import Foundation

@inline(__always)
func f32ToBf16Bits(_ f: Float) -> UInt16 {
    let x = f.bitPattern
    let lsb = (x >> 16) & 1
    let rounding = 0x7FFF &+ lsb
    let out = (x &+ rounding) >> 16
    return UInt16(truncatingIfNeeded: out)
}

enum TensorDType: String {
    case f32 = "F32"
    case f16 = "F16"
    case bf16 = "BF16"
    case u32 = "U32"
    case i64 = "I64"

    var elementSize: Int {
        switch self {
        case .f32, .u32: return 4
        case .f16, .bf16: return 2
        case .i64: return 8
        }
    }
}

struct TensorInfo {
    let name: String
    let dtype: TensorDType
    let shape: [Int]
    let offset: Int   // relative to start of data section
    let length: Int   // bytes
}

final class Safetensors {
    private let data: NSData
    private let dataStart: Int
    let tensors: [String: TensorInfo]

    init(path: String) throws {
        self.data = try NSData(contentsOfFile: path, options: [.mappedIfSafe])
        let base = data.bytes
        var headerLen: UInt64 = 0
        memcpy(&headerLen, base, 8)
        let headerStart = 8
        let headerEnd = 8 + Int(headerLen)
        let headerData = Data(bytes: base + headerStart, count: Int(headerLen))
        let json = try JSONSerialization.jsonObject(with: headerData) as! [String: Any]
        self.dataStart = headerEnd
        var tensors: [String: TensorInfo] = [:]
        for (name, v) in json {
            guard let info = v as? [String: Any],
                  let dtypeStr = info["dtype"] as? String,
                  dtypeStr != "__metadata__",
                  let dtype = TensorDType(rawValue: dtypeStr),
                  let shape = info["shape"] as? [Int],
                  let offs = info["data_offsets"] as? [Int] else { continue }
            tensors[name] = TensorInfo(name: name, dtype: dtype, shape: shape,
                                       offset: offs[0], length: offs[1] - offs[0])
        }
        self.tensors = tensors
    }

    func info(_ name: String) -> TensorInfo {
        guard let t = tensors[name] else { fatalError("missing tensor \(name)") }
        return t
    }

    private func rawPointer(_ name: String) -> UnsafeRawPointer {
        let info = info(name)
        return UnsafeRawPointer(data.bytes + dataStart + info.offset)
    }

    func rawPointerPublic(_ name: String) -> UnsafeRawPointer { rawPointer(name) }

    func copyTensor(_ name: String, into dst: UnsafeMutableRawPointer) {
        let info = info(name)
        memcpy(dst, rawPointer(name), info.length)
    }

    func u32(_ name: String) -> [UInt32] {
        let info = info(name)
        let count = info.length / 4
        let p = rawPointer(name).bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    /// Raw bfloat16 elements. Converts F16/F32 when encountered.
    func bf16(_ name: String) -> [UInt16] {
        let info = info(name)
        switch info.dtype {
        case .bf16:
            let count = info.length / 2
            let p = rawPointer(name).bindMemory(to: UInt16.self, capacity: count)
            return Array(UnsafeBufferPointer(start: p, count: count))
        case .f32:
            let count = info.length / 4
            let p = rawPointer(name).bindMemory(to: Float.self, capacity: count)
            return (0..<count).map { f32ToBf16Bits(p[$0]) }
        case .f16:
            let count = info.length / 2
            let p = rawPointer(name).bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { f32ToBf16Bits(Float(Float16(bitPattern: p[$0]))) }
        default:
            fatalError("unsupported dtype \(info.dtype) for \(name)")
        }
    }

    func f32(_ name: String) -> [Float] {
        let info = info(name)
        switch info.dtype {
        case .f32:
            let count = info.length / 4
            let p = rawPointer(name).bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: p, count: count))
        case .bf16, .f16:
            let count = info.length / 2
            let p = rawPointer(name).bindMemory(to: UInt16.self, capacity: count)
            var out = [Float](repeating: 0, count: count)
            if info.dtype == .bf16 {
                for i in 0..<count { out[i] = Float(bitPattern: UInt32(p[i]) << 16) }
            } else {
                for i in 0..<count { out[i] = Float(Float16(bitPattern: p[i])) }
            }
            return out
        default:
            fatalError("unsupported dtype \(info.dtype) for \(name)")
        }
    }
}
