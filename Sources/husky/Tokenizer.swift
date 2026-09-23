import Foundation

final class Tokenizer {
    private var vocab: [String: Int] = [:]
    private var ranks: [String: Int] = [:]     // "a b" -> rank
    private var idToToken: [Int: String] = [:]
    private var addedTokens: [(content: String, id: Int)] = []
    private let byteToChar: [Character]
    private let charToByte: [Character: UInt8]
    private let regex: NSRegularExpression

    static let byteLevelPattern =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    init(path: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let model = root["model"] as! [String: Any]
        let v = model["vocab"] as! [String: Int]
        self.vocab = v
        for (tok, id) in v { idToToken[id] = tok }
        let merges = model["merges"] as! [[String]]
        for (i, m) in merges.enumerated() { ranks["\(m[0]) \(m[1])"] = i }

        if let added = root["added_tokens"] as? [[String: Any]] {
            for a in added {
                if let c = a["content"] as? String, let id = a["id"] as? Int {
                    addedTokens.append((c, id))
                    if vocab[c] == nil { vocab[c] = id; idToToken[id] = c }
                }
            }
        }
        // longer special tokens first to avoid partial matches
        addedTokens.sort { $0.content.count > $1.content.count }

        // GPT-2 byte <-> unicode mapping
        var bs: [Int] = []
        var cs: [Int] = []
        for b in 33...126 { bs.append(b); cs.append(b) }
        for b in 161...172 { bs.append(b); cs.append(b) }
        for b in 174...255 { bs.append(b); cs.append(b) }
        var n = 0
        for b in 0...255 where !bs.contains(b) {
            bs.append(b); cs.append(256 + n); n += 1
        }
        var b2c = [Character](repeating: " ", count: 256)
        var c2b = [Character: UInt8]()
        for i in 0..<bs.count {
            let ch = Character(UnicodeScalar(cs[i])!)
            b2c[bs[i]] = ch
            c2b[ch] = UInt8(bs[i])
        }
        self.byteToChar = b2c
        self.charToByte = c2b
        self.regex = try NSRegularExpression(pattern: Tokenizer.byteLevelPattern)
    }

    var eosIds: Set<Int> { [248044, 248046] }

    private func byteLevelEncode(_ bytes: [UInt8]) -> [Character] {
        bytes.map { byteToChar[Int($0)] }
    }

    private func bpe(_ symbols: [String]) -> [String] {
        if symbols.count <= 1 { return symbols }
        var syms = symbols
        while true {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(syms.count - 1) {
                if let r = ranks["\(syms[i]) \(syms[i + 1])"], r < bestRank {
                    bestRank = r; bestIdx = i
                }
            }
            if bestIdx < 0 { break }
            var next: [String] = []
            var i = 0
            while i < syms.count {
                if i == bestIdx {
                    next.append(syms[i] + syms[i + 1]); i += 2
                } else {
                    next.append(syms[i]); i += 1
                }
            }
            syms = next
            if syms.count == 1 { break }
        }
        return syms
    }

    private func encodePretoken(_ token: String) -> [Int] {
        let chars = byteLevelEncode(Array(token.utf8))
        let syms = bpe(chars.map { String($0) })
        var ids: [Int] = []
        for s in syms {
            if let id = vocab[s] {
                ids.append(id)
            } else {
                // fall back to per-character bytes
                for ch in s {
                    if let b = charToByte[ch], let id = vocab[String(byteToChar[Int(b)])] {
                        ids.append(id)
                    }
                }
            }
        }
        return ids
    }

    func encode(_ text: String) -> [Int] {
        var result: [Int] = []
        let ns = text as NSString
        var cursor = 0
        // handle special tokens by scanning
        while cursor < ns.length {
            var bestRange = NSRange(location: NSNotFound, length: 0)
            var bestToken: (content: String, id: Int)? = nil
            for a in addedTokens {
                let r = ns.range(of: a.content, options: [], range: NSRange(location: cursor, length: ns.length - cursor))
                if r.location != NSNotFound {
                    if bestRange.location == NSNotFound || r.location < bestRange.location {
                        bestRange = r; bestToken = a
                    }
                }
            }
            let textEnd = bestToken == nil ? ns.length : bestRange.location
            let plain = ns.substring(with: NSRange(location: cursor, length: textEnd - cursor))
            result.append(contentsOf: encodePlain(plain))
            if let bt = bestToken {
                result.append(bt.id)
                cursor = bestRange.location + bestRange.length
            } else {
                break
            }
        }
        return result
    }

    private func encodePlain(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        var ids: [Int] = []
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            ids.append(contentsOf: encodePretoken(ns.substring(with: m.range)))
        }
        return ids
    }

    func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard let tok = idToToken[id] else { continue }
            if addedTokens.contains(where: { $0.id == id }) { continue }
            for ch in tok {
                if let b = charToByte[ch] { bytes.append(b) }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
