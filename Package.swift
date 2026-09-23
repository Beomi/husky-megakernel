// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "husky",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "husky",
            path: "Sources/husky",
            resources: [.copy("kernels.metal"), .copy("megakernel.metal"), .copy("batchkernel.metal"), .copy("prefillkernel.metal")]
        )
    ]
)
