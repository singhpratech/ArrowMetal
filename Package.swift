// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ArrowMetal",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "ArrowMetal", targets: ["ArrowMetal"]),
        .library(name: "CArrowABI", targets: ["CArrowABI"]),
        .executable(name: "arrowmetal-bench", targets: ["ArrowMetalBench"]),
    ],
    targets: [
        // Verbatim Arrow C Data / C Device / C Stream ABI structs (no dependencies).
        .target(name: "CArrowABI", path: "Sources/CArrowABI"),
        .target(
            name: "ArrowMetal",
            dependencies: ["CArrowABI"],
            linkerSettings: [.linkedFramework("Metal")]
        ),
        .executableTarget(name: "ArrowMetalBench", dependencies: ["ArrowMetal"]),
        .testTarget(name: "ArrowMetalTests", dependencies: ["ArrowMetal", "CArrowABI"]),
    ]
)
