// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ArrowMetal",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "ArrowMetal", targets: ["ArrowMetal"]),
        .library(name: "CArrowABI", targets: ["CArrowABI"]),
        // C ABI as a dynamic library for Python, Rust, Go, C#, R, C++ and C consumers.
        .library(name: "ArrowMetalC", type: .dynamic, targets: ["ArrowMetalC"]),
        .executable(name: "arrowmetal-bench", targets: ["ArrowMetalBench"]),
        .executable(name: "arrowmetal-examples", targets: ["ArrowMetalExamples"]),
    ],
    targets: [
        // Verbatim Arrow C Data / C Device / C Stream ABI structs (no dependencies).
        .target(name: "CArrowABI", path: "Sources/CArrowABI"),
        .target(
            name: "ArrowMetal",
            dependencies: ["CArrowABI"],
            linkerSettings: [.linkedFramework("Metal")]
        ),
        .target(name: "ArrowMetalC", dependencies: ["ArrowMetal", "CArrowABI"]),
        .executableTarget(name: "ArrowMetalBench", dependencies: ["ArrowMetal"]),
        .executableTarget(name: "ArrowMetalExamples", dependencies: ["ArrowMetal", "CArrowABI"]),
        .testTarget(name: "ArrowMetalTests", dependencies: ["ArrowMetal", "CArrowABI"]),
    ]
)
