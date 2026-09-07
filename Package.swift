// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ArrowMetal",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "ArrowMetal", targets: ["ArrowMetal"]),
        .library(name: "CArrowABI", targets: ["CArrowABI"]),
        // C ABI as a dynamic library, for C and C++ and for any language with a C FFI. It takes and
        // returns Arrow C Data Interface structs, which are copy-free out and copy-free in when the
        // producer's buffers are page aligned (one copy otherwise). Python, Rust, Go, R and Node
        // bindings live in this repository (python/, rust/, go/, r/, node/, polars-plugin/); C# is on
        // the roadmap.
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
