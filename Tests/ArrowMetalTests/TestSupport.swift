import XCTest
@testable import ArrowMetal

/// GPU kernel tests need a real Apple GPU. GitHub's hosted runners expose an "Apple Paravirtual device"
/// whose Metal compiler fails sporadically; there we only validate build, interop and CPU paths.
func requireRealGPU() throws {
    if MetalContext.shared.isVirtualDevice && ProcessInfo.processInfo.environment["ARROWMETAL_FORCE_GPU_TESTS"] == nil {
        throw XCTSkip("GPU tests skipped on virtual Metal device \(MetalContext.shared.device.name)")
    }
}
