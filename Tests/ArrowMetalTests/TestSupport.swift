import XCTest
@testable import ArrowMetal

/// GPU kernel tests need a real Apple GPU. GitHub's hosted runners expose an "Apple Paravirtual device"
/// whose Metal compiler fails sporadically; there we only validate build, interop and CPU paths.
func requireRealGPU() throws {
    routerTestDefault()
    if MetalContext.shared.isVirtualDevice && ProcessInfo.processInfo.environment["ARROWMETAL_FORCE_GPU_TESTS"] == nil {
        throw XCTSkip("GPU tests skipped on virtual Metal device \(MetalContext.shared.device.name)")
    }
}

/// The suites run small inputs that the router's `auto` mode would send to the CPU, so they pin the
/// router to the GPU and keep exercising the kernels. `ARROWMETAL_ROUTER`, when set, wins: running
/// the whole suite with `ARROWMETAL_ROUTER=cpu` exercises every CPU path against the same tests. The
/// router's own tests (RouterTests) choose paths per call and run gpu, cpu and auto.
private let routerDefaultApplied: Void = {
    if Router.environmentMode == nil { Router.mode = .gpu }
}()
func routerTestDefault() { _ = routerDefaultApplied }

/// The pin rides on `requireRealGPU()`, which the GPU test files call in `setUp` or at the top of each
/// test. A test that runs a routed operation without calling it runs under the process default (`auto`).
final class RouterTestDefaults: XCTestCase {
    func testRouterDefaultForSuites() throws {
        try requireRealGPU()
        XCTAssertEqual(Router.mode, Router.environmentMode ?? .gpu)
    }
}
