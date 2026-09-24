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

/// XCTest asks every test class for its suite before the first test runs, so this pins the default
/// for the tests that never call `requireRealGPU()` too.
final class RouterTestDefaults: XCTestCase {
    override class var defaultTestSuite: XCTestSuite {
        routerTestDefault()
        return super.defaultTestSuite
    }
    func testRouterDefaultForSuites() {
        routerTestDefault()
        XCTAssertEqual(Router.mode, Router.environmentMode ?? .gpu)
    }
}
