import XCTest
import Metal
@testable import ArrowMetal

/// Prints which pipeline variants the current Metal device builds. It never fails: it is a report for a
/// device that cannot be reproduced locally (GitHub's "Apple Paravirtual device"). Runs only when
/// ARROWMETAL_DEVICE_DIAG is set.
final class DeviceDiagnosticTests: XCTestCase {
    func testPipelineVariants() throws {
        guard ProcessInfo.processInfo.environment["ARROWMETAL_DEVICE_DIAG"] != nil else { throw XCTSkip("set ARROWMETAL_DEVICE_DIAG") }
        guard let device = MTLCreateSystemDefaultDevice() else { print("DIAG: no Metal device"); return }
        print("DIAG: device \(device.name) family apple7=\(device.supportsFamily(.apple7)) mac2=\(device.supportsFamily(.mac2)) common3=\(device.supportsFamily(.common3))")
        let scalar = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void k_scalar(device const float* a [[buffer(0)]], device float* o [[buffer(1)]], constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) { if (i < n) o[i] = fabs(a[i]); }
        """
        let folded = Dispatch.foldGridPositions(scalar.replacingOccurrences(of: "k_scalar", with: "k_folded"))
        let byHand = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void k_hand(device const float* a [[buffer(0)]], device float* o [[buffer(1)]], constant uint& n [[buffer(2)]],
                           uint2 p [[thread_position_in_grid]]) { uint i = (p.y << 24u) + p.x; if (i < n) o[i] = fabs(a[i]); }
        """
        let real = MetalArray<Double>.mathSource.0   // the library's own Float64 abs kernel source, as the harness compiles it
        let variants: [(String, String, String)] = [("scalar", scalar, "k_scalar"), ("folded", folded, "k_folded"), ("by-hand uint2", byHand, "k_hand"),
                                                    ("library math_unary_abs (folded)", Dispatch.foldGridPositions(real), "math_unary_abs")]
        for safe in [false, true] {
            for (label, src, fn) in variants {
                let opts = MTLCompileOptions()
                if safe { if #available(macOS 15.0, *) { opts.mathMode = .safe } else { opts.fastMathEnabled = false } }
                do {
                    let lib = try device.makeLibrary(source: src, options: opts)
                    guard let f = lib.makeFunction(name: fn) else { print("DIAG: \(label) safe=\(safe): function missing"); continue }
                    do { _ = try device.makeComputePipelineState(function: f); print("DIAG: \(label) safe=\(safe): pipeline OK") }
                    catch { print("DIAG: \(label) safe=\(safe): PIPELINE FAILED: \(MetalContext.describe(error))") }
                } catch { print("DIAG: \(label) safe=\(safe): LIBRARY FAILED: \(MetalContext.describe(error))") }
            }
        }
        print("DIAG: folded source follows\n\(folded)\nDIAG: end")
    }
}
