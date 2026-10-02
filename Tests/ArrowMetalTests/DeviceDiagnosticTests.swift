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
        let pre = "#include <metal_stdlib>\nusing namespace metal;\n"
        let lidK = pre + "kernel void k_lid(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) { o[i] = float(lid); }"
        let tpgK = pre + "kernel void k_tpg(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) { for (uint j = i; j < 1000u; j += n) o[j] = 1.0f; }"
        let tgK = pre + "kernel void k_tg(device atomic_uint* o [[buffer(0)]], uint tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) { threadgroup atomic_uint h[4]; if (lid < 4u) atomic_store_explicit(&h[lid], 0u, memory_order_relaxed); threadgroup_barrier(mem_flags::mem_threadgroup); atomic_fetch_add_explicit(&h[lid & 3u], 1u, memory_order_relaxed); threadgroup_barrier(mem_flags::mem_threadgroup); if (lid == 0u) atomic_fetch_add_explicit(o, atomic_load_explicit(&h[0], memory_order_relaxed) + tgid, memory_order_relaxed); }"
        let tgOnlyK = pre + "kernel void k_tgonly(device uint* o [[buffer(0)]], uint tgid [[threadgroup_position_in_grid]]) { o[tgid] = tgid; }"
        let radix = SortSource.source(K: "ulong")
        var variants: [(String, String, String)] = [("scalar", scalar, "k_scalar"), ("folded", folded, "k_folded"), ("by-hand uint2", byHand, "k_hand"),
                                                    ("library math_unary_abs (folded)", Dispatch.foldGridPositions(real), "math_unary_abs")]
        for (label, src, fn) in [("i+lid", lidK, "k_lid"), ("i+threads_per_grid loop", tpgK, "k_tpg"), ("tgid+lid+threadgroup atomics", tgK, "k_tg"), ("tgid only", tgOnlyK, "k_tgonly"), ("library radix_histogram", radix, "radix_histogram")] {
            variants.append((label + " scalar", src, fn)); variants.append((label + " folded", Dispatch.foldGridPositions(src), fn))
        }
        for safe in [true] {
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
        print("DIAG: end")
    }
}
