import XCTest
import Metal
@testable import ArrowMetal

/// Prints which kernel variants the current Metal device builds. It never fails: it is a report for a device
/// that cannot be reproduced locally (GitHub's "Apple Paravirtual device"). Runs only when ARROWMETAL_DEVICE_DIAG
/// is set; ARROWMETAL_DIAG_ONLY=label,label,... runs just those variants, in that order, in this one process.
final class DeviceDiagnosticTests: XCTestCase {
    func testPipelineVariants() throws {
        guard ProcessInfo.processInfo.environment["ARROWMETAL_DEVICE_DIAG"] != nil else { throw XCTSkip("set ARROWMETAL_DEVICE_DIAG") }
        guard let device = MTLCreateSystemDefaultDevice() else { print("DIAG: no Metal device"); return }
        let pre = "#include <metal_stdlib>\nusing namespace metal;\n"
        let scalar = pre + "kernel void k_scalar(device const float* a [[buffer(0)]], device float* o [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) o[i] = fabs(a[i]); }"
        // The radix histogram kernel alone, with the sort source's defines, in its current form and with the loop it had
        // before the 2026-09-26 32-bit wrap rewrite.
        let sortSrc = SortSource.source(K: "ulong")
        let start = sortSrc.range(of: "kernel void radix_histogram")!.lowerBound
        let afterStart = sortSrc.index(start, offsetBy: 30)
        let end = sortSrc.range(of: "kernel void", range: afterStart..<sortSrc.endIndex)?.lowerBound ?? sortSrc.endIndex
        let histogram = String(sortSrc[start..<end])
        let defines = "#define RADIX 256u\n#define DIGIT_BITS 8u\n#define DIGIT_MASK 0xFFu\n#define SIMDS (TG / 32u)\n"
        let radixCurrent = KernelSource.prelude + defines + histogram
        let oldLoop = histogram
            .replacingOccurrences(of: "len = (start < n) ? min(elemsPerBlock, n - start) : 0u;", with: "end = min(n, start + elemsPerBlock);")
            .replacingOccurrences(of: "for (uint off = lid; off < len; off += TG) { uint i = start + off;", with: "for (uint i = start + lid; i < end; i += TG) {")
        let radixOld = KernelSource.prelude + defines + oldLoop
        let radixNoTernary = KernelSource.prelude + defines + histogram.replacingOccurrences(of: "len = (start < n) ? min(elemsPerBlock, n - start) : 0u;", with: "len = min(elemsPerBlock, n - start);")
        let radixEndLoop = KernelSource.prelude + defines + histogram.replacingOccurrences(of: "for (uint off = lid; off < len; off += TG) { uint i = start + off;", with: "for (uint i = start + lid; i < start + len; i += TG) {")
        let all: [(String, String, String)] = [
            ("scalar", scalar, "k_scalar"),
            ("abs-lib", Dispatch.foldGridPositions(MetalArray<Double>.mathSource.0), "math_unary_abs"),
            ("radix-current", radixCurrent, "radix_histogram"),
            ("radix-oldloop", radixOld, "radix_histogram"),
            ("radix-noternary", radixNoTernary, "radix_histogram"),
            ("radix-endloop", radixEndLoop, "radix_histogram"),
            ("radix-current-folded", Dispatch.foldGridPositions(radixCurrent), "radix_histogram"),
        ]
        let only = ProcessInfo.processInfo.environment["ARROWMETAL_DIAG_ONLY"].map { $0.split(separator: ",").map(String.init) }
        let run = only.map { names in names.compactMap { n in all.first { $0.0 == n } } } ?? all
        print("DIAG: device \(device.name) apple7=\(device.supportsFamily(.apple7)) mac2=\(device.supportsFamily(.mac2)); order \(run.map(\.0).joined(separator: ","))")
        for (label, src, fn) in run {
            let opts = MTLCompileOptions()
            if #available(macOS 15.0, *) { opts.mathMode = .safe } else { opts.fastMathEnabled = false }
            do {
                let lib = try device.makeLibrary(source: src, options: opts)
                guard let f = lib.makeFunction(name: fn) else { print("DIAG: \(label): function missing"); continue }
                do { _ = try device.makeComputePipelineState(function: f); print("DIAG: \(label): pipeline OK") }
                catch { print("DIAG: \(label): PIPELINE FAILED: \(MetalContext.describe(error))") }
            } catch { print("DIAG: \(label): LIBRARY FAILED: \(MetalContext.describe(error).prefix(300))") }
        }
        print("DIAG: end")
    }
}
