import Foundation
import Accelerate
import ArrowMetal

// ArrowMetal benchmark. GPU kernels vs:
//   - "CPU 1-core": a tight, typed, null-aware Swift loop (what a careful CPU Arrow kernel does)
//   - "CPU all-cores": the same loop split across all cores with concurrentPerform
//   - Accelerate vDSP where an equivalent exists (Float32, no nulls)
// Usage: arrowmetal-bench [rows] [iterations]

let rows = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 50_000_000
let iters = CommandLine.arguments.count > 2 ? Int(CommandLine.arguments[2])! : 5
let cores = ProcessInfo.processInfo.activeProcessorCount

@inline(never) func sink<T>(_ x: T) { withExtendedLifetime(x) {} }
/// Launders a value so the optimiser cannot hoist a pure computation out of the timing loop.
@inline(never) func opaque<T>(_ x: T) -> T { x }

// Typed single-core CPU baselines over the Arrow layout (what a careful CPU kernel does).
func cpuMinInt64(_ a: MetalArray<Int64>) -> Int64 {
    let p = a.valuePointer
    guard let v = a.validity else { var m = Int64.max; for i in 0..<a.length { m = min(m, p[i]) }; return m }
    let bm = v.typed(UInt8.self)
    var m = Int64.max
    for i in 0..<a.length where (bm[i >> 3] >> (i & 7)) & 1 == 1 { m = min(m, p[i]) }
    return m
}
/// compare(a > s) into a packed bitmap, one byte (8 elements) at a time.
func cpuCompareGt(_ a: MetalArray<Int64>, _ sc: Int64, into out: UnsafeMutablePointer<UInt8>) {
    let p = a.valuePointer
    let n = a.length
    var i = 0
    while i + 8 <= n {
        var b: UInt8 = 0
        for j in 0..<8 where p[i + j] > sc { b |= 1 << j }
        out[i >> 3] = b
        i += 8
    }
    var b: UInt8 = 0
    for j in 0..<(n - i) where p[i + j] > sc { b |= 1 << j }
    if i < n { out[i >> 3] = b }
}
/// filter by a selection bitmap into a preallocated output; returns the number kept.
func cpuFilterInt64(_ a: MetalArray<Int64>, sel: UnsafePointer<UInt8>, into out: UnsafeMutablePointer<Int64>) -> Int {
    let p = a.valuePointer
    var k = 0
    let n = a.length
    var i = 0
    while i < n {
        var byte = sel[i >> 3]
        if i + 8 > n { byte &= UInt8((1 << (n - i)) - 1) }
        while byte != 0 {
            let j = byte.trailingZeroBitCount
            out[k] = p[i + j]; k += 1
            byte &= byte - 1
        }
        i += 8
    }
    return k
}

var results: [(String, String, Double, Double)] = []  // section, label, ms, GB/s

func time(_ label: String, bytes: Int, section: String, _ body: () throws -> Void) rethrows {
    try body() // warm-up (compiles pipelines)
    var best = Double.infinity
    for _ in 0..<iters {
        let t0 = DispatchTime.now().uptimeNanoseconds
        try body()
        best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
    }
    let gbps = Double(bytes) / best / 1e9
    print(String(format: "  %-40s %9.2f ms  %7.1f GB/s", (label as NSString).utf8String!, best * 1000, gbps))
    results.append((section, label, best * 1000, gbps))
}

// Null-aware Int64 sum over an Arrow layout, single core, range [lo, hi).
func cpuSumInt64(_ a: MetalArray<Int64>, _ lo: Int, _ hi: Int) -> Int64 {
    let p = a.valuePointer
    guard let v = a.validity else { var s: Int64 = 0; for i in lo..<hi { s &+= p[i] }; return s }
    let bm = v.typed(UInt8.self)
    var s: Int64 = 0
    var i = lo
    while i < hi {
        if i & 7 == 0 && i + 8 <= hi {
            let byte = bm[i >> 3]
            if byte == 0xFF { for j in i..<(i + 8) { s &+= p[j] }; i += 8; continue }
            if byte == 0 { i += 8; continue }
        }
        if (bm[i >> 3] >> (i & 7)) & 1 == 1 { s &+= p[i] }
        i += 1
    }
    return s
}

func parallel<R: AdditiveArithmetic>(_ n: Int, _ f: (Int, Int) -> R) -> R {
    let chunks = cores * 4
    var partial = [R](repeating: .zero, count: chunks)
    partial.withUnsafeMutableBufferPointer { out in
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            let lo = n * c / chunks, hi = n * (c + 1) / chunks
            out[c] = f(lo, hi)
        }
    }
    return partial.reduce(.zero, +)
}

let ctx = MetalContext.shared
print("ArrowMetal bench on \(ctx.device.name), rows=\(rows), iterations=\(iters) (best of), CPU cores=\(cores)\n")

// --- Int64 column with ~10% nulls ---
var g = SystemRandomNumberGenerator()
var i64: [Int64?] = []
i64.reserveCapacity(rows)
for _ in 0..<rows { i64.append(Int.random(in: 0..<10, using: &g) == 0 ? nil : Int64.random(in: -1000...1000, using: &g)) }
let colI64 = try MetalArray<Int64>(i64)
let i64raw = colI64.toRawArray()
sink(i64raw)
let bytesI64 = rows * 8

var sec = "sum(Int64, 10% nulls)"; print(sec)
try time("Metal  sum", bytes: bytesI64, section: sec) { sink(try colI64.sum()) }
time("CPU 1-core  sum (null-aware loop)", bytes: bytesI64, section: sec) { sink(cpuSumInt64(colI64, 0, rows)) }
time("CPU \(cores)-core sum (null-aware loop)", bytes: bytesI64, section: sec) { sink(parallel(rows) { cpuSumInt64(colI64, $0, $1) }) }

sec = "min(Int64, 10% nulls)"; print("\n" + sec)
try time("Metal  min", bytes: bytesI64, section: sec) { sink(try colI64.min()) }
time("CPU 1-core  min (null-aware loop)", bytes: bytesI64, section: sec) { sink(cpuMinInt64(opaque(colI64))) }

sec = "compare(Int64 > 0) -> boolean bitmap"; print("\n" + sec)
try time("Metal  compare", bytes: bytesI64, section: sec) { sink(try colI64.compare(.gt, 0)) }
let cmpOut = UnsafeMutablePointer<UInt8>.allocate(capacity: rows / 8 + 1)
time("CPU 1-core  compare (packed bitmap)", bytes: bytesI64, section: sec) { cpuCompareGt(opaque(colI64), 0, into: cmpOut); sink(cmpOut[0]) }

sec = "filter(Int64 where > 0) -> compacted (~45% kept)"; print("\n" + sec)
let mask = try colI64.compare(.gt, 0)
try time("Metal  filter", bytes: bytesI64, section: sec) { sink(try colI64.filter(mask)) }
let selBits = try mask.and(mask).values  // materialised selection bitmap (mask has no nulls here)
let filtOut = UnsafeMutablePointer<Int64>.allocate(capacity: rows)
time("CPU 1-core  filter (bit-scan loop)", bytes: bytesI64, section: sec) { sink(cpuFilterInt64(opaque(colI64), sel: selBits.typed(UInt8.self), into: filtOut)) }

sec = "compare + filter pipeline"; print("\n" + sec)
try time("Metal  compare then filter", bytes: bytesI64, section: sec) { sink(try colI64.filter(try colI64.compare(.gt, 0))) }
time("Swift  [Int64].filter { $0 > 0 } (no nulls)", bytes: bytesI64, section: sec) { sink(i64raw.filter { $0 > 0 }) }

sec = "multiply(Int64 * 3)"; print("\n" + sec)
try time("Metal  multiply scalar", bytes: bytesI64 * 2, section: sec) { sink(try colI64.multiply(3)) }
try time("CPU 1-core  multiply scalar", bytes: bytesI64 * 2, section: sec) { sink(try CPUReference.arithmetic(colI64, .mul, scalar: 3)) }

// --- Float32 column, no nulls: compare with vDSP ---
let f32 = (0..<rows).map { _ in Float.random(in: -1...1, using: &g) }
let colF32 = try MetalArray<Float>(f32)
let bytesF32 = rows * 4
sec = "sum(Float32, no nulls)"; print("\n" + sec)
try time("Metal  sum", bytes: bytesF32, section: sec) { sink(try colF32.sum()) }
time("vDSP   sum (Accelerate, 1 core)", bytes: bytesF32, section: sec) { sink(vDSP.sum(f32)) }

sec = "max(Float32, no nulls)"; print("\n" + sec)
try time("Metal  max", bytes: bytesF32, section: sec) { sink(try colF32.max()) }
time("vDSP   maximum (Accelerate, 1 core)", bytes: bytesF32, section: sec) { sink(vDSP.maximum(f32)) }

sec = "multiply(Float32 * 2.5)"; print("\n" + sec)
try time("Metal  multiply scalar", bytes: bytesF32 * 2, section: sec) { sink(try colF32.multiply(2.5)) }
var outF = [Float](repeating: 0, count: rows)
time("vDSP   multiply scalar (Accelerate, 1 core)", bytes: bytesF32 * 2, section: sec) { vDSP.multiply(2.5, f32, result: &outF); sink(outF[0]) }

sec = "compare(Float32 > 0) then filter"; print("\n" + sec)
try time("Metal  compare then filter", bytes: bytesF32, section: sec) { sink(try colF32.filter(try colF32.compare(.gt, 0))) }
time("Swift  [Float].filter { $0 > 0 }", bytes: bytesF32, section: sec) { sink(f32.filter { $0 > 0 }) }

// Markdown table for the README.
print("\n\n| Operation | Implementation | Time (ms) | Throughput (GB/s) |\n|---|---|---:|---:|")
for (s, l, ms, gb) in results { print("| \(s) | \(l) | \(String(format: "%.2f", ms)) | \(String(format: "%.1f", gb)) |") }
