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

/// Splits [0, n) into chunks aligned to 64 elements (keeps bitmap bytes/words chunk-private) and runs
/// `f` on all cores. Returns one result per chunk.
func parallelChunks<R>(_ n: Int, _ f: (Int, Int) -> R) -> [R] {
    let chunks = max(1, min(cores * 4, (n + 63) / 64))
    let per = ((n + chunks - 1) / chunks + 63) / 64 * 64
    var partial = [R?](repeating: nil, count: chunks)
    partial.withUnsafeMutableBufferPointer { out in
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            let lo = min(n, c * per), hi = min(n, (c + 1) * per)
            out[c] = f(lo, hi)
        }
    }
    return partial.map { $0! }
}
func parallel<R: AdditiveArithmetic>(_ n: Int, _ f: (Int, Int) -> R) -> R { parallelChunks(n, f).reduce(.zero, +) }

/// Parallel filter over an Arrow layout: per-chunk count, prefix, per-chunk scatter.
func cpuFilterParallel<T>(_ p: UnsafePointer<T>, n: Int, sel: UnsafePointer<UInt8>, into out: UnsafeMutablePointer<T>) -> Int {
    func popcountRange(_ lo: Int, _ hi: Int) -> Int {
        var c = 0; var i = lo
        while i < hi { var b = sel[i >> 3]; if i + 8 > hi { b &= UInt8((1 << (hi - i)) - 1) }; c += b.nonzeroBitCount; i += 8 }
        return c
    }
    let counts = parallelChunks(n, popcountRange)
    var offsets = [Int](repeating: 0, count: counts.count); var acc = 0
    for (i, c) in counts.enumerated() { offsets[i] = acc; acc += c }
    let chunks = counts.count
    let per = ((n + chunks - 1) / chunks + 63) / 64 * 64
    DispatchQueue.concurrentPerform(iterations: chunks) { c in
        let lo = min(n, c * per), hi = min(n, (c + 1) * per)
        var k = offsets[c]; var i = lo
        while i < hi {
            var byte = sel[i >> 3]
            if i + 8 > hi { byte &= UInt8((1 << (hi - i)) - 1) }
            while byte != 0 { let j = byte.trailingZeroBitCount; out[k] = p[i + j]; k += 1; byte &= byte - 1 }
            i += 8
        }
    }
    return acc
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
time("CPU \(cores)-core sum (null-aware loop)", bytes: bytesI64, section: sec) { sink(parallel(rows) { cpuSumInt64(colI64, $0, $1) }) }

sec = "min(Int64, 10% nulls)"; print("\n" + sec)
try time("Metal  min", bytes: bytesI64, section: sec) { sink(try colI64.min()) }
time("CPU \(cores)-core min (null-aware loop)", bytes: bytesI64, section: sec) {
    let a = opaque(colI64); let p = a.valuePointer; let bm = a.validity!.typed(UInt8.self)
    sink(parallelChunks(rows) { lo, hi in var m = Int64.max; for i in lo..<hi where (bm[i >> 3] >> (i & 7)) & 1 == 1 { m = min(m, p[i]) }; return m }.min()!)
}

sec = "compare(Int64 > 0) -> boolean bitmap"; print("\n" + sec)
try time("Metal  compare", bytes: bytesI64, section: sec) { sink(try colI64.compare(.gt, 0)) }
let cmpOut = UnsafeMutablePointer<UInt8>.allocate(capacity: rows / 8 + 64)
time("CPU \(cores)-core compare (packed bitmap)", bytes: bytesI64, section: sec) {
    let p = opaque(colI64).valuePointer
    _ = parallelChunks(rows) { lo, hi in
        var i = lo
        while i < hi { var b: UInt8 = 0; let lim = min(8, hi - i); for j in 0..<lim where p[i + j] > 0 { b |= 1 << j }; cmpOut[i >> 3] = b; i += 8 }
        return 0
    }
    sink(cmpOut[0])
}

sec = "filter(Int64 where > 0) -> compacted (~45% kept)"; print("\n" + sec)
let mask = try colI64.compare(.gt, 0)
try time("Metal  filter", bytes: bytesI64, section: sec) { sink(try colI64.filter(mask)) }
let selBits = try mask.and(mask).values  // materialised selection bitmap (mask has no nulls here)
let filtOut = UnsafeMutablePointer<Int64>.allocate(capacity: rows)
time("CPU \(cores)-core filter (count, prefix, scatter)", bytes: bytesI64, section: sec) {
    sink(cpuFilterParallel(opaque(colI64).valuePointer, n: rows, sel: selBits.typed(UInt8.self), into: filtOut))
}

sec = "compare + filter pipeline"; print("\n" + sec)
try time("Metal  compare then filter", bytes: bytesI64, section: sec) { sink(try colI64.filter(try colI64.compare(.gt, 0))) }
time("CPU \(cores)-core compare then filter", bytes: bytesI64, section: sec) {
    let p = opaque(colI64).valuePointer
    _ = parallelChunks(rows) { lo, hi in
        var i = lo
        while i < hi { var b: UInt8 = 0; let lim = min(8, hi - i); for j in 0..<lim where p[i + j] > 0 { b |= 1 << j }; cmpOut[i >> 3] = b; i += 8 }
        return 0
    }
    sink(cpuFilterParallel(p, n: rows, sel: cmpOut, into: filtOut))
}
time("Swift  [Int64].filter { $0 > 0 } (1 core, no nulls)", bytes: bytesI64, section: sec) { sink(i64raw.filter { $0 > 0 }) }

sec = "fused filter(where Int64 > 0)"; print("\n" + sec)
try time("Metal  filter(where:) fused predicate", bytes: bytesI64, section: sec) { sink(try colI64.filter(where: .gt, 0)) }

sec = "group-by sum(Int64) by 5 keys"; print("\n" + sec)
let keys5 = try MetalArray<Int32>((0..<rows).map { _ in Int32.random(in: 0..<5, using: &g) })
let gb5 = try keys5.groupBy(keyCount: 5)
try time("Metal  group-by sum (privatised)", bytes: bytesI64 + rows * 4, section: sec) { sink(try gb5.sum(colI64)) }
time("CPU \(cores)-core group-by sum", bytes: bytesI64 + rows * 4, section: sec) {
    let p = opaque(colI64).valuePointer, kp = keys5.valuePointer, bm = colI64.validity!.typed(UInt8.self)
    let parts = parallelChunks(rows) { lo, hi -> [Int64] in
        var acc = [Int64](repeating: 0, count: 5)
        for i in lo..<hi where (bm[i >> 3] >> (i & 7)) & 1 == 1 { acc[Int(kp[i])] &+= p[i] }
        return acc
    }
    var total = [Int64](repeating: 0, count: 5)
    for part in parts { for k in 0..<5 { total[k] &+= part[k] } }
    sink(total)
}

sec = "group-by sum(Int64) by 1000 keys"; print("\n" + sec)
let keys1k = try MetalArray<Int32>((0..<rows).map { _ in Int32.random(in: 0..<1000, using: &g) })
let gb1k = try keys1k.groupBy(keyCount: 1000)
try time("Metal  group-by sum (privatised)", bytes: bytesI64 + rows * 4, section: sec) { sink(try gb1k.sum(colI64)) }

sec = "group-by sum(Int64) by 100000 keys"; print("\n" + sec)
let keys100k = try MetalArray<Int32>((0..<rows).map { _ in Int32.random(in: 0..<100_000, using: &g) })
let gb100k = try keys100k.groupBy(keyCount: 100_000)
try time("Metal  group-by sum (device atomics)", bytes: bytesI64 + rows * 4, section: sec) { sink(try gb100k.sum(colI64)) }

sec = "query: sum(amount) where region == 2 and amount > 100 (50M rows)"; print("\n" + sec)
let amountF = try MetalArray<Float>((0..<rows).map { _ in Float.random(in: 0...500, using: &g) })
try time("Metal  filter + filter + sum", bytes: rows * (4 + 4), section: sec) {
    let m = try keys5.compare(.eq, 2).and(try amountF.compare(.gt, 100))
    sink(try amountF.filter(m).sum())
}
time("CPU \(cores)-core fused loop", bytes: rows * (4 + 4), section: sec) {
    let kp = opaque(keys5).valuePointer, ap = amountF.valuePointer
    sink(parallel(rows) { lo, hi -> Double in var s: Float = 0; for i in lo..<hi where kp[i] == 2 && ap[i] > 100 { s += ap[i] }; return Double(s) })
}

sec = "multiply(Int64 * 3)"; print("\n" + sec)
try time("Metal  multiply scalar", bytes: bytesI64 * 2, section: sec) { sink(try colI64.multiply(3)) }
let mulOut = UnsafeMutablePointer<Int64>.allocate(capacity: rows)
time("CPU \(cores)-core multiply scalar", bytes: bytesI64 * 2, section: sec) {
    let p = opaque(colI64).valuePointer
    _ = parallelChunks(rows) { lo, hi in for i in lo..<hi { mulOut[i] = p[i] &* 3 }; return 0 }
    sink(mulOut[0])
}

sec = "take(Int64, 25M random indices)"; print("\n" + sec)
let takeIdx = try MetalArray<Int32>((0..<(rows / 2)).map { _ in Int32.random(in: 0..<Int32(rows), using: &g) })
try time("Metal  take", bytes: rows / 2 * (8 + 4 + 8), section: sec) { sink(try colI64.take(takeIdx)) }
let takeOut = UnsafeMutablePointer<Int64>.allocate(capacity: rows / 2)
time("CPU \(cores)-core take (gather loop)", bytes: rows / 2 * (8 + 4 + 8), section: sec) {
    let p = opaque(colI64).valuePointer, ip = takeIdx.valuePointer
    _ = parallelChunks(rows / 2) { lo, hi in for i in lo..<hi { takeOut[i] = p[Int(ip[i])] }; return 0 }
    sink(takeOut[0])
}

sec = "Float64: compare(> 500) then filter"; print("\n" + sec)
let colF64 = try MetalArray<Double>((0..<rows).map { _ in Double.random(in: 0...1000, using: &g) })
try time("Metal  compare then filter (bit-pattern kernels)", bytes: rows * 8, section: sec) { sink(try colF64.filter(try colF64.compare(.gt, 500))) }
let f64Out = UnsafeMutablePointer<Double>.allocate(capacity: rows)
time("CPU \(cores)-core compare then filter", bytes: rows * 8, section: sec) {
    let p = opaque(colF64).valuePointer
    _ = parallelChunks(rows) { lo, hi in
        var i = lo
        while i < hi { var b: UInt8 = 0; let lim = min(8, hi - i); for j in 0..<lim where p[i + j] > 500 { b |= 1 << j }; cmpOut[i >> 3] = b; i += 8 }
        return 0
    }
    sink(cpuFilterParallel(p, n: rows, sel: cmpOut, into: f64Out))
}

sec = "cast(Int64 -> Float32)"; print("\n" + sec)
try time("Metal  cast", bytes: rows * 12, section: sec) { sink(try colI64.cast(to: Float.self)) }
let castOut = UnsafeMutablePointer<Float>.allocate(capacity: rows)
time("CPU \(cores)-core convert loop", bytes: rows * 12, section: sec) {
    let p = opaque(colI64).valuePointer
    _ = parallelChunks(rows) { lo, hi in for i in lo..<hi { castOut[i] = Float(p[i]) }; return 0 }
    sink(castOut[0])
}

// --- Float32 column, no nulls: compare with vDSP ---
let f32 = (0..<rows).map { _ in Float.random(in: -1...1, using: &g) }
let colF32 = try MetalArray<Float>(f32)
let bytesF32 = rows * 4
sec = "sum(Float32, no nulls)"; print("\n" + sec)
try time("Metal  sum", bytes: bytesF32, section: sec) { sink(try colF32.sum()) }
time("vDSP   sum (Accelerate, 1 core)", bytes: bytesF32, section: sec) { sink(vDSP.sum(f32)) }
time("vDSP   sum (Accelerate, \(cores) cores)", bytes: bytesF32, section: sec) {
    let p = opaque(colF32).valuePointer
    sink(parallel(rows) { lo, hi in vDSP.sum(UnsafeBufferPointer(start: p + lo, count: hi - lo)) })
}

sec = "max(Float32, no nulls)"; print("\n" + sec)
try time("Metal  max", bytes: bytesF32, section: sec) { sink(try colF32.max()) }
time("vDSP   maximum (Accelerate, 1 core)", bytes: bytesF32, section: sec) { sink(vDSP.maximum(f32)) }
time("vDSP   maximum (Accelerate, \(cores) cores)", bytes: bytesF32, section: sec) {
    let p = opaque(colF32).valuePointer
    sink(parallelChunks(rows) { lo, hi in vDSP.maximum(UnsafeBufferPointer(start: p + lo, count: hi - lo)) }.max()!)
}

sec = "multiply(Float32 * 2.5)"; print("\n" + sec)
try time("Metal  multiply scalar", bytes: bytesF32 * 2, section: sec) { sink(try colF32.multiply(2.5)) }
var outF = [Float](repeating: 0, count: rows)
time("vDSP   multiply scalar (Accelerate, 1 core)", bytes: bytesF32 * 2, section: sec) { vDSP.multiply(2.5, f32, result: &outF); sink(outF[0]) }
let mulF = UnsafeMutablePointer<Float>.allocate(capacity: rows)
time("vDSP   multiply scalar (Accelerate, \(cores) cores)", bytes: bytesF32 * 2, section: sec) {
    let p = opaque(colF32).valuePointer
    _ = parallelChunks(rows) { lo, hi in
        var dst = UnsafeMutableBufferPointer(start: mulF + lo, count: hi - lo)
        vDSP.multiply(2.5, UnsafeBufferPointer(start: p + lo, count: hi - lo), result: &dst); return 0
    }
    sink(mulF[0])
}

sec = "compare(Float32 > 0) then filter"; print("\n" + sec)
try time("Metal  compare then filter", bytes: bytesF32, section: sec) { sink(try colF32.filter(try colF32.compare(.gt, 0))) }
let f32Out = UnsafeMutablePointer<Float>.allocate(capacity: rows)
time("CPU \(cores)-core compare then filter", bytes: bytesF32, section: sec) {
    let p = opaque(colF32).valuePointer
    _ = parallelChunks(rows) { lo, hi in
        var i = lo
        while i < hi { var b: UInt8 = 0; let lim = min(8, hi - i); for j in 0..<lim where p[i + j] > 0 { b |= 1 << j }; cmpOut[i >> 3] = b; i += 8 }
        return 0
    }
    sink(cpuFilterParallel(p, n: rows, sel: cmpOut, into: f32Out))
}

// Markdown table for the README.
print("\n\n| Operation | Implementation | Time (ms) | Throughput (GB/s) |\n|---|---|---:|---:|")
for (s, l, ms, gb) in results { print("| \(s) | \(l) | \(String(format: "%.2f", ms)) | \(String(format: "%.1f", gb)) |") }
