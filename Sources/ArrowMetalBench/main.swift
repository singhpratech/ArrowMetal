import Foundation
import Accelerate
import ArrowMetal

// ArrowMetal benchmark. GPU kernels vs:
//   - "CPU 1-core": a tight, typed, null-aware Swift loop (what a careful CPU Arrow kernel does)
//   - "CPU all-cores": the same loop split across all cores with concurrentPerform
//   - Accelerate vDSP where an equivalent exists (Float32, no nulls)
// Usage: arrowmetal-bench [rows] [iterations]

let latencyMode = CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "latency"
let rows = CommandLine.arguments.count > 1 && !latencyMode ? Int(CommandLine.arguments[1])! : 50_000_000
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

var results: [(String, String, Double, Double, Double)] = []  // section, label, ms, GB/s, CPU ms

/// Process CPU time (user + system) in seconds: what the operation takes away from the rest of the app.
func cpuSeconds() -> Double {
    var ru = rusage()
    getrusage(RUSAGE_SELF, &ru)
    return Double(ru.ru_utime.tv_sec) + Double(ru.ru_utime.tv_usec) / 1e6 + Double(ru.ru_stime.tv_sec) + Double(ru.ru_stime.tv_usec) / 1e6
}

func time(_ label: String, bytes: Int, section: String, _ body: () throws -> Void) rethrows {
    try body() // warm-up (compiles pipelines)
    var best = Double.infinity
    var bestCPU = Double.infinity
    for _ in 0..<iters {
        let c0 = cpuSeconds()
        let t0 = DispatchTime.now().uptimeNanoseconds
        try body()
        let wall = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        let cpu = cpuSeconds() - c0
        if wall < best { best = wall; bestCPU = cpu }
    }
    let gbps = Double(bytes) / best / 1e9
    print(String(format: "  %-44s %9.2f ms  %7.1f GB/s  %8.1f CPU-ms", (label as NSString).utf8String!, best * 1000, gbps, bestCPU * 1000))
    results.append((section, label, best * 1000, gbps, bestCPU * 1000))
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
if ctx.isVirtualDevice {
    print("ArrowMetal bench: virtual Metal device (\(ctx.device.name)) detected; benchmarks need real Apple silicon. Skipping.")
    exit(0)
}

// ---- Latency mode: fixed cost per call at small sizes, GPU vs 1 core (which is what small sizes get on the CPU side).
if latencyMode {
    print("ArrowMetal latency on \(ctx.device.name): microseconds per call, best of \(max(iters, 20))\n")
    print(String(format: "  %-10@ %12@ %12@ %14@ %14@ %14@ %14@ %14@", "rows" as NSString, "GPU sum" as NSString, "CPU sum" as NSString, "GPU filter" as NSString, "CPU filter" as NSString, "GPU group-by" as NSString, "5-op chain" as NSString, "chain batched" as NSString))
    func best(_ body: () throws -> Void) rethrows -> Double {
        try body()
        var b = Double.infinity
        for _ in 0..<max(iters, 20) { let t0 = DispatchTime.now().uptimeNanoseconds; try body(); b = min(b, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e3) }
        return b
    }
    for n in [1_000, 10_000, 100_000, 1_000_000, 10_000_000] {
        var gen = SystemRandomNumberGenerator()
        let col = try MetalArray<Int64>((0..<n).map { _ in Int64.random(in: -1000...1000, using: &gen) })
        let keys = try MetalArray<Int32>((0..<n).map { _ in Int32.random(in: 0..<16, using: &gen) })
        let gb = try keys.groupBy(keyCount: 16)
        let out = UnsafeMutablePointer<Int64>.allocate(capacity: n)
        let gs = try best { sink(try col.sum()) }
        let cs = best { sink(cpuSumInt64(opaque(col), 0, n)) }
        let gf = try best { sink(try col.filter(where: .gt, 0)) }
        let cf = best { let p = opaque(col).valuePointer; var k = 0; for i in 0..<n where p[i] > 0 { out[k] = p[i]; k += 1 }; sink(k) }
        let gg = try best { sink(try gb.sum(col)) }
        // A 5-kernel chain: two compares, and, filter, sum. Unbatched = 5 round trips; batched = 1.
        let chain = try best { sink(try col.filter(try col.compare(.gt, -500).and(try col.compare(.lt, 500))).sum()) }
        let chainB = try best { sink(try ctx.batch { try col.filter(try col.compare(.gt, -500).and(try col.compare(.lt, 500))).sum() }) }
        print(String(format: "  %-10d %9.0f µs %9.0f µs %11.0f µs %11.0f µs %11.0f µs %11.0f µs %11.0f µs", n, gs, cs, gf, cf, gg, chain, chainB))
        out.deallocate()
    }
    print("\nFixed cost per GPU call is the small-row number; the crossover with one CPU core is where the columns meet.")
    exit(0)
}

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
try time("Metal  same, batched (one command buffer)", bytes: rows * (4 + 4), section: sec) {
    sink(try ctx.batch { try amountF.filter(try keys5.compare(.eq, 2).and(try amountF.compare(.gt, 100))).sum() })
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

// ---------------------------------------------------------------------------------------------------
// Sorting: GPU LSD radix sort vs an all-core CPU sort (chunk sort on every core, then a merge tree whose
// merges also run in parallel). Both start from the same Arrow buffer and produce the same permutation.
// ---------------------------------------------------------------------------------------------------

/// Sorts `a[0..<n]` using every core: `cores` chunks sorted in parallel, then a pairwise merge tree.
/// `scratch` must hold `n` elements; the sorted result always ends up in `a`.
func parallelSort<T: Comparable>(_ a: UnsafeMutablePointer<T>, _ scratch: UnsafeMutablePointer<T>, _ n: Int) {
    if n < 2 { return }
    let chunks = max(1, min(cores, (n + 65_535) / 65_536))
    let per = (n + chunks - 1) / chunks
    var bounds = (0...chunks).map { min(n, $0 * per) }
    let first = bounds
    DispatchQueue.concurrentPerform(iterations: chunks) { c in
        let lo = first[c], hi = first[c + 1]
        if hi > lo { var b = UnsafeMutableBufferPointer(start: a + lo, count: hi - lo); b.sort() }
    }
    var src = a, dst = scratch
    while bounds.count > 2 {
        let b = bounds
        let runs = b.count - 1
        let pairs = (runs + 1) / 2
        let s = src, d = dst
        DispatchQueue.concurrentPerform(iterations: pairs) { p in
            let lo = b[2 * p], mid = b[2 * p + 1], hi = 2 * p + 2 < b.count ? b[2 * p + 2] : n
            var i = lo, j = mid, k = lo
            while i < mid && j < hi { if s[j] < s[i] { d[k] = s[j]; j += 1 } else { d[k] = s[i]; i += 1 }; k += 1 }
            while i < mid { d[k] = s[i]; i += 1; k += 1 }
            while j < hi { d[k] = s[j]; j += 1; k += 1 }
        }
        var next = (0..<pairs).map { b[2 * $0] }
        next.append(n)
        bounds = next
        swap(&src, &dst)
    }
    if src != a { a.update(from: src, count: n) }
}

/// (key, original index) pair; ordering by key then index makes the CPU sort stable like the GPU one.
struct KeyIndex: Comparable {
    var key: Int64
    var idx: Int32
    static func < (a: KeyIndex, b: KeyIndex) -> Bool { a.key != b.key ? a.key < b.key : a.idx < b.idx }
}

let sortI64 = try MetalArray<Int64>((0..<rows).map { _ in Int64.random(in: Int64.min...Int64.max, using: &g) })
let sortF64 = try MetalArray<Double>((0..<rows).map { _ in Double.random(in: -1e9...1e9, using: &g) })

sec = "argsort(Int64, \(rows) rows, no nulls)"; print("\n" + sec)
try time("Metal  argsort (GPU LSD radix, 8 passes)", bytes: rows * 12, section: sec) { sink(try sortI64.argsort()) }
let kiA = UnsafeMutablePointer<KeyIndex>.allocate(capacity: rows)
let kiB = UnsafeMutablePointer<KeyIndex>.allocate(capacity: rows)
time("CPU \(cores)-core argsort (chunk sort + merge tree)", bytes: rows * 12, section: sec) {
    let p = opaque(sortI64).valuePointer
    _ = parallelChunks(rows) { lo, hi in for i in lo..<hi { kiA[i] = KeyIndex(key: p[i], idx: Int32(i)) }; return 0 }
    parallelSort(kiA, kiB, rows)
    sink(kiA[rows - 1].idx)
}

sec = "sort(Float64, \(rows) rows, no nulls)"; print("\n" + sec)
try time("Metal  sort (radix argsort + take)", bytes: rows * 16, section: sec) { sink(try sortF64.sorted()) }
let dblA = UnsafeMutablePointer<Double>.allocate(capacity: rows)
let dblB = UnsafeMutablePointer<Double>.allocate(capacity: rows)
time("CPU \(cores)-core sort (chunk sort + merge tree)", bytes: rows * 16, section: sec) {
    dblA.update(from: opaque(sortF64).valuePointer, count: rows)
    parallelSort(dblA, dblB, rows)
    sink(dblA[rows - 1])
}
time("Swift  [Double].sort() (1 core, introsort)", bytes: rows * 16, section: sec) {
    dblA.update(from: opaque(sortF64).valuePointer, count: rows)
    var b = UnsafeMutableBufferPointer(start: dblA, count: rows)
    b.sort()
    sink(dblA[rows - 1])
}

sec = "top_k(100 of \(rows) Int64)"; print("\n" + sec)
let topKCount = 100
try time("Metal  top_k (full radix argsort + slice)", bytes: rows * 8, section: sec) { sink(try sortI64.topK(topKCount)) }
time("CPU \(cores)-core top_k (per-core running top-k)", bytes: rows * 8, section: sec) {
    let p = opaque(sortI64).valuePointer
    let parts = parallelChunks(rows) { lo, hi -> [Int64] in
        var best = [Int64](repeating: Int64.min, count: topKCount)
        var thr = Int64.min
        for i in lo..<hi {
            let v = p[i]
            if v > thr {
                var j = 1
                while j < topKCount && best[j] < v { best[j - 1] = best[j]; j += 1 }
                best[j - 1] = v
                thr = best[0]
            }
        }
        return best
    }
    sink(parts.flatMap { $0 }.sorted().suffix(topKCount))
}

// ---------------------------------------------------------------------------------------------------
// Strings (Arrow utf8): predicates, filter, and dictionary-encoded group-by over 10M values.
// ---------------------------------------------------------------------------------------------------

let strRows = min(rows, 10_000_000)
let distinct = 1000
let regions = ["north", "south", "east", "west"]
let vocab = (0..<distinct).map { i in "cust_" + String(format: "%03d", i) + "_" + regions[i % 4] }
var strCodesRaw = [Int32](); strCodesRaw.reserveCapacity(strRows)
var strVals = [String?](); strVals.reserveCapacity(strRows)
for _ in 0..<strRows {
    let c = Int32.random(in: 0..<Int32(distinct), using: &g)
    strCodesRaw.append(c); strVals.append(vocab[Int(c)])
}
let strCol = try MetalStringArray(strVals)
strVals = []
let strBytes = strCol.totalBytes + (strRows + 1) * 4
let sOff = strCol.offsets.typed(Int32.self)
let sDat = strCol.data.typed(UInt8.self)
let strBitmap = UnsafeMutablePointer<UInt8>.allocate(capacity: strRows / 8 + 64)

/// All-core CPU string predicate into a packed Arrow bitmap. kind: 0 equals, 1 starts_with, 2 contains.
func cpuStrPredicate(_ pattern: String, kind: Int) {
    let pat = Array(pattern.utf8)
    pat.withUnsafeBufferPointer { pp in
        let m = pp.count, pb = pp.baseAddress!
        _ = parallelChunks(strRows) { lo, hi -> Int in
            var i = lo
            while i < hi {
                var b: UInt8 = 0
                let lim = min(8, hi - i)
                for j in 0..<lim {
                    let s = Int(sOff[i + j]), len = Int(sOff[i + j + 1]) - s
                    var hit = false
                    if kind == 0 { hit = len == m && memcmp(sDat + s, pb, m) == 0 }
                    else if kind == 1 { hit = len >= m && memcmp(sDat + s, pb, m) == 0 }
                    else if len >= m {
                        var q = s
                        let last = s + len - m
                        while q <= last { if memcmp(sDat + q, pb, m) == 0 { hit = true; break }; q += 1 }
                    }
                    if hit { b |= 1 << j }
                }
                strBitmap[i >> 3] = b
                i += 8
            }
            return 0
        }
    }
}

print("\nstrings: \(strRows) utf8 values, \(distinct) distinct, \(strCol.totalBytes / 1_000_000) MB of bytes")

sec = "string contains(\"north\") over \(strRows) strings (25% hit)"; print("\n" + sec)
try time("Metal  contains", bytes: strBytes, section: sec) { sink(try strCol.contains("north")) }
time("CPU \(cores)-core contains (byte scan)", bytes: strBytes, section: sec) { cpuStrPredicate("north", kind: 2); sink(strBitmap[0]) }

sec = "string starts_with(\"cust_1\") over \(strRows) strings (10% hit)"; print("\n" + sec)
try time("Metal  starts_with", bytes: strBytes, section: sec) { sink(try strCol.startsWith("cust_1")) }
time("CPU \(cores)-core starts_with", bytes: strBytes, section: sec) { cpuStrPredicate("cust_1", kind: 1); sink(strBitmap[0]) }

sec = "string equals(\"cust_042_east\") over \(strRows) strings"; print("\n" + sec)
try time("Metal  equals", bytes: strBytes, section: sec) { sink(try strCol.equals("cust_042_east")) }
time("CPU \(cores)-core equals", bytes: strBytes, section: sec) { cpuStrPredicate("cust_042_east", kind: 0); sink(strBitmap[0]) }

sec = "string filter over \(strRows) strings (~30% kept)"; print("\n" + sec)
let strCodes = try MetalArray<Int32>(strCodesRaw)
let strMask = try strCodes.compare(.lt, Int32(distinct * 3 / 10))       // uniform codes: keeps ~30%
let strSel = try strMask.and(strMask).values
try time("Metal  filter (scan + gather bytes)", bytes: strBytes, section: sec) { sink(try strCol.filter(strMask)) }
let fOffOut = UnsafeMutablePointer<Int32>.allocate(capacity: strRows + 1)
let fDatOut = UnsafeMutablePointer<UInt8>.allocate(capacity: max(1, strCol.totalBytes))
time("CPU \(cores)-core filter (count, prefix, copy)", bytes: strBytes, section: sec) {
    let sel = opaque(strSel).typed(UInt8.self)
    let counts = parallelChunks(strRows) { lo, hi -> (Int, Int) in
        var r = 0, b = 0
        var i = lo
        while i < hi {
            var byte = sel[i >> 3]
            if i + 8 > hi { byte &= UInt8((1 << (hi - i)) - 1) }
            while byte != 0 { let j = byte.trailingZeroBitCount; r += 1; b += Int(sOff[i + j + 1] - sOff[i + j]); byte &= byte - 1 }
            i += 8
        }
        return (r, b)
    }
    var rowOff = [Int](), byteOff = [Int](); var ra = 0, ba = 0
    for (r, b) in counts { rowOff.append(ra); byteOff.append(ba); ra += r; ba += b }
    let chunks = counts.count
    let per = ((strRows + chunks - 1) / chunks + 63) / 64 * 64
    DispatchQueue.concurrentPerform(iterations: chunks) { c in
        let lo = min(strRows, c * per), hi = min(strRows, (c + 1) * per)
        var k = rowOff[c], pos = byteOff[c]
        var i = lo
        while i < hi {
            var byte = sel[i >> 3]
            if i + 8 > hi { byte &= UInt8((1 << (hi - i)) - 1) }
            while byte != 0 {
                let j = byte.trailingZeroBitCount
                let s = Int(sOff[i + j]), len = Int(sOff[i + j + 1]) - s
                fOffOut[k] = Int32(pos); memcpy(fDatOut + pos, sDat + s, len); pos += len; k += 1
                byte &= byte - 1
            }
            i += 8
        }
    }
    fOffOut[ra] = Int32(ba)
    sink(ra)
}

sec = "dictionary_encode + group-by sum over \(strRows) strings (\(distinct) distinct)"; print("\n" + sec)
let strAmounts = try sortI64.slice(offset: 0, length: strRows)
let dictBytes = strBytes + strRows * 8
try time("Metal  dictionary_encode + group-by sum", bytes: dictBytes, section: sec) {
    let (codes, uniq) = try strCol.dictionaryEncode()
    sink(try codes.groupBy(keyCount: uniq.length).sum(strAmounts))
}
let cachedCodes = try strCol.dictionaryEncode().codes
let cachedGB = try cachedCodes.groupBy(keyCount: distinct)
try time("Metal  group-by sum on cached codes (GPU only)", bytes: dictBytes, section: sec) { sink(try cachedGB.sum(strAmounts)) }
time("CPU \(cores)-core hash dictionary + group-by sum", bytes: dictBytes, section: sec) {
    let vp = opaque(strAmounts).valuePointer
    let parts = parallelChunks(strRows) { lo, hi -> [UInt64: Int64] in
        var acc = [UInt64: Int64](minimumCapacity: 4096)
        for i in lo..<hi {
            let s = Int(sOff[i]), e = Int(sOff[i + 1])
            var h: UInt64 = 0xcbf2_9ce4_8422_2325
            for q in s..<e { h = (h ^ UInt64(sDat[q])) &* 0x100_0000_01b3 }
            acc[h, default: 0] &+= vp[i]
        }
        return acc
    }
    var total = [UInt64: Int64](minimumCapacity: 4096)
    for part in parts { for (k, v) in part { total[k, default: 0] &+= v } }
    sink(total.count)
}

// Markdown table for the README.
print("\n\n| Operation | Implementation | Time (ms) | Throughput (GB/s) | CPU time (ms) |\n|---|---|---:|---:|---:|")
for (s, l, ms, gb, cpu) in results { print("| \(s) | \(l) | \(String(format: "%.2f", ms)) | \(String(format: "%.1f", gb)) | \(String(format: "%.1f", cpu)) |") }
