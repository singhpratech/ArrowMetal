import XCTest
@testable import ArrowMetal

/// Covers the resident-worker investigation (docs/RESIDENT.md) and the low-latency dispatch path
/// that replaced it: correctness of many small ops, equivalence of the two wait strategies, and an
/// opt-in latency benchmark.
final class ResidentTests: XCTestCase {

    // MARK: The negative result

    func testResidentModeIsUnavailable() throws {
        let ctx = MetalContext.shared
        XCTAssertFalse(ResidentMode.available)
        XCTAssertFalse(ctx.residentMode)
        XCTAssertFalse(ctx.setResidentMode(true), "resident mode must never silently claim to be on")
        XCTAssertTrue(ctx.setResidentMode(false))
        XCTAssertFalse(ResidentMode.unavailableReason.isEmpty)
    }

    /// Reproduces the coherence experiment: a kernel spinning on shared storage does not exchange
    /// stores with the CPU during its dispatch. Asserted loosely on purpose — visibility does
    /// eventually arrive by cache eviction, at a latency far too large and too variable to build on.
    /// What must hold is that the probe runs, the kernel really spins, and the round trip never gets
    /// anywhere near the ~65 µs command-buffer floor it would have to beat.
    func testPersistentKernelCoherenceProbe() throws {
        try requireRealGPU()
        guard let r = try ResidentProbe.run(iterations: 1_000_000, windowMilliseconds: 80) else {
            throw XCTSkip("probe needs a real GPU")
        }
        print("\n" + r.summary + "\n")
        XCTAssertGreaterThan(r.iterations, 0, "the probe kernel did not run")
        XCTAssertGreaterThan(r.doorbellsWrittenByCPU, 10, "the CPU should have written many doorbells")
        if r.residentWorkerViable {
            // Not a failure: record it loudly, because it would mean the platform changed.
            print("NOTE: both directions were visible during the dispatch on this machine. "
                  + "Re-measure the latency before trusting it; docs/RESIDENT.md has the method.")
        } else {
            XCTAssertFalse(r.cpuStoresReachedRunningGPU && r.gpuStoresReachedRunningCPU)
        }
    }

    // MARK: Correctness of the low-latency path

    /// 10,000 small ops through the ordinary path, every result checked against a CPU reference.
    /// This is the guard on `lowLatencyWait`: an event-signalled wait that returned early would show
    /// up here as a stale buffer read.
    func testTenThousandSmallOpsMatchCPUReference() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let previous = ctx.lowLatencyWait
        ctx.lowLatencyWait = true
        defer { ctx.lowLatencyWait = previous }

        var g = SystemRandomNumberGenerator()
        // A handful of columns, reused so the pool and the pipeline cache are warm (the interesting case).
        var columns: [[Int64]] = []
        for _ in 0..<8 {
            columns.append((0..<257).map { _ in Int64.random(in: -1000...1000, using: &g) })
        }
        let arrays = try columns.map { try MetalArray<Int64>($0) }

        var checked = 0
        for i in 0..<10_000 {
            let k = i % arrays.count
            let a = arrays[k], raw = columns[k]
            switch i % 5 {
            case 0:
                let want = raw.reduce(Int64(0), &+)
                XCTAssertEqual(try a.sum(), .int(want), "sum mismatch at op \(i)")
            case 1:
                XCTAssertEqual(try a.min(), raw.min(), "min mismatch at op \(i)")
            case 2:
                XCTAssertEqual(try a.max(), raw.max(), "max mismatch at op \(i)")
            case 3:
                let threshold = Int64(i % 500) - 250
                let got = try a.filter(where: .gt, threshold).toArray().compactMap { $0 }
                XCTAssertEqual(got, raw.filter { $0 > threshold }, "filter mismatch at op \(i)")
            default:
                let got = try a.multiply(3).toArray().compactMap { $0 }
                XCTAssertEqual(got, raw.map { $0 &* 3 }, "multiply mismatch at op \(i)")
            }
            checked += 1
        }
        XCTAssertEqual(checked, 10_000)
    }

    /// The shared-event wait and `waitUntilCompleted` must produce identical results.
    func testLowLatencyWaitMatchesBlockingWait() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let previous = ctx.lowLatencyWait
        defer { ctx.lowLatencyWait = previous }

        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 33, 1000, 65_537] {
            var vals: [Int64?] = []
            for _ in 0..<n { vals.append(Int.random(in: 0..<8, using: &g) == 0 ? nil : Int64.random(in: -500...500, using: &g)) }
            let a = try MetalArray<Int64>(vals)

            func measureAll() throws -> (SumResult?, Int64?, Int64?, [Int64?], [Int64?]) {
                (try a.sum(), try a.min(), try a.max(),
                 try a.filter(where: .gt, 0).toArray(),
                 try a.multiply(2).toArray())
            }
            ctx.lowLatencyWait = false
            let blocking = try measureAll()
            ctx.lowLatencyWait = true
            let fast = try measureAll()
            XCTAssertEqual(blocking.0, fast.0, "sum differs at n=\(n)")
            XCTAssertEqual(blocking.1, fast.1, "min differs at n=\(n)")
            XCTAssertEqual(blocking.2, fast.2, "max differs at n=\(n)")
            XCTAssertEqual(blocking.3, fast.3, "filter differs at n=\(n)")
            XCTAssertEqual(blocking.4, fast.4, "multiply differs at n=\(n)")
        }
    }

    /// The event value is handed out under the commit lock, so concurrent submitters must not
    /// release each other early. Without that lock this test reads buffers the GPU has not written.
    func testConcurrentCallsAreNotReleasedEarly() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let columns: [[Int64]] = (0..<8).map { c in (0..<4096).map { Int64($0 &* (c + 1)) } }
        let arrays = try columns.map { try MetalArray<Int64>($0) }
        let expected = columns.map { $0.reduce(Int64(0), &+) }
        let failures = ManagedAtomicCounter()
        DispatchQueue.concurrentPerform(iterations: 64) { i in
            let k = i % arrays.count
            if (try? arrays[k].sum()) != .int(expected[k]) { failures.increment() }
        }
        XCTAssertEqual(failures.value, 0, "a concurrent caller saw a result its command buffer had not produced")
    }

    /// The lazy MSL source must not change what gets compiled: the same cache key has to keep
    /// resolving to the same kernel across element types.
    func testLazyShaderSourceStillSpecialisesPerType() throws {
        try requireRealGPU()
        let i32 = try MetalArray<Int32>([1, 2, 3, 4])
        let i64 = try MetalArray<Int64>([1, 2, 3, 4])
        let f32 = try MetalArray<Float>([1, 2, 3, 4])
        XCTAssertEqual(try i32.sum(), .int(10))
        XCTAssertEqual(try i64.sum(), .int(10))
        XCTAssertEqual(try f32.sum()?.asDouble, 10)
        XCTAssertEqual(try i32.compare(.gt, 2).toArray(), [false, false, true, true])
        XCTAssertEqual(try i64.compare(.gt, 2).toArray(), [false, false, true, true])
    }

    // MARK: Latency benchmark (opt in with ARROWMETAL_LATENCY_BENCH=1)

    func testLatencyBenchmark() throws {
        try requireRealGPU()
        guard ProcessInfo.processInfo.environment["ARROWMETAL_LATENCY_BENCH"] != nil else {
            throw XCTSkip("set ARROWMETAL_LATENCY_BENCH=1 to run the latency benchmark")
        }
        try LatencyBenchmark.run()
    }
}

/// Per-op latency through the public API, GPU against one CPU core and all cores.
/// Build with `-c release` for meaningful CPU baselines.
enum LatencyBenchmark {
    static var timebase: mach_timebase_info_data_t = {
        var t = mach_timebase_info_data_t(); mach_timebase_info(&t); return t
    }()
    static func microseconds(_ ticks: UInt64) -> Double {
        Double(ticks) * Double(timebase.numer) / Double(timebase.denom) / 1000.0
    }
    static func median(reps: Int = 200, _ body: () throws -> Void) rethrows -> Double {
        for _ in 0..<20 { try body() }
        var v: [Double] = []
        for _ in 0..<reps {
            let t = mach_absolute_time()
            try body()
            v.append(microseconds(mach_absolute_time() - t))
        }
        v.sort()
        return v[v.count / 2]
    }

    @inline(never) static func cpuSum(_ p: UnsafePointer<Int64>, _ n: Int, _ bias: Int64) -> Int64 {
        var a = bias
        for i in 0..<n { a &+= p[i] }
        return a
    }
    @inline(never) static func cpuFilter(_ p: UnsafePointer<Int64>, _ n: Int, _ bias: Int64,
                                         _ out: UnsafeMutablePointer<Int64>) -> Int {
        var k = 0
        for i in 0..<n where p[i] > bias { out[k] = p[i]; k += 1 }
        return k
    }
    @inline(never) static func keep<T>(_ x: T) { withExtendedLifetime(x) {} }
    @inline(never) static func opaque<T>(_ x: T) -> T { x }

    static func run() throws {
        let ctx = MetalContext.shared
        let cores = ProcessInfo.processInfo.activeProcessorCount
        // ARROWMETAL_LOW_LATENCY=0 measures the same build with the old `waitUntilCompleted` path,
        // which isolates the shared-event half of the win from the lazy-shader-source half.
        if ProcessInfo.processInfo.environment["ARROWMETAL_LOW_LATENCY"] == "0" { ctx.lowLatencyWait = false }
        defer { ctx.lowLatencyWait = true }
        print("\nArrowMetal per-op latency on \(ctx.device.name) — median µs, lowLatencyWait=\(ctx.lowLatencyWait)")
        print("  rows        sum   compare    filter  multiply  add(arr)  chain10  chain10-batched   cpu1-sum  cpu1-filter  cpu\(cores)-sum")

        for n in [1_000, 10_000, 100_000, 1_000_000] {
            var g = SystemRandomNumberGenerator()
            let col = try MetalArray<Int64>((0..<n).map { _ in Int64.random(in: -1000...1000, using: &g) })
            let colB = try MetalArray<Int64>((0..<n).map { _ in Int64.random(in: -1000...1000, using: &g) })
            let out = UnsafeMutablePointer<Int64>.allocate(capacity: n)
            defer { out.deallocate() }

            let gSum = try median { keep(try col.sum()) }
            let gCmp = try median { keep(try col.compare(.gt, 0)) }
            let gFil = try median { keep(try col.filter(where: .gt, 0)) }
            let gMul = try median { keep(try col.multiply(3)) }
            let gAdd = try median { keep(try col.add(colB)) }
            let chain = try median(reps: 50) {
                var a = try col.multiply(1)
                for _ in 0..<9 { a = try a.multiply(1) }
                keep(a)
            }
            let chainB = try median(reps: 50) {
                keep(try ctx.batch { () -> MetalArray<Int64> in
                    var a = try col.multiply(1)
                    for _ in 0..<9 { a = try a.multiply(1) }
                    return a
                })
            }
            var tick: Int64 = 0
            let cSum = median { tick &+= 1; keep(cpuSum(opaque(col).valuePointer, n, tick)) }
            let cFil = median { tick &+= 1; keep(cpuFilter(opaque(col).valuePointer, n, tick, out)) }
            let cnSum = median { tick &+= 1; keep(parallelSum(opaque(col).valuePointer, n, cores)) }

            print(String(format: "  %-9d %8.1f %9.1f %9.1f %9.1f %9.1f %8.1f %16.1f %10.1f %12.1f %10.1f",
                         n, gSum, gCmp, gFil, gMul, gAdd, chain, chainB, cSum, cFil, cnSum))
        }
        print("")
    }

    static func parallelSum(_ p: UnsafePointer<Int64>, _ n: Int, _ chunks: Int) -> Int64 {
        let per = (n + chunks - 1) / chunks
        var parts = [Int64](repeating: 0, count: chunks)
        parts.withUnsafeMutableBufferPointer { pb in
            let base = p
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                let lo = c * per, hi = Swift.min(n, lo + per)
                var a: Int64 = 0
                if lo < hi { for i in lo..<hi { a &+= base[i] } }
                pb[c] = a
            }
        }
        return parts.reduce(0, &+)
    }
}
