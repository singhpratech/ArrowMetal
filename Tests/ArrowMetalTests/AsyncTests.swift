import XCTest
@testable import ArrowMetal

/// A one-way flag readable from another thread.
private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var v = false
    func set() { lock.lock(); v = true; lock.unlock() }
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return v }
}

/// Spins the current thread, counting iterations, until `flag` is set or `limitNanos` elapse.
/// Both the calibration run and the measured run use this exact loop so the rates are comparable.
private func spinCounting(until flag: Flag, limitNanos: UInt64) -> (spins: Int, nanos: UInt64) {
    let start = DispatchTime.now().uptimeNanoseconds
    var spins = 0
    var now = start
    while !flag.isSet {
        spins &+= 1
        now = DispatchTime.now().uptimeNanoseconds
        if now &- start >= limitNanos { break }
    }
    return (spins, Swift.max(now &- start, 1))
}

final class AsyncTests: XCTestCase {

    // MARK: The pitch: the GPU does the scan, the CPU does your app

    func testCPUIsFreeWhileGPUWorks() async throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let n = 50_000_000
        let a = try MetalArray<Int32>.allocate(length: n, withValidity: false)
        let p = a.mutableValuePointer
        for i in 0..<n { p[i] = Int32(i % 1001) - 500 }

        // Synchronous reference. Also warms the pipeline cache, so the async runs measure GPU work
        // and not one-off shader compilation.
        let reference = try ctx.batch { try a.filter(where: .gt, 0).multiply(2) }
        let referenceLength = reference.length
        let referenceSum = try reference.sum()
        XCTAssertFalse(ctx.isBatching)
        XCTAssertEqual(ctx.openBatches.value, 0)

        // How fast does this thread spin with nothing else going on? Same loop shape as below, so the
        // two rates are comparable.
        let (calibrationSpins, calibrationNanos) = spinCounting(until: Flag(), limitNanos: 50_000_000)
        let calibrationRate = Double(calibrationSpins) / Double(calibrationNanos)
        XCTAssertGreaterThan(calibrationSpins, 0)

        // 1. Same-thread proof, callback form. The call returns as soon as the kernels are recorded and
        //    committed, so *this* thread — the one that entered batchAsync — is free to spin while the
        //    GPU scans 50M rows. Anything that waited on the GPU would set the flag before returning and
        //    this loop would count zero.
        let doneCallback = Flag()
        var callbackResult: Result<MetalArray<Int32>, Error>? = nil
        ctx.batchAsync({ try a.filter(where: .gt, 0).multiply(2) }) { r in
            callbackResult = r
            doneCallback.set()
        }
        XCTAssertFalse(ctx.isBatching, "batchAsync must not leave a batch open on the calling thread")
        let (callbackSpins, callbackNanos) = spinCounting(until: doneCallback, limitNanos: 60_000_000_000)
        XCTAssertTrue(doneCallback.isSet, "callback batch did not complete within 60s")
        let callbackRate = Double(callbackSpins) / Double(callbackNanos)
        XCTAssertGreaterThan(callbackSpins, 50_000, "calling thread made no progress; it was blocked")
        XCTAssertGreaterThan(callbackRate, calibrationRate * 0.25,
                             "calling thread ran at \(callbackRate) spins/ns vs \(calibrationRate) idle")
        let fromCallback = try XCTUnwrap(callbackResult).get()
        XCTAssertFalse(fromCallback.pending, "the completion path resolves the pending length")
        XCTAssertEqual(fromCallback.length, referenceLength)
        XCTAssertEqual(try fromCallback.sum(), referenceSum)

        // 2. Swift concurrency form: the same work awaited from a task, with the calling task spinning.
        let doneAwait = Flag()
        let work = Task.detached { () -> (Int, SumResult?, Bool) in
            defer { doneAwait.set() }
            let kept = try await ctx.batchAsync { try a.filter(where: .gt, 0).multiply(2) }
            // Resolved by the completion handler: reading the length here does not touch the GPU.
            let resolved = !kept.pending
            let s = try await kept.sumAsync()
            return (kept.length, s, resolved)
        }
        let (spins, nanos) = spinCounting(until: doneAwait, limitNanos: 60_000_000_000)
        let rate = Double(spins) / Double(nanos)

        let (length, sum, resolved) = try await work.value
        XCTAssertTrue(doneAwait.isSet, "async batch did not complete within 60s")
        XCTAssertEqual(length, referenceLength)
        XCTAssertEqual(sum, referenceSum)
        XCTAssertTrue(resolved, "the pending filter result should be resolved when the await returns")
        XCTAssertGreaterThan(spins, 50_000, "calling task made almost no progress")
        XCTAssertGreaterThan(rate, calibrationRate * 0.25,
                             "calling task ran at \(rate) spins/ns vs \(calibrationRate) idle")

        XCTAssertFalse(ctx.isBatching)
        XCTAssertEqual(ctx.openBatches.value, 0)
    }

    // MARK: Results match the synchronous path

    func testAsyncResultsMatchSynchronousPath() async throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 1000, 300_000] {
            var vals: [Int64?] = []
            for _ in 0..<n { vals.append(Int.random(in: 0..<8, using: &g) == 0 ? nil : Int64.random(in: -500...500, using: &g)) }
            let a = try MetalArray<Int64>(vals)

            let syncKept = try ctx.batch { try a.filter(try a.compare(.gt, 0)) }
            let syncSum = try syncKept.sum()
            let syncMean = try syncKept.mean()

            let asyncKept = try await ctx.batchAsync { try a.filter(try a.compare(.gt, 0)) }
            XCTAssertFalse(asyncKept.pending, "n=\(n)")
            XCTAssertEqual(asyncKept.length, syncKept.length, "n=\(n)")
            XCTAssertEqual(asyncKept.nullCount, syncKept.nullCount, "n=\(n)")
            XCTAssertEqual(asyncKept.toArray(), syncKept.toArray(), "n=\(n)")
            let keptSum = try await asyncKept.sumAsync()
            let keptMean = try await asyncKept.meanAsync()
            let wholeSum = try await a.sumAsync()
            XCTAssertEqual(keptSum, syncSum, "n=\(n)")
            XCTAssertEqual(keptMean, syncMean, "n=\(n)")
            XCTAssertEqual(wholeSum, try a.sum(), "n=\(n)")

            XCTAssertFalse(ctx.isBatching)
            XCTAssertEqual(ctx.openBatches.value, 0)
        }
    }

    // MARK: Errors surface from the await

    func testDeferredErrorThrowsFromAwait() async throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let a = try MetalArray<Int32>([1, 2, 3])

        // The bounds check runs on the completion path, so it must be reported by the await.
        do {
            _ = try await ctx.batchAsync { try a.take(try MetalArray<Int32>([0, 7])) }
            XCTFail("out-of-range take should throw from the await")
        } catch let e as ArrowMetalError {
            guard case .invalidArrowArray = e else { return XCTFail("unexpected error \(e)") }
        }
        XCTAssertFalse(ctx.isBatching)
        XCTAssertEqual(ctx.openBatches.value, 0)

        // An error raised while recording drains the partial batch and rethrows too.
        do {
            _ = try await ctx.batchAsync { try a.filter(try MetalBooleanArray([true, false])) }
            XCTFail("length mismatch should throw")
        } catch let e as ArrowMetalError {
            guard case .lengthMismatch = e else { return XCTFail("unexpected error \(e)") }
        }
        XCTAssertFalse(ctx.isBatching)
        XCTAssertEqual(ctx.openBatches.value, 0)

        // The context is still usable afterwards.
        let ok = try await ctx.batchAsync { try a.take(try MetalArray<Int32>([2, 0])) }
        XCTAssertEqual(ok.toRawArray(), [3, 1])
        XCTAssertEqual(ctx.openBatches.value, 0)
    }

    // MARK: Completion-handler form

    func testCompletionHandlerForm() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let a = try MetalArray<Int32>((0..<10_000).map { $0 % 5 == 0 ? nil : Int32($0) })
        let expected = (0..<10_000).filter { $0 % 5 != 0 && $0 > 100 }

        let ready = expectation(description: "filter completed")
        var got: Result<MetalArray<Int32>, Error>? = nil
        ctx.batchAsync({ try a.filter(where: .gt, 100) }) { r in
            got = r
            ready.fulfill()
        }
        // Returned without waiting: nothing is open on this thread any more.
        XCTAssertFalse(ctx.isBatching)
        wait(for: [ready], timeout: 30)
        XCTAssertEqual(try got?.get().toRawArray(), expected.map { Int32($0) })
        XCTAssertEqual(ctx.openBatches.value, 0)

        // Errors arrive as .failure.
        let failed = expectation(description: "take failed")
        var err: Error? = nil
        ctx.batchAsync({ try a.take(try MetalArray<Int32>([99_999])) }) { (r: Result<MetalArray<Int32>, Error>) in
            if case .failure(let e) = r { err = e }
            failed.fulfill()
        }
        wait(for: [failed], timeout: 30)
        XCTAssertNotNil(err)
        XCTAssertEqual(ctx.openBatches.value, 0)
    }

    // MARK: Nesting

    func testNestedUse() async throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let a = try MetalArray<Int32>((0..<5_000).map { Int32($0) })
        let expected = (1_000..<5_000).map { Int32($0) }

        // A synchronous batch nested inside an async one joins it: one command buffer, one round trip.
        let kept = try await ctx.batchAsync { () -> MetalArray<Int32> in
            try ctx.batch { () -> MetalArray<Int32> in
                XCTAssertTrue(ctx.isBatching)
                let r = try a.filter(where: .ge, 1_000)
                XCTAssertTrue(r.pending, "nested batch must not flush")
                return r
            }
        }
        XCTAssertEqual(kept.toRawArray(), expected)
        XCTAssertFalse(ctx.isBatching)
        XCTAssertEqual(ctx.openBatches.value, 0)

        // The completion form nested in an open batch joins it and calls back synchronously, leaving
        // the outer batch to actually run the work.
        var innerResult: MetalArray<Int32>? = nil
        var calledBeforeReturn = false
        let outer = try ctx.batch { () -> MetalArray<Int32> in
            var called = false
            ctx.batchAsync({ try a.filter(where: .ge, 1_000) }) { r in
                called = true
                innerResult = try? r.get()
            }
            calledBeforeReturn = called
            XCTAssertTrue(ctx.isBatching, "nested completion form must leave the outer batch open")
            return try a.filter(where: .lt, 10)
        }
        XCTAssertTrue(calledBeforeReturn)
        XCTAssertEqual(innerResult?.toRawArray(), expected)
        XCTAssertEqual(outer.toRawArray(), (0..<10).map { Int32($0) })
        XCTAssertFalse(ctx.isBatching)
        XCTAssertEqual(ctx.openBatches.value, 0)
    }
}
