import Foundation
import Metal

// MARK: - Non-blocking batches
//
// `batch { }` records kernels into one command buffer and then blocks the calling thread until the GPU
// is done. `batchAsync` records exactly the same way, but instead of waiting it hangs a completion
// handler off the command buffer: the calling thread returns immediately and the deferred fix-ups
// (`afterFlush` closures, `pool.releaseParked()`, the `openBatches` decrement) run on the GPU's
// completion thread. That is the whole point of the library — the GPU does the scan, the CPU does
// the app — so nothing here spins or waits.

extension MetalContext {
    /// Non-blocking `batch { }`: records `body`'s kernels into one command buffer, commits it, and
    /// suspends the calling task until the GPU reports completion. The calling thread is released
    /// while the GPU works; no spinning, no `waitUntilCompleted`.
    ///
    /// `body` runs synchronously, on the thread that entered this call, before the first suspension —
    /// batches are per-thread, so the batch is opened, filled and detached without ever hopping threads.
    ///
    /// **What `body` should return.** Sync points still exist *inside* `body`: anything that reads a GPU
    /// result on the CPU — a reduction (`sum`, `min`, `max`, `mean`), `length` or `nullCount` of a pending
    /// filter result, a subscript, `toArray`, an export — commits the open batch and blocks right there,
    /// exactly as it does in `batch { }`. So return something that does *not* force a sync: a pending
    /// array from `filter`, `compare`, `take`, `cast` or arithmetic. Its `length`, `nullCount` and contents
    /// are resolved by the time the `await` returns, so reading them afterwards costs nothing.
    /// For a scalar, use an async accessor such as ``MetalArray/sumAsync()`` instead of calling `sum()`
    /// inside `body`.
    ///
    /// Errors from deferred checks (an out-of-range `take` index, a failed command buffer) are thrown
    /// from the `await`, not from the recording. Nested calls join the enclosing batch and return as soon
    /// as `body` does, leaving the outer `batch`/`batchAsync` to run and finish the work.
    public func batchAsync<R>(_ body: () throws -> R) async throws -> R {
        if currentBatch != nil { return try body() }   // nested: the outer batch owns the fix-ups
        try openBatch()
        let result: R
        do {
            result = try body()
        } catch {
            // Recording failed part way. Still run the partially recorded batch to completion so the
            // pool and `openBatches` stay balanced, then report the original error.
            try? await commitAndFinish()
            throw error
        }
        try await commitAndFinish()
        return result
    }

    /// Callback form of ``batchAsync(_:)`` for callers that are not in an async context (AppKit/UIKit
    /// event handlers, C ABI shims, GCD code). Records `body` synchronously, commits, and returns
    /// immediately; `completion` is invoked on a Metal-owned completion thread once the GPU is done and
    /// the deferred fix-ups have run. The same guidance about what `body` should return applies.
    ///
    /// When called inside an open batch the work joins it and `completion` fires synchronously, before
    /// this function returns, with the value `body` produced.
    public func batchAsync<R>(_ body: () throws -> R, completion: @escaping (Result<R, Error>) -> Void) {
        if currentBatch != nil {
            do { completion(.success(try body())) } catch { completion(.failure(error)) }
            return
        }
        let result: R
        do {
            try openBatch()
            result = try body()
        } catch {
            finish(detachBatch()) { _ in completion(.failure(error)) }
            return
        }
        finish(detachBatch()) { finishError in
            if let e = finishError { completion(.failure(e)) } else { completion(.success(result)) }
        }
    }

    /// Commits the open batch (if any) and suspends until the GPU has finished with it, then runs the
    /// deferred fix-ups and rethrows the first error they produced.
    private func commitAndFinish() async throws {
        guard let b = detachBatch() else { return }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            // The handler must be registered before `commit()`.
            b.commandBuffer.addCompletedHandler { _ in continuation.resume() }
            b.commandBuffer.commit()
        }
        try finishBatch(b)
    }

    /// Commits `b` (if non-nil) and calls `then` with the fix-up error, if any, once the GPU is done.
    /// With no batch to run, `then` is called straight away.
    private func finish(_ b: Batch?, then: @escaping (Error?) -> Void) {
        guard let b else { then(nil); return }
        b.commandBuffer.addCompletedHandler { [self] _ in
            do { try finishBatch(b); then(nil) } catch { then(error) }
        }
        b.commandBuffer.commit()
    }
}

// MARK: - Non-blocking scalar results

extension MetalArray {
    /// ``sum()`` without blocking the calling thread: the reduction kernel is recorded in a
    /// ``MetalContext/batchAsync(_:)``, and the per-threadgroup partials are combined once the GPU
    /// completion handler has resumed the task. Returns nil when there are no valid values, matching
    /// Arrow semantics and ``sum()`` exactly.
    ///
    /// Inside an already-open batch there is nothing to overlap with, so this falls back to the
    /// synchronous ``sum()`` (which syncs the enclosing batch and keeps it open).
    public func sumAsync() async throws -> SumResult? {
        if context.isBatching { return try sum() }
        if !pending && validCount == 0 { return nil }
        let (partials, counts, groups) = try await context.batchAsync { try self.recordReduction("reduce_sum") }
        if validCount == 0 { return nil }   // all-null after a pending filter (now resolved)
        return finaliseSum(partials, counts, groups)
    }

    /// ``mean()`` without blocking the calling thread.
    public func meanAsync() async throws -> Double? {
        guard let s = try await sumAsync() else { return nil }
        return s.asDouble / Double(validCount)
    }
}
