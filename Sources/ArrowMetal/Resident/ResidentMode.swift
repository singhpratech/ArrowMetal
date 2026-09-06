import Foundation
import Metal

/// Persistent ("resident") GPU worker mode, and the experiment that shows why it does not exist.
///
/// The idea: keep one compute dispatch alive whose threadgroups spin on a ring buffer of op
/// descriptors in unified memory. The CPU writes a descriptor and polls a completion flag, so a
/// small kernel costs a memory round trip instead of a command-buffer round trip (~65 µs on M4 Max).
///
/// It does not work on Apple silicon. A running kernel and the CPU are not coherent through
/// `.storageModeShared` memory: neither side's stores are guaranteed to reach the other before the
/// dispatch ends, and in practice they only do so when something evicts the line from the GPU's
/// cache — at an unbounded, effectively random latency. `ResidentProbe` reproduces it; the numbers
/// and the reasoning are in docs/RESIDENT.md.
///
/// What replaced it is in `MetalContext`: `lowLatencyWait` (spin on an `MTLSharedEvent` the command
/// buffer signals) and lazily generated MSL, which together took a 1,000-row `sum` from 114 µs to
/// 88 µs and a batched chain of ten ops from 137 µs to 95 µs.
public enum ResidentMode {
    /// Always false on Apple silicon. See `unavailableReason` and docs/RESIDENT.md.
    public static var available: Bool { false }

    public static let unavailableReason = """
        A Metal compute kernel and the CPU are not cache coherent through shared storage while the \
        kernel is running. MSL offers only memory_order_relaxed and no system-scope atomics. Over a \
        3 s window with a doorbell every 5 ms (599 stores), a spinning kernel observed at best 5 of \
        them, the first after 420 ms to 1.4 s; with device atomics it observed none at all. In the \
        other direction the CPU first saw the kernel's heartbeat somewhere between 33 ms and 2.0 s. \
        Against a ~65 µs command-buffer round trip that is four orders of magnitude too slow, and it \
        is eviction-driven, so it has no bound. Run ResidentProbe.run() to reproduce.
        """
}

/// Result of one persistent-kernel coherence experiment.
public struct ResidentProbeResult: Sendable {
    /// How long the spinning dispatch ran, wall clock.
    public let dispatchMilliseconds: Double
    /// How long the CPU interacted with the running kernel.
    public let windowMilliseconds: Double
    /// Doorbell values the CPU wrote into shared memory while the kernel was spinning.
    public let doorbellsWrittenByCPU: Int
    /// Distinct doorbell values the kernel actually observed (read back after the dispatch ended,
    /// so this measurement does not itself depend on GPU -> CPU visibility).
    public let doorbellsObservedByGPU: Int
    /// The first few values the kernel saw, in order.
    public let observedValues: [UInt32]
    /// Whether the CPU ever saw the kernel's heartbeat advance while the kernel was running.
    public let heartbeatSeenByCPU: Bool
    /// Time from the start of the window to the first heartbeat the CPU saw, if any.
    public let heartbeatFirstSeenMicroseconds: Double?
    /// Spin iterations the kernel completed.
    public let iterations: UInt32
    /// Process CPU seconds consumed while the kernel was spinning (a proxy for the power a resident
    /// worker would cost: there is no user-space GPU power counter without `sudo powermetrics`).
    public let cpuSecondsDuringSpin: Double

    /// True when a CPU store reached the running kernel at all.
    public var cpuStoresReachedRunningGPU: Bool { doorbellsObservedByGPU > 0 }
    /// True when a kernel store reached the CPU while the kernel was still running.
    public var gpuStoresReachedRunningCPU: Bool { heartbeatSeenByCPU }
    /// The condition a resident worker needs: both directions visible during the dispatch.
    public var residentWorkerViable: Bool { cpuStoresReachedRunningGPU && gpuStoresReachedRunningCPU }

    public var summary: String {
        """
        resident-worker probe on a \(String(format: "%.0f", dispatchMilliseconds)) ms dispatch \
        (\(iterations) spin iterations, \(String(format: "%.0f", windowMilliseconds)) ms CPU window):
          CPU -> running GPU : \(cpuStoresReachedRunningGPU ? "visible" : "NOT visible") \
        (\(doorbellsWrittenByCPU) doorbells written, \(doorbellsObservedByGPU) observed\
        \(observedValues.isEmpty ? "" : ", first: \(observedValues)"))
          running GPU -> CPU : \(gpuStoresReachedRunningCPU ? "visible" : "NOT visible")\
        \(heartbeatFirstSeenMicroseconds.map { String(format: " (first at %.0f µs)", $0) } ?? "")
          viable resident worker: \(residentWorkerViable ? "YES" : "no")
          CPU seconds burned while spinning: \(String(format: "%.3f", cpuSecondsDuringSpin))
        """
    }
}

/// Runs a single-threadgroup kernel that spins on a shared-storage buffer and measures, in both
/// directions independently, whether stores made during the dispatch are visible to the other side.
///
/// The CPU -> GPU direction is logged inside the kernel and read back *after* the dispatch ends, so
/// it is measured even when the GPU -> CPU direction is broken. The GPU -> CPU direction is polled
/// while the kernel runs.
///
/// The control buffer is page sized and the doorbell, the heartbeat and the log sit in separate
/// 128-byte lines: when they shared a line, the GPU's eventual write-back clobbered the CPU's store.
public enum ResidentProbe {
    /// The kernel spins a bounded number of iterations rather than waiting for a quit flag, because
    /// the quit flag is exactly what it cannot see. Sized for roughly 175 ms on an M4 Max; an
    /// unbounded spin here would run for minutes (the longest measured was 382 s, uninterrupted).
    public static let defaultIterations: UInt32 = 2_000_000

    /// `volatile device coherent(device)` plus a device-scope fence is the *most* permissive
    /// combination MSL offers, and the only family that ever let a CPU store through in the sweep:
    /// plain `device`, `atomic_load_explicit`, `atomic_fetch_add` and `atomic_exchange` observed
    /// nothing at all over three seconds. The probe therefore uses the best case, not a straw man.
    public static let qualifier = "volatile device coherent(device) uint* + threadgroup_barrier(mem_device)"

    static let source = """
    #include <metal_stdlib>
    using namespace metal;
    // ctl (uint indices; the two directions are 128 B apart so they never share a cache line —
    // when they shared one, the GPU's eventual write-back clobbered the CPU's store):
    //    0 : doorbell   CPU -> GPU
    //   32 : heartbeat  GPU -> CPU
    // out:
    //    0 : iterations, 1 : distinct doorbell values observed, 2 : last value observed
    //   16.. : the first observed values, in order
    kernel void resident_probe(volatile device coherent(device) uint* ctl [[buffer(0)]],
                               device uint* out [[buffer(1)]],
                               constant uint& maxIters [[buffer(2)]],
                               uint tid [[thread_position_in_threadgroup]]) {
        if (tid != 0) return;
        uint iters = 0, lastSeen = 0, changes = 0;
        while (iters < maxIters) {
            iters++;
            uint d = ctl[0];
            if (d != lastSeen) {
                lastSeen = d;
                if (changes < 16u) out[16u + changes] = d;
                changes++;
            }
            ctl[32] = iters;
            threadgroup_barrier(mem_flags::mem_device);
        }
        out[0] = iters; out[1] = changes; out[2] = lastSeen;
    }
    """

    /// Runs the experiment. Returns nil when the device cannot run it (virtualised GPU).
    public static func run(context: MetalContext = .shared,
                           iterations: UInt32 = defaultIterations,
                           windowMilliseconds: Double = 120,
                           doorbellIntervalMicroseconds: UInt64 = 500) throws -> ResidentProbeResult? {
        if context.isVirtualDevice { return nil }
        let pso = try context.pipeline(source: source, function: "resident_probe", cacheKey: "resident/probe")

        let ctl = try MetalArrowBuffer.allocate(byteCount: 4096, context: context)
        let out = try MetalArrowBuffer.allocate(byteCount: 4096, context: context)
        let c = ctl.mutableTyped(UInt32.self)
        let o = out.mutableTyped(UInt32.self)

        guard let cb = context.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw ArrowMetalError.noMetalDevice
        }
        var iters = iterations
        enc.setComputePipelineState(pso)
        enc.setBuffer(ctl.mtl, offset: 0, index: 0)
        enc.setBuffer(out.mtl, offset: 0, index: 1)
        enc.setBytes(&iters, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()

        let started = DispatchTime.now().uptimeNanoseconds
        let cpu0 = processCPUSeconds()
        cb.commit()

        // Interact with the running kernel: raise a doorbell on a fixed interval, watch the heartbeat.
        var doorbell: UInt32 = 0
        var heartbeatSeen = false
        var heartbeatFirst: Double? = nil
        var lastHeartbeat: UInt32 = 0
        var lastWrite: UInt64 = 0
        let windowNs = UInt64(windowMilliseconds * 1_000_000)
        while true {
            let now = DispatchTime.now().uptimeNanoseconds
            let elapsed = now - started
            if elapsed >= windowNs { break }
            if elapsed - lastWrite >= doorbellIntervalMicroseconds * 1000 {
                lastWrite = elapsed
                doorbell &+= 1
                OSMemoryBarrier()
                c[0] = doorbell
                OSMemoryBarrier()
            }
            OSMemoryBarrier()
            let h = c[32]
            if h != lastHeartbeat {
                lastHeartbeat = h
                if !heartbeatSeen { heartbeatSeen = true; heartbeatFirst = Double(elapsed) / 1000.0 }
            }
            if cb.status == .completed || cb.status == .error { break }
        }
        let cpuBurned = processCPUSeconds() - cpu0
        cb.waitUntilCompleted()
        let dispatchMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6

        let changes = Int(o[1])
        var observed: [UInt32] = []
        for k in 0..<Swift.min(changes, 16) { observed.append(o[16 + k]) }
        return ResidentProbeResult(dispatchMilliseconds: dispatchMs,
                                   windowMilliseconds: windowMilliseconds,
                                   doorbellsWrittenByCPU: Int(doorbell),
                                   doorbellsObservedByGPU: changes,
                                   observedValues: observed,
                                   heartbeatSeenByCPU: heartbeatSeen,
                                   heartbeatFirstSeenMicroseconds: heartbeatFirst,
                                   iterations: o[0],
                                   cpuSecondsDuringSpin: cpuBurned)
    }
}

/// Process CPU seconds (user + system) across all threads.
func processCPUSeconds() -> Double {
    var ru = rusage()
    getrusage(RUSAGE_SELF, &ru)
    return Double(ru.ru_utime.tv_sec) + Double(ru.ru_utime.tv_usec) / 1e6
         + Double(ru.ru_stime.tv_sec) + Double(ru.ru_stime.tv_usec) / 1e6
}

extension MetalContext {
    /// Whether ops are routed through a persistent GPU worker. Always false: see `ResidentMode`.
    public var residentMode: Bool { ResidentMode.available }

    /// Requests resident mode. Returns whether it took effect, which on Apple silicon is always
    /// false; `ResidentMode.unavailableReason` says why. Kept so callers and the C ABI can ask.
    @discardableResult
    public func setResidentMode(_ on: Bool) -> Bool { on ? false : true }
}
