# The resident GPU worker: why it does not work on Apple silicon today, and what took its place

Every unbatched ArrowMetal call is one command buffer, and a command-buffer round trip on an M4 Max
costs about 65 µs of which the GPU runs for 1.7 (1.5 at best). Below roughly a million rows one CPU
core wins, and below 100k the fixed cost is the whole story. The obvious escape is a **resident
worker**: keep one compute dispatch alive whose threadgroups spin on a ring buffer of op descriptors
in unified memory, so submitting an op is a store and collecting a result is a load. No encode, no
commit, no completion notification.

It does not work on this platform. This document records the experiments, the numbers, and the
latency work that replaced it.

Everything here was measured on an **Apple M4 Max, macOS 26.6.2 (25G83), Xcode 26.6**, release build.


## 1. What was attempted

```
                 shared MTLBuffer (.storageModeShared, page aligned)
   CPU  ──write op descriptor──►  ring[i] = { opcode, gpuAddress[], n, params }
                                  ring[i].state = READY          (release)
                                        │
   one long-running dispatch,  threadgroups spin:  while (ring[cursor].state != READY) ;
   never ends                            │        execute; ring[cursor].state = DONE
                                        ▼
   CPU  ◄──poll completion flag──  ring[i].state == DONE          (acquire)
```

Two things have to be true for this to beat 65 µs:

1. a CPU store into shared memory must become visible to a **running** kernel, quickly;
2. a store by that running kernel must become visible to the CPU, quickly.

Both were measured directly. Both fail.


## 2. Coherence: the crux, measured

`Sources/ArrowMetal/Resident/ResidentMode.swift` holds the in-tree reproduction (`ResidentProbe`),
exercised by `ResidentTests.testPersistentKernelCoherenceProbe`. The method separates the two
directions so neither measurement depends on the other:

- **CPU → GPU.** The CPU raises a doorbell counter every 5 ms for 3 s (599 stores). The kernel logs
  every *distinct* value it observes into an output buffer. That log is read **after the dispatch
  ends**, where end-of-dispatch cache flushes make it visible unconditionally. So this direction is
  measured even when the other direction is completely broken.
- **GPU → CPU.** The kernel writes a rising heartbeat. The CPU samples it *while the kernel runs*.

The doorbell and the heartbeat are 128 bytes apart. They must be: when they shared one 64-byte line,
the GPU's eventual write-back **clobbered the CPU's store**, so the mailbox read back as zero after
the kernel exited. That alone disqualifies a naive descriptor layout.

Every memory qualifier MSL offers was swept. 3 s window, 599 doorbells:

| MSL qualifier on the shared buffer | CPU stores the GPU saw | first one seen after | CPU first saw the heartbeat after |
|---|---:|---:|---:|
| `device uint*` (plain) | **0 of 599** | never | 227 ms |
| `volatile device uint*` | 5 of 599 | ~420 ms | 380 ms |
| `device coherent(device) uint*` | 3 of 599 | ~815 ms | 148 ms |
| `volatile device coherent(device) uint*` | 2 of 599 | ~1.36 s | 76 ms |
| `volatile device coherent(device)` + `threadgroup_barrier(mem_flags::mem_device)` | 4 of 599 | ~990 ms | **33 ms** |
| `atomic_load_explicit(…, memory_order_relaxed)` | 2 of 599 | ~1.21 s | 1.21 s |
| `atomic_fetch_add_explicit(…, 0, relaxed)` (device RMW) | **0 of 599** | never | 990 ms |
| device RMW + `mem_device` fence | **0 of 599** | never | **never** |
| `coherent(device)` + device RMW | **0 of 599** | never | 1.99 s |

Read that table twice. The best case is `volatile` + `coherent(device)` + a device fence, and it let
**four of 599 stores through, the first after about a second**. Atomics — the tool the design calls
for — are the *worst*: a device-scope read-modify-write never observed a CPU store at all.

The behaviour is consistent with visibility arriving only when the line is evicted from the GPU's
cache, i.e. driven by unrelated memory traffic. Forcing evictions with a large device-memory sweep on
every poll brings the latency down to about 5–6 ms, reliably — still **two orders of magnitude worse
than the 65 µs command buffer it is meant to replace**, and it burns full memory bandwidth to do it.

There is no architectural fix available:

- MSL exposes exactly two memory orders, `memory_order_relaxed` and `memory_order_seq_cst`, and
  `seq_cst` is **not accepted** for `atomic_load_explicit` on `device` address space (compile error:
  *no matching function for call to 'atomic_load_explicit'*).
- The only coherence scope the language has is `coherent(device)` — GPU-device scope. There is no
  system scope, so there is no way to express "coherent with the CPU".
- `MTLBuffer.didModifyRange` does not apply: it is a managed-storage operation, and shared storage on
  Apple silicon has no such hint. There is no `synchronize` that can run mid-dispatch.

**Conclusion: no MSL qualifier available today delivers the coherence a persistent spinning worker
needs, so it cannot be built on this platform as it stands.** The memory model simply does not promise
CPU/GPU coherence inside a dispatch, and no qualifier delivers it in practice.

`ResidentMode.available` is `false`, `MetalContext.setResidentMode(true)` returns `false`, and
`am_resident_mode(1)` returns `0`. Nothing silently pretends otherwise.


## 3. The GPU watchdog: not the blocker

Worth recording, because it was expected to be the limit and is not.

Kernels that spun 2,000,000,000 iterations ran to completion **uninterrupted for 65.4 s, 369.7 s,
381.2 s and 382.4 s**, with `MTLCommandBuffer.error == nil` and `status == .completed` every time. No
watchdog fired, no `MTLCommandBufferError.timeout`, no device reset, on a Mac with an attached
display driving the desktop.

So the re-arm machinery a resident worker would have needed (bounded loop, relaunch from a completion
handler) is unnecessary — but it is also moot. The GPU is fully occupied for the whole dispatch and
unavailable to anything else, which makes a minutes-long spin a liability rather than a design.


## 4. Where the 65 µs goes

With the worker ruled out, the question becomes how much of ArrowMetal's per-call cost is Metal's and
how much is ours. Per-call stages for an empty (`nop`) kernel, 500 calls, µs:

| stage | min | median |
|---|---:|---:|
| `makeCommandBufferWithUnretainedReferences` | 0.21 | 0.33 |
| `makeComputeCommandEncoder` | 1.33 | 1.71 |
| `setComputePipelineState` + `setBuffer` + `dispatchThreadgroups` | 0.25 | 0.42 |
| `endEncoding` | 0.00 | 0.12 |
| `commit` | 1.08 | 1.58 |
| **`waitUntilCompleted`** | **63.12** | **86.17** |
| total round trip | 67.54 | 90.67 |
| *of which* driver work (`kernelStartTime`..`kernelEndTime`) | 5.21 | 18.25 |
| *of which* GPU execution (`gpuStartTime`..`gpuEndTime`) | **1.50** | **1.71** |

Encoding costs 2.6 µs. The GPU runs for 1.7 µs. Everything else — about 60 µs — is submission and
completion notification inside the driver, and it is not ours to remove.

(docs/BENCHMARKS.md round 4 quotes ~116 µs for the same empty round trip. These runs measure 67.5 µs
min / 90.7 µs median over 500 warm calls in one process; the older figure was taken over 30 calls
including the first ones. The floor is therefore lower than previously recorded, but the shape —
GPU 1.7 µs, everything else driver — is the same.)

Wait strategies, same empty kernel:

| strategy | min | median |
|---|---:|---:|
| `waitUntilCompleted` | 68.62 | 77.83 |
| spin on `MTLCommandBuffer.status` | 61.33 | 79.75 |
| spin 300 µs then `waitUntilCompleted` (what ArrowMetal did) | 61.29 | 80.08 |
| `addCompletedHandler` + `DispatchSemaphore` | 67.33 | 76.50 |
| **`encodeSignalEvent` + CPU spin on `MTLSharedEvent.signaledValue`** | **56.17** | **65.54** |

An `MTLSharedEvent`'s `signaledValue` is readable straight out of memory, so polling it skips the
objc/driver path that `MTLCommandBuffer.status` goes through. That is 12 µs of median, for free.

Can the floor itself move?

| | min | median | CPU µs/op |
|---|---:|---:|---:|
| shared-event spin, idle GPU | 56.67 | 75.58 | 105 |
| same, thread at `QOS_CLASS_USER_INTERACTIVE` | 55.67 | 68.75 | 91 |
| same, a second queue kept busy (GPU never idles) | 29.50 | 68.12 | 116 |

Raising thread QoS is worth ~7 µs of median but is the *caller's* decision, not a library's, so
ArrowMetal does not set it; do it in your own thread if you want it. Keeping the GPU warm improves
the best case a lot and the median not at all.

**Amortisation is the only real lever.** Marginal cost per op when work is not waited on one at a time:

| ops | N separate command buffers, one wait | N dispatches in one command buffer |
|---:|---:|---:|
| 1 | 68.8 µs | 68.8 µs |
| 2 | 38.4 µs/op | 34.3 µs/op |
| 5 | 22.3 µs/op | 13.8 µs/op |
| 10 | 15.6 µs/op | **6.9 µs/op** |
| 20 | 11.5 µs/op | **3.9 µs/op** |

Submission is about 11 µs; the ~57 µs remainder is the completion notification, paid **once** per
wait. Ten dispatches in one command buffer complete in 69 µs total. That is what
`MetalContext.batch { }` already does, and it is the only route below 20 µs per op on this hardware.


## 5. What changed

Three changes, all on the ordinary path. Nothing new is enabled by default that alters results.

1. **`MTLSharedEvent` completion wait** (`MetalContext.lowLatencyWait`, on by default). Every
   synchronous command buffer signals a shared event; the CPU spins on `signaledValue` and falls back
   to `waitUntilCompleted` after `spinMicroseconds`. The event value is taken under `commitLock`
   together with the signal encode and the commit, so values are issued in commit order and
   `signaledValue >= v` really does mean "this command buffer finished" — without that lock a second
   thread could signal a higher value first and release a waiter whose own work had not run
   (`ResidentTests.testConcurrentCallsAreNotReleasedEarly` covers it).

2. **MSL source generated lazily.** `MetalContext.pipeline(source:function:cacheKey:)` and
   `Dispatch.pipeline` now take the source as an `@autoclosure`, and the hot call sites pass the
   generator expression inline instead of binding it to a `let` first. Profiling a 1,000-row `sum`
   showed **20% of the call spent in `CFStringFindWithOptionsAndLocale` and `_StringGuts.append`** —
   `KernelSource.reductions` rebuilding an MSL string for a kernel that was already compiled and
   cached, roughly 20 µs a call. On a cache hit the string is now never built. Applied to reductions,
   compare, filter, arithmetic, take and cast.

3. **A spin that polls instead of timing itself.** The old spin called `DispatchTime.now()` on every
   iteration; the profile put **60% of the call's CPU samples inside `dispatch_time` →
   `mach_absolute_time`, against 4% reading `MTLCommandBuffer.status`**. It now reads
   `mach_absolute_time()` directly and only once per 64 polls.

The 20% and 60% figures come from `sample(1)` on a tight `col.sum()` loop; throughput on that loop
went from 108,781 to 168,190 iterations in 12 s (110 µs → 71 µs per op).


## 6. Measured per-op latency

Same standalone harness against the release library, median of 200, quiet machine. "before" is
the library with `lowLatencyWait = false` and the eager MSL string build; "after" is 0.1.0.
`ARROWMETAL_LOW_LATENCY=0` reproduces the event-wait half only; the lazy MSL build has no runtime switch.

| rows | op | before | after | Δ | 1 CPU core | 16 CPU cores |
|---:|---|---:|---:|---:|---:|---:|
| 1,000 | sum | 114.4 | **88.4** | −23% | 0.1 | 0.5 |
| 1,000 | compare | 93.3 | **70.1** | −25% | | |
| 1,000 | filter | 92.2 | **81.6** | −11% | 0.5 | |
| 1,000 | multiply | 81.7 | **69.7** | −15% | | |
| 1,000 | add (array) | 81.8 | **70.5** | −14% | | |
| 1,000 | 10 ops unbatched | 820.0 | **695.5** | −15% | | |
| 1,000 | **10 ops batched** | 137.4 | **95.0** | −31% | | |
| 10,000 | sum | 101.2 | **69.1** | −32% | 0.6 | 1.0 |
| 10,000 | compare | 80.8 | **68.8** | −15% | | |
| 10,000 | filter | 92.7 | **81.6** | −12% | 6.3 | |
| 10,000 | 10 ops batched | 138.1 | **96.7** | −30% | | |
| 100,000 | sum | 101.2 | **68.5** | −32% | 7.2 | 8.2 |
| 100,000 | compare | 87.8 | **75.6** | −14% | | |
| 100,000 | filter | 101.9 | **86.7** | −15% | 192.2 | |
| 100,000 | 10 ops batched | 157.1 | **116.0** | −26% | | |
| 1,000,000 | sum | 126.8 | **97.0** | −23% | 78.8 | 74.5 |
| 1,000,000 | compare | 84.8 | **83.2** | −2% | | |
| 1,000,000 | filter | 118.1 | **107.8** | −9% | 2068.6 | |
| 1,000,000 | 10 ops batched | 370.1 | **330.7** | −11% | | |

Repeated three times interleaved (before, after, before, after, before, after) on a thermally loaded
machine to rule out drift: absolute numbers rise about 30% across the board, the deltas hold — sum
−16% to −24%, batched chain −16% to −19%, arithmetic −8% to −17%.

Reproduce in-tree:

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  ARROWMETAL_LATENCY_BENCH=1 swift test -c release --filter "ResidentTests/testLatencyBenchmark"
# ARROWMETAL_LOW_LATENCY=0 runs the same build with waitUntilCompleted, isolating the event half
```
The XCTest-hosted benchmark reads 15–30% higher than the standalone harness because of the test
process; use it for relative comparisons.

### Against the targets

- **≤ 20 µs for one small op, synchronously, from Swift: not reachable.** The platform floor for a
  single command-buffer round trip is 56 µs best case and ~65–69 µs typical, of which 1.7 µs is the
  GPU. There is nothing left in ArrowMetal to remove — encode is 2.6 µs.
- **A chain of 10 small ops in < 100 µs: reached.** `ctx.batch { }` over ten 1,000-row ops is
  **95 µs**, down from 137 µs. The raw Metal equivalent is 69 µs, so ArrowMetal now sits ~26 µs above
  the floor for a ten-op chain rather than ~68 µs above it.
- The GPU/CPU crossover for `sum` is unchanged in shape and moved slightly: one core still wins below
  about 1M rows.


## 7. Power

There is no user-space GPU power counter without `sudo powermetrics`, which is not available in this
environment, so CPU time (`getrusage`) is used as the proxy. 1,000-row `sum`, 2,000 calls:

| configuration | wall median µs | CPU µs/op | CPU per wall µs |
|---|---:|---:|---:|
| `lowLatencyWait`, `spinMicroseconds = 300` (default) | 101.0 | 128.0 | 1.27 |
| `lowLatencyWait`, `spinMicroseconds = 0` | 109.4 | **31.6** | **0.29** |
| `waitUntilCompleted`, `spinMicroseconds = 300` | 110.6 | 136.8 | 1.24 |
| `waitUntilCompleted`, `spinMicroseconds = 0` | 109.3 | 30.7 | 0.28 |

Spinning costs slightly more than one full core per in-flight op (the excess over 1.0 is Metal's own
helper threads). **`spinMicroseconds = 0` gives 4x of that CPU back for 8% more latency** — the right
setting for a throughput-oriented caller, and available as `am_spin_microseconds(0)` /
`arrowmetal.spin_microseconds(0)`. With spinning off the two wait strategies are identical, as
expected: the event is never polled.

For scale, the resident worker it replaced would have burned **one full CPU core plus one GPU
threadgroup continuously and indefinitely** — `ResidentProbe` measures 0.080 CPU-seconds over an
80 ms window, i.e. a core pinned, while the GPU is 100% occupied for the whole dispatch and
unavailable to anything else. That cost would have been justified by a 20 µs op. It is not justified
by a 5 ms one.


## 8. Limits and what is left

- The 56 µs floor is the driver's submit + notify path. Nothing in ArrowMetal can touch it.
- Everything below 20 µs per op comes from amortisation: `batch { }` (6.9 µs/op at ten dispatches) or
  `batchAsync` for pipelining (11.5 µs/op with 20 command buffers in flight). Small-op workloads
  should batch; the API already supports it in Swift, C and Python.
- `filter` is still the furthest from the floor (~82 µs at 1,000 rows against ~69 µs for compare)
  because it encodes three dispatches and allocates. Fusing its count and scan passes is the next win
  and would take it to roughly the floor.
- Not attempted, and probably not worth it given where the time goes: `MTLIndirectCommandBuffer` with
  patched arguments. It removes encoding, and encoding is 2.6 µs of a 65 µs call.
- Not attempted: Metal 4 (`MTL4CommandQueue`, residency sets). Its committed-command-buffer path may
  have a shorter notification route; that is the one remaining idea that could move the floor itself.
- If a future OS gives MSL a system-scope atomic or a documented CPU/GPU coherence guarantee inside a
  dispatch, `ResidentProbe` is the test to re-run first; `ResidentTests` prints a loud note rather
  than failing if both directions ever start working.
