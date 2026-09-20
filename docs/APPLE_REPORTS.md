# What was reported to Apple

Metal has no public issue tracker, so the three findings about it went to Apple through Feedback
Assistant on 2026-09-20. Feedback Assistant is private: the reports, their attachments and any reply
are visible only to Apple and to the account that filed them. This page is the readable record of
what each report says, what was measured, and what it asks for, so that nobody has to take the
tracker's one-line summaries on trust. The measurements behind each one are in the repository; the
probes can be run by anyone with an Apple silicon Mac.

| Report | Type | Reference | Status |
|---|---|---|---|
| 64-bit atomics: the documentation and the compiler disagree | incorrect or unexpected behaviour | FB24858110 | open, no similar reports at filing |
| Pipeline creation fails sporadically on GitHub's virtual GPU | incorrect or unexpected behaviour | FB24858160 | open |
| A running kernel cannot reliably see CPU stores | suggestion | FB24858235 | open |

Every number below is from an Apple M4 Max on macOS 26.6.2 (25G83) with Xcode 26.6, unless the text
says otherwise. The rows move in [UPSTREAM.md](UPSTREAM.md) when Apple answers.

## 1. 64-bit atomics: the Feature Set Tables promise more than the shading language accepts (FB24858110)

**What the documents say.** Apple's Metal Feature Set Tables list 64-bit atomics for the Apple9 GPU
family, which the M4 series belongs to, and state in a footnote that "the full set of 64-bit atomic
operations is supported on all platforms starting with Apple9".

**What the compiler accepts.** Exactly two 64-bit atomic operations, on device memory only:
`atomic_min_explicit` and `atomic_max_explicit` on `device atomic_ulong`, void-returning, relaxed
order. Every other 64-bit atomic is a compile error at every language version from 3.1 to 4.0:
fetch-add, fetch-sub, the value-returning min and max, exchange, compare-exchange, load and store,
and there is no signed `atomic_long`. The shipped header agrees with the compiler, and so does the
shading-language specification's own table of 64-bit atomic functions, which lists min and max and
nothing else. So either the hardware has only min and max and the Feature Set Tables overstate it,
or the hardware has more and the language does not expose it. Neither document says, per GPU family,
which operations exist. A 64-bit add cannot be built from parts either, because there is no 64-bit
compare-exchange to loop on.

**How it was measured.** A short Swift program with no dependencies compiles one tiny kernel per
operation at three language versions and prints the compiler's verdict for each, then runs the two
operations that do compile over ten million elements and checks their results. It was run on the
day of filing with the same outcome as two weeks earlier. The program and its output were attached.

**What it costs this project.** Every 64-bit accumulation in the engine carries a workaround: an
int64 sum per group is a split 32-bit add with an explicit carry; a hash table publishes a 32-bit row
index because a 64-bit key cannot be published atomically; and grouped variance and standard
deviation, which accumulate in software binary64, cannot use per-group atomic accumulators at all
and run a counting sort by group first. On the matrix's 50-million-row, 1,000-group rows the grouped
sum, where the carry trick suffices, is 3.8x ahead of pyarrow's threaded engine, while grouped
variance, which cannot use the trick, is level with it. The design notes are in
[DESIGN.md](DESIGN.md) and [DECISIONS.md](DECISIONS.md).

**What was asked.** On Apple9, the operations the footnote already claims. Failing that, a precise
per-family statement of which `ulong` operations exist, in place of "the full set".

## 2. Pipeline creation fails sporadically on the virtual GPU of GitHub's macOS runners (FB24858160)

**What happens.** On GitHub-hosted macOS runners the Metal device is an "Apple Paravirtual device".
Compiling a shader library from source succeeds every time. Creating a compute pipeline for a
function from that library then fails at random, with the error text "Compilation failed" and no
other detail, for ordinary kernels that succeed on the same runner image in other runs and in the
same process moments later.

**How it was measured.** Two complete CI logs were attached, from runner image
macos-15-arm64 version 20260829.0321.1 with Xcode 16.4. In the first run, one test process saw 110
pipeline-creation failures across 23 distinct kernels and no library-compilation failure; the most
frequent was a filter kernel, 64 times. In the second run the engine retried each creation once
after 20 ms and skipped GPU tests on virtual devices; the test step then created few pipelines and
passed, and the benchmark step, which still created pipelines, failed on a group-by kernel even
with the retry. On real Apple silicon the same source has never produced one such failure across
every local run of the full suite.

**What it costs.** GPU code cannot be validated on GitHub-hosted macOS runners. This project's CI
proves only the build, the language interop and the CPU paths there; every GPU test and benchmark
runs on local hardware, and the test harness skips GPU tests when the device name contains
"Paravirtual". Any project that uses Metal compute in hosted CI meets the same failure, with an error
that gives it nothing to act on.

**What was asked.** Deterministic pipeline creation for a function from a library the same device
just compiled; or, if the virtual device cannot support some construct, an error that names it and a
device property that reports the limitation before any work is submitted.

## 3. A running kernel cannot reliably observe CPU stores to shared memory (FB24858235)

**Why it matters.** Small operations on the GPU are dominated by the fixed cost of a dispatch, about
60 microseconds of submission and completion notification on this machine. The standard escape on
other platforms is a persistent worker: one long-running kernel whose threads spin on a ring of work
descriptors in shared memory, so that submitting work is a CPU store and collecting a result is a
CPU load. That needs a CPU store to become visible to a running kernel within a bounded time, and
the reverse.

**How it was measured.** A standalone Swift program compiles one kernel per memory qualifier the
shading language offers, from plain pointers through `volatile`, `coherent(device)`, device-scope
atomics and device-memory barriers. One threadgroup spins on a doorbell word in a shared-storage
buffer for three seconds while the CPU increments that word every 5 ms, and the kernel writes a
heartbeat the CPU samples. The program was run five times across three days.

**What it found.** Visibility is sporadic and has no bound. In one run no store was seen under any
qualifier. In another, 24 of 433 stores were seen under one qualifier and a handful under others,
the first after 160 ms or more. In another, a single store of 598. The heartbeat in the other
direction arrived anywhere from 129 ms to 2.3 s, or never within the window. The behaviour is
consistent with visibility arriving only when a cache line happens to be evicted, which nothing in
the language can request. `coherent(device)` is the widest scope the shading language has, there is
no system scope, and there is no synchronisation call that can run in the middle of a dispatch.
The full study, including the measurement of where the 60 microseconds go and the finding that the
GPU watchdog is not the obstacle, is [RESIDENT.md](RESIDENT.md).

**What was asked.** Either a system-scope coherence primitive in the shading language, with a stated
visibility bound, so that a running kernel can observe CPU stores on a shared-storage buffer; or a
documented, explicitly costed way to do the same without a command-buffer boundary; or, failing
both, a documented submit-and-notify path under 20 microseconds, which removes the motive.

## What is not in these reports

No attachment on this page. The reports carried the probe programs, their outputs, the CI logs and
the kernel sources that hold the workarounds; all of those are either in this repository already or
reproducible from it. The reports contain no personal data beyond the filing account Apple requires.
