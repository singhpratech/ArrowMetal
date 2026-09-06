# Testing

Every GPU kernel has a plain-Swift CPU reference (`Sources/ArrowMetal/CPUReference.swift` and per-test
oracles) and is compared against it across types, sizes, null densities and input shapes. Tests skip on
virtual Metal devices (GitHub runners) via `requireRealGPU()`; run them on a real Mac.

Run everything:
```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift test                # debug
swift test -c release     # optimised: required, a release-only miscompile has bitten this project once
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests -q
```

## Swift suites (Tests/ArrowMetalTests)

| Suite | Covers |
|---|---|
| KernelTests | Reductions on every type with nulls, compare scalar/array, boolean logic, arithmetic incl. Float64, filter |
| TakeCastSliceTests | take with null indices and bounds errors, cast between all types, zero-copy vs copying slices |
| Float64AndBooleanTests | Float64 compare/min/max/filter via bit-pattern kernels, NaN and signed zero, boolean filter/count/any/all |
| DoubleMathTests | Software IEEE-754 add/sub/mul/div bit-exact against Swift over 1M random and edge-case pairs; Float64 sum |
| ScenarioMatrixTests | Every kernel × every type × null ratio {0, 0.3, 1.0} × sizes 0..70001 × plain/sliced against the oracle; concurrent use from 64 threads; buffer-pool zeroing semantics |
| BatchTests | Batched chains equal unbatched, deferred lengths and null counts, deferred take errors, pool parking, export inside a batch |
| AsyncTests | Calling thread stays free during async batches (spin-rate proof), results equal sync path, error propagation |
| GroupByTests | Dense-key group-by sum/count/mean/min/max across key counts 1..100000, 64-bit carry, float sums, sliced inputs |
| UniqueTests | unique, value_counts, dictionary_encode, group-by over arbitrary keys; NaN/-0.0 normalisation |
| SortTests | Radix argsort stable both directions, nulls last, IEEE total order, top-k, batch sort |
| JoinTests | Inner/left hash join vs a dictionary oracle: many-to-many, null keys, empty sides, batch joins |
| StringTests | Lengths, predicates, MurmurHash3 reference vectors, GPU filter/take, dictionary encode, scans |
| CInteropTests | C Data / C Device Interface export and import, zero-copy detection, foreign buffers with offsets, release callbacks |
| RecordBatchTests | Multi-column filter/take/slice/select, struct (+s) round trips, ArrowArrayStream import |
| Plus suites added by the current wave | Temporal, IPC, Structural, MathKernel, StringTransform, Segmented/TopK |

## Python suite (python/tests)

- `test_arrowmetal.py`: 83 tests over the ctypes API against pyarrow.compute and plain Python, including Polars interop and the wheel loader.
- `test_differential.py` and `differential_report.py`: randomised differential testing of every exposed operation against pyarrow.compute (see EVALUATION.md).

## What "passing" means before a push

1. `swift test -c release` green on real Apple silicon.
2. `pytest python/tests` green.
3. `differential_report.py` exits 0, or every failure is recorded as an open finding in EVALUATION.md with inputs and expected/actual values.
