# Upstream tracker

What ArrowMetal's differential matrix and its platform work found in other projects, and where each
report stands. A row is added when a finding is confirmed by a test in this repository, and updated when
the report is filed, answered or fixed. Nothing here is a complaint about another project: every entry
is a divergence the matrix has to carry, and each has a test that fails on the day upstream fixes it, so
the workaround can be removed.

| Project | Finding | Our evidence | Report | Status |
|---|---|---|---|---|
| pyarrow 25.0.1 | `pc.utf8_normalize` decomposes whatever `form` says (its NFC output is NFD) | `test_pyarrow_utf8_normalize_ignores_its_form_option` | not yet filed; no existing report found | open, reproduced 2026-09-07, draft ready |
| pyarrow 25.0.1 | `pc.pairwise_diff` on a sliced array ignores `ArrowArray.offset` | `test_pyarrow_pairwise_diff_ignores_the_array_offset` | reported by others: [apache/arrow#50524](https://github.com/apache/arrow/issues/50524), fixed by [#50858](https://github.com/apache/arrow/pull/50858) | fixed in 26.0.0; the test retires when the venv moves to 26 |
| pyarrow 25.0.1 | `pc.fill_null_forward`, `pc.fill_null_backward`, `pc.replace_with_mask` on a sliced boolean array ignore the offset | `test_pyarrow_boolean_fill_null_forward_ignores_the_array_offset` | not yet filed; distinct from [#45086](https://github.com/apache/arrow/issues/45086) (chunked-output sizing, fixed in 26.0.0) | open, reproduced 2026-09-07, draft ready |
| pyarrow 25.0.1 | `pc.winsorize` on a sliced array nulls the wrong rows and places values where the input has nulls | `test_pyarrow_winsorize_ignores_the_array_offset` | not yet filed; no existing report found | open, reproduced 2026-09-07, draft ready |
| pyarrow 25.0.1 | `pc.binary_slice` with its own default `stop` raises `ArrowInvalid: Negative buffer resize` | `test_pyarrow_binary_slice_overflows_on_its_own_default_stop` | not yet filed; the same overflow was fixed for `utf8_slice_codeunits` in [#36575](https://github.com/apache/arrow/pull/36575), the binary path was not; umbrella [#34929](https://github.com/apache/arrow/issues/34929) | open, reproduced 2026-09-07, draft ready |
| pyarrow 25.0.1 | `pc.year_month_day` and `pc.iso_calendar` crashed the process in the first matrix run | the matrix never calls them; `temporal_struct` compares field by field ([EVALUATION.md](EVALUATION.md)) | withheld: not reproduced on 2026-09-07 under guard malloc | unexplained, observed once |
| Arrow (vendored date library) | DST rules stop being applied after the last tabulated transition (2037), so a summer instant in 2050 gets the winter offset, on a fat TZif file | finding 14 in [EVALUATION.md](EVALUATION.md), 144 matrix cases | not yet filed; related [#42157](https://github.com/apache/arrow/issues/42157) (open) covers slim files only | open, reproduced 2026-09-07, draft ready |
| Apple Metal | the Feature Set Tables say the Apple9 family has "the full set of 64-bit atomic operations", but the shading-language headers admit `device ulong` only for atomic min and max: no 64-bit add, compare-exchange, load or store compile, whatever language version is asked for (measured on an M4 Max, macOS 26.6.2). Grouped sums and hash tables carry 32-bit workarounds because of it | [DECISIONS.md](DECISIONS.md), [DESIGN.md](DESIGN.md); the probe is attached to the draft report | Feedback Assistant, not yet filed, draft ready | open |
| Apple Metal | `makeComputePipelineState` fails sporadically on the "Apple Paravirtual device" GitHub-hosted runners expose | retry in `MetalContext.pipeline`; CI logs: 97 pipeline-creation failures across ten kernels in one run and 0 in another on the same runner image, `makeLibrary` succeeding every time | Feedback Assistant, not yet filed, draft ready with the log excerpts | open |
| Apple Metal | a persistent GPU worker polling shared memory cannot be made to work: CPU/GPU coherence and the driver's submit-and-notify floor | [RESIDENT.md](RESIDENT.md), two measured probes | Feedback Assistant, not yet filed, draft ready | open |
| Apache Arrow JS | no C Data Interface export or import in the JavaScript library, so the Node binding builds the `ArrowArray`/`ArrowSchema` structs itself from a `Vector`'s buffers; and `Data.slice` advances the values and offsets buffers but not the validity bitmap, leaving the row offset in `Data.offset`, which a C Data consumer must undo | `node/src/addon.cc` (the rewind on import, the advance on export); [TYPESCRIPT.md](TYPESCRIPT.md) | github.com/apache/arrow-js, not yet filed; whether a C Data Interface is already tracked there is being checked | open, candidate |
| Swift 6.3.3 | a `withUnsafeBytes` closure inside a throwing generic convenience init miscompiled under `-O` (crash on entry, release builds only) | the plain-loop workaround in `MetalArray.init(_:)`, [FINDINGS.md](FINDINGS.md) | github.com/swiftlang/swift, not yet filed; a minimal reproducer is being built | open, reproducer needed |

Nothing found in pandas, numpy, Polars or DuckDB is their defect: the costs the matrix measures there are
crossings and idioms, not bugs.

## Where each report goes

| Project | Route | Public tracker |
|---|---|---|
| Apache Arrow (pyarrow, the vendored date library) | GitHub issues, and a pull request when the fix is ours | github.com/apache/arrow |
| Swift compiler | GitHub issues, pull requests welcome | github.com/swiftlang/swift |
| Apple Metal, the shading language, the driver | Feedback Assistant (private; the FB number is what goes in the Report column), the Apple Developer Forums for a public thread, a Developer Technical Support incident or a WWDC lab for a conversation | none: Metal is not open source and has no public issue tracker |
| Apple's open-source projects that touch this work (MLX, Swift packages) | GitHub, like any other project | github.com/ml-explore/mlx, github.com/apple |

## How a row moves

1. **open**: confirmed here, test in place, nothing filed.
2. **filed**: the report link goes in the Report column with the date.
3. **fixed in X**: upstream released the fix; the test that pins the workaround starts failing on that
   version and is retired with it, and the row stays as history.
