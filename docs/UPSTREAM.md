# Upstream tracker

What ArrowMetal's differential matrix and its platform work found in other projects, and where each
report stands. A row is added when a finding is confirmed by a test in this repository, and updated when
the report is filed, answered or fixed. Nothing here is a complaint about another project: every entry
is a divergence the matrix has to carry, and each has a test that fails on the day upstream fixes it, so
the workaround can be removed.

| Project | Finding | Our evidence | Report | Status |
|---|---|---|---|---|
| pyarrow 25.0.1 | `pc.utf8_normalize` decomposes whatever `form` says (its NFC output is NFD) | `test_pyarrow_utf8_normalize_ignores_its_form_option` | not yet filed | open |
| pyarrow 25.0.1 | `pc.pairwise_diff` on a sliced array ignores `ArrowArray.offset` | `test_pyarrow_pairwise_diff_ignores_the_array_offset` | not yet filed | open |
| pyarrow 25.0.1 | `pc.fill_null_forward`, `pc.fill_null_backward`, `pc.replace_with_mask` on a sliced boolean array ignore the offset | `test_pyarrow_boolean_fill_null_forward_ignores_the_array_offset` | not yet filed | open |
| pyarrow 25.0.1 | `pc.winsorize` on a sliced array nulls the wrong rows and places values where the input has nulls | `test_pyarrow_winsorize_ignores_the_array_offset` | not yet filed | open |
| pyarrow 25.0.1 | `pc.binary_slice` with its own default `stop` raises `ArrowInvalid: Negative buffer resize` | `test_pyarrow_binary_slice_overflows_on_its_own_default_stop` | not yet filed | open |
| pyarrow 25.0.1 | `pc.year_month_day` and `pc.iso_calendar` corrupt the heap and segfault a few allocations later | the matrix never calls them; `temporal_struct` compares field by field ([EVALUATION.md](EVALUATION.md)) | not yet filed | open |
| Arrow (bundled tz database) | DST rules stop being applied after January 2038, so a summer instant in 2050 gets the winter offset | finding 14 in [EVALUATION.md](EVALUATION.md), 144 matrix cases | not yet filed | open |
| Apple Metal | no 64-bit atomic operations in the Metal Shading Language on Apple GPUs; grouped moments and hash tables carry 32-bit workarounds | [DECISIONS.md](DECISIONS.md), [DESIGN.md](DESIGN.md) | Feedback Assistant, not yet filed | open |
| Apple Metal | `makeComputePipelineState` fails sporadically on the "Apple Paravirtual device" GitHub-hosted runners expose | retry in `MetalContext.pipeline` | Feedback Assistant, not yet filed | open |

Nothing found in pandas, numpy, Polars or DuckDB is their defect: the costs the matrix measures there are
crossings and idioms, not bugs.

## How a row moves

1. **open**: confirmed here, test in place, nothing filed.
2. **filed**: the report link goes in the Report column with the date.
3. **fixed in X**: upstream released the fix; the test that pins the workaround starts failing on that
   version and is retired with it, and the row stays as history.
