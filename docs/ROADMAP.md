# Roadmap

ArrowMetal's roadmap is its measured record. What each release added is in
[CHANGELOG.md](../CHANGELOG.md), with the results file behind every number; what was measured and learned
along the way is in [FINDINGS.md](FINDINGS.md); the shapes where a CPU library is ahead, with their
numbers, are in [TO_IMPROVE.md](TO_IMPROVE.md). This page carries no dates and no plans.

## Scope

- Apple silicon only: every kernel is Metal, on the GPU of an M-series Mac.
- No dataframe API of its own: ArrowMetal runs under Polars, DuckDB, DataFusion and pandas, and over
  Arrow arrays directly.
- Exact answers where Arrow specifies exact ones; each documented divergence is listed in
  [EVALUATION.md](EVALUATION.md).
- Python, Swift, C, Rust, Go, TypeScript and R, all over one C ABI (`include/arrowmetal.h`).

Ideas and requests are welcome as GitHub issues.
