# datafusion-arrowmetal

A physical optimizer rule for [Apache DataFusion](https://datafusion.apache.org) 55.1 that runs full
`ORDER BY` sorts on Apple silicon GPUs through ArrowMetal, with DataFusion's answers.

User documentation — registering the rule, what the default takes and leaves, the semantics matched,
the differential grid, the measured numbers and the limits — is in
[`docs/DATAFUSION.md`](../docs/DATAFUSION.md).

```sh
# The GPU library, from the repository root
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --product ArrowMetalC

cd datafusion
export ARROWMETAL_LIB=$PWD/../.build/release/libArrowMetalC.dylib
cargo run --example quickstart          # the example in docs/DATAFUSION.md
cargo test                              # the differential grid, the rule tests, the plan-runner tests
cargo test --test rule -- --ignored     # the default configuration at 2,000,000 rows
```

macOS on Apple silicon only: `build.rs` stops the build for any other target. DataFusion is pinned
to `=55.1.0`.

| Path | What it is |
|---|---|
| `src/lib.rs` | the registration helpers: `session_context`, `with_arrowmetal`, `physical_optimizer_rules` |
| `src/rule.rs` | `ArrowMetalRule`, `ArrowMetalConfig` (the default take-list), `Report`, `Decision` |
| `src/exec.rs` | `MetalExec`: collects the input, runs the GPU plan, hands the node back to DataFusion on an error |
| `src/translate.rs` | the checks on node shapes and types, and predicates to ArrowMetal expressions |
| `src/gpu.rs` | the plans sent to ArrowMetal, the chunked import, the run-time checks on the data |
| `tests/` | `grid.rs` (4,656 query pairs, rule off against rule on), `rule.rs`, `arrowmetal_repros.rs` |
| `examples/quickstart.rs` | the example |
| `examples/bench.rs` | the rule off / rule on benchmark |
| `results/` | the benchmark CSVs the documentation's tables come from |
