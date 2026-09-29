# datafusion-arrowmetal

A physical optimizer rule for [Apache DataFusion](https://datafusion.apache.org) 55.1 that runs full
`ORDER BY` sorts on Apple silicon GPUs through ArrowMetal, with DataFusion's answers. The default
also takes the aggregate shapes a measured table takes (`count(*)` and `DISTINCT` over two int32
keys of a `MemTable` of at least 10,000,000 rows): such an aggregate estimates its number of groups
when it runs and runs on the GPU or hands the node back to DataFusion's own operators.

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
| `src/rule.rs` | `ArrowMetalRule`, `ArrowMetalConfig` (the default take-list), `AggregateChoice`, `Report`, `Decision` |
| `src/exec.rs` | `MetalExec`: collects the input, the aggregates' run-time choice, runs the GPU plan, hands the node back to DataFusion |
| `src/probe.rs` | the group-count estimate from a sample of the keys |
| `src/choice.rs`, `src/agg_table.rs` | an aggregate's shape and the measured table (generated) |
| `scripts/groupby_table.py` | generates `src/agg_table.rs` from the result CSVs (`--check`) |
| `src/translate.rs` | the checks on node shapes and types, and predicates to ArrowMetal expressions |
| `src/gpu.rs` | the plans sent to ArrowMetal, the chunked import, the run-time checks on the data |
| `tests/` | `grid.rs` (7,656 query pairs, rule off against rule on), `rule.rs`, `arrowmetal_repros.rs` |
| `examples/quickstart.rs` | the example |
| `examples/bench.rs` | the rule off / rule on benchmark |
| `examples/coldstart.rs`, `examples/plancost.rs` | pipeline compilation per process; the rule's planning cost |
| `results/` | the benchmark CSVs the documentation's tables come from |
