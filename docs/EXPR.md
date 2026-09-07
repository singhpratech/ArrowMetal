# Fused expression queries

One Metal kernel for a whole Arrow compute expression, instead of one kernel per operator.

`(a * 2 + b) / (c + 1) - d` is five Arrow kernels: five reads of a column, five writes of a temporary,
five command buffers (or five dispatches inside one). At 50M rows on an M4 Max that costs 24 ms even
when every kernel runs at memory bandwidth, because the data is moved five times.

The expression compiler lowers the whole tree — arithmetic, comparisons, null logic, casts, string
predicates — into **one runtime-generated MSL kernel**, so the inputs are read once and the outputs are
written once. Same expression, 16 ms, and the CPU does 0.7 ms of work instead of 2.8 ms.

```swift
import ArrowMetal

let batch = try MetalRecordBatch(names: ["region", "amount"], columns: [.int32(region), .float32(amount)])
let total = try batch.query(
    query().filter((col("region") == 2) && (col("amount") > 100)).sum(col("amount"))
).onlyScalar                                   // .double(...)
```

```python
import arrowmetal as am
am.query(table, am.filter((am.col("region") == 2) & (am.col("amount") > 100)).sum(am.col("amount")))
am.query(table, ((am.col("a") * 2 + am.col("b")) / (am.col("c") + 1) - am.col("d")).alias("r").project())
am.query(table, am.group_by(am.col("k"), 1000).sum(am.col("v") * 3 + 1, "s"))
```

---

## What it is

| File | What lives there |
|---|---|
| `Sources/ArrowMetal/Expr/Expr.swift` | The `Expr` tree, the operators, the Swift DSL, the canonical text |
| `Sources/ArrowMetal/Expr/ExprPlan.swift` | `ExprQuery` (filter + group-by + terminal), the builder, `MetalRecordBatch.query(_:)` |
| `Sources/ArrowMetal/Expr/ExprSource.swift` | The lowering: type checking, promotion, CSE, MSL emission, the shared MSL helpers |
| `Sources/ArrowMetal/Expr/ExprCompiler.swift` | Kernel assembly, the compiled-kernel cache, buffer binding |
| `Sources/ArrowMetal/Expr/ExprKernels.swift` | project / filter+project / aggregate terminals |
| `Sources/ArrowMetal/Expr/ExprGroupBy.swift` | The group-by terminal (privatised and device-wide atomic tables) |
| `Sources/ArrowMetal/Expr/ExprText.swift` | The serialised grammar, parsed and printed |
| `Sources/ArrowMetalC/ArrowMetalC_Expr.swift` | `am_query` and the result handle |
| the block at the end of `python/arrowmetal/__init__.py` | `am.col`, `am.filter`, `am.group_by`, `am.query` |

## How the lowering works

The emitter walks the tree once, bottom up, and appends straight-line MSL to one
`inline void am_row(...)` function. Every node becomes two registers: the value and a `bool` validity.

- **Common subexpression elimination** is keyed on the node's canonical text, so a subtree that occurs
  four times is evaluated once. `ExprTests.testDeepMixedTreeWithSharedSubtrees` builds a 20+ node tree
  with a shared subtree used four times and checks it against a Swift oracle.
- **Validity is compiled in, not carried in buffers.** A node that cannot be null carries the literal
  `true` instead of a register, and the Metal front end folds the whole null chain away. A node that
  can be null gets one `bool`; the *input* bitmaps are read one 32-bit word per 32 rows into a register
  (`am_vword`) and the output bitmap is built in a register and stored once per 32 rows.
- **Vectorised loads.** The kernels are word shaped: one thread owns 32 consecutive rows (the same
  shape the existing compare and filter kernels use). When every leaf is a plain numeric column the
  thread loads its 32 rows as eight four-wide vector loads per column; a boolean or `utf8` leaf, whose
  access is not a contiguous vector, falls back to the scalar path.
- **Float64** has no hardware support on Apple GPUs, so it travels as a raw 64-bit pattern in a `ulong`
  and goes through the correctly rounded software binary64 of `Kernels/DoubleMath.swift`
  (`d_add`/`d_sub`/`d_mul`/`d_div`) and the transcendentals of `Kernels/DoubleTranscendental.swift`.
  `d_add`/`d_sub`/`d_mul`/`d_div` and `d_sqrt` are correctly rounded, so those results are **bit
  identical** to Swift's `Double` — `ExprTests.testFloat64ArithmeticIsBitExact` asserts add, sub, mul,
  div, negate, abs and round on bit patterns, and `test_expr.py` asserts against pyarrow. The
  transcendentals are bounded to 1-2 ulp, not correctly rounded.
- **Pipeline caching.** A compiled query is cached on the canonical query text plus the referenced
  columns' types and nullability, and the pipeline itself on a hash of the generated source (the same
  process-wide cache every other kernel uses). Running the same shape again costs one dictionary
  lookup: `ExprTests.testPipelineCacheHit` asserts the compile counter does not move.
- **Batching.** A query joins an open `MetalContext.batch { }` like any other kernel, and a filtered
  project inside a batch produces *pending* arrays whose length the GPU decides, so a second query can
  consume them without a CPU round trip (`ExprTests.testChainedQueriesInOneBatch`).

## Kernel shapes

| Terminal | Dispatches | Passes over the input |
|---|---|---|
| `project`, no filter | 1 | 1 read, 1 write per output |
| `project` + `filter` | 3 (count, scan, scatter) | 2 reads, 1 write — the standard compaction pipeline |
| `aggregate` (± filter) | 1 | 1 read; threadgroup partials, finished on the CPU |
| `group_by` (± filter) | 2 (accumulate, merge) | 1 read; privatised 32-bit atomic tables |

The filtered project reads its inputs twice on purpose: the output positions are only known after the
scan, and re-evaluating the expression for the selected rows is cheaper than materialising every
intermediate. The predicate is compiled into the counting pass, so no boolean array is ever built.

Group-by privatises a table of `keyCount × aggregates` slots in threadgroup memory while that fits in
2048 slots (24 KB of the 32 KB budget); above that it uses device-wide atomics, as `Kernels/GroupBy.swift`
does. Sums are 64-bit through a lo/hi carry pair, because MSL has no 64-bit atomic add.

## The grammar

Every query serialises to an s-expression. It is what `Expr.description` prints, what
`ExprQuery(text:)` parses, what `am_query` takes over the C ABI, and what Python's `.sexpr()` produces.
`include/arrowmetal.h` carries the same grammar for C consumers.

```
query   := "(query" filter? group_by? terminal ")"
filter  := "(filter" expr ")"
group_by:= "(group_by" INT "\"name\"" expr ")"
terminal:= "(project" ("(as \"name\"" expr ")")+ ")"
         | "(aggregate" agg+ ")"
agg     := "(" ("sum"|"min"|"max"|"mean"|"count") "\"name\"" expr? ")"

expr    := "(col \"name\")"
         | "(int" INT ")" | "(float" NUM ")"              -- untyped: adapts to the other operand
         | "(i8"|"i16"|"i32"|"i64"|"u8"|"u16"|"u32"|"u64" INT ")"
         | "(f32"|"f64" NUM ")" | "(bool" true|false ")" | "(str \"...\")" | "(null" TYPE ")"
         | "(" BINOP expr expr ")" | "(" UNOP expr ")"
         | "(cast" expr TYPE ")"
         | "(if_else" expr expr expr ")" | "(coalesce" expr+ ")" | "(fill_null" expr expr ")"
         | "(is_null" expr ")" | "(is_valid" expr ")" | "(is_in" expr literal+ ")"
         | "(str_eq"|"starts_with"|"contains" expr "\"pattern\"" ")"
BINOP   := add sub mul div | eq ne lt le gt ge | and or and_kleene or_kleene
         | bit_and bit_or bit_xor shl shr
UNOP    := negate abs sqrt exp ln round not bit_not
TYPE    := i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 bool str
```

Example:

```
(query (filter (and (eq (col "region") (int 2)) (gt (col "amount") (int 100))))
       (aggregate (sum "total" (col "amount"))))
```

## Types and promotion

Checked against `pyarrow.compute` (`test_expr.py`), and applied to both operands of every binary
operator:

1. `float64` with anything numeric → `float64`.
2. `float32` with any integer → `float32` (Arrow's rule; `int64 + float32` is `float32`, not `float64`).
3. Two integers of the same signedness → the wider one.
4. Mixed signedness → a **signed** type of `max(signed width, unsigned width × 2)`, capped at `int64`
   (so `uint8 + int8 → int16`, `uint32 + int32 → int64`, `uint64 + int64 → int64`).
5. `boolean` and `utf8` have no numeric common type: mixing them is an error naming both types.

An **untyped literal** (`col("x") > 100`, `(int 100)`) takes the type of the other operand when that is
numeric, so comparing a `float32` column with `100` stays in `float32`. Pin a literal with
`Expr.typedInt(100, .int32)` / `am.lit(100, "int32")` when you want a specific width.

It takes the other operand's type only when it **fits** there; a literal that does not fit widens the
pair instead of being truncated to it, because truncating would silently answer a different question.
An integer literal outside the column's range promotes as if it carried its own smallest type
(`int8_column > 200` compares in `int16`, not against `(char)200 == -56`; `uint8_column == -1` compares
in `int16` and is false everywhere), and a **floating** literal against an integer column promotes the
pair to `float64` under rule 1 (`int64_column >= 2.5` really does compare against 2.5). The same rule
applies inside `if_else`, `fill_null`, `coalesce` and `is_in`.

`sqrt`, `exp` and `ln` on an integer column promote to `float64`, as Arrow's do. `round` puts halves
**away from zero** (Arrow's `half_towards_infinity`), matching ArrowMetal's existing `round`.

## How nulls are compiled

| Node | Result validity |
|---|---|
| arithmetic, comparison, bit ops, casts, `abs`/`negate`/`sqrt`/`exp`/`ln`/`round` | valid iff every input is valid |
| `and` / `or` | valid iff both inputs are valid |
| `and_kleene` | `false` as soon as either side is a valid `false`; otherwise null if either side is null |
| `or_kleene` | `true` as soon as either side is a valid `true`; otherwise null if either side is null |
| `not` | validity of the operand |
| `if_else(c, a, b)` | null when `c` is null, otherwise the validity of the chosen branch |
| `coalesce`, `fill_null` | valid when any argument is valid |
| `is_null`, `is_valid`, `is_in` | never null (a null input `is_in` anything is `false`, as pyarrow does) |
| string predicates | validity of the string column |
| `filter(pred)` | a row is kept when `pred` is **true and not null** (Arrow's `null_selection_behavior="drop"`) |
| `sum`/`min`/`max`/`mean` | skip nulls; null when nothing contributed. `min`/`max` also skip NaN |
| `count(expr)` | non-null values of `expr`; `count()` counts rows that pass the filter |
| group-by | rows whose key is null or outside `[0, keyCount)` are skipped; a group with no value is null |

Integer division by zero yields **0**, which is what ArrowMetal's other integer kernels do (pyarrow
raises). A shift amount outside `[0, width)` yields 0.

## Numbers

M4 Max, 50M rows, best of 5, in-process against Polars 1.44 (16 threads), pyarrow 25, pandas 3, numpy 2.5.
`Benchmarks/expr_bench.py`. "op-by-op" is the same expression built from ArrowMetal's per-operator
kernels, all batched into one command buffer — today's best without fusion.

| case | implementation | wall ms | GB/s | CPU ms |
|---|---|---:|---:|---:|
| (a) `sum(amount) where region == 2 and amount > 100` | **ArrowMetal fused (1 kernel)** | **1.69** | **237** | 0.4 |
| | ArrowMetal op-by-op (batched, 6 kernels) | 2.06 | 195 | 0.5 |
| | polars lazy | 16.72 | 24 | 29.6 |
| | pyarrow.compute | 63.73 | 6.3 | 63.7 |
| | pandas (arrow-backed) | 42.01 | 9.5 | 42.0 |
| | numpy masked sum | 119.22 | 3.4 | 119.2 |
| (b) `(a*2 + b) / (c + 1) - d`, float64, 5% nulls | **ArrowMetal fused (1 kernel)** | **15.94** | **126** | 0.7 |
| | ArrowMetal op-by-op (batched, 5 kernels) | 23.82 | 84 | 2.8 |
| | polars lazy | 123.55 | 16 | 123.5 |
| | pyarrow.compute | 126.97 | 16 | 127.0 |
| | numpy (no nulls) | 41.26 | 49 | 41.3 |
| (c) `filter(sel < 5)` then project 3 columns | **ArrowMetal fused** | **10.27** | **146** | 1.3 |
| | ArrowMetal op-by-op (batched) | 10.80 | 139 | 1.2 |
| | polars lazy | 42.11 | 36 | 77.6 |
| | pyarrow.compute | 209.89 | 7.1 | 209.8 |
| (d) group-by `sum(v * 3 + 1)` by 1000 keys | **ArrowMetal fused (1 kernel + merge)** | **1.99** | **201** | 0.5 |
| | ArrowMetal op-by-op (batched) | 3.66 | 109 | 0.5 |
| | polars lazy | 124.35 | 3.2 | 1203.2 |
| | pyarrow group_by | 58.41 | 6.8 | 260.8 |

At **100 000 rows** the fixed cost dominates and the GPU is the wrong tool:

| case | implementation | wall ms |
|---|---|---:|
| (e) `sum(amount) where region == 2 and amount > 100` | ArrowMetal fused | 0.25 |
| | ArrowMetal op-by-op (batched) | 0.29 |
| | polars lazy | 0.12 |
| | pyarrow.compute | 0.14 |
| (e) `(a*2 + b) / (c + 1) - d`, float64 | ArrowMetal fused | 0.86 |
| | ArrowMetal op-by-op (batched) | 0.62 |
| | polars lazy | 0.12 |

Reading the small-input row: the command-buffer round trip is about 60-70 µs measured
([RESIDENT.md](RESIDENT.md)), 110-230 µs as an all-in per-call floor in the matrix's latency family,
and it is paid once per query however many operators the expression has — that is what fusion buys at
the low end. What it cannot buy back is the round trip itself, nor the cost of allocating and
first-touching a fresh output buffer
(the reason the float64 *project* costs more than the *aggregate*, which only writes threadgroup
partials). Crossover against Polars for these shapes is around 1M rows.

Where fusion wins most is where the operator count is highest relative to the bytes moved: (b) is 1.5×
the batched op-by-op chain, (d) is 1.8×, and (a) is 1.2× — while (c), which is compaction bound rather
than operator bound, is a wash. Against the CPU engines in these tables — Polars, pyarrow and pandas —
the margin is 4× to 62×, at a sixtieth to a two-thousandth of the CPU time; against numpy it is 2.6×
to 70×.

## Limits

- **No string outputs.** String columns can be read by `str_eq` / `starts_with` / `contains` against a
  literal pattern; the compiler will not materialise a `utf8` output column, and rejects a projection
  that asks for one by name.
- **Only these column types are read**: int8–int64, uint8–uint64, float32, float64, boolean, utf8.
  Temporal, decimal, dictionary, list, struct, map and union columns are rejected with an error naming
  the column and its Arrow format. Extension columns are read through their storage.
- **Group-by needs a dense integer key** in `[0, keyCount)`, exactly like `GroupBy`. Hash the keys with
  `GroupByKeys` first if they are not dense.
- **Group-by aggregate types** are limited to the aggregates 32-bit atomics support (MSL has no
  64-bit atomic add): `sum`/`mean` accept any
  integer or `float32` (not `float64` — cast first); `min`/`max` accept a 32-bit or narrower integer or
  `float32`. Both errors name the aggregate and say what to cast.
- **`is_in` takes a small literal set** and expands to a chain of comparisons; it is not a hash lookup,
  so keep the set to a few dozen values. A null input is `false`.
- **No window functions, no joins, no sorts, no aggregates over strings** inside a query.
- **A `count(expr)` of a boolean expression** is allowed; `sum` of a boolean is not (cast it).
- **`pyarrow.compute.Expression` is not accepted.** Converting one means walking `str(expr)`, whose
  rendering is ambiguous for string literals and field names with spaces, or going through Substrait,
  which is a larger dependency than this feature warrants. Build the expression with `am.col` instead —
  the two read almost identically.
- Arrays above 2^32 elements are not supported, as elsewhere in the package.

## Correctness

- `Tests/ArrowMetalTests/ExprTests.swift` — 20 test cases against Swift oracles at 0 / 1 / 33 / 4097 /
  1 000 003 rows with nulls: every operator alone, deep mixed trees with shared subtrees, bit-exact
  float64 arithmetic and comparisons (including ±0, subnormals, infinities and NaN), Kleene predicates,
  all four terminals, batched vs unbatched, the pipeline-cache hit, the error cases, and the canonical
  text round trip.
- `python/tests/test_expr.py` — 40 cases against `pyarrow.compute` compositions and Polars lazy for the
  same expressions, including a bit-identical check of the six-operator float64 expression.

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter ExprTests
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test -c release --filter ExprTests
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests/test_expr.py -q
PYTHONPATH=python python Benchmarks/expr_bench.py
```
