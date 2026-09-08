# Findings and gotchas

Things learned the hard way. Add to this whenever something surprises you.

## Round 10 (2026-09-08): arrow-go's span iterator, found while designing its aggregates

**TL;DR**

- Arrow Go's `compute` package has no aggregate functions ([apache/arrow-go#1296](https://github.com/apache/arrow-go/issues/1296)). The maintainer asked for them; a design note with a tested prototype of `sum`, `mean`, `min_max`, `min`, `max`, `count`, `any` and `all` is [on the issue](https://github.com/apache/arrow-go/issues/1296#issuecomment-5593008436).
- Reviewing that prototype the way this project reviews its own kernels found a bug in arrow-go itself: `ArraySpan.SetSlice` carries a stale null count into the next slice, so `and_kleene` and `or_kleene` return `false` where the answer is null once `ExecCtx.ChunkSize` is below the input length, on main and in v18.7.0. Reported as [#1305](https://github.com/apache/arrow-go/issues/1305), fixed by [#1306](https://github.com/apache/arrow-go/pull/1306).
- Two more things the C++ reference does that a straightforward Go port gets wrong: C's `fmin`/`fmax` rank -0.0 below +0.0, and float sums are pairwise, not a left fold.
- Two documentation pull requests from the same day, [#1302](https://github.com/apache/arrow-go/pull/1302) and [#1303](https://github.com/apache/arrow-go/pull/1303), were merged within hours.

### The setting

`compute.GetFunctionRegistry()` in arrow-go v18.7.0 holds 85 functions, none of aggregate kind
(89 on main: 76 scalar, 8 vector, 5 meta). `FuncScalarAgg` exists as a kind; `execInternal`
returns `ErrNotImplemented` for it. The Go binding here therefore uses plain loops as its reference
for reductions ([GO.md](GO.md)). The maintainer's reply to the report was that the kernels had been
on his list for a long time and he would love to see them; the design note was promised before
any code, so the interface would be one he wants to maintain.

### Research steps

1. Read the C++ `ScalarAggregateKernel` (`kernel.h`), the executor (`exec.cc`), the options
   (`api_aggregate.h`) and the kernels (`aggregate_basic.cc`, `aggregate_basic.inc.cc`,
   `aggregate_internal.h`) at f251bc3, and the Go `compute` package at 67ef40b, and write the design.
2. Three independent reviews of the note: one testing every row of the semantics table against
   pyarrow built from that C++ commit, one checking every Go identifier and compiling the
   proposed constraint change in a worktree, one reading it as the maintainer would.
3. Build the prototype from the corrected note, run arrow-go's own lint under the Go version it
   pins, and cross-check every expected test value against pyarrow.
4. Attack the code: sliced inputs at bitmap-hostile offsets, forced small `ChunkSize`, executor pool
   reuse, the race detector, a checked allocator on every call, bad options and input kinds.
5. Differential fuzz: 31,760 cases across the twelve input types, plain, sliced, chunked and scalar,
   under every `skip_nulls`/`min_count` combination, compared bit for bit with pyarrow.
6. Fix what was found, rerun everything, and only then post.

### Finding 1: the note's own errors, caught before posting

| The note said | What C++ does (pyarrow from the same commit agrees) |
|---|---|
| `mean` of an empty input follows the same rule as `sum` (0 with `min_count=0`) | NaN for numeric and boolean inputs, 0.0 for the null type |
| in `min_max`, NaN never wins | true against a value; when every non-null value is NaN the result is NaN, because the state starts at NaN |
| `mode`, `quantile` and `tdigest` are scalar aggregates for later | `mode` and `quantile` are vector functions; `tdigest` returns an array, so a scalar-returning Finalize cannot implement it |
| no other options type in the package inverts a field for the zero value | `ArithmeticOptions.NoCheckOverflow` does, and `TakeOptions` uses a constructor |

### Finding 2: signed zeros in `min`, `max` and `min_max`

One divergence in the first 31,760 fuzz cases, 2,884 times, all the same cause. C's `fmin` and
`fmax`, which the C++ kernels use, order -0.0 below +0.0, so pyarrow returns `min` -0.0 and `max`
+0.0 whenever both appear, in any order. A comparison-based port keeps whichever zero came first:

```go
// before: b < a is false for (-0.0, +0.0), so the incumbent stays
case b < a:
    return b
// after: the sign is observable, so it is a case of its own
case b < a:
    return b
case b == 0 && a == 0 && math.Signbit(float64(b)):
    return b
```

After the fix the rerun matched on every case.

### Finding 3: float sums are pairwise

C++ `SumArray` sums floats pairwise in blocks of sixteen, the same algorithm numpy uses, and
`mean` accumulates in float64 for every input type. A plain left fold over 100,001 values gives
75247.35643756675 where pyarrow gives 75247.35643756694; the port of the pairwise loop matches bit
for bit, and the test that proves it is in the prototype.

### Finding 4: `ArraySpan.SetSlice` carries a stale null count

Every prototype kernel returned wrong answers once `ExecCtx.ChunkSize` was smaller than the input:
`count` of 339 valid values in 493 gave 193 at chunk size 1, 490 at 2, 467 at 4. The cause is not
in the kernels. `iterateExecSpans` reuses one `ArraySpan` per argument and advances it with
`SetSlice`, which on main reads:

```go
if a.Type.ID() != arrow.NULL {
    if a.Nulls != 0 {
        if a.Nulls == a.Len {
            a.Nulls = length          // "all null" carries over
        } else {
            a.Nulls = array.UnknownNullCount
        }
    }                                 // "no nulls" carries over too
} else {
    a.Nulls = length
}
```

That is right only while `Nulls` describes the whole array. A kernel that calls
`UpdateNullCount()` on the shared span stores the current slice's count; the next `SetSlice` then
treats the following slice as all valid or all null. Two kernels on main do exactly that, and the
reproducer shows it without any new code:

| ChunkSize | `and_kleene([T, null, T, null, F, T, null, T], [null, T, T, F, null, null, T, T])` |
|---|---|
| default | `[null null true false false null null true]` |
| 1 | `[null false true false false null false true]` |
| 2 | `[null null true false false false false true]` |
| 3 and above | correct |

`or_kleene` fails the same way; the `SetSlice` code is identical in v18.7.0. The C++
`ArraySpan::SetSlice` never carries a count when a validity bitmap is present, and the fix is that
rule:

```go
if a.Type.ID() != arrow.NULL {
    if a.Nulls != 0 || len(a.Buffers[0].Buf) != 0 {
        a.Nulls = array.UnknownNullCount
    }
} else {
    a.Nulls = length
}
```

With it, both Kleene kernels pass at every chunk size and the whole compute suite still passes.
The prototype's kernels, independently of that fix, count each span from its bitmap and never
write to the shared span, with a test at chunk sizes 1 to 65.

### What shipped, and what is open

- Merged the same day: [#1302](https://github.com/apache/arrow-go/pull/1302) (allocator alignment and
  which allocator to use when C keeps a buffer, closing [#1297](https://github.com/apache/arrow-go/issues/1297))
  and [#1303](https://github.com/apache/arrow-go/pull/1303) (what the compute registry holds).
- Open: the design note on [#1296](https://github.com/apache/arrow-go/issues/1296) with two questions
  for the maintainer; [#1305](https://github.com/apache/arrow-go/issues/1305) with its fix
  [#1306](https://github.com/apache/arrow-go/pull/1306) by singhpratech. The aggregate branch is pushed
  once the interface is agreed.
- Status of each is on the [Upstream tracker](UPSTREAM.md) and updates itself from arrow-go's tracker.

Logs, scripts and raw outputs of every run above (the audit reports, the build and lint log, the
fuzz fixtures and results, the reproducer and its output on both commits) are kept with the
release records.

## Round 9 (2026-09-06): radix select, and the LSD radix sort's blocking at small n

**TL;DR**

- A GPU radix select needs two passes over the column, not three, when the histogram keeps its
  counts per sub-block: at 50M rows that is 1 ms saved out of 3.
- After selection, ordering the ~200k survivors was the bottleneck, and the cause was a fixed
  4096-element block in the LSD radix sort. Halving the block until there are at least 64 blocks took
  `argsort` of 20k from 1.99 ms to 0.41 ms and is most of why `top_k(100)` went from 5.6 ms to 2.9 ms.
- The selected bin's size depends entirely on the key distribution's top byte. Float keys concentrate
  in a handful of bins, so a 50M Float64 median varies from run to run while Int64 keys are stable.

### The setting

A GPU radix select normally costs three passes over the column: histogram the top digit, count how
many rows fall in the selected range per block, then scatter them in order. This round built the
select behind `top_k`, then measured where the time went once selection itself was cheap: the final
ordering of the candidates, and the size of the bin the selection leaves behind.

### Research steps

1. Lay the histogram out per sub-block, so the count pass is redundant and the select is two passes.
2. Make the sub-block a simdgroup's slice rather than a threadgroup's, removing the barriers from the
   scatter.
3. Time the final `argsort` over the ~200k candidates; trace its cost to the fixed
   `elemsPerBlock = 4096`.
4. Halve the block size until there are at least 64 blocks; remeasure `argsort` at 20k and 200k rows
   and `top_k(100)` end to end.
5. Run the histogram and compaction a second time over the compacted candidates.
6. Read the benchmark generator's key distribution, and the top byte of a float64 key, against the
   bin sizes observed.

### Finding 1: the two-pass trick

The count pass is redundant if the histogram keeps its counts **per sub-block** instead of only
globally. The digit-major table `counts[digit * subBlocks + sub]` is simultaneously the global
histogram (summed over sub-blocks) and the offset table the scatter needs (summed over digits <=
target). At 50M rows that is 1 ms saved out of 3.

### Finding 2: one sub-block per simdgroup, not per threadgroup

Making the output granularity a simdgroup's slice rather than a threadgroup's removes every
`threadgroup_barrier` from the scatter: the rank of a selected row within its slice is
`simd_prefix_exclusive_sum` over 32 lanes plus a running base each lane computes identically. The
cost is an 8x larger count table (256 * 8 * groups words, ~8 MB at 50M rows), which is noise next to
the 400 MB the pass reads anyway.

### Finding 3: the final sort was the bottleneck, and it was a blocking bug

After selection there are only ~200k candidates left to order, but `argsort` of 200k UInt64 measured
2.2-2.9 ms, the time of sorting 800k. The cause was `elemsPerBlock = 4096` fixed: 200k rows is 49
threadgroups, 20k rows is *five*, on a GPU with 40 cores, and `radix_scatter` does an O(TG) rank
loop per element that nothing else can overlap. Halving the block size until there are at least 64
blocks (inputs above ~256k rows are untouched, so the 50M argsort is unchanged) gave:

| Input | Before | After |
|---|---|---|
| `argsort` of 20k | 1.99 ms | 0.41 ms |
| `argsort` of 200k | 2.20 ms | 1.34 ms |
| `top_k(100)` | 5.6 ms | 2.9 ms |

The block-size change is most of the `top_k(100)` improvement. The rule as it stands in
`Sources/ArrowMetal/Kernels/Sort.swift`:

```swift
func blockPlan(_ rows: Int) -> (elemsPerBlock: Int, blocks: Int) {
    var e = Swift.max(4096, ((rows + 127) / 128 + 255) / 256 * 256)
    while e > 256 && (rows + e - 1) / e < 64 { e >>= 1 }
    return (e, Swift.max(1, (rows + e - 1) / e))
}
```

### Finding 4: refine on the compacted array, not the column

The selected bin is ~n/256 rows, which still dominates the final ordering. Running the same
histogram + compaction *again* over the compacted candidates (a few hundred thousand keys,
microseconds) shrinks it by another 256x. One extra round trip, and it takes the survivors under the
2048 pairs a single-threadgroup bitonic sort can order in one dispatch.

### Finding 5: benchmark data hides skew

`rng.integers(-(2**62), 2**62)` spreads over only 128 of the 256 top-digit bins, so the candidate bin
is n/128, not n/256. Worth remembering when reading a selection benchmark: the bin size, and
therefore the final sort, depends entirely on the key distribution's top byte.

### Finding 6: a float key's top byte is the exponent, so float columns are the skewed case

The top byte of a float64 key is the sign plus seven exponent bits, so a column of uniform doubles
concentrates in a handful of bins rather than spreading over 256, and the bin holding the wanted rank
can be a large fraction of the column. When it is over the compaction budget the search narrows
another digit over the column instead, one more full pass, which is why the same 50M Float64 median
varies from run to run depending on exactly where the rank lands, while Int64 keys are stable.
Raising the budget so the big bin gets compacted instead is faster still, but a 25M-row bin needs
300 MB of scratch for a 400 MB column, and a median already 80x faster than pyarrow (4.61 ms against
368.88 ms at 50M rows, `Benchmarks/results/full_matrix_2026-09-07-parallel.csv`) is not worth a 75%
memory overhead.

### What shipped

- The two-pass select with per-simdgroup sub-blocks, planned by `TopK.radixPlan(n:)` in
  `Sources/ArrowMetal/Kernels/RadixSelect.swift`; the single-dispatch bitonic limit is `bitonicCap = 2048`
  in the same file.
- The block-size rule `blockPlan` in `Sources/ArrowMetal/Kernels/Sort.swift`, quoted above.
- The second refinement pass over the compacted candidates; the compaction budget left as it is, for the
  memory reason in Finding 6.
- The median figures are the `quantile(float64, 0.5)` rows at 50M in
  `Benchmarks/results/full_matrix_2026-09-07-parallel.csv`.

## Round 8 (2026-09-06): a threadgroup atomic read that is not uniform (`top_k`)

**TL;DR**

- One cell of a 13,000-case differential run diverged and then passed on every rerun. A stress test
  against a CPU oracle reproduced it at roughly 1 in 900 calls.
- Every thread of `topk_select` read the threadgroup atomic count `held` itself with a relaxed load;
  one simdgroup out of eight occasionally saw a newer value, so the sentinel fill left holes of zeroed
  memory that read as `(key 0, row 0)` and sorted to the front.
- Fix: thread 0 reads `held` once per chunk into a plain threadgroup variable between two barriers.
  The cost is below the noise floor: 3.11 / 3.31 / 3.38 ms with the fix against 3.22 / 3.30 / 4.37 ms
  without it for `top_k(100 of 20M Int64)`.
- Rule: never let a barrier-carrying branch, or a loop bound whose strides must tile a range, come from
  a per-thread `atomic_load_explicit(..., memory_order_relaxed)`.

### The setting

**Symptom.** One cell of a 13,000-case differential run diverged and then passed on every rerun:
`top_k / float64`, 100,003 rows, 30% nulls, k = 17, descending. `result[0]` was row 0 where pyarrow
says row 35254. A randomised stress against a CPU oracle (`TopKTests.testStressAgainstCPUOracle`)
reproduced it at roughly 1 in 900 calls across every key type, both directions, n from 32k to 1M and
k from 1 to 1024. The failure always looked the same: a run of zeros at the front of the result, the
true answer after them.

### Research steps

1. A stress test comparing against a CPU oracle (not against `argsort`, which shares the sort keys),
   with pool-churning kernels in between; this is `testStressAgainstCPUOracle` in
   `Tests/ArrowMetalTests/TopKTests.swift`.
2. Poison the candidate buffers before the dispatch, to prove the kernel really wrote those zeros
   rather than leaving stale bytes.
3. Add a debug buffer carrying each threadgroup's `held`, arrival count and compaction count, which
   showed the holes were inside `[0, c)` and `[c, cap)` in simdgroup-sized runs.
4. Instrument the kernel to count the holes per threadgroup.
5. Fix, measure the cost, and check the other `atomic_load_explicit` sites for the same pattern.

### Finding 1: a relaxed atomic load is not uniform across the threadgroup

**Root cause.** `topk_select` keeps a per-threadgroup candidate buffer in threadgroup memory with an
atomic count `held`. Every thread read that count itself:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
uint c = atomic_load_explicit(&held, memory_order_relaxed);   // Kernels/TopKSource.swift
...
for (uint i = c + lid; i < cap; i += TG) { bufKey[i] = keyMax; bufRow[i] = TK_NOROW; }
```

`c` bounds the range each thread pads with sentinels, and the union over `lid` covers `[c, cap)`
**only if every thread has the same `c`**. It does not: on an M4 Max, one simdgroup out of eight
occasionally comes back with a newer value than the rest. A relaxed atomic load is not ordered by the
preceding barrier the way a plain threadgroup read is, so it can be satisfied late, after other
threads have already run ahead and incremented `held`. The threads holding the larger `c` start their
stride later and the slots they should have covered are never written. Instrumenting the kernel showed
7 to 63 such holes per threadgroup, clustered at 32 and 64: one and two simdgroups.

### Finding 2: what a hole does to the answer

Those holes hold zeroed pool memory, which reads as `(key 0, row 0)`. That pair is the *minimum* of
the `(key, row)` order, so the bitonic sort moves it to the front of the buffer and it becomes the
block's answer; the host then sees row 0 as the top candidate. Worse, if a block accumulates k or more
holes the compaction sets its threshold to `bufKey[k-1] = (0, 0)`, which nothing can beat, and the
block is blind for the rest of the scan. The same read also decides `if (c + TG > cap)`, a branch
containing `threadgroup_barrier`, so a non-uniform `c` was undefined behaviour in its own right.

### What shipped

- **Fix.** Thread 0 reads `held` once per chunk into a plain `threadgroup uint shared_c` (clamped to
  `cap`) between two barriers, and every thread takes `c` from there; the branch and the fill ranges
  are now uniform by construction. The kernel as it stands in `Sources/ArrowMetal/Kernels/TopKSource.swift`:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid == 0u) shared_c = min(atomic_load_explicit(&held, memory_order_relaxed), cap);
threadgroup_barrier(mem_flags::mem_threadgroup);
uint c = shared_c;
if (c + TG > cap) {
    for (uint i = c + lid; i < cap; i += TG) { bufKey[i] = \(keyMax); bufRow[i] = TK_NOROW; }
```

- In addition, the buffer is filled with sentinels once at kernel entry, so a slot no one writes reads
  as "no row" and the host drops it instead of it masquerading as row 0.
- **Cost.** One extra `threadgroup_barrier` per 256-row chunk, and one `cap`-element sentinel fill per
  block (2 to 8 strided stores per thread, once). Below the noise floor: `top_k(100 of 20M Int64)` on
  M4 Max runs 3.11 / 3.31 / 3.38 ms with the fix against 3.22 / 3.30 / 4.37 ms without it. The pass is
  still about one read per row.
- **Test.** `TopKTests.testStressAgainstCPUOracle`, the stress against the CPU oracle.
- **Rule.** In MSL, never let a value that decides a barrier-carrying branch, or a per-thread loop
  bound whose strides must tile a range, come from a per-thread
  `atomic_load_explicit(..., memory_order_relaxed)`. Broadcast it through a plain threadgroup variable
  between barriers. The other `atomic_load_explicit` sites (`SortSource`, `GroupBySource`,
  `JoinSource`, `StringExtraSource`) were checked: each reads a slot the calling thread owns, so none
  of them tiles or branches on a shared count.

## Round 8b (2026-09-06): the differential matrix over the whole type surface

**TL;DR**

- Extending `python/tests/test_differential.py` to every type ArrowMetal imports (45 columns, 181
  operations, 33,156 cases at the time; 212 operations and 39,069 cases today) turned up three bugs
  and one unreproduced crash **in pyarrow 25.0.1**, not in ArrowMetal.
- Each has a workaround in the harness and a test that fails if a later pyarrow fixes it.
- Two properties of the harness itself: time-of-day cases have to drop their null rows, and a 38-digit
  `Decimal` has to be read off `as_tuple().digits`.

### The setting

The differential matrix compares every ArrowMetal operation against pyarrow over every imported type.
Extending it to the whole type surface is what surfaced the pyarrow behaviours below. They are
recorded here because the harness has to work around them, and each has a test that fails if a later
pyarrow fixes it.

### Research steps

1. Extend the matrix to 45 columns and 181 operations, 33,156 cases (212 operations and 39,069 cases
   today).
2. Trace one segfault back from the faulting frame, `Array.nbytes` inside the harness's own array
   cache, to the preceding `pc.year_month_day` call, which cost an hour.
3. Compare `pc.utf8_normalize` against Python's `unicodedata` and the Unicode annex for each form.
4. Run `pc.pairwise_diff` on a sliced column whose values are not an arithmetic progression, then the
   same slice check on the boolean fill and replace kernels.
5. Give each oracle a workaround, and pin each pyarrow behaviour with a test that fails when it changes.

### Finding 1: a single unreproduced segfault after `pc.year_month_day`

A **single unreproduced segfault** was observed a few allocations after `pc.year_month_day` on a
`timestamp[s]` array with nulls; it has not recurred. It landed in whatever unrelated call happened
next: the faulting frame was `Array.nbytes` inside the harness's own array cache, which cost an hour
to trace back. Out of caution the matrix does not use the two struct-valued temporal kernels as
oracles: it compares `iso_calendar` and `year_month_day` field by field against `pc.iso_year`/
`pc.iso_week`/`pc.day_of_week` and `pc.year`/`pc.month`/`pc.day`, which is a stronger check anyway.

### Finding 2: `pc.utf8_normalize` ignores its `form` option

`pc.utf8_normalize` **ignores its `form` option**: NFC and NFKC come back decomposed, so its NFC is
NFD. Python's `unicodedata` and ArrowMetal agree with each other and with the Unicode annex; the
matrix uses `unicodedata` as the oracle:

```python
for form in ("NFC", "NFKC", "NFD", "NFKD"):
    got.append(arrow(x.utf8_normalize(form)))
    expected.append(pa.array([None if s is None else unicodedata.normalize(form, s)
```

Pinned by `test_pyarrow_utf8_normalize_ignores_its_form_option`.

### Finding 3: `pc.pairwise_diff` ignores `ArrowArray.offset`

`pc.pairwise_diff` **ignores `ArrowArray.offset`**: on a sliced column it reads the values buffer
from the start and answers with the wrong rows. Only visible when the values are not an arithmetic
progression, which is why it hid for a while. The matrix hands that oracle a materialised copy while
ArrowMetal still gets the slice, so the case remains a test of the offset handling:

```python
def _materialised(src):
    """The same values with the Arrow offset folded away."""
    if src.offset == 0:
        return src
    return pa.concat_arrays([src.slice(0, 0), src])
```

Fixed upstream in pyarrow 26.0.0; the workaround stays only while 25.0.1 is the pinned version.

### Finding 4: the same offset bug on boolean columns

`pc.fill_null_forward`, `pc.fill_null_backward` and `pc.replace_with_mask` have the same offset bug
on a **boolean** column (the values bitmap, not the validity one). Same mitigation.

### Finding 5: `pc.add` on time-of-day validates the values under the validity bitmap

`pc.add` on a `time32`/`time64` column validates the *values under the validity bitmap*, so a null
row whose hidden value would leave `[0, 86400)` makes the oracle raise even though the row is null.
The generator deliberately puts real numbers under the null bits, so the time-of-day cases have to
drop their null rows rather than mask them.

### Finding 6: a 38-digit `Decimal` and the default context

A `Decimal` that came out of a 38-digit column cannot be scaled with `Decimal.scaleb` or multiplied
by `10 ** scale` under the default decimal context; 28 digits of precision silently round it. Read
the unscaled magnitude off `as_tuple().digits` instead:

```python
def _unscaled_magnitude(value):
    return int("".join(str(d) for d in value.as_tuple().digits) or "0")
```

### What changed

- The oracles in `python/tests/test_differential.py`: field-by-field comparison for `iso_calendar`
  and `year_month_day`, `unicodedata` for normalisation, `_materialised` copies for `pc.pairwise_diff`
  and the boolean fill and replace kernels, dropped null rows for time-of-day addition, and
  `_unscaled_magnitude` for decimals.
- The pinning tests in the same file: `test_pyarrow_utf8_normalize_ignores_its_form_option`,
  `test_pyarrow_pairwise_diff_ignores_the_array_offset` and
  `test_pyarrow_boolean_fill_null_forward_ignores_the_array_offset`; each fails when a later pyarrow
  changes the behaviour it pins.

## Round 7 (2026-09-06): strings and sort

**TL;DR**

- Generated MSL written through a shell heredoc needs `\(K)` (one backslash) for Swift interpolation.
- LSD radix sort with 8-bit digits, 4096-element blocks and a stable in-chunk ranking is correct
  across all types and sizes; descending order must invert keys, not reverse the ascending result.
- MurmurHash3 x86_32 matches the reference vectors; four collisions among 100k 32-bit hashes is the
  birthday bound, not a bug.

### The setting

The string kernels (hashing) and the LSD radix sort, with their MSL generated from Swift source
written through a shell heredoc.

### Research steps

1. Generate MSL through a shell heredoc and check what reaches the Metal compiler.
2. Run the LSD radix sort across all types and sizes, in both directions, and check ties.
3. Check MurmurHash3 x86_32 against its reference vectors and count collisions among 100k hashes.

### Finding 1: heredoc interpolation

Generated MSL written through a shell heredoc must use `\(K)` (one backslash) for Swift
interpolation; a doubled backslash reaches the Metal compiler as literal text. Same newline rule as
before for `#define` (Round 6, Finding 3).

### Finding 2: the radix sort, and descending order

LSD radix sort with 8-bit digits, 4096-element blocks and a stable in-chunk ranking is correct across
all types and sizes; descending order must invert keys rather than reverse the ascending result, or
ties flip.

### Finding 3: MurmurHash3 reference vectors

MurmurHash3 x86_32 reference vectors (seed 0):

| Input | Hash |
|---|---|
| "" | 0 |
| "a" | 0x3c2569b2 |
| "abc" | 0xb3dd93fa |
| "hello" | 0x248bfa47 |

Four collisions among 100k 32-bit hashes is normal (birthday bound), not a bug.

### What shipped

- The heredoc rule for generated MSL.
- Key inversion for descending sorts in `Sources/ArrowMetal/Kernels/Sort.swift`.
- The reference vectors above are checked in `Tests/ArrowMetalTests/StringTests.swift`, from "" -> 0
  through "hello" -> 0x248bfa47.

## Round 6 (2026-09-06): software IEEE-754 double on the GPU

**TL;DR**

- Add, subtract and multiply on `ulong` with 3 guard bits and sticky are bit-exact against Swift's
  `Double` for 1M random pairs including subnormals, signed zeros, infinities and NaN.
- A float-seeded Newton division was within 1 ulp only 93% of the time and wrong for subnormal
  inputs; restoring long division on the significands (57 quotient bits) replaced it.
- Generated MSL fragments that start with a `#define` must begin with a newline, because Swift
  multi-line strings drop the final newline.
- The Float64 GPU sum accumulates with `d_add` in tree order and differs from a sequential CPU sum
  only by normal floating-point reordering.

### The setting

Float64 arithmetic implemented in MSL on `ulong` bit patterns, tested against Swift's `Double`.

### Research steps

1. Implement add, subtract and multiply with 3 guard bits and sticky; compare against `Double` on 1M
   random pairs including subnormals, signed zeros, infinities and NaN.
2. Implement division as a float-seeded Newton iteration and measure how often it is within 1 ulp.
3. Replace it with restoring long division on the significands, and test on ratios above and below 1.
4. Fix the `#define` that glued to the previous line in the generated source.
5. Sum Float64 on the GPU with `d_add` and combine the partials on the CPU.

### Finding 1: add, subtract and multiply are bit-exact

Add, subtract and multiply implemented on `ulong` with 3 guard bits and sticky are bit-exact against
Swift's `Double` for 1M random pairs including subnormals, signed zeros, infinities and NaN
(`d_finish` handles normalisation, subnormal shift-with-sticky, round-to-nearest-even, and rounding
carry).

### Finding 2: division

A float-seeded Newton iteration was within 1 ulp only 93% of the time and wrong for subnormal inputs.
Replaced with restoring long division on the significands (57 quotient bits). Two bugs on the way: one
extra quotient bit shifted every result by 2x, and the restoring loop needs `rem < mb` before the first
step (take the first quotient bit explicitly). Lesson: test division on ratios above and below 1.

### Finding 3: a `#define` glued to the previous line

A preprocessor `#define` glued to the previous line (`}#define`) because Swift multi-line strings drop
the final newline. Generated MSL fragments that start with a directive must begin with a newline.

### Finding 4: the Float64 sum

Float64 sum on the GPU accumulates with `d_add` in tree order; per-threadgroup partials are combined
on the CPU in `Double`. Results differ from a sequential CPU sum only by normal floating-point
reordering.

### What shipped

- `d_add`, `d_sub`, `d_mul`, `d_div` and `d_finish` in `Sources/ArrowMetal/Kernels/DoubleMath.swift`.
- Tests in `Tests/ArrowMetalTests/DoubleMathTests.swift`: `testAddSubMulBitExact`, `testDivBitExact`
  and `testDoubleSumOnGPU`.
- The newline rule for generated fragments that start with a directive.

## Round 5 (2026-09-06): pipeline creation on the paravirtual GPU, and lengths that stay on the device

**TL;DR**

- The paravirtual GPU on GitHub runners fails `makeComputePipelineState` for arbitrary trivial kernels
  with no diagnostic while identical kernels pass in the same run. Mitigation: one retry, and GPU tests
  `XCTSkip` on devices whose name contains "Paravirtual".
- Kernels now read their element count from a device buffer, so a filter followed by a sum is one
  round trip.

### The setting

Round 4 left two things open: `makeComputePipelineState` failures on GitHub's paravirtual GPU, and the
second round trip a `sum()` on a batched filter result paid to learn the filtered length.

### Research steps

1. Enable full compiler logs on the paravirtual GPU and compare the kernels that fail with those that
   pass in the same run.
2. Add one retry on pipeline creation, and skip GPU tests on virtual devices.
3. Move the element count into a device buffer and bind a pending filter's GPU-written total as that
   buffer.

### Finding 1: the paravirtual GPU fails arbitrary trivial kernels

With full compiler logs enabled, the paravirtual GPU on GitHub runners fails `makeComputePipelineState`
for arbitrary trivial kernels (`bitmap_not`, `cast_kernel`) with no diagnostic while identical kernels
pass in the same run. It is the virtual Metal stack, not a construct. The retry in
`Sources/ArrowMetal/MetalContext.swift`:

```swift
do { p = try device.makeComputePipelineState(function: fn) }
catch {
    // Virtualised GPUs (GitHub's "Apple Paravirtual device") fail pipeline creation sporadically; retry once.
    usleep(20_000)
    p = try device.makeComputePipelineState(function: fn)
}
```

### Finding 2: lengths that stay on the device

Kernels now read their element count from a device buffer (`device const uint* nPtr`). A pending
filter result binds its GPU-written total as that buffer, so compare / arithmetic / cast / bitmap ops /
another filter / a reduction can consume it inside the same command buffer with worst-case dispatch
sizes. Reductions still sync to read partials, but a filter followed by a sum is now one round trip.

### What shipped

- One retry on pipeline creation, quoted above.
- GPU tests `XCTSkip` on devices whose name contains "Paravirtual" (`ARROWMETAL_FORCE_GPU_TESTS=1`
  overrides): `requireRealGPU()` in `Tests/ArrowMetalTests/TestSupport.swift`. CI validates the
  build, interop and CPU paths; GPU correctness runs on real hardware.
- `device const uint* nPtr` as the element count of every kernel, so pending lengths flow on the GPU.

## Round 4 (2026-09-06): the paravirtual GPU on GitHub runners

**TL;DR**

- GitHub's `macos-15` Apple silicon runners expose an "Apple Paravirtual device" GPU with 3 CPU cores,
  ~16 GB/s for a sum. Use CI for correctness only; never publish its numbers.
- The paravirtual GPU failed `makeComputePipelineState` for kernels that pass on M4 Max;
  `ARROWMETAL_DEBUG_SHADERS=1` now dumps the generated MSL and the compiler log on failure.
- The ~120 µs per-call floor is the round trip; batching is what removes it, and a `sum()` on a batched
  filter still paid a second trip, which Round 5 removed.

### The setting

The first CI runs on GitHub's hosted Apple silicon runners, after the Round 3 push, and the per-call
latency of a single dispatch on M4 Max.

### Research steps

1. Measure a sum on the `macos-15` runner's GPU.
2. Read the pipeline creation failures on the runner for kernels that pass on M4 Max, and add a way to
   see the generated source and the compiler log.
3. Try `makeCommandBufferWithUnretainedReferences` plus a spin-wait against the per-call latency.
4. Count the round trips of a `sum()` on a batched filter result.

### Finding 1: the runner GPU

GitHub's `macos-15` Apple silicon runners expose an "Apple Paravirtual device" GPU with 3 CPU cores:
~16 GB/s for a sum. Use CI for correctness only; never publish its numbers.

### Finding 2: pipeline creation fails on the runner

The paravirtual GPU failed `makeComputePipelineState` for kernels that pass on M4 Max (round 3 push).
`ARROWMETAL_DEBUG_SHADERS=1` now dumps generated MSL and the full compiler log on failure; CI sets it.

### Finding 3: the per-call floor

`makeCommandBufferWithUnretainedReferences` plus a spin-wait did not measurably reduce per-call
latency; the ~120 µs floor is the round trip. Batching is what works: chains pay it once.

### Finding 4: a reduction on a batched filter

A `sum()` on a batched filter result still costs a second round trip because the reduction needs the
filtered length on the CPU. Done in Round 5 above: kernels read `n` from a device buffer so pending
lengths flow on the GPU.

### What shipped

- `ARROWMETAL_DEBUG_SHADERS=1`, read in `Sources/ArrowMetal/MetalContext.swift`; the CI job in
  `.github/workflows/ci.yml` runs `ARROWMETAL_DEBUG_SHADERS=1 swift test`.
- The rule that CI numbers are never published.

## Round 3 (2026-09-06): integer division by zero, and the group-by atomics

**TL;DR**

- Metal integer division by zero returns an unspecified value; ArrowMetal defines it as 0 in both the
  GPU kernels and the CPU reference, and `Int.min / -1` wraps.
- Threadgroup-privatised group-by with 32-bit atomics reaches ~300 GB/s for up to 1024 keys, the same
  rate as a plain sum; device atomics at 100k keys halve it.
- Crossing the C boundary from Python, and exporting a result back to pyarrow, cost nothing
  measurable; importing pyarrow buffers is one memcpy.

### The setting

The arithmetic kernels, the group-by, and the Python binding's path across the C boundary.

### Research steps

1. Observe what Metal returns for integer division by zero, and define it the same way in the GPU
   kernels and the CPU reference.
2. Measure the threadgroup-privatised group-by at up to 1024 keys and with device atomics at 100k keys.
3. Time a call across the C boundary from Python, with and without exporting the result to pyarrow.
4. Check the alignment of pyarrow's buffers against what a zero-copy import needs.

### Finding 1: integer division by zero

Metal integer division by zero returns an unspecified value (observed 1 on M4 Max). ArrowMetal now
defines it as 0 in both the GPU kernels and the CPU reference, and `Int.min / -1` wraps instead of
trapping. The generator in `Sources/ArrowMetal/Kernels/KernelSource.swift`:

```swift
/// Integer division by zero yields 0 (defined here, matching the CPU reference); overflow wraps.
if name == "div" && isInt {
    sclArray = "(b[i] == (\(T))0) ? (\(T))0 : a[i] / b[i]"
```

### Finding 2: the group-by atomics

Threadgroup-privatised group-by with 32-bit atomics reaches ~300 GB/s for up to 1024 keys, the same
rate as a plain sum: the atomics are not the bottleneck at that key count. Device atomics at 100k keys
halve it.

### Finding 3: the C boundary from Python

Crossing the C boundary from Python costs nothing measurable per call (ctypes overhead is ~10 µs);
exporting a result back to pyarrow costs nothing measurable:

| Call | Time |
|---|---|
| without export | 3.55 ms |
| with export | 3.57 ms |

### Finding 4: importing pyarrow buffers

Importing pyarrow buffers is one memcpy: pyarrow's allocator is 64-byte aligned, not page aligned.

### What shipped

- Division by zero defined as 0 in the GPU kernels and the CPU reference; `Int.min / -1` wraps.
- The threadgroup-privatised group-by.

## Background

Reference material rather than investigations: what the ecosystem offers, what the toolchain does,
and what the hardware gives a kernel.

### Ecosystem research (2026-09-06)

**TL;DR**

- No existing Arrow library computes on Metal; the pieces that exist are types, IPC, device buffer
  wrappers and tensors.

**What exists**

- `apache/arrow-swift` v21: types, IPC, Flight, C Data Interface. No compute kernels, nothing Metal.
- `arrow-nanoarrow` device extension: C wrapper for `ARROW_DEVICE_METAL` buffers, no compute.
- cuDF: CUDA only. MLX: unified-memory tensors, zero-copy `MTLBuffer` access, no nulls or columnar
  semantics.
- A DuckDB extension with Metal aggregates exists (gpudb); it does not use Arrow buffers.

**The device interface**

- The Arrow C Device Data Interface defines `ARROW_DEVICE_METAL = 8` and expects `MTLEvent*` as
  `sync_event`.

### Toolchain

**TL;DR**

- Run tests with Xcode's `DEVELOPER_DIR` and always in release as well as debug.
- Swift 6.3.3 miscompiles a `withUnsafeBytes` closure in a generic `throws` function under `-O`
  (swiftlang/swift#90477, open); `MetalArray.init(_:)` uses a plain loop instead.
- The `macos-15` runner's Swift 6.1 refuses closures Swift 6.3 accepts; release builds shorten object
  lifetimes; `posix_memalign` memory is not zero.

**Running the tests**

- XCTest is not in the Command Line Tools. Run tests with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Always run `swift test -c release`.

**Compilers**

- Swift 6.3.3 miscompiles a `withUnsafeBytes` closure inside a generic `throws` function under `-O`
  when the caller is in another module: the closure clobbers the error register around an Objective-C
  message send, so the function reports a phantom error and the caller crashes retaining it
  (swiftlang/swift#90477, open). `MetalArray.init(_:)` uses a plain element loop instead; the
  Metal-free reproducer is in UPSTREAM.md.
- GitHub `macos-15` runners come with Xcode 16.4 / Swift 6.1, which refuses to type-check dense
  closures that Swift 6.3 accepts. Keep test expressions simple.

**Memory**

- Swift release builds shorten object lifetimes to last use: a raw pointer taken from a buffer object
  can outlive the object. Use `withExtendedLifetime` or the closure accessors.
- `posix_memalign` memory is not zero for small blocks (recycled heap); only fresh mmap pages are
  zero.

### Metal: what the M4 Max gives a kernel

**TL;DR**

- 32 KB threadgroup memory, SIMD width 32, unified memory, no 64-bit atomic add from MSL.
- Runtime `makeLibrary(source:)` needs no `metal` toolchain; `makeBuffer(length:)` buffers are not
  page aligned.
- Memory-bound element-wise kernels run at ~375 GB/s against ~350 GB/s on the 16-core CPU; reductions
  and compaction favour the GPU by 2x to 3x.

**The hardware**

- Apple M4 Max: 32 KB threadgroup memory, SIMD width 32, unified memory, no 64-bit atomic add from MSL
  (`atomic_ulong` fetch_add fails to compile; min and max do). 64-bit sums use split 32-bit atomics
  with carry.

**The API**

- Runtime `makeLibrary(source:)` works without the `metal` toolchain. Errors carry line numbers of the
  generated source.
- `makeBuffer(length:)` buffers are heap sub-allocated and not page aligned; `bytesNoCopy` requires
  page alignment of pointer and length.

**Bandwidth**

- Memory-bound element-wise kernels run at ~375 GB/s on M4 Max once allocation overhead is removed;
  the 16-core CPU reaches ~350 GB/s on the same loops. Reductions and compaction favour the GPU by 2x
  to 3x.
- Fusing the predicate into the filter's counting pass avoids materialising a boolean array.
