# ArrowMetal for Go

`github.com/singhpratech/ArrowMetal/go/arrowmetal` — Apache Arrow compute on Apple silicon GPUs,
from Go, over the Arrow C Data Interface.

The full guide is [docs/GO.md](../docs/GO.md): install, the copy rule, what is covered, what is not,
and the limits. This page is the short version and the measured timing.

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product ArrowMetalC
go get github.com/singhpratech/ArrowMetal/go/arrowmetal
```

The `go get` line works once the repository is public and carries a `go/arrowmetal/vX.Y.Z` tag; the
module sits in a subdirectory, so the proxy wants the tag prefixed with that path. Until then, use a
`replace` directive against a checkout, or work inside `go/arrowmetal`.

```go
b := array.NewInt64Builder(am.NewPageAlignedAllocator())
b.AppendValues([]int64{5, 3, 9, 1}, []bool{true, true, false, true})
src := b.NewArray()

gpu, err := am.Import(src)              // arrow.Array -> GPU
sum, err := gpu.Sum()                   // 9; nulls skipped, like Arrow
mask, err := gpu.CompareScalar(am.Gt, int64(2))
kept, err := gpu.Filter(mask)
out, err := kept.Export()               // back to arrow-go, no copy
```

The binding `dlopen`s `libArrowMetalC.dylib`: `$ARROWMETAL_LIB` (the full path to the file) first,
then `.build/release/` at up to three levels above the working directory and above the binary. When
it finds nothing the error names the variable, every path it tried, and the `swift build` line.

## Test

```bash
cd go/arrowmetal
ARROWMETAL_LIB=$PWD/../../.build/release/libArrowMetalC.dylib go test ./...
GOEXPERIMENT=cgocheck2 ARROWMETAL_LIB=$PWD/../../.build/release/libArrowMetalC.dylib go test -count=1 ./...
```

46 test functions, 177 cases with subtests, run both ways. Every wrapped operation is checked
against Arrow Go's own compute where arrow-go has the function, and against a plain Go loop where it
does not (arrow-go's compute package registers no aggregate function at all — no `sum`, `mean` or
`min_max`). Lengths 0, 1, 1000 and 1,000,001; nulls at several densities, in keys as well as values;
sliced input at offsets 1 through 1000.

The second run arms cgo's pointer checker. `Import` pins the buffers it hands to C, because
arrow-go's `cdata` publishes Go-heap pointers into C memory by default
([apache/arrow-go#70](https://github.com/apache/arrow-go/issues/70)) and that is what cgocheck2
catches — see [docs/GO.md](../docs/GO.md#go-pointers-cgo-and-why-pinning-is-not-optional).

## Timing

10M Int64 rows (76 MB), no nulls, M4 Max, one process, wall clock, **best of 5 timed runs after one
untimed warm-up**. ArrowMetal 0.1.0, Go 1.27.1, arrow-go v18.7.0. Reproduce with
`go run ./cmd/amtiming`. The filter predicate is `x > 0`, keeping 50.0% of the rows.

| Op | Method | Best of 5 |
|---|---|---:|
| Import | arrow-go → ArrowMetal, page-aligned buffer (borrowed) | 1.13 ms |
| Import | arrow-go → ArrowMetal, buffer 64 B past a page (one copy) | 1.41 ms |
| Import | arrow-go → ArrowMetal, `memory.NewGoAllocator` | 1.40 ms |
| Export | ArrowMetal → arrow-go (always copy-free) | 917 ns |
| Sum | plain Go loop | 2.46 ms |
| Sum | Arrow Go `arrow/math` (NEON) | **1.16 ms** |
| Sum | ArrowMetal, array already resident | **296 µs** |
| Sum | ArrowMetal, end to end from an `arrow.Array` | 2.22 ms |
| Filter | plain Go loop | 29.18 ms |
| Filter | Arrow Go compute (`greater` then `filter`) | 48.93 ms |
| Filter | ArrowMetal, array already resident | **1.51 ms** |
| Filter | ArrowMetal, end to end from an `arrow.Array` | **3.22 ms** |

**ArrowMetal loses at Sum end to end**: 2.22 ms against Arrow Go's 1.16 ms. Summing 76 MB once is a
bandwidth problem the CPU already handles well, and about 1.2 ms of the ArrowMetal number is the
import. Resident, the same sum is 296 µs — 3.9× faster than arrow-go. If the shape of your program
is "load an array, sum it once, drop it", this is the wrong tool.

**ArrowMetal wins at Filter in every shape**: 3.22 ms end to end against 48.93 ms is 15×, and 1.51 ms
resident is 32×. Filter does enough work per byte that the fixed cost stops dominating.

`arrow/math.Int64.Sum` ignores nulls and the data here has none; it is the fastest sum arrow-go
offers, which is why it is the comparison. The plain Go filter writes a `[]int64` rather than
building an `arrow.Array`, so it is doing less work than the other two Filter rows.

The full table, including what the copy rule costs and where the numbers are inside the noise, is in
[docs/GO.md](../docs/GO.md#timing).
