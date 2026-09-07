// Command amtiming measures Sum and Filter at 10M Int64 rows three ways: ArrowMetal from Go, Arrow
// Go's own code, and a plain Go loop. It prints the table that goes in go/README.md and docs/GO.md.
//
//	go run ./cmd/amtiming
//
// Method: best of 5 timed runs after one untimed warm-up, wall clock, one process, no nulls. Each
// case is timed as a whole (`time.Since` around the work), and the result is used afterwards so the
// compiler cannot elide it.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	armath "github.com/apache/arrow-go/v18/arrow/math"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

var (
	rows   = flag.Int("rows", 10_000_000, "number of int64 rows")
	trials = flag.Int("trials", 5, "timed runs per case; the best is reported")
)

// bestOf runs fn once untimed and then `n` times timed, returning the shortest wall time.
func bestOf(n int, fn func()) time.Duration {
	fn() // warm-up: first-touch faults, Metal pipeline compilation, Go's own lazy init
	best := time.Duration(1<<63 - 1)
	for i := 0; i < n; i++ {
		start := time.Now()
		fn()
		if d := time.Since(start); d < best {
			best = d
		}
	}
	return best
}

func mustf(err error, f string, a ...any) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "amtiming: "+f+": %v\n", append(a, err)...)
		os.Exit(1)
	}
}

// misalignedAllocator hands out buffers that deliberately start 64 bytes past a page boundary, so
// that ArrowMetal has to copy them on import. Without it the "does the copy rule cost anything?"
// question is answered by luck: whether the Go heap happened to place the buffer on a 16 KiB
// boundary, which it does maybe a quarter of the time at this size.
type misalignedAllocator struct{}

func (misalignedAllocator) Allocate(n int) []byte {
	page := am.PageSize()
	raw := make([]byte, n+2*page)
	pad := (page - int(uintptr(unsafe.Pointer(&raw[0]))%uintptr(page))) % page
	off := pad + 64
	return raw[off : off+n : off+n]
}

func (a misalignedAllocator) Reallocate(n int, b []byte) []byte {
	out := a.Allocate(n)
	copy(out, b)
	return out
}

func (misalignedAllocator) Free([]byte) {} // the Go collector owns it

// pageOffset is how far an array's values buffer sits past a page boundary. Zero means ArrowMetal
// can borrow it; anything else means one copy on import.
func pageOffset(a arrow.Array) uintptr {
	bufs := a.Data().Buffers()
	if len(bufs) < 2 || bufs[1] == nil || bufs[1].Len() == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&bufs[1].Bytes()[0])) % uintptr(am.PageSize())
}

func buildInt64(mem memory.Allocator, vals []int64) arrow.Array {
	b := array.NewInt64Builder(mem)
	defer b.Release()
	b.AppendValues(vals, nil)
	return b.NewArray()
}

type result struct {
	op, method string
	d          time.Duration
	note       string
}

func main() {
	flag.Parse()
	mustf(am.Init(), "loading the library")
	dev, _ := am.DeviceName()
	ver, _ := am.Version()

	n := *rows
	vals := make([]int64, n)
	x := int64(1)
	for i := range vals {
		x = x*6364136223846793005 + 1442695040888963407
		vals[i] = x >> 40
	}

	goMem := memory.NewGoAllocator()
	alignedMem := am.NewPageAlignedAllocator()

	srcGo := buildInt64(goMem, vals)
	defer srcGo.Release()
	srcAligned := buildInt64(alignedMem, vals)
	defer srcAligned.Release()
	srcMisaligned := buildInt64(misalignedAllocator{}, vals)
	defer srcMisaligned.Release()

	// The array already resident on the GPU, for the kernel-only rows.
	resident, err := am.Import(srcAligned)
	mustf(err, "importing the values")
	defer resident.Release()

	ctx := context.Background()
	var out []result
	add := func(op, method string, d time.Duration, note string) {
		out = append(out, result{op, method, d, note})
	}

	// ---- interop on its own ------------------------------------------------------------------------
	// These two rows are the copy rule, measured: the only difference between them is whether the
	// producer's buffer starts on a page boundary, which is what decides borrow-or-copy on the way in.
	add("Import", "arrow-go -> ArrowMetal, page-aligned", bestOf(*trials, func() {
		h, err := am.Import(srcAligned)
		mustf(err, "Import")
		h.Release()
	}), "buffer from arrowmetal.PageAlignedAllocator")

	add("Import", "arrow-go -> ArrowMetal, one copy", bestOf(*trials, func() {
		h, err := am.Import(srcMisaligned)
		mustf(err, "Import")
		h.Release()
	}), "buffer deliberately 64 bytes past a page boundary")

	add("Import", "arrow-go -> ArrowMetal, Go allocator", bestOf(*trials, func() {
		h, err := am.Import(srcGo)
		mustf(err, "Import")
		h.Release()
	}), fmt.Sprintf("memory.NewGoAllocator; this run's buffer sits %d bytes past a page boundary",
		pageOffset(srcGo)))

	add("Export", "ArrowMetal -> arrow-go", bestOf(*trials, func() {
		a, err := resident.Export()
		mustf(err, "Export")
		a.Release()
	}), "always copy-free")

	// ---- Sum -------------------------------------------------------------------------------------
	var sink int64

	add("Sum", "plain Go loop", bestOf(*trials, func() {
		var s int64
		for _, v := range vals {
			s += v
		}
		sink += s
	}), "one pass over a []int64")

	add("Sum", "Arrow Go (arrow/math, NEON)", bestOf(*trials, func() {
		sink += armath.Int64.Sum(srcGo.(*array.Int64))
	}), "arrow-go registers no `sum` compute function; arrow/math is its own vectorised sum")

	add("Sum", "ArrowMetal, array resident", bestOf(*trials, func() {
		s, err := resident.Sum()
		mustf(err, "Sum")
		sink += s.Int64()
	}), "kernel plus the GPU sync; no import")

	add("Sum", "ArrowMetal, end to end (page-aligned)", bestOf(*trials, func() {
		h, err := am.Import(srcAligned)
		mustf(err, "Import")
		s, err := h.Sum()
		mustf(err, "Sum")
		sink += s.Int64()
		h.Release()
	}), "import is copy-free: the buffer is page aligned")

	add("Sum", "ArrowMetal, end to end (one copy in)", bestOf(*trials, func() {
		h, err := am.Import(srcMisaligned)
		mustf(err, "Import")
		s, err := h.Sum()
		mustf(err, "Sum")
		sink += s.Int64()
		h.Release()
	}), "the same, with an import that has to copy 76 MB")

	// ---- Filter ----------------------------------------------------------------------------------
	// The predicate is x > 0, roughly half the rows.
	var kept int

	add("Filter", "plain Go loop", bestOf(*trials, func() {
		dst := make([]int64, 0, len(vals))
		for _, v := range vals {
			if v > 0 {
				dst = append(dst, v)
			}
		}
		kept = len(dst)
	}), "one fused pass; produces a []int64, not an arrow.Array")

	zero := compute.NewDatum(scalar.NewInt64Scalar(0))
	defer zero.Release()
	add("Filter", "Arrow Go compute", bestOf(*trials, func() {
		// NewDatum retains, so this Release is the matching one. (Its sibling
		// NewDatumWithoutOwning does not retain and must not be released -- an easy way to
		// drive an arrow.Array's refcount negative and free its buffers underneath you.)
		lhs := compute.NewDatum(srcGo)
		maskD, err := compute.CallFunction(ctx, "greater", nil, lhs, zero)
		mustf(err, "compute.greater")
		mask := maskD.(*compute.ArrayDatum).MakeArray()
		res, err := compute.FilterArray(ctx, srcGo, mask,
			compute.FilterOptions{NullSelection: compute.SelectionDropNulls})
		mustf(err, "compute.FilterArray")
		kept = res.Len()
		res.Release()
		mask.Release()
		maskD.Release()
		lhs.Release()
	}), "greater + filter, two passes, produces an arrow.Array")

	add("Filter", "ArrowMetal, array resident", bestOf(*trials, func() {
		mask, err := resident.CompareScalar(am.Gt, int64(0))
		mustf(err, "CompareScalar")
		res, err := resident.Filter(mask)
		mustf(err, "Filter")
		kept = int(res.Len())
		res.Release()
		mask.Release()
	}), "compare + filter on the GPU; no import, no export")

	add("Filter", "ArrowMetal, end to end (page-aligned)", bestOf(*trials, func() {
		h, err := am.Import(srcAligned)
		mustf(err, "Import")
		mask, err := h.CompareScalar(am.Gt, int64(0))
		mustf(err, "CompareScalar")
		res, err := h.Filter(mask)
		mustf(err, "Filter")
		back, err := res.Export()
		mustf(err, "Export")
		kept = back.Len()
		back.Release()
		res.Release()
		mask.Release()
		h.Release()
	}), "import, compare, filter, export back to arrow-go")

	add("Filter", "ArrowMetal, end to end (one copy in)", bestOf(*trials, func() {
		h, err := am.Import(srcMisaligned)
		mustf(err, "Import")
		mask, err := h.CompareScalar(am.Gt, int64(0))
		mustf(err, "CompareScalar")
		res, err := h.Filter(mask)
		mustf(err, "Filter")
		back, err := res.Export()
		mustf(err, "Export")
		kept = back.Len()
		back.Release()
		res.Release()
		mask.Release()
		h.Release()
	}), "the same, with an import that has to copy 76 MB")

	// ---- report ----------------------------------------------------------------------------------
	fmt.Printf("ArrowMetal %s on %s\n", ver, dev)
	fmt.Printf("Go %s, arrow-go v18.7.0, %d int64 rows (%d MB), no nulls, best of %d after a warm-up\n",
		runtime.Version(), n, n*8/(1<<20), *trials)
	fmt.Printf("filter predicate x > 0 keeps %d rows (%.1f%%)\n", kept, 100*float64(kept)/float64(n))
	fmt.Printf("values buffer offset past a %d-byte page: PageAlignedAllocator %d, GoAllocator %d, misaligned %d\n\n",
		am.PageSize(), pageOffset(srcAligned), pageOffset(srcGo), pageOffset(srcMisaligned))

	fmt.Printf("| %-6s | %-38s | %10s | %s\n", "op", "method", "best", "notes")
	fmt.Printf("|%s|%s|%s|%s\n", dash(8), dash(40), dash(12), dash(40))
	for _, r := range out {
		fmt.Printf("| %-6s | %-38s | %10s | %s\n", r.op, r.method, fmtDur(r.d), r.note)
	}
	fmt.Printf("\n(sink %d, kept %d)\n", sink, kept)
}

func dash(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = '-'
	}
	return string(b)
}

func fmtDur(d time.Duration) string {
	switch {
	case d < time.Microsecond:
		return fmt.Sprintf("%d ns", d.Nanoseconds())
	case d < time.Millisecond:
		return fmt.Sprintf("%.1f us", float64(d.Nanoseconds())/1e3)
	default:
		return fmt.Sprintf("%.2f ms", float64(d.Nanoseconds())/1e6)
	}
}
