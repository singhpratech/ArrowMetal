package arrowmetal_test

import (
	"runtime"
	"syscall"
	"testing"

	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// maxRSS is the process's high-water resident set in bytes. On Darwin ru_maxrss is already in bytes
// (it is in kilobytes on Linux, which this binding does not target).
func maxRSS(t *testing.T) int64 {
	t.Helper()
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		t.Skipf("getrusage: %v", err)
	}
	return int64(ru.Maxrss)
}

// TestNoLeakOverManyRoundTrips runs a full import/compare/filter/export/release cycle a few thousand
// times and checks that the process's high-water memory stops growing.
//
// A missed release anywhere in that chain leaks a whole column per iteration, which at this size
// would be gigabytes; the C Data Interface retain/release pairing is the easiest thing in this
// binding to get wrong and the hardest to notice, because a leak is not an error.
func TestNoLeakOverManyRoundTrips(t *testing.T) {
	requireLib(t)
	if testing.Short() {
		t.Skip("long")
	}
	const (
		n     = 200_000 // 1.6 MB of int64 per iteration
		iters = 2000
	)
	alloc := am.NewPageAlignedAllocator()
	vals := genInt64(n)

	// One source array, reused: the loop is about the handles, not about arrow-go's builders.
	b := newInt64BuilderWith(t, alloc)
	b.AppendValues(vals, nullEvery(n, 5))
	src := b.NewArray()
	b.Release()
	defer src.Release()

	runOnce := func() {
		h, err := am.Import(src)
		if err != nil {
			t.Fatal(err)
		}
		mask, err := h.CompareScalar(am.Gt, int64(0))
		if err != nil {
			t.Fatal(err)
		}
		kept, err := h.Filter(mask)
		if err != nil {
			t.Fatal(err)
		}
		out, err := kept.Export()
		if err != nil {
			t.Fatal(err)
		}
		out.Release()
		kept.Release()
		mask.Release()
		h.Release()
	}

	// Warm up past the one-off costs (Metal pipelines, Go heap growth) before taking the baseline.
	for i := 0; i < 200; i++ {
		runOnce()
	}
	runtime.GC()
	before := maxRSS(t)

	for i := 0; i < iters; i++ {
		runOnce()
	}
	runtime.GC()
	after := maxRSS(t)

	grew := after - before
	// Leaking one 1.6 MB column per iteration would be about 3.2 GB; the measured growth is a couple
	// of megabytes. 64 MB of headroom absorbs Metal's own pooling and Go's heap while still catching
	// a leak of more than about 32 KB per round trip.
	const budget = 64 << 20
	t.Logf("max RSS %d -> %d bytes over %d round trips (grew %d)", before, after, iters, grew)
	if grew > budget {
		t.Fatalf("max RSS grew by %d bytes over %d round trips, budget %d: something is not being released",
			grew, iters, budget)
	}
	if got := alloc.AllocatedBytes(); got <= 0 {
		t.Fatalf("AllocatedBytes() = %d while the source array is still alive", got)
	}
}
