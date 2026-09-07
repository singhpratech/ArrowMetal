package arrowmetal_test

import (
	"runtime"
	"testing"

	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
	"github.com/singhpratech/ArrowMetal/go/arrowmetal/internal/memstat"
)

func footprint(t *testing.T) int64 {
	t.Helper()
	v, err := memstat.PhysFootprint()
	if err != nil {
		t.Skipf("cannot read the process footprint: %v", err)
	}
	return int64(v)
}

// TestNoLeakOverManyRoundTrips runs a full import/compare/filter/export/release cycle a few thousand
// times and requires the process's footprint to come back to where it started.
//
// What this can catch: anything that is allocated once per iteration and never freed — a missed
// Release on any of the four handles, a Pinner that is never unpinned, a C struct that is never
// freed, an ArrowArray whose release callback is never called. The budget below is small enough that
// leaking even the smallest of the four handles (the boolean mask, ~25 KB of payload but a
// page-rounded Metal buffer and an ArrowMetal box behind it) fails the test.
//
// What it cannot catch: a leak that is bounded — a fixed-size cache, or one allocation per distinct
// array rather than per call — and anything freed lazily by Metal's own pooling.
//
// The measure is TASK_VM_INFO.phys_footprint (see internal/memstat), which is current usage rather
// than a high-water mark. ru_maxrss is unusable here: it only rises, and by the time this test runs
// the rest of the suite has already pushed the peak into the hundreds of megabytes, so a leak has to
// exceed that peak before it can even be seen.
func TestNoLeakOverManyRoundTrips(t *testing.T) {
	requireLib(t)
	if testing.Short() {
		t.Skip("long")
	}
	const (
		n     = 200_000 // 1.6 MB of int64 values plus a 25 KB validity bitmap per iteration
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
	before := footprint(t)

	for i := 0; i < iters; i++ {
		runOnce()
	}
	runtime.GC()
	after := footprint(t)

	grew := after - before
	// Sized against the smallest thing a single missed Release would leak. The mask is a boolean
	// array over 200,000 rows: 25 KB of bits, rounded up to a 16 KiB page by Metal, so at least
	// 32 KB per iteration, or 64 MB over the run. 16 MB is a quarter of that and still leaves room
	// for Metal's pooling and Go's heap, which together move by a couple of megabytes.
	const budget = 16 << 20
	t.Logf("phys_footprint %.1f MB -> %.1f MB over %d round trips (grew %.1f MB, budget %d MB)",
		float64(before)/(1<<20), float64(after)/(1<<20), iters,
		float64(grew)/(1<<20), budget>>20)
	if grew > budget {
		t.Fatalf("footprint grew by %d bytes over %d round trips, budget %d: something is not being released",
			grew, iters, budget)
	}
	if got := alloc.AllocatedBytes(); got <= 0 {
		t.Fatalf("AllocatedBytes() = %d while the source array is still alive", got)
	}
}
