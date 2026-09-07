package arrowmetal_test

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"

	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
	"github.com/singhpratech/ArrowMetal/go/arrowmetal/internal/memstat"
)

const leakChildEnv = "ARROWMETAL_LEAK_CHILD"

// TestNoLeakOverManyRoundTrips runs a full import/compare/filter/export/release cycle a few thousand
// times and requires the process's footprint to come back to where it started.
//
// The loop runs in a fresh child process, and that is the whole point of the arrangement. Measured
// in-process it shares a footprint with the rest of the suite, which by this point has allocated and
// freed several 80 MB arrays in TestAllocatorAlignment; the footprint is still *falling* as the
// allocator returns those pages, so a deliberate 2,200-handle leak measured as −25.7 MB and passed.
// A child that has done nothing else has a flat baseline, and growth means growth.
//
// What this can catch: anything allocated once per iteration and never freed — a missed Release on
// any of the four handles, a Pinner that is never unpinned, a C struct never freed, an ArrowArray
// whose release callback is never called. Verified by commenting out mask.Release(), the smallest of
// the four: the child reports about 35 MB of growth and the test fails.
//
// What it cannot catch: a leak that is bounded (a fixed-size cache, or one allocation per distinct
// array rather than per call), and anything Metal frees lazily on its own schedule.
//
// The measure is TASK_VM_INFO.phys_footprint (see internal/memstat), which is current usage rather
// than a high-water mark; ru_maxrss only ever rises and would hide a leak behind the suite's peak.
func TestNoLeakOverManyRoundTrips(t *testing.T) {
	if os.Getenv(leakChildEnv) == "1" {
		runLeakLoop(t)
		return
	}
	requireLib(t)
	if testing.Short() {
		t.Skip("long")
	}

	exe, err := os.Executable()
	if err != nil {
		t.Skipf("cannot find the test binary: %v", err)
	}
	cmd := exec.Command(exe, "-test.run", "^TestNoLeakOverManyRoundTrips$", "-test.v")
	cmd.Env = append(os.Environ(), leakChildEnv+"=1")
	out, err := cmd.CombinedOutput()
	t.Logf("child:\n%s", strings.TrimSpace(string(out)))
	if err != nil {
		t.Fatalf("the leak loop failed in a child process: %v", err)
	}
	if !strings.Contains(string(out), "LEAK-CHECK OK") {
		t.Fatalf("the child did not report a completed leak check")
	}
}

func runLeakLoop(t *testing.T) {
	if err := am.Init(); err != nil {
		t.Fatalf("ArrowMetal is not loadable: %v", err)
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
	fmt.Printf("phys_footprint %.1f MB -> %.1f MB over %d round trips (grew %.1f MB, budget %d MB)\n",
		float64(before)/(1<<20), float64(after)/(1<<20), iters,
		float64(grew)/(1<<20), budget>>20)
	if grew > budget {
		t.Fatalf("footprint grew by %d bytes over %d round trips, budget %d: something is not being released",
			grew, iters, budget)
	}
	if got := alloc.AllocatedBytes(); got <= 0 {
		t.Fatalf("AllocatedBytes() = %d while the source array is still alive", got)
	}
	fmt.Println("LEAK-CHECK OK")
}

func footprint(t *testing.T) int64 {
	t.Helper()
	v, err := memstat.PhysFootprint()
	if err != nil {
		t.Skipf("cannot read the process footprint: %v", err)
	}
	return int64(v)
}
