// Command ambench times the binding's call overhead on the existing sort and import calls, one row
// per call and size, so two builds of the binding can be compared row by row in one session.
//
//	go run ./cmd/ambench [-reps 30] [-only argsort]
//
// Every row: calls of the same shape run untimed for at least 100 ms first, then the process
// sleeps 500 ms and times one call on its own (first_after_idle_ms: after an idle gap the GPU runs
// a small job slower), then `reps` timed calls give best and median wall time and the process CPU
// time (getrusage user + system) per call. The result handle is released inside the timed call, as
// a program would. Output is CSV on stdout. It uses only calls that exist since 0.3.0.
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

var (
	reps  = flag.Int("reps", 30, "timed calls per row")
	only  = flag.String("only", "", "run only the rows whose name contains this")
	label = flag.String("label", "", "a label printed in the first column (e.g. old / new)")
)

func cpuNow() time.Duration {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "ambench:", err)
		os.Exit(1)
	}
}

func timeRow(name string, n int, fn func()) {
	if *only != "" && !strings.Contains(name, *only) {
		return
	}
	start := time.Now()
	for time.Since(start) < 100*time.Millisecond {
		fn()
	}
	time.Sleep(500 * time.Millisecond)
	t0 := time.Now()
	fn()
	idle := time.Since(t0)

	d := make([]float64, *reps)
	c0 := cpuNow()
	for i := range d {
		t := time.Now()
		fn()
		d[i] = float64(time.Since(t).Nanoseconds()) / 1e6
	}
	cpu := float64((cpuNow() - c0).Nanoseconds()) / 1e6 / float64(*reps)
	sort.Float64s(d)
	fmt.Printf("%s,go,%s,%d,%.4f,%.4f,%.4f,%.4f\n", *label, name, n, idle.Seconds()*1e3, d[0], d[len(d)/2], cpu)
}

func int64Array(mem memory.Allocator, n int, nullEvery int) arrow.Array {
	b := array.NewInt64Builder(mem)
	defer b.Release()
	x := int64(1)
	for i := 0; i < n; i++ {
		x = x*6364136223846793005 + 1442695040888963407
		if nullEvery > 0 && i%nullEvery == 0 {
			b.AppendNull()
		} else {
			b.Append(x >> 40)
		}
	}
	return b.NewArray()
}

func float64Array(mem memory.Allocator, n int, nullEvery int) arrow.Array {
	b := array.NewFloat64Builder(mem)
	defer b.Release()
	x := int64(7)
	for i := 0; i < n; i++ {
		x = x*6364136223846793005 + 1442695040888963407
		if nullEvery > 0 && i%nullEvery == 0 {
			b.AppendNull()
		} else {
			b.Append(float64(x>>40) / 1024)
		}
	}
	return b.NewArray()
}

func main() {
	flag.Parse()
	must(am.Init())
	fmt.Println("label,binding,row,rows,first_after_idle_ms,best_ms,median_ms,cpu_ms_per_call")
	aligned := am.NewPageAlignedAllocator()

	// Import (+ Release): page-aligned buffers, so the import borrows; and Go-heap buffers.
	for _, n := range []int{1000, 1_000_000, 10_000_000} {
		src := int64Array(aligned, n, 0)
		timeRow("import_int64_aligned", n, func() {
			h, err := am.Import(src)
			must(err)
			h.Release()
		})
		src.Release()
	}
	{
		src := float64Array(memory.NewGoAllocator(), 10_000_000, 10)
		timeRow("import_float64_nulls_goheap", 10_000_000, func() {
			h, err := am.Import(src)
			must(err)
			h.Release()
		})
		src.Release()
	}

	// Argsort, Sort and Lexsort on resident columns.
	for _, n := range []int{1000, 1_000_000, 10_000_000} {
		f := float64Array(aligned, n, 10)
		i := int64Array(aligned, n, 0)
		hf, err := am.Import(f)
		must(err)
		hi, err := am.Import(i)
		must(err)
		timeRow("argsort_float64_nulls", n, func() {
			o, err := hf.Argsort(false)
			must(err)
			o.Release()
		})
		timeRow("argsort_int64_desc", n, func() {
			o, err := hi.Argsort(true)
			must(err)
			o.Release()
		})
		timeRow("sort_float64_nulls", n, func() {
			o, err := hf.Sort(false)
			must(err)
			o.Release()
		})
		timeRow("lexsort_int64_float64", n, func() {
			o, err := am.Lexsort([]*am.Array{hi, hf}, []bool{false, true})
			must(err)
			o.Release()
		})
		hf.Release()
		hi.Release()
		f.Release()
		i.Release()
	}
}
