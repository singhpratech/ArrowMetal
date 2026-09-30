// Command amchunks times the chunked import (ImportChunks) against what a program does without it:
// array.Concatenate the chunks, then Import the result. Both are timed end to end, the handle (and
// the concatenated array) released inside the timed call.
//
//	go run ./cmd/amchunks [-rows 10000000,50000000] [-reps 10]
//
// Each chunk is its own allocation from Arrow Go's default allocator, as record batches from a
// reader are. Method as in cmd/ambench: at least 100 ms of untimed calls, a 500 ms idle and one
// timed call on its own, then `reps` timed calls for best, median and CPU time per call. CSV out.
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

var (
	rowsFlag = flag.String("rows", "10000000,50000000", "comma-separated total row counts")
	reps     = flag.Int("reps", 10, "timed calls per row")
	label    = flag.String("label", "", "a label printed in the first column")
)

func cpuNow() time.Duration {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "amchunks:", err)
		os.Exit(1)
	}
}

func timeRow(name string, n, chunkRows int, fn func()) {
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
	fmt.Printf("%s,go,%s,%d,%d,%.3f,%.3f,%.3f,%.3f\n", *label, name, n, chunkRows,
		idle.Seconds()*1e3, d[0], d[len(d)/2], cpu)
}

func makeChunks(dt arrow.DataType, n, chunkRows int) []arrow.Array {
	mem := memory.NewGoAllocator()
	var out []arrow.Array
	x := int64(3)
	for done := 0; done < n; done += chunkRows {
		m := min(chunkRows, n-done)
		switch dt.ID() {
		case arrow.INT64:
			b := array.NewInt64Builder(mem)
			b.Reserve(m)
			for i := 0; i < m; i++ {
				x = x*6364136223846793005 + 1442695040888963407
				b.UnsafeAppend(x >> 40)
			}
			out = append(out, b.NewArray())
			b.Release()
		case arrow.FLOAT64:
			b := array.NewFloat64Builder(mem)
			b.Reserve(m)
			for i := 0; i < m; i++ {
				x = x*6364136223846793005 + 1442695040888963407
				if (done+i)%10 == 0 {
					b.UnsafeAppendBoolToBitmap(false)
				} else {
					b.UnsafeAppend(float64(x>>40) / 1024)
				}
			}
			out = append(out, b.NewArray())
			b.Release()
		}
	}
	return out
}

func main() {
	flag.Parse()
	must(am.Init())
	fmt.Println("label,binding,row,rows,chunk_rows,first_after_idle_ms,best_ms,median_ms,cpu_ms_per_call")
	for _, s := range strings.Split(*rowsFlag, ",") {
		n, err := strconv.Atoi(s)
		must(err)
		for _, dt := range []arrow.DataType{arrow.PrimitiveTypes.Int64, arrow.PrimitiveTypes.Float64} {
			tag := "int64"
			if dt.ID() == arrow.FLOAT64 {
				tag = "float64_nulls"
			}
			for _, chunkRows := range []int{65_536, 1_000_000} {
				chunks := makeChunks(dt, n, chunkRows)
				timeRow("chunked_import_"+tag, n, chunkRows, func() {
					h, err := am.ImportChunks(chunks)
					must(err)
					h.Release()
				})
				timeRow("concatenate_then_import_"+tag, n, chunkRows, func() {
					merged, err := array.Concatenate(chunks, memory.NewGoAllocator())
					must(err)
					h, err := am.Import(merged)
					must(err)
					merged.Release()
					h.Release()
				})
				for _, c := range chunks {
					c.Release()
				}
			}
		}
	}
}
