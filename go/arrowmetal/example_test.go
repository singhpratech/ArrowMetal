package arrowmetal_test

import (
	"fmt"
	"math"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// Example is the fifteen lines in docs/GO.md, compiled and run so that the documentation cannot
// drift from the API.
func Example() {
	// PageAlignedAllocator is a memory.Allocator whose buffers ArrowMetal can borrow rather than copy.
	b := array.NewInt64Builder(am.NewPageAlignedAllocator())
	defer b.Release()
	b.AppendValues([]int64{5, 3, 9, 1}, []bool{true, true, false, true}) // 9 is null

	src := b.NewArray()
	defer src.Release()

	gpu, err := am.Import(src) // arrow.Array -> GPU
	if err != nil {
		panic(err)
	}
	defer gpu.Release()

	sum, _ := gpu.Sum() // nulls skipped, like Arrow
	mask, _ := gpu.CompareScalar(am.Gt, int64(2))
	kept, _ := gpu.Filter(mask)
	out, _ := kept.Export() // back to arrow-go, no copy
	defer out.Release()

	fmt.Println(sum, out)
	// Output: 9 [5 3]
}

// ExampleArray_ArgsortWith is the sort-options example in docs/GO.md.
func ExampleArray_ArgsortWith() {
	b := array.NewFloat64Builder(memory.NewGoAllocator())
	defer b.Release()
	b.AppendValues([]float64{2, 0, math.NaN(), math.Copysign(0, -1), 7}, []bool{true, false, true, true, true})
	src := b.NewArray() // [2, null, NaN, -0, 7]
	defer src.Release()

	gpu, err := am.Import(src)
	if err != nil {
		panic(err)
	}
	defer gpu.Release()

	plain, _ := gpu.Argsort(true) // nulls and NaN stay last
	opts := am.SortOptions{Descending: true, Nulls: am.NullsFirst, FloatOrder: am.FloatTotal}
	total, _ := gpu.ArgsortWith(opts) // null first, then +NaN > 7 > 2 > -0
	top, _ := gpu.TopKWith(2, opts)
	for _, h := range []*am.Array{plain, total, top} {
		out, _ := h.Export()
		fmt.Println(out)
		out.Release()
		h.Release()
	}
	// Output:
	// [4 0 3 2 1]
	// [1 2 4 0 3]
	// [1 2]
}

// ExampleImportChunks is the chunked-import example in docs/GO.md.
func ExampleImportChunks() {
	mem := memory.NewGoAllocator()
	b := array.NewInt64Builder(mem)
	defer b.Release()
	b.AppendValues([]int64{1, 2}, nil)
	c1 := b.NewArray()
	b.AppendValues([]int64{3, 0, 5}, []bool{true, false, true})
	c2 := b.NewArray()
	chunked := arrow.NewChunked(arrow.PrimitiveTypes.Int64, []arrow.Array{c1, c2})
	c1.Release()
	c2.Release()
	defer chunked.Release()

	gpu, err := am.ImportChunked(chunked) // one array of 5 rows, no concatenation first
	if err != nil {
		panic(err)
	}
	defer gpu.Release()
	sum, _ := gpu.Sum()
	fmt.Println(gpu.Len(), gpu.NullCount(), sum)
	// Output: 5 1 11
}
