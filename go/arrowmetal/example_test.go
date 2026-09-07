package arrowmetal_test

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow/array"
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
