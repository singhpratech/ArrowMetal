package arrowmetal_test

import (
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

func buildWith(mem memory.Allocator, vals []int64) arrow.Array {
	b := array.NewInt64Builder(mem)
	defer b.Release()
	b.AppendValues(vals, nil)
	return b.NewArray()
}

// TestRepeatedImportOfSameArray pins that one arrow.Array can be imported over and over. Each import
// retains the source's buffers through the C Data Interface release callback and each Release drops
// that reference again; getting the pairing wrong frees the buffers underneath the Go array, which
// then exports as an array with no buffers at all.
func TestRepeatedImportOfSameArray(t *testing.T) {
	requireLib(t)
	vals := []int64{1, 2, 3}
	for _, alloc := range []struct {
		name string
		mem  memory.Allocator
	}{
		{"go", memory.NewGoAllocator()},
		{"aligned", am.NewPageAlignedAllocator()},
	} {
		t.Run(alloc.name, func(t *testing.T) {
			src := buildWith(alloc.mem, vals)
			defer src.Release()
			for i := 0; i < 10; i++ {
				h, err := am.Import(src)
				if err != nil {
					t.Fatalf("import %d: %v (buffers=%d len=%d)",
						i, err, len(src.Data().Buffers()), src.Len())
				}
				s, err := h.Sum()
				if err != nil {
					t.Fatalf("sum %d: %v", i, err)
				}
				if s.Int64() != 6 {
					t.Fatalf("sum %d = %v", i, s)
				}
				h.Release()
			}
		})
	}
}

// TestConcurrentHandlesOnOneArray checks that a long-lived handle and a stream of short-lived ones
// over the same arrow.Array coexist.
func TestConcurrentHandlesOnOneArray(t *testing.T) {
	requireLib(t)
	vals := []int64{1, 2, 3}
	src := buildWith(am.NewPageAlignedAllocator(), vals)
	defer src.Release()

	resident, err := am.Import(src)
	if err != nil {
		t.Fatal(err)
	}
	defer resident.Release()
	if _, err := resident.Sum(); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 10; i++ {
		h, err := am.Import(src)
		if err != nil {
			t.Fatalf("import %d: %v (buffers=%d len=%d)", i, err, len(src.Data().Buffers()), src.Len())
		}
		if _, err := h.Sum(); err != nil {
			t.Fatalf("sum %d: %v", i, err)
		}
		h.Release()
	}
}
