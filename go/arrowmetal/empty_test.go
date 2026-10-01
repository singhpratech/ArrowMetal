package arrowmetal_test

import (
	"fmt"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// emptyTypes are the element types the empty-input tests cover.
var emptyTypes = []arrow.DataType{
	arrow.PrimitiveTypes.Int8, arrow.PrimitiveTypes.Uint8,
	arrow.PrimitiveTypes.Int16, arrow.PrimitiveTypes.Uint16,
	arrow.PrimitiveTypes.Int32, arrow.PrimitiveTypes.Uint32,
	arrow.PrimitiveTypes.Int64, arrow.PrimitiveTypes.Uint64,
	arrow.PrimitiveTypes.Float32, arrow.PrimitiveTypes.Float64,
	arrow.FixedWidthTypes.Boolean, arrow.BinaryTypes.String,
}

// oneValue builds a one-row array of dt, for a zero-length slice that keeps a non-empty buffer.
func oneValue(dt arrow.DataType) arrow.Array {
	b := array.NewBuilder(mem, dt)
	defer b.Release()
	switch x := b.(type) {
	case *array.BooleanBuilder:
		x.Append(true)
	case *array.StringBuilder:
		x.Append("abc")
	default:
		b.AppendEmptyValue()
	}
	return b.NewArray()
}

// TestImportEmpty imports zero-length arrays of every type four ways (a builder with no rows, which
// has no buffers; MakeArrayOfNull with no rows; a zero-length slice of a one-row array, which keeps
// its buffers, at offset 0 and 1), exports each back and sorts it; ImportChunks takes empty chunks
// next to a non-empty one and with nothing but empty ones.
func TestImportEmpty(t *testing.T) {
	requireLib(t)
	for _, dt := range emptyTypes {
		one := oneValue(dt)
		defer one.Release()
		b := array.NewBuilder(mem, dt)
		built := b.NewArray()
		b.Release()
		defer built.Release()
		nulls := array.MakeArrayOfNull(mem, dt, 0)
		defer nulls.Release()
		s0 := array.NewSlice(one, 0, 0)
		defer s0.Release()
		s1 := array.NewSlice(one, 1, 1)
		defer s1.Release()
		inputs := []struct {
			name string
			src  arrow.Array
		}{{"builder", built}, {"null", nulls}, {"slice0", s0}, {"slice1", s1}}
		for _, in := range inputs {
			t.Run(fmt.Sprintf("%s/%s", dt, in.name), func(t *testing.T) {
				h, err := am.Import(in.src)
				if err != nil {
					t.Fatalf("Import: %v", err)
				}
				defer h.Release()
				if h.Len() != 0 || h.NullCount() != 0 {
					t.Fatalf("Len %d, NullCount %d", h.Len(), h.NullCount())
				}
				out := exportArr(t, h)
				if out.Len() != 0 || !arrow.TypeEqual(out.DataType(), dt) {
					t.Fatalf("Export: %d rows of %s", out.Len(), out.DataType())
				}
				if dt.ID() != arrow.BOOL {
					idx, err := h.ArgsortWith(am.SortOptions{FloatOrder: am.FloatNanLargest})
					if err != nil {
						t.Fatalf("ArgsortWith: %v", err)
					}
					if n := idx.Len(); n != 0 {
						t.Fatalf("ArgsortWith: %d indices", n)
					}
					idx.Release()
				}
			})
		}
		t.Run(fmt.Sprintf("%s/chunks", dt), func(t *testing.T) {
			for _, chunks := range [][]arrow.Array{{built, s1}, {s0, one, built}, {nulls, built}} {
				h, err := am.ImportChunks(chunks)
				if err != nil {
					t.Fatalf("ImportChunks: %v", err)
				}
				want := 0
				for _, c := range chunks {
					want += c.Len()
				}
				if int(h.Len()) != want {
					t.Fatalf("ImportChunks: %d rows, want %d", h.Len(), want)
				}
				out := exportArr(t, h)
				if out.Len() != want {
					t.Fatalf("Export: %d rows, want %d", out.Len(), want)
				}
				h.Release()
			}
		})
	}
}
