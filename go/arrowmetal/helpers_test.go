package arrowmetal_test

import (
	"math"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// testLens covers the shapes the project tests everywhere else: empty, one element, a plain middling
// size, and a length that crosses a threadgroup boundary and is not a multiple of anything.
var testLens = []int{0, 1, 1000, 1000001}

// mem is the allocator the tests build their inputs with. It is Arrow Go's default, deliberately:
// the point of the alignment measurement in alignment_test.go is what a Go user gets without doing
// anything special.
var mem = memory.NewGoAllocator()

func requireLib(t *testing.T) {
	t.Helper()
	if err := am.Init(); err != nil {
		t.Fatalf("ArrowMetal is not loadable: %v", err)
	}
}

// nullEvery marks every k-th element null; k <= 0 means no nulls.
func nullEvery(n, k int) []bool {
	if k <= 0 {
		return nil
	}
	v := make([]bool, n)
	for i := range v {
		v[i] = i%k != 0
	}
	return v
}

func genInt64(n int) []int64 {
	v := make([]int64, n)
	// A cheap deterministic spread with both signs and repeats, so sorts and group-bys see ties.
	x := int64(1)
	for i := range v {
		x = (x*6364136223846793005 + 1442695040888963407)
		v[i] = x >> 40 // ~24 bits, signed
	}
	return v
}

func genFloat64(n int) []float64 {
	src := genInt64(n)
	v := make([]float64, n)
	for i, x := range src {
		v[i] = float64(x) / 4096.0
	}
	return v
}

func newInt64BuilderWith(t *testing.T, mem memory.Allocator) *array.Int64Builder {
	t.Helper()
	return array.NewInt64Builder(mem)
}

func buildInt64(t *testing.T, vals []int64, valid []bool) arrow.Array {
	t.Helper()
	b := array.NewInt64Builder(mem)
	defer b.Release()
	b.AppendValues(vals, valid)
	return b.NewArray()
}

func buildFloat64(t *testing.T, vals []float64, valid []bool) arrow.Array {
	t.Helper()
	b := array.NewFloat64Builder(mem)
	defer b.Release()
	b.AppendValues(vals, valid)
	return b.NewArray()
}

func buildInt32(t *testing.T, vals []int32, valid []bool) arrow.Array {
	t.Helper()
	b := array.NewInt32Builder(mem)
	defer b.Release()
	b.AppendValues(vals, valid)
	return b.NewArray()
}

// importArr imports and registers the release with the test.
func importArr(t *testing.T, a arrow.Array) *am.Array {
	t.Helper()
	h, err := am.Import(a)
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	t.Cleanup(h.Release)
	return h
}

// exportArr exports back to Arrow Go and registers the release with the test.
func exportArr(t *testing.T, h *am.Array) arrow.Array {
	t.Helper()
	a, err := h.Export()
	if err != nil {
		t.Fatalf("Export: %v", err)
	}
	t.Cleanup(a.Release)
	return a
}

// int64sOf reads an arrow.Array of int64 into plain Go values plus a validity mask.
func int64sOf(t *testing.T, a arrow.Array) ([]int64, []bool) {
	t.Helper()
	arr, ok := a.(*array.Int64)
	if !ok {
		t.Fatalf("expected int64 array, got %s", a.DataType())
	}
	v := make([]int64, arr.Len())
	valid := make([]bool, arr.Len())
	for i := range v {
		valid[i] = arr.IsValid(i)
		if valid[i] {
			v[i] = arr.Value(i)
		}
	}
	return v, valid
}

func float64sOf(t *testing.T, a arrow.Array) ([]float64, []bool) {
	t.Helper()
	arr, ok := a.(*array.Float64)
	if !ok {
		t.Fatalf("expected float64 array, got %s", a.DataType())
	}
	v := make([]float64, arr.Len())
	valid := make([]bool, arr.Len())
	for i := range v {
		valid[i] = arr.IsValid(i)
		if valid[i] {
			v[i] = arr.Value(i)
		}
	}
	return v, valid
}

func int32sOf(t *testing.T, a arrow.Array) ([]int32, []bool) {
	t.Helper()
	arr, ok := a.(*array.Int32)
	if !ok {
		t.Fatalf("expected int32 array, got %s", a.DataType())
	}
	v := make([]int32, arr.Len())
	valid := make([]bool, arr.Len())
	for i := range v {
		valid[i] = arr.IsValid(i)
		if valid[i] {
			v[i] = arr.Value(i)
		}
	}
	return v, valid
}

func boolsOf(t *testing.T, a arrow.Array) ([]bool, []bool) {
	t.Helper()
	arr, ok := a.(*array.Boolean)
	if !ok {
		t.Fatalf("expected boolean array, got %s", a.DataType())
	}
	v := make([]bool, arr.Len())
	valid := make([]bool, arr.Len())
	for i := range v {
		valid[i] = arr.IsValid(i)
		if valid[i] {
			v[i] = arr.Value(i)
		}
	}
	return v, valid
}

func closeEnough(got, want, relTol float64) bool {
	if math.IsNaN(got) && math.IsNaN(want) {
		return true
	}
	d := math.Abs(got - want)
	if d == 0 {
		return true
	}
	scale := math.Max(math.Abs(got), math.Abs(want))
	return d/scale <= relTol
}
