package arrowmetal_test

import (
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// TestIndexArraysAreUint32 checks the element type of every index array the module returns:
// Argsort, ArgsortWith, TopK, TopKWith, Lexsort and LexsortWith are uint32, and a plan window's
// row_number / rank / dense_rank columns are uint32 too.
func TestIndexArraysAreUint32(t *testing.T) {
	requireLib(t)
	vals := []int64{30, 10, 20, 10, 50, 40}
	src := buildInt64(t, vals, []bool{true, true, true, true, false, true})
	defer src.Release()
	h := importArr(t, src)

	check := func(what string, got *am.Array, err error, want []uint32) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		out := exportArr(t, got)
		got.Release()
		if out.DataType().ID() != arrow.UINT32 {
			t.Fatalf("%s: type %s, want uint32", what, out.DataType())
		}
		u, _ := uint32sOf(t, out)
		if len(u) != len(want) {
			t.Fatalf("%s: %v, want %v", what, u, want)
		}
		for i := range u {
			if u[i] != want[i] {
				t.Fatalf("%s: %v, want %v", what, u, want)
			}
		}
	}
	a, err := h.Argsort(false)
	check("Argsort", a, err, []uint32{1, 3, 2, 0, 5, 4})
	a, err = h.ArgsortWith(am.SortOptions{Descending: true, Nulls: am.NullsFirst})
	check("ArgsortWith", a, err, []uint32{4, 5, 0, 2, 1, 3})
	a, err = h.TopK(2, true)
	check("TopK", a, err, []uint32{5, 0})
	a, err = h.TopKWith(2, am.SortOptions{})
	check("TopKWith", a, err, []uint32{1, 3})
	a, err = am.Lexsort([]*am.Array{h}, []bool{true})
	check("Lexsort", a, err, []uint32{5, 0, 2, 1, 3, 4})
	a, err = am.LexsortWith([]*am.Array{h}, []am.SortOptions{{}})
	check("LexsortWith", a, err, []uint32{1, 3, 2, 0, 5, 4})

	// The engine's window functions number rows as uint32 as well.
	g := buildInt64(t, []int64{1, 1, 1, 2, 2}, nil)
	defer g.Release()
	v := buildInt64(t, []int64{5, 5, 9, 7, 1}, nil)
	defer v.Release()
	srcPlan, err := am.NewSource("t", []string{"g", "v"}, []*am.Array{importArr(t, g), importArr(t, v)})
	if err != nil {
		t.Fatalf("NewSource: %v", err)
	}
	defer srcPlan.Release()
	plan := `{"op":"window","input":{"op":"scan","source":"t"},"specs":[
	  {"name":"rn","fn":"row_number","partition_by":["g"],"order_by":[["v",false]]},
	  {"name":"rk","fn":"rank","partition_by":["g"],"order_by":[["v",false]]},
	  {"name":"dr","fn":"dense_rank","partition_by":["g"],"order_by":[["v",false]]}]}`
	res, err := am.RunPlan(plan, true, srcPlan)
	if err != nil {
		t.Fatalf("RunPlan: %v", err)
	}
	defer res.Release()
	want := map[string][]uint32{"rn": {1, 2, 3, 2, 1}, "rk": {1, 1, 3, 2, 1}, "dr": {1, 1, 2, 2, 1}}
	for i := 0; i < res.NumColumns(); i++ {
		w, ok := want[res.ColumnName(i)]
		if !ok {
			continue
		}
		c, err := res.Column(i)
		check("window "+res.ColumnName(i), c, err, w)
	}
}

// TestTakeIndexTypes takes with int32, int64 and uint32 index arrays (nulls included) and gets the
// same answer from each; an index past the end is an error in every type.
func TestTakeIndexTypes(t *testing.T) {
	requireLib(t)
	src := buildInt64(t, []int64{10, 11, 12, 13, 14}, []bool{true, true, false, true, true})
	defer src.Release()
	h := importArr(t, src)

	idx := []int{4, 0, 2, 3, 1}
	valid := []bool{true, true, true, false, true}
	b32 := array.NewInt32Builder(mem)
	b64 := array.NewInt64Builder(mem)
	bu := array.NewUint32Builder(mem)
	defer b32.Release()
	defer b64.Release()
	defer bu.Release()
	for i, x := range idx {
		if !valid[i] {
			b32.AppendNull()
			b64.AppendNull()
			bu.AppendNull()
			continue
		}
		b32.Append(int32(x))
		b64.Append(int64(x))
		bu.Append(uint32(x))
	}
	wantV := []int64{14, 10, 0, 0, 11}
	wantValid := []bool{true, true, false, false, true}
	for _, ia := range []arrow.Array{b32.NewArray(), b64.NewArray(), bu.NewArray()} {
		name := ia.DataType().String()
		hi := importArr(t, ia)
		ia.Release()
		out, err := h.Take(hi)
		if err != nil {
			t.Fatalf("Take(%s): %v", name, err)
		}
		gv, gvalid := int64sOf(t, exportArr(t, out))
		out.Release()
		for i := range wantV {
			if gvalid[i] != wantValid[i] || (gvalid[i] && gv[i] != wantV[i]) {
				t.Fatalf("Take(%s) element %d: got (%d, %v), want (%d, %v)", name, i, gv[i], gvalid[i], wantV[i], wantValid[i])
			}
		}
	}

	// A uint32 index past the end, including one above 2^31 that an int32 would read as negative.
	for _, bad := range []uint32{5, 1 << 31, 1<<32 - 1} {
		bu.Append(bad)
		ia := bu.NewArray()
		hi := importArr(t, ia)
		ia.Release()
		if out, err := h.Take(hi); err == nil {
			out.Release()
			t.Fatalf("Take with uint32 index %d did not fail", bad)
		}
	}

	// The uint32 indices Argsort returns feed Take directly: the sorted column.
	ord, err := h.Argsort(false)
	if err != nil {
		t.Fatal(err)
	}
	sorted, err := h.Take(ord)
	ord.Release()
	if err != nil {
		t.Fatalf("Take(Argsort): %v", err)
	}
	sv, svalid := int64sOf(t, exportArr(t, sorted))
	sorted.Release()
	wantSorted := []int64{10, 11, 13, 14, 0}
	for i := range wantSorted {
		if svalid[i] != (i < 4) || (svalid[i] && sv[i] != wantSorted[i]) {
			t.Fatalf("Take(Argsort) = %v %v, want %v with the null last", sv, svalid, wantSorted)
		}
	}
}
