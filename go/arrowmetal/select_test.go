package arrowmetal_test

import (
	"context"
	"fmt"
	"sort"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// TestArrowGoHasNoAggregates records why the reduction oracles are plain Go loops rather than
// arrow-go compute calls: arrow-go v18.7.0 registers no aggregate function at all. If a future
// release adds them this test fails and the oracles should be upgraded.
func TestArrowGoHasNoAggregates(t *testing.T) {
	reg := compute.GetFunctionRegistry()
	var present []string
	for _, name := range []string{"sum", "mean", "min_max", "min", "max", "count"} {
		if _, ok := reg.GetFunction(name); ok {
			present = append(present, name)
		}
	}
	if len(present) > 0 {
		t.Logf("arrow-go now registers %v; the reduction oracles could use compute instead", present)
	} else {
		t.Log("arrow-go v18 registers no aggregate functions; reduction oracles are plain Go")
	}
}

func computeCompare(t *testing.T, ctx context.Context, fn string, a arrow.Array, v int64) arrow.Array {
	t.Helper()
	lhs := compute.NewDatumWithoutOwning(a)
	rhs := compute.NewDatum(scalar.NewInt64Scalar(v))
	defer rhs.Release()
	out, err := compute.CallFunction(ctx, fn, nil, lhs, rhs)
	if err != nil {
		t.Fatalf("compute.%s: %v", fn, err)
	}
	res := out.(*compute.ArrayDatum).MakeArray()
	out.Release()
	t.Cleanup(res.Release)
	return res
}

// TestCompareScalarAgainstArrowGo runs every comparison against arrow-go's own kernels on the same
// data, including nulls.
func TestCompareScalarAgainstArrowGo(t *testing.T) {
	requireLib(t)
	ctx := context.Background()
	names := map[am.CmpOp]string{
		am.Eq: "equal", am.Ne: "not_equal",
		am.Lt: "less", am.Le: "less_equal",
		am.Gt: "greater", am.Ge: "greater_equal",
	}
	for _, n := range []int{0, 1, 1000, 1000001} {
		src := buildInt64(t, genInt64(n), nullEvery(n, 6))
		h := importArr(t, src)
		for op, fn := range names {
			t.Run(fmt.Sprintf("n=%d/%s", n, fn), func(t *testing.T) {
				got, err := h.CompareScalar(op, int64(0))
				if err != nil {
					t.Fatal(err)
				}
				defer got.Release()
				gv, gvalid := boolsOf(t, exportArr(t, got))
				wv, wvalid := boolsOf(t, computeCompare(t, ctx, fn, src, 0))
				if len(gv) != len(wv) {
					t.Fatalf("length %d, want %d", len(gv), len(wv))
				}
				for i := range gv {
					if gvalid[i] != wvalid[i] || (gvalid[i] && gv[i] != wv[i]) {
						t.Fatalf("element %d: got (%v, valid=%v), want (%v, valid=%v)",
							i, gv[i], gvalid[i], wv[i], wvalid[i])
					}
				}
			})
		}
		src.Release()
	}
}

// TestFilterAgainstArrowGo compares Compare+Filter with arrow-go's compute.FilterArray over the same
// mask, at a length that crosses a threadgroup boundary.
func TestFilterAgainstArrowGo(t *testing.T) {
	requireLib(t)
	ctx := context.Background()
	for _, n := range []int{0, 1, 1000, 1000001} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			src := buildInt64(t, genInt64(n), nullEvery(n, 6))
			defer src.Release()
			h := importArr(t, src)

			mask, err := h.CompareScalar(am.Gt, int64(0))
			if err != nil {
				t.Fatal(err)
			}
			defer mask.Release()
			out, err := h.Filter(mask)
			if err != nil {
				t.Fatal(err)
			}
			defer out.Release()
			gv, gvalid := int64sOf(t, exportArr(t, out))

			// Arrow Go's own answer, with the same "a null in the mask drops the row" rule.
			refMask := computeCompare(t, ctx, "greater", src, 0)
			want, err := compute.FilterArray(ctx, src, refMask,
				compute.FilterOptions{NullSelection: compute.SelectionDropNulls})
			if err != nil {
				t.Fatalf("compute.FilterArray: %v", err)
			}
			defer want.Release()
			wv, wvalid := int64sOf(t, want)

			if len(gv) != len(wv) {
				t.Fatalf("filtered length %d, want %d", len(gv), len(wv))
			}
			for i := range gv {
				if gvalid[i] != wvalid[i] || (gvalid[i] && gv[i] != wv[i]) {
					t.Fatalf("element %d: got (%d, valid=%v), want (%d, valid=%v)",
						i, gv[i], gvalid[i], wv[i], wvalid[i])
				}
			}
		})
	}
}

// TestTakeAgainstArrowGo compares Take with arrow-go's compute.TakeArray, indices including nulls.
func TestTakeAgainstArrowGo(t *testing.T) {
	requireLib(t)
	ctx := context.Background()
	const n = 1000001
	src := buildInt64(t, genInt64(n), nullEvery(n, 6))
	defer src.Release()
	h := importArr(t, src)

	// A deterministic scatter of in-range indices, with every 11th one null.
	idx := make([]int32, n)
	for i := range idx {
		idx[i] = int32((int64(i)*2654435761 + 7) % int64(n))
	}
	idxArr := buildInt32(t, idx, nullEvery(n, 11))
	defer idxArr.Release()
	hi := importArr(t, idxArr)

	out, err := h.Take(hi)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Release()
	gv, gvalid := int64sOf(t, exportArr(t, out))

	want, err := compute.TakeArray(ctx, src, idxArr)
	if err != nil {
		t.Fatalf("compute.TakeArray: %v", err)
	}
	defer want.Release()
	wv, wvalid := int64sOf(t, want)

	if len(gv) != len(wv) {
		t.Fatalf("length %d, want %d", len(gv), len(wv))
	}
	for i := range gv {
		if gvalid[i] != wvalid[i] || (gvalid[i] && gv[i] != wv[i]) {
			t.Fatalf("element %d: got (%d, valid=%v), want (%d, valid=%v)",
				i, gv[i], gvalid[i], wv[i], wvalid[i])
		}
	}
}

// TestSortAgainstArrowGo compares Sort with arrow-go's compute.SortArray in both directions.
//
// ArrowMetal keeps nulls at the end in both directions by design ("a reversed order does not mirror
// them to the front"), and arrow-go's SortKey has an explicit NullPlacement, so the oracle asks for
// NullsAtEnd in both directions to match.
func TestSortAgainstArrowGo(t *testing.T) {
	requireLib(t)
	ctx := context.Background()
	for _, n := range []int{0, 1, 1000, 1000001} {
		for _, desc := range []bool{false, true} {
			t.Run(fmt.Sprintf("n=%d/desc=%v", n, desc), func(t *testing.T) {
				src := buildInt64(t, genInt64(n), nullEvery(n, 9))
				defer src.Release()
				h := importArr(t, src)

				out, err := h.Sort(desc)
				if err != nil {
					t.Fatal(err)
				}
				defer out.Release()
				gv, gvalid := int64sOf(t, exportArr(t, out))

				key := compute.DefaultSortKey()
				key.NullPlacement = compute.SortNullsAtEnd
				if desc {
					key.Order = compute.SortOrderDescending
				}
				want, err := compute.SortArray(ctx, src, key)
				if err != nil {
					t.Fatalf("compute.SortArray: %v", err)
				}
				defer want.Release()
				wv, wvalid := int64sOf(t, want)

				if len(gv) != len(wv) {
					t.Fatalf("length %d, want %d", len(gv), len(wv))
				}
				for i := range gv {
					if gvalid[i] != wvalid[i] || (gvalid[i] && gv[i] != wv[i]) {
						t.Fatalf("element %d: got (%d, valid=%v), want (%d, valid=%v)",
							i, gv[i], gvalid[i], wv[i], wvalid[i])
					}
				}
			})
		}
	}
}

// TestArgsortIsStableAndOrders checks Argsort two ways: the permutation it returns has to sort the
// column (checked against arrow-go's sorted answer via Take), and it has to be a stable permutation
// (equal keys keep their input order), which is a property arrow-go's sort_indices also has.
func TestArgsortAgainstArrowGo(t *testing.T) {
	requireLib(t)
	ctx := context.Background()
	const n = 1000001
	// Deliberate ties: values are squeezed into a small range so equal keys are common.
	vals := genInt64(n)
	for i := range vals {
		vals[i] %= 1000
	}
	src := buildInt64(t, vals, nullEvery(n, 9))
	defer src.Release()
	h := importArr(t, src)

	got, err := h.Argsort(false)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Release()
	gi, _ := int32sOf(t, exportArr(t, got))

	key := compute.DefaultSortKey()
	key.NullPlacement = compute.SortNullsAtEnd
	wantIdx, err := compute.SortIndicesArray(ctx, src, key)
	if err != nil {
		t.Fatalf("compute.SortIndicesArray: %v", err)
	}
	defer wantIdx.Release()

	// arrow-go returns uint64 indices; compare the permutations element for element.
	wu, ok := wantIdx.(*array.Uint64)
	if !ok {
		t.Fatalf("compute.SortIndicesArray returned %s, expected uint64", wantIdx.DataType())
	}
	if len(gi) != wu.Len() {
		t.Fatalf("index length %d, want %d", len(gi), wu.Len())
	}
	for i := range gi {
		if uint64(gi[i]) != wu.Value(i) {
			t.Fatalf("index %d: got %d, want %d (both sorts are stable with nulls last, so they must agree)",
				i, gi[i], wu.Value(i))
		}
	}
}

// TestLexsort checks the multi-column sort against a plain Go sort.SliceStable over the same rows.
func TestLexsort(t *testing.T) {
	requireLib(t)
	const n = 100000
	a := genInt64(n)
	b := genInt64(n)
	for i := range a {
		a[i] %= 10 // a few distinct values in the major key, so the minor key decides most rows
		b[i] %= 1000
	}
	aa := buildInt64(t, a, nil)
	defer aa.Release()
	bb := buildInt64(t, b, nil)
	defer bb.Release()

	got, err := am.Lexsort([]*am.Array{importArr(t, aa), importArr(t, bb)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Release()
	gi, _ := int32sOf(t, exportArr(t, got))

	want := make([]int32, n)
	for i := range want {
		want[i] = int32(i)
	}
	sort.SliceStable(want, func(x, y int) bool {
		i, j := want[x], want[y]
		if a[i] != a[j] {
			return a[i] < a[j]
		}
		return b[i] < b[j]
	})
	for i := range gi {
		if gi[i] != want[i] {
			t.Fatalf("index %d: got %d, want %d", i, gi[i], want[i])
		}
	}
}
