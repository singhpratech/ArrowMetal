package arrowmetal_test

import (
	"context"
	"fmt"
	"math"
	"sort"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// The float values the two orders disagree on: NaN of both signs and several payloads, both zeros,
// both infinities and subnormals.
var awkwardFloats = []float64{
	math.NaN(),
	math.Float64frombits(0xFFF8_0000_0000_0000), // -NaN
	math.Float64frombits(0x7FF0_0000_0000_0001), // signalling NaN payload
	math.Float64frombits(0xFFF8_0000_0000_0042), // -NaN, another payload
	math.Float64frombits(0x7FF8_0000_0000_0007),
	0.0,
	math.Copysign(0, -1),
	math.Inf(1),
	math.Inf(-1),
	math.SmallestNonzeroFloat64,
	-math.SmallestNonzeroFloat64 * 3,
	1.5,
	-1.5,
}

// awkwardFloat64 is mostly the awkward set, the rest small values with many ties; null every
// nullEvery rows (0: none).
func awkwardFloat64(n, nullEvery int, seed int64) ([]float64, []bool) {
	raw := genInt64Seed(n, seed)
	v := make([]float64, n)
	valid := make([]bool, n)
	for i, x := range raw {
		valid[i] = nullEvery == 0 || i%nullEvery != 0
		if x%3 == 0 {
			v[i] = awkwardFloats[(uint64(x)>>3)%uint64(len(awkwardFloats))]
		} else {
			v[i] = float64(x%200) / 4
		}
	}
	return v, valid
}

// totalKey maps a float64 to a uint64 whose unsigned order is IEEE 754 totalOrder.
func totalKey(f float64) uint64 {
	b := math.Float64bits(f)
	if b>>63 != 0 {
		return ^b
	}
	return b | 1<<63
}

// refFloatCmp is the reference comparison of two valid floats: -1, 0, +1 in ascending order.
func refFloatCmp(a, b float64, order am.FloatOrder) int {
	if order == am.FloatTotal {
		ka, kb := totalKey(a), totalKey(b)
		switch {
		case ka < kb:
			return -1
		case ka > kb:
			return 1
		}
		return 0
	}
	// ieee and nan_largest: every NaN is one value, larger than +Inf; -0.0 == +0.0. (ieee's NaN rows
	// are moved next to the nulls by refKeyCmp before this is reached.)
	na, nb := math.IsNaN(a), math.IsNaN(b)
	switch {
	case na && nb:
		return 0
	case na:
		return 1
	case nb:
		return -1
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// refKeyLess orders two rows of one float key under o, the way the header documents it: nulls where
// o.Nulls says in both directions; with FloatIEEE the NaN rows stay next to the nulls in both
// directions; with FloatTotal a descending sort is the exact mirror; with FloatNanLargest every NaN
// is one value above +Inf, so it comes last ascending and first among the values descending.
func refKeyCmp(v []float64, valid []bool, i, j int32, o am.SortOptions) int {
	vi, vj := valid[i], valid[j]
	if !vi || !vj {
		switch {
		case !vi && !vj:
			return 0
		case !vi: // i is null
			if o.Nulls == am.NullsFirst {
				return -1
			}
			return 1
		default:
			if o.Nulls == am.NullsFirst {
				return 1
			}
			return -1
		}
	}
	a, b := v[i], v[j]
	if o.FloatOrder == am.FloatIEEE {
		na, nb := math.IsNaN(a), math.IsNaN(b)
		if na || nb {
			// NaN is null-like: after the values at the end, between nulls and values at the start.
			switch {
			case na && nb:
				return 0
			case na:
				if o.Nulls == am.NullsFirst {
					return -1
				}
				return 1
			default:
				if o.Nulls == am.NullsFirst {
					return 1
				}
				return -1
			}
		}
	}
	c := refFloatCmp(a, b, o.FloatOrder)
	if o.Descending {
		c = -c
	}
	return c
}

func refArgsort(v []float64, valid []bool, o am.SortOptions) []int32 {
	idx := make([]int32, len(v))
	for i := range idx {
		idx[i] = int32(i)
	}
	sort.SliceStable(idx, func(x, y int) bool { return refKeyCmp(v, valid, idx[x], idx[y], o) < 0 })
	return idx
}

func allSortOptions() []am.SortOptions {
	var out []am.SortOptions
	for _, desc := range []bool{false, true} {
		for _, nulls := range []am.NullPlacement{am.NullsLast, am.NullsFirst} {
			for _, fo := range []am.FloatOrder{am.FloatIEEE, am.FloatTotal, am.FloatNanLargest} {
				out = append(out, am.SortOptions{Descending: desc, Nulls: nulls, FloatOrder: fo})
			}
		}
	}
	return out
}

func optName(o am.SortOptions) string {
	return fmt.Sprintf("desc=%v/%v/%v", o.Descending, o.Nulls, o.FloatOrder)
}

func indicesOf(t *testing.T, h *am.Array) []int32 {
	t.Helper()
	defer h.Release()
	gi, _ := int32sOf(t, exportArr(t, h))
	return gi
}

func equalIdx(t *testing.T, what string, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d indices, want %d", what, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: index %d is %d, want %d", what, i, got[i], want[i])
		}
	}
}

// TestArgsortWithFloat64 runs every option combination over Float64 columns full of NaN of both
// signs, ±0.0, ±Inf and subnormals, with nulls, against a stable reference sort index for index.
func TestArgsortWithFloat64(t *testing.T) {
	requireLib(t)
	for _, n := range []int{0, 1, 33, 1025, 100001} {
		v, valid := awkwardFloat64(n, 7, int64(n)+3)
		src := buildFloat64(t, v, valid)
		h := importArr(t, src)
		for _, o := range allSortOptions() {
			t.Run(fmt.Sprintf("n=%d/%s", n, optName(o)), func(t *testing.T) {
				got, err := h.ArgsortWith(o)
				if err != nil {
					t.Fatal(err)
				}
				equalIdx(t, "ArgsortWith", indicesOf(t, got), refArgsort(v, valid, o))
			})
		}
		src.Release()
	}
}

// TestArgsortWithIEEEAgainstArrowGo checks the ieee order against arrow-go's own sort_indices, which
// is Arrow C++'s order: nulls where NullPlacement says, NaN next to them in both directions.
func TestArgsortWithIEEEAgainstArrowGo(t *testing.T) {
	requireLib(t)
	ctx := context.Background()
	const n = 100001
	v, valid := awkwardFloat64(n, 5, 11)
	src := buildFloat64(t, v, valid)
	defer src.Release()
	h := importArr(t, src)
	for _, desc := range []bool{false, true} {
		for _, nulls := range []am.NullPlacement{am.NullsLast, am.NullsFirst} {
			o := am.SortOptions{Descending: desc, Nulls: nulls}
			t.Run(optName(o), func(t *testing.T) {
				key := compute.DefaultSortKey()
				if desc {
					key.Order = compute.SortOrderDescending
				}
				if nulls == am.NullsFirst {
					key.NullPlacement = compute.SortNullsAtStart
				}
				want, err := compute.SortIndicesArray(ctx, src, key)
				if err != nil {
					t.Fatalf("compute.SortIndicesArray: %v", err)
				}
				defer want.Release()
				wu := want.(*array.Uint64)
				wi := make([]int32, wu.Len())
				for i := range wi {
					wi[i] = int32(wu.Value(i))
				}
				got, err := h.ArgsortWith(o)
				if err != nil {
					t.Fatal(err)
				}
				equalIdx(t, "ArgsortWith vs arrow-go", indicesOf(t, got), wi)
			})
		}
	}
}

// TestIntegerNullPlacementAgainstArrowGo covers an Int64 column (float order ignored) in all four
// direction and placement combinations against arrow-go.
func TestIntegerNullPlacementAgainstArrowGo(t *testing.T) {
	requireLib(t)
	ctx := context.Background()
	const n = 100001
	vals := genInt64(n)
	for i := range vals {
		vals[i] %= 500 // ties
	}
	src := buildInt64(t, vals, nullEvery(n, 6))
	defer src.Release()
	h := importArr(t, src)
	for _, o := range allSortOptions() {
		t.Run(optName(o), func(t *testing.T) {
			key := compute.DefaultSortKey()
			if o.Descending {
				key.Order = compute.SortOrderDescending
			}
			if o.Nulls == am.NullsFirst {
				key.NullPlacement = compute.SortNullsAtStart
			}
			want, err := compute.SortIndicesArray(ctx, src, key)
			if err != nil {
				t.Fatalf("compute.SortIndicesArray: %v", err)
			}
			defer want.Release()
			wu := want.(*array.Uint64)
			wi := make([]int32, wu.Len())
			for i := range wi {
				wi[i] = int32(wu.Value(i))
			}
			got, err := h.ArgsortWith(o)
			if err != nil {
				t.Fatal(err)
			}
			equalIdx(t, "ArgsortWith", indicesOf(t, got), wi)

			sorted, err := h.SortWith(o)
			if err != nil {
				t.Fatal(err)
			}
			defer sorted.Release()
			gv, gvalid := int64sOf(t, exportArr(t, sorted))
			wantSorted, err := compute.SortArray(ctx, src, key)
			if err != nil {
				t.Fatal(err)
			}
			defer wantSorted.Release()
			wv, wvalid := int64sOf(t, wantSorted)
			for i := range gv {
				if gvalid[i] != wvalid[i] || (gvalid[i] && gv[i] != wv[i]) {
					t.Fatalf("SortWith element %d: got (%d, %v), want (%d, %v)", i, gv[i], gvalid[i], wv[i], wvalid[i])
				}
			}
		})
	}
}

// TestSortWithFloat64Bits compares SortWith's output bit for bit (NaN payloads and zero signs
// included) with the reference order applied to the input.
func TestSortWithFloat64Bits(t *testing.T) {
	requireLib(t)
	const n = 30011
	v, valid := awkwardFloat64(n, 9, 5)
	src := buildFloat64(t, v, valid)
	defer src.Release()
	h := importArr(t, src)
	for _, o := range allSortOptions() {
		t.Run(optName(o), func(t *testing.T) {
			// With ieee, NaNs of different payloads and ±0.0 tie, so the bits depend on the stable
			// order of ties, which the reference reproduces exactly.
			ref := refArgsort(v, valid, o)
			out, err := h.SortWith(o)
			if err != nil {
				t.Fatal(err)
			}
			defer out.Release()
			gv, gvalid := float64sOf(t, exportArr(t, out))
			for i, r := range ref {
				if gvalid[i] != valid[r] {
					t.Fatalf("row %d: valid=%v, want %v", i, gvalid[i], valid[r])
				}
				if valid[r] && math.Float64bits(gv[i]) != math.Float64bits(v[r]) {
					t.Fatalf("row %d: bits %016x, want %016x", i, math.Float64bits(gv[i]), math.Float64bits(v[r]))
				}
			}
		})
	}
}

// TestSortOptionsFloat32 runs totalOrder on a Float32 column with NaN of both signs and ±0.0.
func TestSortOptionsFloat32(t *testing.T) {
	requireLib(t)
	special := []float32{
		float32(math.NaN()), math.Float32frombits(0xFFC00000), math.Float32frombits(0x7F800001),
		0, float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.Inf(-1)),
		math.SmallestNonzeroFloat32, 2.5, -2.5,
	}
	const n = 10007
	raw := genInt64Seed(n, 77)
	v32 := make([]float32, n)
	v64 := make([]float64, n)
	valid := make([]bool, n)
	for i, x := range raw {
		valid[i] = i%11 != 0
		if x%2 == 0 {
			v32[i] = special[(uint64(x)>>4)%uint64(len(special))]
		} else {
			v32[i] = float32(x%50) / 2
		}
		// Widening is exact for every finite value and the infinities. A NaN is widened by hand: the
		// hardware conversion quiets a signalling NaN, which moves it past the quiet ones in
		// totalOrder, so the payload is carried over as it is.
		if f := v32[i]; f != f {
			w := math.Float32bits(f)
			v64[i] = math.Float64frombits(uint64(w>>31)<<63 | 0x7ff<<52 | uint64(w&0x7fffff)<<29)
		} else {
			v64[i] = float64(f)
		}
	}
	b := array.NewFloat32Builder(mem)
	b.AppendValues(v32, valid)
	src := b.NewArray()
	b.Release()
	defer src.Release()
	h := importArr(t, src)
	for _, o := range allSortOptions() {
		t.Run(optName(o), func(t *testing.T) {
			got, err := h.ArgsortWith(o)
			if err != nil {
				t.Fatal(err)
			}
			equalIdx(t, "ArgsortWith float32", indicesOf(t, got), refArgsort(v64, valid, o))
		})
	}
}

// TestSortOptionsDefaultsMatchPlainCalls pins that the zero SortOptions (and Descending alone) give
// exactly what Argsort, Sort and TopK give.
func TestSortOptionsDefaultsMatchPlainCalls(t *testing.T) {
	requireLib(t)
	const n = 50001
	v, valid := awkwardFloat64(n, 13, 2)
	src := buildFloat64(t, v, valid)
	defer src.Release()
	h := importArr(t, src)
	for _, desc := range []bool{false, true} {
		plain, err := h.Argsort(desc)
		if err != nil {
			t.Fatal(err)
		}
		with, err := h.ArgsortWith(am.SortOptions{Descending: desc})
		if err != nil {
			t.Fatal(err)
		}
		equalIdx(t, fmt.Sprintf("Argsort(%v) vs ArgsortWith", desc), indicesOf(t, with), indicesOf(t, plain))

		tk, err := h.TopK(100, desc)
		if err != nil {
			t.Fatal(err)
		}
		tkw, err := h.TopKWith(100, am.SortOptions{Descending: desc})
		if err != nil {
			t.Fatal(err)
		}
		equalIdx(t, fmt.Sprintf("TopK(%v) vs TopKWith", desc), indicesOf(t, tkw), indicesOf(t, tk))
	}
}

// TestTopKWith checks TopKWith(k, o) against the first k indices of ArgsortWith(o), for every
// option combination and several k, including 0 and k past the length.
func TestTopKWith(t *testing.T) {
	requireLib(t)
	const n = 100001
	v, valid := awkwardFloat64(n, 8, 21)
	src := buildFloat64(t, v, valid)
	defer src.Release()
	h := importArr(t, src)
	for _, o := range allSortOptions() {
		ref := refArgsort(v, valid, o)
		for _, k := range []int64{0, 1, 10, 1000, 30000, n + 5} {
			t.Run(fmt.Sprintf("%s/k=%d", optName(o), k), func(t *testing.T) {
				got, err := h.TopKWith(k, o)
				if err != nil {
					t.Fatal(err)
				}
				want := ref
				if k < int64(len(ref)) {
					want = ref[:k]
				}
				equalIdx(t, "TopKWith", indicesOf(t, got), want)
			})
		}
	}
	if _, err := h.TopKWith(-1, am.SortOptions{}); err == nil {
		t.Fatal("TopKWith(-1) did not fail")
	}
	if _, err := h.ArgsortWith(am.SortOptions{Nulls: 7}); err == nil {
		t.Fatal("an out-of-range NullPlacement did not fail")
	}
	if _, err := h.ArgsortWith(am.SortOptions{FloatOrder: 3}); err == nil {
		t.Fatal("an out-of-range FloatOrder did not fail")
	}
}

// TestLexsortWith sorts an Int32 key with nulls and an awkward Float64 key with nulls, each key with
// its own options, against sort.SliceStable over the same rows.
func TestLexsortWith(t *testing.T) {
	requireLib(t)
	const n = 60013
	raw := genInt64Seed(n, 0xabc)
	k1 := make([]int32, n)
	k1valid := make([]bool, n)
	for i, x := range raw {
		k1[i] = int32(x % 6)
		k1valid[i] = i%10 != 3
	}
	k2, k2valid := awkwardFloat64(n, 9, 0x5eed)
	a1 := buildInt32(t, k1, k1valid)
	defer a1.Release()
	a2 := buildFloat64(t, k2, k2valid)
	defer a2.Release()
	cols := []*am.Array{importArr(t, a1), importArr(t, a2)}

	f1 := make([]float64, n) // the int32 key through the float reference: exact for these values
	for i := range k1 {
		f1[i] = float64(k1[i])
	}
	for _, o1 := range allSortOptions() {
		for _, o2 := range allSortOptions() {
			if o1.FloatOrder != am.FloatIEEE {
				continue // an integer key ignores the float order; one float order is enough
			}
			keys := []am.SortOptions{o1, o2}
			t.Run(optName(o1)+"+"+optName(o2), func(t *testing.T) {
				got, err := am.LexsortWith(cols, keys)
				if err != nil {
					t.Fatal(err)
				}
				want := make([]int32, n)
				for i := range want {
					want[i] = int32(i)
				}
				sort.SliceStable(want, func(x, y int) bool {
					i, j := want[x], want[y]
					if c := refKeyCmp(f1, k1valid, i, j, o1); c != 0 {
						return c < 0
					}
					return refKeyCmp(k2, k2valid, i, j, o2) < 0
				})
				equalIdx(t, "LexsortWith", indicesOf(t, got), want)
			})
		}
	}
	if _, err := am.LexsortWith(cols, []am.SortOptions{{}}); err == nil {
		t.Fatal("LexsortWith with one SortOptions for two columns did not fail")
	}
}

// TestLexsortWithIEEEAgainstArrowGo checks a two-key ieee lexsort against arrow-go's own
// SortIndicesRecordBatch with per-key null placement.
func TestLexsortWithIEEEAgainstArrowGo(t *testing.T) {
	requireLib(t)
	ctx := context.Background()
	const n = 40009
	raw := genInt64Seed(n, 9)
	k1 := make([]int64, n)
	for i, x := range raw {
		k1[i] = x % 4
	}
	k2, k2valid := awkwardFloat64(n, 6, 99)
	a1 := buildInt64(t, k1, nullEvery(n, 5))
	defer a1.Release()
	a2 := buildFloat64(t, k2, k2valid)
	defer a2.Release()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "a", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
		{Name: "b", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil)
	rb := array.NewRecordBatch(schema, []arrow.Array{a1, a2}, n)
	defer rb.Release()
	cols := []*am.Array{importArr(t, a1), importArr(t, a2)}
	keys := []am.SortOptions{{Descending: true, Nulls: am.NullsFirst}, {Nulls: am.NullsFirst}}
	want, err := compute.SortIndicesRecordBatch(ctx, rb, []compute.SortKey{
		{ColumnIndex: 0, Order: compute.SortOrderDescending, NullPlacement: compute.SortNullsAtStart},
		{ColumnIndex: 1, Order: compute.SortOrderAscending, NullPlacement: compute.SortNullsAtStart},
	})
	if err != nil {
		t.Fatalf("compute.SortIndicesRecordBatch: %v", err)
	}
	defer want.Release()
	wu := want.(*array.Uint64)
	wi := make([]int32, wu.Len())
	for i := range wi {
		wi[i] = int32(wu.Value(i))
	}
	got, err := am.LexsortWith(cols, keys)
	if err != nil {
		t.Fatal(err)
	}
	equalIdx(t, "LexsortWith vs arrow-go", indicesOf(t, got), wi)
}

// TestPlanSortKeyOptions runs a plan whose sort key carries {"nulls","float_order"} and compares
// it with ArgsortWith + Take of the same column.
func TestPlanSortKeyOptions(t *testing.T) {
	requireLib(t)
	const n = 20011
	v, valid := awkwardFloat64(n, 4, 31)
	src := buildFloat64(t, v, valid)
	defer src.Release()
	h := importArr(t, src)
	s, err := am.NewSource("t", []string{"x"}, []*am.Array{h})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Release()
	plan := `{"op":"sort","by":[{"column":"x","descending":true,"nulls":"first","float_order":"total"}],` +
		`"input":{"op":"scan","source":"t"}}`
	res, err := am.RunPlan(plan, true, s)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Release()
	col, err := res.Column(0)
	if err != nil {
		t.Fatal(err)
	}
	defer col.Release()
	gv, gvalid := float64sOf(t, exportArr(t, col))
	ref := refArgsort(v, valid, am.SortOptions{Descending: true, Nulls: am.NullsFirst, FloatOrder: am.FloatTotal})
	for i, r := range ref {
		if gvalid[i] != valid[r] || (valid[r] && math.Float64bits(gv[i]) != math.Float64bits(v[r])) {
			t.Fatalf("row %d: got (%v, %v), want (%v, %v)", i, gv[i], gvalid[i], v[r], valid[r])
		}
	}
}

// TestFloatNanLargestHandPicked pins the nan_largest order on a column small enough to read: NaN of
// both signs, both zeros, both infinities and nulls, in both directions and both null placements,
// for ArgsortWith, SortWith, TopKWith and a two-key LexsortWith.
func TestFloatNanLargestHandPicked(t *testing.T) {
	requireLib(t)
	negNaN := math.Float64frombits(0xFFF8_0000_0000_0000)
	//                  0    1           2    3       4    5             6    7       8
	v := []float64{2, math.NaN(), 0, math.Inf(1), -1, math.Inf(-1), 0, negNaN, math.Copysign(0, -1)}
	valid := []bool{true, true, false, true, true, true, true, true, true}
	src := buildFloat64(t, v, valid)
	defer src.Release()
	h := importArr(t, src)
	cases := []struct {
		o    am.SortOptions
		want []int32
	}{
		// -Inf, -1, then 0 / -0.0 tied in input order, 2, +Inf, the NaNs (sign ignored), the null.
		{am.SortOptions{FloatOrder: am.FloatNanLargest}, []int32{5, 4, 6, 8, 0, 3, 1, 7, 2}},
		{am.SortOptions{Nulls: am.NullsFirst, FloatOrder: am.FloatNanLargest}, []int32{2, 5, 4, 6, 8, 0, 3, 1, 7}},
		{am.SortOptions{Descending: true, FloatOrder: am.FloatNanLargest}, []int32{1, 7, 3, 0, 6, 8, 4, 5, 2}},
		{am.SortOptions{Descending: true, Nulls: am.NullsFirst, FloatOrder: am.FloatNanLargest}, []int32{2, 1, 7, 3, 0, 6, 8, 4, 5}},
	}
	for _, c := range cases {
		t.Run(optName(c.o), func(t *testing.T) {
			equalIdx(t, "reference", refArgsort(v, valid, c.o), c.want)
			got, err := h.ArgsortWith(c.o)
			if err != nil {
				t.Fatal(err)
			}
			equalIdx(t, "ArgsortWith", indicesOf(t, got), c.want)
			for _, k := range []int64{0, 1, 3, 9, 20} {
				tk, err := h.TopKWith(k, c.o)
				if err != nil {
					t.Fatal(err)
				}
				want := c.want
				if k < int64(len(want)) {
					want = want[:k]
				}
				equalIdx(t, fmt.Sprintf("TopKWith(%d)", k), indicesOf(t, tk), want)
			}
			sorted, err := h.SortWith(c.o)
			if err != nil {
				t.Fatal(err)
			}
			defer sorted.Release()
			gv, gvalid := float64sOf(t, exportArr(t, sorted))
			for i, r := range c.want {
				if gvalid[i] != valid[r] || (valid[r] && math.Float64bits(gv[i]) != math.Float64bits(v[r])) {
					t.Fatalf("SortWith row %d: got (%v, %v), want row %d (%v, %v)", i, gv[i], gvalid[i], r, v[r], valid[r])
				}
			}
		})
	}
	// Two keys: a group key that splits the rows in two, then the float key descending, nan_largest.
	g := buildInt32(t, []int32{1, 0, 1, 0, 1, 0, 1, 0, 1}, nil)
	defer g.Release()
	cols := []*am.Array{importArr(t, g), h}
	got, err := am.LexsortWith(cols, []am.SortOptions{{}, {Descending: true, FloatOrder: am.FloatNanLargest}})
	if err != nil {
		t.Fatal(err)
	}
	// group 0: rows 1 (NaN), 3 (+Inf), 5 (-Inf), 7 (-NaN); group 1: 0 (2), 2 (null), 4 (-1), 6 (0), 8 (-0.0)
	equalIdx(t, "LexsortWith", indicesOf(t, got), []int32{1, 7, 3, 5, 0, 6, 8, 4, 2})
	if s := am.FloatNanLargest.String(); s != "nan_largest" {
		t.Fatalf("FloatNanLargest.String() = %q", s)
	}
}

// TestPlanSortNanLargest runs "float_order": "nan_largest" through the plan JSON in each key form
// and as the sort-level default, with and without a limit (the top-k path), against the reference.
func TestPlanSortNanLargest(t *testing.T) {
	requireLib(t)
	const n = 20011
	v, valid := awkwardFloat64(n, 4, 37)
	src := buildFloat64(t, v, valid)
	defer src.Release()
	h := importArr(t, src)
	s, err := am.NewSource("t", []string{"x"}, []*am.Array{h})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Release()
	const scan = `{"op":"scan","source":"t"}`
	for _, desc := range []bool{false, true} {
		for _, nulls := range []am.NullPlacement{am.NullsLast, am.NullsFirst} {
			nj := "last"
			if nulls == am.NullsFirst {
				nj = "first"
			}
			sorts := map[string]string{
				"object": fmt.Sprintf(`{"op":"sort","by":[{"column":"x","descending":%v,"nulls":%q,"float_order":"nan_largest"}],"input":%s}`, desc, nj, scan),
				"array":  fmt.Sprintf(`{"op":"sort","by":[["x",%v,{"nulls":%q,"float_order":"nan_largest"}]],"input":%s}`, desc, nj, scan),
				"level":  fmt.Sprintf(`{"op":"sort","by":[["x",%v]],"nulls":%q,"float_order":"nan_largest","input":%s}`, desc, nj, scan),
			}
			ref := refArgsort(v, valid, am.SortOptions{Descending: desc, Nulls: nulls, FloatOrder: am.FloatNanLargest})
			for form, sortPlan := range sorts {
				for _, limit := range []int{-1, 100} {
					plan := sortPlan
					want := ref
					if limit >= 0 {
						plan = fmt.Sprintf(`{"op":"limit","count":%d,"input":%s}`, limit, sortPlan)
						want = ref[:limit]
					}
					t.Run(fmt.Sprintf("desc=%v/%s/%s/limit=%d", desc, nj, form, limit), func(t *testing.T) {
						res, err := am.RunPlan(plan, true, s)
						if err != nil {
							t.Fatal(err)
						}
						defer res.Release()
						col, err := res.Column(0)
						if err != nil {
							t.Fatal(err)
						}
						defer col.Release()
						gv, gvalid := float64sOf(t, exportArr(t, col))
						if len(gv) != len(want) {
							t.Fatalf("%d rows, want %d", len(gv), len(want))
						}
						for i, r := range want {
							if gvalid[i] != valid[r] || (valid[r] && math.Float64bits(gv[i]) != math.Float64bits(v[r])) {
								t.Fatalf("row %d: got (%v, %v), want (%v, %v)", i, gv[i], gvalid[i], v[r], valid[r])
							}
						}
					})
				}
			}
		}
	}
}
