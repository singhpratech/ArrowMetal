package arrowmetal_test

import (
	"sort"
	"strings"
	"testing"

	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// planSource builds a two-column table the plan tests scan: `region` (a small int64 key) and
// `amount` (an int64 measure), plus the plain Go answer to the query the tests run.
func planSource(t *testing.T, n int) (*am.Source, []int64, []int64) {
	t.Helper()
	raw := genInt64(n)
	region := make([]int64, n)
	amount := make([]int64, n)
	for i := range raw {
		region[i] = ((raw[i] % 5) + 5) % 5
		amount[i] = ((raw[i] / 3) % 1000)
	}
	ra := buildInt64(t, region, nil)
	t.Cleanup(ra.Release)
	aa := buildInt64(t, amount, nil)
	t.Cleanup(aa.Release)

	src, err := am.NewSource("sales",
		[]string{"region", "amount"},
		[]*am.Array{importArr(t, ra), importArr(t, aa)})
	if err != nil {
		t.Fatalf("NewSource: %v", err)
	}
	t.Cleanup(src.Release)
	return src, region, amount
}

// The example from the C header, minus the limit: sum(amount) by region where amount > 100.
const groupPlan = `{"op":"sort","by":[["region",false]],"input":
  {"op":"group_by","keys":[["region","(col \"region\")"]],
   "aggs":[["sum","total","(col \"amount\")"]],"input":
    {"op":"filter","predicate":"(gt (col \"amount\") (int 100))","input":
      {"op":"scan","source":"sales"}}}}`

func TestPlanGroupBySumAgainstGo(t *testing.T) {
	requireLib(t)
	const n = 1000001
	src, region, amount := planSource(t, n)

	res, err := am.RunPlan(groupPlan, true, src)
	if err != nil {
		t.Fatalf("RunPlan: %v", err)
	}
	defer res.Release()

	if res.NumColumns() != 2 {
		t.Fatalf("NumColumns() = %d, want 2", res.NumColumns())
	}
	names := []string{res.ColumnName(0), res.ColumnName(1)}
	if names[0] != "region" || names[1] != "total" {
		t.Fatalf("column names = %v, want [region total]", names)
	}

	want := map[int64]int64{}
	for i := range region {
		if amount[i] > 100 {
			want[region[i]] += amount[i]
		}
	}
	if int64(len(want)) != res.NumRows() {
		t.Fatalf("NumRows() = %d, want %d", res.NumRows(), len(want))
	}

	c0, err := res.Column(0)
	if err != nil {
		t.Fatal(err)
	}
	defer c0.Release()
	c1, err := res.Column(1)
	if err != nil {
		t.Fatal(err)
	}
	defer c1.Release()
	gotKeys, _ := int64sOf(t, exportArr(t, c0))
	gotSums, _ := int64sOf(t, exportArr(t, c1))

	wantKeys := make([]int64, 0, len(want))
	for k := range want {
		wantKeys = append(wantKeys, k)
	}
	sort.Slice(wantKeys, func(i, j int) bool { return wantKeys[i] < wantKeys[j] })
	if len(gotKeys) != len(wantKeys) {
		t.Fatalf("%d result rows, want %d", len(gotKeys), len(wantKeys))
	}
	for i, k := range gotKeys {
		if k != wantKeys[i] {
			t.Fatalf("row %d: region = %d, want %d", i, k, wantKeys[i])
		}
		if gotSums[i] != want[k] {
			t.Fatalf("region %d: total = %d, want %d", k, gotSums[i], want[k])
		}
	}
}

// TestPlanRecordBatch checks the convenience that hands a whole plan result back to Arrow Go.
func TestPlanRecordBatch(t *testing.T) {
	requireLib(t)
	src, region, amount := planSource(t, 1000)

	res, err := am.RunPlan(groupPlan, true, src)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Release()

	rb, err := res.RecordBatch()
	if err != nil {
		t.Fatal(err)
	}
	defer rb.Release()
	if rb.NumCols() != 2 {
		t.Fatalf("NumCols() = %d, want 2", rb.NumCols())
	}
	if rb.Schema().Field(0).Name != "region" || rb.Schema().Field(1).Name != "total" {
		t.Fatalf("schema = %v", rb.Schema())
	}
	// Nullability is read off each column rather than asserted: neither result column here can
	// contain a null (every group has a key and a sum), so both must come back non-nullable.
	for i := 0; i < int(rb.NumCols()); i++ {
		f := rb.Schema().Field(i)
		if got, want := f.Nullable, rb.Column(i).NullN() != 0; got != want {
			t.Fatalf("field %q Nullable = %v but the column has %d nulls",
				f.Name, got, rb.Column(i).NullN())
		}
		if f.Nullable {
			t.Fatalf("field %q is marked nullable but no group can be null here", f.Name)
		}
	}

	want := map[int64]int64{}
	for i := range region {
		if amount[i] > 100 {
			want[region[i]] += amount[i]
		}
	}
	keys, _ := int64sOf(t, rb.Column(0))
	sums, _ := int64sOf(t, rb.Column(1))
	for i, k := range keys {
		if sums[i] != want[k] {
			t.Fatalf("region %d: total = %d, want %d", k, sums[i], want[k])
		}
	}
}

// TestPlanLimitAndSort runs the header's own example, top 3 regions by total, biggest first.
func TestPlanLimitAndSort(t *testing.T) {
	requireLib(t)
	src, region, amount := planSource(t, 100000)

	const plan = `{"op":"limit","count":3,"input":
	  {"op":"sort","by":[["total",true]],"input":
	    {"op":"group_by","keys":[["region","(col \"region\")"]],
	     "aggs":[["sum","total","(col \"amount\")"]],"input":
	      {"op":"filter","predicate":"(gt (col \"amount\") (int 100))","input":
	        {"op":"scan","source":"sales"}}}}}`

	res, err := am.RunPlan(plan, true, src)
	if err != nil {
		t.Fatalf("RunPlan: %v", err)
	}
	defer res.Release()
	if res.NumRows() != 3 {
		t.Fatalf("NumRows() = %d, want 3", res.NumRows())
	}

	totals := map[int64]int64{}
	for i := range region {
		if amount[i] > 100 {
			totals[region[i]] += amount[i]
		}
	}
	type kv struct{ k, v int64 }
	all := make([]kv, 0, len(totals))
	for k, v := range totals {
		all = append(all, kv{k, v})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].v > all[j].v })

	c1, _ := res.Column(1)
	defer c1.Release()
	gotSums, _ := int64sOf(t, exportArr(t, c1))
	for i := range gotSums {
		if gotSums[i] != all[i].v {
			t.Fatalf("row %d: total = %d, want %d", i, gotSums[i], all[i].v)
		}
	}
}

// TestPlanOptimizedEqualsUnoptimized runs the same plan with and without the optimizer; the answers
// must agree, which is the property the Swift side's optimizer tests assert too.
func TestPlanOptimizedEqualsUnoptimized(t *testing.T) {
	requireLib(t)
	src, _, _ := planSource(t, 100000)

	read := func(optimize bool) ([]int64, []int64) {
		res, err := am.RunPlan(groupPlan, optimize, src)
		if err != nil {
			t.Fatalf("RunPlan(optimize=%v): %v", optimize, err)
		}
		defer res.Release()
		c0, _ := res.Column(0)
		defer c0.Release()
		c1, _ := res.Column(1)
		defer c1.Release()
		k, _ := int64sOf(t, exportArr(t, c0))
		v, _ := int64sOf(t, exportArr(t, c1))
		return k, v
	}
	k1, v1 := read(true)
	k0, v0 := read(false)
	if len(k1) != len(k0) {
		t.Fatalf("optimized gave %d rows, unoptimized %d", len(k1), len(k0))
	}
	for i := range k1 {
		if k1[i] != k0[i] || v1[i] != v0[i] {
			t.Fatalf("row %d: optimized (%d, %d), unoptimized (%d, %d)", i, k1[i], v1[i], k0[i], v0[i])
		}
	}
}

func TestPlanExplain(t *testing.T) {
	requireLib(t)
	src, _, _ := planSource(t, 1000)
	txt, err := am.ExplainPlan(groupPlan, true, src)
	if err != nil {
		t.Fatalf("ExplainPlan: %v", err)
	}
	if txt == "" {
		t.Fatal("ExplainPlan returned empty text")
	}
	t.Logf("plan:\n%s", txt)
}

func TestPlanErrorIsReported(t *testing.T) {
	requireLib(t)
	src, _, _ := planSource(t, 100)

	// A column that does not exist has to fail the engine's type check, with a message.
	const bad = `{"op":"select","exprs":[["x","(col \"nope\")"]],"input":{"op":"scan","source":"sales"}}`
	_, err := am.RunPlan(bad, true, src)
	if err == nil {
		t.Fatal("a plan referencing a missing column returned no error")
	}
	if !strings.Contains(err.Error(), "arrowmetal:") {
		t.Fatalf("error is not from this binding: %v", err)
	}
	t.Logf("bad plan error: %v", err)
}
