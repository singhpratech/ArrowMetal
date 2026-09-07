package arrowmetal_test

import (
	"fmt"
	"testing"

	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// TestGroupBySumAgainstGo compares GroupBy(keys).Sum(values) with a plain Go map over the same rows.
// arrow-go has no hash aggregation to compare against (its compute package registers no aggregate
// functions at all), so the oracle is Go.
//
// Group order is ascending by key for a numeric key column, which is what the ABI documents, so the
// key column that comes back is compared against the sorted distinct keys.
func TestGroupBySumAgainstGo(t *testing.T) {
	requireLib(t)
	for _, n := range []int{1, 1000, 1000001} {
		for _, groups := range []int64{1, 7, 1000} {
			// nullKeys says whether the key column carries nulls as well as the value column. A
			// null key is its own group, so it moves the group count and the key column that comes
			// back, not only the aggregate.
			for _, nullKeys := range []bool{false, true} {
				name := fmt.Sprintf("n=%d/groups=%d/nullKeys=%v", n, groups, nullKeys)
				t.Run(name, func(t *testing.T) {
					raw := genInt64(n)
					keys := make([]int64, n)
					vals := make([]int64, n)
					for i := range keys {
						k := raw[i] % groups
						if k < 0 {
							k += groups
						}
						keys[i] = k
						vals[i] = raw[i] % 10000
					}
					valueValid := nullEvery(n, 13)
					var keyValid []bool
					if nullKeys {
						keyValid = nullEvery(n, 17)
					}
					ka := buildInt64(t, keys, keyValid)
					defer ka.Release()
					va := buildInt64(t, vals, valueValid)
					defer va.Release()

					g, err := am.NewGroupBy(importArr(t, ka))
					if err != nil {
						t.Fatal(err)
					}
					defer g.Release()

					// Plain Go oracle. A null key is labelled nullGroup, which no real key can take
					// because the keys are all in [0, groups).
					const nullGroup = int64(-1)
					groupOf := func(i int) int64 {
						if keyValid != nil && !keyValid[i] {
							return nullGroup
						}
						return keys[i]
					}
					wantSum := map[int64]int64{}
					wantCount := map[int64]int64{}
					wantRows := map[int64]int64{}
					for i := range keys {
						gid := groupOf(i)
						wantRows[gid]++
						if valueValid[i] {
							wantSum[gid] += vals[i]
							wantCount[gid]++
						}
					}
					if got, want := g.NumGroups(), int64(len(wantRows)); got != want {
						t.Fatalf("NumGroups() = %d, want %d", got, want)
					}

					keyOut, err := g.Key(0)
					if err != nil {
						t.Fatal(err)
					}
					defer keyOut.Release()
					rawKeys, keyOutValid := int64sOf(t, exportArr(t, keyOut))
					// Label each result row the way the oracle labels an input row.
					gotKeys := make([]int64, len(rawKeys))
					for i := range rawKeys {
						gotKeys[i] = nullGroup
						if keyOutValid[i] {
							gotKeys[i] = rawKeys[i]
						}
					}
					// Ascending by key, nulls last, every group distinct.
					seenNull := false
					for i := range gotKeys {
						if gotKeys[i] == nullGroup {
							if i != len(gotKeys)-1 {
								t.Fatalf("the null-key group is at row %d of %d, expected last",
									i, len(gotKeys))
							}
							seenNull = true
							continue
						}
						if i > 0 && gotKeys[i] <= gotKeys[i-1] {
							t.Fatalf("group keys are not ascending and distinct at %d: %d then %d",
								i, gotKeys[i-1], gotKeys[i])
						}
					}
					if seenNull != nullKeys {
						t.Fatalf("a null-key group is present = %v, want %v", seenNull, nullKeys)
					}

					sumOut, err := g.Sum(importArr(t, va))
					if err != nil {
						t.Fatal(err)
					}
					defer sumOut.Release()
					gotSums, _ := int64sOf(t, exportArr(t, sumOut))
					if len(gotSums) != len(gotKeys) {
						t.Fatalf("%d sums for %d groups", len(gotSums), len(gotKeys))
					}
					for i, k := range gotKeys {
						if gotSums[i] != wantSum[k] {
							t.Fatalf("group %d (key %d): sum = %d, want %d", i, k, gotSums[i], wantSum[k])
						}
					}

					cntOut, err := g.Count(importArr(t, va))
					if err != nil {
						t.Fatal(err)
					}
					defer cntOut.Release()
					gotCounts, _ := int64sOf(t, exportArr(t, cntOut))
					for i, k := range gotKeys {
						if gotCounts[i] != wantCount[k] {
							t.Fatalf("group %d (key %d): count = %d, want %d", i, k, gotCounts[i], wantCount[k])
						}
					}

					rowsOut, err := g.CountAll()
					if err != nil {
						t.Fatal(err)
					}
					defer rowsOut.Release()
					gotRows, _ := int64sOf(t, exportArr(t, rowsOut))
					for i, k := range gotKeys {
						if gotRows[i] != wantRows[k] {
							t.Fatalf("group %d (key %d): rows = %d, want %d", i, k, gotRows[i], wantRows[k])
						}
					}
				})
			}
		}
	}
}

// TestGroupByNullKey pins the documented rule that a null key is not skipped: it forms its own group.
func TestGroupByNullKey(t *testing.T) {
	requireLib(t)
	keys := []int64{1, 2, 1, 0, 2, 0}
	valid := []bool{true, true, false, true, true, false} // two null keys
	vals := []int64{10, 20, 30, 40, 50, 60}

	ka := buildInt64(t, keys, valid)
	defer ka.Release()
	va := buildInt64(t, vals, nil)
	defer va.Release()

	g, err := am.NewGroupBy(importArr(t, ka))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Release()
	if got, want := g.NumGroups(), int64(4); got != want { // 0, 1, 2 and null
		t.Fatalf("NumGroups() = %d, want %d", got, want)
	}

	keyOut, err := g.Key(0)
	if err != nil {
		t.Fatal(err)
	}
	defer keyOut.Release()
	kv, kvalid := int64sOf(t, exportArr(t, keyOut))

	sumOut, err := g.Sum(importArr(t, va))
	if err != nil {
		t.Fatal(err)
	}
	defer sumOut.Release()
	sv, _ := int64sOf(t, exportArr(t, sumOut))

	got := map[string]int64{}
	for i := range kv {
		if kvalid[i] {
			got[fmt.Sprint(kv[i])] = sv[i]
		} else {
			got["null"] = sv[i]
		}
	}
	want := map[string]int64{"0": 40, "1": 10, "2": 70, "null": 90}
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("group %q: sum = %d, want %d (all groups: %v)", k, got[k], v, got)
		}
	}
}

// TestGroupByIDsLabelEveryRow checks that IDs() labels every row with a valid dense group id and that
// summing the values by that label reproduces the grouped sum.
func TestGroupByIDsLabelEveryRow(t *testing.T) {
	requireLib(t)
	const n = 100000
	raw := genInt64(n)
	keys := make([]int64, n)
	for i := range keys {
		keys[i] = ((raw[i] % 50) + 50) % 50
	}
	ka := buildInt64(t, keys, nil)
	defer ka.Release()

	g, err := am.NewGroupBy(importArr(t, ka))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Release()

	idsOut, err := g.IDs()
	if err != nil {
		t.Fatal(err)
	}
	defer idsOut.Release()
	ids, valid := int32sOf(t, exportArr(t, idsOut))
	if len(ids) != n {
		t.Fatalf("IDs() length %d, want %d", len(ids), n)
	}
	ng := g.NumGroups()
	seen := map[int64]int64{}
	for i, id := range ids {
		if !valid[i] {
			t.Fatalf("row %d has a null group id; the ABI says the ids are never null", i)
		}
		if int64(id) < 0 || int64(id) >= ng {
			t.Fatalf("row %d has group id %d outside [0, %d)", i, id, ng)
		}
		if prev, ok := seen[int64(id)]; ok && prev != keys[i] {
			t.Fatalf("group id %d covers both key %d and key %d", id, prev, keys[i])
		}
		seen[int64(id)] = keys[i]
	}
	if int64(len(seen)) != ng {
		t.Fatalf("%d distinct group ids used, want %d", len(seen), ng)
	}
}

// TestGroupByTwoKeys checks a two-column group-by against a plain Go map.
func TestGroupByTwoKeys(t *testing.T) {
	requireLib(t)
	const n = 100000
	raw := genInt64(n)
	k1 := make([]int64, n)
	k2 := make([]int64, n)
	vals := make([]int64, n)
	for i := range raw {
		k1[i] = ((raw[i] % 5) + 5) % 5
		k2[i] = ((raw[i] / 7 % 11) + 11) % 11
		vals[i] = raw[i] % 100
	}
	a1 := buildInt64(t, k1, nil)
	defer a1.Release()
	a2 := buildInt64(t, k2, nil)
	defer a2.Release()
	va := buildInt64(t, vals, nil)
	defer va.Release()

	g, err := am.NewGroupBy(importArr(t, a1), importArr(t, a2))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Release()

	type key struct{ a, b int64 }
	want := map[key]int64{}
	for i := range k1 {
		want[key{k1[i], k2[i]}] += vals[i]
	}
	if got := g.NumGroups(); got != int64(len(want)) {
		t.Fatalf("NumGroups() = %d, want %d", got, len(want))
	}

	c1, err := g.Key(0)
	if err != nil {
		t.Fatal(err)
	}
	defer c1.Release()
	c2, err := g.Key(1)
	if err != nil {
		t.Fatal(err)
	}
	defer c2.Release()
	sumOut, err := g.Sum(importArr(t, va))
	if err != nil {
		t.Fatal(err)
	}
	defer sumOut.Release()

	g1, _ := int64sOf(t, exportArr(t, c1))
	g2, _ := int64sOf(t, exportArr(t, c2))
	gs, _ := int64sOf(t, exportArr(t, sumOut))
	for i := range g1 {
		k := key{g1[i], g2[i]}
		if gs[i] != want[k] {
			t.Fatalf("group %v: sum = %d, want %d", k, gs[i], want[k])
		}
	}
}

// TestGroupByFloatValues checks the grouped mean of a float64 column against Go.
func TestGroupByFloatValues(t *testing.T) {
	requireLib(t)
	const n = 100000
	raw := genInt64(n)
	keys := make([]int64, n)
	for i := range keys {
		keys[i] = ((raw[i] % 8) + 8) % 8
	}
	vals := genFloat64(n)
	ka := buildInt64(t, keys, nil)
	defer ka.Release()
	va := buildFloat64(t, vals, nil)
	defer va.Release()

	g, err := am.NewGroupBy(importArr(t, ka))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Release()

	sums := map[int64]float64{}
	counts := map[int64]float64{}
	for i := range keys {
		sums[keys[i]] += vals[i]
		counts[keys[i]]++
	}

	keyOut, _ := g.Key(0)
	defer keyOut.Release()
	gotKeys, _ := int64sOf(t, exportArr(t, keyOut))

	meanOut, err := g.Mean(importArr(t, va))
	if err != nil {
		t.Fatal(err)
	}
	defer meanOut.Release()
	gm, _ := float64sOf(t, exportArr(t, meanOut))
	for i, k := range gotKeys {
		want := sums[k] / counts[k]
		if !closeEnough(gm[i], want, 1e-10) {
			t.Fatalf("group %d: mean = %v, want %v", k, gm[i], want)
		}
	}
}

// TestGroupByEmpty checks a zero-row group-by, which must produce zero groups rather than an error.
func TestGroupByEmpty(t *testing.T) {
	requireLib(t)
	ka := buildInt64(t, nil, nil)
	defer ka.Release()
	va := buildInt64(t, nil, nil)
	defer va.Release()

	g, err := am.NewGroupBy(importArr(t, ka))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Release()
	if got := g.NumGroups(); got != 0 {
		t.Fatalf("NumGroups() = %d, want 0", got)
	}
	sumOut, err := g.Sum(importArr(t, va))
	if err != nil {
		t.Fatal(err)
	}
	defer sumOut.Release()
	out := exportArr(t, sumOut)
	if out.Len() != 0 {
		t.Fatalf("grouped sum of no rows has length %d, want 0", out.Len())
	}
}
