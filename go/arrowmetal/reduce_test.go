package arrowmetal_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/array"
)

// The oracle for the reductions is a plain Go loop over the same data. Arrow Go's compute package
// registers no aggregate functions at all in v18.7.0 (no "sum", "min_max" or "mean" in its function
// registry — see TestArrowGoHasNoAggregates), so there is nothing in arrow-go to compare against.

func goSumInt64(a *array.Int64) (int64, bool) {
	var s int64
	any := false
	for i := 0; i < a.Len(); i++ {
		if a.IsValid(i) {
			s += a.Value(i) // wrapping, as Arrow's sum does
			any = true
		}
	}
	return s, any
}

func goMinMaxInt64(a *array.Int64) (min, max int64, ok bool) {
	for i := 0; i < a.Len(); i++ {
		if !a.IsValid(i) {
			continue
		}
		v := a.Value(i)
		if !ok {
			min, max, ok = v, v, true
			continue
		}
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return
}

func goMeanInt64(a *array.Int64) (float64, bool) {
	var s float64
	n := 0
	for i := 0; i < a.Len(); i++ {
		if a.IsValid(i) {
			s += float64(a.Value(i))
			n++
		}
	}
	if n == 0 {
		return 0, false
	}
	return s / float64(n), true
}

func TestReduceInt64(t *testing.T) {
	requireLib(t)
	for _, n := range testLens {
		for _, nullK := range []int{0, 3} {
			t.Run(fmt.Sprintf("n=%d/nullEvery=%d", n, nullK), func(t *testing.T) {
				src := buildInt64(t, genInt64(n), nullEvery(n, nullK))
				defer src.Release()
				ref := src.(*array.Int64)
				h := importArr(t, src)

				wantSum, wantAny := goSumInt64(ref)
				got, err := h.Sum()
				if err != nil {
					t.Fatal(err)
				}
				if got.Valid != wantAny {
					t.Fatalf("Sum valid = %v, want %v", got.Valid, wantAny)
				}
				if wantAny && got.Int64() != wantSum {
					t.Fatalf("Sum = %d, want %d", got.Int64(), wantSum)
				}

				wantMin, wantMax, ok := goMinMaxInt64(ref)
				gotMin, err := h.Min()
				if err != nil {
					t.Fatal(err)
				}
				gotMax, err := h.Max()
				if err != nil {
					t.Fatal(err)
				}
				if gotMin.Valid != ok || gotMax.Valid != ok {
					t.Fatalf("Min/Max valid = %v/%v, want %v", gotMin.Valid, gotMax.Valid, ok)
				}
				if ok && (gotMin.Int64() != wantMin || gotMax.Int64() != wantMax) {
					t.Fatalf("Min/Max = %d/%d, want %d/%d", gotMin.Int64(), gotMax.Int64(), wantMin, wantMax)
				}

				wantMean, ok := goMeanInt64(ref)
				gotMean, err := h.Mean()
				if err != nil {
					t.Fatal(err)
				}
				if gotMean.Valid != ok {
					t.Fatalf("Mean valid = %v, want %v", gotMean.Valid, ok)
				}
				// The GPU reduction reassociates, so the float64 mean of a million values is compared
				// with a relative tolerance rather than bit for bit.
				if ok && !closeEnough(gotMean.Float64(), wantMean, 1e-12) {
					t.Fatalf("Mean = %v, want %v", gotMean.Float64(), wantMean)
				}
			})
		}
	}
}

func TestReduceFloat64(t *testing.T) {
	requireLib(t)
	for _, n := range testLens {
		for _, nullK := range []int{0, 4} {
			t.Run(fmt.Sprintf("n=%d/nullEvery=%d", n, nullK), func(t *testing.T) {
				src := buildFloat64(t, genFloat64(n), nullEvery(n, nullK))
				defer src.Release()
				ref := src.(*array.Float64)
				h := importArr(t, src)

				var wantSum float64
				wantMin, wantMax := math.Inf(1), math.Inf(-1)
				cnt := 0
				for i := 0; i < ref.Len(); i++ {
					if !ref.IsValid(i) {
						continue
					}
					v := ref.Value(i)
					wantSum += v
					wantMin = math.Min(wantMin, v)
					wantMax = math.Max(wantMax, v)
					cnt++
				}
				any := cnt > 0

				gotSum, err := h.Sum()
				if err != nil {
					t.Fatal(err)
				}
				if gotSum.Valid != any {
					t.Fatalf("Sum valid = %v, want %v", gotSum.Valid, any)
				}
				// A float sum reassociates on the GPU; compare with a relative tolerance.
				if any && !closeEnough(gotSum.Float64(), wantSum, 1e-12) {
					t.Fatalf("Sum = %v, want %v", gotSum.Float64(), wantSum)
				}

				gotMin, err := h.Min()
				if err != nil {
					t.Fatal(err)
				}
				gotMax, err := h.Max()
				if err != nil {
					t.Fatal(err)
				}
				if any {
					// min and max pick an element, so they are exact.
					if gotMin.Float64() != wantMin {
						t.Fatalf("Min = %v, want %v", gotMin.Float64(), wantMin)
					}
					if gotMax.Float64() != wantMax {
						t.Fatalf("Max = %v, want %v", gotMax.Float64(), wantMax)
					}
				} else if gotMin.Valid || gotMax.Valid {
					t.Fatalf("Min/Max of an all-null column reported a value")
				}

				gotMean, err := h.Mean()
				if err != nil {
					t.Fatal(err)
				}
				if any && !closeEnough(gotMean.Float64(), wantSum/float64(cnt), 1e-12) {
					t.Fatalf("Mean = %v, want %v", gotMean.Float64(), wantSum/float64(cnt))
				}
			})
		}
	}
}

// TestReduceNaNAndInf pins the documented float rules: min and max skip NaN, and a column that is
// nothing but NaN reports null.
func TestReduceNaNAndInf(t *testing.T) {
	requireLib(t)
	nan, inf := math.NaN(), math.Inf(1)

	t.Run("NaN is skipped by min/max", func(t *testing.T) {
		src := buildFloat64(t, []float64{nan, 3, nan, -1, 2, inf}, nil)
		defer src.Release()
		h := importArr(t, src)
		mn, err := h.Min()
		if err != nil {
			t.Fatal(err)
		}
		mx, err := h.Max()
		if err != nil {
			t.Fatal(err)
		}
		if mn.Float64() != -1 {
			t.Fatalf("Min = %v, want -1", mn.Float64())
		}
		if !math.IsInf(mx.Float64(), 1) {
			t.Fatalf("Max = %v, want +Inf", mx.Float64())
		}
	})

	t.Run("all NaN is null", func(t *testing.T) {
		src := buildFloat64(t, []float64{nan, nan, nan}, nil)
		defer src.Release()
		h := importArr(t, src)
		mn, err := h.Min()
		if err != nil {
			t.Fatal(err)
		}
		if mn.Valid {
			t.Fatalf("Min of an all-NaN column = %v, want null", mn)
		}
	})
}
