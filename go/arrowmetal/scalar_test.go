package arrowmetal_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// TestCompareScalarRangeCheck pins that a scalar which does not fit the column's element type is an
// error rather than a silent truncation.
//
// The ABI takes a bare `const void*` and reads it as the column's own type, so an unchecked
// int64(1000) against an int8 column is read as -24 and every comparison answers about -24.
func TestCompareScalarRangeCheck(t *testing.T) {
	requireLib(t)

	int8Arr := func() arrow.Array {
		b := array.NewInt8Builder(mem)
		defer b.Release()
		b.AppendValues([]int8{-128, -1, 0, 1, 127}, nil)
		return b.NewArray()
	}
	uint8Arr := func() arrow.Array {
		b := array.NewUint8Builder(mem)
		defer b.Release()
		b.AppendValues([]uint8{0, 1, 255}, nil)
		return b.NewArray()
	}
	f32Arr := func() arrow.Array {
		b := array.NewFloat32Builder(mem)
		defer b.Release()
		b.AppendValues([]float32{-1, 0, 1}, nil)
		return b.NewArray()
	}
	f64Arr := func() arrow.Array {
		b := array.NewFloat64Builder(mem)
		defer b.Release()
		b.AppendValues([]float64{-1, 0, 1}, nil)
		return b.NewArray()
	}

	for _, tc := range []struct {
		name   string
		build  func() arrow.Array
		scalar any
		ok     bool
	}{
		{"int8 vs 1000", int8Arr, int64(1000), false},
		{"int8 vs -1000", int8Arr, int64(-1000), false},
		{"int8 vs 128", int8Arr, 128, false},
		{"int8 vs 127", int8Arr, 127, true},
		{"int8 vs -128", int8Arr, -128, true},
		{"int8 vs uint64 max", int8Arr, uint64(math.MaxUint64), false},
		{"uint8 vs -1", uint8Arr, -1, false},
		{"uint8 vs 256", uint8Arr, 256, false},
		{"uint8 vs 255", uint8Arr, 255, true},
		{"uint8 vs 0", uint8Arr, 0, true},
		{"float32 vs 1e300", f32Arr, 1e300, false},
		{"float32 vs -1e300", f32Arr, -1e300, false},
		{"float32 vs 1e30", f32Arr, 1e30, true},
		{"float32 vs +Inf", f32Arr, math.Inf(1), true},
		// Underflow is as wrong as overflow: float32(5e-46) is 0, so an unchecked Eq would ask
		// about zero and match every zero in the column.
		{"float32 vs 5e-46", f32Arr, 5e-46, false},
		{"float32 vs -5e-46", f32Arr, -5e-46, false},
		{"float32 vs the smallest float32 subnormal", f32Arr, 1.4e-45, true},
		{"float32 vs 0", f32Arr, 0.0, true},
		// Ordinary rounding stays allowed: float32(0.1) != 0.1, but that is what every Arrow
		// implementation compares against.
		{"float32 vs 0.1", f32Arr, 0.1, true},
		// Integers past the mantissa width would land on a neighbouring float.
		{"float32 vs 1<<24", f32Arr, int64(1) << 24, true},
		{"float32 vs 1<<24 + 1", f32Arr, int64(1)<<24 + 1, false},
		{"float64 vs 1<<53", f64Arr, int64(1) << 53, true},
		{"float64 vs 1<<53 + 1", f64Arr, int64(1)<<53 + 1, false},
		{"float64 vs int64 max", f64Arr, int64(math.MaxInt64), false},
		{"float64 vs uint64 max", f64Arr, uint64(math.MaxUint64), false},
		{"float64 vs 5e-46", f64Arr, 5e-46, true}, // float64 names it exactly
		{"int8 vs a string", int8Arr, "nope", false},
		{"int8 vs a bool", int8Arr, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := tc.build()
			defer src.Release()
			h := importArr(t, src)

			out, err := h.CompareScalar(am.Gt, tc.scalar)
			if out != nil {
				defer out.Release()
			}
			if tc.ok {
				if err != nil {
					t.Fatalf("CompareScalar(%v) on %s: %v", tc.scalar, src.DataType(), err)
				}
				return
			}
			if err == nil {
				t.Fatalf("CompareScalar(%v) on %s returned no error; the scalar does not fit",
					tc.scalar, src.DataType())
			}
			// The message has to name the Go value (or its type) and the Arrow type, so the caller
			// can see both halves of the mismatch.
			msg := err.Error()
			if !strings.Contains(msg, src.DataType().Name()) {
				t.Fatalf("error does not name the Arrow type %q: %v", src.DataType().Name(), err)
			}
			t.Logf("%v", err)
		})
	}
}

// TestCompareScalarUnderflowIsNotZero is the reproducer for the float32 underflow case, kept
// separate because the wrong answer was not a crash but a plausible-looking mask.
//
// float32(5e-46) is 0, so before the check `Eq 5e-46` against [0, 1e-30, 1] answered
// [true false false] — it matched the zero, having quietly become a comparison against zero.
func TestCompareScalarUnderflowIsNotZero(t *testing.T) {
	requireLib(t)
	b := array.NewFloat32Builder(mem)
	defer b.Release()
	b.AppendValues([]float32{0, 1e-30, 1}, nil)
	src := b.NewArray()
	defer src.Release()
	h := importArr(t, src)

	out, err := h.CompareScalar(am.Eq, 5e-46)
	if out != nil {
		defer out.Release()
	}
	if err == nil {
		got, _ := boolsOf(t, exportArr(t, out))
		t.Fatalf("CompareScalar(Eq, 5e-46) on a float32 column returned %v; "+
			"the scalar underflows to zero and must be refused", got)
	}
	t.Logf("%v", err)
}

// TestCompareScalarBoundariesAgreeWithGo checks that the values which do fit are compared correctly
// at the edges of each width, against a plain Go loop.
func TestCompareScalarBoundariesAgreeWithGo(t *testing.T) {
	requireLib(t)

	vals := []int8{-128, -127, -1, 0, 1, 126, 127}
	b := array.NewInt8Builder(mem)
	defer b.Release()
	b.AppendValues(vals, nil)
	src := b.NewArray()
	defer src.Release()
	h := importArr(t, src)

	for _, pivot := range []int{-128, -1, 0, 1, 127} {
		t.Run(fmt.Sprintf("gt %d", pivot), func(t *testing.T) {
			out, err := h.CompareScalar(am.Gt, pivot)
			if err != nil {
				t.Fatal(err)
			}
			defer out.Release()
			got, _ := boolsOf(t, exportArr(t, out))
			for i, v := range vals {
				want := int(v) > pivot
				if got[i] != want {
					t.Fatalf("element %d (%d) > %d: got %v, want %v", i, v, pivot, got[i], want)
				}
			}
		})
	}
}
