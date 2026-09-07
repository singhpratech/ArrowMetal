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
