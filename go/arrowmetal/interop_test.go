package arrowmetal_test

import (
	"fmt"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/array"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

func TestLoaderReportsWhatItTried(t *testing.T) {
	// The message a Go user gets with no dylib has to name the environment variable and the paths;
	// this pins that contract without unloading the real library (which the shim cannot do).
	if am.LibraryEnv != "ARROWMETAL_LIB" {
		t.Fatalf("LibraryEnv = %q", am.LibraryEnv)
	}
	if am.LibraryName != "libArrowMetalC.dylib" {
		t.Fatalf("LibraryName = %q", am.LibraryName)
	}
}

func TestVersionAndDevice(t *testing.T) {
	requireLib(t)
	v, err := am.Version()
	if err != nil || v == "" {
		t.Fatalf("Version() = %q, %v", v, err)
	}
	d, err := am.DeviceName()
	if err != nil || d == "" {
		t.Fatalf("DeviceName() = %q, %v", d, err)
	}
	if p := am.PageSize(); p <= 0 || p&(p-1) != 0 {
		t.Fatalf("PageSize() = %d, want a positive power of two", p)
	}
	t.Logf("ArrowMetal %s on %s, page size %d, loaded from %s", v, d, am.PageSize(), am.LibraryPath())
}

// TestRoundTripInt64 imports an arrow.Array and exports it back, and checks that every value and
// every null survived, over the empty / one / middling / threadgroup-crossing lengths.
func TestRoundTripInt64(t *testing.T) {
	requireLib(t)
	for _, n := range testLens {
		for _, nullK := range []int{0, 7} {
			t.Run(fmt.Sprintf("n=%d/nullEvery=%d", n, nullK), func(t *testing.T) {
				vals := genInt64(n)
				src := buildInt64(t, vals, nullEvery(n, nullK))
				defer src.Release()

				h := importArr(t, src)
				if h.Len() != int64(n) {
					t.Fatalf("Len() = %d, want %d", h.Len(), n)
				}
				if got, want := h.Format(), "l"; got != want {
					t.Fatalf("Format() = %q, want %q", got, want)
				}
				if got, want := h.NullCount(), int64(src.NullN()); got != want {
					t.Fatalf("NullCount() = %d, want %d", got, want)
				}
				out := exportArr(t, h)
				gotV, gotValid := int64sOf(t, out)
				wantV, wantValid := int64sOf(t, src)
				if len(gotV) != len(wantV) {
					t.Fatalf("length %d, want %d", len(gotV), len(wantV))
				}
				for i := range gotV {
					if gotValid[i] != wantValid[i] || (gotValid[i] && gotV[i] != wantV[i]) {
						t.Fatalf("element %d: got (%d, valid=%v), want (%d, valid=%v)",
							i, gotV[i], gotValid[i], wantV[i], wantValid[i])
					}
				}
			})
		}
	}
}

func TestRoundTripFloat64(t *testing.T) {
	requireLib(t)
	for _, n := range testLens {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			vals := genFloat64(n)
			src := buildFloat64(t, vals, nullEvery(n, 5))
			defer src.Release()

			out := exportArr(t, importArr(t, src))
			gotV, gotValid := float64sOf(t, out)
			wantV, wantValid := float64sOf(t, src)
			for i := range wantV {
				if gotValid[i] != wantValid[i] || (gotValid[i] && gotV[i] != wantV[i]) {
					t.Fatalf("element %d: got (%v, valid=%v), want (%v, valid=%v)",
						i, gotV[i], gotValid[i], wantV[i], wantValid[i])
				}
			}
		})
	}
}

// TestRoundTripSliced covers input with offset != 0, which is a different code path on both sides:
// Arrow Go exports the whole buffer with array.offset set, and ArrowMetal has to honour it.
func TestRoundTripSliced(t *testing.T) {
	requireLib(t)
	const n = 1000001
	vals := genInt64(n)
	full := buildInt64(t, vals, nullEvery(n, 3))
	defer full.Release()

	for _, off := range []int64{1, 7, 31, 32, 33, 63, 64, 1000} {
		t.Run(fmt.Sprintf("offset=%d", off), func(t *testing.T) {
			sl := array.NewSlice(full, off, int64(n))
			defer sl.Release()
			if sl.Data().Offset() == 0 {
				t.Fatalf("expected a non-zero array offset, got 0")
			}
			out := exportArr(t, importArr(t, sl))
			gotV, gotValid := int64sOf(t, out)
			wantV, wantValid := int64sOf(t, sl)
			if len(gotV) != len(wantV) {
				t.Fatalf("length %d, want %d", len(gotV), len(wantV))
			}
			for i := range gotV {
				if gotValid[i] != wantValid[i] || (gotValid[i] && gotV[i] != wantV[i]) {
					t.Fatalf("element %d: got (%d, valid=%v), want (%d, valid=%v)",
						i, gotV[i], gotValid[i], wantV[i], wantValid[i])
				}
			}
		})
	}
}

func TestAllNullAndEmpty(t *testing.T) {
	requireLib(t)
	t.Run("empty", func(t *testing.T) {
		src := buildInt64(t, nil, nil)
		defer src.Release()
		h := importArr(t, src)
		if h.Len() != 0 {
			t.Fatalf("Len() = %d, want 0", h.Len())
		}
		s, err := h.Sum()
		if err != nil {
			t.Fatal(err)
		}
		if s.Valid {
			t.Fatalf("Sum of an empty array = %v, want null", s)
		}
	})
	t.Run("all-null", func(t *testing.T) {
		const n = 1000
		valid := make([]bool, n) // every entry false
		src := buildInt64(t, genInt64(n), valid)
		defer src.Release()
		h := importArr(t, src)
		if h.NullCount() != n {
			t.Fatalf("NullCount() = %d, want %d", h.NullCount(), n)
		}
		for name, fn := range map[string]func() (am.Scalar, error){
			"Sum": h.Sum, "Min": h.Min, "Max": h.Max, "Mean": h.Mean,
		} {
			s, err := fn()
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if s.Valid {
				t.Fatalf("%s of an all-null array = %v, want null", name, s)
			}
		}
	})
}

func TestSliceHandle(t *testing.T) {
	requireLib(t)
	const n = 1000001
	vals := genInt64(n)
	src := buildInt64(t, vals, nil)
	defer src.Release()
	h := importArr(t, src)

	sl, err := h.Slice(5, 1000)
	if err != nil {
		t.Fatal(err)
	}
	defer sl.Release()
	if sl.Len() != 1000 {
		t.Fatalf("sliced Len() = %d, want 1000", sl.Len())
	}
	got, _ := int64sOf(t, exportArr(t, sl))
	for i := range got {
		if got[i] != vals[5+i] {
			t.Fatalf("element %d = %d, want %d", i, got[i], vals[5+i])
		}
	}
}

func TestReleasedArrayIsAnError(t *testing.T) {
	requireLib(t)
	src := buildInt64(t, []int64{1, 2, 3}, nil)
	defer src.Release()
	h, err := am.Import(src)
	if err != nil {
		t.Fatal(err)
	}
	h.Release()
	h.Release() // must be idempotent
	if _, err := h.Sum(); err == nil {
		t.Fatal("Sum on a released array returned no error")
	}
	if _, err := h.Export(); err == nil {
		t.Fatal("Export on a released array returned no error")
	}
}

func TestErrorCarriesLastErrorMessage(t *testing.T) {
	requireLib(t)
	a := importArr(t, buildInt64(t, []int64{1, 2, 3}, nil))
	b := importArr(t, buildInt64(t, []int64{1, 2}, nil))
	_, err := a.CompareArray(am.Eq, b)
	if err == nil {
		t.Fatal("comparing arrays of different lengths returned no error")
	}
	var e *am.Error
	if !asError(err, &e) {
		t.Fatalf("error is %T, want *arrowmetal.Error", err)
	}
	if e.Msg == "" {
		t.Fatalf("error carries no am_last_error() message: %v", err)
	}
	t.Logf("length-mismatch error: %v", err)
}

func asError(err error, target **am.Error) bool {
	e, ok := err.(*am.Error)
	if ok {
		*target = e
	}
	return ok
}
