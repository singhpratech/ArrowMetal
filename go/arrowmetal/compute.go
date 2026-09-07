package arrowmetal

/*
#cgo CFLAGS: -I${SRCDIR}
#include <stdlib.h>
#include <string.h>
#include "amshim.h"
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// ScalarKind says which field of a Scalar the ABI filled in.
type ScalarKind int

const (
	KindInt64   ScalarKind = 0
	KindUint64  ScalarKind = 1
	KindFloat64 ScalarKind = 2
)

// Scalar is the result of a reduction. Valid is false when there was no value to answer with, which
// is Arrow's null: an empty array, or an array whose every element is null.
type Scalar struct {
	Kind  ScalarKind
	Valid bool

	i int64   // int64 or uint64 (reinterpreted) result
	f float64 // float64 result
}

// Int64 returns the value as an int64 whatever kind it is.
func (s Scalar) Int64() int64 {
	switch s.Kind {
	case KindFloat64:
		return int64(s.f)
	default:
		return s.i
	}
}

// Uint64 returns the value as a uint64 whatever kind it is.
func (s Scalar) Uint64() uint64 {
	switch s.Kind {
	case KindFloat64:
		return uint64(s.f)
	default:
		return uint64(s.i)
	}
}

// Float64 returns the value as a float64 whatever kind it is. An invalid scalar is NaN.
func (s Scalar) Float64() float64 {
	if !s.Valid {
		return math.NaN()
	}
	switch s.Kind {
	case KindFloat64:
		return s.f
	case KindUint64:
		return float64(uint64(s.i))
	default:
		return float64(s.i)
	}
}

func (s Scalar) String() string {
	if !s.Valid {
		return "null"
	}
	switch s.Kind {
	case KindFloat64:
		return fmt.Sprintf("%v", s.f)
	case KindUint64:
		return fmt.Sprintf("%v", uint64(s.i))
	default:
		return fmt.Sprintf("%v", s.i)
	}
}

const (
	opSum  = 0
	opMin  = 1
	opMax  = 2
	opMean = 3
)

func (a *Array) reduce(name string, op C.int) (Scalar, error) {
	h, err := a.ptr()
	if err != nil {
		return Scalar{}, err
	}
	defer runtime.KeepAlive(a)
	var oi C.int64_t
	var of C.double
	var kind, isNull C.int
	if err := call(name, func() C.int {
		return C.amx_reduce(h, op, &oi, &of, &kind, &isNull)
	}); err != nil {
		return Scalar{}, err
	}
	return Scalar{Kind: ScalarKind(kind), Valid: isNull == 0, i: int64(oi), f: float64(of)}, nil
}

// Sum is Arrow's `sum`: nulls are skipped, and an all-null or empty array gives an invalid Scalar.
// Integer columns accumulate in 64 bits and wrap; float32 accumulates in float64, as Arrow does.
func (a *Array) Sum() (Scalar, error) { return a.reduce("am_reduce(sum)", opSum) }

// Min is Arrow's `min`. NaN is skipped; an all-NaN column reports invalid.
func (a *Array) Min() (Scalar, error) { return a.reduce("am_reduce(min)", opMin) }

// Max is Arrow's `max`, with the same NaN rule as Min.
func (a *Array) Max() (Scalar, error) { return a.reduce("am_reduce(max)", opMax) }

// Mean is Arrow's `mean`: the float64 average of the non-null values.
func (a *Array) Mean() (Scalar, error) { return a.reduce("am_reduce(mean)", opMean) }

// CmpOp selects a comparison for CompareScalar and CompareArray.
type CmpOp int

const (
	Eq CmpOp = 0
	Ne CmpOp = 1
	Lt CmpOp = 2
	Le CmpOp = 3
	Gt CmpOp = 4
	Ge CmpOp = 5
)

func (o CmpOp) String() string {
	switch o {
	case Eq:
		return "eq"
	case Ne:
		return "ne"
	case Lt:
		return "lt"
	case Le:
		return "le"
	case Gt:
		return "gt"
	case Ge:
		return "ge"
	}
	return fmt.Sprintf("CmpOp(%d)", int(o))
}

// scalarBytes writes v into freshly malloc'd C memory in the element type named by the Arrow format
// string, which is what the ABI expects for every `const void* scalar` argument. The caller frees it.
func scalarBytes(format string, v any) (unsafe.Pointer, error) {
	fail := func() (unsafe.Pointer, error) {
		return nil, fmt.Errorf("arrowmetal: scalar of Go type %T does not fit an array of Arrow type %q", v, format)
	}
	// A Go int is offered to every integer width; anything else has to match exactly.
	asInt := func() (int64, bool) {
		switch x := v.(type) {
		case int:
			return int64(x), true
		case int8:
			return int64(x), true
		case int16:
			return int64(x), true
		case int32:
			return int64(x), true
		case int64:
			return x, true
		case uint8:
			return int64(x), true
		case uint16:
			return int64(x), true
		case uint32:
			return int64(x), true
		case uint64:
			return int64(x), true
		}
		return 0, false
	}
	asFloat := func() (float64, bool) {
		switch x := v.(type) {
		case float32:
			return float64(x), true
		case float64:
			return x, true
		case int:
			return float64(x), true
		case int64:
			return float64(x), true
		}
		return 0, false
	}
	alloc := func(n C.size_t) unsafe.Pointer { return C.calloc(1, n) }

	switch format {
	case "c", "C": // int8, uint8
		n, ok := asInt()
		if !ok {
			return fail()
		}
		p := alloc(1)
		*(*uint8)(p) = uint8(n)
		return p, nil
	case "s", "S": // int16, uint16
		n, ok := asInt()
		if !ok {
			return fail()
		}
		p := alloc(2)
		*(*uint16)(p) = uint16(n)
		return p, nil
	case "i", "I": // int32, uint32
		n, ok := asInt()
		if !ok {
			return fail()
		}
		p := alloc(4)
		*(*uint32)(p) = uint32(n)
		return p, nil
	case "l", "L": // int64, uint64
		n, ok := asInt()
		if !ok {
			return fail()
		}
		p := alloc(8)
		*(*uint64)(p) = uint64(n)
		return p, nil
	case "f": // float32
		x, ok := asFloat()
		if !ok {
			return fail()
		}
		p := alloc(4)
		*(*float32)(p) = float32(x)
		return p, nil
	case "g": // float64
		x, ok := asFloat()
		if !ok {
			return fail()
		}
		p := alloc(8)
		*(*float64)(p) = x
		return p, nil
	case "b": // boolean: one byte, non-zero is true
		b, ok := v.(bool)
		if !ok {
			return fail()
		}
		p := alloc(1)
		if b {
			*(*uint8)(p) = 1
		}
		return p, nil
	}
	return nil, fmt.Errorf("arrowmetal: scalar comparison is not wrapped for Arrow type %q", format)
}

// CompareScalar compares every element against v and returns a boolean mask. v must be a Go value of
// the array's element type (an untyped integer constant or a plain `int` works for any integer width).
// Nulls in the input stay null in the mask, as they do in Arrow.
func (a *Array) CompareScalar(op CmpOp, v any) (*Array, error) {
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	sp, err := scalarBytes(a.Format(), v)
	if err != nil {
		return nil, err
	}
	defer C.free(sp)
	var out *C.am_array
	if err := call("am_compare_scalar("+op.String()+")", func() C.int {
		return C.amx_compare_scalar(h, C.int(op), sp, &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// CompareArray compares two arrays of the same type and length element by element.
func (a *Array) CompareArray(op CmpOp, b *Array) (*Array, error) {
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	hb, err := b.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(b)
	var out *C.am_array
	if err := call("am_compare_array("+op.String()+")", func() C.int {
		return C.amx_compare_array(h, C.int(op), hb, &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// Filter keeps the elements where mask is true. mask must be a boolean array of the same length; a
// null in the mask drops the element, which is Arrow's "drop" null-selection behaviour.
func (a *Array) Filter(mask *Array) (*Array, error) {
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	hm, err := mask.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(mask)
	var out *C.am_array
	if err := call("am_filter", func() C.int { return C.amx_filter(h, hm, &out) }); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// Take gathers a[i] for each i in indices, which must be an int32 array. A null index gives a null
// element.
func (a *Array) Take(indices *Array) (*Array, error) {
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	hi, err := indices.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(indices)
	var out *C.am_array
	if err := call("am_take", func() C.int { return C.amx_take(h, hi, &out) }); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// Argsort returns the int32 indices that order the array. The sort is stable, nulls come last, and
// NaN sorts after +Inf — in both directions: descending does not mirror nulls and NaN to the front.
func (a *Array) Argsort(descending bool) (*Array, error) {
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	var out *C.am_array
	if err := call("am_argsort", func() C.int { return C.amx_argsort(h, cbool(descending), &out) }); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// Sort returns a sorted copy of the array, with the ordering Argsort documents.
func (a *Array) Sort(descending bool) (*Array, error) {
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	var out *C.am_array
	if err := call("am_sort", func() C.int { return C.amx_sort(h, cbool(descending), &out) }); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// Lexsort returns the int32 indices ordering the rows by each column in turn, the first column being
// the most significant. descending may be nil for all-ascending, otherwise one entry per column.
func Lexsort(columns []*Array, descending []bool) (*Array, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if len(columns) == 0 {
		return nil, fmt.Errorf("arrowmetal: Lexsort needs at least one column")
	}
	if descending != nil && len(descending) != len(columns) {
		return nil, fmt.Errorf("arrowmetal: Lexsort: %d columns but %d descending flags",
			len(columns), len(descending))
	}
	hs, free, err := handleVec(columns)
	if err != nil {
		return nil, err
	}
	defer free()
	defer runtime.KeepAlive(columns)
	var dp *C.int
	if descending != nil {
		d := (*C.int)(C.calloc(C.size_t(len(descending)), C.sizeof_int))
		defer C.free(unsafe.Pointer(d))
		ds := unsafe.Slice(d, len(descending))
		for i, v := range descending {
			ds[i] = cbool(v)
		}
		dp = d
	}
	var out *C.am_array
	if err := call("am_lexsort", func() C.int {
		return C.amx_lexsort(hs, dp, C.int64_t(len(columns)), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

func cbool(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// handleVec copies the handles into a C array, which is what the ABI's `am_array**` arguments want.
func handleVec(arrays []*Array) (**C.am_array, func(), error) {
	n := len(arrays)
	p := (**C.am_array)(C.calloc(C.size_t(n), C.size_t(unsafe.Sizeof((*C.am_array)(nil)))))
	s := unsafe.Slice(p, n)
	for i, a := range arrays {
		h, err := a.ptr()
		if err != nil {
			C.free(unsafe.Pointer(p))
			return nil, nil, fmt.Errorf("arrowmetal: column %d: %w", i, err)
		}
		s[i] = h
	}
	return p, func() { C.free(unsafe.Pointer(p)) }, nil
}
