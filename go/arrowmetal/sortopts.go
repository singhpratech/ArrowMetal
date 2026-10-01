package arrowmetal

/*
#cgo CFLAGS: -I${SRCDIR}
#include <stdlib.h>
#include "amshim.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// NullPlacement says where the null rows of a sort key go. It holds in both directions: a descending
// sort does not move them.
type NullPlacement int

const (
	// NullsLast puts the null rows after every value (Arrow's default, and what Argsort and Sort do).
	NullsLast NullPlacement = 0
	// NullsFirst puts the null rows before every value.
	NullsFirst NullPlacement = 1
)

func (p NullPlacement) String() string {
	switch p {
	case NullsLast:
		return "nulls last"
	case NullsFirst:
		return "nulls first"
	}
	return fmt.Sprintf("NullPlacement(%d)", int(p))
}

// FloatOrder says how a float key orders -0.0, +0.0 and NaN. Integer, string and temporal keys
// ignore it.
type FloatOrder int

const (
	// FloatIEEE is the order of Argsort and Sort, which is Arrow C++'s: -0.0 ties +0.0, every NaN is
	// one value, and the NaN rows sit next to the nulls in both directions (after the values with
	// NullsLast, between the nulls and the values with NullsFirst).
	FloatIEEE FloatOrder = 0
	// FloatTotal is IEEE 754 totalOrder, as arrow-rs and Rust's total_cmp define it:
	// -NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN, NaNs by payload, only identical bits tie,
	// and a descending sort is the exact mirror (+NaN first).
	FloatTotal FloatOrder = 1
	// FloatNanLargest is Polars' and NumPy's order: IEEE comparison with -0.0 and +0.0 equal, and every
	// NaN one value greater than every number, +Inf included, in both directions (last ascending,
	// first among the values descending). The null placement is independent of it.
	FloatNanLargest FloatOrder = 2
)

func (o FloatOrder) String() string {
	switch o {
	case FloatIEEE:
		return "ieee"
	case FloatTotal:
		return "total"
	case FloatNanLargest:
		return "nan_largest"
	}
	return fmt.Sprintf("FloatOrder(%d)", int(o))
}

// SortOptions are the options of one sort key. The zero value is an ascending sort with the order
// Argsort and Sort use, so ArgsortWith(SortOptions{}) gives the indices Argsort(false) gives.
//
// Neither option adds a pass on the GPU: the null placement is where the partition that takes the
// null rows out of the radix sort puts them, and FloatTotal and FloatNanLargest are the key map in
// front of the radix passes.
type SortOptions struct {
	Descending bool
	Nulls      NullPlacement
	FloatOrder FloatOrder
}

func (o SortOptions) check(op string) error {
	if o.Nulls != NullsLast && o.Nulls != NullsFirst {
		return fmt.Errorf("arrowmetal: %s: %v is not NullsLast or NullsFirst", op, o.Nulls)
	}
	if o.FloatOrder != FloatIEEE && o.FloatOrder != FloatTotal && o.FloatOrder != FloatNanLargest {
		return fmt.Errorf("arrowmetal: %s: %v is not FloatIEEE, FloatTotal or FloatNanLargest", op, o.FloatOrder)
	}
	return nil
}

// requireSortOptions reports a dylib that predates the option-taking sorts, which then still loads
// for everything else.
func requireSortOptions(op string) error {
	if err := Init(); err != nil {
		return err
	}
	if C.amshim_has_sort_options() == 0 {
		return fmt.Errorf("arrowmetal: %s: the loaded %s (%s) has no am_argsort_ex2 / am_sort_ex2 / "+
			"am_top_k_ex / am_lexsort_ex2; it predates sort options, so rebuild it", op, LibraryName, loadPath)
	}
	return nil
}

// ArgsortWith returns the int32 indices that order the array under o. The sort is stable.
func (a *Array) ArgsortWith(o SortOptions) (*Array, error) {
	if err := requireSortOptions("ArgsortWith"); err != nil {
		return nil, err
	}
	if err := o.check("ArgsortWith"); err != nil {
		return nil, err
	}
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	var out *C.am_array
	if err := call("am_argsort_ex2", func() C.int {
		return C.amx_argsort_ex2(h, cbool(o.Descending), C.int(o.Nulls), C.int(o.FloatOrder), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// SortWith returns a sorted copy of the array, ordered as ArgsortWith orders it.
func (a *Array) SortWith(o SortOptions) (*Array, error) {
	if err := requireSortOptions("SortWith"); err != nil {
		return nil, err
	}
	if err := o.check("SortWith"); err != nil {
		return nil, err
	}
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	var out *C.am_array
	if err := call("am_sort_ex2", func() C.int {
		return C.amx_sort_ex2(h, cbool(o.Descending), C.int(o.Nulls), C.int(o.FloatOrder), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// TopK returns the int32 indices of the k largest (or, with largest false, the k smallest) values,
// in sorted order: the first k indices Argsort(largest) gives.
func (a *Array) TopK(k int64, largest bool) (*Array, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if k < 0 {
		return nil, fmt.Errorf("arrowmetal: TopK: k is %d; it must be >= 0", k)
	}
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	var out *C.am_array
	if err := call("am_top_k", func() C.int {
		return C.amx_top_k(h, C.int64_t(k), cbool(largest), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// TopKWith returns the first k indices ArgsortWith(o) gives, found by GPU selection rather than a
// whole sort; o.Descending asks for the largest.
func (a *Array) TopKWith(k int64, o SortOptions) (*Array, error) {
	if err := requireSortOptions("TopKWith"); err != nil {
		return nil, err
	}
	if err := o.check("TopKWith"); err != nil {
		return nil, err
	}
	if k < 0 {
		return nil, fmt.Errorf("arrowmetal: TopKWith: k is %d; it must be >= 0", k)
	}
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	var out *C.am_array
	if err := call("am_top_k_ex", func() C.int {
		return C.amx_top_k_ex(h, C.int64_t(k), cbool(o.Descending), C.int(o.Nulls), C.int(o.FloatOrder), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// LexsortWith returns the int32 indices ordering the rows by each column in turn, the first column
// being the most significant, each under its own options. keys holds one SortOptions per column.
func LexsortWith(columns []*Array, keys []SortOptions) (*Array, error) {
	if err := requireSortOptions("LexsortWith"); err != nil {
		return nil, err
	}
	if len(columns) == 0 {
		return nil, fmt.Errorf("arrowmetal: LexsortWith needs at least one column")
	}
	if len(keys) != len(columns) {
		return nil, fmt.Errorf("arrowmetal: LexsortWith: %d columns but %d SortOptions",
			len(columns), len(keys))
	}
	for i, o := range keys {
		if err := o.check(fmt.Sprintf("LexsortWith key %d", i)); err != nil {
			return nil, err
		}
	}
	hs, free, err := handleVec(columns)
	if err != nil {
		return nil, err
	}
	defer free()
	defer runtime.KeepAlive(columns)

	n := len(columns)
	// One C block for the three per-key arrays: descending, null placement, float order.
	opts := (*C.int)(C.calloc(C.size_t(3*n), C.sizeof_int))
	defer C.free(unsafe.Pointer(opts))
	s := unsafe.Slice(opts, 3*n)
	for i, o := range keys {
		s[i] = cbool(o.Descending)
		s[n+i] = C.int(o.Nulls)
		s[2*n+i] = C.int(o.FloatOrder)
	}
	var out *C.am_array
	if err := call("am_lexsort_ex2", func() C.int {
		return C.amx_lexsort_ex2(hs, &s[0], &s[n], &s[2*n], C.int64_t(n), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}
