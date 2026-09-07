package arrowmetal

/*
#cgo CFLAGS: -I${SRCDIR}
#include <stdlib.h>
#include "amshim.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
)

// GroupAgg names one grouped aggregate. The numbering is the ABI's am_group_agg_ex table; only the
// aggregates this binding tests are named here.
type GroupAgg int

const (
	AggSum      GroupAgg = 0  // hash_sum: int64/uint64/double, integers wrap in 64 bits
	AggCountAll GroupAgg = 1  // hash_count_all: rows per group, nulls included
	AggCount    GroupAgg = 2  // hash_count: non-null values per group
	AggMean     GroupAgg = 3  // hash_mean: double
	AggMin      GroupAgg = 4  // hash_min: the values' type
	AggMax      GroupAgg = 5  // hash_max: the values' type
	AggProduct  GroupAgg = 16 // hash_product
)

// GroupBy is a set of key columns mapped to dense group ids on the GPU.
//
// Group order is deterministic but is not Arrow's first-seen order: it is ascending by key for
// numeric, boolean, temporal and decimal columns (nulls last), first-seen for utf8 and binary, and
// lexicographic in column order for several columns. Label the rows with Key.
type GroupBy struct {
	gb    *C.am_groupby
	nkeys int
}

// NewGroupBy maps one or more key columns to dense group ids. A null key is not skipped; it forms its
// own group, as Arrow's hash aggregation does.
func NewGroupBy(keys ...*Array) (*GroupBy, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if len(keys) == 0 {
		return nil, errors.New("arrowmetal: NewGroupBy needs at least one key column")
	}
	hs, free, err := handleVec(keys)
	if err != nil {
		return nil, err
	}
	defer free()
	defer runtime.KeepAlive(keys)
	var out *C.am_groupby
	if err := call("am_group_by_keys", func() C.int {
		return C.amx_group_by_keys(hs, C.int64_t(len(keys)), &out)
	}); err != nil {
		return nil, err
	}
	g := &GroupBy{gb: out, nkeys: len(keys)}
	runtime.SetFinalizer(g, (*GroupBy).Release)
	return g, nil
}

// Release frees the group-by handle. Safe to call more than once.
func (g *GroupBy) Release() {
	if g == nil || g.gb == nil {
		return
	}
	C.amx_group_by_release(g.gb)
	g.gb = nil
	runtime.SetFinalizer(g, nil)
}

// NumGroups is the number of distinct groups.
func (g *GroupBy) NumGroups() int64 {
	if g == nil || g.gb == nil {
		return -1
	}
	defer runtime.KeepAlive(g)
	return int64(C.amx_group_by_group_count(g.gb))
}

// Key returns the i-th key column with one row per group, in group order and in the input column's
// Arrow type.
func (g *GroupBy) Key(i int) (*Array, error) {
	if g == nil || g.gb == nil {
		return nil, errReleased
	}
	if i < 0 || i >= g.nkeys {
		return nil, fmt.Errorf("arrowmetal: key column %d out of range (%d columns)", i, g.nkeys)
	}
	defer runtime.KeepAlive(g)
	var out *C.am_array
	if err := call("am_group_by_keys_result", func() C.int {
		return C.amx_group_by_keys_result(g.gb, C.int64_t(i), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// IDs is the dense group id of every input row, as an int32 array that is never null.
func (g *GroupBy) IDs() (*Array, error) {
	if g == nil || g.gb == nil {
		return nil, errReleased
	}
	defer runtime.KeepAlive(g)
	var out *C.am_array
	if err := call("am_group_by_ids", func() C.int { return C.amx_group_by_ids(g.gb, &out) }); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// Agg runs one grouped aggregate and returns one row per group, in group order. values may be nil
// only for AggCountAll. p1 is the extra parameter some aggregates take (the quantile, for instance)
// and is ignored by the rest.
func (g *GroupBy) Agg(op GroupAgg, values *Array, p1 float64) (*Array, error) {
	if g == nil || g.gb == nil {
		return nil, errReleased
	}
	defer runtime.KeepAlive(g)
	var hv *C.am_array
	if values != nil {
		h, err := values.ptr()
		if err != nil {
			return nil, err
		}
		hv = h
		defer runtime.KeepAlive(values)
	} else if op != AggCountAll {
		return nil, fmt.Errorf("arrowmetal: grouped aggregate %d needs a values column", int(op))
	}
	var out *C.am_array
	if err := call("am_group_agg_ex", func() C.int {
		return C.amx_group_agg_ex(g.gb, hv, C.int(op), C.double(p1), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// Sum is the grouped sum of values, one row per group in group order.
func (g *GroupBy) Sum(values *Array) (*Array, error) { return g.Agg(AggSum, values, 0) }

// Count is the number of non-null values per group.
func (g *GroupBy) Count(values *Array) (*Array, error) { return g.Agg(AggCount, values, 0) }

// CountAll is the number of rows per group, nulls included.
func (g *GroupBy) CountAll() (*Array, error) { return g.Agg(AggCountAll, nil, 0) }

// Mean is the grouped mean of values as a float64 column.
func (g *GroupBy) Mean(values *Array) (*Array, error) { return g.Agg(AggMean, values, 0) }

// Min is the grouped minimum of values.
func (g *GroupBy) Min(values *Array) (*Array, error) { return g.Agg(AggMin, values, 0) }

// Max is the grouped maximum of values.
func (g *GroupBy) Max(values *Array) (*Array, error) { return g.Agg(AggMax, values, 0) }
