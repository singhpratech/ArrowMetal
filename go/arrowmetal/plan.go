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
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
)

// Source is a named table the query engine can scan. It retains the column handles it was built
// with, so releasing the Arrays that went into it does not invalidate the source.
type Source struct {
	s *C.am_plan_source
}

// NewSource registers a table under name. names and columns must be the same length; the columns are
// the table's fields in order.
func NewSource(name string, names []string, columns []*Array) (*Source, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if len(names) != len(columns) {
		return nil, fmt.Errorf("arrowmetal: NewSource(%q): %d names but %d columns", name, len(names), len(columns))
	}
	if len(columns) == 0 {
		return nil, fmt.Errorf("arrowmetal: NewSource(%q) needs at least one column", name)
	}
	hs, freeH, err := handleVec(columns)
	if err != nil {
		return nil, err
	}
	defer freeH()
	defer runtime.KeepAlive(columns)

	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	cnames := (**C.char)(C.calloc(C.size_t(len(names)), C.size_t(unsafe.Sizeof((*C.char)(nil)))))
	defer C.free(unsafe.Pointer(cnames))
	ns := unsafe.Slice(cnames, len(names))
	for i, n := range names {
		ns[i] = C.CString(n)
	}
	defer func() {
		for i := range ns {
			C.free(unsafe.Pointer(ns[i]))
		}
	}()

	var out *C.am_plan_source
	if err := call("am_plan_source_create", func() C.int {
		return C.amx_plan_source_create(cname, hs, cnames, C.int64_t(len(columns)), &out)
	}); err != nil {
		return nil, err
	}
	s := &Source{s: out}
	runtime.SetFinalizer(s, (*Source).Release)
	return s, nil
}

// Release frees the source. Safe to call more than once.
func (s *Source) Release() {
	if s == nil || s.s == nil {
		return
	}
	C.amx_plan_source_release(s.s)
	s.s = nil
	runtime.SetFinalizer(s, nil)
}

func sourceVec(sources []*Source) (**C.am_plan_source, func(), error) {
	n := len(sources)
	if n == 0 {
		return nil, func() {}, nil
	}
	p := (**C.am_plan_source)(C.calloc(C.size_t(n), C.size_t(unsafe.Sizeof((*C.am_plan_source)(nil)))))
	sl := unsafe.Slice(p, n)
	for i, s := range sources {
		if s == nil || s.s == nil {
			C.free(unsafe.Pointer(p))
			return nil, nil, fmt.Errorf("arrowmetal: source %d has been released", i)
		}
		sl[i] = s.s
	}
	return p, func() { C.free(unsafe.Pointer(p)) }, nil
}

// PlanResult is the output of RunPlan: a set of named columns.
type PlanResult struct {
	r *C.am_plan_result
}

// RunPlan type-checks, optionally optimizes, lowers and runs a JSON plan over the given sources. The
// plan grammar is documented in the C header and in docs/ENGINE.md.
func RunPlan(planJSON string, optimize bool, sources ...*Source) (*PlanResult, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	sp, freeS, err := sourceVec(sources)
	if err != nil {
		return nil, err
	}
	defer freeS()
	defer runtime.KeepAlive(sources)
	cp := C.CString(planJSON)
	defer C.free(unsafe.Pointer(cp))
	var out *C.am_plan_result
	if err := call("am_plan_run", func() C.int {
		return C.amx_plan_run(cp, sp, C.int64_t(len(sources)), cbool(optimize), &out)
	}); err != nil {
		return nil, err
	}
	r := &PlanResult{r: out}
	runtime.SetFinalizer(r, (*PlanResult).Release)
	return r, nil
}

// ExplainPlan returns the optimized logical plan and the physical plan it lowers to, as text.
func ExplainPlan(planJSON string, optimize bool, sources ...*Source) (string, error) {
	if err := Init(); err != nil {
		return "", err
	}
	sp, freeS, err := sourceVec(sources)
	if err != nil {
		return "", err
	}
	defer freeS()
	defer runtime.KeepAlive(sources)
	cp := C.CString(planJSON)
	defer C.free(unsafe.Pointer(cp))

	// am_plan_explain returns text valid until the next call on this thread, and reports failure by
	// returning NULL with the reason in am_last_error(); both need the thread pinned.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	txt := C.amx_plan_explain(cp, sp, C.int64_t(len(sources)), cbool(optimize))
	if txt == nil {
		return "", &Error{Op: "am_plan_explain", Code: 1, Msg: C.GoString(C.amx_last_error())}
	}
	return C.GoString(txt), nil
}

// Release frees the result. Safe to call more than once.
func (r *PlanResult) Release() {
	if r == nil || r.r == nil {
		return
	}
	C.amx_plan_result_release(r.r)
	r.r = nil
	runtime.SetFinalizer(r, nil)
}

// NumColumns is the number of result columns.
func (r *PlanResult) NumColumns() int {
	if r == nil || r.r == nil {
		return -1
	}
	defer runtime.KeepAlive(r)
	return int(C.amx_plan_column_count(r.r))
}

// NumRows is the number of result rows.
func (r *PlanResult) NumRows() int64 {
	if r == nil || r.r == nil {
		return -1
	}
	defer runtime.KeepAlive(r)
	return int64(C.amx_plan_row_count(r.r))
}

// ColumnName is the name of the i-th result column.
func (r *PlanResult) ColumnName(i int) string {
	if r == nil || r.r == nil {
		return ""
	}
	defer runtime.KeepAlive(r)
	return C.GoString(C.amx_plan_column_name(r.r, C.int64_t(i)))
}

// Column hands out a new handle on the i-th result column; release it when you are done.
func (r *PlanResult) Column(i int) (*Array, error) {
	if r == nil || r.r == nil {
		return nil, errReleased
	}
	defer runtime.KeepAlive(r)
	var out *C.am_array
	if err := call("am_plan_column", func() C.int {
		return C.amx_plan_column(r.r, C.int64_t(i), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// RecordBatch exports every result column back to Arrow Go as one record batch. The caller releases
// it. No element bytes are copied.
func (r *PlanResult) RecordBatch() (arrow.RecordBatch, error) {
	n := r.NumColumns()
	if n < 0 {
		return nil, errReleased
	}
	if n == 0 {
		return nil, errors.New("arrowmetal: plan result has no columns")
	}
	fields := make([]arrow.Field, n)
	cols := make([]arrow.Array, 0, n)
	ok := false
	defer func() {
		if !ok {
			for _, c := range cols {
				c.Release()
			}
		}
	}()
	for i := 0; i < n; i++ {
		h, err := r.Column(i)
		if err != nil {
			return nil, err
		}
		a, err := h.Export()
		h.Release()
		if err != nil {
			return nil, err
		}
		cols = append(cols, a)
		fields[i] = arrow.Field{Name: r.ColumnName(i), Type: a.DataType(), Nullable: true}
	}
	rb := array.NewRecordBatch(arrow.NewSchema(fields, nil), cols, r.NumRows())
	for _, c := range cols {
		c.Release() // the record batch retained them
	}
	ok = true
	return rb, nil
}
