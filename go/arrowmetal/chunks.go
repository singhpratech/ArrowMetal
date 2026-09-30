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
	"github.com/apache/arrow-go/v18/arrow/cdata"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// ImportChunks imports a column held as several arrow.Arrays of one type — the chunks of an
// arrow.Chunked, one column of a run of record batches — as one Array of their total length, without
// concatenating them first.
//
// Each chunk's offset, length, validity bitmap (or its absence) and null count are honoured, and
// each chunk's buffers are copied straight into the final Metal buffers, on the CPU cores in
// parallel (am_import_chunks). One chunk is Import, with Import's copy rule. A type the chunked
// import does not take (dictionary, nested, run-end encoded, extension) is concatenated with
// array.Concatenate and imported, which is what a caller would otherwise do by hand;
// ChunksSupported reports which case a type falls into.
//
// Every chunk must have the first one's type. The chunks stay the caller's: release them as usual.
func ImportChunks(chunks []arrow.Array) (*Array, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if len(chunks) == 0 {
		return nil, errors.New("arrowmetal: ImportChunks needs at least one chunk (the type comes " +
			"from it); use ImportChunked for an arrow.Chunked, which carries its type")
	}
	for i, c := range chunks {
		if c == nil {
			return nil, fmt.Errorf("arrowmetal: ImportChunks: chunk %d is nil", i)
		}
		if !arrow.TypeEqual(c.DataType(), chunks[0].DataType()) {
			return nil, fmt.Errorf("arrowmetal: ImportChunks: every chunk must have one type; chunk 0 "+
				"is %s and chunk %d is %s", chunks[0].DataType(), i, c.DataType())
		}
	}
	if len(chunks) == 1 {
		return Import(chunks[0])
	}
	if C.amshim_has_import_chunks() == 0 {
		return nil, fmt.Errorf("arrowmetal: ImportChunks: the loaded %s (%s) has no "+
			"am_import_chunks; it predates the chunked import, so rebuild it", LibraryName, loadPath)
	}

	n := len(chunks)
	cs := (*C.struct_ArrowSchema)(C.calloc(1, C.sizeof_struct_ArrowSchema))
	defer C.free(unsafe.Pointer(cs))
	// The n structs am_import_chunks takes, contiguous, in C memory.
	cas := (*C.struct_ArrowArray)(C.calloc(C.size_t(n), C.sizeof_struct_ArrowArray))
	defer C.free(unsafe.Pointer(cas))
	arrs := unsafe.Slice(cas, n)

	pin := new(runtime.Pinner)
	pinBuffers(pin, chunks[0].Data())
	cdata.ExportArrowArray(chunks[0],
		cdata.ArrayFromPtr(uintptr(unsafe.Pointer(&arrs[0]))),
		cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cs))))
	releaseSchema := func() { cdata.ReleaseCArrowSchema(cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cs)))) }
	// Every export that still has its release set is ours; am_import_chunks nulls the release of a
	// chunk it moved, so this releases exactly the ones it did not take, once each.
	releaseLeft := func() {
		for i := range arrs {
			if arrs[i].release != nil {
				cdata.ReleaseCArrowArray(cdata.ArrayFromPtr(uintptr(unsafe.Pointer(&arrs[i]))))
			}
		}
	}

	if C.amx_import_chunks_supported(cs) == 0 {
		releaseLeft()
		releaseSchema()
		pin.Unpin()
		return importConcatenated(chunks)
	}
	for i := 1; i < n; i++ {
		pinBuffers(pin, chunks[i].Data())
		cdata.ExportArrowArray(chunks[i], cdata.ArrayFromPtr(uintptr(unsafe.Pointer(&arrs[i]))), nil)
	}

	var h *C.am_array
	err := call("am_import_chunks", func() C.int { return C.amx_import_chunks(cs, cas, C.int64_t(n), &h) })
	// The schema is read, never taken, so it is released here either way.
	releaseSchema()
	if err != nil {
		releaseLeft()
		pin.Unpin()
		return nil, err
	}
	out := wrap(h)
	// The chunks were copied, but the pin is held to Release all the same, as Import's is: it costs
	// nothing and does not depend on when the library runs each chunk's release callback.
	out.pin = pin
	return out, nil
}

// importConcatenated is the path for a type the chunked import does not take.
func importConcatenated(chunks []arrow.Array) (*Array, error) {
	merged, err := array.Concatenate(chunks, memory.NewGoAllocator())
	if err != nil {
		return nil, fmt.Errorf("arrowmetal: concatenating the chunks: %w", err)
	}
	defer merged.Release() // the import's own export holds a reference for as long as it needs one
	return Import(merged)
}

// ImportChunked imports an arrow.Chunked as one Array: ImportChunks over its chunks. A Chunked with
// no chunks gives an empty Array of its type.
func ImportChunked(c *arrow.Chunked) (*Array, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if c == nil {
		return nil, errors.New("arrowmetal: ImportChunked: nil *arrow.Chunked")
	}
	if len(c.Chunks()) == 0 {
		empty := array.MakeArrayOfNull(memory.NewGoAllocator(), c.DataType(), 0)
		defer empty.Release()
		return Import(empty)
	}
	return ImportChunks(c.Chunks())
}

// ImportColumn imports column i of every batch as one Array (ImportChunks over the batches'
// columns), so a run of record batches needs no concatenation first.
func ImportColumn(batches []arrow.RecordBatch, i int) (*Array, error) {
	if len(batches) == 0 {
		return nil, errors.New("arrowmetal: ImportColumn needs at least one record batch")
	}
	chunks := make([]arrow.Array, len(batches))
	for b, rb := range batches {
		if rb == nil {
			return nil, fmt.Errorf("arrowmetal: ImportColumn: batch %d is nil", b)
		}
		if i < 0 || i >= int(rb.NumCols()) {
			return nil, fmt.Errorf("arrowmetal: ImportColumn: column %d is out of range for batch %d, "+
				"which has %d columns", i, b, rb.NumCols())
		}
		chunks[b] = rb.Column(i)
	}
	return ImportChunks(chunks)
}

// ChunksSupported reports whether ImportChunks takes chunks of type dt through the chunked import
// (true) or concatenates them first (false).
func ChunksSupported(dt arrow.DataType) (bool, error) {
	if err := Init(); err != nil {
		return false, err
	}
	if C.amshim_has_import_chunks() == 0 {
		return false, nil
	}
	empty := array.MakeArrayOfNull(memory.NewGoAllocator(), dt, 0)
	defer empty.Release()
	cs := (*C.struct_ArrowSchema)(C.calloc(1, C.sizeof_struct_ArrowSchema))
	ca := (*C.struct_ArrowArray)(C.calloc(1, C.sizeof_struct_ArrowArray))
	defer C.free(unsafe.Pointer(cs))
	defer C.free(unsafe.Pointer(ca))
	// An empty array still has Go-heap buffers, which the export publishes into C memory.
	var pin runtime.Pinner
	pinBuffers(&pin, empty.Data())
	defer pin.Unpin()
	cdata.ExportArrowArray(empty, cdata.ArrayFromPtr(uintptr(unsafe.Pointer(ca))),
		cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cs))))
	ok := C.amx_import_chunks_supported(cs) != 0
	cdata.ReleaseCArrowArray(cdata.ArrayFromPtr(uintptr(unsafe.Pointer(ca))))
	cdata.ReleaseCArrowSchema(cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cs))))
	return ok, nil
}

// NewSourceFromBatches registers a run of record batches as one table: each column is imported with
// ImportColumn, so the batches are never concatenated. Every batch must have the first one's schema.
func NewSourceFromBatches(name string, batches []arrow.RecordBatch) (*Source, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if len(batches) == 0 {
		return nil, fmt.Errorf("arrowmetal: NewSourceFromBatches(%q) needs at least one batch (the "+
			"schema comes from it)", name)
	}
	schema := batches[0].Schema()
	for b, rb := range batches {
		if rb == nil {
			return nil, fmt.Errorf("arrowmetal: NewSourceFromBatches(%q): batch %d is nil", name, b)
		}
		if !rb.Schema().Equal(schema) {
			return nil, fmt.Errorf("arrowmetal: NewSourceFromBatches(%q): every batch must have one "+
				"schema; batch 0 is %s and batch %d is %s", name, schema, b, rb.Schema())
		}
	}
	names := make([]string, schema.NumFields())
	cols := make([]*Array, 0, schema.NumFields())
	fail := func(err error) (*Source, error) {
		for _, c := range cols {
			c.Release()
		}
		return nil, err
	}
	for i, f := range schema.Fields() {
		names[i] = f.Name
		c, err := ImportColumn(batches, i)
		if err != nil {
			return fail(fmt.Errorf("arrowmetal: NewSourceFromBatches(%q): column %q: %w", name, f.Name, err))
		}
		cols = append(cols, c)
	}
	src, err := NewSource(name, names, cols)
	if err != nil {
		return fail(err)
	}
	// The source keeps the handles and releases them after itself: a single batch is imported with
	// Import, which may borrow (and so pins) the batch's own buffers.
	src.owned = cols
	return src, nil
}
