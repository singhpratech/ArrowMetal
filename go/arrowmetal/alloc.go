package arrowmetal

/*
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// posix_memalign at page granularity. Returns NULL on failure, as malloc does.
static void* am_page_alloc(size_t n) {
    void* p = 0;
    size_t page = (size_t)getpagesize();
    if (posix_memalign(&p, page, n) != 0) return 0;
    return p;
}
*/
import "C"

import (
	"sync/atomic"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow/memory"
)

// PageAlignedAllocator is an arrow-go memory.Allocator whose every allocation starts on a page
// boundary, which is exactly the condition ArrowMetal needs to borrow an incoming buffer instead of
// copying it (see Import).
//
// Two things follow from the memory being C memory rather than Go heap memory:
//
//   - Import is copy-free. Arrow Go's default allocator (memory.NewGoAllocator, which is what
//     memory.DefaultAllocator is unless the module is built with the `mallocator` build tag) makes no
//     alignment promise; what it actually returns is measured in the binding's tests and reported in
//     docs/GO.md.
//   - No Go pointer is handed to C and retained there, so the arrangement does not lean on the fact
//     that Go's collector happens not to move heap objects.
//
// The zero value is ready to use. It is safe for concurrent use.
type PageAlignedAllocator struct {
	allocated atomic.Int64
}

var _ memory.Allocator = (*PageAlignedAllocator)(nil)

// NewPageAlignedAllocator returns an allocator whose buffers ArrowMetal can borrow without copying.
func NewPageAlignedAllocator() *PageAlignedAllocator { return &PageAlignedAllocator{} }

// Allocate returns a zeroed, page-aligned buffer of n bytes.
func (a *PageAlignedAllocator) Allocate(n int) []byte {
	if n < 0 {
		panic("arrowmetal: PageAlignedAllocator.Allocate: negative size")
	}
	if n == 0 {
		// arrow-go expects a non-nil, zero-length slice here; one page keeps the alignment promise
		// uniform and costs nothing measurable.
		n = 1
		p := C.am_page_alloc(C.size_t(n))
		if p == nil {
			panic("arrowmetal: PageAlignedAllocator: out of memory")
		}
		C.memset(p, 0, C.size_t(n))
		a.allocated.Add(int64(n))
		return unsafe.Slice((*byte)(p), n)[:0]
	}
	p := C.am_page_alloc(C.size_t(n))
	if p == nil {
		panic("arrowmetal: PageAlignedAllocator: out of memory")
	}
	C.memset(p, 0, C.size_t(n))
	a.allocated.Add(int64(n))
	return unsafe.Slice((*byte)(p), n)
}

// Reallocate grows or shrinks b to n bytes, keeping the page alignment. The contents are preserved up
// to min(len(b), n) and any new tail is zeroed.
func (a *PageAlignedAllocator) Reallocate(n int, b []byte) []byte {
	if n < 0 {
		panic("arrowmetal: PageAlignedAllocator.Reallocate: negative size")
	}
	out := a.Allocate(n)
	// Allocate(0) hands back a zero-length slice over a one-byte allocation; copy is a no-op then.
	copy(out, b)
	a.Free(b)
	return out
}

// Free releases a buffer returned by Allocate or Reallocate.
func (a *PageAlignedAllocator) Free(b []byte) {
	if cap(b) == 0 {
		return
	}
	p := unsafe.Pointer(&b[:1][0])
	a.allocated.Add(-int64(cap(b)))
	C.free(p)
}

// AllocatedBytes is the number of bytes currently held, for tests and diagnostics.
func (a *PageAlignedAllocator) AllocatedBytes() int64 { return a.allocated.Load() }
