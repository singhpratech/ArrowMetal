package arrowmetal_test

import (
	"testing"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// bufferAddr is the address of an array's values buffer, which is the buffer ArrowMetal decides to
// borrow or copy on import.
func bufferAddr(a arrow.Array) uintptr {
	bufs := a.Data().Buffers()
	if len(bufs) < 2 || bufs[1] == nil {
		return 0
	}
	b := bufs[1].Bytes()
	if len(b) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&b[0]))
}

// TestAllocatorAlignment is the measurement behind the copy rule in docs/GO.md: does Arrow Go's
// default allocator hand ArrowMetal a page-aligned buffer, at 1M and 10M elements?
//
// Copy-free import needs the buffer's start address to be a multiple of the page size (16 KiB on
// Apple silicon); ArrowMetal copies the bytes into a page-aligned Metal buffer otherwise.
//
// What this test asserts is the *shape* of the answer, not a hit rate. Every offset a Go-heap
// buffer produces is a multiple of 8192 — Go's own page — so it is page aligned when that multiple
// happens to be even and 8192 bytes short when it is odd. Which one you get depends on the state of
// the heap: across fresh runs of this same test the count at 10M swung between 5/20 and 15/20, so a
// fraction here would be noise dressed up as a property. The invariant is what gets asserted; the
// count is logged for the record.
//
// PageAlignedAllocator, by contrast, has a contract, and that is asserted exactly.
func TestAllocatorAlignment(t *testing.T) {
	requireLib(t)
	page := uintptr(am.PageSize())

	// docs/GO.md says memory.DefaultAllocator *is* the Go allocator unless the program is built
	// with the `mallocator` tag. Pin that rather than asserting it in prose.
	if _, ok := memory.DefaultAllocator.(*memory.GoAllocator); !ok {
		t.Logf("memory.DefaultAllocator is %T, not *memory.GoAllocator; "+
			"docs/GO.md's claim about the default allocator needs revisiting", memory.DefaultAllocator)
	}

	type row struct {
		alloc      string
		goHeap     bool
		n          int
		aligned    int
		trials     int
		offsets    []uintptr
		distinctOK bool
	}
	var rows []row

	const trials = 20
	for _, n := range []int{1_000_000, 10_000_000} {
		vals := make([]int64, n)
		for i := range vals {
			vals[i] = int64(i)
		}
		for _, a := range []struct {
			name   string
			mem    memory.Allocator
			goHeap bool
		}{
			{"memory.NewGoAllocator (== memory.DefaultAllocator)", memory.NewGoAllocator(), true},
			{"arrowmetal.PageAlignedAllocator", am.NewPageAlignedAllocator(), false},
		} {
			r := row{alloc: a.name, goHeap: a.goHeap, n: n, trials: trials, distinctOK: true}
			for i := 0; i < trials; i++ {
				b := array.NewInt64Builder(a.mem)
				b.AppendValues(vals, nil)
				arr := b.NewArray()
				off := bufferAddr(arr) % page
				r.offsets = append(r.offsets, off)
				if off == 0 {
					r.aligned++
				}
				arr.Release()
				b.Release()
			}
			rows = append(rows, r)
		}
	}

	t.Logf("page size: %d bytes", page)
	for _, r := range rows {
		t.Logf("%-50s n=%-10d page-aligned %d/%d this run  (offsets mod page: %v)",
			r.alloc, r.n, r.aligned, r.trials, r.offsets)

		if !r.goHeap {
			if r.aligned != r.trials {
				t.Fatalf("PageAlignedAllocator produced an unaligned buffer: %v", r.offsets)
			}
			continue
		}
		// The reproducible claim: a Go-heap buffer is aligned to Go's 8 KiB page, so its offset
		// past a 16 KiB page is only ever 0 or 8192 — never 64, never 4096.
		half := page / 2
		for i, off := range r.offsets {
			if off != 0 && off != half {
				t.Fatalf("%s n=%d trial %d: offset %d past a %d-byte page; expected only 0 or %d "+
					"(Go's heap aligns large spans to %d)", r.alloc, r.n, i, off, page, half, half)
			}
		}
	}
}

// TestPageAlignedAllocatorRoundTrip checks that an array built with the page-aligned allocator gives
// the same answers as one built with the Go allocator — the allocator changes whether the import
// copies, never what comes out.
func TestPageAlignedAllocatorRoundTrip(t *testing.T) {
	requireLib(t)
	alloc := am.NewPageAlignedAllocator()
	const n = 1000001
	vals := genInt64(n)
	valid := nullEvery(n, 7)

	b := array.NewInt64Builder(alloc)
	b.AppendValues(vals, valid)
	src := b.NewArray()
	b.Release()
	defer src.Release()

	if got := bufferAddr(src) % uintptr(am.PageSize()); got != 0 {
		t.Fatalf("values buffer is %d bytes past a page boundary", got)
	}

	h := importArr(t, src)
	if h.Len() != n {
		t.Fatalf("Len() = %d, want %d", h.Len(), n)
	}
	s, err := h.Sum()
	if err != nil {
		t.Fatal(err)
	}
	var want int64
	for i := range vals {
		if valid[i] {
			want += vals[i]
		}
	}
	if s.Int64() != want {
		t.Fatalf("Sum = %d, want %d", s.Int64(), want)
	}

	out := exportArr(t, h)
	gotV, gotValid := int64sOf(t, out)
	for i := range gotV {
		if gotValid[i] != valid[i] || (valid[i] && gotV[i] != vals[i]) {
			t.Fatalf("element %d: got (%d, valid=%v), want (%d, valid=%v)",
				i, gotV[i], gotValid[i], vals[i], valid[i])
		}
	}
	if alloc.AllocatedBytes() <= 0 {
		t.Fatalf("AllocatedBytes() = %d while an array is alive", alloc.AllocatedBytes())
	}
}

// TestPageAlignedAllocatorFrees checks the allocator's own bookkeeping: everything it hands out comes
// back.
func TestPageAlignedAllocatorFrees(t *testing.T) {
	alloc := am.NewPageAlignedAllocator()
	if got := alloc.AllocatedBytes(); got != 0 {
		t.Fatalf("a fresh allocator holds %d bytes", got)
	}
	b1 := alloc.Allocate(1000)
	if len(b1) != 1000 {
		t.Fatalf("Allocate(1000) returned %d bytes", len(b1))
	}
	for _, v := range b1 {
		if v != 0 {
			t.Fatal("Allocate returned memory that is not zeroed")
		}
	}
	copy(b1, []byte("hello"))
	b2 := alloc.Reallocate(4000, b1)
	if len(b2) != 4000 || string(b2[:5]) != "hello" {
		t.Fatalf("Reallocate lost the contents: len=%d head=%q", len(b2), b2[:5])
	}
	for _, v := range b2[1000:] {
		if v != 0 {
			t.Fatal("Reallocate left the new tail unzeroed")
		}
	}
	b3 := alloc.Reallocate(0, b2)
	if len(b3) != 0 {
		t.Fatalf("Reallocate(0) returned %d bytes", len(b3))
	}
	alloc.Free(b3)
	if got := alloc.AllocatedBytes(); got != 0 {
		t.Fatalf("after freeing everything the allocator still holds %d bytes", got)
	}
}
