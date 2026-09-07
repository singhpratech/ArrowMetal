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
// Copy-free import needs the buffer's start address to be a multiple of the page size
// (16 KiB on Apple silicon); ArrowMetal copies the bytes into a page-aligned Metal buffer otherwise.
// The test asserts nothing about the Go allocator's answer — it has no alignment contract to assert —
// it records it, and it does assert that PageAlignedAllocator delivers what it promises.
func TestAllocatorAlignment(t *testing.T) {
	requireLib(t)
	page := uintptr(am.PageSize())

	type row struct {
		alloc   string
		n       int
		aligned int
		trials  int
		offsets []uintptr
	}
	var rows []row

	const trials = 20
	for _, n := range []int{1_000_000, 10_000_000} {
		vals := make([]int64, n)
		for i := range vals {
			vals[i] = int64(i)
		}
		for _, a := range []struct {
			name string
			mem  memory.Allocator
		}{
			{"memory.NewGoAllocator (arrow-go default)", memory.NewGoAllocator()},
			{"memory.DefaultAllocator", memory.DefaultAllocator},
			{"arrowmetal.PageAlignedAllocator", am.NewPageAlignedAllocator()},
		} {
			r := row{alloc: a.name, n: n, trials: trials}
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
		t.Logf("%-42s n=%-10d page-aligned %d/%d  (offsets mod page: %v)",
			r.alloc, r.n, r.aligned, r.trials, r.offsets)
		if r.alloc == "arrowmetal.PageAlignedAllocator" && r.aligned != r.trials {
			t.Fatalf("PageAlignedAllocator produced an unaligned buffer: %v", r.offsets)
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
