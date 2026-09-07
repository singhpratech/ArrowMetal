// Package arrowmetal is a Go binding for ArrowMetal: Apache Arrow compute on Apple silicon GPUs.
//
// Arrays cross the boundary through the Arrow C Data Interface, so an arrow.Array from
// github.com/apache/arrow-go/v18 goes in and comes back out without this package ever touching the
// element bytes itself. See Import, (*Array).Export and the copy rule documented on Import.
//
// The binding needs libArrowMetalC.dylib, which is a build product of the Swift package and cannot be
// fetched by `go get`. It is opened with dlopen the first time the package is used; see LibraryPath
// and the ARROWMETAL_LIB environment variable.
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
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/cdata"
)

// LibraryName is the file the loader looks for.
const LibraryName = "libArrowMetalC.dylib"

// LibraryEnv is the environment variable that overrides the search: set it to the full path of the
// dylib. When it is set and the file cannot be loaded, the loader reports that and does not fall back.
const LibraryEnv = "ARROWMETAL_LIB"

var (
	loadOnce sync.Once
	loadPath string
	loadErr  error
)

// searchDirs are tried in order, relative to the working directory and then to the directory holding
// the running binary. `.build/release` is where SwiftPM puts the dylib, so a program run from the
// repository root, from go/arrowmetal (where `go test` runs), or from go/arrowmetal/<pkg> finds it.
var searchDirs = []string{
	".build/release",
	"../.build/release",
	"../../.build/release",
	"../../../.build/release",
}

func candidates() []string {
	if p := os.Getenv(LibraryEnv); p != "" {
		return []string{p}
	}
	var out []string
	seen := map[string]bool{}
	add := func(p string) {
		if p != "" && !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	roots := []string{""}
	if exe, err := os.Executable(); err == nil {
		roots = append(roots, filepath.Dir(exe))
	}
	for _, root := range roots {
		for _, d := range searchDirs {
			if root == "" {
				add(filepath.Join(d, LibraryName))
			} else {
				add(filepath.Join(root, d, LibraryName))
			}
		}
	}
	// Last resort: let dyld search its own paths (DYLD_LIBRARY_PATH, /usr/local/lib, ...).
	add(LibraryName)
	return out
}

// shortDlErr keeps a dlopen failure to one readable line.
//
// dyld's message is "dlopen(<path>, <flags>): tried: '<path>' (<reason>), '<fallback>' (<reason>), ..."
// — a screenful per candidate, listing prefixed variants of a path the caller can already see. The
// path is printed alongside this, so all that is wanted is the reason.
func shortDlErr(s string) string {
	if i := strings.Index(s, "): "); i >= 0 {
		s = s[i+3:] // drop the "dlopen(<path>, <flags>): " prefix
	}
	// Keep the parenthesised reason for the first entry, which is the path that was actually asked
	// for; the rest are dyld's own prefixed variants of it. The reason can itself contain
	// parentheses ("(have 'x86_64', need 'arm64')"), so match them rather than scanning for ", ".
	if strings.HasPrefix(s, "tried:") {
		if i := strings.Index(s, "' ("); i >= 0 {
			rest := s[i+len("' ("):]
			depth := 1
			for j := 0; j < len(rest); j++ {
				switch rest[j] {
				case '(':
					depth++
				case ')':
					if depth--; depth == 0 {
						return strings.TrimSpace(rest[:j])
					}
				}
			}
		}
	}
	return strings.TrimSpace(s)
}

func load() {
	cands := candidates()
	var reasons []string
	for _, p := range cands {
		cp := C.CString(p)
		rc := C.amshim_load(cp)
		C.free(unsafe.Pointer(cp))
		if rc == 0 {
			loadPath = p
			return
		}
		reasons = append(reasons, fmt.Sprintf("  %s: %s", p, shortDlErr(C.GoString(C.amshim_error()))))
	}
	var b strings.Builder
	fmt.Fprintf(&b, "arrowmetal: could not load %s.\n", LibraryName)
	if os.Getenv(LibraryEnv) != "" {
		fmt.Fprintf(&b, "%s is set to %q and that path did not load:\n", LibraryEnv, os.Getenv(LibraryEnv))
	} else {
		fmt.Fprintf(&b, "Set %s to the full path of the dylib, or run from a directory where one of\n"+
			"these relative paths resolves. Tried:\n", LibraryEnv)
	}
	b.WriteString(strings.Join(reasons, "\n"))
	b.WriteString("\nBuild it with: DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release")
	loadErr = errors.New(b.String())
}

// Init loads the dylib if it has not been loaded yet and reports whether it succeeded. Every other
// entry point in this package calls it first, so calling Init directly is only useful to fail early
// with a clear message.
func Init() error {
	loadOnce.Do(load)
	return loadErr
}

// LibraryPath is the path the dylib was loaded from, or "" if it has not loaded.
func LibraryPath() string {
	if Init() != nil {
		return ""
	}
	return loadPath
}

// Version is the ArrowMetal version string reported by the loaded dylib.
func Version() (string, error) {
	if err := Init(); err != nil {
		return "", err
	}
	return C.GoString(C.amx_version()), nil
}

// DeviceName is the Metal device ArrowMetal is running on.
func DeviceName() (string, error) {
	if err := Init(); err != nil {
		return "", err
	}
	return C.GoString(C.amx_device_name()), nil
}

// PageSize is the page size the Swift side uses to decide whether an imported buffer can be borrowed
// rather than copied. A buffer whose start address is a multiple of this can be borrowed.
func PageSize() int {
	if Init() != nil {
		return os.Getpagesize()
	}
	return int(C.amshim_page_size())
}

// Error is a failure reported by the ArrowMetal C ABI. Msg is the am_last_error() text.
type Error struct {
	Op   string
	Code int
	Msg  string
}

func (e *Error) Error() string {
	if e.Msg == "" {
		return fmt.Sprintf("arrowmetal: %s failed (code %d)", e.Op, e.Code)
	}
	return fmt.Sprintf("arrowmetal: %s: %s", e.Op, e.Msg)
}

// check turns a non-zero ABI return code into an *Error carrying am_last_error().
//
// am_last_error() is thread-local, so the message has to be read on the same OS thread that made the
// failing call. Every call site wraps the pair in runtime.LockOSThread (see call).
func check(op string, rc C.int) error {
	if rc == 0 {
		return nil
	}
	return &Error{Op: op, Code: int(rc), Msg: C.GoString(C.amx_last_error())}
}

// call runs one ABI call with the goroutine pinned to its OS thread, so that the thread-local error
// slot read by check belongs to the thread that produced it. Go may otherwise move a goroutine to a
// different M between two cgo calls.
func call(op string, fn func() C.int) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	return check(op, fn())
}

// Array is a handle on an Arrow array held by ArrowMetal in GPU-visible unified memory.
//
// It is not safe for concurrent use. Release it when you are done; a finalizer is set as a backstop
// but GPU memory should not wait for the garbage collector.
type Array struct {
	h *C.am_array
}

func wrap(h *C.am_array) *Array {
	a := &Array{h: h}
	runtime.SetFinalizer(a, (*Array).Release)
	return a
}

// Release frees the handle. It is safe to call more than once.
func (a *Array) Release() {
	if a == nil || a.h == nil {
		return
	}
	C.amx_release(a.h)
	a.h = nil
	runtime.SetFinalizer(a, nil)
}

var errReleased = errors.New("arrowmetal: array has been released")

func (a *Array) ptr() (*C.am_array, error) {
	if a == nil || a.h == nil {
		return nil, errReleased
	}
	return a.h, nil
}

// Len is the number of elements.
func (a *Array) Len() int64 {
	if a == nil || a.h == nil {
		return -1
	}
	defer runtime.KeepAlive(a)
	return int64(C.amx_length(a.h))
}

// NullCount is the number of null elements.
func (a *Array) NullCount() int64 {
	if a == nil || a.h == nil {
		return -1
	}
	defer runtime.KeepAlive(a)
	return int64(C.amx_null_count(a.h))
}

// Format is the Arrow C Data Interface format string of the array's type ("l" for int64, "g" for
// float64, "b" for boolean, and so on).
func (a *Array) Format() string {
	if a == nil || a.h == nil {
		return ""
	}
	defer runtime.KeepAlive(a)
	return C.GoString(C.amx_format(a.h))
}

// Import moves an arrow.Array into ArrowMetal through the Arrow C Data Interface.
//
// The copy rule, exactly as the Swift core states it: copy-free out always; copy-free in when the
// producer's buffers are page aligned, one copy otherwise. ArrowMetal borrows an incoming buffer with
// MTLBuffer(bytesNoCopy:) only when the buffer's start address is a multiple of PageSize(); otherwise
// it copies the bytes into a page-aligned Metal buffer. Arrow Go's default allocator does not
// guarantee page alignment — see PageAlignedAllocator and docs/GO.md for what was measured.
//
// The returned Array holds a reference on the source array's buffers until it is released, so the
// arrow.Array stays alive for as long as the handle does.
func Import(arr arrow.Array) (*Array, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if arr == nil {
		return nil, errors.New("arrowmetal: Import: nil array")
	}
	cs := (*C.struct_ArrowSchema)(C.calloc(1, C.sizeof_struct_ArrowSchema))
	ca := (*C.struct_ArrowArray)(C.calloc(1, C.sizeof_struct_ArrowArray))
	defer C.free(unsafe.Pointer(cs))
	defer C.free(unsafe.Pointer(ca))

	cdata.ExportArrowArray(arr,
		cdata.ArrayFromPtr(uintptr(unsafe.Pointer(ca))),
		cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cs))))

	var h *C.am_array
	err := call("am_import", func() C.int { return C.amx_import(cs, ca, &h) })
	// am_import reads the schema but does not take it; the caller always releases it. It moves the
	// array on success, and on failure may or may not have moved it, so release only if it is intact.
	cdata.ReleaseCArrowSchema(cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cs))))
	if err != nil {
		if ca.release != nil {
			cdata.ReleaseCArrowArray(cdata.ArrayFromPtr(uintptr(unsafe.Pointer(ca))))
		}
		return nil, err
	}
	return wrap(h), nil
}

// Export hands the array back to Arrow Go through the C Data Interface. No element bytes are copied:
// the returned arrow.Array references the same GPU-visible memory and keeps it alive.
func (a *Array) Export() (arrow.Array, error) {
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	cs := (*C.struct_ArrowSchema)(C.calloc(1, C.sizeof_struct_ArrowSchema))
	ca := (*C.struct_ArrowArray)(C.calloc(1, C.sizeof_struct_ArrowArray))
	defer C.free(unsafe.Pointer(cs))
	defer C.free(unsafe.Pointer(ca))

	if err := call("am_export", func() C.int { return C.amx_export(h, cs, ca) }); err != nil {
		return nil, err
	}
	// ImportCArray moves the array's contents and releases the schema; it does not take ownership of
	// the two structs themselves, so the deferred frees above are correct.
	_, out, err := cdata.ImportCArray(
		cdata.ArrayFromPtr(uintptr(unsafe.Pointer(ca))),
		cdata.SchemaFromPtr(uintptr(unsafe.Pointer(cs))))
	if err != nil {
		return nil, fmt.Errorf("arrowmetal: export: %w", err)
	}
	return out, nil
}

// Slice is a zero-copy view of length elements starting at offset.
func (a *Array) Slice(offset, length int64) (*Array, error) {
	h, err := a.ptr()
	if err != nil {
		return nil, err
	}
	defer runtime.KeepAlive(a)
	var out *C.am_array
	if err := call("am_slice", func() C.int {
		return C.amx_slice(h, C.int64_t(offset), C.int64_t(length), &out)
	}); err != nil {
		return nil, err
	}
	return wrap(out), nil
}
