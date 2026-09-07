package arrowmetal_test

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestHeadersMatchRepository guards the one duplication this module has to carry.
//
// A Go module can only see files inside its own directory, so `go/arrowmetal/include/` holds copies
// of the repository's `include/arrowmetal.h` and `include/arrow_abi.h`. The cgo shim compiles
// against those copies, which is what makes a signature change on the Swift side a compile error
// rather than a crash — but only if the copies are current. Inside a checkout this test compares
// them; outside one (a `go get` of the published module) there is nothing to compare and it skips.
func TestHeadersMatchRepository(t *testing.T) {
	for _, name := range []string{"arrowmetal.h", "arrow_abi.h"} {
		upstream := filepath.Join("..", "..", "include", name)
		want, err := os.ReadFile(upstream)
		if err != nil {
			t.Skipf("no repository checkout at %s: %v", upstream, err)
		}
		got, err := os.ReadFile(filepath.Join("include", name))
		if err != nil {
			t.Fatalf("vendored %s is missing: %v", name, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("go/arrowmetal/include/%s has drifted from include/%s; copy it across:\n"+
				"    cp include/%s go/arrowmetal/include/%s", name, name, name, name)
		}
	}
}
