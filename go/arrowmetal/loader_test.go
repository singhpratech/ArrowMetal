package arrowmetal_test

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// The loader resolves once per process, so the failure path can only be exercised in a fresh
// process. TestLoaderErrorNamesEverything re-runs this test binary with a bad ARROWMETAL_LIB and
// checks what comes out; the child is this same function, selected by the marker variable.
const loaderChildEnv = "ARROWMETAL_LOADER_CHILD"

func TestLoaderErrorNamesEverything(t *testing.T) {
	if os.Getenv(loaderChildEnv) == "1" {
		// Child: report the loader's error on stdout and exit cleanly, so the parent can read it.
		err := am.Init()
		if err == nil {
			os.Stdout.WriteString("UNEXPECTED: the library loaded\n")
			os.Exit(0)
		}
		os.Stdout.WriteString(err.Error())
		os.Exit(0)
	}

	for _, tc := range []struct {
		name string
		env  []string
		want []string
	}{
		{
			name: "ARROWMETAL_LIB points at nothing",
			env:  []string{am.LibraryEnv + "=/nonexistent/path/to/libArrowMetalC.dylib"},
			want: []string{
				"ARROWMETAL_LIB",
				"/nonexistent/path/to/libArrowMetalC.dylib",
				"swift build -c release",
			},
		},
		{
			name: "nothing set and nothing on the search path",
			// An empty ARROWMETAL_LIB is the same as unset, and the working directory is a temporary
			// one with no .build anywhere above it.
			env: []string{am.LibraryEnv + "="},
			want: []string{
				"ARROWMETAL_LIB",
				".build/release/libArrowMetalC.dylib",
				"../../.build/release/libArrowMetalC.dylib",
				"swift build -c release",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			exe, err := os.Executable()
			if err != nil {
				t.Skipf("cannot find the test binary: %v", err)
			}
			cmd := exec.Command(exe, "-test.run", "^TestLoaderErrorNamesEverything$")
			cmd.Dir = t.TempDir()
			cmd.Env = append(append(os.Environ(), loaderChildEnv+"=1"), tc.env...)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("child failed: %v\n%s", err, out)
			}
			got := string(out)
			if strings.Contains(got, "UNEXPECTED") {
				t.Fatalf("the child found a library it should not have:\n%s", got)
			}
			for _, want := range tc.want {
				if !strings.Contains(got, want) {
					t.Fatalf("the loader error does not mention %q:\n%s", want, got)
				}
			}
			t.Logf("loader error:\n%s", got)
		})
	}
}
