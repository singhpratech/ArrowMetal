package arrowmetal

import "testing"

// TestShortDlErr pins the trimming of dyld's dlopen message. Real strings, copied from a failing
// load on macOS 15 / Apple silicon.
func TestShortDlErr(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{
			in: "dlopen(/nope/libArrowMetalC.dylib, 0x0005): tried: '/nope/libArrowMetalC.dylib' " +
				"(no such file), '/System/Volumes/Preboot/Cryptexes/OS/nope/libArrowMetalC.dylib' " +
				"(no such file), '/nope/libArrowMetalC.dylib' (no such file)",
			want: "no such file",
		},
		{
			in: "dlopen(libArrowMetalC.dylib, 0x0005): tried: 'libArrowMetalC.dylib' " +
				"(no such file), '/usr/lib/libArrowMetalC.dylib' (no such file, not in dyld cache)",
			want: "no such file",
		},
		{
			in: "dlopen(/x/lib.dylib, 0x0005): tried: '/x/lib.dylib' (mach-o file, but is an " +
				"incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64')), '/y/lib.dylib' (no such file)",
			want: "mach-o file, but is an incompatible architecture " +
				"(have 'x86_64', need 'arm64e' or 'arm64')",
		},
		{in: "tried: 'x' (no such file)", want: "no such file"},
		{in: "", want: ""},
		{in: "something else entirely", want: "something else entirely"},
	} {
		if got := shortDlErr(tc.in); got != tc.want {
			t.Errorf("shortDlErr(%q)\n got %q\nwant %q", tc.in, got, tc.want)
		}
	}
}

// TestSearchPathShape pins the relative paths the loader tries, because docs/GO.md documents them.
func TestSearchPathShape(t *testing.T) {
	want := []string{
		".build/release",
		"../.build/release",
		"../../.build/release",
		"../../../.build/release",
	}
	if len(searchDirs) != len(want) {
		t.Fatalf("searchDirs = %v, want %v", searchDirs, want)
	}
	for i := range want {
		if searchDirs[i] != want[i] {
			t.Fatalf("searchDirs[%d] = %q, want %q", i, searchDirs[i], want[i])
		}
	}
}
