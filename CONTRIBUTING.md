# Contributing

Thanks for looking. This project is small enough to hold in your head; please keep it that way.

## Ground rules
- Every kernel is tested against a CPU oracle — the plain-Swift implementations in
  `Sources/ArrowMetal/CPUReference.swift`, a hand-computed vector, or a `pyarrow.compute` answer pinned
  as a literal — across sizes 0, 1, word boundaries (31/32/33), threadgroup and simdgroup boundaries
  (255/256/257 and 8191/8192/8193 in some suites, 1023/1024/1025 and 65535/65536/65537 in others; both
  sets occur, and [docs/TESTING.md](docs/TESTING.md) lists the second) and something large. Add both the oracle and the test when you add a kernel.
- Follow Arrow semantics (nulls, wrapping arithmetic, bit order). When Arrow has a documented behaviour,
  match it; when it does not, document what you chose.
- Kernels are MSL strings in the `*Source.swift` files under `Sources/ArrowMetal/Kernels/`
  (`KernelSource.swift` holds the element-wise family; there are 47 of them), generated per element type.
  Keep them readable; a slower obvious kernel beats a clever one until a benchmark says otherwise.
- Run `swift test` in **both** debug and release (`swift test -c release`). We have already hit one
  release-only miscompile (see the comment at `Sources/ArrowMetal/MetalArray.swift:283`).
- Run the binding suites too, and in release: `PYTHONPATH=python python -m pytest python/tests -q`
  and `cd rust && cargo test --release`. The Rust suite compares against arrow-rs's own compute
  kernels on the same data; if you touch `include/arrowmetal.h`, `rust/arrowmetal/tests/signatures.rs`
  will tell you whether `rust/arrowmetal-sys` still matches it. Full list in
  [docs/TESTING.md](docs/TESTING.md).
- Run the binding suites when you touch the C ABI or `include/arrowmetal.h`:
  `PYTHONPATH=python python -m pytest python/tests -q` and, from `go/arrowmetal`,
  `ARROWMETAL_LIB=$PWD/../../.build/release/libArrowMetalC.dylib go test ./...` — then once more
  with `GOEXPERIMENT=cgocheck2`, which is where a Go pointer handed to C without being pinned turns
  into a hard failure instead of luck. The Go module compiles against a copy of the header under
  `go/arrowmetal/include/`; if you change the real one, copy it across
  (`TestHeadersMatchRepository` tells you so).
- Run `swift run -c release arrowmetal-bench` before and after a performance change and paste both tables in
  the PR.
- Run the binding suites when you touch the C ABI: `PYTHONPATH=python python -m pytest python/tests -q`
  and, for the Node binding, `(cd node && npm install && npm test)` — see [docs/TESTING.md](docs/TESTING.md).
- Run the binding suites when you touch them: `PYTHONPATH=python python -m pytest python/tests -q` for
  Python, and for R
  `ARROWMETAL_LIB=$PWD/.build/release/libArrowMetalC.dylib Rscript -e 'testthat::test_local("r/arrowmetal")'`
  (see [docs/R.md](docs/R.md); `r/arrowmetal/src/arrowmetal.h` is a copy of `include/arrowmetal.h` and a test
  fails when it goes stale).

## Setup
```
git clone <repo>
cd ArrowMetal
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
swift run -c release arrowmetal-bench 10000000 3
```
XCTest needs Xcode; the library itself builds with Command Line Tools only.

## CI and real hardware
GitHub's hosted Apple silicon runners expose an "Apple Paravirtual device" whose Metal compiler fails
sporadically, so GPU tests are skipped there (`requireRealGPU()`), and CI proves the build, interop and CPU
paths only. Run the full suite on a real Mac before merging; set `ARROWMETAL_FORCE_GPU_TESTS=1` to run GPU
tests on a virtual device anyway. `.github/workflows/ci.yml` therefore runs on pull requests and by hand
(`workflow_dispatch`), not on every push.

## Debugging GPU kernels
- `ARROWMETAL_DEBUG_SHADERS=1` dumps the generated MSL and the full compiler log when a shader or pipeline fails.
- `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift run ...` enables API and shader validation.
- Compilation errors from `makeLibrary(source:)` include line numbers relative to the generated source; print
  `KernelSource.<family>(T:)` to see it.

## Pull requests
Small and focused. One kernel or one feature per PR. Update [docs/ROADMAP.md](docs/ROADMAP.md) if you
finish an item.
