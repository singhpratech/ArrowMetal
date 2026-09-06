# Contributing

Thanks for looking. This project is small enough to hold in your head; please keep it that way.

## Ground rules
- Every kernel has a CPU reference in `Sources/ArrowMetal/CPUReference.swift` and a test that compares the
  GPU result to it across sizes 0, 1, word boundaries (31/32/33), threadgroup boundaries (255/256/257,
  8191/8192/8193) and something large. Add both when you add a kernel.
- Follow Arrow semantics (nulls, wrapping arithmetic, bit order). When Arrow has a documented behaviour,
  match it; when it does not, document what you chose.
- Kernels are MSL strings in `Sources/ArrowMetal/Kernels/KernelSource.swift`, generated per element type.
  Keep them readable; a slower obvious kernel beats a clever one until a benchmark says otherwise.
- Run `swift test` in **both** debug and release (`swift test -c release`). We have already hit one
  release-only miscompile (see the comment in `MetalArray.init(_:[T])`).
- Run `swift run -c release arrowmetal-bench` before and after a performance change and paste both tables in
  the PR.

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
tests on a virtual device anyway.

## Debugging GPU kernels
- `ARROWMETAL_DEBUG_SHADERS=1` dumps the generated MSL and the full compiler log when a shader or pipeline fails.
- `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 swift run ...` enables API and shader validation.
- Compilation errors from `makeLibrary(source:)` include line numbers relative to the generated source; print
  `KernelSource.<family>(T:)` to see it.

## Pull requests
Small and focused. One kernel or one feature per PR. Update `ROADMAP.md` if you finish an item.
