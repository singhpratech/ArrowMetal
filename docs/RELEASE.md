# Releasing

How a release of ArrowMetal is built, verified and published. Every step is the maintainer's to run;
the repository itself never publishes anything.

**The version number is changed only when a release is decided, in every manifest at once**, and not
between releases: work after a release goes under an "Unreleased" heading in CHANGELOG.md until the
next one. Read `X.Y.Z` below as the version being released.

## 0. Preconditions

Run these and read the output; do not proceed on a red result.

```
git status --porcelain                                     # must be empty
git switch main && git pull --ff-only

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test               # debug
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test -c release    # release

swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests -q
```

The binding suites run as well ([TESTING.md](TESTING.md), "Run everything"). Confirm the Python
version is stated in exactly one place and that every binding's own manifest agrees:

```
grep -rn '__version__' python/arrowmetal/__init__.py       # the single source
grep -n 'version' python/pyproject.toml                    # must be `dynamic`, reading the above
grep -nE 'X\.Y\.Z' rust/Cargo.toml rust/arrowmetal/Cargo.toml polars-plugin/Cargo.toml \
    polars-plugin/arrowmetal-sys/Cargo.toml node/package.json r/arrowmetal/DESCRIPTION \
    duckdb-extension/CMakeLists.txt duckdb-extension/build.sh   # all must read X.Y.Z
```

## 1. Rehearse the upload on TestPyPI

```
python -m pip install --upgrade build twine
python/build_wheel.sh
twine check python/dist/*.whl
twine upload --repository testpypi python/dist/*.whl

# Install what TestPyPI served, in a throwaway environment, from a directory outside this
# repository so no development dylib can be found:
python -m venv /tmp/am-testpypi && /tmp/am-testpypi/bin/pip install \
    --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple arrowmetal
cd /tmp && /tmp/am-testpypi/bin/python -c "import arrowmetal as am; print(am.__version__, am.device_name())"
```

Only when TestPyPI installs and imports cleanly does the real upload happen (step 4).

## 2. Freeze the changelog

`CHANGELOG.md` opens with `## X.Y.Z`. Confirm nothing above that heading names a later version, leave
the content alone, and commit if anything changed. No date: the repository does not date its entries.

`python/README.md` is the PyPI long description (`readme = "README.md"` in `python/pyproject.toml`):
its links must be absolute URLs, because PyPI does not rewrite relative ones.

## 3. Tag the Swift package

Swift Package Manager resolves versions from git tags, so the tag *is* the Swift release. Tag the commit
from step 2, annotated, on `main`; the Go module takes its own tag.

```
git tag -a vX.Y.Z -m "ArrowMetal X.Y.Z"
git tag -a go/arrowmetal/vX.Y.Z -m "ArrowMetal Go binding X.Y.Z"
git push origin main
git push origin vX.Y.Z go/arrowmetal/vX.Y.Z
```

Verify a consumer can resolve it, from a scratch directory outside this repository:

```
mkdir /tmp/am-consumer && cd /tmp/am-consumer && swift package init --type executable
# add .package(url: "<repository URL>", from: "X.Y.Z") to Package.swift, then
swift package resolve
```

## 4. Build and upload the wheel

The wheel is built from the tagged tree, not from a dirty working copy.

```
git switch --detach vX.Y.Z
scripts/build_wheel.sh                  # swift build -c release, then python/build_wheel.sh
```

This produces `python/dist/arrowmetal-X.Y.Z-py3-none-macosx_14_0_arm64.whl` with
`libArrowMetalC.dylib` and the Polars plugin `libarrowmetal_polars.dylib` bundled at
`arrowmetal/_lib/`. Check it before uploading:

```
twine check python/dist/*.whl
unzip -l python/dist/*.whl | grep _lib          # both dylibs must be in the archive
PYTHON=python3.13 scripts/check_wheel.sh        # fresh virtualenv + Polars from PyPI: all four tiers
python -m venv /tmp/am-wheel && /tmp/am-wheel/bin/pip install python/dist/*.whl
cd /tmp && /tmp/am-wheel/bin/python -c "import arrowmetal as am; print(am.device_name())"
cd - && /tmp/am-wheel/bin/pip install "$(echo python/dist/arrowmetal-X.Y.Z-*.whl)[polars,duckdb,pandas]"
```

Then upload:

```
twine upload python/dist/arrowmetal-X.Y.Z-py3-none-macosx_14_0_arm64.whl
```

There is no source distribution (`sdist`). An sdist would promise a build from source on the installing
machine, which needs a Swift toolchain and a Metal device; the wheel is macOS 14+ arm64 only and says so
in its platform tag.

Finally, confirm the published artefact from a clean environment:

```
python -m venv /tmp/am-pypi && /tmp/am-pypi/bin/pip install arrowmetal
cd /tmp && /tmp/am-pypi/bin/python -c "import arrowmetal as am; print(am.__version__, am.device_name())"
```

## 5. The GitHub release

Cut a release from the `vX.Y.Z` tag, with the changelog section as the body and the wheel attached, so
the wheel is installable without PyPI.

```
gh release create vX.Y.Z python/dist/arrowmetal-X.Y.Z-py3-none-macosx_14_0_arm64.whl \
    --title "ArrowMetal X.Y.Z" --notes-file <(sed -n '/^## X.Y.Z/,/^## /p' CHANGELOG.md)
```

The README badges point at CI on `main`, PyPI and pkg.go.dev; dispatch `ci.yml` once on the release
commit (`gh workflow run ci.yml --ref main`).

## 6. The crates

`arrowmetal-sys` and `arrowmetal` are published from the `vX.Y.Z` tag in that order
(`cargo publish -p arrowmetal-sys`, then `-p arrowmetal`, each with `ARROWMETAL_LIB` set so the
packaging build finds the dylib; docs.rs builds skip the search). The crates link the dylib the user
already has, from the wheel or a Swift build, and [RUST.md](RUST.md) says how a binary carries the
run-time path. Checked after publication: a scratch crate outside the repository depending on
`arrowmetal = "X.Y.Z"` and pointed at the wheel's dylib prints the version and the device name.

`polars-plugin/` is a Rust `cdylib` that Polars loads by path; it carries its own copy of
`arrowmetal-sys` under the same name, so it stays `publish = false` on crates.io. The wheel build
compiles it and ships it inside the wheel, at `arrowmetal/_lib/libarrowmetal_polars.dylib`. The DuckDB extension (`duckdb-extension/`) is built from
the repository and is not distributed through the DuckDB community extensions repository. The Node
package (`node/`) is `"private": true` and is not published to npm: the addon links the dylib built
from this checkout.

## 7. The site

The site is published to GitHub Pages from the `gh-pages` branch. Its figures are checked against the
documents they cite (the benchmark matrix, the coverage tables, the to-improve list) before it goes live.

## 8. Numbers to re-run before announcing

Benchmark tables name the machine they were measured on, and the README quotes them. Re-run on the
release machine and update `docs/BENCHMARKS.md`, `docs/BENCHMARKS_MATRIX.md` and `docs/TO_IMPROVE.md` if any
headline moved:

```
swift run -c release arrowmetal-bench
PYTHONPATH=python python Benchmarks/python_gpu_bench.py
PYTHONPATH=python python Benchmarks/engine_bench.py
```

Never publish numbers from a GitHub-hosted runner: its GPU is virtual.

## 9. After the release

Watch the PyPI project page for the platform tag: `pip install arrowmetal` on Intel macOS, Linux or
Windows must fail with "no matching distribution", not install something that cannot import.
