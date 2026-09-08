# Releasing 0.1.0

The ordered checklist for turning this checkout into published artefacts. Nothing here has been done
yet: 0.1.0 is unreleased, there is no PyPI package, no crates.io crate and no tagged Swift version.
Every step is the maintainer's to run; the repository itself never publishes anything.

**The version number stays `0.1.0` throughout.** It is not bumped by any step below, and it is not
bumped after the release either — the next number is a decision, not a formality.

---

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

Confirm the version is stated in exactly one place and reads `0.1.0`:

```
grep -rn '__version__' python/arrowmetal/__init__.py       # the single source
grep -n 'version' python/pyproject.toml                    # must be `dynamic`, reading the above
grep -rn '0\.1\.0' README.md CHANGELOG.md docs/*.md | head # narrative mentions only
```

## 1. Reserve the name on PyPI, before anything else

The project name is the one irreversible thing in the release. Claim `arrowmetal` on PyPI **first**, so
that no one else takes it between the tag and the upload.

```
python -m pip install --upgrade build twine

# 1a. A PyPI account with 2FA, and an API token scoped to "entire account" for the first upload.
#     Store it as a keyring entry or in ~/.pypirc (never in this repository).

# 1b. Rehearse the whole upload on TestPyPI, which is a separate namespace and separate account:
python/build_wheel.sh
twine check python/dist/*.whl
twine upload --repository testpypi python/dist/*.whl

# 1c. Install what TestPyPI actually served, in a throwaway environment, and smoke test it
#     from a directory outside this repository so no development dylib can be found:
python -m venv /tmp/am-testpypi && /tmp/am-testpypi/bin/pip install \
    --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple arrowmetal
cd /tmp && /tmp/am-testpypi/bin/python -c "import arrowmetal as am; print(am.__version__, am.device_name())"
```

Only when TestPyPI installs and imports cleanly does the real upload happen (step 4). After the first
successful PyPI upload, replace the account-wide token with a project-scoped one.

## 2. Freeze the changelog

`CHANGELOG.md` opens with `## 0.1.0 (unreleased, in development)`. Drop the parenthetical, leave the
content alone, and commit. No date: the repository does not date its entries.

```
git add CHANGELOG.md && git commit -m "0.1.0"
```

## 3. Tag the Swift package

Swift Package Manager resolves versions from git tags, so the tag *is* the Swift release. Tag the commit
from step 2, annotated, on `main`.

```
git tag -a v0.1.0 -m "ArrowMetal 0.1.0"
git push origin main
git push origin v0.1.0
```

Verify a consumer can resolve it, from a scratch directory outside this repository:

```
mkdir /tmp/am-consumer && cd /tmp/am-consumer && swift package init --type executable
# add .package(url: "<repository URL>", from: "0.1.0") to Package.swift, then
swift package resolve
```

## 4. Build and upload the wheel

The wheel must be built from the tagged tree, not from a dirty working copy.

```
git switch --detach v0.1.0
scripts/build_wheel.sh                  # swift build -c release, then python/build_wheel.sh
```

This produces `python/dist/arrowmetal-0.1.0-py3-none-macosx_14_0_arm64.whl` (the script prints its size) with
`libArrowMetalC.dylib` bundled at `arrowmetal/_lib/`. Check it before uploading:

```
twine check python/dist/*.whl
unzip -l python/dist/*.whl | grep _lib          # the dylib must be in the archive
python -m venv /tmp/am-wheel && /tmp/am-wheel/bin/pip install python/dist/*.whl
cd /tmp && /tmp/am-wheel/bin/python -c "import arrowmetal as am; print(am.device_name())"
cd - && /tmp/am-wheel/bin/pip install "$(echo python/dist/arrowmetal-0.1.0-*.whl)[polars,duckdb,pandas]"
```

Then upload:

```
twine upload python/dist/arrowmetal-0.1.0-py3-none-macosx_14_0_arm64.whl
```

There is no source distribution (`sdist`). An sdist would promise a build from source on the installing
machine, which needs a Swift toolchain and a Metal device; the wheel is macOS 14+ arm64 only and says so
in its platform tag. If an sdist is ever added it must fail loudly on any other platform.

Finally, confirm the published artefact from a clean environment:

```
python -m venv /tmp/am-pypi && /tmp/am-pypi/bin/pip install arrowmetal
cd /tmp && /tmp/am-pypi/bin/python -c "import arrowmetal as am; print(am.__version__, am.device_name())"
```

## 5. The GitHub release

Cut a release from the `v0.1.0` tag, with the changelog section as the body and the wheel attached, so
the wheel is installable without PyPI.

```
gh release create v0.1.0 python/dist/arrowmetal-0.1.0-py3-none-macosx_14_0_arm64.whl \
    --title "ArrowMetal 0.1.0" --notes-file <(sed -n '/^## 0.1.0/,/^## /p' CHANGELOG.md)
```

## 6. The Polars plugin crate — after the release, not during it

`polars-plugin/` is a Rust `cdylib` whose `build.rs` links `libArrowMetalC.dylib` by rpath from a local
Swift build. crates.io publication is therefore **deferred**: a crate downloaded from crates.io has no
Swift checkout to link against, so it needs either a vendored prebuilt dylib or a `build.rs` that fetches
one, and neither exists yet. Until then the plugin is built from this repository:

```
cd polars-plugin && cargo build --release
```

Two things to settle before any crate is published. There are two crates named `arrowmetal-sys`
in this repository, `rust/arrowmetal-sys` (the binding's, 296 lines of declarations) and
`polars-plugin/arrowmetal-sys` (the plugin's, 718); crates.io can hold one crate of that name, so
the plugin must depend on the binding's crate, or its copy must be renamed, before either goes
up. And every crate carries `publish = false` today, which `cargo publish` refuses; flip it only on
the crate being published.

The Node package (`node/`) is `"private": true` and is not published to npm at 0.1.0 for the same
reason as the crates: the addon links a dylib built from this checkout. `npm pack --dry-run` lists
what a later publication would carry.

When it is ready, the order is `cargo publish --dry-run`, then reserving the crate name, then
`cargo publish` — and the crate version tracks the dylib ABI it was built against, so it cannot be
published ahead of a tagged 0.1.0.

The DuckDB extension (`duckdb-extension/`) is in the same position: it is built from the repository and
is not distributed through the DuckDB community extensions repository yet.

## 7. The GitHub organisation rename

The repository lives on a personal account today and is meant to move to an organisation of the project's
own. Do this **after** the tag and the PyPI upload, never between them — a rename mid-release breaks the
URLs the release refers to.

1. Create the organisation, transfer the repository into it. GitHub keeps redirects from the old path, so
   existing clones and `swift package resolve` keep working; do not delete or re-create the old repository.
2. Update every URL that names the old path, then commit:
   ```
   grep -rn 'github.com' README.md CHANGELOG.md CONTRIBUTING.md docs/ python/ Package.swift \
       polars-plugin/ duckdb-extension/ rust/ go/ node/ r/ .github/ | grep -v Binary
   ```
   `python/pyproject.toml` carries `[project.urls] Homepage` and `Source`; a changed URL there needs a
   re-upload to be visible on PyPI, so prefer settling the organisation name *before* step 4 if possible.
3. Re-check the PyPI project's links and the GitHub release page after the move.

## 8. Re-verify the site's hand-typed figures

The marketing site is built from a working directory that is deliberately git-ignored, so none of it is in
this repository. Its facts file separates figures the build script computes from the repository on every
build from figures typed by hand; the build script prints the hand-typed ones as a **RELEASE CHECK** list
precisely so that they get re-verified at a release. Run the site build, read that list, and confirm every
figure on it against the document it cites — the benchmark matrix, the coverage tables, the to-improve list —
before the site goes live. Any figure that no longer matches gets corrected in the facts file, never in the
page templates.

The same applies to the comparison table of other projects: it is sourced from each project's public docs
and goes stale on their schedule, not ours.

## 8b. Publish the website

The site ships as one static page plus its icon set. It is published to GitHub Pages from the `gh-pages`
branch by a script kept outside the repository (`private/tools/publish_pages.sh`), which rebuilds both site
outputs from the current sources, refuses to run on a dirty `main` or on a page containing a personal
address, adds the `CNAME` for the domain and force-pushes the standalone build. Run it after step 8, and
once only per release state:

```
private/tools/publish_pages.sh arrowmetal.org
```

The first time, set the repository's Pages settings by hand: source "Deploy from a branch", branch
`gh-pages`, folder `/`, custom domain `arrowmetal.org`, "Enforce HTTPS" on; point the domain's DNS at
GitHub Pages (four A records to `185.199.108.153` … `.111.153`, and `www` as a CNAME to the account's
`github.io` host). Then open the published page and check, in this order: the tab icon is the ArrowMetal
mark; every tab renders on first arrival; the Compare table fits at a normal window width; a doc page
shows its "On this page" list; dark mode shows amber links; the footer links resolve now that the
repository is public. The claude.ai preview is a separate copy and is never the deployed page.

## 9. Numbers to re-run before announcing

Benchmark tables name the machine they were measured on, and the README quotes them. Re-run on the
release machine and update `docs/BENCHMARKS.md`, `docs/BENCHMARKS_MATRIX.md` and `docs/TO_IMPROVE.md` if any
headline moved:

```
swift run -c release arrowmetal-bench
PYTHONPATH=python python Benchmarks/python_gpu_bench.py
PYTHONPATH=python python Benchmarks/engine_bench.py
```

Never publish numbers from a GitHub-hosted runner: its GPU is virtual.

## 10. After the release

- Leave the version at `0.1.0`. The next version number is chosen when there is something to put in it.
- Reopen `CHANGELOG.md` with a new unreleased section only when the first post-release change lands.
- Watch the PyPI project page for the platform tag: `pip install arrowmetal` on Intel macOS, Linux or
  Windows must fail with "no matching distribution", not install something that cannot import.
