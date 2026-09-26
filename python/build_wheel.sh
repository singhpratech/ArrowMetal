#!/usr/bin/env bash
# Build the `arrowmetal` wheel from an already-built dylib.
#
# It bundles .build/release/libArrowMetalC.dylib into arrowmetal/_lib/, builds the Polars expression
# plugin (polars-plugin/, tier 2) with cargo against that same dylib and bundles it beside it as
# arrowmetal/_lib/libarrowmetal_polars.dylib, and builds a macOS arm64 wheel, so that `pip install`
# needs no Swift toolchain, no Rust toolchain and no build step.
#
# Requirements:
#   - macOS on Apple silicon (arm64). The wheel embeds Metal-linked arm64 dylibs and runs nowhere else.
#   - .build/release/libArrowMetalC.dylib, from `swift build -c release --product ArrowMetalC`.
#     This script does not build it (use scripts/build_wheel.sh to do both).
#   - cargo (https://rustup.rs), for the Polars plugin. ARROWMETAL_POLARS_PLUGIN_DYLIB skips the build.
#   - The wheel build frontend:  python3 -m pip install build
#
# Usage:
#   python/build_wheel.sh                                  # -> python/dist/arrowmetal-<version>-*.whl
#   PYTHON=python3.12 python/build_wheel.sh
#   ARROWMETAL_DYLIB=/path/to/libArrowMetalC.dylib python/build_wheel.sh
#   ARROWMETAL_POLARS_PLUGIN_DYLIB=/path/to/libarrowmetal_polars.dylib python/build_wheel.sh
#   PLAT_TAG=macosx_15_0_arm64 python/build_wheel.sh
#
# scripts/check_wheel.sh then installs the wheel into a fresh virtualenv and runs all four Polars tiers.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-python3}"
# The deployment target the wheel claims. macOS 14 is the floor the Metal kernels are built against;
# pip installs it on 14 and every later release.
PLAT_TAG="${PLAT_TAG:-macosx_14_0_arm64}"
SRC="${ARROWMETAL_DYLIB:-$ROOT/.build/release/libArrowMetalC.dylib}"
DEST="$HERE/arrowmetal/_lib"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: arrowmetal builds only on macOS arm64 (found $(uname -s) $(uname -m))" >&2
    exit 1
fi

if [[ ! -f "$SRC" ]]; then
    echo "error: $SRC not found." >&2
    echo "       Build it first:  swift build -c release --product ArrowMetalC" >&2
    echo "       Or point ARROWMETAL_DYLIB at an existing libArrowMetalC.dylib." >&2
    exit 1
fi

PLUGIN_SRC="${ARROWMETAL_POLARS_PLUGIN_DYLIB:-}"
if [[ -z "$PLUGIN_SRC" ]] && ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found; the wheel carries the Polars plugin, which is built with cargo." >&2
    echo "       Install Rust (https://rustup.rs), or point ARROWMETAL_POLARS_PLUGIN_DYLIB at a built" >&2
    echo "       libarrowmetal_polars.dylib." >&2
    exit 1
fi

if ! "$PYTHON" -c "import build" >/dev/null 2>&1; then
    echo "error: the 'build' frontend is missing. Install it with: $PYTHON -m pip install build" >&2
    exit 1
fi

echo "==> bundling $(basename "$SRC") into python/arrowmetal/_lib/"
mkdir -p "$DEST"
cp "$SRC" "$DEST/libArrowMetalC.dylib"

# SwiftPM bakes the build machine's toolchain directory in as an LC_RPATH entry. Nothing the dylib
# links is resolved through it (the Swift runtime is linked by absolute /usr/lib/swift paths), so
# drop every rpath under /Applications or /Users and re-sign, or the wheel carries a path from the
# machine it was built on. Editing the load commands invalidates the signature, and an arm64 dylib
# with a broken signature does not load, so the ad-hoc re-sign is not optional.
for rp in $(otool -l "$DEST/libArrowMetalC.dylib" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}' | grep -E '^/(Applications|Users)/' || true); do
    install_name_tool -delete_rpath "$rp" "$DEST/libArrowMetalC.dylib"
done
codesign --force --sign - "$DEST/libArrowMetalC.dylib" 2>/dev/null

# The Polars expression plugin (tier 2), linked against the very dylib bundled above: arrowmetal-sys's
# build.rs takes the directory from ARROWMETAL_LIB_DIR.
if [[ -z "$PLUGIN_SRC" ]]; then
    echo "==> cargo build --release  (polars-plugin, linked against $SRC)"
    ( cd "$ROOT/polars-plugin" && ARROWMETAL_LIB_DIR="$(dirname "$SRC")" cargo build --release )
    PLUGIN_SRC="$ROOT/polars-plugin/target/release/libarrowmetal_polars.dylib"
fi
if [[ ! -f "$PLUGIN_SRC" ]]; then
    echo "error: $PLUGIN_SRC not found" >&2
    exit 1
fi
echo "==> bundling $(basename "$PLUGIN_SRC") into python/arrowmetal/_lib/"
PLUGIN="$DEST/libarrowmetal_polars.dylib"
cp "$PLUGIN_SRC" "$PLUGIN"
chmod u+w "$PLUGIN"
# cargo bakes the directory it linked libArrowMetalC.dylib from in as an LC_RPATH entry. In the wheel
# the plugin sits beside the bundled copy, so every rpath goes and `@loader_path` takes their place:
# `@rpath/libArrowMetalC.dylib` then resolves to arrowmetal/_lib/libArrowMetalC.dylib, the same file
# Python's ctypes loads, and no DYLD_LIBRARY_PATH is needed.
for rp in $(otool -l "$PLUGIN" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}'); do
    install_name_tool -delete_rpath "$rp" "$PLUGIN"
done
install_name_tool -add_rpath @loader_path "$PLUGIN"
install_name_tool -id @rpath/libarrowmetal_polars.dylib "$PLUGIN"
# Local symbols only: the exported entry points Polars looks up by name stay.
strip -x "$PLUGIN" 2>/dev/null
codesign --force --sign - "$PLUGIN" 2>/dev/null
# The load commands must now name nothing on this machine: libArrowMetalC through @rpath, the rest
# from the system (/usr/lib, /System). CPython symbols are resolved from the interpreter at dlopen.
if ! otool -L "$PLUGIN" | tail -n +2 | awk '{print $1}' | grep -qx '@rpath/libArrowMetalC.dylib'; then
    echo "error: $PLUGIN does not link @rpath/libArrowMetalC.dylib" >&2
    exit 1
fi
if otool -L "$PLUGIN" | tail -n +2 | awk '{print $1}' | grep -vE '^(@rpath/|/usr/lib/|/System/)'; then
    echo "error: $PLUGIN links a library outside /usr/lib and /System (listed above)" >&2
    exit 1
fi
if [[ "$(otool -l "$PLUGIN" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}')" != "@loader_path" ]]; then
    echo "error: $PLUGIN rpaths are not exactly @loader_path" >&2
    exit 1
fi

# A stale copy from the pre-0.1 layout would ship twice and shadow the new one.
rm -f "$HERE/arrowmetal/libArrowMetalC.dylib"
rm -rf "$HERE/build" "$HERE/dist"

echo "==> $PYTHON -m build --wheel  (platform tag $PLAT_TAG)"
cd "$HERE"
"$PYTHON" -m build --wheel --config-setting=--build-option=--plat-name="$PLAT_TAG"

WHEEL="$(ls -t "$HERE/dist"/*.whl 2>/dev/null | head -1)"
if [[ -z "$WHEEL" ]]; then
    echo "error: no wheel produced in python/dist" >&2
    exit 1
fi

# Both dylibs have to be inside the wheel, or the install is not self-contained.
for lib in libArrowMetalC.dylib libarrowmetal_polars.dylib; do
    if ! "$PYTHON" -c "import zipfile,sys; sys.exit(0 if any(n == 'arrowmetal/_lib/' + sys.argv[2] for n in zipfile.ZipFile(sys.argv[1]).namelist()) else 1)" "$WHEEL" "$lib"; then
        echo "error: $WHEEL does not contain arrowmetal/_lib/$lib" >&2
        exit 1
    fi
done

echo
echo "wheel: $WHEEL"
echo "size:  $(/usr/bin/du -h "$WHEEL" | cut -f1) ($(/usr/bin/stat -f%z "$WHEEL") bytes)"
echo
echo "install it with:  $PYTHON -m pip install '$WHEEL'"
echo "check it with:    scripts/check_wheel.sh '$WHEEL'"
