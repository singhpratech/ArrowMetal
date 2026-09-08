#!/usr/bin/env bash
# Build the `arrowmetal` wheel from an already-built dylib.
#
# It bundles .build/release/libArrowMetalC.dylib into arrowmetal/_lib/ and builds a macOS
# arm64 wheel, so that `pip install` needs no Swift toolchain and no build step.
#
# Requirements:
#   - macOS on Apple silicon (arm64). The wheel embeds a Metal-linked arm64 dylib and runs nowhere else.
#   - .build/release/libArrowMetalC.dylib, from `swift build -c release --product ArrowMetalC`.
#     This script does not build it (use scripts/build_wheel.sh to do both).
#   - The wheel build frontend:  python3 -m pip install build
#
# Usage:
#   python/build_wheel.sh                                  # -> python/dist/arrowmetal-<version>-*.whl
#   PYTHON=python3.12 python/build_wheel.sh
#   ARROWMETAL_DYLIB=/path/to/libArrowMetalC.dylib python/build_wheel.sh
#   PLAT_TAG=macosx_15_0_arm64 python/build_wheel.sh
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

# The dylib has to be inside the wheel, or the install is not self-contained.
if ! "$PYTHON" -c "import zipfile,sys; sys.exit(0 if any(n.endswith('arrowmetal/_lib/libArrowMetalC.dylib') for n in zipfile.ZipFile(sys.argv[1]).namelist()) else 1)" "$WHEEL"; then
    echo "error: $WHEEL does not contain arrowmetal/_lib/libArrowMetalC.dylib" >&2
    exit 1
fi

echo
echo "wheel: $WHEEL"
echo "size:  $(/usr/bin/du -h "$WHEEL" | cut -f1) ($(/usr/bin/stat -f%z "$WHEEL") bytes)"
echo
echo "install it with:  $PYTHON -m pip install '$WHEEL'"
