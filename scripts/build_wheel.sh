#!/usr/bin/env bash
# Build the arrowmetal Python wheel: compile the C ABI dylib, bundle it into the package, build the wheel.
#
# Requirements:
#   - macOS on Apple silicon (arm64). The wheel embeds a Metal-linked arm64 dylib and runs nowhere else.
#   - Swift toolchain (Command Line Tools are enough; shaders compile at runtime).
#   - The Python build frontend:  pip install build
#
# Usage:
#   scripts/build_wheel.sh              # wheel into python/dist/
#   PYTHON=python3.12 scripts/build_wheel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DYLIB=".build/release/libArrowMetalC.dylib"
DEST="$ROOT/python/arrowmetal"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: arrowmetal builds only on macOS arm64 (found $(uname -s) $(uname -m))" >&2
    exit 1
fi

echo "==> swift build -c release --product ArrowMetalC"
cd "$ROOT"
swift build -c release --product ArrowMetalC

if [[ ! -f "$ROOT/$DYLIB" ]]; then
    echo "error: $DYLIB not found after the build" >&2
    exit 1
fi

echo "==> bundling $DYLIB into python/arrowmetal/"
mkdir -p "$DEST"
cp "$ROOT/$DYLIB" "$DEST/libArrowMetalC.dylib"

if ! "$PYTHON" -c "import build" >/dev/null 2>&1; then
    echo "error: the 'build' frontend is missing. Install it with: $PYTHON -m pip install build" >&2
    exit 1
fi

# The wheel is not pure Python (it carries an arm64 dylib), so tag it for macOS arm64
# rather than letting setuptools emit py3-none-any.
MACOS_VERSION="$("$PYTHON" -c 'import platform; v = platform.mac_ver()[0].split("."); print(f"{v[0]}_0" if int(v[0]) >= 11 else f"{v[0]}_{v[1]}")')"
PLAT_TAG="macosx_${MACOS_VERSION}_arm64"

echo "==> python -m build --wheel (plat tag $PLAT_TAG)"
cd "$ROOT/python"
"$PYTHON" -m build --wheel --config-setting=--build-option=--plat-name="$PLAT_TAG"

echo
echo "wheel(s) in $ROOT/python/dist:"
ls -1 "$ROOT/python/dist"/*.whl
