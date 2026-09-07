#!/usr/bin/env bash
# Build the arrowmetal Python wheel end to end: compile the C ABI dylib, then bundle and package it.
#
# This is the two-step convenience wrapper. The packaging half lives in python/build_wheel.sh and
# can be run on its own when .build/release/libArrowMetalC.dylib already exists.
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
DYLIB=".build/release/libArrowMetalC.dylib"

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

exec "$ROOT/python/build_wheel.sh"
