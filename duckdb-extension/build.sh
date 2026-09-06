#!/usr/bin/env bash
# Build duckdb-extension/build/arrowmetal.duckdb_extension.
#
#   ./duckdb-extension/build.sh                 # C API v1.2.0, osx_arm64, whatever python3 is first
#   DUCKDB_PLATFORM=osx_amd64 ./build.sh        # cross-target the platform string
#   PYTHON=/path/to/venv/bin/python ./build.sh  # use a specific interpreter to read the duckdb version
#
# Two prerequisites, and no others:
#
#   1. libArrowMetalC.dylib, from the repository root:
#        DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
#          swift build -c release --product ArrowMetalC
#   2. DuckDB's C headers, duckdb.h and duckdb_extension.h. The script downloads them into
#      third_party/ on first run (they are the ones shipped in the libduckdb release archive) and
#      reuses them afterwards, so only the first build needs the network.
#
# cmake is optional. CMakeLists.txt drives exactly the same compile for people who want it; this
# script exists because a C-API extension is one translation unit and one link, and requiring a
# build system for that is not worth anyone's afternoon.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="$HERE/build"
THIRD_PARTY="$HERE/third_party"

PYTHON="${PYTHON:-python3}"
DUCKDB_PLATFORM="${DUCKDB_PLATFORM:-}"
# The C extension API version the extension declares, NOT the DuckDB version. A C_STRUCT extension
# built against v1.2.0 loads into every DuckDB whose C API is at least that; see docs/DUCKDB.md.
DUCKDB_C_API_VERSION="${DUCKDB_C_API_VERSION:-v1.2.0}"
# The DuckDB release whose headers are fetched. Only the headers come from here; the extension never
# links against libduckdb.
DUCKDB_HEADER_VERSION="${DUCKDB_HEADER_VERSION:-v1.5.5}"
EXTENSION_VERSION="${EXTENSION_VERSION:-0.1.0}"

ARROWMETAL_LIB="${ARROWMETAL_LIB:-$ROOT/.build/release/libArrowMetalC.dylib}"

if [ -z "$DUCKDB_PLATFORM" ]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) DUCKDB_PLATFORM=osx_arm64 ;;
    Darwin-x86_64) DUCKDB_PLATFORM=osx_amd64 ;;
    *) echo "set DUCKDB_PLATFORM explicitly for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
  esac
fi

if [ ! -f "$ARROWMETAL_LIB" ]; then
  echo "libArrowMetalC.dylib not found at $ARROWMETAL_LIB" >&2
  echo "build it first: DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \\" >&2
  echo "  swift build -c release --product ArrowMetalC" >&2
  exit 1
fi

mkdir -p "$THIRD_PARTY" "$BUILD"
for header in duckdb.h duckdb_extension.h; do
  if [ ! -f "$THIRD_PARTY/$header" ]; then
    url="https://raw.githubusercontent.com/duckdb/duckdb/$DUCKDB_HEADER_VERSION/src/include/$header"
    echo "fetching $header from DuckDB $DUCKDB_HEADER_VERSION"
    if ! curl -sSfL --max-time 120 -o "$THIRD_PARTY/$header" "$url"; then
      rm -f "$THIRD_PARTY/$header"
      echo "could not fetch $header." >&2
      echo "Download libduckdb-osx-universal.zip from" >&2
      echo "  https://github.com/duckdb/duckdb/releases/tag/$DUCKDB_HEADER_VERSION" >&2
      echo "and put duckdb.h and duckdb_extension.h in $THIRD_PARTY/" >&2
      exit 1
    fi
  fi
done

echo "compiling arrowmetal_extension.cpp"
clang++ -std=c++17 -O2 -fPIC -fvisibility=hidden -Wall \
  -I"$THIRD_PARTY" -I"$ROOT/include" \
  -shared -o "$BUILD/libarrowmetal_extension.dylib" \
  "$HERE/src/arrowmetal_extension.cpp" \
  "$ARROWMETAL_LIB" \
  -Wl,-rpath,"$(cd "$(dirname "$ARROWMETAL_LIB")" && pwd)"

"$PYTHON" "$HERE/scripts/append_metadata.py" \
  -l "$BUILD/libarrowmetal_extension.dylib" \
  -o "$BUILD/arrowmetal.duckdb_extension" \
  -p "$DUCKDB_PLATFORM" \
  -dv "$DUCKDB_C_API_VERSION" \
  -ev "$EXTENSION_VERSION"

cat <<EOF

Built $BUILD/arrowmetal.duckdb_extension

Load it (unsigned extensions have to be allowed explicitly):

  import duckdb
  con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
  con.execute("LOAD '$BUILD/arrowmetal.duckdb_extension'")
  con.sql("SELECT arrowmetal_device()").show()
EOF
