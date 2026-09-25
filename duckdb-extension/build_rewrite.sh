#!/usr/bin/env bash
# Build duckdb-extension/build/arrowmetal_rewrite.duckdb_extension: the optimizer extension that runs
# eligible aggregates of ordinary SQL on the GPU (docs/DUCKDB.md §4b).
#
#   ./duckdb-extension/build_rewrite.sh
#   PYTHON=/path/to/venv/bin/python ./duckdb-extension/build_rewrite.sh   # the Python whose duckdb it targets
#
# Unlike build.sh, this one needs DuckDB's C++ headers: an optimizer hook exists only in the C++ API.
# The script shallow-clones DuckDB at the release tag into build/duckdb-<version>/ on first run (about
# 60 MB, headers only are used; nothing of DuckDB is compiled) and reuses it afterwards.
#
# A C++ extension is tied to one DuckDB release: the metadata footer names it, and DuckDB refuses the
# file in any other version. The release is read from the target Python's duckdb module, and the
# source tree's commit is checked against that module's source_id so the headers match the binary.
#
# The source is written to the DuckDB 1.5 C++ API. The C++ API changes between releases: against the
# 1.4.5 headers it does not compile (OptimizerExtension::Register and PhysicalOperator::GetDataInternal
# are not there yet), so another release may need changes to src/arrowmetal_rewrite.cpp.
#
# The extension leaves DuckDB's symbols unresolved (-undefined dynamic_lookup); the host process - the
# duckdb Python module, or the duckdb CLI - provides them when it loads the file. The last step checks
# that every DuckDB symbol the extension needs is exported by the target Python's duckdb module.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="$HERE/build"
PYTHON="${PYTHON:-python3}"
EXTENSION_VERSION="${EXTENSION_VERSION:-0.2.0}"
ARROWMETAL_LIB="${ARROWMETAL_LIB:-$ROOT/.build/release/libArrowMetalC.dylib}"

if [ ! -f "$ARROWMETAL_LIB" ]; then
  echo "libArrowMetalC.dylib not found at $ARROWMETAL_LIB" >&2
  echo "build it first: DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \\" >&2
  echo "  swift build -c release --product ArrowMetalC" >&2
  exit 1
fi

read -r DUCKDB_VERSION DUCKDB_SOURCE_ID DUCKDB_PLATFORM_DETECTED DUCKDB_MODULE < <("$PYTHON" -c '
import duckdb, importlib
v, sid, _ = duckdb.sql("pragma version").fetchone()
plat = duckdb.sql("pragma platform").fetchone()[0]
mod = importlib.import_module("_duckdb").__file__
print(v, sid, plat, mod)')
if [ -z "${DUCKDB_MODULE:-}" ]; then
  echo "$PYTHON could not import duckdb; point PYTHON at the interpreter whose duckdb the extension is for" >&2
  exit 1
fi
DUCKDB_PLATFORM="${DUCKDB_PLATFORM:-$DUCKDB_PLATFORM_DETECTED}"
SOURCE="${DUCKDB_SOURCE_DIR:-$BUILD/duckdb-$DUCKDB_VERSION}"

if [ ! -f "$SOURCE/src/include/duckdb.hpp" ]; then
  echo "fetching DuckDB $DUCKDB_VERSION source into $SOURCE"
  mkdir -p "$BUILD"
  git clone --quiet --depth 1 --branch "$DUCKDB_VERSION" https://github.com/duckdb/duckdb.git "$SOURCE"
fi
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
case "$SOURCE_COMMIT" in
  "$DUCKDB_SOURCE_ID"*) ;;
  *) echo "DuckDB source at $SOURCE is commit $SOURCE_COMMIT, but the target duckdb is $DUCKDB_SOURCE_ID" >&2
     exit 1 ;;
esac

mkdir -p "$BUILD"
echo "compiling arrowmetal_rewrite.cpp against DuckDB $DUCKDB_VERSION ($DUCKDB_SOURCE_ID)"
clang++ -std=c++17 -O2 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -Wall -Wno-unused-function \
  -DNDEBUG -DDUCKDB_BUILD_LOADABLE_EXTENSION \
  -I"$SOURCE/src/include" -I"$ROOT/include" \
  -shared -undefined dynamic_lookup \
  -o "$BUILD/libarrowmetal_rewrite.dylib" \
  "$HERE/src/arrowmetal_rewrite.cpp" \
  "$ARROWMETAL_LIB" \
  -Wl,-rpath,"$(cd "$(dirname "$ARROWMETAL_LIB")" && pwd)"

# Every duckdb:: symbol left unresolved has to be exported by the host, or the first call to it would
# abort the process instead of failing the LOAD. Check against the target Python's duckdb module.
missing="$(comm -23 \
  <(nm -u "$BUILD/libarrowmetal_rewrite.dylib" | grep '__ZN6duckdb\|__ZNK6duckdb\|__ZTIN6duckdb\|__ZTVN6duckdb' | sort -u) \
  <(nm -gU "$DUCKDB_MODULE" | awk '{print $3}' | sort -u))"
if [ -n "$missing" ]; then
  echo "the extension needs DuckDB symbols that $DUCKDB_MODULE does not export:" >&2
  echo "$missing" | c++filt >&2
  exit 1
fi

"$PYTHON" "$HERE/scripts/append_metadata.py" \
  -l "$BUILD/libarrowmetal_rewrite.dylib" \
  -o "$BUILD/arrowmetal_rewrite.duckdb_extension" \
  -p "$DUCKDB_PLATFORM" \
  -dv "$DUCKDB_VERSION" \
  -ev "$EXTENSION_VERSION" \
  --abi-type CPP

cat <<EOF

Built $BUILD/arrowmetal_rewrite.duckdb_extension for DuckDB $DUCKDB_VERSION ($DUCKDB_PLATFORM)

  import duckdb
  con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
  con.execute("LOAD '$BUILD/arrowmetal_rewrite.duckdb_extension'")
  con.sql("EXPLAIN SELECT k, sum(v) FROM t GROUP BY k").show()   # ARROWMETAL_AGGREGATE when rewritten
EOF
