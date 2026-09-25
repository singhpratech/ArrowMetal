#!/usr/bin/env bash
# Install a built wheel into a fresh virtualenv and run all four Polars tiers, and
# `python -m arrowmetal.bench --parquet`, from it.
#
# The check the release runs before any upload (docs/RELEASE.md step 4): the wheel alone, with Polars
# from PyPI, must give every tier. The virtualenv lives in a new temporary directory outside this
# repository and the checks run from there with a scrubbed environment (no PYTHONPATH,
# ARROWMETAL_LIB, ARROWMETAL_POLARS_PLUGIN, DYLD_* or DEVELOPER_DIR, and a PATH without cargo), so
# no development build and no toolchain can be reached. It also asserts that the process loaded
# exactly one libArrowMetalC.dylib, the packaged one, which is the library the packaged plugin
# resolves through its @loader_path rpath.
#
# Usage:
#   scripts/check_wheel.sh                                   # the newest python/dist/arrowmetal-*.whl
#   scripts/check_wheel.sh python/dist/arrowmetal-X.Y.Z-py3-none-macosx_14_0_arm64.whl
#   PYTHON=python3.12 POLARS_SPEC='polars==1.44.1' scripts/check_wheel.sh
#   KEEP_VENV=1 scripts/check_wheel.sh                       # leave the virtualenv for inspection
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
POLARS_SPEC="${POLARS_SPEC:-polars}"
WHEEL="${1:-$(ls -t "$ROOT"/python/dist/arrowmetal-*.whl 2>/dev/null | head -1)}"
if [[ -z "$WHEEL" || ! -f "$WHEEL" ]]; then
    echo "error: no wheel given and none in python/dist; build one with scripts/build_wheel.sh" >&2
    exit 1
fi
WHEEL="$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"
PY_ABS="$(command -v "$PYTHON")"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/am-wheel-check.XXXXXX")"
if [[ -z "${KEEP_VENV:-}" ]]; then trap 'rm -rf "$WORK"' EXIT; fi
CLEAN=(env -i "HOME=$HOME" "PATH=/usr/bin:/bin:/usr/sbin:/sbin" "TMPDIR=${TMPDIR:-/tmp}")

echo "==> fresh virtualenv in $WORK/venv ($("$PY_ABS" --version))"
"${CLEAN[@]}" "$PY_ABS" -m venv "$WORK/venv"
echo "==> pip install $(basename "$WHEEL") $POLARS_SPEC"
(cd "$WORK" && "${CLEAN[@]}" "$WORK/venv/bin/python" -m pip install -q --disable-pip-version-check "$WHEEL" "$POLARS_SPEC")

echo "==> the four tiers, from $WORK"
cd "$WORK"
"${CLEAN[@]}" "$WORK/venv/bin/python" - <<'PY'
import ctypes, importlib.util, os, shutil, subprocess, sys

assert shutil.which("cargo") is None, "cargo is on PATH; the check must run without it"
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import arrowmetal as am
from arrowmetal import polars_plugin

pkg = os.path.dirname(os.path.abspath(am.__file__))
lib_dir = os.path.join(pkg, "_lib")
core, plugin = am._find_library(), polars_plugin.plugin_path()
print("arrowmetal", am.__version__, "| polars", pl.__version__, "| pyarrow", pa.__version__,
      "| python", sys.version.split()[0], "| numpy installed:", importlib.util.find_spec("numpy") is not None)
print("device    ", am.device_name())
print("core      ", core)
print("plugin    ", plugin)
assert os.path.dirname(core) == lib_dir, core
assert os.path.dirname(plugin) == lib_dir, plugin

n = 2_000_000
df = pl.DataFrame({"k": pl.int_range(0, n, eager=True) % 100, "v": pl.int_range(0, n, eager=True)})
lf = df.lazy()
want_sum = df["v"].sum()
want_groups = df.group_by("k").agg(pl.col("v").sum()).sort("k")

# tier 1: the bridge
t1 = df.arrowmetal.group_by("k").sum("v")
print("tier 1    ", "df.arrowmetal.group_by('k').sum('v') ->", t1.height, "groups")
assert t1.sort("k")["v"].to_list() == want_groups["v"].to_list()

# tier 2: the expression plugin, inside a lazy plan
t2 = lf.select(pl.col("v").arrowmetal.sum()).collect().item()
fs = lf.select(pl.col("v").arrowmetal.filter_sum(pl.col("k") == 7)).collect().item()
dev = lf.select(pl.col("v").arrowmetal.device()).collect().item()
print("tier 2    ", "pl.col('v').arrowmetal.sum() ->", t2, "| filter_sum(k == 7) ->", fs)
print("          ", "pl.col('v').arrowmetal.device() ->", dev)
assert t2 == want_sum
assert fs == df.filter(pl.col("k") == 7)["v"].sum()

# tier 3: Polars runs the plan, ArrowMetal finishes it
t3 = lf.filter(pl.col("k") < 50).arrowmetal.collect_gpu(lambda d: d.arrowmetal.group_by("k").sum("v"))
print("tier 3    ", "lf.filter(k < 50).arrowmetal.collect_gpu(group_by) ->", t3.height, "groups")
assert t3.height == 50

# tier 4: MetalEngine
engine = am.MetalEngine(shapes="all", min_rows=0)   # take every translatable subtree
t4 = lf.group_by("k").agg(pl.col("v").sum()).sort("k").collect(engine=engine)
print("tier 4    ", "collect(engine=am.MetalEngine(shapes='all')) ->", t4.height, "rows, equal to Polars:", t4.equals(want_groups))
assert t4.equals(want_groups)
print(str(engine.last_report).rstrip())
assert engine.last_report.taken, "MetalEngine took no subtree"

# Exactly one libArrowMetalC.dylib in the process, the packaged one: the plugin resolved it through
# @loader_path rather than a path on the build machine.
dyld = ctypes.CDLL(None)
dyld._dyld_image_count.restype = ctypes.c_uint32
dyld._dyld_get_image_name.restype = ctypes.c_char_p
dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
images = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
ours = [p for p in images if os.path.basename(p) in ("libArrowMetalC.dylib", "libarrowmetal_polars.dylib")]
for p in ours:
    print("loaded    ", p)
assert sorted(os.path.realpath(p) for p in ours) == sorted(
    os.path.realpath(os.path.join(lib_dir, b)) for b in ("libArrowMetalC.dylib", "libarrowmetal_polars.dylib")), ours

# The bench on a Parquet file, from the installed package.
n = 1_000_000
pq.write_table(pa.table({"k": pa.array([i % 7 for i in range(n)], pa.int32()),
                         "v": pa.array([float(i) for i in range(n)])}), "bench.parquet")
r = subprocess.run([sys.executable, "-m", "arrowmetal.bench", "--parquet", "bench.parquet", "--quiet"],
                   capture_output=True, text=True)
print("bench      python -m arrowmetal.bench --parquet bench.parquet --quiet")
print(r.stdout.rstrip())
assert r.returncode == 0, r.stderr
print("OK: all four tiers and bench --parquet from the wheel, no cargo, no DYLD_LIBRARY_PATH")
PY
