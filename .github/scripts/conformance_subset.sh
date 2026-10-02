#!/bin/bash
# The conformance subset every pull request runs (.github/workflows/ci.yml), runnable as is on a Mac:
#
#   .github/scripts/conformance_subset.sh [differential] [standalone] [coverage] [engines] [datafusion]
#
# With no argument it runs all five. Run it from the repository root after
# `swift build -c release --product ArrowMetalC`; PY names the Python (default python3), OUT the
# directory for the logs (default a new temporary one). It prints the same summary lines as the full
# local run (docs/TESTING.md, "What green means"), each part on the subset:
#
#   differential            2,000 cases of the differential matrix, sampled with a fixed seed
#   standalone              test_differential.py without the matrix (findings, fixed findings, guards)
#   coverage                test_coverage_report.py: the per-function table in docs/COVERAGE.md against
#                           the matrix and the record, and a seeded sample of cells rerun
#   engines                 the Polars engine grid without its 100,000-row tables, three column types
#   datafusion              the DataFusion crate's rule tests (tests/rule.rs), not the grid
#
# On a virtual Metal device (GitHub's "Apple Paravirtual device") some kernels cannot be built, and
# after two failed pipeline creations in a process every later one fails (docs/TESTING.md). There
# differential_report.py counts such a case as skipped; a pytest part whose failures all carry that
# error counts as passed with those tests noted as skipped; the engines and datafusion parts, which
# need every kernel, are skipped. Nothing is rerun. The log says so in each case.
set -u
PY=${PY:-python3}
OUT=${OUT:-$(mktemp -d)}
SEED=20261001
PARTS=${*:-differential standalone coverage engines datafusion}
export PYTHONPATH=$PWD/python
export ARROWMETAL_LIB=${ARROWMETAL_LIB:-$PWD/.build/release/libArrowMetalC.dylib}
[ -f "$ARROWMETAL_LIB" ] || { echo "SUBSET: no $ARROWMETAL_LIB (swift build -c release --product ArrowMetalC)"; exit 1; }
PIPELINE="Metal pipeline creation failed"
# A virtual Metal device (GitHub's runner) is recognised before any part runs, so the parts that need
# every kernel can be skipped even when the Python parts are not asked for.
virtual=0
case "$(system_profiler SPDisplaysDataType 2>/dev/null)" in *[Pp]aravirtual*) virtual=1; echo "metal device (system_profiler): Apple Paravirtual device" ;; esac

has() { case " $PARTS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if has differential || has standalone || has coverage || has engines; then
  loaded=$($PY -c "import arrowmetal as am; print(am._find_library())")
  echo "python loads: $loaded"; [ "$loaded" = "$ARROWMETAL_LIB" ] || { echo "SUBSET: python loads the wrong dylib"; exit 1; }
  device=$($PY -c "import arrowmetal as am; print(am.device_name())")
  echo "metal device: $device"
  case "$device" in *[Pp]aravirtual*) virtual=1 ;; esac
fi

# pytest with one rerun of the failed tests, on a virtual device only, when every failure carries the
# pipeline-creation error. Prints pytest's last line (and the rerun's).
run_pytest() {
  local log=$1; shift
  $PY -m pytest "$@" -q -rs -o cache_dir="$OUT/.pytest_cache" > "$log" 2>&1
  local rc=$?
  local line; line=$(tail -1 "$log")
  if [ $rc -ne 0 ] && [ "$virtual" = 1 ] && grep -q "$PIPELINE" "$log" \
     && [ "$(grep -E '^(FAILED|ERROR) ' "$log" | grep -vc "$PIPELINE")" = "0" ]; then
    rc=0
    line="$line; $(grep -E '^(FAILED|ERROR) ' "$log" | wc -l | tr -d ' ') test(s) failed only on pipeline creation on the virtual device: counted as skipped"
  fi
  echo "$line"
  return $rc
}

diff_rc=-; py_ok=-; eng_rc=-; df_rc=-

if has differential; then
  $PY python/tests/differential_report.py -q --sample 2000 --seed $SEED > "$OUT/diff.txt" 2>&1; diff_rc=$?
  echo "differential: $(grep -E '^total:' "$OUT/diff.txt") exit=$diff_rc"
  grep -E '^pipeline creation' "$OUT/diff.txt"; grep -A6 "UNCLASSIFIED" "$OUT/diff.txt" | head -8
fi

if has standalone; then
  dline=$(run_pytest "$OUT/diffpy.txt" python/tests/test_differential.py -k "not test_matches_pyarrow"); rc=$?
  echo "differential-standalone: $dline"; grep -E "^FAILED" "$OUT/diffpy.txt" | head -10
  py_ok=$([ $rc -eq 0 ] && echo 1 || echo 0)
fi

if has coverage; then
  cline=$(run_pytest "$OUT/coverage.txt" python/tests/test_coverage_report.py); rc=$?
  echo "coverage-table: $cline"; grep -E "^FAILED" "$OUT/coverage.txt" | head -10
  [ $rc -eq 0 ] || py_ok=0; [ "$py_ok" = "-" ] && py_ok=1
fi

if has engines && [ "$virtual" = 1 ]; then
  echo "engines: skipped on the virtual device (the engine grids need every kernel)"
elif has engines; then
  $PY python/tests/engine_report.py --engine polars --quick --dtypes int64,float64,string -q \
    > "$OUT/engines.txt" 2>&1; eng_rc=$?
  echo "engines: $(grep -E '^engine ' "$OUT/engines.txt" | tr '\n' ' ') exit=$eng_rc"
  grep -A8 "UNCLASSIFIED" "$OUT/engines.txt" | head -10
fi

if has datafusion && [ "$virtual" = 1 ]; then
  echo "datafusion: skipped on the virtual device (the rule tests run sorts and group-bys on the GPU)"
elif has datafusion; then
  (cd datafusion && cargo test --test rule -- --nocapture) > "$OUT/df.txt" 2>&1; df_rc=$?
  echo "datafusion: rule tests only (tests/rule.rs) | $(grep -E '^test result: ' "$OUT/df.txt" | tail -1) exit=$df_rc"
  grep -E 'FAILED|panicked|^error' "$OUT/df.txt" | head -10
fi

echo "GATE swift_ok=- diff_rc=$diff_rc py_ok=$py_ok eng_rc=$eng_rc df_rc=$df_rc (subset: $PARTS; logs in $OUT)"
ok=1
for v in "$diff_rc" "$eng_rc" "$df_rc"; do [ "$v" = "-" ] || [ "$v" = "0" ] || ok=0; done
[ "$py_ok" = "-" ] || [ "$py_ok" = "1" ] || ok=0
[ $ok = 1 ]
