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
# On a virtual Metal device (GitHub's "Apple Paravirtual device") pipeline creation fails at random
# (docs/FINDINGS.md); differential_report.py reruns a case that failed that way, and a pytest part
# whose failures all carry that error is rerun once on its failed tests. Both say so in the log.
set -u
PY=${PY:-python3}
OUT=${OUT:-$(mktemp -d)}
SEED=20261001
PARTS=${*:-differential standalone coverage engines datafusion}
export PYTHONPATH=$PWD/python
export ARROWMETAL_LIB=${ARROWMETAL_LIB:-$PWD/.build/release/libArrowMetalC.dylib}
[ -f "$ARROWMETAL_LIB" ] || { echo "SUBSET: no $ARROWMETAL_LIB (swift build -c release --product ArrowMetalC)"; exit 1; }
PIPELINE="Metal pipeline creation failed"
virtual=0

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
    $PY -m pytest "$@" -q -rs --last-failed -o cache_dir="$OUT/.pytest_cache" > "$log.rerun" 2>&1
    rc=$?
    line="$line; rerun of the pipeline-creation failures: $(tail -1 "$log.rerun")"
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

if has engines; then
  $PY python/tests/engine_report.py --engine polars --quick --dtypes int64,float64,string -q \
    > "$OUT/engines.txt" 2>&1; eng_rc=$?
  echo "engines: $(grep -E '^engine ' "$OUT/engines.txt" | tr '\n' ' ') exit=$eng_rc"
  grep -A8 "UNCLASSIFIED" "$OUT/engines.txt" | head -10
fi

if has datafusion; then
  (cd datafusion && cargo test --test rule -- --nocapture) > "$OUT/df.txt" 2>&1; df_rc=$?
  echo "datafusion: rule tests only (tests/rule.rs) | $(grep -E '^test result: ' "$OUT/df.txt" | tail -1) exit=$df_rc"
  grep -E 'FAILED|panicked|^error' "$OUT/df.txt" | head -10
fi

echo "GATE swift_ok=- diff_rc=$diff_rc py_ok=$py_ok eng_rc=$eng_rc df_rc=$df_rc (subset: $PARTS; logs in $OUT)"
ok=1
for v in "$diff_rc" "$eng_rc" "$df_rc"; do [ "$v" = "-" ] || [ "$v" = "0" ] || ok=0; done
[ "$py_ok" = "-" ] || [ "$py_ok" = "1" ] || ok=0
[ $ok = 1 ]
