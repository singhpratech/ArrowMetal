"""The complete ArrowMetal vs Polars / pyarrow.compute / pandas matrix.

Every operation family the Python package exposes, measured against the fastest idiom of each CPU
library on the same in-process data, at 10M and 50M rows (1M and 10M for string columns).

    PYTHONPATH=python python Benchmarks/full_matrix.py            # the full run (~30-40 min)
    PYTHONPATH=python python Benchmarks/full_matrix.py --quick    # 1M-row smoke, a few minutes

Writes `Benchmarks/results/full_matrix_<date>.csv` and `docs/BENCHMARKS_MATRIX.md`.

Methodology (the same rules as Benchmarks/README.md):
- one warm-up call, then best-of-5 wall time; a call is repeated fewer times (never fewer than
  twice) once the repetitions have used the per-measurement budget, so a six-second pyarrow sort
  does not cost a minute. The iteration count actually used is recorded in the CSV.
- CPU time is the process user+system time delta across the call (`resource.getrusage`), which
  counts every thread the library spawned, so a 16-thread CPU kernel shows ~16x its wall time.
- Every library is handed the same values. pyarrow gets the Arrow array; Polars a Series built from
  it; pandas an Arrow-backed Series when the column has nulls (the only faithful representation) and
  a numpy-backed one when it has none (pandas' fastest idiom); numpy stands in where pandas has no
  vectorised equivalent. No Python loops anywhere in a baseline.
- Bytes counted are the bytes the operation must touch (input + output), so GB/s is comparable.
- An operation ArrowMetal does not have, or that raises, is recorded as an error row, never skipped.
"""
import argparse, csv, datetime, decimal, gc, os, resource, sys, time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SEED = 20260906

# ---------------------------------------------------------------- measurement

ROWS = []           # dicts: family, op, rows, library, wall_ms, cpu_ms, gbs, iters, status, note
_ORDER = []         # (family, op, rows) in first-seen order


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Bench:
    def __init__(self, iters, budget):
        self.iters, self.budget = iters, budget

    def run(self, fn):
        """(wall_ms, cpu_ms, iterations) for the best of up to `iters` calls after one warm-up."""
        fn()
        best_w, best_c, total, n = float("inf"), float("inf"), 0.0, 0
        while n < self.iters and (n < 2 or total < self.budget):
            c0 = cpu_seconds()
            t0 = time.perf_counter()
            fn()
            w = time.perf_counter() - t0
            c = cpu_seconds() - c0
            if w < best_w:
                best_w, best_c = w, c
            total += w
            n += 1
        return best_w * 1000.0, best_c * 1000.0, n


BENCH = Bench(5, 1.2)


def record(family, op, rows, library, wall_ms, cpu_ms, gbs, iters, status, note):
    key = (family, op, rows)
    if key not in _ORDER:
        _ORDER.append(key)
    ROWS.append(dict(family=family, op=op, rows=rows, library=library, wall_ms=wall_ms,
                     cpu_ms=cpu_ms, gbs=gbs, iters=iters, status=status, note=note))


def case(family, op, rows, nbytes, impls, notes=None):
    """Measure one operation across the libraries.

    `impls` maps a library name to a zero-argument callable, or to None when the library has no
    equivalent (recorded as such, not silently dropped). `notes` maps a library name to a string.
    """
    notes = notes or {}
    for library, fn in impls.items():
        note = notes.get(library, "")
        if fn is None:
            record(family, op, rows, library, None, None, None, 0, "no equivalent", note)
            print(f"  {family:<12} {op:<34} {rows:>10,} {library:<11} -- no equivalent")
            continue
        try:
            wall, cpu, iters = BENCH.run(fn)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:220]
            record(family, op, rows, library, None, None, None, 0, "error", msg)
            print(f"  {family:<12} {op:<34} {rows:>10,} {library:<11} !! {msg[:90]}")
            continue
        gbs = (nbytes / (wall / 1000.0) / 1e9) if wall > 0 else None
        record(family, op, rows, library, wall, cpu, gbs, iters, "ok", note)
        print(f"  {family:<12} {op:<34} {rows:>10,} {library:<11} {wall:9.3f} ms "
              f"{cpu:9.1f} cpu-ms {gbs:7.1f} GB/s")
    gc.collect()


# ---------------------------------------------------------------- data

class Col:
    """One column, materialised per library only when a benchmark asks for it."""

    def __init__(self, arrow, numpy_=None, has_nulls=None):
        self.a = arrow
        self._np = numpy_
        self._has_nulls = arrow.null_count > 0 if has_nulls is None else has_nulls
        self._g = self._p = self._d = None

    @property
    def g(self):                       # ArrowMetal (device memory)
        if self._g is None:
            self._g = am.array(self.a)
        return self._g

    @property
    def p(self):                       # polars
        if self._p is None:
            self._p = pl.Series("x", self.a)
        return self._p

    @property
    def d(self):                       # pandas
        if self._d is None:
            if self._np is not None and not self._has_nulls:
                self._d = pd.Series(self._np)
            else:
                self._d = pd.Series(pd.arrays.ArrowExtensionArray(self.a))
        return self._d

    @property
    def n(self):                       # numpy
        if self._np is None:
            self._np = self.a.to_numpy(zero_copy_only=False)
        return self._np

    @property
    def nbytes(self):
        return self.a.nbytes


def masked(values, rng, null_fraction=0.10):
    if null_fraction <= 0:
        return pa.array(values)
    return pa.array(values, mask=rng.random(len(values)) < null_fraction)


class Data:
    """The columns for one row count. Everything is lazy; drop the object to free the memory."""

    def __init__(self, n):
        self.n = n
        self.rng = np.random.default_rng(SEED)
        self._c = {}

    def __call__(self, name):
        if name not in self._c:
            self._c[name] = getattr(self, "_build_" + name)()
        return self._c[name]

    # -- integers
    def _build_i64(self):
        v = self.rng.integers(-1000, 1001, size=self.n, dtype=np.int64)
        return Col(masked(v, self.rng), v)

    def _build_i64b(self):
        v = self.rng.integers(1, 1001, size=self.n, dtype=np.int64)
        return Col(pa.array(v), v)

    def _build_i64_nn(self):
        v = self.rng.integers(-(2 ** 62), 2 ** 62, size=self.n, dtype=np.int64)
        return Col(pa.array(v), v)

    # -- floats
    def _build_f64(self):
        v = self.rng.random(self.n) * 1000.0
        return Col(masked(v, self.rng), v)

    def _build_f64_nn(self):
        v = self.rng.random(self.n) * 2e9 - 1e9
        return Col(pa.array(v), v)

    def _build_f64b(self):
        v = self.rng.random(self.n) * 1000.0 + 1.0
        return Col(pa.array(v), v)

    def _build_f32(self):
        v = (self.rng.random(self.n, dtype=np.float32) * 2 - 1).astype(np.float32)
        return Col(pa.array(v), v)

    # -- booleans
    def _build_bool(self):
        v = self.rng.random(self.n) < 0.5
        return Col(masked(v, self.rng), v)

    def _build_bool_nn(self):
        v = self.rng.random(self.n) < 0.5
        return Col(pa.array(v), v)

    def _build_mask30(self):
        v = self.rng.random(self.n) < 0.30
        return Col(pa.array(v), v)

    def _build_mask90(self):
        v = self.rng.random(self.n) < 0.90
        return Col(pa.array(v), v)

    # -- selection
    def _build_idx(self):
        v = self.rng.integers(0, self.n, size=self.n // 2, dtype=np.int32)
        return Col(pa.array(v), v)

    # -- temporal: microsecond timestamps across ~20 years
    def _build_ts(self):
        base = 1_100_000_000_000_000
        v = base + self.rng.integers(0, 631_000_000_000_000, size=self.n, dtype=np.int64)
        return Col(pa.array(v).cast(pa.timestamp("us")), v)

    def _build_ts2(self):
        base = 1_400_000_000_000_000
        v = base + self.rng.integers(0, 631_000_000_000_000, size=self.n, dtype=np.int64)
        return Col(pa.array(v).cast(pa.timestamp("us")), v)

    # -- decimal128(18, 4), built straight from the unscaled 128-bit values
    def _build_dec(self):
        v = self.rng.integers(0, 10 ** 10, size=self.n, dtype=np.int64)
        return Col(decimal_array(v), None, has_nulls=False)

    def _build_dec2(self):
        v = self.rng.integers(0, 10 ** 8, size=self.n, dtype=np.int64)
        return Col(decimal_array(v), None, has_nulls=False)

    # -- group-by keys
    def keys(self, distinct):
        name = f"keys_{distinct}"
        if name not in self._c:
            v = self.rng.integers(0, min(distinct, self.n), size=self.n, dtype=np.int32)
            self._c[name] = Col(pa.array(v), v)
        return self._c[name]

    def str_keys(self, distinct):
        """utf8 keys of `distinct` cardinality, built straight into Arrow buffers (12 bytes each)."""
        name = f"str_keys_{distinct}"
        if name not in self._c:
            codes = self.rng.integers(0, min(distinct, self.n), size=self.n, dtype=np.int32)
            self._c[name] = Col(fixed_width_utf8(codes), None, has_nulls=False)
        return self._c[name]


def decimal_array(unscaled, precision=18, scale=4):
    """A decimal128 Arrow array from int64 unscaled values (two 64-bit halves, little-endian)."""
    n = len(unscaled)
    buf = np.zeros((n, 2), dtype=np.int64)
    buf[:, 0] = unscaled
    buf[:, 1] = np.where(unscaled < 0, -1, 0)
    return pa.Array.from_buffers(pa.decimal128(precision, scale), n,
                                 [None, pa.py_buffer(buf.reshape(-1))])


def fixed_width_utf8(codes, width=12, lead=b"k"):
    """A utf8 Arrow array of `len(codes)` fixed-width keys, vectorised (no Python string loop)."""
    n = len(codes)
    hexdigits = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)
    body = np.empty((n, width), dtype=np.uint8)
    body[:, 0] = lead[0]
    acc = codes.astype(np.uint64)
    for j in range(width - 1, 0, -1):
        body[:, j] = hexdigits[(acc & 0xF).astype(np.intp)]
        acc >>= 4
    offsets = np.arange(n + 1, dtype=np.int32) * width
    return pa.Array.from_buffers(pa.utf8(), n, [None, pa.py_buffer(offsets),
                                                pa.py_buffer(body.reshape(-1))])


class StrData:
    """utf8 columns: 1000 distinct `cust_NNN_region` values, the shape used everywhere else here."""

    REGIONS = ("north", "south", "east", "west")

    def __init__(self, n, distinct=1000):
        self.n, self.distinct = n, distinct
        self.rng = np.random.default_rng(SEED + 1)
        self.vocab = [f"cust_{i:03d}_{self.REGIONS[i % 4]}" for i in range(distinct)]
        self.codes = self.rng.integers(0, distinct, size=n, dtype=np.int32)
        arr = pa.DictionaryArray.from_arrays(pa.array(self.codes), pa.array(self.vocab)).cast(pa.string())
        self.s = Col(arr, None, has_nulls=False)
        self._c = {}

    def __call__(self, name):
        if name not in self._c:
            self._c[name] = getattr(self, "_build_" + name)()
        return self._c[name]

    def _build_s2(self):
        codes = self.rng.integers(0, self.distinct, size=self.n, dtype=np.int32)
        arr = pa.DictionaryArray.from_arrays(pa.array(codes), pa.array(self.vocab)).cast(pa.string())
        return Col(arr, None, has_nulls=False)

    def _build_padded(self):
        """The same values with leading and trailing spaces, for trim."""
        vocab = pa.array(["  " + v + "  " for v in self.vocab])
        arr = pa.DictionaryArray.from_arrays(pa.array(self.codes), vocab).cast(pa.string())
        return Col(arr, None, has_nulls=False)

    def _build_ints(self):
        v = self.rng.integers(0, 1_000_000, size=self.n, dtype=np.int64)
        return Col(pa.array(v), v)

    def _build_numeric_text(self):
        codes = self.rng.integers(0, 1_000_000, size=self.n, dtype=np.int32)
        digits = np.frombuffer(b"0123456789", dtype=np.uint8)
        width = 8
        body = np.empty((self.n, width), dtype=np.uint8)
        acc = codes.astype(np.uint64)
        for j in range(width - 1, -1, -1):
            body[:, j] = digits[(acc % 10).astype(np.intp)]
            acc //= 10
        offsets = np.arange(self.n + 1, dtype=np.int32) * width
        arr = pa.Array.from_buffers(pa.utf8(), self.n, [None, pa.py_buffer(offsets),
                                                        pa.py_buffer(body.reshape(-1))])
        return Col(arr, None, has_nulls=False)


# ---------------------------------------------------------------- families

def family_reductions(d, n):
    f = "reductions"
    i, fl, b = d("i64"), d("f64"), d("bool")
    B = n * 8
    case(f, "sum(int64, 10% nulls)", n, B, {
        "arrowmetal": lambda: i.g.sum(),
        "polars": lambda: i.p.sum(),
        "pyarrow": lambda: pc.sum(i.a),
        "pandas": lambda: i.d.sum()})
    case(f, "mean(int64, 10% nulls)", n, B, {
        "arrowmetal": lambda: i.g.mean(),
        "polars": lambda: i.p.mean(),
        "pyarrow": lambda: pc.mean(i.a),
        "pandas": lambda: i.d.mean()})
    case(f, "min(int64, 10% nulls)", n, B, {
        "arrowmetal": lambda: i.g.min(),
        "polars": lambda: i.p.min(),
        "pyarrow": lambda: pc.min(i.a),
        "pandas": lambda: i.d.min()})
    case(f, "max(int64, 10% nulls)", n, B, {
        "arrowmetal": lambda: i.g.max(),
        "polars": lambda: i.p.max(),
        "pyarrow": lambda: pc.max(i.a),
        "pandas": lambda: i.d.max()})
    case(f, "min_max(int64)", n, B, {
        "arrowmetal": lambda: i.g.min_max(),
        "polars": lambda: (i.p.min(), i.p.max()),
        "pyarrow": lambda: pc.min_max(i.a),
        "pandas": lambda: (i.d.min(), i.d.max())},
        notes={"polars": "no single-pass min_max; two reductions",
               "pandas": "no single-pass min_max; two reductions"})
    case(f, "variance(float64, ddof=1)", n, B, {
        "arrowmetal": lambda: fl.g.variance(ddof=1),
        "polars": lambda: fl.p.var(),
        "pyarrow": lambda: pc.variance(fl.a, ddof=1),
        "pandas": lambda: fl.d.var()})
    case(f, "stddev(float64, ddof=1)", n, B, {
        "arrowmetal": lambda: fl.g.stddev(ddof=1),
        "polars": lambda: fl.p.std(),
        "pyarrow": lambda: pc.stddev(fl.a, ddof=1),
        "pandas": lambda: fl.d.std()})
    case(f, "count_distinct(int64)", n, B, {
        "arrowmetal": lambda: i.g.count_distinct(),
        "polars": lambda: i.p.n_unique(),
        "pyarrow": lambda: pc.count_distinct(i.a),
        "pandas": lambda: i.d.nunique()})
    case(f, "quantile(float64, 0.5)", n, B, {
        "arrowmetal": lambda: fl.g.quantile(0.5),
        "polars": lambda: fl.p.quantile(0.5),
        "pyarrow": lambda: pc.quantile(fl.a, q=0.5),
        "pandas": lambda: fl.d.quantile(0.5)})
    case(f, "mode(int64)", n, B, {
        "arrowmetal": lambda: i.g.mode(),
        "polars": lambda: i.p.mode(),
        "pyarrow": lambda: pc.mode(i.a),
        "pandas": lambda: i.d.mode()})
    case(f, "product(int64)", n, B, {
        "arrowmetal": lambda: i.g.product(),
        "polars": lambda: i.p.product(),
        "pyarrow": lambda: pc.product(i.a),
        "pandas": lambda: i.d.prod()})
    case(f, "any(bool, 10% nulls)", n, n // 8, {
        "arrowmetal": lambda: b.g.any(),
        "polars": lambda: b.p.any(),
        "pyarrow": lambda: pc.any(b.a),
        "pandas": lambda: b.d.any()})
    case(f, "all(bool, 10% nulls)", n, n // 8, {
        "arrowmetal": lambda: b.g.all(),
        "polars": lambda: b.p.all(),
        "pyarrow": lambda: pc.all(b.a),
        "pandas": lambda: b.d.all()})
    case(f, "first(int64)", n, B, {
        "arrowmetal": lambda: i.g.first(),
        "polars": lambda: i.p.drop_nulls().first(),
        "pyarrow": lambda: pc.first(i.a),
        "pandas": lambda: i.d.iloc[i.d.first_valid_index()]},
        notes={"polars": "no skip-null first(); drop_nulls().first()",
               "pandas": "no skip-null first(); iloc[first_valid_index()]"})
    case(f, "last(int64)", n, B, {
        "arrowmetal": lambda: i.g.last(),
        "polars": lambda: i.p.drop_nulls().last(),
        "pyarrow": lambda: pc.last(i.a),
        "pandas": lambda: i.d.iloc[i.d.last_valid_index()]})
    fl_np = pd.Series(fl.n)          # arrow-backed pandas has no skew/kurtosis kernel
    case(f, "skew(float64)", n, B, {
        "arrowmetal": lambda: fl.g.skew(),
        "polars": lambda: fl.p.skew(bias=True),
        "pyarrow": lambda: pc.skew(fl.a),
        "pandas": lambda: fl_np.skew()},
        notes={"pandas": "numpy-backed Series (arrow-backed pandas has no skew kernel); "
                         "pandas' skew is bias-corrected, the same amount of work"})
    case(f, "kurtosis(float64)", n, B, {
        "arrowmetal": lambda: fl.g.kurtosis(),
        "polars": lambda: fl.p.kurtosis(bias=True),
        "pyarrow": lambda: pc.kurtosis(fl.a),
        "pandas": lambda: fl_np.kurt()},
        notes={"pandas": "numpy-backed Series (arrow-backed pandas has no kurtosis kernel)"})
    case(f, "tdigest(float64, q=0.5)", n, B, {
        "arrowmetal": lambda: fl.g.tdigest(0.5),
        "polars": None,
        "pyarrow": lambda: pc.tdigest(fl.a, q=0.5),
        "pandas": None},
        notes={"polars": "polars has no t-digest sketch",
               "pandas": "pandas has no t-digest sketch"})


def family_elementwise(d, n):
    f = "element-wise"
    i, ib, fl, flb, f32 = d("i64"), d("i64b"), d("f64"), d("f64b"), d("f32")
    B = n * 16
    case(f, "add scalar (int64)", n, B, {
        "arrowmetal": lambda: i.g + 1,
        "polars": lambda: i.p + 1,
        "pyarrow": lambda: pc.add(i.a, 1),
        "pandas": lambda: i.d + 1})
    case(f, "multiply scalar (int64)", n, B, {
        "arrowmetal": lambda: i.g * 3,
        "polars": lambda: i.p * 3,
        "pyarrow": lambda: pc.multiply(i.a, 3),
        "pandas": lambda: i.d * 3})
    case(f, "add array (int64 + int64)", n, n * 24, {
        "arrowmetal": lambda: i.g + ib.g,
        "polars": lambda: i.p + ib.p,
        "pyarrow": lambda: pc.add(i.a, ib.a),
        "pandas": lambda: i.d + ib.d})
    case(f, "multiply array (int64 * int64)", n, n * 24, {
        "arrowmetal": lambda: i.g * ib.g,
        "polars": lambda: i.p * ib.p,
        "pyarrow": lambda: pc.multiply(i.a, ib.a),
        "pandas": lambda: i.d * ib.d})
    case(f, "divide (float64 / float64)", n, n * 24, {
        "arrowmetal": lambda: fl.g / flb.g,
        "polars": lambda: fl.p / flb.p,
        "pyarrow": lambda: pc.divide(fl.a, flb.a),
        "pandas": lambda: fl.d / flb.d})
    case(f, "power (float32 ** 2)", n, n * 8, {
        "arrowmetal": lambda: f32.g.power(2.0),
        "polars": lambda: f32.p ** 2.0,
        "pyarrow": lambda: pc.power(f32.a, pa.scalar(2.0, pa.float32())),
        "pandas": lambda: f32.d ** 2.0},
        notes={"arrowmetal": "power() is not implemented for float64; float32 column used"})
    case(f, "sqrt (float64)", n, B, {
        "arrowmetal": lambda: flb.g.sqrt(),
        "polars": lambda: flb.p.sqrt(),
        "pyarrow": lambda: pc.sqrt(flb.a),
        "numpy": lambda: np.sqrt(flb.n)},
        notes={"numpy": "pandas has no Series.sqrt(); numpy is the pandas idiom"})
    case(f, "exp (float32)", n, n * 8, {
        "arrowmetal": lambda: f32.g.exp(),
        "polars": lambda: f32.p.exp(),
        "pyarrow": lambda: pc.exp(f32.a),
        "numpy": lambda: np.exp(f32.n)})
    case(f, "ln (float64)", n, B, {
        "arrowmetal": lambda: flb.g.ln(),
        "polars": lambda: flb.p.log(),
        "pyarrow": lambda: pc.ln(flb.a),
        "numpy": lambda: np.log(flb.n)})
    case(f, "sin (float64)", n, B, {
        "arrowmetal": lambda: fl.g.sin(),
        "polars": lambda: fl.p.sin(),
        "pyarrow": lambda: pc.sin(fl.a),
        "numpy": lambda: np.sin(fl.n)})
    case(f, "round (float64)", n, B, {
        "arrowmetal": lambda: fl.g.round(),
        "polars": lambda: fl.p.round(0),
        "pyarrow": lambda: pc.round(fl.a),
        "pandas": lambda: fl.d.round(0)})
    case(f, "abs (float64)", n, B, {
        "arrowmetal": lambda: fl.g.abs(),
        "polars": lambda: fl.p.abs(),
        "pyarrow": lambda: pc.abs(fl.a),
        "pandas": lambda: fl.d.abs()})
    case(f, "negate (int64)", n, B, {
        "arrowmetal": lambda: i.g.negate(),
        "polars": lambda: -i.p,
        "pyarrow": lambda: pc.negate(i.a),
        "pandas": lambda: -i.d})
    case(f, "add_checked (int64 + int64)", n, n * 24, {
        "arrowmetal": lambda: i.g.add_checked(ib.g),
        "polars": None,
        "pyarrow": lambda: pc.add_checked(i.a, ib.a),
        "pandas": None},
        notes={"polars": "polars addition is unchecked (wraps); no checked kernel",
               "pandas": "pandas/numpy addition is unchecked"})
    case(f, "bit_wise_and (int64)", n, n * 24, {
        "arrowmetal": lambda: i.g.bitwise_and(ib.g),
        "polars": lambda: i.p & ib.p,
        "pyarrow": lambda: pc.bit_wise_and(i.a, ib.a),
        "numpy": lambda: i.n & ib.n})
    case(f, "shift_left (int64 << 2)", n, B, {
        "arrowmetal": lambda: i.g.shift_left(2),
        "polars": None,
        "pyarrow": lambda: pc.shift_left(i.a, 2),
        "numpy": lambda: i.n << 2},
        notes={"polars": "polars Series has no bit-shift operator"})
    cond = d("mask30")
    c2 = d("mask90")
    ew = pl.DataFrame({"a": i.p, "b": ib.p, "c": cond.p, "c2": c2.p})
    case(f, "if_else (bool ? int64 : int64)", n, n * 24, {
        "arrowmetal": lambda: cond.g.if_else(i.g, ib.g),
        "polars": lambda: ew.select(pl.when(pl.col("c")).then(pl.col("a")).otherwise(pl.col("b"))),
        "pyarrow": lambda: pc.if_else(cond.a, i.a, ib.a),
        "numpy": lambda: np.where(cond.n, i.n, ib.n)},
        notes={"numpy": "numpy has no null representation; the same values without the validity bitmap"})
    case(f, "coalesce (2 int64 columns)", n, n * 24, {
        "arrowmetal": lambda: am.coalesce(i.g, ib.g),
        "polars": lambda: ew.select(pl.coalesce(pl.col("a"), pl.col("b"))),
        "pyarrow": lambda: pc.coalesce(i.a, ib.a),
        "pandas": lambda: i.d.fillna(ib.d)})
    case(f, "fill_null (int64)", n, B, {
        "arrowmetal": lambda: i.g.fill_null(0),
        "polars": lambda: i.p.fill_null(0),
        "pyarrow": lambda: pc.fill_null(i.a, 0),
        "pandas": lambda: i.d.fillna(0)})
    case(f, "fill_null_forward (int64)", n, B, {
        "arrowmetal": lambda: i.g.fill_null_forward(),
        "polars": lambda: i.p.fill_null(strategy="forward"),
        "pyarrow": lambda: pc.fill_null_forward(i.a),
        "pandas": lambda: i.d.ffill()})
    case(f, "is_nan (float64)", n, n * 9, {
        "arrowmetal": lambda: fl.g.is_nan(),
        "polars": lambda: fl.p.is_nan(),
        "pyarrow": lambda: pc.is_nan(fl.a),
        "numpy": lambda: np.isnan(fl.n)})
    set_np = np.arange(0, 1000, 10, dtype=np.int64)
    set_pa = pa.array(set_np)
    set_g = am.array(set_pa)
    set_pl = pl.Series(set_np)
    case(f, "is_in (int64, 100-value set)", n, n * 9, {
        "arrowmetal": lambda: i.g.is_in(set_g),
        "polars": lambda: i.p.is_in(set_pl),
        "pyarrow": lambda: pc.is_in(i.a, value_set=set_pa),
        "pandas": lambda: i.d.isin(set_np)})
    case(f, "index_in (int64, 100-value set)", n, n * 12, {
        "arrowmetal": lambda: i.g.index_in(set_g),
        "polars": None,
        "pyarrow": lambda: pc.index_in(i.a, value_set=set_pa),
        "numpy": lambda: np.searchsorted(set_np, i.n)},
        notes={"polars": "no index_in; polars has is_in only",
               "numpy": "searchsorted is the closest vectorised equivalent (no membership check)"})
    case(f, "case_when (2 conditions)", n, n * 40, {
        "arrowmetal": lambda: am.case_when([cond.g, c2.g], [i.g, ib.g], ib.g),
        "polars": lambda: ew.select(pl.when(pl.col("c")).then(pl.col("a"))
                                      .when(pl.col("c2")).then(pl.col("b")).otherwise(pl.col("b"))),
        "pyarrow": lambda: pc.case_when(pc.make_struct(cond.a, c2.a), i.a, ib.a, ib.a),
        "numpy": lambda: np.select([cond.n, c2.n], [i.n, ib.n], ib.n)})
    case(f, "hash64 (int64)", n, B, {
        "arrowmetal": lambda: i.g.hash64(),
        "polars": lambda: i.p.hash(),
        "pyarrow": None,
        "pandas": lambda: pd.util.hash_array(ib.n)},
        notes={"pyarrow": "pyarrow.compute has no element-wise hash function",
               "pandas": "pd.util.hash_array over the no-null column"})


def family_select(d, n):
    f = "compare+select"
    i, ib = d("i64"), d("i64b")
    m30, m90, idx = d("mask30"), d("mask90"), d("idx")
    B = n * 9
    case(f, "compare scalar (int64 > 0)", n, B, {
        "arrowmetal": lambda: i.g > 0,
        "polars": lambda: i.p > 0,
        "pyarrow": lambda: pc.greater(i.a, 0),
        "pandas": lambda: i.d > 0})
    case(f, "compare array (int64 > int64)", n, n * 17, {
        "arrowmetal": lambda: i.g > ib.g,
        "polars": lambda: i.p > ib.p,
        "pyarrow": lambda: pc.greater(i.a, ib.a),
        "pandas": lambda: i.d > ib.d})
    case(f, "filter int64 (30% kept)", n, int(n * 8 * 1.3), {
        "arrowmetal": lambda: i.g.filter(m30.g),
        "polars": lambda: i.p.filter(m30.p),
        "pyarrow": lambda: pc.filter(i.a, m30.a),
        "pandas": lambda: i.d[m30.d]})
    case(f, "filter int64 (90% kept)", n, int(n * 8 * 1.9), {
        "arrowmetal": lambda: i.g.filter(m90.g),
        "polars": lambda: i.p.filter(m90.p),
        "pyarrow": lambda: pc.filter(i.a, m90.a),
        "pandas": lambda: i.d[m90.d]})
    case(f, "take (n/2 random indices)", n, n // 2 * 20, {
        "arrowmetal": lambda: i.g.take(idx.g),
        "polars": lambda: i.p.gather(idx.p),
        "pyarrow": lambda: pc.take(i.a, idx.a),
        "pandas": lambda: i.d.take(idx.n)})
    case(f, "drop_null (int64, 10% nulls)", n, int(n * 8 * 1.9), {
        "arrowmetal": lambda: i.g.drop_null(),
        "polars": lambda: i.p.drop_nulls(),
        "pyarrow": lambda: pc.drop_null(i.a),
        "pandas": lambda: i.d.dropna()})
    case(f, "slice (zero-copy view)", n, 0, {
        "arrowmetal": lambda: i.g.slice(1000, n - 2000),
        "polars": lambda: i.p.slice(1000, n - 2000),
        "pyarrow": lambda: i.a.slice(1000, n - 2000),
        "pandas": lambda: i.d.iloc[1000:n - 1000]})
    repl = Col(pa.array(np.zeros(int(m30.n.sum()), dtype=np.int64)))
    zeros_pl = pl.Series(np.zeros(n, dtype=np.int64))
    case(f, "replace_with_mask (30% replaced)", n, n * 17, {
        "arrowmetal": lambda: ib.g.replace_with_mask(m30.g, repl.g),
        "polars": lambda: zeros_pl.zip_with(m30.p, ib.p),
        "pyarrow": lambda: pc.replace_with_mask(ib.a, m30.a, repl.a),
        "pandas": lambda: ib.d.mask(m30.d, 0)},
        notes={"polars": "no replace_with_mask; zip_with against a full zero column",
               "pandas": "Series.mask with a scalar (no positional replacement list)"})
    case(f, "indices_nonzero (bool)", n, n * 9, {
        "arrowmetal": lambda: m30.g.indices_nonzero(),
        "polars": lambda: m30.p.arg_true(),
        "pyarrow": lambda: pc.indices_nonzero(m30.a),
        "numpy": lambda: np.flatnonzero(m30.n)})


def family_sort(d, n, sd):
    f = "sort"
    ii, ff = d("i64_nn"), d("f64_nn")
    case(f, "argsort int64", n, n * 12, {
        "arrowmetal": lambda: ii.g.argsort(),
        "polars": lambda: ii.p.arg_sort(),
        "pyarrow": lambda: pc.array_sort_indices(ii.a),
        "numpy": lambda: np.argsort(ii.n)},
        notes={"numpy": "pandas argsort delegates to numpy"})
    case(f, "argsort float64", n, n * 12, {
        "arrowmetal": lambda: ff.g.argsort(),
        "polars": lambda: ff.p.arg_sort(),
        "pyarrow": lambda: pc.array_sort_indices(ff.a),
        "numpy": lambda: np.argsort(ff.n)})
    if sd is not None:
        case(f, "argsort utf8", sd.n, sd.s.nbytes, {
            "arrowmetal": lambda: sd.s.g.argsort(),
            "polars": lambda: sd.s.p.arg_sort(),
            "pyarrow": lambda: pc.array_sort_indices(sd.s.a),
            "pandas": lambda: sd.s.d.argsort()})
    case(f, "sort float64", n, n * 16, {
        "arrowmetal": lambda: ff.g.sort(),
        "polars": lambda: ff.p.sort(),
        "pyarrow": lambda: pc.take(ff.a, pc.array_sort_indices(ff.a)),
        "numpy": lambda: np.sort(ff.n)})
    case(f, "top_k (k=100, int64)", n, n * 8, {
        "arrowmetal": lambda: ii.g.top_k(100),
        "polars": lambda: ii.p.top_k(100),
        "pyarrow": lambda: pc.select_k_unstable(ii.a, k=100, sort_keys=[("", "descending")]),
        "numpy": lambda: np.argpartition(ii.n, n - 100)[n - 100:]})
    case(f, "top_k (k=10000, int64)", n, n * 8, {
        "arrowmetal": lambda: ii.g.top_k(10000),
        "polars": lambda: ii.p.top_k(10000),
        "pyarrow": lambda: pc.select_k_unstable(ii.a, k=10000, sort_keys=[("", "descending")]),
        "numpy": lambda: np.argpartition(ii.n, n - 10000)[n - 10000:]})
    case(f, "rank (min)", n, n * 16, {
        "arrowmetal": lambda: ii.g.rank(),
        "polars": lambda: ii.p.rank("min"),
        "pyarrow": lambda: pc.rank(ii.a, sort_keys="ascending", tiebreaker="min"),
        "pandas": lambda: ii.d.rank(method="min")})
    case(f, "dense_rank", n, n * 16, {
        "arrowmetal": lambda: ii.g.dense_rank(),
        "polars": lambda: ii.p.rank("dense"),
        "pyarrow": lambda: pc.rank(ii.a, sort_keys="ascending", tiebreaker="dense"),
        "pandas": lambda: ii.d.rank(method="dense")})
    k1, k2 = d("keys_lex_a"), d("keys_lex_b")
    tbl_lex = pa.table({"a": k1.a, "b": k2.a})
    df_lex = pl.DataFrame({"a": k1.p, "b": k2.p})
    case(f, "lexsort (2 int32 keys)", n, n * 16, {
        "arrowmetal": lambda: am.lexsort_indices([k1.g, k2.g]),
        "polars": lambda: df_lex.select(pl.arg_sort_by(["a", "b"])),
        "pyarrow": lambda: pc.sort_indices(tbl_lex, sort_keys=[("a", "ascending"), ("b", "ascending")]),
        "numpy": lambda: np.lexsort((k2.n, k1.n))})
    case(f, "partition_nth_indices (n/2)", n, n * 12, {
        "arrowmetal": lambda: ii.g.partition_nth_indices(n // 2),
        "polars": None,
        "pyarrow": lambda: pc.partition_nth_indices(ii.a, pivot=n // 2),
        "numpy": lambda: np.argpartition(ii.n, n // 2)},
        notes={"polars": "polars has no partial-partition function"})
    keys = d("keys_1000")
    case(f, "unique (int32, 1000 distinct)", n, n * 4, {
        "arrowmetal": lambda: keys.g.unique(),
        "polars": lambda: keys.p.unique(),
        "pyarrow": lambda: pc.unique(keys.a),
        "pandas": lambda: pd.unique(keys.n)})
    case(f, "value_counts (int32, 1000 distinct)", n, n * 4, {
        "arrowmetal": lambda: keys.g.value_counts(),
        "polars": lambda: keys.p.value_counts(),
        "pyarrow": lambda: pc.value_counts(keys.a),
        "pandas": lambda: keys.d.value_counts()})
    case(f, "dictionary_encode (int32)", n, n * 8, {
        "arrowmetal": lambda: keys.g.dictionary_encode(),
        "polars": None,
        "pyarrow": lambda: pc.dictionary_encode(keys.a),
        "pandas": lambda: keys.d.astype("category")},
        notes={"polars": "polars has no eager dictionary_encode for an integer Series"})
    if sd is not None:
        case(f, "dictionary_encode (utf8)", sd.n, sd.s.nbytes, {
            "arrowmetal": lambda: sd.s.g.dictionary_encode(),
            "polars": lambda: sd.s.p.cast(pl.Categorical),
            "pyarrow": lambda: pc.dictionary_encode(sd.s.a),
            "pandas": lambda: sd.s.d.astype("category")})


def _lex_builders():
    def a(self):
        v = self.rng.integers(0, 1000, size=self.n, dtype=np.int32)
        return Col(pa.array(v), v)

    def b(self):
        v = self.rng.integers(0, 100000, size=self.n, dtype=np.int32)
        return Col(pa.array(v), v)
    Data._build_keys_lex_a = a
    Data._build_keys_lex_b = b
    Data._build_keys_1000 = lambda self: self.keys(1000)


_lex_builders()


def family_groupby(d, n):
    f = "group-by"
    vals = d("i64")
    for distinct in (1_000, 100_000, 10_000_000):
        if distinct > n:
            continue
        k = d.keys(distinct)
        df = pl.DataFrame({"k": k.p, "x": vals.p})
        tbl = pa.table({"k": k.a, "x": vals.a})
        pdf = pd.DataFrame({"k": k.n, "x": vals.d})
        B = n * 12
        for agg, am_fn, pl_agg, pa_agg, pd_agg in [
            ("sum", "sum", pl.col("x").sum(), "sum", "sum"),
            ("mean", "mean", pl.col("x").mean(), "mean", "mean"),
            ("min", "min", pl.col("x").min(), "min", "min"),
            ("max", "max", pl.col("x").max(), "max", "max"),
            ("count", "count", pl.col("x").count(), "count", "count"),  # noqa: E501
        ]:
            def make_am(name=am_fn, _k=k, _v=vals):
                return lambda: getattr(am.group_by([_k.g]), name)(_v.g)
            case(f, f"{agg} by int32 key ({distinct} groups)", n, B, {
                "arrowmetal": make_am(),
                "polars": (lambda _d=df, _a=pl_agg: _d.group_by("k").agg(_a)),
                "pyarrow": (lambda _t=tbl, _a=pa_agg: _t.group_by("k").aggregate([("x", _a)])),
                "pandas": (lambda _p=pdf, _a=pd_agg: _p.groupby("k", sort=False)["x"].agg(_a)),
            })
        # utf8 keys of the same cardinality
        sk = d.str_keys(distinct)
        df_s = pl.DataFrame({"k": sk.p, "x": vals.p})
        tbl_s = pa.table({"k": sk.a, "x": vals.a})
        pdf_s = pd.DataFrame({"k": sk.d, "x": vals.d})
        SB = n * 20
        case(f, f"sum by utf8 key ({distinct} distinct)", n, SB, {
            "arrowmetal": lambda _k=sk: am.group_by([_k.g]).sum(vals.g),
            "polars": lambda _d=df_s: _d.group_by("k").agg(pl.col("x").sum()),
            "pyarrow": lambda _t=tbl_s: _t.group_by("k").aggregate([("x", "sum")]),
            "pandas": lambda _p=pdf_s: _p.groupby("k", sort=False)["x"].sum()})
        # two key columns
        side = max(2, int(np.ceil(np.sqrt(distinct))))
        ka = Col(pa.array(d.rng.integers(0, side, size=n, dtype=np.int32)))
        kb = Col(pa.array(d.rng.integers(0, side, size=n, dtype=np.int32)))
        df_2 = pl.DataFrame({"a": ka.p, "b": kb.p, "x": vals.p})
        tbl_2 = pa.table({"a": ka.a, "b": kb.a, "x": vals.a})
        pdf_2 = pd.DataFrame({"a": ka.n, "b": kb.n, "x": vals.d})
        case(f, f"sum by two int32 keys (~{side * side} groups)", n, n * 16, {
            "arrowmetal": lambda _a=ka, _b=kb: am.group_by([_a.g, _b.g]).sum(vals.g),
            "polars": lambda _d=df_2: _d.group_by(["a", "b"]).agg(pl.col("x").sum()),
            "pyarrow": lambda _t=tbl_2: _t.group_by(["a", "b"]).aggregate([("x", "sum")]),
            "pandas": lambda _p=pdf_2: _p.groupby(["a", "b"], sort=False)["x"].sum()})
        del ka, kb, df_2, tbl_2, pdf_2
        gc.collect()

    # the remaining grouped aggregates, at 100k groups
    distinct = 100_000 if n >= 100_000 else 1_000
    k = d.keys(distinct)
    fl = d("f64")
    df = pl.DataFrame({"k": k.p, "x": fl.p})
    tbl = pa.table({"k": k.a, "x": fl.a})
    pdf = pd.DataFrame({"k": k.n, "x": fl.d})
    B = n * 12
    case(f, f"variance by key ({distinct} groups)", n, B, {
        "arrowmetal": lambda: am.group_by([k.g]).variance(fl.g, ddof=1),
        "polars": lambda: df.group_by("k").agg(pl.col("x").var()),
        "pyarrow": lambda: tbl.group_by("k").aggregate([("x", "variance")]),
        "pandas": lambda: pdf.groupby("k", sort=False)["x"].var()})
    case(f, f"count_distinct by key ({distinct} groups)", n, B, {
        "arrowmetal": lambda: am.group_by([k.g]).count_distinct(fl.g),
        "polars": lambda: df.group_by("k").agg(pl.col("x").n_unique()),
        "pyarrow": lambda: tbl.group_by("k").aggregate([("x", "count_distinct")]),
        "pandas": lambda: pdf.groupby("k", sort=False)["x"].nunique()})
    case(f, f"first by key ({distinct} groups)", n, B, {
        "arrowmetal": lambda: am.group_by([k.g]).first(fl.g),
        "polars": lambda: df.group_by("k").agg(pl.col("x").first()),
        "pyarrow": lambda: tbl.group_by("k", use_threads=False).aggregate([("x", "first")]),
        "pandas": lambda: pdf.groupby("k", sort=False)["x"].first()})
    case(f, f"list by key ({distinct} groups)", n, n * 20, {
        "arrowmetal": lambda: am.group_by([k.g]).list(fl.g),
        "polars": lambda: df.group_by("k").agg(pl.col("x")),
        "pyarrow": lambda: tbl.group_by("k").aggregate([("x", "list")]),
        "pandas": lambda: pdf.groupby("k", sort=False)["x"].apply(list)},
        notes={"pandas": "pandas has no vectorised list aggregation; apply(list) is its idiom"})


def family_join(d, n):
    f = "join"
    build_n = max(1000, n // 50)
    rng = np.random.default_rng(SEED + 7)
    right_keys = np.arange(build_n, dtype=np.int64)
    right_vals = rng.integers(0, 1000, size=build_n, dtype=np.int64)
    left_keys = rng.integers(0, build_n * 2, size=n, dtype=np.int64)   # ~50% match
    left_vals = rng.integers(0, 1000, size=n, dtype=np.int64)
    a_rk, a_rv = pa.array(right_keys), pa.array(right_vals)
    a_lk, a_lv = pa.array(left_keys), pa.array(left_vals)
    g_rk, g_rv = am.array(a_rk), am.array(a_rv)
    g_lk, g_lv = am.array(a_lk), am.array(a_lv)
    left_pl = pl.DataFrame({"k": a_lk, "v": a_lv})
    right_pl = pl.DataFrame({"k": a_rk, "w": a_rv})
    left_tb = pa.table({"k": a_lk, "v": a_lv})
    right_tb = pa.table({"k": a_rk, "w": a_rv})
    left_pd = pd.DataFrame({"k": left_keys, "v": left_vals})
    right_pd = pd.DataFrame({"k": right_keys, "w": right_vals})

    def am_join():
        pos = g_lk.index_in(g_rk)          # int32 position in the build side, null when absent
        keep = pos.is_valid()
        return g_lv.filter(keep), g_rv.take(pos.drop_null())

    case(f, f"inner hash join ({n} x {build_n} on int64)", n, n * 16 + build_n * 16, {
        "arrowmetal": am_join,
        "polars": lambda: left_pl.join(right_pl, on="k", how="inner"),
        "pyarrow": lambda: left_tb.join(right_tb, keys="k", join_type="inner"),
        "pandas": lambda: left_pd.merge(right_pd, on="k", how="inner")},
        notes={"arrowmetal": "no join kernel: composed from index_in + is_valid + filter + take "
                             "(requires unique build-side keys)"})
    del g_rk, g_rv, g_lk, g_lv, left_pl, right_pl, left_tb, right_tb, left_pd, right_pd
    gc.collect()


def family_strings(sd):
    f = "strings"
    n = sd.n
    s, SB = sd.s, sd.s.nbytes
    s2 = sd("s2")
    padded = sd("padded")
    ints = sd("ints")
    numtext = sd("numeric_text")
    case(f, "char_length", n, SB, {
        "arrowmetal": lambda: s.g.char_length(),
        "polars": lambda: s.p.str.len_chars(),
        "pyarrow": lambda: pc.utf8_length(s.a),
        "pandas": lambda: s.d.str.len()})
    case(f, "upper", n, SB * 2, {
        "arrowmetal": lambda: s.g.upper(),
        "polars": lambda: s.p.str.to_uppercase(),
        "pyarrow": lambda: pc.utf8_upper(s.a),
        "pandas": lambda: s.d.str.upper()})
    case(f, "lower", n, SB * 2, {
        "arrowmetal": lambda: s.g.lower(),
        "polars": lambda: s.p.str.to_lowercase(),
        "pyarrow": lambda: pc.utf8_lower(s.a),
        "pandas": lambda: s.d.str.lower()})
    case(f, 'contains("north")', n, SB, {
        "arrowmetal": lambda: s.g.str_contains("north"),
        "polars": lambda: s.p.str.contains("north", literal=True),
        "pyarrow": lambda: pc.match_substring(s.a, "north"),
        "pandas": lambda: s.d.str.contains("north", regex=False)})
    case(f, 'starts_with("cust_1")', n, SB, {
        "arrowmetal": lambda: s.g.starts_with("cust_1"),
        "polars": lambda: s.p.str.starts_with("cust_1"),
        "pyarrow": lambda: pc.starts_with(s.a, "cust_1"),
        "pandas": lambda: s.d.str.startswith("cust_1")})
    case(f, 'match_like("cust%") [pure prefix]', n, SB, {
        "arrowmetal": lambda: s.g.match_like("cust%"),
        "polars": lambda: s.p.str.starts_with("cust"),
        "pyarrow": lambda: pc.match_like(s.a, "cust%"),
        "pandas": lambda: s.d.str.match("cust")},
        notes={"arrowmetal": "a pure prefix pattern, which Regex.likePredicate maps onto the GPU "
                             "starts_with kernel",
               "polars": "no SQL LIKE; the equivalent prefix predicate"})
    case(f, 'match_like("cust_1%") [_ wildcard]', n, SB, {
        "arrowmetal": lambda: s.g.match_like("cust_1%"),
        "polars": lambda: s.p.str.contains(r"^cust.1"),
        "pyarrow": lambda: pc.match_like(s.a, "cust_1%"),
        "pandas": lambda: s.d.str.match(r"cust.1")},
        notes={"arrowmetal": "`_` is LIKE's single-character wildcard, so this is not a pure prefix "
                             "and takes the ICU host path",
               "polars": "no SQL LIKE; the equivalent anchored regex"})
    case(f, "match_substring_regex (literal)", n, SB, {
        "arrowmetal": lambda: s.g.match_substring_regex("north"),
        "polars": lambda: s.p.str.contains("north"),
        "pyarrow": lambda: pc.match_substring_regex(s.a, "north"),
        "pandas": lambda: s.d.str.contains("north", regex=True)})
    case(f, "match_substring_regex (real regex)", n, SB, {
        "arrowmetal": lambda: s.g.match_substring_regex(r"_[0-9]{2}3_(?:north|west)"),
        "polars": lambda: s.p.str.contains(r"_[0-9]{2}3_(?:north|west)"),
        "pyarrow": lambda: pc.match_substring_regex(s.a, r"_[0-9]{2}3_(?:north|west)"),
        "pandas": lambda: s.d.str.contains(r"_[0-9]{2}3_(?:north|west)", regex=True)})
    case(f, 'replace_substring("cust" -> "cx")', n, SB * 2, {
        "arrowmetal": lambda: s.g.replace("cust", "cx"),
        "polars": lambda: s.p.str.replace_all("cust", "cx", literal=True),
        "pyarrow": lambda: pc.replace_substring(s.a, "cust", "cx"),
        "pandas": lambda: s.d.str.replace("cust", "cx", regex=False)})
    case(f, 'split_pattern("_")', n, SB * 2, {
        "arrowmetal": lambda: s.g.split_pattern("_"),
        "polars": lambda: s.p.str.split("_"),
        "pyarrow": lambda: pc.split_pattern(s.a, "_"),
        "pandas": lambda: s.d.str.split("_")})
    case(f, "trim (whitespace)", n, padded.nbytes * 2, {
        "arrowmetal": lambda: padded.g.utf8_trim(),
        "polars": lambda: padded.p.str.strip_chars(),
        "pyarrow": lambda: pc.utf8_trim_whitespace(padded.a),
        "pandas": lambda: padded.d.str.strip()})
    case(f, "pad_left (width 20)", n, SB * 3, {
        "arrowmetal": lambda: s.g.pad_left(20, "*"),
        "polars": lambda: s.p.str.pad_start(20, "*"),
        "pyarrow": lambda: pc.utf8_lpad(s.a, 20, padding="*"),
        "pandas": lambda: s.d.str.rjust(20, "*")})
    case(f, "slice_codeunits [5:10]", n, SB, {
        "arrowmetal": lambda: s.g.slice_codeunits(5, 10),
        "polars": lambda: s.p.str.slice(5, 5),
        "pyarrow": lambda: pc.utf8_slice_codeunits(s.a, 5, 10),
        "pandas": lambda: s.d.str[5:10]})
    case(f, "concat (a + '-' + b)", n, SB * 3, {
        "arrowmetal": lambda: s.g.str_concat(s2.g, "-"),
        "polars": lambda: s.p + "-" + s2.p,
        "pyarrow": lambda: pc.binary_join_element_wise(s.a, s2.a, "-"),
        "pandas": lambda: s.d + "-" + s2.d})
    case(f, "is_alpha", n, SB, {
        "arrowmetal": lambda: s.g.utf8_is_alpha(),
        "polars": lambda: s.p.str.contains(r"^[^\W\d_]+$"),
        "pyarrow": lambda: pc.utf8_is_alpha(s.a),
        "pandas": lambda: s.d.str.isalpha()},
        notes={"polars": "no is_alpha predicate; the equivalent anchored regex"})
    case(f, "to_strings (int64 -> utf8)", n, n * 16, {
        "arrowmetal": lambda: ints.g.to_strings(),
        "polars": lambda: ints.p.cast(pl.Utf8),
        "pyarrow": lambda: pc.cast(ints.a, pa.string()),
        "pandas": lambda: ints.d.astype(str)})
    case(f, "parse (utf8 -> int64)", n, numtext.nbytes + n * 8, {
        "arrowmetal": lambda: numtext.g.parse("int64"),
        "polars": lambda: numtext.p.cast(pl.Int64),
        "pyarrow": lambda: pc.cast(numtext.a, pa.int64()),
        "pandas": lambda: numtext.d.astype("int64")})
    subset = pa.array(sd.vocab[:100])
    subset_g = am.array(subset)
    subset_list = sd.vocab[:100]
    case(f, "is_in (utf8, 100-value set)", n, SB, {
        "arrowmetal": lambda: s.g.is_in(subset_g),
        "polars": lambda: s.p.is_in(subset_list),
        "pyarrow": lambda: pc.is_in(s.a, value_set=subset),
        "pandas": lambda: s.d.isin(subset_list)})


def family_temporal(d, n):
    f = "temporal"
    ts, ts2 = d("ts"), d("ts2")
    pdt = pd.Series(ts.n.astype("datetime64[us]"))
    pdt2 = pd.Series(ts2.n.astype("datetime64[us]"))
    B = n * 12
    case(f, "year", n, B, {
        "arrowmetal": lambda: ts.g.year(),
        "polars": lambda: ts.p.dt.year(),
        "pyarrow": lambda: pc.year(ts.a),
        "pandas": lambda: pdt.dt.year})
    case(f, "month", n, B, {
        "arrowmetal": lambda: ts.g.month(),
        "polars": lambda: ts.p.dt.month(),
        "pyarrow": lambda: pc.month(ts.a),
        "pandas": lambda: pdt.dt.month})
    case(f, "day", n, B, {
        "arrowmetal": lambda: ts.g.day(),
        "polars": lambda: ts.p.dt.day(),
        "pyarrow": lambda: pc.day(ts.a),
        "pandas": lambda: pdt.dt.day})
    case(f, "floor_temporal (day)", n, n * 16, {
        "arrowmetal": lambda: ts.g.floor_temporal("day"),
        "polars": lambda: ts.p.dt.truncate("1d"),
        "pyarrow": lambda: pc.floor_temporal(ts.a, unit="day"),
        "pandas": lambda: pdt.dt.floor("D")})
    case(f, "add_duration (+1h)", n, n * 16, {
        "arrowmetal": lambda: ts.g.add_duration(3_600_000_000),
        "polars": lambda: ts.p.dt.offset_by("1h"),
        "pyarrow": lambda: pc.add(ts.a, pa.scalar(3_600_000_000, pa.duration("us"))),
        "pandas": lambda: pdt + pd.Timedelta("1h")})
    case(f, "days_between", n, n * 24, {
        "arrowmetal": lambda: ts.g.days_between(ts2.g),
        "polars": lambda: (ts2.p - ts.p).dt.total_days(),
        "pyarrow": lambda: pc.days_between(ts.a, ts2.a),
        "pandas": lambda: (pdt2 - pdt).dt.days},
        notes={"polars": "no days_between; duration difference in whole days",
               "pandas": "same"})
    case(f, "week", n, B, {
        "arrowmetal": lambda: ts.g.week(),
        "polars": lambda: ts.p.dt.week(),
        "pyarrow": lambda: pc.week(ts.a),
        "pandas": lambda: pdt.dt.isocalendar().week})
    case(f, "strftime (%Y-%m-%d)", n, n * 20, {
        "arrowmetal": lambda: ts.g.strftime("%Y-%m-%d"),
        "polars": lambda: ts.p.dt.strftime("%Y-%m-%d"),
        "pyarrow": lambda: pc.strftime(ts.a, format="%Y-%m-%d"),
        "pandas": lambda: pdt.dt.strftime("%Y-%m-%d")})
    case(f, "assume_timezone (America/New_York)", n, n * 16, {
        "arrowmetal": lambda: ts.g.assume_timezone("America/New_York", ambiguous="earliest",
                                                   nonexistent="latest"),
        "polars": lambda: ts.p.dt.replace_time_zone("America/New_York", ambiguous="earliest",
                                                    non_existent="null"),
        "pyarrow": lambda: pc.assume_timezone(ts.a, "America/New_York", ambiguous="earliest",
                                              nonexistent="latest"),
        "pandas": lambda: pdt.dt.tz_localize("America/New_York", ambiguous=True,
                                             nonexistent="shift_forward")})


def family_window(d, n):
    f = "window"
    fl = d("f64_nn")
    i = d("i64")
    B = n * 16
    case(f, "cumulative_sum (float64)", n, B, {
        "arrowmetal": lambda: fl.g.cumulative_sum(),
        "polars": lambda: fl.p.cum_sum(),
        "pyarrow": lambda: pc.cumulative_sum(fl.a),
        "pandas": lambda: fl.d.cumsum()})
    case(f, "cumulative_prod (float64)", n, B, {
        "arrowmetal": lambda: fl.g.cumulative_prod(),
        "polars": lambda: fl.p.cum_prod(),
        "pyarrow": lambda: pc.cumulative_prod(fl.a),
        "pandas": lambda: fl.d.cumprod()})
    case(f, "shift (lag 1, int64)", n, B, {
        "arrowmetal": lambda: i.g.shift(1),
        "polars": lambda: i.p.shift(1),
        "pyarrow": None,
        "pandas": lambda: i.d.shift(1)},
        notes={"pyarrow": "pyarrow.compute has no shift/lag kernel"})
    case(f, "pairwise_diff (float64)", n, B, {
        "arrowmetal": lambda: fl.g.pairwise_diff(),
        "polars": lambda: fl.p.diff(),
        "pyarrow": lambda: pc.pairwise_diff(fl.a),
        "pandas": lambda: fl.d.diff()})
    case(f, "rolling_sum (window 64)", n, B, {
        "arrowmetal": lambda: fl.g.rolling_sum(64),
        "polars": lambda: fl.p.rolling_sum(64),
        "pyarrow": None,
        "pandas": lambda: fl.d.rolling(64).sum()},
        notes={"pyarrow": "pyarrow.compute has no rolling-window kernels"})
    case(f, "rolling_mean (window 64)", n, B, {
        "arrowmetal": lambda: fl.g.rolling_mean(64),
        "polars": lambda: fl.p.rolling_mean(64),
        "pyarrow": None,
        "pandas": lambda: fl.d.rolling(64).mean()})


def family_decimal(d, n):
    f = "decimal"
    a, b = d("dec"), d("dec2")
    pl_a = pl.Series("x", a.a)
    pl_b = pl.Series("y", b.a)
    pd_a = pd.Series(pd.arrays.ArrowExtensionArray(a.a))
    pd_b = pd.Series(pd.arrays.ArrowExtensionArray(b.a))
    B = n * 48
    case(f, "decimal add (128-bit)", n, B, {
        "arrowmetal": lambda: a.g.decimal_add(b.g),
        "polars": lambda: pl_a + pl_b,
        "pyarrow": lambda: pc.add(a.a, b.a),
        "pandas": lambda: pd_a + pd_b})
    dec3 = pa.scalar(decimal.Decimal("3"), pa.decimal128(2, 0))
    lim = decimal.Decimal("500000.0000")
    dec_lim = pa.scalar(lim, pa.decimal128(18, 4))
    case(f, "decimal multiply (by scalar)", n, n * 32, {
        "arrowmetal": lambda: a.g.decimal_mul(3),
        "polars": lambda: pl_a * 3,
        "pyarrow": lambda: pc.multiply(a.a, dec3),
        "pandas": lambda: pd_a * 3})
    case(f, "decimal compare (> scalar)", n, n * 17, {
        "arrowmetal": lambda: a.g > lim,
        "polars": lambda: pl_a > lim,
        "pyarrow": lambda: pc.greater(a.a, dec_lim),
        "pandas": lambda: pd_a > lim})
    case(f, "decimal sum", n, n * 16, {
        "arrowmetal": lambda: a.g.sum(),
        "polars": lambda: pl_a.sum(),
        "pyarrow": lambda: pc.sum(a.a),
        "pandas": lambda: pd_a.sum()})
    case(f, "decimal round (2 places)", n, n * 32, {
        "arrowmetal": lambda: a.g.decimal_round(2),
        "polars": lambda: pl_a.round(2),
        "pyarrow": lambda: pc.round(a.a, ndigits=2),
        "pandas": lambda: pd_a.round(2)})


def family_nested(d, n):
    f = "nested"
    m = min(n, 10_000_000)
    rng = np.random.default_rng(SEED + 3)
    per_row = 4
    child = rng.integers(0, 1000, size=m * per_row, dtype=np.int64)
    offsets = np.arange(m + 1, dtype=np.int32) * per_row
    lst = pa.ListArray.from_arrays(pa.array(offsets), pa.array(child))
    a_x = pa.array(rng.integers(0, 1000, size=m, dtype=np.int64))
    a_y = pa.array(rng.random(m))
    st = pa.StructArray.from_arrays([a_x, a_y], names=["x", "y"])
    g_lst, g_st = am.array(lst), am.array(st)
    pl_lst = pl.Series("l", lst)
    pl_st = pl.Series("s", st)
    pd_lst = pd.Series(pd.arrays.ArrowExtensionArray(lst))
    pd_st = pd.Series(pd.arrays.ArrowExtensionArray(st))
    B = m * per_row * 8
    case(f, "list_value_length", m, B, {
        "arrowmetal": lambda: g_lst.list_value_length(),
        "polars": lambda: pl_lst.list.len(),
        "pyarrow": lambda: pc.list_value_length(lst),
        "pandas": lambda: pd_lst.list.len()})
    case(f, "list_flatten", m, B * 2, {
        "arrowmetal": lambda: g_lst.list_flatten(),
        "polars": lambda: pl_lst.explode(),
        "pyarrow": lambda: pc.list_flatten(lst),
        "pandas": lambda: pd_lst.explode()})
    case(f, "list_element (index 1)", m, B, {
        "arrowmetal": lambda: g_lst.list_element(1),
        "polars": lambda: pl_lst.list.get(1),
        "pyarrow": lambda: pc.list_element(lst, 1),
        "pandas": lambda: pd_lst.list[1]})
    case(f, "struct_field", m, m * 16, {
        "arrowmetal": lambda: g_st.struct_field("y"),
        "polars": lambda: pl_st.struct.field("y"),
        "pyarrow": lambda: pc.struct_field(st, "y"),
        "pandas": lambda: pd_st.struct.field("y")})
    del g_lst, g_st, pl_lst, pl_st, pd_lst, pd_st, lst, st, child
    gc.collect()


def family_chains(d, n):
    f = "chains"
    rng = np.random.default_rng(SEED + 5)
    region = rng.integers(0, 5, size=n, dtype=np.int32)
    amount = (rng.random(n) * 500)
    a_r, a_a = pa.array(region), pa.array(amount)
    g_r, g_a = am.array(a_r), am.array(a_a)
    keys = d.keys(1000)
    g_k = keys.g
    df = pl.DataFrame({"region": a_r, "amount": a_a, "k": keys.a})
    lz = df.lazy()
    tbl = pa.table({"region": a_r, "amount": a_a, "k": keys.a})
    pdf = pd.DataFrame({"region": region, "amount": amount, "k": keys.n})
    B = n * 12

    def q1():
        return g_a.filter((g_r == 2) & (g_a > 100.0)).sum()

    def q1_batched():
        with am.batch():
            return g_a.filter((g_r == 2) & (g_a > 100.0)).sum()

    case(f, "filter two columns + sum", n, B, {
        "arrowmetal": q1,
        "polars": lambda: lz.filter((pl.col("region") == 2) & (pl.col("amount") > 100.0))
                            .select(pl.col("amount").sum()).collect(),
        "pyarrow": lambda: pc.sum(pc.filter(a_a, pc.and_(pc.equal(a_r, 2), pc.greater(a_a, 100.0)))),
        "numpy": lambda: amount[(region == 2) & (amount > 100.0)].sum()})
    case(f, "filter two columns + sum [batched]", n, B, {
        "arrowmetal": q1_batched,
        "polars": None, "pyarrow": None, "pandas": None},
        notes={"arrowmetal": "same chain inside `with am.batch()`; compare with the unbatched row",
               "polars": "batching is an ArrowMetal-only concept",
               "pyarrow": "batching is an ArrowMetal-only concept",
               "pandas": "batching is an ArrowMetal-only concept"})

    def q2():
        m = g_r == 2
        return am.group_by([g_k.filter(m)]).sum(g_a.filter(m))

    def q2_batched():
        with am.batch():
            m = g_r == 2
            return am.group_by([g_k.filter(m)]).sum(g_a.filter(m))

    case(f, "group-by after filter", n, n * 20, {
        "arrowmetal": q2,
        "polars": lambda: lz.filter(pl.col("region") == 2).group_by("k")
                            .agg(pl.col("amount").sum()).collect(),
        "pyarrow": lambda: tbl.filter(pc.equal(a_r, 2)).group_by("k").aggregate([("amount", "sum")]),
        "pandas": lambda: pdf[pdf.region == 2].groupby("k", sort=False)["amount"].sum()})
    case(f, "group-by after filter [batched]", n, n * 20, {
        "arrowmetal": q2_batched,
        "polars": None, "pyarrow": None, "pandas": None},
        notes={"arrowmetal": "same chain inside `with am.batch()`"})

    # Indices safely inside the ~50% that survives the filter, so no side has to measure the
    # filtered length first (which would force a sync inside the batch).
    small_idx_np = rng.integers(0, max(1, n // 4), size=max(1, n // 8), dtype=np.int32)
    small_idx_pa = pa.array(small_idx_np)
    small_idx_pl = pl.Series(small_idx_np)
    g_idx = am.array(small_idx_pa)
    pl_amount = pl.Series("a", a_a)

    def q3():
        return g_a.filter(g_a > 250.0).take(g_idx)

    def q3_batched():
        with am.batch():
            return g_a.filter(g_a > 250.0).take(g_idx)

    case(f, "compare + filter + take", n, n * 20, {
        "arrowmetal": q3,
        "polars": lambda: pl_amount.filter(pl_amount > 250.0).gather(small_idx_pl),
        "pyarrow": lambda: pc.take(pc.filter(a_a, pc.greater(a_a, 250.0)), small_idx_pa),
        "numpy": lambda: amount[amount > 250.0][small_idx_np]})
    case(f, "compare + filter + take [batched]", n, n * 20, {
        "arrowmetal": q3_batched,
        "polars": None, "pyarrow": None, "pandas": None},
        notes={"arrowmetal": "same chain inside `with am.batch()`"})
    del g_r, g_a, df, lz, tbl, pdf
    gc.collect()


def family_small(sizes):
    """The latency floor: the same three operations at small row counts, batched and unbatched."""
    f = "latency"
    for n in sizes:
        rng = np.random.default_rng(SEED + 11)
        v = rng.integers(-1000, 1001, size=n, dtype=np.int64)
        a = pa.array(v)
        g = am.array(a)
        p = pl.Series("x", a)
        dser = pd.Series(v)
        k = rng.integers(0, min(1000, n), size=n, dtype=np.int32)
        a_k = pa.array(k)
        g_k = am.array(a_k)
        df = pl.DataFrame({"k": a_k, "x": a})
        tbl = pa.table({"k": a_k, "x": a})
        pdf = pd.DataFrame({"k": k, "x": v})
        B = n * 8
        case(f, "sum(int64)", n, B, {
            "arrowmetal": lambda: g.sum(),
            "polars": lambda: p.sum(),
            "pyarrow": lambda: pc.sum(a),
            "pandas": lambda: dser.sum()})

        def sum_batched():
            with am.batch():
                return g.sum()
        case(f, "sum(int64) [batched]", n, B, {
            "arrowmetal": sum_batched, "polars": None, "pyarrow": None, "pandas": None},
            notes={"arrowmetal": "one-call batch; shows the fixed cost of the batch itself"})
        case(f, "filter(int64 > 0)", n, B, {
            "arrowmetal": lambda: g.filter_where(">", 0),
            "polars": lambda: p.filter(p > 0),
            "pyarrow": lambda: pc.filter(a, pc.greater(a, 0)),
            "pandas": lambda: dser[dser > 0]})

        def filter_batched():
            with am.batch():
                return g.filter_where(">", 0)
        case(f, "filter(int64 > 0) [batched]", n, B, {
            "arrowmetal": filter_batched, "polars": None, "pyarrow": None, "pandas": None})
        case(f, "group-by sum (1000 keys)", n, n * 12, {
            "arrowmetal": lambda: am.group_by([g_k]).sum(g),
            "polars": lambda: df.group_by("k").agg(pl.col("x").sum()),
            "pyarrow": lambda: tbl.group_by("k").aggregate([("x", "sum")]),
            "pandas": lambda: pdf.groupby("k", sort=False)["x"].sum()})

        def gb_batched():
            with am.batch():
                return am.group_by([g_k]).sum(g)
        case(f, "group-by sum (1000 keys) [batched]", n, n * 12, {
            "arrowmetal": gb_batched, "polars": None, "pyarrow": None, "pandas": None})
        del g, g_k, p, dser, df, tbl, pdf
        gc.collect()


# ---------------------------------------------------------------- reporting

# Numbers copied verbatim from docs/BENCHMARKS.md (round 7 and round 6, Apple M4 Max, 50M rows).
# The Swift benchmark is NOT rebuilt or re-run here; these are quoted for context only.
SWIFT_BASELINE = [
    # (family, op, rows, metal_ms, cpu16_ms, cpu16_cpu_ms, source)
    ("sort", "argsort int64", 50_000_000, 128.52, 653.97, 5017.1, "round 7"),
    ("sort", "sort float64", 50_000_000, 138.52, 591.97, 4468.6, "round 7"),
    ("sort", "top_k (k=100, int64)", 50_000_000, 128.67, 1.96, 25.6, "round 7"),
    ("strings", 'contains("north")', 10_000_000, 1.65, 17.77, 250.0, "round 7"),
    ("strings", 'starts_with("cust_1")', 10_000_000, 1.75, 3.10, 42.7, "round 7"),
    ("reductions", "sum(int64, 10% nulls)", 50_000_000, 1.07, 4.87, 66.6, "round 6"),
    ("compare+select", "filter int64 (30% kept)", 50_000_000, 2.95, 3.49, 48.8, "round 6"),
    ("compare+select", "take (n/2 random indices)", 50_000_000, 5.91, 11.75, 154.6, "round 6"),
    ("chains", "filter two columns + sum", 50_000_000, 1.75, 6.27, 86.5, "round 6"),
]

BASELINE_LIBS = ("polars", "pyarrow", "pandas", "numpy")


def fmt_ratio(r):
    """baseline / ArrowMetal, with enough digits to show how far off a bad row is."""
    if r is None:
        return "--"
    return f"{r:.3f}x" if r < 0.1 else f"{r:.2f}x"


def fmt_ms(v):
    """Milliseconds with enough digits to stay meaningful under the dispatch floor."""
    if v is None:
        return "--"
    if v < 1:
        return f"{v:.3f}"
    if v < 100:
        return f"{v:.2f}"
    return f"{v:.0f}"


def fmt_gbs(g):
    if g is None:
        return "--"
    if g >= 100:
        return f"{g:.0f}"
    if g >= 10:
        return f"{g:.1f}"
    return f"{g:.2f}"


def verdict(ratio):
    if ratio is None:
        return "n/a"
    if ratio >= 3.0:
        return "OK"
    if ratio >= 1.0:
        return "WARN"
    return "FAIL"


VERDICT_MARK = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌", "n/a": "—"}


def build_report(csv_path, elapsed_s):
    by_key = {}
    for r in ROWS:
        by_key.setdefault((r["family"], r["op"], r["rows"]), {})[r["library"]] = r

    families = []
    for key in _ORDER:
        if key[0] not in families:
            families.append(key[0])

    shortfalls = []
    verdicts = []
    errors = 0
    lines = []
    lines.append("# The complete comparison matrix")
    lines.append("")
    lines.append(f"Generated by `Benchmarks/full_matrix.py` on {datetime.date.today().isoformat()}, "
                 f"Apple M4 Max, 16 CPU cores, 64 GB unified memory, Darwin {os.uname().release}.")
    lines.append("")
    lines.append(f"ArrowMetal {am.__version__} on `{am.device_name()}` against "
                 f"Polars {pl.__version__} ({pl.thread_pool_size()} threads), "
                 f"pyarrow {pa.__version__} ({pa.cpu_count()} threads), "
                 f"pandas {pd.__version__}, numpy {np.__version__}.")
    lines.append("")
    lines.append(f"Run time {elapsed_s / 60:.1f} minutes; raw numbers in `{os.path.relpath(csv_path, ROOT)}`.")
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- Every number is the best wall time of up to five calls after one warm-up, with the "
                 "process CPU time (all threads) of that same call beside it. A slow call is repeated "
                 "fewer times, never fewer than twice; the CSV records the count.")
    lines.append("- **ratio** is the fastest baseline's wall time divided by ArrowMetal's. The verdict is "
                 "the project's own bar: ✅ at or above 3x, ⚠️ between 1x and 3x, "
                 "❌ slower than the fastest CPU library.")
    lines.append("- `--` in a library column means that library has no equivalent operation (the reason "
                 "is in the CSV's `note` column); `err` means the call raised, and the message is in the "
                 "CSV. Nothing is skipped silently.")
    lines.append("- Bandwidth (GB/s) is bytes touched (input + output) over wall time. Where ArrowMetal "
                 "and the best baseline are both near the machine's ~400 GB/s unified-memory ceiling the "
                 "operation is memory-bound and no ratio above ~1.5x is available to either side.")
    lines.append("")
    summary_at = len(lines)          # the verdict tally is filled in once every row is scored
    lines.append("")

    for fam in families:
        lines.append(f"## {fam}")
        lines.append("")
        lines.append("| op | rows | ArrowMetal wall / CPU ms | ArrowMetal GB/s | polars | pyarrow | "
                     "pandas / numpy | fastest baseline | ratio | verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|:--:|")
        for key in _ORDER:
            if key[0] != fam:
                continue
            row = by_key[key]
            amr = row.get("arrowmetal")
            cells = []
            best_lib, best_wall = None, None
            for lib in BASELINE_LIBS:
                r = row.get(lib)
                if r is None:
                    continue
                if r["status"] != "ok":
                    cells.append((lib, "--" if r["status"] == "no equivalent" else "err"))
                    continue
                cells.append((lib, f"{fmt_ms(r['wall_ms'])} / {fmt_ms(r['cpu_ms'])}"))
                if best_wall is None or r["wall_ms"] < best_wall:
                    best_lib, best_wall = lib, r["wall_ms"]
            cellmap = dict(cells)
            pandas_cell = cellmap.get("pandas") or cellmap.get("numpy") or "--"
            pandas_lib = "pandas" if "pandas" in cellmap else ("numpy" if "numpy" in cellmap else "")
            if pandas_lib == "numpy" and pandas_cell != "--":
                pandas_cell += " (numpy)"
            if amr is None or amr["status"] != "ok":
                am_cell = "err" if (amr and amr["status"] == "error") else "--"
                am_gbs = "--"
                ratio, vd = None, "FAIL" if amr and amr["status"] == "error" else "n/a"
                if amr and amr["status"] == "error":
                    shortfalls.append((key, None, best_lib, best_wall, amr, row))
                    errors += 1
            else:
                am_cell = f"**{fmt_ms(amr['wall_ms'])}** / {fmt_ms(amr['cpu_ms'])}"
                am_gbs = fmt_gbs(amr["gbs"])
                ratio = (best_wall / amr["wall_ms"]) if best_wall else None
                vd = verdict(ratio)
                if vd in ("WARN", "FAIL"):
                    shortfalls.append((key, ratio, best_lib, best_wall, amr, row))
            verdicts.append(vd)
            lines.append(
                f"| {key[1]} | {key[2]:,} | {am_cell} | {am_gbs} | {cellmap.get('polars', '--')} | "
                f"{cellmap.get('pyarrow', '--')} | {pandas_cell} | {best_lib or '--'} | "
                f"{fmt_ratio(ratio)} | {VERDICT_MARK[vd]} |")
        lines.append("")

    tally = {"OK": 0, "WARN": 0, "FAIL": 0, "n/a": 0}
    for v in verdicts:
        tally[v] = tally.get(v, 0) + 1
    lines[summary_at] = (
        f"**{len(_ORDER)} operations measured.** "
        f"{tally['OK']} at or above 3x (✅), {tally['WARN']} between 1x and 3x (⚠️), "
        f"{tally['FAIL']} slower than the fastest CPU library (❌), of which {errors} are operations "
        f"ArrowMetal does not have at all (the call raised). "
        f"{tally['OK'] * 100 // max(len(_ORDER), 1)}% of the surface meets the bar.\n")

    # ---- the Swift / vDSP baselines, copied
    lines.append("## The 16-core Swift / vDSP baselines (copied from docs/BENCHMARKS.md)")
    lines.append("")
    lines.append("These are **not** re-measured here. They are the numbers already recorded in "
                 "`docs/BENCHMARKS.md` for the in-process Swift benchmark (Metal vs all 16 CPU cores on "
                 "the same Arrow buffers), quoted so the CPU-side ceiling is visible next to the Python "
                 "libraries. Rows are matched to this matrix by operation name where one exists.")
    lines.append("")
    lines.append("| op | rows | Metal (Swift bench) | 16-core Swift wall / CPU ms | source |")
    lines.append("|---|---:|---:|---:|---|")
    for fam, op, rws, metal, cpu16, cpu16cpu, src in SWIFT_BASELINE:
        lines.append(f"| {op} | {rws:,} | {metal:.2f} | {cpu16:.2f} / {cpu16cpu:.1f} | "
                     f"docs/BENCHMARKS.md {src} |")
    lines.append("")

    # ---- shortfalls
    lines.append("## SHORTFALL: every row that is not at 3x")
    lines.append("")
    lines.append("Sorted by how far short of the 3x bar the row is (worst first). "
                 "❌ means a CPU library is faster than ArrowMetal outright.")
    lines.append("")
    lines.append("| op | rows | ratio | ArrowMetal ms | fastest baseline | baseline ms | "
                 "AM GB/s | baseline GB/s | likely cause |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---|")

    def sort_key(item):
        _key, ratio, _bl, _bw, _amr, _row = item
        return -1e9 if ratio is None else ratio
    for key, ratio, best_lib, best_wall, amr, row in sorted(shortfalls, key=sort_key):
        cause = diagnose(key, ratio, best_lib, amr, row)
        if amr is None or amr["status"] != "ok":
            lines.append(f"| {key[1]} | {key[2]:,} | err | -- | {best_lib or '--'} | "
                         f"{fmt_ms(best_wall)} | -- | -- | {cause} |")
            continue
        b = row.get(best_lib)
        lines.append(
            f"| {key[1]} | {key[2]:,} | {fmt_ratio(ratio)} | {fmt_ms(amr['wall_ms'])} | {best_lib} | "
            f"{fmt_ms(best_wall)} | {fmt_gbs(amr['gbs'])} | {fmt_gbs(b['gbs'])} | {cause} |")
    lines.append("")
    lines.append(f"{len(shortfalls)} of {len(_ORDER)} measured operations are below the 3x bar.")
    lines.append("")
    lines.append("## Reproducing")
    lines.append("")
    lines.append("```")
    lines.append("DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \\")
    lines.append("  swift build -c release --product ArrowMetalC")
    lines.append("PYTHONPATH=python python Benchmarks/full_matrix.py            # full run")
    lines.append("PYTHONPATH=python python Benchmarks/full_matrix.py --quick    # 1M-row smoke")
    lines.append("```")
    lines.append("")
    return "\n".join(lines), shortfalls


SORT_BASED = ("count_distinct", "quantile", "mode(", "median", "tdigest", "unique", "value_counts")

CAUSE_HINTS = [
    # (predicate on (family, op), cause)
    (lambda fam, op: "match_like" in op and "_ wildcard" in op,
     "Host fallback, correctly: `Regex.likePredicate` (Sources/ArrowMetal/Kernels/Regex.swift) maps "
     "only a pure prefix / suffix / contains / equality LIKE pattern onto a GPU predicate. `_` is "
     "LIKE's single-character wildcard, so this pattern is translated to an anchored regex and matched "
     "row by row with NSRegularExpression (ICU) on the CPU. A GPU wildcard matcher is the fix."),
    (lambda fam, op: "match_like" in op,
     "`match_like` should take the GPU prefix path here (Regex.likePredicate); if this row is short of "
     "3x the GPU predicate itself is the cost, not a fallback."),
    (lambda fam, op: "argsort utf8" in op,
     "`am_argsort` goes through `withPrimitive`, and `Sort.swift`'s type switch has no utf8 case, so "
     "there is no string sort at all - on the GPU or through the C ABI."),
    (lambda fam, op: "dictionary_encode (int32)" in op,
     "The C ABI only has `am_str_dictionary_encode`, which takes utf8. Swift already has the primitive "
     "path (`DictionaryCompute.dictionaryEncoded()`); it is simply not exported."),
    (lambda fam, op: fam == "decimal" and "sum" in op,
     "`am_reduce` goes through the primitive path, which rejects `d:p,s`. There is no decimal "
     "reduction kernel behind the C ABI."),
    (lambda fam, op: any(t in op for t in SORT_BASED),
     "Sort-based path: ArrowMetal answers this with a full GPU radix sort plus a run scan, where the "
     "CPU libraries use a hash table (count_distinct, mode, unique, value_counts) or a partial "
     "selection (quantile, tdigest). A hash/sketch kernel is the fix."),
    (lambda fam, op: op.startswith("min_max"),
     "`min_max` is two separate `am_reduce_ex` dispatches here (min_of_min_max then max_of_min_max), "
     "so it pays the dispatch floor twice for one pass' worth of work."),
    (lambda fam, op: fam == "strings" and ("parse" in op or "to_strings" in op),
     "Variable-length output: the kernel measures the lengths, prefix-sums them and writes in a second "
     "pass, against one streaming pass on the CPU."),
    (lambda fam, op: "top_k" in op,
     "`top_k` is a full GPU radix argsort plus a slice; a CPU running top-k touches each value once "
     "and rarely writes. Needs a partial radix / threadgroup selection kernel."),
    (lambda fam, op: op.startswith("partition_nth"),
     "`partition_nth_indices` is documented as the full stable argsort; there is no partial-partition "
     "kernel yet."),
    (lambda fam, op: fam == "latency",
     "Below the ~150 us dispatch floor: encode + commit + wait dominates the kernel. Batching removes "
     "most of it, but a single small call cannot beat an in-cache CPU loop."),
    (lambda fam, op: fam == "reductions" and ("first" in op or "last" in op or "any" in op or "all" in op),
     "A scalar answer the CPU can short-circuit or read in O(1); the GPU still pays a full dispatch "
     "plus a pass over the column."),
    (lambda fam, op: op.startswith("slice ("),
     "`MetalArray.slice` (Sources/ArrowMetal/Slice.swift) is zero-copy only when the offset is a "
     "multiple of 32; any other offset falls into a **single-threaded host loop** copying element by "
     "element, plus a `recomputeNullCount()` scan. Polars, pyarrow and pandas all return a view. "
     "Carrying an Arrow `offset` on the array, as the C Data interface allows, makes this free."),
    (lambda fam, op: "dictionary_encode" in op,
     "`MetalStringArray.dictionaryEncode()` now takes the GPU hash path "
     "(`dictionaryEncodeGPU`, up to three hash rounds); what is left is the uniques buffer being "
     "rebuilt as a string array on the host."),
    (lambda fam, op: fam == "strings" and "split" in op,
     "`Regex.splitPattern` is documented **always CPU**: it builds a Swift `[String]` per row under "
     "`concurrentPerform`, then rebuilds a MetalStringArray. Nothing runs on the GPU."),
    (lambda fam, op: fam == "strings" and "regex" in op,
     "ICU host fallback: only a metacharacter-free pattern (or `^literal`) takes the GPU path, "
     "everything else is NSRegularExpression row by row across 4096-row chunks."),
    (lambda fam, op: fam == "strings" and "replace" in op,
     "Variable-length output: the kernel measures every row's new length, prefix-sums, then writes."),
    (lambda fam, op: fam == "temporal" and ("strftime" in op or "timezone" in op),
     "Documented host path: strftime and the tz database are CPU-side (`assume_timezone` says so)."),
    (lambda fam, op: fam == "temporal",
     "Civil-calendar arithmetic: days-from-civil and its inverse are a few dozen integer operations "
     "per row on both sides, so this is compute-bound rather than bandwidth-bound and the GPU's only "
     "advantage is its lane count. Both sides land within a factor of three."),
    (lambda fam, op: op.startswith(("rank", "dense_rank")),
     "`rank` is the full GPU argsort plus a segmented scan, so it inherits the radix sort's traffic; "
     "see the sort family above."),
    (lambda fam, op: "replace_with_mask" in op,
     "Three passes: the mask's prefix sum, a gather of the replacements, then the merge. pyarrow "
     "fuses them into one streaming pass over the column."),
    (lambda fam, op: fam == "chains" and "group-by" in op,
     "The chain's group-by rebuilds the dense key mapping after the filter, which is most of the "
     "measured time; the filter and the aggregate themselves are each well inside the bar."),
    (lambda fam, op: fam == "join",
     "A GPU hash join exists in Swift (`MetalRecordBatch.join`, Sources/ArrowMetal/Kernels/Join.swift) "
     "but is **not exported through the C ABI**, so the Python row is composed from index_in + "
     "is_valid + filter + take: four dispatches and four full passes against one fused CPU hash join. "
     "Exporting the existing kernel is the fix."),
    (lambda fam, op: fam == "nested",
     "Nested kernels are one thread per row over an offsets buffer; the CPU equivalents are often "
     "metadata-only (a zero-copy child view) and so cannot be beaten by any amount of bandwidth."),
    (lambda fam, op: fam == "group-by" and any(a in op for a in ("min ", "max ", "variance",
                                                                 "count_distinct", "first ", "list ")),
     "These aggregates go through `GroupBy.segments()` (Sources/ArrowMetal/Kernels/Segmented.swift), "
     "which runs a **full stable GPU argsort of the key column** before the segmented reduction. Only "
     "sum / mean / count take the cheap atomic accumulation path. A segmented min/max and a Welford "
     "variance over that atomic path would remove the sort."),
    (lambda fam, op: fam == "group-by",
     "The dense key mapping (`am_group_by_keys`) is rebuilt on every call and, at low cardinality, "
     "costs more than the aggregation itself; the CPU libraries' hash table over a thousand keys sits "
     "in L2. Reusing one `am.group_by([...])` object across aggregates (as "
     "Benchmarks/python_gpu_bench.py does) removes that part."),
    (lambda fam, op: fam == "sort" and op.startswith(("argsort", "sort")),
     "The LSD radix sort makes one full read/write pass over the column per digit, plus a gather for "
     "`sort`; that is several times the column in traffic, so it is bandwidth-bound where Polars' "
     "multi-threaded pattern-defeating sort touches the data far fewer times. Wider digits, or an "
     "in-threadgroup first pass, is the lever."),
    (lambda fam, op: op.startswith(("sin", "cos", "tan", "divide")) and "float64" in op,
     "Metal has no `double`: float64 transcendentals and division run ArrowMetal's **software "
     "binary64** (Sources/ArrowMetal/Kernels/DoubleTranscendental.swift), tens of integer instructions "
     "per element against one vectorised hardware instruction on the CPU. Compute-bound, not "
     "bandwidth-bound."),
    (lambda fam, op: fam == "group-by" and "10000000 groups" in op,
     "At 10M groups the group table no longer fits in threadgroup memory and the kernel falls back to "
     "device atomics, which serialise on contention."),
    (lambda fam, op: fam == "decimal",
     "128-bit decimal arithmetic is emulated from 32-bit lanes on the GPU, so each element costs "
     "several instructions where the CPU has native 128-bit adds."),
    (lambda fam, op: fam == "window" and "rolling" in op,
     "Rolling sum/mean are prefix-sum differences: two full passes over the column plus a scan, "
     "against one streaming pass on the CPU."),
]


def diagnose(key, ratio, best_lib, amr, row):
    fam, op, _rows = key
    if amr is not None and amr["status"] == "error":
        return f"ArrowMetal raised: `{amr['note']}`"
    if amr is not None and amr["status"] == "no equivalent":
        return "ArrowMetal has no equivalent operation."
    for pred, cause in CAUSE_HINTS:
        try:
            if pred(fam, op):
                return cause
        except Exception:
            pass
    # memory-bound tie?
    b = row.get(best_lib) if best_lib else None
    if amr and amr.get("gbs") and b and b.get("gbs"):
        # Anything far above the machine's ~400 GB/s ceiling is not moving the data at all.
        if b["gbs"] > 800:
            return (f"The baseline is not moving the data: {best_lib} reports "
                    f"{fmt_gbs(b['gbs'])} GB/s, far above this machine's ~400 GB/s ceiling, so it "
                    "returns a view or a metadata change rather than a materialised column, while "
                    "ArrowMetal materialises the result.")
        if 50 < amr["gbs"] <= 800 and 50 < b["gbs"] <= 800:
            return (f"Memory-bound tie: ArrowMetal {fmt_gbs(amr['gbs'])} GB/s vs {best_lib} "
                    f"{fmt_gbs(b['gbs'])} GB/s, both within reach of the ~400 GB/s unified-memory "
                    "ceiling; there is no 3x available to either side on this operation.")
    if amr and amr["wall_ms"] < 0.5:
        return ("Under half a millisecond: the ~150 us dispatch floor is a large share of the "
                "measurement.")
    return ("Not yet diagnosed from the kernel; the ArrowMetal path is doing more passes over the "
            "column than the CPU library's fused one.")


def load_csv(path):
    """Reload a results CSV into ROWS / _ORDER so the report can be rebuilt without re-measuring."""
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            record(r["family"], r["op"], int(r["rows"]), r["library"],
                   float(r["wall_ms"]) if r["wall_ms"] else None,
                   float(r["cpu_ms"]) if r["cpu_ms"] else None,
                   float(r["gb_per_s"]) if r["gb_per_s"] else None,
                   int(r["iterations"]), r["status"], r["note"])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="1M-row smoke run")
    ap.add_argument("--iters", type=int, default=None, help="max repetitions per measurement")
    ap.add_argument("--budget", type=float, default=1.2,
                    help="seconds of repetitions before a measurement stops early")
    ap.add_argument("--sizes", type=str, default=None, help="comma-separated row counts")
    ap.add_argument("--families", type=str, default=None,
                    help="comma-separated subset of families to run")
    ap.add_argument("--no-report", action="store_true", help="write the CSV only")
    ap.add_argument("--report-from", type=str, default=None,
                    help="rebuild the Markdown report from an existing results CSV and exit")
    ap.add_argument("--elapsed-min", type=float, default=0.0,
                    help="with --report-from: the run time to record, in minutes")
    args = ap.parse_args()

    global BENCH
    BENCH = Bench(args.iters or (3 if args.quick else 5), args.budget)

    if args.report_from:
        load_csv(args.report_from)
        report, shortfalls = build_report(args.report_from, args.elapsed_min * 60.0)
        md_path = (os.path.join(os.path.dirname(args.report_from), "BENCHMARKS_MATRIX_quick.md")
                   if args.quick else os.path.join(ROOT, "docs", "BENCHMARKS_MATRIX.md"))
        with open(md_path, "w") as fh:
            fh.write(report)
        print(f"rebuilt {md_path} from {args.report_from}; "
              f"{len(shortfalls)} of {len(_ORDER)} operations below 3x")
        return

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
    elif args.quick:
        sizes = [1_000_000]
    else:
        sizes = [10_000_000, 50_000_000]
    str_sizes = [1_000_000] if args.quick else [1_000_000, 10_000_000]
    small_sizes = [1_000, 100_000] if args.quick else [1_000, 100_000, 1_000_000]
    wanted = set(args.families.split(",")) if args.families else None

    def want(name):
        return wanted is None or name in wanted

    print(f"ArrowMetal {am.__version__} on {am.device_name()}; polars {pl.__version__} "
          f"({pl.thread_pool_size()} threads), pyarrow {pa.__version__}, pandas {pd.__version__}, "
          f"numpy {np.__version__}")
    print(f"sizes={sizes} string sizes={str_sizes} small sizes={small_sizes} "
          f"iters<={BENCH.iters} budget={BENCH.budget}s\n")

    t_start = time.perf_counter()
    for n in sizes:
        print(f"\n===== {n:,} rows =====")
        d = Data(n)
        sd = StrData(min(n, max(str_sizes)))
        if want("reductions"):
            family_reductions(d, n)
        if want("element-wise"):
            family_elementwise(d, n)
        if want("compare+select"):
            family_select(d, n)
        if want("sort"):
            family_sort(d, n, sd if n == sizes[0] else None)
        if want("group-by"):
            family_groupby(d, n)
        if want("temporal"):
            family_temporal(d, n)
        if want("window"):
            family_window(d, n)
        if want("decimal"):
            family_decimal(d, n)
        if want("nested"):
            family_nested(d, n)
        if want("chains"):
            family_chains(d, n)
        if want("join"):
            family_join(d, n)
        del d, sd
        gc.collect()

    for n in str_sizes:
        print(f"\n===== strings, {n:,} rows =====")
        sd = StrData(n)
        if want("strings"):
            family_strings(sd)
        del sd
        gc.collect()

    if want("latency"):
        print("\n===== latency floor =====")
        family_small(small_sizes)

    elapsed = time.perf_counter() - t_start

    results_dir = os.path.join(ROOT, "Benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"full_matrix_{datetime.date.today().isoformat()}"
                                         f"{'_quick' if args.quick else ''}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "op", "rows", "library", "wall_ms", "cpu_ms",
                    "gb_per_s", "iterations", "status", "note"])
        for r in ROWS:
            w.writerow([r["family"], r["op"], r["rows"], r["library"],
                        "" if r["wall_ms"] is None else f"{r['wall_ms']:.4f}",
                        "" if r["cpu_ms"] is None else f"{r['cpu_ms']:.4f}",
                        "" if r["gbs"] is None else f"{r['gbs']:.3f}",
                        r["iters"], r["status"], r["note"]])
    print(f"\nwrote {csv_path} ({len(ROWS)} rows, {elapsed / 60:.1f} min)")

    if not args.no_report:
        report, shortfalls = build_report(csv_path, elapsed)
        md_path = (os.path.join(results_dir, "BENCHMARKS_MATRIX_quick.md") if args.quick
                   else os.path.join(ROOT, "docs", "BENCHMARKS_MATRIX.md"))
        with open(md_path, "w") as fh:
            fh.write(report)
        print(f"wrote {md_path}; {len(shortfalls)} of {len(_ORDER)} operations below 3x")


if __name__ == "__main__":
    main()
