"""Differential tests: ArrowMetal against pyarrow.compute, Arrow's reference implementation.

Every kernel reachable from `arrowmetal.MetalArray` is run on generated pyarrow arrays, and the result
is compared with the same computation done by `pyarrow.compute` on the same input. Nothing here is
compared against a hand-written expectation: the oracle is always Arrow itself, so a kernel that is
wrong in a way the author never considered still fails.

    PYTHONPATH=python python -m pytest python/tests/test_differential.py -q
    DIFF_QUICK=1 ...    # drop the 100k row datasets
    DIFF_LARGE=1 ...    # add 5,000,000 row datasets

`python/tests/differential_report.py` runs the same matrix outside pytest and prints an
operation x type table of pass/fail/skip counts. `docs/EVALUATION.md` describes the method and lists
the divergences the harness found.

Needs a real Metal device and libArrowMetalC.dylib (see python/README.md).
"""
import math
import os
import re
import struct

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am


# ------------------------------------------------------------------ types

SIGNED = ["int8", "int16", "int32", "int64"]
UNSIGNED = ["uint8", "uint16", "uint32", "uint64"]
INTEGER = SIGNED + UNSIGNED
FLOATING = ["float32", "float64"]
NUMERIC = INTEGER + FLOATING
ALL_TYPES = NUMERIC + ["bool", "utf8"]

ARROW_TYPE = {n: pa.type_for_alias("string" if n == "utf8" else n) for n in ALL_TYPES}
NUMPY_TYPE = {n: np.dtype(n) for n in NUMERIC}
_LIMITS = {n: (int(np.iinfo(NUMPY_TYPE[n]).min), int(np.iinfo(NUMPY_TYPE[n]).max)) for n in INTEGER}

#: Relative tolerance where the two engines legitimately accumulate in a different order.
FLOAT_TOL = {"float32": 1e-6, "float64": 1e-12}


#: pyarrow spells float32 "float" and float64 "double"; map back to the names used here.
_NAME_OF_TYPE = {ARROW_TYPE[n]: n for n in ALL_TYPES}


def type_name_of(arr):
    return _NAME_OF_TYPE[arr.type]


# ------------------------------------------------------------------ dataset shapes

class Shape:
    """One generated dataset: row count, null ratio, and value flavor."""

    __slots__ = ("size", "null_ratio", "flavor", "offset")

    def __init__(self, size, null_ratio, flavor="random", offset=0):
        self.size, self.null_ratio, self.flavor, self.offset = size, null_ratio, flavor, offset

    @property
    def id(self):
        base = f"{self.size}rows-nulls{self.null_ratio:g}-{self.flavor}"
        return f"{base}@{self.offset}" if self.offset else base

    def __repr__(self):
        return f"Shape({self.id})"


def sizes():
    if os.environ.get("DIFF_QUICK") == "1":
        return [0, 1, 33, 1000]
    base = [0, 1, 33, 1000, 100_003]
    if os.environ.get("DIFF_LARGE") == "1":
        base.append(5_000_000)
    return base


NULL_RATIOS = [0.0, 0.3, 1.0]


def build_shapes():
    """Sizes x null ratios, plus a sliced flavor (unaligned Arrow offsets) and a special-value flavor.

    The 100k and 5M datasets get a trimmed set of null ratios: at those sizes the extra ratio buys
    no new code path, only runtime.
    """
    out = []
    for size in sizes():
        big = size > 10_000
        for ratio in ([0.0, 0.3] if big else NULL_RATIOS):
            out.append(Shape(size, ratio, "random"))
        # Slices at offsets that are neither byte- nor 8-bit aligned, so the import path has to honour
        # ArrowArray.offset for the values buffer, the validity bitmap and the utf8 offsets alike.
        out.append(Shape(size, 0.3, "sliced", offset=5 if size else 0))
        if not big:
            out.append(Shape(size, 0.0, "sliced", offset=3 if size else 0))
    for size in (33, 1000):
        out.append(Shape(size, 0.0, "special"))
        out.append(Shape(size, 0.3, "special"))
    return out


SHAPES = build_shapes()


# ------------------------------------------------------------------ value generators

_UNICODE = ["héllo", "日本語", "naïve", "Ωμέγα", "🙂🙃", "é", "ÅNGSTRÖM", "á"]
_SHORT_STRINGS = ["", "a", "apple", "banana", "app", "Apple", "  padded  ", "tab\tsep",
                  "line\nbreak", "0", "-1", "null", "aa"] + _UNICODE
_STRING_POOL = _SHORT_STRINGS + ["x" * 300, "x" * 4096]

#: Above this many rows the long strings drop out of the pool: at 5M rows a pool averaging 200 bytes
#: would make a gigabyte-scale array per case, and every long-string path is already covered below it.
_LONG_STRING_LIMIT = 200_000


def string_pool(n):
    return _STRING_POOL if n <= _LONG_STRING_LIMIT else _SHORT_STRINGS


def special_values(name):
    """The values a kernel is most likely to get wrong, for one type."""
    if name in INTEGER:
        lo, hi = _LIMITS[name]
        vals = [0, 1, lo, hi, lo + 1, hi - 1, hi // 2, hi // 2 + 1]
        return vals + ([-1, lo // 2] if name in SIGNED else [hi - 2])
    if name in FLOATING:
        info = np.finfo(NUMPY_TYPE[name])
        tiny = float(info.tiny)
        subnormal = tiny * (2.0 ** -52 if name == "float64" else 2.0 ** -23)
        big = float(info.max)
        return [0.0, -0.0, 1.0, -1.0, 0.5, math.nan, math.inf, -math.inf, tiny, -tiny,
                subnormal, -subnormal, big, -big, 1e-8, 123456.75, float(info.eps)]
    if name == "bool":
        return [True, False]
    return _STRING_POOL          # the special flavor is only generated at 33 and 1000 rows


def _null_mask(rng, n, ratio):
    if ratio <= 0.0 or n == 0:
        return None
    if ratio >= 1.0:
        return np.ones(n, dtype=bool)
    return rng.random(n) < ratio


def _values(rng, name, n, flavor):
    """The n values that sit under the validity bitmap, null positions included."""
    if flavor == "special":
        pool = special_values(name)
        idx = rng.integers(0, len(pool), n)
        picked = [pool[i] for i in idx]
        if name in ("utf8", "bool"):
            return picked
        return np.array(picked, dtype=NUMPY_TYPE[name])
    if name == "utf8":
        pool = string_pool(n)
        return [pool[i] for i in rng.integers(0, len(pool), n)]
    if name == "bool":
        return rng.random(n) < 0.5
    if name in FLOATING:
        scale = rng.choice(np.array([1.0, 1e6, 1e-6]), n)
        return (rng.standard_normal(n) * scale).astype(NUMPY_TYPE[name])
    lo, hi = _LIMITS[name]
    # Magnitudes stay small enough that a 64-bit sum accumulator cannot overflow, so the reduction
    # oracle keeps its meaning; the extremes are covered by the "special" flavor.
    span = min(hi - lo, 2_000_000)
    base = rng.integers(0, span + 1, n, dtype=np.int64) + max(lo, -(span // 2))
    return np.clip(base, lo, hi).astype(NUMPY_TYPE[name])


class _ByteBudgetCache:
    """Generated arrays are reused across operations; the budget keeps the 5M datasets from piling up."""

    def __init__(self, budget=512 * 1024 * 1024):
        self._budget, self._used, self._items = budget, 0, {}

    def get(self, key):
        return self._items.get(key, (None, 0))[0]

    def put(self, key, arr):
        size = arr.nbytes or 64
        if size > self._budget:
            return
        while self._used + size > self._budget and self._items:
            oldest = next(iter(self._items))         # dicts keep insertion order: evict FIFO
            self._used -= self._items.pop(oldest)[1]
        self._items[key] = (arr, size)
        self._used += size


_CACHE = _ByteBudgetCache()


def make_array(name, shape, seed=0):
    """A pyarrow array of type `name` matching `shape`."""
    key = (name, shape.id, seed)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    rng = np.random.default_rng([seed, shape.size, int(shape.null_ratio * 1000), shape.offset,
                                 len(shape.flavor), ALL_TYPES.index(name)])
    n = shape.size + (2 * shape.offset if shape.flavor == "sliced" else 0)
    values = _values(rng, name, n, shape.flavor)
    mask = _null_mask(rng, n, shape.null_ratio)

    ty = ARROW_TYPE[name]
    if name in ("utf8", "bool") or shape.flavor == "special":
        rows = list(values)
        if mask is not None:
            rows = [None if m else v for v, m in zip(rows, mask)]
        arr = pa.array(rows, type=ty)
    else:
        # pa.array(ndarray, mask=...) leaves the original numbers under the null bits, which is what a
        # real column looks like and what a kernel is required to ignore.
        arr = pa.array(values, mask=mask, type=ty)

    if shape.flavor == "sliced" and shape.offset:
        arr = arr.slice(shape.offset, shape.size)
    assert len(arr) == shape.size, f"generator produced {len(arr)} rows for {shape.id}"
    _CACHE.put(key, arr)
    return arr


# ------------------------------------------------------------------ comparison

def _floats_equal(a, b, tol):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)      # NaN payloads are outside the Arrow contract
    if tol is None:
        return struct.pack("<d", a) == struct.pack("<d", b)   # bit exact, so -0.0 differs from 0.0
    if math.isinf(a) or math.isinf(b):
        return a == b
    rel, absolute = tol if isinstance(tol, tuple) else (tol, 0.0)
    return a == b or abs(a - b) <= absolute + rel * max(abs(a), abs(b), 1e-300)


_FLOAT_ARROW = (pa.float32(), pa.float64())
_BITS_OF = {pa.float32(): np.int32, pa.float64(): np.int64}


def _float_arrays_bit_equal(got, expected):
    """Bit-exact float comparison, vectorised: -0.0 does not match 0.0, any NaN matches any NaN, and
    the null positions have to line up. Arrow's own equals() gets both of the first two backwards."""
    if not pc.is_null(got).equals(pc.is_null(expected)):
        return False
    a = got.to_numpy(zero_copy_only=False)          # nulls arrive as NaN, and are masked out below
    b = expected.to_numpy(zero_copy_only=False)
    bits = _BITS_OF[got.type]
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (a.view(bits) == b.view(bits))))


def _diff_arrays(got, expected, tol):
    """Compare two pyarrow arrays. The type is part of the comparison: a kernel that returns the
    right numbers in the wrong type is wrong."""
    if got.type != expected.type:
        return f"type: got {got.type}, expected {expected.type}"
    if len(got) != len(expected):
        return f"length: got {len(got)}, expected {len(expected)}"
    if tol is None:
        # Buffer- and vector-level fast paths. Arrow's equals() is only usable for non-float types:
        # for floats it calls -0.0 and 0.0 equal and NaN and NaN unequal, the opposite of what a
        # differential test of IEEE-754 kernels needs.
        if got.type in _FLOAT_ARROW:
            if _float_arrays_bit_equal(got, expected):
                return None
        elif got.equals(expected):
            return None
    return diff(got.to_pylist(), expected.to_pylist(), tol)


def diff(got, expected, tol=None):
    """None when the two agree; otherwise a short description of the first difference."""
    if isinstance(got, pa.Array) or isinstance(expected, pa.Array):
        if not (isinstance(got, pa.Array) and isinstance(expected, pa.Array)):
            return f"shape: got {type(got).__name__}, expected {type(expected).__name__}"
        return _diff_arrays(got, expected, tol)
    if isinstance(got, (list, tuple)) or isinstance(expected, (list, tuple)):
        if not isinstance(got, (list, tuple)) or not isinstance(expected, (list, tuple)):
            return f"shape: got {type(got).__name__}, expected {type(expected).__name__}"
        if len(got) != len(expected):
            return f"length: got {len(got)}, expected {len(expected)}"
        for i, (g, e) in enumerate(zip(got, expected)):
            d = diff(g, e, tol)
            if d is not None:
                return f"[{i}] {d}"
        return None
    if got is None or expected is None:
        return None if (got is None and expected is None) else f"got {got!r}, expected {expected!r}"
    if isinstance(got, bool) != isinstance(expected, bool):
        return f"got {got!r}, expected {expected!r}"
    if isinstance(got, float) or isinstance(expected, float):
        return None if _floats_equal(float(got), float(expected), tol) else \
            f"got {got!r}, expected {expected!r}"
    return None if got == expected else f"got {got!r}, expected {expected!r}"


# ------------------------------------------------------------------ supported vs. broken

class Unsupported(Exception):
    """The operation is not implemented for this input, and said so clearly."""


#: Errors ArrowMetal raises for combinations it deliberately does not implement. Anything else out of
#: a kernel is a failure, not a gap.
_GAP = re.compile(
    r"Unsupported Arrow type"
    r"|scalar operations need a primitive array"
    r"|group-by (min/max|agg|keys|over)"
    r"|needs a 32-bit or narrower"
    r"|cast to Float32 first"
    r"|count over non-numeric"
    r"|take indices must be"
    r"|are not supported"
)


def arrow(x):
    """MetalArray -> pyarrow.Array, through the Arrow C Data Interface. Comparisons work on these
    rather than on Python lists: the type is checked too, and a 5M-row cell stays affordable."""
    return x.to_arrow()


def pylist(x):
    """MetalArray -> plain Python list, through the Arrow C Data Interface."""
    return x.to_arrow().to_pylist()


# ------------------------------------------------------------------ operation registry

class Op:
    """One differentially tested operation. `fn(src, shape)` returns (got, expected)."""

    def __init__(self, name, types, fn, tol=None, note=""):
        self.name, self.types, self.fn, self.tol, self.note = name, list(types), fn, tol, note

    def __repr__(self):
        return f"Op({self.name})"


OPS = []
ABSENT = []          # (name, method) for optional operations the library does not expose yet


def op(name, types, tol=None, note=""):
    def wrap(fn):
        OPS.append(Op(name, types, fn, tol, note))
        return fn
    return wrap


def scalar_for(name):
    """A small constant representable in every supported type."""
    return 0.5 if name in FLOATING else 3


# ---- interop -------------------------------------------------------

@op("roundtrip", ALL_TYPES)
def _roundtrip(src, shape):
    x = am.array(src)
    assert x.type == src.type, f"type changed on import: {x.type} != {src.type}"
    assert len(x) == len(src), f"length changed on import: {len(x)} != {len(src)}"
    assert x.null_count == src.null_count, f"null_count {x.null_count} != {src.null_count}"
    return x.to_arrow(), src


# ---- reductions ----------------------------------------------------

def _drop_nan(src):
    if type_name_of(src) not in FLOATING:
        return src
    return src.filter(pc.invert(pc.is_nan(pc.fill_null(src, pa.scalar(0.0, src.type)))))


def wrap64(total, name):
    """A running total as ArrowMetal's 64-bit accumulator holds it."""
    if name in UNSIGNED:
        return total % 2 ** 64
    return (total + 2 ** 63) % 2 ** 64 - 2 ** 63


def _int_sum_oracle(src, name):
    """pc.sum, or the same total wrapped to 64 bits when Arrow refuses to produce an overflowed one."""
    try:
        return pc.sum(src).as_py()
    except (pa.ArrowInvalid, OverflowError):
        valid = [v for v in src.to_pylist() if v is not None]
        return wrap64(sum(valid), name) if valid else None


def _int_mean_oracle(src, name):
    """ArrowMetal divides the wrapped 64-bit sum by the count. pyarrow's mean accumulates in double,
    so the two only agree while the exact total fits; test_integer_mean_wraps_where_pyarrow_widens
    pins the difference."""
    valid = src.drop_null()
    if len(valid) == 0:
        return None
    wide = np.uint64 if name in UNSIGNED else np.int64
    with np.errstate(over="ignore"):
        # NumPy wraps in the accumulator's width; int() then makes the division exact.
        total = int(valid.to_numpy(zero_copy_only=False).astype(wide).sum())
    return wrap64(total, name) / len(valid)


#: Unit roundoff of each float type: half an ulp at 1.0.
_EPS = {"float32": 2.0 ** -24, "float64": 2.0 ** -53}


def float_sum_bound(src, name, count=None, scale=None):
    """Worst-case absolute error of naive summation, n * eps * sum|x|, relaxed to sqrt(n) * eps * sum|x|
    -- Arrow accumulates float32 in double and blocks its float64 sums, so some drift is expected;
    a bound that scales with the input is the only comparison that stays meaningful at 100k rows."""
    if name not in _EPS:
        return 0.0
    if scale is None:
        scale = pc.sum(pc.abs(src)).as_py()
    if count is None:
        count = len(src) - src.null_count
    if scale is None or not math.isfinite(scale):
        return 0.0                                   # inf/NaN inputs: only exact agreement counts
    return 8.0 * _EPS[name] * math.sqrt(max(count, 1)) * abs(scale)


@op("sum", NUMERIC, tol="float", note="float sums compared to an n-scaled roundoff bound")
def _sum(src, shape):
    name = type_name_of(src)
    if name not in FLOATING:
        return am.array(src).sum(), _int_sum_oracle(src, name)
    return (am.array(src).sum(), pc.sum(src).as_py(),
            (FLOAT_TOL[name], float_sum_bound(src, name)))


@op("min_max", NUMERIC, note="NaN dropped from the input; NaN semantics pinned separately")
def _min_max(src, shape):
    clean = _drop_nan(src)
    x = am.array(clean)
    return [x.min(), x.max()], [pc.min(clean).as_py(), pc.max(clean).as_py()]


@op("mean", NUMERIC, tol="result_float", note="NaN dropped from the input")
def _mean(src, shape):
    name = type_name_of(src)
    clean = _drop_nan(src)
    if name not in FLOATING:
        return am.array(clean).mean(), _int_mean_oracle(clean, name)
    count = max(len(clean) - clean.null_count, 1)
    return (am.array(clean).mean(), pc.mean(clean).as_py(),
            (FLOAT_TOL[name], float_sum_bound(clean, name) / count))


# ---- element-wise --------------------------------------------------

_CMP = [("==", pc.equal), ("!=", pc.not_equal), ("<", pc.less), ("<=", pc.less_equal),
        (">", pc.greater), (">=", pc.greater_equal)]


@op("compare_scalar", NUMERIC)
def _compare_scalar(src, shape):
    s = scalar_for(type_name_of(src))
    x = am.array(src)
    got, expected = [], []
    for symbol, fn in _CMP:
        got.append(arrow(x.compare(symbol, s)))
        expected.append(fn(src, pa.scalar(s, src.type)))
    return got, expected


@op("compare_array", NUMERIC)
def _compare_array(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    x, y = am.array(src), am.array(other)
    got, expected = [], []
    for symbol, fn in _CMP:
        got.append(arrow(x.compare(symbol, y)))
        expected.append(fn(src, other))
    return got, expected


_ARITH = [("+", pc.add), ("-", pc.subtract), ("*", pc.multiply), ("/", pc.divide)]


@op("arith_scalar", NUMERIC, note="pyarrow's unchecked add/subtract/multiply; both wrap on overflow")
def _arith_scalar(src, shape):
    name = type_name_of(src)
    x = am.array(src)
    got, expected = [], []
    for symbol, fn in _ARITH:
        s = scalar_for(name)
        if symbol == "/" and name in INTEGER:
            s = 3                     # never 0: the divergence is pinned in its own test below
        got.append(arrow(x.arith(symbol, s)))
        expected.append(fn(src, pa.scalar(s, src.type)))
    return got, expected


@op("arith_array", NUMERIC, note="pyarrow's unchecked add/subtract/multiply; both wrap on overflow")
def _arith_array(src, shape):
    name = type_name_of(src)
    x = am.array(src)
    other = make_array(name, shape, seed=1)
    # Two integer divisors are excluded and pinned in tests of their own: 0, where ArrowMetal returns
    # 0 and pyarrow raises, and -1, where INT_MIN / -1 is undefined in C and the two engines differ.
    divisor = other
    if name in INTEGER:
        bad = pc.is_in(other, value_set=pa.array([0, -1] if name in SIGNED else [0], other.type))
        divisor = pc.if_else(bad, pa.scalar(1, other.type), other)
    got, expected = [], []
    for symbol, fn in _ARITH:
        rhs = divisor if symbol == "/" else other
        got.append(arrow(x.arith(symbol, am.array(rhs))))
        expected.append(fn(src, rhs))
    return got, expected


@op("cast", NUMERIC, note="pyarrow safe=False; out-of-range float->int excluded and pinned separately")
def _cast(src, shape):
    name = type_name_of(src)
    x = am.array(src)
    got, expected = [], []
    for target in NUMERIC:
        source, xs = src, x
        if name in FLOATING and target in INTEGER:
            # Out-of-range float -> int saturates in Arrow and wraps on the GPU; both are undefined
            # in C. Keep those values out of the matrix (test_out_of_range_float_to_int_cast_diverges
            # pins them) with a strict upper bound, since float(INT64_MAX) rounds up to 2**63.
            lo, hi = _LIMITS[target]
            filled = pc.fill_null(src, pa.scalar(0.0, src.type))
            in_range = pc.and_(pc.invert(pc.is_nan(filled)),
                               pc.and_(pc.greater_equal(filled, pa.scalar(float(lo), src.type)),
                                       pc.less(filled, pa.scalar(float(hi), src.type))))
            source = src.filter(in_range)
            xs = am.array(source)
        got.append(arrow(xs.cast(target)))
        expected.append(source.cast(ARROW_TYPE[target], safe=False))
    return got, expected


@op("bool_logic", ["bool"], note="pyarrow and_/or_/invert, not the Kleene variants")
def _bool_logic(src, shape):
    other = make_array("bool", shape, seed=1)
    x, y = am.array(src), am.array(other)
    return ([arrow(x & y), arrow(x | y), arrow(~x)],
            [pc.and_(src, other), pc.or_(src, other), pc.invert(src)])


# ---- selection -----------------------------------------------------

@op("filter", ALL_TYPES)
def _filter(src, shape):
    mask = make_array("bool", shape, seed=2)
    return arrow(am.array(src).filter(am.array(mask))), src.filter(mask)


@op("filter_where", NUMERIC)
def _filter_where(src, shape):
    s = scalar_for(type_name_of(src))
    x = am.array(src)
    got, expected = [], []
    for symbol, fn in _CMP:
        got.append(arrow(x.filter_where(symbol, s)))
        expected.append(src.filter(fn(src, pa.scalar(s, src.type))))
    return got, expected


@op("take", ALL_TYPES)
def _take(src, shape):
    n = len(src)
    if n == 0:
        idx = pa.array([], pa.int32())
    else:
        rng = np.random.default_rng(11)
        raw = rng.integers(0, n, min(n, 977)).astype(np.int32)
        idx = pa.array(raw, mask=rng.random(len(raw)) < 0.1, type=pa.int32())
    return arrow(am.array(src).take(am.array(idx))), src.take(idx)


@op("slice", ALL_TYPES)
def _slice(src, shape):
    n = len(src)
    off = n // 3
    return arrow(am.array(src).slice(off, n - off)), src.slice(off, n - off)


# ---- sorting -------------------------------------------------------

@op("sort", NUMERIC)
def _sort(src, shape):
    x = am.array(src)
    return ([arrow(x.sort()), arrow(x.sort(True))],
            [src.take(pc.array_sort_indices(src)),
             src.take(pc.array_sort_indices(src, order="descending"))])


@op("argsort", NUMERIC, note="both engines document a stable sort with nulls last")
def _argsort(src, shape):
    x = am.array(src)
    return ([arrow(x.argsort()), arrow(x.argsort(True))],
            [pc.array_sort_indices(src).cast(pa.int32()),
             pc.array_sort_indices(src, order="descending").cast(pa.int32())])


@op("top_k", NUMERIC)
def _top_k(src, shape):
    x = am.array(src)
    k = min(len(src), 17)
    return ([arrow(x.top_k(k)), arrow(x.top_k(k, False))],
            [pc.array_sort_indices(src, order="descending").cast(pa.int32()).slice(0, k),
             pc.array_sort_indices(src).cast(pa.int32()).slice(0, k)])


# ---- strings -------------------------------------------------------

@op("str_length", ["utf8"])
def _str_length(src, shape):
    x = am.array(src)
    return ([arrow(x.byte_length()), arrow(x.char_length())],
            [pc.binary_length(src).cast(pa.int32()), pc.utf8_length(src).cast(pa.int32())])


_PATTERNS = ["a", "", "app", "Apple", "x", "é", "日", "\t", "zzz", "banana", "x" * 300]


@op("str_match", ["utf8"])
def _str_match(src, shape):
    x = am.array(src)
    got, expected = [], []
    for p in _PATTERNS:
        got += [arrow(x.starts_with(p)), arrow(x.ends_with(p)),
                arrow(x.str_contains(p)), arrow(x.str_equals(p))]
        expected += [pc.starts_with(src, p), pc.ends_with(src, p), pc.match_substring(src, p),
                     pc.equal(src, pa.scalar(p, pa.string()))]
    return got, expected


@op("str_equals_array", ["utf8"])
def _str_equals_array(src, shape):
    other = make_array("utf8", shape, seed=1)
    return arrow(am.array(src).str_equals(am.array(other))), pc.equal(src, other)


@op("str_hash32", ["utf8"], note="no pyarrow oracle: checked for null propagation and injectivity")
def _str_hash32(src, shape):
    """Arrow has no murmur3 to compare against, so check the two properties a hash owes its caller:
    nulls stay null, and the map from string to hash is a bijection over the distinct values."""
    hashes = arrow(am.array(src).hash32())
    if hashes.type != pa.uint32():
        return f"hash type {hashes.type}", "uint32"
    if not pc.is_null(hashes).equals(pc.is_null(src)):
        return "null positions differ between input and hashes", "identical null positions"
    distinct_values = pc.count_distinct(src).as_py()
    distinct_hashes = pc.count_distinct(hashes).as_py()
    pairs = pa.table({"v": src, "h": hashes}).filter(pc.is_valid(src))
    distinct_pairs = pairs.group_by(["v", "h"]).aggregate([([], "count_all")]).num_rows
    if distinct_pairs != distinct_values:
        return f"{distinct_values} strings produced {distinct_pairs} (string, hash) pairs", \
               "one hash per string"
    if distinct_hashes != distinct_values:
        return f"{distinct_values} strings collided onto {distinct_hashes} hashes", "no collisions"
    return "consistent", "consistent"


@op("dictionary_encode", ["utf8"])
def _dictionary_encode(src, shape):
    codes, unique = am.array(src).dictionary_encode()
    assert codes.type == pa.int32(), f"codes are {codes.type}, expected int32"
    assert unique.type == pa.string(), f"dictionary is {unique.type}, expected string"
    dictionary = arrow(unique)
    decoded = dictionary.take(arrow(codes))
    first_seen = pa.array(dict.fromkeys(v for v in src.to_pylist() if v is not None), pa.string())
    return [decoded, dictionary], [src, first_seen]


# ---- group-by ------------------------------------------------------

GROUP_KEY_COUNT = 8


def group_keys(shape):
    n = shape.size
    if n == 0:
        return pa.array([], pa.int32()), 1
    rng = np.random.default_rng(23)
    key_count = min(GROUP_KEY_COUNT, max(1, n))
    raw = rng.integers(0, key_count, n).astype(np.int32)
    return pa.array(raw, mask=rng.random(n) < 0.05, type=pa.int32()), key_count


def group_oracle(keys, values, key_count, agg):
    """pa.Table.group_by, projected onto the dense key range ArrowMetal reports."""
    table = pa.table({"k": keys, "v": values})
    spec = [([], "count_all")] if agg == "count_all" else [("v", agg)]
    grouped = table.group_by("k").aggregate(spec)
    column = "count_all" if agg == "count_all" else "v_" + agg
    by_key = dict(zip(grouped["k"].to_pylist(), grouped[column].to_pylist()))
    empty = 0 if agg in ("count", "count_all") else None
    return [by_key.get(k, empty) for k in range(key_count)]


def _paired(keys, values, key_count, finite_only=False):
    """The (key, value) rows both engines see, as numpy: neither side null, key inside the range."""
    keep = pc.and_(pc.is_valid(keys), pc.is_valid(values))
    k = keys.filter(keep).to_numpy(zero_copy_only=False).astype(np.int64)
    v = values.filter(keep).to_numpy(zero_copy_only=False)
    inside = (k >= 0) & (k < key_count)
    if finite_only:
        inside &= np.isfinite(v)
    return k[inside], v[inside]


def group_mean_oracle(keys, values, key_count, name):
    """Per-group wrapped-64-bit sum over count, matching the scalar mean accumulator."""
    k, v = _paired(keys, values, key_count)
    counts = np.bincount(k, minlength=key_count)
    # NumPy wraps in the accumulator's own width, exactly as the 64-bit kernel does.
    sums = np.zeros(key_count, dtype=np.uint64 if name in UNSIGNED else np.int64)
    with np.errstate(over="ignore"):
        np.add.at(sums, k, v.astype(sums.dtype))
    return [int(sums[i]) / int(counts[i]) if counts[i] else None for i in range(key_count)]


def group_float_bound(keys, values, key_count, name, per_element):
    """A roundoff bound valid for every group: the worst any single group can suffer."""
    k, v = _paired(keys, values, key_count, finite_only=True)
    if not len(k):
        return 0.0
    counts = np.bincount(k, minlength=key_count)
    scales = np.bincount(k, weights=np.abs(v.astype(np.float64)), minlength=key_count)
    worst = 0.0
    for i in range(key_count):
        if not counts[i]:
            continue
        bound = float_sum_bound(None, name, count=int(counts[i]), scale=float(scales[i]))
        worst = max(worst, bound / int(counts[i]) if per_element else bound)
    return worst


def _register_group_op(agg, arrow_agg, tol=None):
    @op("group_by_" + agg, NUMERIC, tol=tol)
    def run(src, shape, agg=agg, arrow_agg=arrow_agg):
        keys, key_count = group_keys(shape)
        g = am.array(keys).group_by(key_count)
        result = g.count() if agg == "count" else getattr(g, agg)(am.array(src))
        name = type_name_of(src)
        if agg == "mean" and name in INTEGER:
            return pylist(result), group_mean_oracle(keys, src, key_count, name)
        expected = group_oracle(keys, src, key_count, arrow_agg)
        if agg in ("sum", "mean") and name in FLOATING:
            bound = group_float_bound(keys, src, key_count, name, per_element=(agg == "mean"))
            return pylist(result), expected, (FLOAT_TOL[name], bound)
        return pylist(result), expected
    return run


_register_group_op("count", "count_all")
_register_group_op("count_values", "count")
_register_group_op("sum", "sum", tol="float")
_register_group_op("min", "min")
_register_group_op("max", "max")
_register_group_op("mean", "mean", tol="result_float")


# ---- operations other agents may append ----------------------------
#
# None of these exist in 0.1.0. Each is registered only if the method is actually on MetalArray by the
# time this module is imported, so the matrix picks up new kernels without any edit here; the ones
# still missing are listed by test_absent_optional_operations_are_reported.

def _register_optional(name, types, methods, fn, tol=None):
    missing = [m for m in methods if not hasattr(am.MetalArray, m)]
    if missing:
        ABSENT.append((name, ", ".join(missing)))
        return
    OPS.append(Op(name, types, fn, tol, note="optional kernel, discovered at import"))


def _unary_optional(name, method, oracle, types, tol=None):
    def fn(src, shape, method=method, oracle=oracle):
        return arrow(getattr(am.array(src), method)()), oracle(src)
    _register_optional(name, types, [method], fn, tol)


_unary_optional("is_null", "is_null", pc.is_null, ALL_TYPES)
_unary_optional("is_valid", "is_valid", pc.is_valid, ALL_TYPES)
_unary_optional("is_nan", "is_nan", pc.is_nan, FLOATING)
_unary_optional("abs", "abs", pc.abs, NUMERIC)
_unary_optional("negate", "negate", pc.negate, SIGNED + FLOATING)
_unary_optional("sign", "sign", pc.sign, NUMERIC)
_unary_optional("upper", "upper", pc.utf8_upper, ["utf8"])
_unary_optional("lower", "lower", pc.utf8_lower, ["utf8"])
_unary_optional("trim", "trim", pc.utf8_trim_whitespace, ["utf8"])
_unary_optional("reverse", "reverse", pc.utf8_reverse, ["utf8"])
_unary_optional("bitwise_not", "bitwise_not", pc.bit_wise_not, INTEGER)
_unary_optional("cumulative_sum", "cumulative_sum", pc.cumulative_sum, NUMERIC, tol="float")
_unary_optional("cumulative_prod", "cumulative_prod", pc.cumulative_prod, NUMERIC, tol="float")
_unary_optional("cumulative_max", "cumulative_max", pc.cumulative_max, NUMERIC)
_unary_optional("cumulative_min", "cumulative_min", pc.cumulative_min, NUMERIC)


def _binary_optional(name, method, oracle, types):
    def fn(src, shape, method=method, oracle=oracle):
        other = make_array(type_name_of(src), shape, seed=1)
        return arrow(getattr(am.array(src), method)(am.array(other))), oracle(src, other)
    _register_optional(name, types, [method], fn)


_binary_optional("bitwise_and", "bitwise_and", pc.bit_wise_and, INTEGER)
_binary_optional("bitwise_or", "bitwise_or", pc.bit_wise_or, INTEGER)
_binary_optional("bitwise_xor", "bitwise_xor", pc.bit_wise_xor, INTEGER)
_binary_optional("shift_left", "shift_left", pc.shift_left, INTEGER)
_binary_optional("shift_right", "shift_right", pc.shift_right, INTEGER)


def _fill_null(src, shape):
    name = type_name_of(src)
    v = "" if name == "utf8" else (False if name == "bool" else scalar_for(name))
    return arrow(am.array(src).fill_null(v)), pc.fill_null(src, pa.scalar(v, src.type))


_register_optional("fill_null", ALL_TYPES, ["fill_null"], _fill_null)


def _if_else(src, shape):
    cond = make_array("bool", shape, seed=2)
    other = make_array(type_name_of(src), shape, seed=1)
    return arrow(am.array(src).if_else(am.array(cond), am.array(other))), pc.if_else(cond, src, other)


_register_optional("if_else", ALL_TYPES, ["if_else"], _if_else)


def _is_in(src, shape):
    values = src.slice(0, min(len(src), 5))
    return arrow(am.array(src).is_in(am.array(values))), pc.is_in(src, value_set=values)


_register_optional("is_in", ALL_TYPES, ["is_in"], _is_in)


# ------------------------------------------------------------------ the runner

PASS, FAIL, SKIP = "pass", "fail", "skip"


def resolve_tol(mode, type_name):
    """None means exact (bit-exact for floats)."""
    if mode is None:
        return None
    if mode == "float":                    # only the float types accumulate differently
        return FLOAT_TOL.get(type_name)
    if mode == "result_float":             # the result is a double whatever the input type is
        return FLOAT_TOL.get(type_name, FLOAT_TOL["float64"])
    return float(mode)


class Finding:
    """A divergence this harness has already reported. Failures that match one are still failures --
    the report still exits 1 -- but they are grouped as known so a new failure stands out."""

    def __init__(self, ident, title, ops, types, flavors=None, needs_nulls=False):
        self.id, self.title = ident, title
        self.ops, self.types, self.flavors, self.needs_nulls = set(ops), set(types), flavors, needs_nulls

    def matches(self, op_name, type_name, shape):
        if op_name not in self.ops or type_name not in self.types:
            return False
        if self.flavors is not None and shape.flavor not in self.flavors:
            return False
        return not self.needs_nulls or shape.null_ratio > 0


FINDINGS = [
    Finding("argsort-null-order",
            "argsort/top_k order the null indices by the bytes under the validity bitmap",
            ["argsort", "top_k"], NUMERIC, needs_nulls=True),
    Finding("float32-subnormal-ftz",
            "Float32 kernels flush subnormals to zero (arithmetic and comparison)",
            ["arith_scalar", "arith_array", "compare_array", "compare_scalar"],
            ["float32"], flavors={"special"}),
    Finding("negative-zero-sort-order",
            "sort/argsort order -0.0 before 0.0 instead of leaving the tie in input order",
            ["sort", "argsort", "top_k"], FLOATING, flavors={"special"}),
    Finding("float32-reduction-accumulator",
            "sum/mean over Float32 accumulate in Float32 and overflow where Arrow uses a double",
            ["sum", "mean", "group_by_sum", "group_by_mean"], ["float32"], flavors={"special"}),
    Finding("uint64-group-sum-signed",
            "group-by sum over UInt64 values comes back as Int64, so totals above 2^63 go negative",
            ["group_by_sum", "group_by_mean"], ["uint64"], flavors={"special"}),
]


def classify(op_name, type_name, shape):
    """The known finding this failure belongs to, or None if it is new."""
    for finding in FINDINGS:
        if finding.matches(op_name, type_name, shape):
            return finding
    return None


def run_case(operation, type_name, shape):
    """Run one (operation, type, shape) case. Returns (status, detail)."""
    if type_name not in operation.types:
        return SKIP, "type out of scope for this operation"
    try:
        src = make_array(type_name, shape)
    except Exception as exc:                              # pragma: no cover - generator bug
        return FAIL, f"generator raised {type(exc).__name__}: {exc}"
    override = None
    try:
        result = operation.fn(src, shape)
        got, expected = result[0], result[1]
        if len(result) == 3:                          # the op supplied its own tolerance
            override = result[2]
    except Unsupported as exc:
        return SKIP, str(exc)
    except am.ArrowMetalError as exc:
        if _GAP.search(str(exc)):
            return SKIP, f"not implemented: {exc}"
        return FAIL, f"ArrowMetalError: {exc}"
    except AssertionError as exc:
        return FAIL, str(exc) or "assertion failed"
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, OverflowError) as exc:
        return FAIL, f"pyarrow oracle raised {type(exc).__name__}: {exc}"

    d = diff(got, expected, override if override is not None else resolve_tol(operation.tol, type_name))
    return (PASS, "") if d is None else (FAIL, d)


def all_cases(shapes=None):
    """Every (operation, type, shape) triple, shape-major so the array cache stays warm."""
    for shape in (shapes or SHAPES):
        for operation in OPS:
            for type_name in operation.types:
                yield operation, type_name, shape


# ------------------------------------------------------------------ pytest

_PARAMS = [(o, t, s) for o in OPS for t in o.types for s in SHAPES]
_IDS = [f"{o.name}-{t}-{s.id}" for o, t, s in _PARAMS]


@pytest.mark.parametrize("operation,type_name,shape", _PARAMS, ids=_IDS)
def test_matches_pyarrow(operation, type_name, shape):
    status, detail = run_case(operation, type_name, shape)
    if status == SKIP:
        pytest.skip(detail)
    if status == FAIL:
        finding = classify(operation.name, type_name, shape)
        if finding is not None:
            # Reported in docs/EVALUATION.md under "Open findings"; xfail so a *new* divergence in
            # this cell still turns the suite red, while the known one does not.
            pytest.xfail(f"open finding {finding.id}: {finding.title} -- {detail[:160]}")
    assert status == PASS, f"{operation.name} / {type_name} / {shape.id}: {detail}"


def test_absent_optional_operations_are_reported():
    """Not a failure: a record of the kernels this harness would test if they existed."""
    for name, methods in ABSENT:
        print(f"absent: {name} (MetalArray.{methods})")
    assert isinstance(ABSENT, list)


# ------------------------------------------------------------------ coverage of the public API

#: MetalArray/GroupBy members that no single pyarrow.compute call corresponds to.
_NOT_DIFFERENTIABLE = {"from_arrow", "to_arrow", "null_count", "format", "type", "group_by"}

#: Method -> the operation that exercises it.
_COVERED_BY = {
    "sum": "sum", "min": "min_max", "max": "min_max", "mean": "mean",
    "compare": "compare_scalar", "arith": "arith_scalar", "cast": "cast",
    "filter": "filter", "filter_where": "filter_where", "take": "take", "slice": "slice",
    "argsort": "argsort", "sort": "sort", "top_k": "top_k",
    "byte_length": "str_length", "char_length": "str_length", "hash32": "str_hash32",
    "starts_with": "str_match", "ends_with": "str_match", "str_contains": "str_match",
    "str_equals": "str_match", "dictionary_encode": "dictionary_encode",
    "count": "group_by_count", "count_values": "group_by_count_values",
}


def public_api():
    return {n for n in dir(am.MetalArray) if not n.startswith("_")} | \
           {n for n in dir(am.GroupBy) if not n.startswith("_")}


def test_every_public_operation_has_a_differential_case():
    """Fails when a method is added to MetalArray or GroupBy without a case here, so the harness
    cannot silently fall behind the library."""
    known = _NOT_DIFFERENTIABLE | set(_COVERED_BY) | {o.name for o in OPS} | {n for n, _ in ABSENT}
    missing = sorted(n for n in public_api() if n not in known)
    assert not missing, (
        "no differential case for: " + ", ".join(missing) +
        " -- add an Op, or an entry in _COVERED_BY / _NOT_DIFFERENTIABLE, "
        "in python/tests/test_differential.py")


# ------------------------------------------------------------------ pinned divergences
#
# Each test below records a place where ArrowMetal and pyarrow.compute knowingly disagree. They are
# assertions rather than skips: if either engine changes its mind, the test says so.


def test_integer_division_by_zero_returns_zero_where_pyarrow_raises():
    a = pa.array([7, -7, 0, 5], pa.int32())
    assert pylist(am.array(a).arith("/", 0)) == [0, 0, 0, 0]
    with pytest.raises(pa.ArrowInvalid):
        pc.divide(a, pa.scalar(0, pa.int32()))
    assert pylist(am.array(a).arith("/", am.array(pa.array([1, 0, 2, 0], pa.int32())))) == [7, 0, 0, 0]


def test_float_division_by_zero_is_ieee_754_in_both():
    a = pa.array([1.0, -1.0, 0.0], pa.float64())
    got = pylist(am.array(a).arith("/", 0.0))
    expected = pc.divide(a, pa.scalar(0.0, pa.float64())).to_pylist()
    assert got[:2] == expected[:2] == [math.inf, -math.inf]
    assert math.isnan(got[2]) and math.isnan(expected[2])


def test_integer_overflow_wraps_where_checked_pyarrow_raises():
    a = pa.array([2 ** 31 - 1], pa.int32())
    assert pylist(am.array(a).arith("+", 1)) == [-2 ** 31]
    assert pc.add(a, pa.scalar(1, pa.int32())).to_pylist() == [-2 ** 31]   # unchecked also wraps
    with pytest.raises(pa.ArrowInvalid):
        pc.add_checked(a, pa.scalar(1, pa.int32()))


def test_int_min_divided_by_minus_one_wraps_where_pyarrow_yields_zero():
    """C leaves INT_MIN / -1 undefined. ArrowMetal returns the hardware answer, INT_MIN;
    pyarrow's unchecked divide returns 0."""
    a = pa.array([-128, 127, 0], pa.int8())
    assert pylist(am.array(a).arith("/", -1)) == [-128, -127, 0]
    assert pc.divide(a, pa.scalar(-1, pa.int8())).to_pylist() == [0, -127, 0]


def test_out_of_range_float_to_int_cast_diverges():
    """Arrow saturates an out-of-range float -> int cast; the GPU wraps. Undefined in C either way."""
    a = pa.array([1e20, -1.7, 2.9, math.nan], pa.float64())
    got = pylist(am.array(a).cast("int32"))
    expected = a.cast(pa.int32(), safe=False).to_pylist()
    assert got[1:] == expected[1:] == [-1, 2, 0]
    assert expected[0] == 2 ** 31 - 1 and got[0] != expected[0]


@pytest.mark.parametrize("ty", [pa.float32(), pa.float64()])
def test_nan_is_skipped_by_min_and_max_in_both(ty):
    a = pa.array([1.0, math.nan, 3.0, -2.0], ty)
    x = am.array(a)
    assert x.min() == pc.min(a).as_py() == -2.0
    assert x.max() == pc.max(a).as_py() == 3.0
    b = pa.array([math.nan, math.inf, -math.inf], ty)
    assert am.array(b).min() == pc.min(b).as_py() == -math.inf
    assert am.array(b).max() == pc.max(b).as_py() == math.inf


@pytest.mark.parametrize("ty", [pa.float32(), pa.float64()])
def test_all_nan_min_max_is_null_in_arrowmetal_and_nan_in_pyarrow(ty):
    """ArrowMetal treats NaN as missing all the way: with nothing but NaN left it reports null, even
    though the array has no nulls. pyarrow returns NaN. Listed in docs/EVALUATION.md."""
    a = pa.array([math.nan, math.nan], ty)
    assert a.null_count == 0
    assert am.array(a).min() is None and am.array(a).max() is None
    assert math.isnan(pc.min(a).as_py()) and math.isnan(pc.max(a).as_py())
    # sum and mean agree: NaN propagates through both.
    assert math.isnan(am.array(a).sum()) and math.isnan(pc.sum(a).as_py())


def test_nan_propagates_through_sum_in_both():
    a = pa.array([1.0, math.nan, 3.0], pa.float64())
    assert math.isnan(am.array(a).sum()) and math.isnan(pc.sum(a).as_py())


def test_boolean_and_or_are_not_kleene():
    """ArrowMetal's & and | propagate nulls like pyarrow's and_/or_, not like and_kleene/or_kleene."""
    a = pa.array([True, False, None, None], pa.bool_())
    b = pa.array([None, None, True, False], pa.bool_())
    x, y = am.array(a), am.array(b)
    assert pylist(x & y) == pc.and_(a, b).to_pylist() == [None] * 4
    assert pylist(x | y) == pc.or_(a, b).to_pylist() == [None] * 4
    assert pc.and_kleene(a, b).to_pylist() == [None, False, None, False]


def test_ties_sort_stably_in_both():
    a = pa.array([3, 1, 3, 2, 3, 1], pa.int32())
    assert pylist(am.array(a).argsort()) == pc.array_sort_indices(a).to_pylist() == [1, 5, 3, 0, 2, 4]
    assert pylist(am.array(a).argsort(True)) == \
        pc.array_sort_indices(a, order="descending").to_pylist() == [0, 2, 4, 3, 1, 5]


def test_sort_places_nan_after_positive_infinity_in_both():
    a = pa.array([1.0, math.nan, math.inf, -math.inf], pa.float64())
    assert pylist(am.array(a).argsort()) == pc.array_sort_indices(a).to_pylist() == [3, 0, 2, 1]


def test_integer_mean_wraps_where_pyarrow_widens():
    """ArrowMetal sums integers into a 64-bit accumulator and divides that; pyarrow's mean
    accumulates in double. The two part company exactly where the 64-bit total wraps -- which is
    also where ArrowMetal's own sum() wraps, and pyarrow's sum() agrees with it."""
    a = pa.array([2 ** 63 - 1, 2 ** 63 - 1, 2], pa.int64())
    assert am.array(a).sum() == pc.sum(a).as_py() == 0          # both wrap
    assert am.array(a).mean() == 0.0
    assert pc.mean(a).as_py() == pytest.approx((2 * (2 ** 63 - 1) + 2) / 3)


def test_group_by_ignores_keys_outside_the_declared_range():
    """pyarrow makes a group for every distinct key; ArrowMetal drops rows outside [0, key_count)."""
    keys = pa.array([0, 1, 5, -1], pa.int32())
    assert pylist(am.array(keys).group_by(2).count()) == [1, 1]
    assert len(pa.table({"k": keys}).group_by("k").aggregate([([], "count_all")])) == 4


# ------------------------------------------------------------------ open findings
#
# Reproductions of bugs this harness found. They are xfail(strict=True), so they start passing --
# and the suite starts failing -- the moment a kernel is fixed. See docs/EVALUATION.md.


def _payload_pair():
    """Two arrays Arrow calls equal, differing only in the bytes under the null bits."""
    mask = np.array([True, True, True, False])
    loud = pa.array(np.array([30, 20, 10, 7], np.int32), mask=mask, type=pa.int32())
    quiet = pa.array(np.array([0, 0, 0, 7], np.int32), mask=mask, type=pa.int32())
    assert loud.equals(quiet)
    return loud, quiet


@pytest.mark.xfail(strict=True, reason="argsort-null-order: argsort orders the null indices by the "
                                       "bytes left under the validity bitmap instead of by position")
def test_argsort_null_block_is_stable():
    loud, quiet = _payload_pair()
    assert pylist(am.array(loud).argsort()) == pylist(am.array(quiet).argsort())
    assert pylist(am.array(loud).argsort()) == pc.array_sort_indices(loud).to_pylist()


@pytest.mark.xfail(strict=True, reason="argsort-null-order: top_k inherits the argsort null ordering")
def test_top_k_null_block_is_stable():
    loud, _ = _payload_pair()
    assert pylist(am.array(loud).top_k(4, False)) == pc.array_sort_indices(loud).to_pylist()


@pytest.mark.xfail(strict=True, reason="float32-subnormal-ftz: Float32 arithmetic flushes subnormal "
                                       "results to zero")
def test_float32_arithmetic_keeps_subnormal_results():
    tiny = float(np.finfo(np.float32).tiny)               # 1.1754944e-38, the smallest normal
    a = pa.array([tiny, 1.0], pa.float32())
    assert pylist(am.array(a).arith("*", 0.5)) == \
        pc.multiply(a, pa.scalar(0.5, pa.float32())).to_pylist()


@pytest.mark.xfail(strict=True, reason="float32-subnormal-ftz: Float32 comparison treats subnormal "
                                       "operands as zero")
def test_float32_comparison_distinguishes_subnormals_from_zero():
    smallest = float(np.finfo(np.float32).tiny) * 2.0 ** -23     # 1.4e-45
    a = pa.array([smallest], pa.float32())
    assert pylist(am.array(a).compare(">", 0.0)) == \
        pc.greater(a, pa.scalar(0.0, pa.float32())).to_pylist() == [True]


def test_float64_arithmetic_and_comparison_keep_subnormals():
    """The Float64 software path is not affected: only the Float32 kernels flush."""
    smallest = 5e-324
    a = pa.array([smallest, smallest * 4], pa.float64())
    assert pylist(am.array(a).arith("*", 0.5)) == \
        pc.multiply(a, pa.scalar(0.5, pa.float64())).to_pylist()
    assert pylist(am.array(a).compare(">", 0.0)) == [True, True]


@pytest.mark.xfail(strict=True, reason="negative-zero-sort-order: the radix sort separates -0.0 from "
                                       "0.0, which Arrow treats as a tie held in input order")
def test_negative_zero_is_a_sort_tie():
    a = pa.array([0.0, -0.0, 0.0, -0.0], pa.float64())
    assert pylist(am.array(a).argsort()) == pc.array_sort_indices(a).to_pylist()


@pytest.mark.xfail(strict=True, reason="float32-reduction-accumulator: sum over Float32 accumulates "
                                       "in Float32 and overflows where Arrow accumulates in double")
def test_float32_sum_does_not_overflow_before_arrow_does():
    big = float(np.finfo(np.float32).max)
    a = pa.array([big, big], pa.float32())
    assert am.array(a).sum() == pc.sum(a).as_py()          # Arrow: 6.805646932770577e+38


@pytest.mark.xfail(strict=True, reason="uint64-group-sum-signed: group-by sum over UInt64 values "
                                       "returns an Int64 array")
def test_group_by_sum_over_uint64_stays_unsigned():
    keys = pa.array([0, 0], pa.int32())
    values = pa.array([2 ** 63, 2 ** 63 - 5], pa.uint64())
    total = am.array(keys).group_by(1).sum(am.array(values))
    assert total.type == pa.uint64()
    assert pylist(total) == group_oracle(keys, values, 1, "sum")


def test_argsort_is_a_permutation_with_nulls_last_and_values_in_order():
    """The weaker properties that do hold on the data the xfail above rejects."""
    rng = np.random.default_rng(5)
    for n in (33, 1000):
        values = rng.integers(-50, 50, n).astype(np.int32)
        a = pa.array(values, mask=rng.random(n) < 0.4, type=pa.int32())
        idx = pylist(am.array(a).argsort())
        valid = n - a.null_count
        assert sorted(idx) == list(range(n))
        assert all(a[i].is_valid for i in idx[:valid])
        assert all(not a[i].is_valid for i in idx[valid:])
        assert a.take(pa.array(idx, pa.int32())).to_pylist() == \
            a.take(pc.array_sort_indices(a)).to_pylist()
