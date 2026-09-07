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
import decimal
import math
import os
import re
import struct
import sys

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

# ------------------------------------------------------------------ the extended type matrix
#
# Everything ArrowMetal imports beyond the twelve flat primitives above. The names are the matrix's
# own; ARROW_TYPE maps each to its pyarrow type and the generators below produce it for every shape.

#: A timezone with a DST rule, so assume_timezone / local_timestamp / is_dst have something to do.
TZ_NAME = "America/New_York"
TIME_UNITS = ["s", "ms", "us", "ns"]
#: Ticks in one second, per unit.
TICKS = {"s": 1, "ms": 10 ** 3, "us": 10 ** 6, "ns": 10 ** 9}

TIMESTAMP_NAIVE = ["ts_" + u for u in TIME_UNITS]
TIMESTAMP_TZ = ["ts_" + u + "_tz" for u in TIME_UNITS]
TIMESTAMP_TYPES = TIMESTAMP_NAIVE + TIMESTAMP_TZ
DATE_TYPES = ["date32", "date64"]
TIME_TYPES = ["time32_s", "time32_ms", "time64_us", "time64_ns"]
DURATION_TYPES = ["duration_" + u for u in TIME_UNITS]
#: Every type that carries a date, and so answers the calendar extractors.
DATE_LIKE = TIMESTAMP_TYPES + DATE_TYPES
#: Every temporal type, including the ones that carry no date.
TEMPORAL_TYPES = TIMESTAMP_TYPES + DATE_TYPES + TIME_TYPES + DURATION_TYPES

#: (precision, scale) pairs: a small one, a mid-range one and Arrow's maximum precision.
DECIMAL128_TYPES = ["decimal128_9_2", "decimal128_18_6", "decimal128_38_10"]
SMALL_DECIMAL_TYPES = ["decimal32_9_2", "decimal64_18_4"]
DECIMAL_TYPES = DECIMAL128_TYPES + SMALL_DECIMAL_TYPES
NESTED_TYPES = ["list_int64", "list_utf8", "struct", "map"]
LIST_TYPES = ["list_int64", "list_utf8"]
OTHER_TYPES = ["float16", "fixed_size_binary", "dict_utf8", "run_end_int64", "null", "mdn_interval"]

EXTENDED_TYPES = (TIMESTAMP_TYPES + DATE_TYPES + TIME_TYPES + DURATION_TYPES +
                  DECIMAL_TYPES + NESTED_TYPES + OTHER_TYPES)
#: Every column type the matrix generates, in report order.
MATRIX_TYPES = ALL_TYPES + EXTENDED_TYPES


def _decimal_parts(name):
    """("decimal128_9_2") -> (128, 9, 2)."""
    kind, precision, scale = name.split("_")
    return int(kind[len("decimal"):]), int(precision), int(scale)


def _extended_arrow_type(name):
    if name in TIMESTAMP_TYPES:
        unit = name.split("_")[1]
        return pa.timestamp(unit, TZ_NAME if name.endswith("_tz") else None)
    if name in DURATION_TYPES:
        return pa.duration(name.split("_")[1])
    if name in TIME_TYPES:
        width, unit = name.split("_")
        return pa.time32(unit) if width == "time32" else pa.time64(unit)
    if name in DECIMAL_TYPES:
        bits, precision, scale = _decimal_parts(name)
        return {32: pa.decimal32, 64: pa.decimal64, 128: pa.decimal128}[bits](precision, scale)
    return {
        "date32": pa.date32(), "date64": pa.date64(),
        "list_int64": pa.list_(pa.int64()), "list_utf8": pa.list_(pa.string()),
        "struct": pa.struct([("a", pa.int64()), ("b", pa.string())]),
        "map": pa.map_(pa.string(), pa.int64()),
        "float16": pa.float16(), "fixed_size_binary": pa.binary(8),
        "dict_utf8": pa.dictionary(pa.int32(), pa.string()),
        "run_end_int64": pa.run_end_encoded(pa.int32(), pa.int64()),
        "null": pa.null(),
        "mdn_interval": pa.month_day_nano_interval(),
    }[name]


ARROW_TYPE.update({n: _extended_arrow_type(n) for n in EXTENDED_TYPES})

#: Relative tolerance where the two engines legitimately accumulate in a different order.
FLOAT_TOL = {"float32": 1e-6, "float64": 1e-12}


#: pyarrow spells float32 "float" and float64 "double"; map back to the names used here.
_NAME_OF_TYPE = {ARROW_TYPE[n]: n for n in MATRIX_TYPES}


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

_UNICODE = ["héllo", "日本語", "naïve", "Ωμέγα", "🙂🙃", "é", "ÅNGSTRÖM", "á",
            # A full-width digit run and a title-case code point: the first is what separates ICU's
            # `\d` from RE2's, the second a title-case predicate from an upper-case one.
            "\uff12\uff10\uff12\uff14", "\u01c5ungla"]
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


# ------------------------------------------------------------------ extended value generators
#
# One builder per extended type. Each returns a pyarrow array of `n` rows with the null ratio already
# applied. Where the type has a numeric payload the nulls go on with `pa.array(values, mask=...)`, so
# the values stay *under* the validity bitmap exactly as they do for the flat types; the variable-length
# and nested builders use None rows, which is the only thing pyarrow will build them from.

#: The window the temporal generators draw from: 1900-01-01 to 2100-01-01, in seconds. Wide enough to
#: cross every leap-year and week-numbering rule, narrow enough that a nanosecond timestamp (which runs
#: out in 2262) and a nanosecond difference both stay inside int64.
_EPOCH_LO, _EPOCH_HI = -2_208_988_800, 4_102_444_800


def _seconds_of(y, mo, d, h=0, mi=0, s=0):
    """Seconds from the epoch for a UTC civil time, without importing datetime's timezone rules."""
    a = (14 - mo) // 12
    days = (y + 4800 - a) * 365 + (y + 4800 - a) // 4 - (y + 4800 - a) // 100 + (y + 4800 - a) // 400
    days += (153 * (mo + 12 * a - 3) + 2) // 5 + d - 32045 - 2440588
    return days * 86400 + h * 3600 + mi * 60 + s


#: Instants a temporal kernel is most likely to get wrong, in seconds.
_SPECIAL_SECONDS = [
    0, 1, -1, 86399, 86400, -86400,                       # the epoch and the day around it
    _seconds_of(1900, 1, 1), _seconds_of(2100, 1, 1),     # the ends of the generated window
    _seconds_of(1969, 12, 31, 23, 59, 59), _seconds_of(2000, 1, 1),
    _seconds_of(2000, 2, 29), _seconds_of(2024, 2, 29), _seconds_of(2023, 2, 28),
    _seconds_of(1900, 3, 1), _seconds_of(2024, 12, 31), _seconds_of(2025, 1, 1),
    _seconds_of(2024, 12, 30), _seconds_of(2021, 1, 1), _seconds_of(2021, 1, 3),
    _seconds_of(2024, 3, 10, 7), _seconds_of(2024, 11, 3, 6),   # the two US DST transitions
    _seconds_of(2024, 3, 10, 6, 59, 59), _seconds_of(2024, 11, 3, 5),
    _seconds_of(2024, 6, 30, 23, 59, 59), _seconds_of(1970, 1, 1, 12),
]


def _temporal_ticks(rng, n, unit, flavor):
    """int64 tick values for a timestamp column of `unit`."""
    per_second = TICKS[unit]
    if flavor == "special":
        pool = [s * per_second for s in _SPECIAL_SECONDS]
        pool += [1, -1, per_second - 1, -(per_second - 1)] if per_second > 1 else []
        idx = rng.integers(0, len(pool), n)
        return np.array([pool[i] for i in idx], dtype=np.int64)
    seconds = rng.integers(_EPOCH_LO, _EPOCH_HI, n, dtype=np.int64)
    sub = rng.integers(0, per_second, n, dtype=np.int64) if per_second > 1 else 0
    return seconds * per_second + sub


def _time_of_day_ticks(rng, n, unit, flavor):
    """int64 ticks since midnight, the range a time32 / time64 column is defined on."""
    span = 86400 * TICKS[unit]
    if flavor == "special":
        pool = [0, 1, span - 1, span // 2, 3600 * TICKS[unit], 86399 * TICKS[unit]]
        idx = rng.integers(0, len(pool), n)
        return np.array([pool[i] for i in idx], dtype=np.int64)
    return rng.integers(0, span, n, dtype=np.int64)


def _duration_ticks(rng, n, unit, flavor):
    """Durations up to about a month either way -- small enough that adding one to any generated
    timestamp stays inside the window, so `add_duration` never has to define an overflow."""
    limit = 30 * 86400 * TICKS[unit]
    if flavor == "special":
        pool = [0, 1, -1, limit, -limit, 86400 * TICKS[unit], -(86400 * TICKS[unit]), TICKS[unit]]
        idx = rng.integers(0, len(pool), n)
        return np.array([pool[i] for i in idx], dtype=np.int64)
    return rng.integers(-limit, limit + 1, n, dtype=np.int64)


def _decimal_unscaled(rng, n, precision, scale, flavor):
    """Unscaled integers for a decimal column, as Python ints (10^38 does not fit a numpy dtype)."""
    limit = 10 ** precision - 1
    if flavor == "special":
        pool = [0, 1, -1, limit, -limit, limit - 1, 10 ** scale, -(10 ** scale),
                5, -5, limit // 2, -(limit // 2)]
        return [pool[i] for i in rng.integers(0, len(pool), n)]
    if precision <= 18:
        return [int(v) for v in rng.integers(-limit, limit + 1, n, dtype=np.int64)]
    # Above 18 digits numpy cannot hold the range: draw the digits in two halves.
    hi = rng.integers(0, 10 ** 19, n, dtype=np.uint64)
    lo = rng.integers(0, 10 ** 19, n, dtype=np.uint64)
    sign = rng.integers(0, 2, n)
    return [int(1 - 2 * s) * ((int(a) * 10 ** 19 + int(b)) % (limit + 1))
            for a, b, s in zip(hi, lo, sign)]


def _decimals(rng, n, precision, scale, flavor):
    import decimal as _decimal
    with _decimal.localcontext() as ctx:
        ctx.prec = max(precision + 2, 40)
        return [_decimal.Decimal(v).scaleb(-scale)
                for v in _decimal_unscaled(rng, n, precision, scale, flavor)]


_FLOAT16_SPECIALS = [0.0, -0.0, 1.0, -1.0, 0.5, math.nan, math.inf, -math.inf,
                     65504.0, -65504.0, 6.103515625e-05, -6.103515625e-05,
                     5.960464477539063e-08, -5.960464477539063e-08, 0.0009765625]

_BINARY_SPECIALS = [b"\x00" * 8, b"\xff" * 8, b"abcdefgh", b"ABCDEFGH", b"\x00\x01\x02\x03\x04\x05\x06\x07",
                    b"        ", b"\x80" * 8, b"01234567"]

_MAP_KEYS = ["k", "j", "", "z", "kk", "é"]


def _apply_none(rows, mask):
    return rows if mask is None else [None if m else v for v, m in zip(rows, mask)]


def _extended_array(name, rng, n, flavor, null_ratio):
    """One generated array of an extended type, nulls applied."""
    ty = ARROW_TYPE[name]
    if name == "null":
        return pa.nulls(n)
    if name in TIMESTAMP_TYPES:
        values = _temporal_ticks(rng, n, name.split("_")[1], flavor)
    elif name in DURATION_TYPES:
        values = _duration_ticks(rng, n, name.split("_")[1], flavor)
    elif name in TIME_TYPES:
        values = _time_of_day_ticks(rng, n, name.split("_")[1], flavor)
    elif name == "date32":
        values = _temporal_ticks(rng, n, "s", flavor) // 86400
    elif name == "date64":
        values = (_temporal_ticks(rng, n, "s", flavor) // 86400) * 86_400_000
    elif name == "float16":
        if flavor == "special":
            picked = [_FLOAT16_SPECIALS[i] for i in rng.integers(0, len(_FLOAT16_SPECIALS), n)]
            values = np.array(picked, dtype=np.float16)
        else:
            scale = rng.choice(np.array([1.0, 100.0, 0.01]), n)
            values = (rng.standard_normal(n) * scale).astype(np.float16)
    else:
        values = None

    if values is not None:
        mask = _null_mask(rng, n, null_ratio)
        if name in TIME_TYPES and name.startswith("time32"):
            values = values.astype(np.int32)
        if name == "date32":
            values = values.astype(np.int32)
        return pa.array(values, mask=mask, type=ty)

    mask = _null_mask(rng, n, null_ratio)
    if name in DECIMAL_TYPES:
        _, precision, scale = _decimal_parts(name)
        rows = _decimals(rng, n, precision, scale, flavor)
    elif name == "fixed_size_binary":
        if flavor == "special":
            rows = [_BINARY_SPECIALS[i] for i in rng.integers(0, len(_BINARY_SPECIALS), n)]
        else:
            raw = rng.integers(0, 256, (n, 8), dtype=np.uint8)
            rows = [bytes(r) for r in raw]
    elif name == "mdn_interval":
        months = rng.integers(-30, 31, n, dtype=np.int32)
        days = rng.integers(-40, 41, n, dtype=np.int32)
        nanos = rng.integers(-86_400_000_000_000, 86_400_000_000_000, n, dtype=np.int64)
        rows = [pa.MonthDayNano([int(a), int(b), int(c)]) for a, b, c in zip(months, days, nanos)]
    elif name in LIST_TYPES:
        lengths = rng.integers(0, 5, n)
        if name == "list_int64":
            pool = [0, 1, -1, 7, 2 ** 62, -(2 ** 62), 99]
        else:
            pool = ["a", "", "bb", "héllo", "日本語", "x" * 40, "Z"]
        drawn = rng.integers(0, len(pool), int(lengths.sum()) if n else 0)
        holes = rng.random(len(drawn)) < 0.1
        cursor, rows = 0, []
        for length in lengths:
            item = [None if holes[cursor + k] else pool[drawn[cursor + k]] for k in range(length)]
            cursor += length
            rows.append(item)
    elif name == "struct":
        a = rng.integers(-1000, 1000, n, dtype=np.int64)
        pool = ["a", "", "bb", "héllo", "x" * 40]
        b = rng.integers(0, len(pool), n)
        holes = rng.random(n) < 0.1
        rows = [{"a": None if h else int(av), "b": pool[bv]} for av, bv, h in zip(a, b, holes)]
    elif name == "map":
        lengths = rng.integers(0, 4, n)
        drawn = rng.integers(0, len(_MAP_KEYS), int(lengths.sum()) if n else 0)
        values = rng.integers(-100, 100, len(drawn), dtype=np.int64)
        cursor, rows = 0, []
        for length in lengths:
            item = [(_MAP_KEYS[drawn[cursor + k]], int(values[cursor + k])) for k in range(length)]
            cursor += length
            rows.append(item)
    elif name == "dict_utf8":
        pool = string_pool(n)
        rows = [pool[i] for i in rng.integers(0, len(pool), n)]
    elif name == "run_end_int64":
        # Runs of 1 to 6 equal values, so the encoding actually compresses and the run boundaries do
        # not line up with any threadgroup width.
        rows, cursor = [], 0
        while cursor < n:
            length = int(rng.integers(1, 7))
            rows += [int(rng.integers(-50, 50))] * min(length, n - cursor)
            cursor += length
    else:                                                  # pragma: no cover - generator bug
        raise AssertionError(f"no generator for {name}")

    rows = _apply_none(rows, mask)
    if name == "dict_utf8":
        return pa.array(rows, type=pa.string()).dictionary_encode().cast(ty)
    if name == "run_end_int64":
        return pc.run_end_encode(pa.array(rows, type=pa.int64()))
    return pa.array(rows, type=ty)


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
                                 len(shape.flavor), MATRIX_TYPES.index(name)])
    n = shape.size + (2 * shape.offset if shape.flavor == "sliced" else 0)
    if name in EXTENDED_TYPES:
        arr = _extended_array(name, rng, n, shape.flavor, shape.null_ratio)
        if shape.flavor == "sliced" and shape.offset:
            arr = arr.slice(shape.offset, shape.size)
        assert len(arr) == shape.size, f"generator produced {len(arr)} rows for {shape.id}"
        _CACHE.put(key, arr)
        return arr
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
    # np.floating covers float16, which pyarrow hands back as numpy scalars rather than Python floats.
    if isinstance(got, (float, np.floating)) or isinstance(expected, (float, np.floating)):
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

def _register_optional(name, types, methods, fn, tol=None, note=""):
    missing = [m for m in methods if not hasattr(am.MetalArray, m)]
    if missing:
        ABSENT.append((name, ", ".join(missing)))
        return
    OPS.append(Op(name, types, fn, tol,
                  note=note or "optional kernel, discovered at import"))


def _unary_optional(name, method, oracle, types, tol=None, note=""):
    def fn(src, shape, method=method, oracle=oracle):
        return arrow(getattr(am.array(src), method)()), oracle(src)
    _register_optional(name, types, [method], fn, tol, note)


def _sign_oracle(src):
    """`pc.sign` narrows an integer column to int8; ArrowMetal's unary math kernels keep the column's
    own type. -1/0/1 are representable in every integer type, so casting the oracle back loses nothing
    from the value comparison, and the type difference itself is pinned by
    test_sign_keeps_the_column_type_where_pyarrow_narrows_to_int8."""
    signs = pc.sign(src)
    return signs.cast(src.type) if signs.type != src.type else signs


_unary_optional("is_null", "is_null", pc.is_null, ALL_TYPES)
_unary_optional("is_valid", "is_valid", pc.is_valid, ALL_TYPES)
_unary_optional("is_nan", "is_nan", pc.is_nan, FLOATING)
_unary_optional("abs", "abs", pc.abs, NUMERIC)
_unary_optional("negate", "negate", pc.negate, SIGNED + FLOATING)
_unary_optional("sign", "sign", _sign_oracle, NUMERIC,
                note="pc.sign returns int8 for an integer column; the oracle is cast back")
_unary_optional("upper", "upper", pc.utf8_upper, ["utf8"])
_unary_optional("lower", "lower", pc.utf8_lower, ["utf8"])
_unary_optional("trim", "trim", pc.utf8_trim_whitespace, ["utf8"])
_unary_optional("reverse", "reverse", pc.utf8_reverse, ["utf8"])
_unary_optional("bitwise_not", "bitwise_not", pc.bit_wise_not, INTEGER)


# ---- cumulative ----------------------------------------------------
#
# Two options have to be spelled out for the oracle to mean what the kernels implement:
#
#   * `skip_nulls`. ArrowMetal's cumulative kernels carry the running value across nulls and leave the
#     output null exactly where the input is, which is Arrow's `skip_nulls=True`. Arrow's *default* is
#     False -- the first null poisons the rest of the column. ArrowMetal has no such mode;
#     test_cumulative_functions_skip_nulls_where_arrow_propagates_them pins the difference.
#   * `start`. Arrow's default start for `cumulative_max` is `numeric_limits<T>::min()`, which on a
#     float column is the smallest positive *normal*, so Arrow's default clamps every negative maximum
#     to 1.18e-38 (and `cumulative_min` clamps +inf down to FLT_MAX). The neutral identity is passed
#     explicitly instead; test_arrow_cumulative_max_default_start_clamps_negative_floats pins why.

def _cumulative_oracle(fn, float_start=None):
    def oracle(src):
        if float_start is not None and type_name_of(src) in FLOATING:
            return fn(src, start=pa.scalar(float_start, src.type), skip_nulls=True)
        return fn(src, skip_nulls=True)
    return oracle


def _cumulative_sum(src, shape):
    """The two engines associate a float running sum differently -- ArrowMetal scans in blocks, Arrow
    accumulates left to right -- so float columns get the same roundoff bound the reductions use,
    with two adjustments:

      * values that can make a partial sum overflow are filtered out of a float column: ±inf, and
        anything above `type_max / n`. Past that point the divergence is not roundoff at all -- one
        association reaches ±inf where the other does not, and `inf - inf` is NaN -- so no bound can
        express it. NaN itself stays in: it poisons every later element in both engines alike, and
        test_cumulative_sum_propagates_nan_and_reassociates_infinities pins both halves.
      * the roundoff bound is the reductions' `8·eps·sqrt(n)·Σ|x|`, measured over the same input."""
    name = type_name_of(src)
    if name in FLOATING:
        filled = pc.fill_null(src, pa.scalar(0.0, src.type))
        limit = pa.scalar(float(np.finfo(NUMPY_TYPE[name]).max) / max(len(src), 1), src.type)
        src = src.filter(pc.or_(pc.is_nan(filled), pc.less_equal(pc.abs(filled), limit)))
    got = arrow(am.array(src).cumulative_sum())
    expected = pc.cumulative_sum(src, skip_nulls=True)
    if name not in FLOATING:
        return got, expected
    return got, expected, (FLOAT_TOL[name], float_sum_bound(_drop_nan(src), name))


_register_optional("cumulative_sum", NUMERIC, ["cumulative_sum"], _cumulative_sum)
_unary_optional("cumulative_prod", "cumulative_prod", _cumulative_oracle(pc.cumulative_prod),
                NUMERIC, tol="float")
_unary_optional("cumulative_max", "cumulative_max",
                _cumulative_oracle(pc.cumulative_max, float_start=-math.inf), NUMERIC)
_unary_optional("cumulative_min", "cumulative_min",
                _cumulative_oracle(pc.cumulative_min, float_start=math.inf), NUMERIC)


def _binary_optional(name, method, oracle, types):
    def fn(src, shape, method=method, oracle=oracle):
        other = make_array(type_name_of(src), shape, seed=1)
        return arrow(getattr(am.array(src), method)(am.array(other))), oracle(src, other)
    _register_optional(name, types, [method], fn)


_binary_optional("bitwise_and", "bitwise_and", pc.bit_wise_and, INTEGER)
_binary_optional("bitwise_or", "bitwise_or", pc.bit_wise_or, INTEGER)
_binary_optional("bitwise_xor", "bitwise_xor", pc.bit_wise_xor, INTEGER)


#: Shift counts both engines define the same way: `std::numeric_limits<T>::digits`, which excludes the
#: sign bit, so int32 agrees on 0..30 and uint32 on 0..31.
def _shift_digits(name):
    return NUMPY_TYPE[name].itemsize * 8 - (1 if name in SIGNED else 0)


def _shift_counts(other, name):
    """The generated column, folded into the shift counts both engines agree on.

    Out of that range the two diverge by design -- ArrowMetal shifts the bits out (0, or the sign fill
    for a signed `shift_right`), pyarrow's unchecked kernel returns the operand untouched and
    `shift_left_checked` raises -- so the matrix stays inside it and
    test_out_of_range_shift_counts_shift_the_bits_out_where_pyarrow_returns_the_operand pins the rest.
    Nulls in the count column are preserved, so null propagation is still under test."""
    wide = pa.uint64() if name in UNSIGNED else pa.int64()
    c = pc.cast(other, wide, safe=False)
    c = pc.bit_wise_and(c, pa.scalar(2 ** 63 - 1, wide))            # drop the sign bit: counts >= 0
    digits = pa.scalar(_shift_digits(name), wide)
    c = pc.subtract(c, pc.multiply(pc.divide(c, digits), digits))   # c % digits, exactly
    return pc.cast(c, ARROW_TYPE[name], safe=False)


def _shift_optional(name, method, oracle):
    def fn(src, shape, method=method, oracle=oracle):
        type_name = type_name_of(src)
        counts = _shift_counts(make_array(type_name, shape, seed=1), type_name)
        return arrow(getattr(am.array(src), method)(am.array(counts))), oracle(src, counts)
    _register_optional(name, INTEGER, [method], fn,
                       note="shift counts folded into [0, digits); out-of-range counts pinned apart")


_shift_optional("shift_left", "shift_left", pc.shift_left)
_shift_optional("shift_right", "shift_right", pc.shift_right)


def _fill_null(src, shape):
    name = type_name_of(src)
    v = "" if name == "utf8" else (False if name == "bool" else scalar_for(name))
    return arrow(am.array(src).fill_null(v)), pc.fill_null(src, pa.scalar(v, src.type))


_register_optional("fill_null", ALL_TYPES, ["fill_null"], _fill_null)


def _if_else(src, shape):
    """In ArrowMetal the boolean condition is the receiver: `cond.if_else(left, right)`."""
    cond = make_array("bool", shape, seed=2)
    other = make_array(type_name_of(src), shape, seed=1)
    return arrow(am.array(cond).if_else(am.array(src), am.array(other))), pc.if_else(cond, src, other)


_register_optional("if_else", ALL_TYPES, ["if_else"], _if_else)


def _is_in(src, shape):
    """`skip_nulls=True` is the mode ArrowMetal implements: a null in the value set is ignored and a
    null element never matches, so the result has no nulls. Arrow's default (`skip_nulls=False`) makes
    a null in the value set match a null element -- pinned by
    test_is_in_never_matches_a_null_where_arrow_matches_null_to_null."""
    values = src.slice(0, min(len(src), 5))
    return (arrow(am.array(src).is_in(am.array(values))),
            pc.is_in(src, value_set=values, skip_nulls=True))


_register_optional("is_in", ALL_TYPES, ["is_in"], _is_in)


def _index_in(src, shape):
    values = src.slice(0, min(len(src), 5))
    return (arrow(am.array(src).index_in(am.array(values))),
            pc.index_in(src, value_set=values, skip_nulls=True).cast(pa.int32()))


_register_optional("index_in", ALL_TYPES, ["index_in"], _index_in)


def _drop_null(src, shape):
    return arrow(am.array(src).drop_null()), pc.drop_null(src)


_register_optional("drop_null", ALL_TYPES, ["drop_null"], _drop_null)


def _kleene(src, shape):
    other = make_array("bool", shape, seed=1)
    x, y = am.array(src), am.array(other)
    return ([arrow(x.and_kleene(y)), arrow(x.or_kleene(y))],
            [pc.and_kleene(src, other), pc.or_kleene(src, other)])


_register_optional("kleene", ["bool"], ["and_kleene", "or_kleene"], _kleene,
                   note="three-valued logic, as opposed to the null-propagating & and |")


# ---- element-wise math beyond + - * / -------------------------------

def _rounding(src, shape):
    """`pc.floor`/`ceil`/`trunc` widen an integer column to double; ArrowMetal's unary kernels are the
    identity there and keep the column's type, so the oracle is cast back. Above 2^53 that cast is
    Arrow's loss, not a kernel's, so those rows are dropped from an integer column. `pc.round`
    defaults to half-to-even; ArrowMetal rounds halves away from zero, Arrow's
    `half_towards_infinity`."""
    name = type_name_of(src)
    if name in INTEGER:
        wide = pc.cast(src, pa.int64() if name in SIGNED else pa.uint64(), safe=False)
        limit = pa.scalar(2 ** 53, wide.type)
        inside = pc.less_equal(wide, limit)
        if name in SIGNED:
            inside = pc.and_(inside, pc.greater_equal(wide, pa.scalar(-2 ** 53, wide.type)))
        src = src.filter(pc.fill_null(inside, True))
    x = am.array(src)
    got, expected = [], []
    for method, fn in [("floor", pc.floor), ("ceil", pc.ceil), ("trunc", pc.trunc),
                       ("round", lambda a: pc.round(a, round_mode="half_towards_infinity"))]:
        result = fn(src)
        got.append(arrow(getattr(x, method)()))
        expected.append(result.cast(src.type) if result.type != src.type else result)
    return got, expected


_register_optional("rounding", NUMERIC, ["floor", "ceil", "round", "trunc"], _rounding)


def _eps_of(src):
    return _EPS["float64"] if pa.types.is_float64(src.type) else _EPS["float32"]


def _float32_representable(src):
    """For a float32 column, the values its `float` kernels can carry. A float64 column is evaluated
    in software binary64 (Kernels/DoubleTranscendental.swift) over the whole double range, so it is
    returned untouched; test_float64_transcendentals_are_true_binary64 pins that."""
    if pa.types.is_float64(src.type):
        return src
    filled = pc.fill_null(src, pa.scalar(0.0, src.type))
    magnitude = pc.abs(filled)
    tiny = pa.scalar(float(np.finfo(np.float32).tiny), src.type)
    big = pa.scalar(float(np.finfo(np.float32).max), src.type)
    inside = pc.and_(pc.greater_equal(magnitude, tiny), pc.less_equal(magnitude, big))
    return src.filter(pc.or_(pc.equal(filled, pa.scalar(0.0, src.type)), inside))


def _evaluated_in_float(src, methods, oracles):
    src = _float32_representable(src)
    x = am.array(src)
    got, expected = [], []
    for method, fn in zip(methods, oracles):
        result = fn(src)
        got.append(arrow(getattr(x, method)()))
        expected.append(result.cast(src.type) if result.type != src.type else result)
    return src, got, expected


def _sqrt(src, shape):
    """Evaluated in `float` on both float types -- about 7 correct significant digits, hence the
    relative 1e-6 -- over the values a float32 can hold
    (test_float64_transcendentals_are_evaluated_in_float32 pins what happens outside)."""
    _, got, expected = _evaluated_in_float(src, ["sqrt"], [pc.sqrt])
    # float32: ~7 digits; float64: correctly rounded, so a couple of ulp covers the oracle's own libm.
    return got, expected, ((1e-6, 0.0) if pa.types.is_float32(src.type) else (4.0 * _EPS["float64"], 0.0))


_register_optional("sqrt", FLOATING, ["sqrt"], _sqrt)


def _exp(src, shape):
    """`exp` needs its *result* inside the float32 range as well as its argument: past |x| ~ 87 a
    float32 exponential is inf or zero while Arrow's double is still finite.

    Its tolerance is not a constant. Rounding the argument to float32 moves it by |x|·eps32, and
    exp(x + d) = exp(x)·(1 + d), so the *relative* error of the result grows with |x| -- 3.5e-6 at
    x = -58, which no 1e-6 bound could accept. The bound below is that mechanism, with a margin."""
    base = _float32_representable(src)
    magnitude = pc.abs(pc.fill_null(base, pa.scalar(0.0, base.type)))
    limit = 87.0 if pa.types.is_float32(base.type) else 700.0
    small = base.filter(pc.less_equal(magnitude, pa.scalar(limit, base.type)))
    _, got, expected = _evaluated_in_float(small, ["exp"], [pc.exp])
    peak = pc.max(pc.abs(pc.fill_null(small, pa.scalar(0.0, small.type)))).as_py()
    peak = 0.0 if peak is None or not math.isfinite(peak) else abs(peak)
    return got, expected, (4.0 * _eps_of(src) * (1.0 + peak), 0.0)


_register_optional("exp", FLOATING, ["exp"], _exp)


def _logarithm(src, shape):
    """The logarithms need an absolute bound as well as a relative one: rounding the argument to
    float32 moves ln(x) by about `eps32` *whatever the size of the result*, so a result near zero --
    ln of anything close to 1.0 -- has no meaningful relative accuracy at all."""
    _, got, expected = _evaluated_in_float(src, ["ln", "log10", "log2"],
                                           [pc.ln, pc.log10, pc.log2])
    return got, expected, (1e-6, 1e-6)


_register_optional("logarithm", FLOATING, ["ln", "log10", "log2"], _logarithm)


def _element_wise_min_max(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    x, y = am.array(src), am.array(other)
    return ([arrow(x.min_element_wise(y)), arrow(x.max_element_wise(y))],
            [pc.min_element_wise(src, other), pc.max_element_wise(src, other)])


_register_optional("element_wise_min_max", NUMERIC, ["min_element_wise", "max_element_wise"],
                   _element_wise_min_max, note="Arrow's skip_nulls default: one null yields the other side")


def _modulo(src, shape):
    """C remainder, built from the truncating division both engines already agree on. 0 and -1 are
    kept out of the divisor for the same reason `arith_array` keeps them out."""
    name = type_name_of(src)
    other = make_array(name, shape, seed=1)
    bad = pc.is_in(other, value_set=pa.array([0, -1] if name in SIGNED else [0], other.type))
    divisor = pc.if_else(bad, pa.scalar(1, other.type), other)
    expected = pc.subtract(src, pc.multiply(pc.divide(src, divisor), divisor))
    return arrow(am.array(src).modulo(am.array(divisor))), expected


_register_optional("modulo", INTEGER, ["modulo"], _modulo)


def _power(src, shape):
    """Exponents are folded into [0, 8]: a negative one is defined as 0 here and raises in Arrow, and
    beyond a handful the wrapped repeated squaring has nothing left to compare."""
    name = type_name_of(src)
    other = make_array(name, shape, seed=1)
    wide = pa.uint64() if name in UNSIGNED else pa.int64()
    e = pc.bit_wise_and(pc.cast(other, wide, safe=False), pa.scalar(7, wide))
    exponent = pc.cast(e, other.type, safe=False)
    return arrow(am.array(src).power(am.array(exponent))), pc.power(src, exponent)


_register_optional("power", INTEGER, ["power"], _power)


# ---- string transforms ---------------------------------------------

for _name, _method, _oracle in [
        ("ascii_upper", "ascii_upper", pc.ascii_upper),
        ("ascii_lower", "ascii_lower", pc.ascii_lower),
        ("swapcase", "swapcase", pc.ascii_swapcase),
        ("capitalize", "capitalize", pc.ascii_capitalize),
        ("ltrim", "ltrim", pc.ascii_ltrim_whitespace),
        ("rtrim", "rtrim", pc.ascii_rtrim_whitespace),
        ("str_reverse", "str_reverse", pc.utf8_reverse),
        ("is_alnum", "is_alnum", pc.ascii_is_alnum),
        ("is_alpha", "is_alpha", pc.ascii_is_alpha),
        ("is_digit", "is_digit", pc.ascii_is_decimal),
        ("is_space", "is_space", pc.ascii_is_space),
        ("is_upper", "is_upper", pc.ascii_is_upper),
        ("is_lower", "is_lower", pc.ascii_is_lower)]:
    _unary_optional(_name, _method, _oracle, ["utf8"])


def _replace(src, shape):
    """The empty pattern is left out: ArrowMetal defines it as the identity and Arrow inserts the
    replacement between every code point (test_empty_replace_pattern_is_the_identity pins it)."""
    x = am.array(src)
    got, expected = [], []
    for pattern, replacement, limit in [("a", "X", -1), ("app", "", -1), ("a", "yy", 1),
                                        ("é", "e", -1), ("zzz", "!", -1)]:
        got.append(arrow(x.replace(pattern, replacement, limit)))
        expected.append(pc.replace_substring(src, pattern=pattern, replacement=replacement,
                                             max_replacements=None if limit < 0 else limit))
    return got, expected


_register_optional("replace", ["utf8"], ["replace"], _replace)


def _repeat(src, shape):
    x = am.array(src)
    return ([arrow(x.repeat(0)), arrow(x.repeat(1)), arrow(x.repeat(3))],
            [pc.binary_repeat(src, 0), pc.binary_repeat(src, 1), pc.binary_repeat(src, 3)])


_register_optional("repeat", ["utf8"], ["repeat"], _repeat)


def _slice_codeunits(src, shape):
    x = am.array(src)
    got, expected = [], []
    for start, stop in [(0, None), (1, 3), (2, None), (-3, None), (0, -1), (5, 2)]:
        got.append(arrow(x.slice_codeunits(start, stop)))
        expected.append(pc.utf8_slice_codeunits(src, start) if stop is None
                        else pc.utf8_slice_codeunits(src, start, stop))
    return got, expected


_register_optional("slice_codeunits", ["utf8"], ["slice_codeunits"], _slice_codeunits)


def _pad(src, shape):
    x = am.array(src)
    got, expected = [], []
    for width, pad in [(0, " "), (6, " "), (9, "*")]:
        got += [arrow(x.pad_left(width, pad)), arrow(x.pad_right(width, pad))]
        expected += [pc.utf8_lpad(src, width, padding=pad), pc.utf8_rpad(src, width, padding=pad)]
    return got, expected


_register_optional("pad", ["utf8"], ["pad_left", "pad_right"], _pad)


def _substring_search(src, shape):
    """The empty pattern is left out of `count_substring`: ArrowMetal counts code points + 1 and
    Arrow counts bytes + 1, which only differ on multi-byte input
    (test_empty_pattern_counts_code_points_where_arrow_counts_bytes)."""
    x = am.array(src)
    got, expected = [], []
    for pattern in ["a", "app", "é", "\t", "zzz", "x" * 300]:
        got += [arrow(x.count_substring(pattern)), arrow(x.find_substring(pattern))]
        expected += [pc.count_substring(src, pattern).cast(pa.int32()),
                     pc.find_substring(src, pattern).cast(pa.int32())]
    return got, expected


_register_optional("substring_search", ["utf8"], ["count_substring", "find_substring"],
                   _substring_search)


def _str_concat(src, shape):
    other = make_array("utf8", shape, seed=1)
    x, y = am.array(src), am.array(other)
    return ([arrow(x.str_concat(y)), arrow(x.str_concat(y, "-"))],
            [pc.binary_join_element_wise(src, other, ""),
             pc.binary_join_element_wise(src, other, "-")])


_register_optional("str_concat", ["utf8"], ["str_concat"], _str_concat)


def _dictionary_decode(src, shape):
    """`decode` materialises a dictionary-encoded column; the oracle is Arrow's own decode of the
    same pa.DictionaryArray."""
    encoded = src.dictionary_encode()
    return arrow(am.array(encoded).decode()), encoded.cast(pa.string())


_register_optional("dictionary_decode", ["utf8"], ["decode"], _dictionary_decode)


# ==================================================================== the extended matrix
#
# Everything below covers the kernels added after the first round: decimals, the nested types, the
# regular expressions and string casts, the whole temporal surface, the window and rolling functions,
# the statistical aggregates, run-end encoding, the remaining Arrow type rows and the trigonometry.
#
# The rules are the ones above. pyarrow is the oracle wherever Arrow has a function, with its options
# spelled out rather than defaulted; where Arrow has none the case says so, names itself in _NO_ORACLE
# and compares against a Python reference implemented here instead.

#: Operations Arrow has no counterpart for. Each is checked against a reference written in this file
#: (or against a property, for the hashes); test_no_oracle_operations_are_reported prints the list.
_NO_ORACLE = {}


def _no_oracle(name, why):
    _NO_ORACLE[name] = why
    return name


def _pylist(arr):
    return arr.to_pylist() if isinstance(arr, pa.Array) else list(arr)


def _valid_positions(src):
    """The indices of the non-null rows, as a Python list."""
    return [i for i, v in enumerate(pc.is_valid(src).to_pylist()) if v]


# ---- interop and selection over the extended types -------------------

#: Every extended type except the run-end encoded one, whose selection kernels decode rather than
#: re-encode and which pyarrow has no filter / take kernel for at all (see the `run_end` op).
SELECTABLE_TYPES = [t for t in EXTENDED_TYPES if t != "run_end_int64"]


@op("roundtrip_ext", EXTENDED_TYPES, note="import/export through the Arrow C Data Interface")
def _roundtrip_ext(src, shape):
    x = am.array(src)
    assert len(x) == len(src), f"length changed on import: {len(x)} != {len(src)}"
    if not pa.types.is_run_end_encoded(src.type):
        # A run-end encoded array keeps its nulls in the values child, so pyarrow reports null_count 0
        # for it whatever the data says; ArrowMetal reports the logical count
        # (test_run_end_null_count_is_logical pins the difference).
        assert x.null_count == src.null_count, f"null_count {x.null_count} != {src.null_count}"
    return x.to_arrow(), src


@op("filter_ext", SELECTABLE_TYPES)
def _filter_ext(src, shape):
    mask = make_array("bool", shape, seed=2)
    return arrow(am.array(src).filter(am.array(mask))), src.filter(mask)


@op("take_ext", SELECTABLE_TYPES)
def _take_ext(src, shape):
    n = len(src)
    if n == 0:
        idx = pa.array([], pa.int32())
    else:
        rng = np.random.default_rng(11)
        raw = rng.integers(0, n, min(n, 977)).astype(np.int32)
        idx = pa.array(raw, mask=rng.random(len(raw)) < 0.1, type=pa.int32())
    return arrow(am.array(src).take(am.array(idx))), src.take(idx)


@op("slice_ext", SELECTABLE_TYPES)
def _slice_ext(src, shape):
    n = len(src)
    off = n // 3
    return arrow(am.array(src).slice(off, n - off)), src.slice(off, n - off)


@op("run_end", ["run_end_int64"],
    note=_no_oracle("run_end", "pyarrow has no filter/take/slice kernel for a run-end encoded array; "
                               "the reference is the decoded column, put through pyarrow's own kernels"))
def _run_end(src, shape):
    """`pc.run_end_encode` is the oracle for the encoding itself. Everything else is compared against
    the decoded column: ArrowMetal's filter, take and slice on a run-end array *decode* it, which is
    the one shape pyarrow cannot answer at all (it has no kernel for the type)."""
    x = am.array(src)
    decoded = pc.run_end_decode(src)
    mask = make_array("bool", shape, seed=2)
    n = len(src)
    idx = pa.array(np.random.default_rng(11).integers(0, max(n, 1), min(n, 977)).astype(np.int32),
                   type=pa.int32()) if n else pa.array([], pa.int32())
    off = n // 3
    got = [arrow(x.run_end_decode()), arrow(am.array(decoded).run_end_encode()),
           arrow(x.filter(am.array(mask))), arrow(x.take(am.array(idx))),
           arrow(x.slice(off, n - off))]
    expected = [decoded, pc.run_end_encode(decoded),
                decoded.filter(mask), decoded.take(idx), decoded.slice(off, n - off)]
    return got, expected


# ---- decimals --------------------------------------------------------
#
# Every decimal kernel is `am_decimal_op`; the scalar form takes 16 little-endian bytes at the column's
# own scale. Arrow's own decimal kernels promote the result type (add gives precision p+1, multiply
# gives p1+p2+1 and scale s1+s2) while ArrowMetal's add and subtract keep the operand type and wrap, so
# the oracle is cast back to the column type -- the values are identical inside the operand's range,
# which is where the generator stays.

def _unscaled_magnitude(value):
    """|unscaled| of a Decimal that came out of a decimal column, without going through the default
    decimal context (whose 28 digits would round a 38-digit value)."""
    return int("".join(str(d) for d in value.as_tuple().digits) or "0")


def _decimal_scalar_for(name):
    """A constant every decimal type in the matrix can hold, as a Decimal at that type's scale."""
    import decimal as _decimal
    _, _, scale = _decimal_parts(name)
    return _decimal.Decimal(3).scaleb(-min(scale, 2))


def _decimal_arith_oracle(fn, src, other):
    """pc.add / pc.subtract on decimals widen the precision by one, and refuse a 38-digit column
    outright because 39 is out of range -- so the oracle runs in decimal256 and is cast back to the
    column's own type, which is what ArrowMetal's wrapping kernels return."""
    if src.type.precision >= 38:
        wide = pa.decimal256(src.type.precision, src.type.scale)
        result = fn(src.cast(wide), other.cast(wide) if isinstance(other, pa.Array) else
                    pa.scalar(other.as_py(), wide))
    else:
        result = fn(src, other)
    return result if result.type == src.type else result.cast(src.type, safe=False)


def _decimal_fits(src, other, headroom=8):
    """The rows on which every form of `decimal_arith` stays inside the column's own precision: the
    sum and difference of the two columns, and the column multiplied by the small integer scalar.

    Past the precision both engines wrap, but not to the same place: ArrowMetal wraps modulo 2^128 (the
    storage width) and Arrow's unchecked decimal cast wraps modulo 10^precision. Neither is a documented
    Arrow behaviour, so the matrix stays inside the range and
    test_decimal_arithmetic_wraps_modulo_two_to_the_128 pins what happens outside it."""
    limit = 10 ** src.type.precision
    scaled = 10 ** src.type.scale
    keep = []
    for a, b in zip(src.to_pylist(), other.to_pylist()):
        if a is None or b is None:
            keep.append(True)
            continue
        keep.append(headroom * max(_unscaled_magnitude(a), _unscaled_magnitude(b)) + scaled < limit)
    mask = pa.array(keep, pa.bool_())
    return src.filter(mask), other.filter(mask)


@op("decimal_compare", DECIMAL128_TYPES)
def _decimal_compare(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    scalar = _decimal_scalar_for(type_name_of(src))
    x, y = am.array(src), am.array(other)
    got, expected = [], []
    for symbol, fn in _CMP:
        got += [arrow(x.compare(symbol, y)), arrow(x.compare(symbol, scalar))]
        expected += [fn(src, other), fn(src, pa.scalar(scalar, src.type))]
    return got, expected


@op("decimal_arith", DECIMAL128_TYPES,
    note="add/subtract keep the column type and wrap; pc.add widens the precision, so it is cast back")
def _decimal_arith(src, shape):
    name = type_name_of(src)
    _, precision, scale = _decimal_parts(name)
    src, other = _decimal_fits(src, make_array(name, shape, seed=1))
    scalar = _decimal_scalar_for(name)
    x, y = am.array(src), am.array(other)
    got = [arrow(x.decimal_add(y)), arrow(x.decimal_sub(y)),
           arrow(x.decimal_add(scalar)), arrow(x.decimal_sub(scalar))]
    expected = [_decimal_arith_oracle(pc.add, src, other), _decimal_arith_oracle(pc.subtract, src, other),
                _decimal_arith_oracle(pc.add, src, pa.scalar(scalar, src.type)),
                _decimal_arith_oracle(pc.subtract, src, pa.scalar(scalar, src.type))]
    # multiply widens to precision p1+p2+1: Arrow refuses past 38, so the widest column is compared
    # only in the scalar form (which keeps the type).
    if 2 * precision + 1 <= 38:
        got.append(arrow(x.decimal_mul(y)))
        expected.append(pc.multiply(src, other))
    # The int-scalar form keeps the column type; Arrow's multiply widens to p1 + p2 + 1, which is out
    # of decimal128's range for a 38-digit column, so that oracle runs in decimal256 as well.
    wide = pa.decimal256(precision, scale)
    got.append(arrow(x.decimal_mul(7)))
    expected.append(pc.multiply(src.cast(wide), pa.scalar(7, pa.decimal256(2, 0)))
                    .cast(src.type, safe=False))
    return got, expected


@op("decimal_unary", DECIMAL128_TYPES, note="pc.sign narrows to int8; ArrowMetal's decimal sign is int32")
def _decimal_unary(src, shape):
    x = am.array(src)
    return ([arrow(x.negate()), arrow(x.abs()), arrow(x.sign())],
            [pc.negate(src), pc.abs(src), pc.sign(src).cast(pa.int32())])


def _decimal_round_targets(precision, scale):
    """The scales `decimal_round` is asked for: down to 0, one either side of the column's own, and as
    far up as the widened precision still fits decimal128 (scaling up multiplies, so the result type is
    `decimal128(precision + target - scale, target)` and 38 is the ceiling)."""
    return sorted({0, 1, max(scale - 1, 0), scale, min(scale + 2, scale + 38 - precision)})


@op("decimal_round", DECIMAL128_TYPES,
    note="ArrowMetal rescales the column to the target scale; pc.round keeps the input scale, so the "
         "oracle is cast to the rescaled type")
def _decimal_round(src, shape):
    name = type_name_of(src)
    _, precision, scale = _decimal_parts(name)
    x = am.array(src)
    got, expected = [], []
    for target in _decimal_round_targets(precision, scale):
        for mode, arrow_mode in [("round", "half_towards_infinity"), ("ceil", "up"),
                                 ("floor", "down"), ("truncate", "towards_zero")]:
            result = arrow(x.decimal_round(target, mode))
            got.append(result)
            expected.append(pc.round(src, ndigits=target, round_mode=arrow_mode)
                            .cast(result.type, safe=False))
    return got, expected


@op("decimal_to_float64", DECIMAL128_TYPES,
    note="ArrowMetal divides the unscaled value by 10^scale; Arrow multiplies by its reciprocal")
def _decimal_to_float64(src, shape):
    return arrow(am.array(src).to_float64()), src.cast(pa.float64())


@op("decimal_reduce", DECIMAL128_TYPES, note="sum/min/max come back as Decimals, like pc.sum's scalar")
def _decimal_reduce(src, shape):
    x = am.array(src)
    return ([x.sum(), x.min(), x.max()],
            [pc.sum(src).as_py(), pc.min(src).as_py(), pc.max(src).as_py()])


@op("decimal_widen_narrow", SMALL_DECIMAL_TYPES)
def _decimal_widen_narrow(src, shape):
    bits, precision, scale = _decimal_parts(type_name_of(src))
    wide = am.array(src).to_decimal128()
    narrowed = wide.to_small_decimal(bits, precision)
    return ([arrow(wide), arrow(narrowed)],
            [src.cast(pa.decimal128(precision, scale)), src])


# ---- nested: list, struct and map ------------------------------------

def _list_oracle_type(src):
    """A map is a list of struct<key, value>; `pc.list_value_length` has no map kernel, so the map is
    cast to that equivalent list first (which Arrow does support)."""
    if pa.types.is_map(src.type):
        entries = pa.struct([("key", src.type.key_type), ("value", src.type.item_type)])
        return src.cast(pa.list_(entries))
    return src


@op("list_length", LIST_TYPES + ["map"],
    note="int32 per-row child count; pc.list_value_length returns int32 for list and has no map kernel")
def _list_length(src, shape):
    view = _list_oracle_type(src)
    return arrow(am.array(src).list_value_length()), pc.list_value_length(view).cast(pa.int32())


@op("list_flatten", LIST_TYPES)
def _list_flatten(src, shape):
    return arrow(am.array(src).list_flatten()), pc.list_flatten(src)


@op("list_element", LIST_TYPES,
    note="a row shorter than the index is null here and an error in Arrow, so the oracle sees only the "
         "rows long enough (test_list_element_of_a_short_row_is_null pins the rest)")
def _list_element(src, shape):
    x = am.array(src)
    got, expected = [], []
    for index in (0, 1, 3):
        lengths = pc.fill_null(pc.list_value_length(src), 0)
        keep = pc.greater(lengths, index)
        subset = src.filter(keep)
        got.append(arrow(am.array(subset).list_element(index)) if len(subset) else
                   pc.list_element(src.slice(0, 0), index))
        expected.append(pc.list_element(subset, index) if len(subset) else
                        pc.list_element(src.slice(0, 0), index))
    del x
    return got, expected


@op("list_parent_indices", LIST_TYPES,
    note="int32 here, int64 in Arrow -- list offsets are int32 throughout this package")
def _list_parent_indices(src, shape):
    return (arrow(am.array(src).list_parent_indices()),
            pc.list_parent_indices(src).cast(pa.int32()))


@op("list_slice", LIST_TYPES,
    note="pc.list_slice's return_fixed_size_list=False, the variable-length form ArrowMetal produces")
def _list_slice(src, shape):
    x = am.array(src)
    got, expected = [], []
    for start, stop, step in [(0, None, 1), (1, None, 1), (0, 2, 1), (1, 3, 1), (0, 4, 2), (2, 3, 1)]:
        got.append(arrow(x.list_slice(start, stop, step)))
        expected.append(pc.list_slice(src, start, stop, step, return_fixed_size_list=False))
    return got, expected


@op("struct_field", ["struct"])
def _struct_field(src, shape):
    x = am.array(src)
    assert x.child_count() == 2, f"child_count {x.child_count()} != 2"
    # struct_field applies the struct's own validity, as pc.struct_field does; child() is the raw
    # child array, which is pyarrow's StructArray.field().
    return ([arrow(x.struct_field("a")), arrow(x.struct_field("b")), arrow(x.child(0))],
            [pc.struct_field(src, "a"), pc.struct_field(src, "b"), src.field(0)])


@op("map_lookup", ["map"])
def _map_lookup(src, shape):
    x = am.array(src)
    got, expected = [], []
    for key in ("k", "", "absent"):
        for occurrence in ("first", "last", "all"):
            got.append(arrow(x.map_lookup(key, occurrence)))
            expected.append(pc.map_lookup(src, query_key=key, occurrence=occurrence))
    return got, expected


@op("binary_join", ["list_utf8"])
def _binary_join(src, shape):
    x = am.array(src)
    separators = make_array("utf8", shape, seed=3)
    return ([arrow(x.binary_join("-")), arrow(x.binary_join("")),
             arrow(x.binary_join(am.array(separators)))],
            [pc.binary_join(src, "-"), pc.binary_join(src, ""), pc.binary_join(src, separators)])


# ---- regular expressions, LIKE and splitting -------------------------
#
# ArrowMetal matches with ICU (NSRegularExpression) and pyarrow with RE2. The two agree on the syntax
# the matrix uses except for one thing: ICU's `\d`, `\w` and `\s` are Unicode-aware and RE2's are ASCII.
# `\d` is in the pattern list on purpose, and the divergence it produces is finding
# `regex-icu-unicode-classes`, classified by the data so a column of plain ASCII still has to agree.

#: Patterns exercised by every regex op: literals, anchors, alternation, classes, a quantifier, a
#: multi-byte literal and one that matches nothing.
_REGEX_PATTERNS = ["a", "app", "l+", "^a", "[0-9]", "a|b", "a.p", r"\d", "é", "^$", "zzz", "(a)(p)"]

#: Patterns whose meaning is the same in ICU and RE2, for the ops where a template or a split makes the
#: Unicode classes irrelevant to what is being tested.
_ASCII_REGEX_PATTERNS = [p for p in _REGEX_PATTERNS if "\\" not in p]


@op("regex_match", ["utf8"], note="pc.match_substring_regex / count / find, both engines unanchored")
def _regex_match(src, shape):
    x = am.array(src)
    got, expected = [], []
    for pattern in _REGEX_PATTERNS:
        got += [arrow(x.match_substring_regex(pattern)),
                arrow(x.count_substring_regex(pattern)),
                arrow(x.find_substring_regex(pattern))]
        expected += [pc.match_substring_regex(src, pattern),
                     pc.count_substring_regex(src, pattern).cast(pa.int32()),
                     pc.find_substring_regex(src, pattern).cast(pa.int32())]
    for pattern in ("APP", "É"):
        got.append(arrow(x.match_substring_regex(pattern, True)))
        expected.append(pc.match_substring_regex(src, pattern, ignore_case=True))
    return got, expected


@op("regex_replace", ["utf8"],
    note="ArrowMetal's replacement is an ICU template ($1); pyarrow's is an RE2 one (\\1)")
def _regex_replace(src, shape):
    x = am.array(src)
    got, expected = [], []
    for pattern, ours, theirs in [("l+", "L", "L"), ("(a)(p)", "$2$1", r"\2\1"),
                                  ("[0-9]", "#", "#"), ("zzz", "!", "!"), (r"\d", "D", "D")]:
        got.append(arrow(x.replace_substring_regex(pattern, ours)))
        expected.append(pc.replace_substring_regex(src, pattern, theirs))
    return got, expected


def _split_result(r):
    """A split result whatever its shape: one list MetalArray, or the older (offsets, values) pair."""
    return _as_list_array(*r) if isinstance(r, tuple) else _as_list_array(r)


def _as_list_array(*parts):
    """A split result as the list<utf8> Arrow produces: either the list array ArrowMetal now returns
    (one MetalArray of format "+l") or the older (offsets, values) pair."""
    if len(parts) == 1:
        return arrow(parts[0])
    offsets, values = parts
    return pa.ListArray.from_arrays(arrow(offsets), arrow(values))


@op("regex_split", ["utf8"], note="pc.split_pattern with its defaults: every occurrence, left to right")
def _regex_split(src, shape):
    x = am.array(src)
    got, expected = [], []
    for separator in (" ", "a", "\t", "pp"):
        got.append(_as_list_array(x.split_pattern(separator)))
        expected.append(pc.split_pattern(src, separator))
    for pattern in ("[0-9]", "l+", "a|b"):
        got.append(_as_list_array(x.split_pattern(pattern, regex=True)))
        expected.append(pc.split_pattern_regex(src, pattern))
    return got, expected


@op("split_whitespace", ["utf8"],
    note="unicode=True against pc.utf8_split_whitespace: a run of whitespace is one separator in both, "
         "except that a trailing run of two or more characters yields one empty piece here and two "
         "in Arrow's Unicode variant (finding split-whitespace-trailing-run)")
def _split_whitespace(src, shape):
    return _as_list_array(am.array(src).split_whitespace(unicode=True)), pc.utf8_split_whitespace(src)


@op("ascii_split_whitespace", ["utf8"],
    note="the default (ASCII) split against pc.ascii_split_whitespace, which gives one empty piece "
         "for a trailing run, as ArrowMetal does")
def _ascii_split_whitespace(src, shape):
    return _as_list_array(am.array(src).split_whitespace()), pc.ascii_split_whitespace(src)


@op("regex_extract", ["utf8"],
    note="ICU spells a named group (?<n>...) and RE2 (?P<n>...); the groups themselves are compared "
         "one at a time, since ArrowMetal returns a dict where Arrow returns a struct")
def _regex_extract(src, shape):
    x = am.array(src)
    got, expected = [], []
    for ours, theirs, names in [(r"(?<head>a)(?<tail>p+)", r"(?P<head>a)(?P<tail>p+)", ["head", "tail"]),
                                (r"(?<digits>[0-9]+)", r"(?P<digits>[0-9]+)", ["digits"]),
                                (r"(?<never>zzz)", r"(?P<never>zzz)", ["never"])]:
        columns = x.extract_regex(ours)
        struct = pc.extract_regex(src, theirs)
        for field in names:
            got.append(arrow(columns[field]))
            expected.append(pc.struct_field(struct, field))
    return got, expected


@op("regex_extract_span", ["utf8"], note="byte offsets and lengths, as pc.extract_regex_span's are")
def _regex_extract_span(src, shape):
    x = am.array(src)
    got, expected = [], []
    for ours, theirs, names in [(r"(?<head>a)(?<tail>p+)", r"(?P<head>a)(?P<tail>p+)", ["head", "tail"]),
                                (r"(?<digits>[0-9]+)", r"(?P<digits>[0-9]+)", ["digits"])]:
        spans = x.extract_regex_span(ours)
        struct = pc.extract_regex_span(src, theirs)
        for field in names:
            # Arrow returns fixed_size_list<int32>[2] per group: [start, length].
            pair = pc.struct_field(struct, field)
            flat = pc.list_flatten(pair)
            valid = pc.is_valid(pair)
            start, length = spans[field]
            got += [arrow(start), arrow(length)]
            expected += [pc.if_else(valid, _every_other(flat, 0, len(pair)), None),
                         pc.if_else(valid, _every_other(flat, 1, len(pair)), None)]
    return got, expected


def _every_other(flat, offset, rows):
    """Element `offset` of each 2-element group of a flattened fixed-size list, padded back to `rows`
    with nulls where the group was absent."""
    if len(flat) != 2 * rows:
        # Null rows drop out of list_flatten; rebuild positionally instead.
        raise Unsupported("extract_regex_span oracle needs every row to have matched")
    return flat.take(pa.array(list(range(offset, 2 * rows, 2)), pa.int32()))


@op("match_like", ["utf8"], note="pc.match_like: % is any run, _ exactly one, backslash escapes both")
def _match_like(src, shape):
    x = am.array(src)
    got, expected = [], []
    for pattern in ["a%", "%a", "%a%", "app", "_pp", "a_p%", "%", "_", "", "1\\%0", "%é%"]:
        got.append(arrow(x.match_like(pattern)))
        expected.append(pc.match_like(src, pattern))
    got.append(arrow(x.match_like("APP%", True)))
    expected.append(pc.match_like(src, "APP%", ignore_case=True))
    return got, expected


# ---- the rest of the Arrow string surface ----------------------------

_STRING_PREDICATES = [
    ("ascii_is_printable", pc.ascii_is_printable), ("ascii_is_title", pc.ascii_is_title),
    ("string_is_ascii", pc.string_is_ascii), ("utf8_is_alnum", pc.utf8_is_alnum),
    ("utf8_is_alpha", pc.utf8_is_alpha), ("utf8_is_decimal", pc.utf8_is_decimal),
    ("utf8_is_digit", pc.utf8_is_digit), ("utf8_is_lower", pc.utf8_is_lower),
    ("utf8_is_numeric", pc.utf8_is_numeric), ("utf8_is_printable", pc.utf8_is_printable),
    ("utf8_is_space", pc.utf8_is_space), ("utf8_is_title", pc.utf8_is_title),
    ("utf8_is_upper", pc.utf8_is_upper),
]


@op("string_predicates", ["utf8"], note="the thirteen character-class predicates, Unicode included")
def _string_predicates(src, shape):
    x = am.array(src)
    got, expected = [], []
    for name, oracle in _STRING_PREDICATES:
        got.append(arrow(getattr(x, name)()))
        expected.append(oracle(src))
    return got, expected


@op("string_case_transforms", ["utf8"])
def _string_case_transforms(src, shape):
    x = am.array(src)
    return ([arrow(x.ascii_title()), arrow(x.utf8_capitalize()), arrow(x.utf8_title())],
            [pc.ascii_title(src), pc.utf8_capitalize(src), pc.utf8_title(src)])


@op("string_pad_and_slice", ["utf8"])
def _string_pad_and_slice(src, shape):
    x = am.array(src)
    got, expected = [], []
    for width, pad in [(0, " "), (7, " "), (8, "*")]:
        got.append(arrow(x.utf8_center(width, pad)))
        expected.append(pc.utf8_center(src, width, padding=pad))
    for start, stop, replacement in [(1, 3, "--"), (0, 0, "^"), (2, 1, "!"), (-3, -1, "…"), (0, 100, "")]:
        got += [arrow(x.utf8_replace_slice(start, stop, replacement)),
                arrow(x.binary_replace_slice(start, stop, replacement))]
        expected += [pc.utf8_replace_slice(src, start, stop, replacement),
                     pc.binary_replace_slice(src.cast(pa.binary()), start, stop, replacement)]
    return got, expected


@op("string_trim", ["utf8"], note="the Unicode trims, with and without a character set")
def _string_trim(src, shape):
    x = am.array(src)
    got = [arrow(x.utf8_trim()), arrow(x.utf8_ltrim()), arrow(x.utf8_rtrim())]
    expected = [pc.utf8_trim_whitespace(src), pc.utf8_ltrim_whitespace(src), pc.utf8_rtrim_whitespace(src)]
    for characters in ("a", "ap", " \t", "aé", ""):
        got += [arrow(x.utf8_trim(characters)), arrow(x.utf8_ltrim(characters)),
                arrow(x.utf8_rtrim(characters))]
        expected += [pc.utf8_trim(src, characters), pc.utf8_ltrim(src, characters),
                     pc.utf8_rtrim(src, characters)]
    return got, expected


@op("string_normalize", ["utf8"],
    note=_no_oracle("string_normalize",
                    "pyarrow 25.0.1's utf8_normalize ignores its `form` option and always decomposes, "
                    "so the oracle is Python's own unicodedata "
                    "(test_pyarrow_utf8_normalize_ignores_its_form_option pins the bug)"))
def _string_normalize(src, shape):
    import unicodedata
    x = am.array(src)
    got, expected = [], []
    for form in ("NFC", "NFKC", "NFD", "NFKD"):
        got.append(arrow(x.utf8_normalize(form)))
        expected.append(pa.array([None if s is None else unicodedata.normalize(form, s)
                                  for s in src.to_pylist()], pa.string()))
    return got, expected


@op("string_set_lookup", ["utf8"],
    note="the GPU string hash table; skip_nulls=True, the mode ArrowMetal implements")
def _string_set_lookup(src, shape):
    values = src.slice(0, min(len(src), 5))
    x = am.array(src)
    return ([arrow(x.is_in(am.array(values))), arrow(x.index_in(am.array(values)))],
            [pc.is_in(src, value_set=values, skip_nulls=True),
             pc.index_in(src, value_set=values, skip_nulls=True).cast(pa.int32())])


# ---- utf8 <-> number casts -------------------------------------------

@op("to_strings", INTEGER + ["bool"], note="Arrow's cast(utf8), exact for integers and booleans")
def _to_strings(src, shape):
    return arrow(am.array(src).to_strings()), src.cast(pa.string())


@op("to_strings_text", FLOATING,
    note="the text itself: ArrowMetal formats a float the way Swift does, Arrow the way its own "
         "formatter does -- finding float-text-swift-format")
def _to_strings_text(src, shape):
    return arrow(am.array(src).to_strings()), src.cast(pa.string())


@op("to_strings_value", FLOATING,
    note="the number the text denotes, bit-exact: whatever the formatting, the digits have to name "
         "exactly the float that went in")
def _to_strings_value(src, shape):
    """Reads ArrowMetal's own text back with Python's `float`, which is correctly rounded, and compares
    the result with the input bit for bit. This is the property the formatting divergence must not
    touch: `-0.0` stays `-0.0`, a subnormal stays that subnormal, and no digit is lost."""
    name = type_name_of(src)
    text = arrow(am.array(src).to_strings()).to_pylist()
    parsed = [None if s is None else (math.nan if s == "nan" else float(s)) for s in text]
    return pa.array(parsed, type=src.type), src


@op("cast_to_string", NUMERIC + ["bool"],
    note="cast('string') has to be exactly to_strings(), on every type")
def _cast_to_string(src, shape):
    x = am.array(src)
    return arrow(x.cast("string")), arrow(x.to_strings())


@op("parse_numbers", NUMERIC + ["bool"],
    note="the values pyarrow itself formatted, parsed back by both engines")
def _parse_numbers(src, shape):
    """Round trip through Arrow's own text: `pc.cast(utf8)` produces the strings, then both engines
    parse them back. Anything ArrowMetal cannot parse it reports as null and Arrow raises on, which is
    why the strings come from Arrow rather than from the string generator
    (test_parse_returns_null_where_pyarrow_raises pins that half)."""
    name = type_name_of(src)
    text = src.cast(pa.string())
    if name in FLOATING:
        # Arrow prints a float in the shortest round-tripping form, which its own parser reads back
        # exactly; ArrowMetal's CPU parser is strtof/strtod, so the two agree on every finite value.
        pass
    return arrow(am.array(text).parse(name)), text.cast(ARROW_TYPE[name], safe=False)


# ---- temporal --------------------------------------------------------
#
# Every kernel is compared in UTC, which is what ArrowMetal extracts in; the two timezone functions
# (assume_timezone / local_timestamp) are the ones that consult the tz database, and they get the
# timezone columns.
#
# `pc.year_month_day` and `pc.iso_calendar` are deliberately *not* used as oracles: in pyarrow 25.0.1
# they corrupt the heap and crash the process a couple of calls later (see docs/FINDINGS.md). The two
# struct-valued kernels are compared field by field against the scalar extractors instead, which is a
# stronger check anyway -- it says the struct agrees with `pc.year`, not merely with another struct.

#: The calendar extractors, on every column that carries a date.
_CALENDAR_FIELDS = [("year", pc.year), ("month", pc.month), ("day", pc.day),
                    ("day_of_year", pc.day_of_year), ("quarter", pc.quarter),
                    ("iso_week", pc.iso_week), ("iso_year", pc.iso_year),
                    ("us_week", pc.us_week), ("us_year", pc.us_year),
                    ("is_leap_year", pc.is_leap_year)]

#: The clock extractors. date32 / date64 answer them here (all zero) where Arrow has no kernel at all,
#: so those two types get their own op below.
_CLOCK_FIELDS = [("hour", pc.hour), ("minute", pc.minute), ("second", pc.second),
                 ("millisecond", pc.millisecond), ("microsecond", pc.microsecond),
                 ("nanosecond", pc.nanosecond), ("subsecond", pc.subsecond)]


def _in_the_result_width(result, expected):
    """The oracle in the result's own integer width.

    ArrowMetal's temporal extractors return **int32** where pyarrow returns int64 -- every value a
    calendar or clock field can take fits in either, so casting the oracle back compares the numbers
    without hiding anything, and the width itself is the `temporal_field_types` operation's job
    (plus test_temporal_extractors_return_int32_where_pyarrow_returns_int64)."""
    return expected.cast(result.type) if expected.type != result.type else expected


@op("temporal_calendar", DATE_LIKE, note="year, month, day, quarter, day_of_year, the two week "
                                        "numbering pairs and is_leap_year, all in UTC")
def _temporal_calendar(src, shape):
    x = am.array(src)
    got, expected = [], []
    for name, oracle in _CALENDAR_FIELDS:
        result = arrow(getattr(x, name)())
        got.append(result)
        expected.append(_in_the_result_width(result, oracle(src)))
    return got, expected


@op("temporal_clock", TIMESTAMP_TYPES + TIME_TYPES)
def _temporal_clock(src, shape):
    x = am.array(src)
    got, expected = [], []
    for name, oracle in _CLOCK_FIELDS:
        result = arrow(getattr(x, name)())
        got.append(result)
        expected.append(_in_the_result_width(result, oracle(src)))
    return got, expected


#: The Arrow type each temporal extractor returns here, which is not always pyarrow's. Recorded so a
#: change of width is a failure rather than something the value comparison quietly absorbs.
_TEMPORAL_RESULT_TYPES = {
    "year": pa.int32(), "month": pa.int32(), "day": pa.int32(), "day_of_year": pa.int32(),
    "quarter": pa.int32(), "iso_week": pa.int32(), "iso_year": pa.int32(),
    "us_week": pa.int64(), "us_year": pa.int64(), "is_leap_year": pa.bool_(),
    "hour": pa.int32(), "minute": pa.int32(), "second": pa.int32(), "millisecond": pa.int32(),
    "microsecond": pa.int32(), "nanosecond": pa.int32(), "subsecond": pa.float64(),
    "day_of_week": pa.int32(), "week": pa.int64(),
}


@op("temporal_field_types", DATE_LIKE,
    note=_no_oracle("temporal_field_types",
                    "the result width is ArrowMetal's own (int32 where pyarrow returns int64), so the "
                    "reference is the recorded contract in _TEMPORAL_RESULT_TYPES"))
def _temporal_field_types(src, shape):
    x = am.array(src)
    got, expected = [], []
    for name, ty in sorted(_TEMPORAL_RESULT_TYPES.items()):
        got.append(str(arrow(getattr(x, name)()).type))
        expected.append(str(ty))
    return got, expected


@op("temporal_clock_on_a_date", DATE_TYPES,
    note=_no_oracle("temporal_clock_on_a_date",
                    "pyarrow has no hour/minute/second/subsecond kernel for date32 or date64; "
                    "ArrowMetal answers with the midnight the date names, which the reference asserts"))
def _temporal_clock_on_a_date(src, shape):
    """ArrowMetal defines the clock fields on a date column as the clock of the midnight it names, so
    every one of them is zero where the row is valid and null where it is not. That is the reference."""
    x = am.array(src)
    valid = pc.is_valid(src)
    got, expected = [], []
    for name, _ in _CLOCK_FIELDS:
        result = arrow(getattr(x, name)())
        got.append(result)
        zero = pa.scalar(0.0 if name == "subsecond" else 0, result.type)
        expected.append(pc.if_else(valid, zero, pa.scalar(None, result.type)))
    return got, expected


@op("temporal_week_options", DATE_LIKE, note="pc.week with every WeekOptions combination")
def _temporal_week_options(src, shape):
    x = am.array(src)
    got, expected = [], []
    for monday in (True, False):
        for from_zero in (True, False):
            for fully_in_year in (True, False):
                got.append(arrow(x.week(monday, from_zero, fully_in_year)))
                expected.append(pc.week(src, week_starts_monday=monday, count_from_zero=from_zero,
                                        first_week_is_fully_in_year=fully_in_year))
    for from_zero in (True, False):
        for week_start in (1, 3, 7):
            result = arrow(x.day_of_week(from_zero, week_start))
            got.append(result)
            expected.append(_in_the_result_width(
                result, pc.day_of_week(src, count_from_zero=from_zero, week_start=week_start)))
    return got, expected


@op("temporal_struct", DATE_LIKE,
    note="iso_calendar and year_month_day compared field by field against the scalar extractors "
         "(pc.iso_calendar / pc.year_month_day crash pyarrow 25.0.1)")
def _temporal_struct(src, shape):
    x = am.array(src)
    iso, ymd = x.iso_calendar(), x.year_month_day()
    got = [arrow(iso.struct_field("iso_year")), arrow(iso.struct_field("iso_week")),
           arrow(iso.struct_field("iso_day_of_week")),
           arrow(ymd.struct_field("year")), arrow(ymd.struct_field("month")),
           arrow(ymd.struct_field("day"))]
    expected = [pc.iso_year(src), pc.iso_week(src),
                pc.day_of_week(src, count_from_zero=False, week_start=1),
                pc.year(src), pc.month(src), pc.day(src)]
    return got, [e.cast(pa.int64()) if e.type != pa.int64() else e for e in expected]


@op("temporal_timezone", TIMESTAMP_TZ, note="is_dst and local_timestamp in the column's own zone")
def _temporal_timezone(src, shape):
    x = am.array(src)
    return ([arrow(x.is_dst()), arrow(x.local_timestamp())],
            [pc.is_dst(src), pc.local_timestamp(src)])


@op("assume_timezone", TIMESTAMP_NAIVE,
    note="pc.assume_timezone with ambiguous / nonexistent spelled out; 'raise' is not compared "
         "because the generator will hit a DST gap")
def _assume_timezone(src, shape):
    x = am.array(src)
    got, expected = [], []
    for zone in (TZ_NAME, "UTC", "Asia/Kolkata", "Australia/Lord_Howe"):
        for handling in ("earliest", "latest"):
            got.append(arrow(x.assume_timezone(zone, handling, handling)))
            expected.append(pc.assume_timezone(src, zone, ambiguous=handling, nonexistent=handling))
    return got, expected


# Rounding. Three corners once separated the two engines and each keeps its own operation so that a
# relapse shows up on its own row rather than inside the main rounding cell:
#
#   * a unit finer than the column's own resolution (`temporal_round_finer`): Arrow converts to the
#     finer unit, rounds there and truncates back, and so does ArrowMetal now;
#   * `ceil` of a value already sitting on a *calendar* boundary (`temporal_ceil_calendar`): Arrow
#     advances a whole month/quarter/year (but keeps a value on a fixed-length boundary), and so does
#     ArrowMetal now;
#   * a multiple of months or quarters that does not divide the epoch's own offset
#     (`temporal_round_unaligned`): both count from 1970-01 now (and years from year 0, as Arrow does).
# All three were findings in the first runs (see docs/EVALUATION.md, "Findings that were fixed").

_FIXED_ROUND_UNITS = ["nanosecond", "microsecond", "millisecond", "second", "minute", "hour", "day"]
_CALENDAR_ROUND_UNITS = ["month", "quarter", "year"]
#: Ticks in one of the fixed-length units, so the matrix can tell which are finer than a column.
_UNIT_NANOSECONDS = {"nanosecond": 1, "microsecond": 10 ** 3, "millisecond": 10 ** 6,
                     "second": 10 ** 9, "minute": 60 * 10 ** 9, "hour": 3600 * 10 ** 9,
                     "day": 86400 * 10 ** 9}
#: Multiples of a calendar unit that land on the same grid whichever origin is used: `month` needs the
#: multiple to divide 1970*12 and `quarter` to divide 1970*4.
_ALIGNED_MULTIPLES = {"month": [1, 2, 3, 4, 5, 6], "quarter": [1, 2, 4, 5], "year": [1, 2, 3, 4, 7]}


def _column_nanoseconds(src):
    """Nanoseconds in one tick of the column, or None when it carries no time at all."""
    if pa.types.is_date32(src.type):
        return 86400 * 10 ** 9
    if pa.types.is_date64(src.type):
        return 10 ** 6
    return {"s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3, "ns": 1}[src.type.unit]


def _round_pairs(src, kind):
    """(unit, multiple) pairs for one rounding operation.

    `kind` "fixed" keeps the units the column can actually represent, "finer" the ones below its
    resolution at a multiple that does not divide a whole tick, and "calendar" the month / quarter /
    year units at multiples both engines put on the same grid."""
    tick = _column_nanoseconds(src)
    if kind == "fixed":
        return [(u, m) for u in _FIXED_ROUND_UNITS for m in (1, 2, 3, 7)
                if _UNIT_NANOSECONDS[u] >= tick]
    if kind == "finer":
        return [(u, m) for u in _FIXED_ROUND_UNITS for m in (3, 7)
                if _UNIT_NANOSECONDS[u] < tick]
    return [(u, m) for u in _CALENDAR_ROUND_UNITS for m in _ALIGNED_MULTIPLES[u]]


def _rounding_case(src, kinds, modes):
    x = am.array(src)
    got, expected = [], []
    for kind in kinds:
        for unit, multiple in _round_pairs(src, kind):
            for mode, oracle in modes:
                got.append(arrow(getattr(x, mode + "_temporal")(unit, multiple)))
                expected.append(oracle(src, multiple=multiple, unit=unit))
    if not got:
        raise Unsupported(f"no {'/'.join(kinds)} rounding units apply to {src.type}")
    return got, expected


_ROUND_MODES = [("floor", pc.floor_temporal), ("ceil", pc.ceil_temporal),
                ("round", pc.round_temporal)]
_FLOOR_AND_ROUND = [m for m in _ROUND_MODES if m[0] != "ceil"]


@op("temporal_round", TIMESTAMP_TYPES + DATE_TYPES + TIME_TYPES,
    note="floor/ceil/round to every unit the column can represent, at multiples 1, 2, 3 and 7")
def _temporal_round(src, shape):
    return _rounding_case(src, ["fixed"], _ROUND_MODES)


@op("temporal_round_finer", TIMESTAMP_TYPES + DATE_TYPES + TIME_TYPES,
    note="a unit below the column's own resolution: ArrowMetal is the identity, Arrow converts")
def _temporal_round_finer(src, shape):
    return _rounding_case(src, ["finer"], _ROUND_MODES)


@op("temporal_round_calendar", DATE_LIKE,
    note="month, quarter and year, at the multiples both engines put on the same grid")
def _temporal_round_calendar(src, shape):
    return _rounding_case(src, ["calendar"], _FLOOR_AND_ROUND)


@op("temporal_ceil_calendar", DATE_LIKE,
    note="ceil of a value already on a month, quarter or year boundary: kept here, advanced in Arrow")
def _temporal_ceil_calendar(src, shape):
    return _rounding_case(src, ["calendar"], [m for m in _ROUND_MODES if m[0] == "ceil"])


@op("temporal_round_unaligned", DATE_LIKE,
    note="month x7 and quarter x3: the two origins part company here")
def _temporal_round_unaligned(src, shape):
    x = am.array(src)
    got, expected = [], []
    for unit, multiple in [("month", 7), ("month", 11), ("quarter", 3), ("quarter", 7)]:
        for mode, oracle in _ROUND_MODES:
            got.append(arrow(getattr(x, mode + "_temporal")(unit, multiple)))
            expected.append(oracle(src, multiple=multiple, unit=unit))
    return got, expected


@op("temporal_round_duration", DURATION_TYPES,
    note=_no_oracle("temporal_round_duration",
                    "pyarrow's floor/ceil/round_temporal have no duration kernel; the reference is "
                    "integer arithmetic on the tick count, which is what the kernel documents"))
def _temporal_round_duration(src, shape):
    """A duration has no calendar, so rounding it is plain integer arithmetic on its ticks. The
    reference is that arithmetic, done in Python: floor divides toward minus infinity, ceil leaves a
    value already on a boundary alone, and a half rounds toward plus infinity."""
    x = am.array(src)
    tick = _column_nanoseconds(src)
    got, expected = [], []
    values = _raw_ticks(src).to_pylist()
    for unit in _FIXED_ROUND_UNITS:
        if _UNIT_NANOSECONDS[unit] < tick:
            continue
        for multiple in (1, 3):
            step = (_UNIT_NANOSECONDS[unit] // tick) * multiple
            for mode in ("floor", "ceil", "round"):
                got.append(arrow(getattr(x, mode + "_temporal")(unit, multiple)))
                expected.append(pa.array([None if v is None else _round_ticks(v, step, mode)
                                          for v in values], type=pa.int64()).cast(src.type,
                                                                                  safe=False))
    return got, expected


def _raw_ticks(src):
    """The int64 tick values under a temporal column. `to_pylist` would hand back datetimes, which
    cannot hold a nanosecond; Arrow's own cast is exact. The 32-bit temporal types have to go through
    int32 first -- Arrow has no direct cast from time32 or date32 to int64."""
    if pa.types.is_time32(src.type) or pa.types.is_date32(src.type):
        return src.cast(pa.int32(), safe=False).cast(pa.int64())
    return src.cast(pa.int64(), safe=False)


def _round_ticks(ticks, step, mode):
    low = (ticks // step) * step
    if mode == "floor":
        return low
    high = low + step
    if mode == "ceil":
        return low if ticks == low else high
    return high if 2 * (ticks - low) >= step else low


# The differences. Every *_between counts boundaries crossed from self to other, both sides truncated
# to the unit first, so it is not the truncated difference; pyarrow's kernels take two arguments of the
# same type, which is what the matrix pairs.

_BETWEEN_CALENDAR = [("years_between", pc.years_between), ("quarters_between", pc.quarters_between),
                     ("days_between", pc.days_between)]
_BETWEEN_CLOCK = [("hours_between", pc.hours_between), ("minutes_between", pc.minutes_between),
                  ("seconds_between", pc.seconds_between),
                  ("milliseconds_between", pc.milliseconds_between),
                  ("microseconds_between", pc.microseconds_between),
                  ("nanoseconds_between", pc.nanoseconds_between)]


@op("temporal_between", DATE_LIKE)
def _temporal_between(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    x, y = am.array(src), am.array(other)
    got, expected = [], []
    for name, oracle in _BETWEEN_CALENDAR + _BETWEEN_CLOCK:
        got.append(arrow(getattr(x, name)(y)))
        expected.append(oracle(src, other))
    return got, expected


@op("temporal_between_clock", TIME_TYPES)
def _temporal_between_clock(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    x, y = am.array(src), am.array(other)
    got, expected = [], []
    for name, oracle in _BETWEEN_CLOCK:
        got.append(arrow(getattr(x, name)(y)))
        expected.append(oracle(src, other))
    return got, expected


@op("weeks_between", DATE_LIKE, note="pc.weeks_between with week_start spelled out, 1 = Monday")
def _weeks_between(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    x, y = am.array(src), am.array(other)
    got, expected = [], []
    for week_start in (1, 2, 3, 4, 5, 6, 7):
        for from_zero in (True, False):
            got.append(arrow(x.weeks_between(y, count_from_zero=from_zero, week_start=week_start)))
            expected.append(pc.weeks_between(src, other, count_from_zero=from_zero,
                                             week_start=week_start))
    return got, expected


@op("months_between", DATE_LIKE,
    note="pyarrow spells the same quantity month_interval_between and returns an interval; the oracle "
         "is the difference of year * 12 + month, built from pc.year and pc.month")
def _months_between(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)

    def index(a):
        return pc.add(pc.multiply(pc.year(a).cast(pa.int64()), 12), pc.month(a).cast(pa.int64()))

    return (arrow(am.array(src).months_between(am.array(other))),
            pc.subtract(index(other), index(src)))


@op("interval_between", DATE_LIKE, note="pc.month_day_nano_interval_between, field for field")
def _interval_between(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    x, y = am.array(src), am.array(other)
    return (arrow(x.month_day_nano_interval_between(y)),
            pc.month_day_nano_interval_between(src, other))


@op("interval_layouts", DATE_LIKE,
    note=_no_oracle("interval_layouts",
                    "pyarrow 25 has no Python type for interval[month] or interval[day_time], so "
                    "neither engine's result can be wrapped; the reference is the month_day_nano "
                    "result, which pyarrow does compute and which the matrix compares against it"))
def _interval_layouts(src, shape):
    """The two narrow interval layouts are read through `interval_field`, and checked against the
    month_day_nano interval the previous operation already compares with Arrow: the month layout must
    carry the same months, and the day/time layout the same days and the same sub-day part truncated
    to milliseconds."""
    other = make_array(type_name_of(src), shape, seed=1)
    x, y = am.array(src), am.array(other)
    mdn = x.month_day_nano_interval_between(y)
    months = x.month_interval_between(y)
    day_time = x.day_time_interval_between(y)
    milli = pa.scalar(10 ** 6, pa.int64())
    reference_nanos = pc.multiply(pc.divide(arrow(mdn.interval_field("nanoseconds")), milli), milli)
    return ([arrow(months.interval_field("months")),
             arrow(day_time.interval_field("days")),
             arrow(day_time.interval_field("nanoseconds"))],
            [arrow(mdn.interval_field("months")),
             # The day/time layout carries whole days, not the calendar remainder the month/day/nano
             # one does; interval_field returns int32, pc.days_between int64.
             pc.days_between(src, other).cast(pa.int32()),
             reference_nanos])


@op("add_duration", TIMESTAMP_TYPES + TIME_TYPES + DURATION_TYPES,
    note="a duration column of the same unit, and a scalar tick count; a time-of-day column keeps "
         "only the rows whose result stays inside the day, which is all pyarrow will answer")
def _add_duration(src, shape):
    unit = src.type.unit
    delta = make_array("duration_" + unit, shape, seed=4)
    if pa.types.is_time32(src.type) or pa.types.is_time64(src.type):
        # pc.add refuses a time-of-day outside [0, 86400) rather than wrapping, so the pairs that
        # would leave the day are dropped (test_time_of_day_addition_wraps_where_pyarrow_raises).
        span = 86400 * TICKS[unit]
        ticks = _raw_ticks(src)

        def inside(total):
            return pc.and_(pc.greater_equal(total, 0), pc.less(total, span))

        # The null rows go too: pc.add validates the values *under* the validity bitmap, which the
        # generator deliberately fills with real numbers, so a null row whose hidden value would
        # leave the day makes the oracle raise. Null propagation is covered by every other type here.
        keep = pc.fill_null(pc.and_(inside(pc.add(ticks, _raw_ticks(delta))),
                                    inside(pc.add(ticks, pa.scalar(1000, pa.int64())))), False)
        src, delta = src.filter(keep), delta.filter(keep)
    x = am.array(src)
    got = [arrow(x.add_duration(am.array(delta))), arrow(x.add_duration(1000))]
    expected = [pc.add(src, delta), pc.add(src, pa.scalar(1000, pa.duration(unit)))]
    return got, expected


@op("subtract_temporal", TIMESTAMP_TYPES + DATE_TYPES + TIME_TYPES + DURATION_TYPES,
    note="pc.subtract of two temporal columns, which yields a duration in the finer resolution")
def _subtract_temporal(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    return arrow(am.array(src).subtract_temporal(am.array(other))), pc.subtract(src, other)


@op("add_interval", TIMESTAMP_TYPES,
    note=_no_oracle("add_interval",
                    "pyarrow 25 has no add(timestamp, month_day_nano_interval) kernel; the days and "
                    "sub-day part are cross-checked against pc.add of the equivalent duration, and "
                    "the month part against a Python reference that clamps the day like Arrow does"))
def _add_interval(src, shape):
    """The interval column is split in two: the day and nanosecond fields are a plain duration, which
    Arrow can add, and the month field is calendar arithmetic, whose reference is written out here --
    the day is clamped to the length of the target month, which is what Arrow's own month addition
    documents."""
    interval = make_array("mdn_interval", shape, seed=5)
    tick = _column_nanoseconds(src)
    months = arrow(am.array(interval).interval_field("months"))
    days = arrow(am.array(interval).interval_field("days"))
    nanos = arrow(am.array(interval).interval_field("nanoseconds"))

    # The day / nanosecond half, as a duration Arrow can add.
    ticks = pc.add(pc.multiply(days.cast(pa.int64()), 86400 * (10 ** 9 // tick)),
                   pc.divide(nanos, pa.scalar(tick, pa.int64())))
    only_time = pc.if_else(pc.equal(months, 0), pa.scalar(True), pa.scalar(False))
    time_only_rows = pc.fill_null(only_time, False)
    src_time, ticks_time = src.filter(time_only_rows), ticks.filter(time_only_rows)
    interval_time = interval.filter(time_only_rows)

    got = [arrow(am.array(src_time).add_interval(am.array(interval_time)))]
    expected = [pc.add(src_time, ticks_time.cast(pa.duration(src.type.unit)))]

    # The whole column, against the reference.
    got.append(arrow(am.array(src).add_interval(am.array(interval))))
    expected.append(_add_interval_reference(src, months, days, nanos, tick))
    return got, expected


def _add_interval_reference(src, months, days, nanos, tick):
    """`timestamp + month_day_nano_interval`, in Python: the months move the calendar month and clamp
    the day, then the days and nanoseconds are added as elapsed time."""
    ticks_per_second = 10 ** 9 // tick
    out = []
    for ticks, month, day, nano in zip(_raw_ticks(src).to_pylist(), months.to_pylist(),
                                       days.to_pylist(), nanos.to_pylist()):
        if ticks is None or month is None:
            out.append(None)
            continue
        whole_days, rest = divmod(ticks, 86400 * ticks_per_second)
        y, m, d = _civil_from_days(whole_days)
        total = (y * 12 + (m - 1)) + month
        ny, nm = divmod(total, 12)
        nd = min(d, _days_in_month(ny, nm + 1))
        shifted = _days_from_civil(ny, nm + 1, nd) * 86400 * ticks_per_second + rest
        # The sub-day part truncates toward zero when the column is coarser than the interval, which
        # is not Python's floor division for a negative interval.
        sub = abs(nano) // tick * (1 if nano >= 0 else -1)
        out.append(shifted + day * 86400 * ticks_per_second + sub)
    return pa.array(out, type=pa.int64()).cast(src.type, safe=False)


def _civil_from_days(days):
    days += 719468
    era = (days if days >= 0 else days - 146096) // 146097
    doe = days - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return (y + (1 if m <= 2 else 0), m, d)


def _days_from_civil(y, m, d):
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _days_in_month(y, m):
    if m == 2:
        leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
        return 29 if leap else 28
    return 30 if m in (4, 6, 9, 11) else 31


@op("strftime", TIMESTAMP_TYPES + DATE_TYPES,
    note="pc.strftime in UTC (ArrowMetal formats the instant, never the column's local time), with "
         "the format spelled out and the locale left at C")
def _strftime(src, shape):
    x = am.array(src)
    # ArrowMetal formats in UTC; pc.strftime formats a zoned column in its own zone, so the oracle
    # sees the same instants written without a zone. The zone difference itself is finding
    # `temporal-extract-in-utc`, which `temporal_utc_semantics` covers.
    naive = _naive_timestamp(src)
    got, expected = [], []
    for fmt in ("%Y-%m-%d", "%H:%M", "%j", "%Y", "%Y/%m/%d %H:%M", "%m-%d"):
        got.append(arrow(x.strftime(fmt)))
        expected.append(pc.strftime(naive, format=fmt))
    return got, expected


def _naive_timestamp(src):
    if pa.types.is_date32(src.type) or pa.types.is_date64(src.type):
        return src.cast(pa.timestamp("s"))
    return src.cast(pa.timestamp(src.type.unit))


@op("strftime_seconds", TIMESTAMP_TYPES + DATE_TYPES,
    note="a format with %S: Arrow prints the sub-second digits with the seconds, C's strftime (and "
         "so ArrowMetal's) does not")
def _strftime_seconds(src, shape):
    x = am.array(src)
    naive = _naive_timestamp(src)
    got, expected = [], []
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%S", "%H:%M:%S"):
        got.append(arrow(x.strftime(fmt)))
        expected.append(pc.strftime(naive, format=fmt))
    return got, expected


@op("strptime", ["utf8"], note="pc.strptime with error_is_null=True, the mode ArrowMetal implements")
def _strptime(src, shape):
    """The strings come from Arrow's own `strftime` of a generated timestamp column, so the format is
    guaranteed to be the one being parsed; the generator's own strings go in too, and both engines have
    to agree that they do not parse."""
    text = src
    return (arrow(am.array(text).strptime("%Y-%m-%d %H:%M:%S")),
            pc.strptime(text, format="%Y-%m-%d %H:%M:%S", unit="us", error_is_null=True))


@op("strptime_roundtrip", TIMESTAMP_NAIVE,
    note="Arrow formats the column, then both engines parse the text back")
def _strptime_roundtrip(src, shape):
    text = pc.strftime(src, format="%Y-%m-%d %H:%M:%S")
    return (arrow(am.array(text).strptime("%Y-%m-%d %H:%M:%S")),
            pc.strptime(text, format="%Y-%m-%d %H:%M:%S", unit="us", error_is_null=True))


@op("cast_unit", TIMESTAMP_TYPES + TIME_TYPES + DURATION_TYPES,
    note="Arrow's cast between resolutions, safe=False: scaling down truncates")
def _cast_unit(src, shape):
    x = am.array(src)
    got, expected = [], []
    for unit in TIME_UNITS:
        target = _rescaled_type(src.type, unit)
        if target is None:
            continue
        got.append(arrow(x.cast_unit(unit)))
        expected.append(src.cast(target, safe=False))
    return got, expected


def _rescaled_type(ty, unit):
    if pa.types.is_timestamp(ty):
        return pa.timestamp(unit, ty.tz)
    if pa.types.is_duration(ty):
        return pa.duration(unit)
    return pa.time32(unit) if unit in ("s", "ms") else pa.time64(unit)


@op("temporal_utc_semantics", TIMESTAMP_TZ,
    note="ArrowMetal extracts in UTC whatever the column's zone: the oracle is Arrow applied to the "
         "same instants with the zone stripped, which must agree exactly")
def _temporal_utc_semantics(src, shape):
    """The companion to finding `temporal-extract-in-utc`. Arrow reads a zoned timestamp in local
    time and ArrowMetal in UTC; this case pins that ArrowMetal's answer is exactly Arrow's answer for
    the same instants written as a naive column, so the divergence is the zone and nothing else."""
    naive = src.cast(pa.timestamp(src.type.unit))
    x = am.array(src)
    got, expected = [], []
    for name, oracle in _CALENDAR_FIELDS + _CLOCK_FIELDS:
        result = arrow(getattr(x, name)())
        got.append(result)
        expected.append(_in_the_result_width(result, oracle(naive)))
    for unit, multiple in _round_pairs(src, "fixed"):
        got.append(arrow(x.floor_temporal(unit, multiple)))
        expected.append(pc.floor_temporal(naive, multiple=multiple, unit=unit).cast(src.type))
    return got, expected


# ---- window, ranking, shifts and rolling windows ---------------------
#
# pyarrow has `rank` with a tiebreaker, which covers row_number ("first"), rank ("min") and dense_rank
# ("dense"). It has nothing for percent_rank, cume_dist, shift, cumulative_mean or the rolling windows,
# so those are checked against references written here and named in _NO_ORACLE.

@op("ranking", NUMERIC, note="pc.rank with null_placement='at_end' and the matching tiebreaker")
def _ranking(src, shape):
    x = am.array(src)
    got, expected = [], []
    for method, tiebreaker in [("row_number", "first"), ("rank", "min"), ("dense_rank", "dense")]:
        got.append(arrow(getattr(x, method)()))
        # null_placement defaults to "at_end", which is where ArrowMetal's ranks put them too.
        expected.append(pc.rank(src, sort_keys="ascending", tiebreaker=tiebreaker).cast(pa.int32()))
    return got, expected


def _sorted_ranks(src):
    """(min-rank, count-at-or-below, n) per row, ascending with nulls last as one tie group -- the
    order every ranking function here uses. Written out so percent_rank and cume_dist, which pyarrow
    does not have, are compared against something other than themselves."""
    values = src.to_pylist()
    n = len(values)
    order = sorted(range(n), key=lambda i: (values[i] is None, ) if values[i] is None else
                   (False, _sort_key(values[i])))
    ranks, cume = [0] * n, [0] * n
    position = 0
    while position < n:
        end = position + 1
        while end < n and _same_rank(values[order[position]], values[order[end]]):
            end += 1
        for k in range(position, end):
            ranks[order[k]] = position + 1
            cume[order[k]] = end
        position = end
    return ranks, cume, n


def _sort_key(value):
    """Arrow's total order for a value inside one column: NaN after every number."""
    if isinstance(value, float) and math.isnan(value):
        return (1, 0.0)
    return (0, value)


def _same_rank(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


@op("percent_rank_and_cume_dist", NUMERIC,
    note=_no_oracle("percent_rank_and_cume_dist",
                    "pyarrow has no percent_rank or cume_dist; the reference is the SQL definition, "
                    "computed here from the same ascending-nulls-last order pc.rank uses"))
def _percent_rank_and_cume_dist(src, shape):
    x = am.array(src)
    ranks, cume, n = _sorted_ranks(src)
    percent = [0.0 if n <= 1 else (r - 1) / (n - 1) for r in ranks]
    distribution = [c / n for c in cume] if n else []
    return ([arrow(x.percent_rank()), arrow(x.cume_dist())],
            [pa.array(percent, pa.float64()), pa.array(distribution, pa.float64())])


@op("shift", NUMERIC,
    note=_no_oracle("shift", "pyarrow has no lag/lead; the reference is the shifted Python list"))
def _shift(src, shape):
    x = am.array(src)
    values = src.to_pylist()
    n = len(values)
    got, expected = [], []
    for by, fill in [(1, None), (2, None), (-1, None), (-3, None), (1, scalar_for(type_name_of(src))),
                     (0, None)]:
        got.append(arrow(x.shift(by) if fill is None else x.shift(by, fill)))
        shifted = [values[i - by] if 0 <= i - by < n else fill for i in range(n)]
        expected.append(pa.array(shifted, type=src.type))
    return got, expected


def _materialised(src):
    """The same values with the Arrow offset folded away.

    `pc.pairwise_diff` reads the values buffer without honouring `ArrowArray.offset` in pyarrow 25.0.1,
    so on a sliced column it answers with the wrong rows entirely; the oracle is given a copy while
    ArrowMetal still gets the slice, which is what makes the case a test of the offset handling rather
    than of pyarrow's bug (test_pyarrow_pairwise_diff_ignores_the_array_offset pins it)."""
    if src.offset == 0:
        return src
    return pa.concat_arrays([src.slice(0, 0), src])


@op("pairwise_diff", NUMERIC, note="pc.pairwise_diff, the unchecked form: it wraps as ArrowMetal's does")
def _pairwise_diff(src, shape):
    x = am.array(src)
    flat = _materialised(src)
    got, expected = [], []
    for period in (1, 2, 5):
        got.append(arrow(x.pairwise_diff(period)))
        expected.append(pc.pairwise_diff(flat, period=period))
    return got, expected


def _within_exact_double(src):
    """A keep-mask that stops the exact running total of an integer column ever leaving 2^53."""
    running, keep = 0, []
    for value in src.to_pylist():
        if value is None:
            keep.append(True)
            continue
        step = running + value
        if abs(step) > 2 ** 53:
            keep.append(False)
        else:
            running = step
            keep.append(True)
    return keep


@op("cumulative_mean", NUMERIC,
    note=_no_oracle("cumulative_mean",
                    "pyarrow has no cumulative_mean; the reference is the exact running sum over the "
                    "running count, on the values a double can hold exactly"))
def _cumulative_mean(src, shape):
    """The running mean of the non-null values so far.

    The running total is accumulated in `double` by a parallel scan, so the reference is the exact
    running sum over the running count -- and the input is trimmed to the range where that is a
    meaningful comparison. A float column loses the values that can overflow a partial sum, exactly as
    `cumulative_sum` does; an integer column loses the values that would take the running total past
    2^53, where a double can no longer hold it and the scan's association starts to decide the answer
    (test_cumulative_mean_accumulates_in_double pins that)."""
    name = type_name_of(src)
    if name in FLOATING:
        filled = pc.fill_null(src, pa.scalar(0.0, src.type))
        limit = pa.scalar(float(np.finfo(NUMPY_TYPE[name]).max) / max(len(src), 1), src.type)
        src = src.filter(pc.or_(pc.is_nan(filled), pc.less_equal(pc.abs(filled), limit)))
    else:
        src = src.filter(pa.array(_within_exact_double(src), pa.bool_()))
    x = am.array(src)
    running, count, reference = 0, 0, []
    for value in src.to_pylist():
        if value is None:
            reference.append(None)
            continue
        count += 1
        running += value
        reference.append(running / count)
    tolerance = (FLOAT_TOL.get(name, FLOAT_TOL["float64"]),
                 float_sum_bound(_drop_nan(src), name) if name in FLOATING else 0.0)
    return arrow(x.cumulative_mean()), pa.array(reference, pa.float64()), tolerance


_ROLLING_WINDOWS = [(1, None), (3, None), (3, 1), (7, 2), (64, None)]


@op("rolling_min_max", NUMERIC,
    note=_no_oracle("rolling_min_max",
                    "pyarrow has no rolling window; the reference is each window taken as an Arrow "
                    "slice and reduced with pc.min / pc.max, so only the windowing is this file's"))
def _rolling_min_max(src, shape):
    if len(src) > 4000:
        raise Unsupported("the rolling reference is O(n * window); the smaller shapes cover it")
    # NaN is dropped for the same reason the scalar min/max drop it: ArrowMetal treats it as missing
    # all the way and answers ±inf for a window that holds nothing else, where pc.min answers NaN
    # (test_rolling_min_of_an_all_nan_window pins the difference).
    src = _drop_nan(src)
    x = am.array(src)
    got, expected = [], []
    for window, min_periods in _ROLLING_WINDOWS:
        for method, reduce in [("rolling_min", pc.min), ("rolling_max", pc.max)]:
            got.append(arrow(getattr(x, method)(window, min_periods)))
            expected.append(_rolling_reference(src, window, min_periods, reduce, method))
    return got, expected


def _finite_only(src, name):
    """A float column with the values that can make a partial sum overflow taken out: ±inf, NaN, and
    anything above `type_max / n`. The rolling sum is a difference of prefix sums, so one infinity
    poisons every later window through `inf - inf`, which no roundoff bound can express."""
    filled = pc.fill_null(src, pa.scalar(0.0, src.type))
    limit = pa.scalar(float(np.finfo(NUMPY_TYPE[name]).max) / max(len(src), 1), src.type)
    return src.filter(pc.and_(pc.invert(pc.is_nan(filled)), pc.less_equal(pc.abs(filled), limit)))


@op("rolling_sum", NUMERIC,
    note=_no_oracle("rolling_sum",
                    "pyarrow has no rolling window; the reference reduces each window with pc.sum. "
                    "Integers are exact and wrap in the column's own width; floats get the "
                    "reductions' roundoff bound, because the kernel is a difference of prefix sums"))
def _rolling_sum(src, shape):
    """The rolling sum is a difference of prefix sums, which is what makes it O(n) rather than
    O(n·window). The error of one window is therefore bounded by the error of the two prefix sums,
    which is the reductions' own `8·eps·sqrt(n)·Σ|x|`; the values that would make a partial sum
    overflow come out first (test_rolling_sum_is_a_prefix_sum_difference pins what happens with
    them in)."""
    name = type_name_of(src)
    if len(src) > 4000:
        raise Unsupported("the rolling reference is O(n * window); the smaller shapes cover it")
    if name in FLOATING:
        src = _finite_only(src, name)
    x = am.array(src)
    got, expected = [], []
    for window, min_periods in _ROLLING_WINDOWS:
        got.append(arrow(x.rolling_sum(window, min_periods)))
        expected.append(_rolling_reference(src, window, min_periods, pc.sum, "rolling_sum"))
    tolerance = (FLOAT_TOL[name], float_sum_bound(src, name)) if name in FLOATING else None
    return got, expected, tolerance


@op("rolling_mean", NUMERIC,
    note=_no_oracle("rolling_mean",
                    "pyarrow has no rolling window; the reference reduces each window with pc.mean, "
                    "over the values whose prefix sums a double can hold exactly"))
def _rolling_mean(src, shape):
    """Like the rolling sum, but the prefix sums are kept in `double` whatever the column type, so an
    integer column is trimmed to the values whose running total stays inside 2^53 -- past that the
    difference of two prefix sums cannot recover the window
    (test_rolling_mean_accumulates_in_double pins it)."""
    name = type_name_of(src)
    if len(src) > 4000:
        raise Unsupported("the rolling reference is O(n * window); the smaller shapes cover it")
    src = _finite_only(src, name) if name in FLOATING else \
        src.filter(pa.array(_within_exact_double(src), pa.bool_()))
    x = am.array(src)
    got, expected = [], []
    for window, min_periods in _ROLLING_WINDOWS:
        got.append(arrow(x.rolling_mean(window, min_periods)))
        expected.append(_rolling_reference(src, window, min_periods, pc.mean, "rolling_mean"))
    bound = float_sum_bound(src, name) if name in FLOATING else 0.0
    return got, expected, (FLOAT_TOL.get(name, FLOAT_TOL["float64"]), bound)


def _rolling_reference(src, window, min_periods, reduce, method):
    """The trailing window ending at each row, reduced by pyarrow. `min_periods` defaults to the whole
    window; a shorter window at the start of the column is answered as soon as it holds that many
    non-null rows, which is what the kernel documents."""
    needed = window if min_periods is None else min_periods
    out = []
    for i in range(len(src)):
        start = max(0, i + 1 - window)
        chunk = src.slice(start, i + 1 - start)
        if len(chunk) - chunk.null_count < needed:
            out.append(None)
            continue
        out.append(reduce(chunk).as_py())
    if method == "rolling_mean":
        return pa.array(out, pa.float64())
    if method == "rolling_sum" and pa.types.is_integer(src.type):
        # The rolling sum keeps the column's own width and wraps in it, as the element-wise
        # arithmetic does; pc.sum widens to 64 bits, so the total is folded back here.
        name = type_name_of(src)
        return pa.array([None if v is None else _wrap_to_width(v, name) for v in out], type=src.type)
    return pa.array(out, type=src.type)


def _wrap_to_width(value, name):
    bits = NUMPY_TYPE[name].itemsize * 8
    value &= (1 << bits) - 1
    if name in SIGNED and value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


@op("lexsort", NUMERIC,
    note="pc.sort_indices over a two-column table, which is what lexsort_indices orders by")
def _lexsort(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    keys = pa.table({"a": src, "b": other})
    got, expected = [], []
    for descending in ([False, False], [True, False], [False, True], [True, True]):
        order = [("a", "descending" if descending[0] else "ascending"),
                 ("b", "descending" if descending[1] else "ascending")]
        # null_placement defaults to "at_end", which is where both engines put the nulls of every key.
        got.append(arrow(am.lexsort_indices([am.array(src), am.array(other)], descending)))
        expected.append(pc.sort_indices(keys, sort_keys=order).cast(pa.int32()))
    return got, expected


# ---- statistical and positional aggregates ---------------------------

@op("product", NUMERIC, note="pc.product; integers wrap in 64 bits in both engines")
def _product(src, shape):
    """The generated magnitudes make a product overflow almost at once, so the column is reduced to the
    values whose running product stays inside int64 / the float range -- past that the two engines
    associate differently, which `cumulative_prod` already documents."""
    name = type_name_of(src)
    trimmed = _trim_for_product(src, name)
    got = am.array(trimmed).product()
    expected = pc.product(trimmed).as_py()
    if name in FLOATING:
        return got, expected, (FLOAT_TOL[name], 0.0)
    return got, expected


def _trim_for_product(src, name):
    """The values on which no association of the product can leave the type's normal range.

    `product` is a parallel tree reduction, so the intermediates are sub-products of arbitrary
    subsets, not the sequential running product; once one of them overflows or underflows the pairing
    decides the answer, exactly as `cumulative_prod` documents. Tracking `Π max(|v|, 1/|v|)` bounds
    *every* sub-product between its reciprocal and itself, so staying inside the band makes every
    association agree (test_product_reassociates_past_the_normal_range pins what happens outside)."""
    if name in INTEGER:
        # An integer product wraps in 64 bits in both engines, so only the magnitude of the whole
        # product has to stay inside the accumulator.
        values, keep, running = src.to_pylist(), [], 1.0
        for v in values:
            if v is None or v == 0:
                keep.append(True)
                continue
            step = running * abs(float(v))
            if step > 2.0 ** 62:
                keep.append(False)
            else:
                running, _ = step, keep.append(True)
        return src.filter(pa.array(keep, pa.bool_()))
    band = 1e30 if name == "float32" else 1e250
    keep, spread = [], 1.0
    for v in src.to_pylist():
        if v is None or v == 0 or not math.isfinite(v):
            keep.append(True)
            continue
        magnitude = abs(float(v))
        step = spread * max(magnitude, 1.0 / magnitude)
        if step > band:
            keep.append(False)
        else:
            spread = step
            keep.append(True)
    return src.filter(pa.array(keep, pa.bool_()))


#: The variance kernel accumulates the two moments on the GPU and subtracts them, which costs it
#: about six digits against pyarrow's exact two-pass algorithm: measured at 6e-10 relative over the
#: matrix, and 2e-8 on a float32 column. These are those numbers with two orders of margin.
_VARIANCE_TOL = {"float32": 1e-6, "float64": 1e-7}


@op("variance_and_stddev", NUMERIC,
    note="pc.variance / pc.stddev with ddof spelled out, to the relative accuracy the kernel states")
def _variance_and_stddev(src, shape):
    clean = _finite_only(_drop_nan(src), type_name_of(src)) if type_name_of(src) in FLOATING \
        else _drop_nan(src)
    x = am.array(clean)
    got = [x.variance(0), x.variance(1), x.stddev(0), x.stddev(1)]
    expected = [pc.variance(clean, ddof=0).as_py(), pc.variance(clean, ddof=1).as_py(),
                pc.stddev(clean, ddof=0).as_py(), pc.stddev(clean, ddof=1).as_py()]
    if len(clean) - clean.null_count < 2:
        got, expected = got[:1] + got[2:3], expected[:1] + expected[2:3]
    return got, expected, (_VARIANCE_TOL.get(type_name_of(src), 1e-7), 0.0)


@op("quantile", NUMERIC, tol="result_float",
    note="pc.quantile with interpolation='linear', the one ArrowMetal implements")
def _quantile(src, shape):
    clean = _drop_nan(src)
    x = am.array(clean)
    got, expected = [], []
    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        got.append(x.quantile(q))
        result = pc.quantile(clean, q=q, interpolation="linear")
        expected.append(result[0].as_py() if len(result) else None)
    got.append(x.median())
    result = pc.quantile(clean, q=0.5, interpolation="linear")
    expected.append(result[0].as_py() if len(result) else None)
    return got, expected


@op("mode_and_count_distinct", NUMERIC + ["bool"])
def _mode_and_count_distinct(src, shape):
    clean = _drop_nan(src)
    x = am.array(clean)
    modes = pc.mode(clean, n=1)
    expected_mode = None if len(modes) == 0 else pc.struct_field(modes, "mode")[0].as_py()
    expected_count = None if len(modes) == 0 else pc.struct_field(modes, "count")[0].as_py()
    got = x.mode()
    return ([None if got is None else got[0], None if got is None else got[1],
             x.count_distinct()],
            [expected_mode, expected_count, pc.count_distinct(clean).as_py()])


@op("first_last_index_min_max", NUMERIC + ["bool"],
    note="pc.first / pc.last / pc.index / pc.min_max; the needle is a value from the column itself")
def _first_last_index_min_max(src, shape):
    clean = _drop_nan(src)
    x = am.array(clean)
    got = [x.first(), x.last(), list(x.min_max())]
    expected = [pc.first(clean).as_py(), pc.last(clean).as_py(),
                [pc.min(clean).as_py(), pc.max(clean).as_py()]]
    needle = _index_needle(clean)
    if needle is not None:
        got.append(x.index(needle))
        expected.append(pc.index(clean, pa.scalar(needle, clean.type)).as_py())
    return got, expected


def _index_needle(clean):
    """A value from the column that `index()` can carry.

    `MetalArray.index` passes the value through a C `double`, so a 64-bit integer above 2^53 does not
    survive the trip (test_index_of_a_large_integer_is_rounded_through_a_double pins that). The needle
    has to round-trip exactly *and* be the only value in the column that rounds to that double, or the
    comparison would be about the rounding rather than about the search; if no value qualifies, index
    is left out of this case."""
    distinct = pc.unique(clean.drop_null()).to_pylist()
    if not distinct:
        return None
    by_double = {}
    for value in distinct:
        if isinstance(value, float) and not math.isfinite(value):
            continue
        by_double.setdefault(float(value), []).append(value)
    for value in clean.drop_null().to_pylist()[:64]:
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if float(value) == value and len(by_double.get(float(value), ())) == 1:
            return value
    return None


@op("any_all", ["bool"], note="pc.any / pc.all with min_count=0, so an empty column is false/true")
def _any_all(src, shape):
    x = am.array(src)
    return ([x.any(), x.all()],
            [pc.any(src, min_count=0).as_py(), pc.all(src, min_count=0).as_py()])


# ---- the remaining type rows ------------------------------------------

@op("float16_casts", ["float16"], note="pc.cast between float16 and float32, both directions")
def _float16_casts(src, shape):
    wide = am.array(src).to_float32()
    return ([arrow(wide), arrow(wide.to_float16())],
            [src.cast(pa.float32()), src])


@op("float16_compute", ["float16"],
    note="a float16 column widened to float32, computed on and narrowed back, against Arrow doing "
         "the same -- ArrowMetal never computes in half precision")
def _float16_compute(src, shape):
    wide = src.cast(pa.float32())
    x = am.array(src).to_float32()
    return ([arrow(x.abs()), arrow(x.negate()), arrow(am.array(src).to_float32().to_float16())],
            [pc.abs(wide), pc.negate(wide), src])


@op("float16_reduce", ["float16"], tol="result_float",
    note="the reductions run on the widened column in both engines")
def _float16_reduce(src, shape):
    clean = _drop_nan(src.cast(pa.float32()))
    narrow = clean.cast(pa.float16())
    x = am.array(narrow)
    return ([x.min(), x.max()], [pc.min(clean).as_py(), pc.max(clean).as_py()])


@op("fixed_binary_compare", ["fixed_size_binary"],
    note="pc.equal / pc.not_equal; ordering comparisons are not defined for the type")
def _fixed_binary_compare(src, shape):
    other = make_array("fixed_size_binary", shape, seed=1)
    x = am.array(src)
    scalar = b"abcdefgh"
    return ([arrow(x.fixed_binary_compare("==", am.array(other))),
             arrow(x.fixed_binary_compare("!=", am.array(other))),
             arrow(x.fixed_binary_compare("==", scalar))],
            [pc.equal(src, other), pc.not_equal(src, other),
             pc.equal(src, pa.scalar(scalar, src.type))])


@op("hash64", NUMERIC + ["bool", "fixed_size_binary"],
    note=_no_oracle("hash64",
                    "Arrow has no element-wise hash; the properties checked are that nulls stay null, "
                    "that equal values hash equal and that distinct values do not collide"))
def _hash64(src, shape):
    """A hash has no reference value, so the case checks the three things its caller relies on: the
    output is uint64 with the input's null positions, Arrow-equal values hash equal (which for a float
    column means `-0.0` hashes as `0.0` and every NaN alike), and the map from distinct value to hash
    is injective on the generated data."""
    hashes = arrow(am.array(src).hash64())
    if hashes.type != pa.uint64():
        return f"hash type {hashes.type}", "uint64"
    if not pc.is_null(hashes).equals(pc.is_null(src)):
        return "null positions differ between input and hashes", "identical null positions"
    canonical = src
    if type_name_of(src) in FLOATING:
        # Arrow calls -0.0 and 0.0 equal and every NaN distinct; the hash canonicalises both, so the
        # count of distinct values is taken the same way before comparing.
        filled = pc.fill_null(src, pa.scalar(0.0, src.type))
        canonical = pc.if_else(pc.is_nan(filled), pa.scalar(math.nan, src.type),
                               pc.add(src, pa.scalar(0.0, src.type)))
    pairs = pa.table({"v": canonical, "h": hashes}).filter(pc.is_valid(src))
    distinct_pairs = pairs.group_by(["v", "h"]).aggregate([([], "count_all")]).num_rows
    distinct_values = pc.count_distinct(canonical).as_py()
    distinct_hashes = pc.count_distinct(hashes).as_py()
    if distinct_pairs != distinct_values:
        return f"{distinct_values} values produced {distinct_pairs} (value, hash) pairs", \
               "one hash per value"
    if distinct_hashes != distinct_values:
        return f"{distinct_values} values collided onto {distinct_hashes} hashes", "no collisions"
    return "consistent", "consistent"


@op("dictionary_ops", ["dict_utf8"], note="decode is Arrow's cast back to the value type")
def _dictionary_ops(src, shape):
    x = am.array(src)
    return arrow(x.decode()), src.cast(pa.string())


@op("null_column", ["null"], note="the null type carries no buffers; filter, take and slice still work")
def _null_column(src, shape):
    x = am.array(src)
    n = len(src)
    mask = make_array("bool", shape, seed=2)
    idx = pa.array(np.random.default_rng(11).integers(0, max(n, 1), min(n, 97)).astype(np.int32),
                   type=pa.int32()) if n else pa.array([], pa.int32())
    return ([arrow(x.filter(am.array(mask))), arrow(x.take(am.array(idx))), arrow(x.slice(0, n // 2))],
            [src.filter(mask), src.take(idx), src.slice(0, n // 2)])


@op("extension_type", INTEGER,
    note=_no_oracle("extension_type",
                    "Arrow has no compute function for extension metadata; the reference is the "
                    "round trip through pyarrow's own extension registry"))
def _extension_type(src, shape):
    """`as_extension_type` writes `ARROW:extension:name` and `:metadata` into the exported schema, so
    pyarrow rebuilds the extension type when it is registered. The reference is pyarrow's own view of
    the same array."""
    tagged = am.array(src).as_extension_type("arrowmetal.differential", b"v1")
    exported = tagged.to_arrow()
    return ([tagged.extension_name, tagged.extension_metadata,
             arrow(tagged.extension_storage()), str(exported.type)],
            ["arrowmetal.differential", b"v1", src, str(src.type)])


# ---- trigonometry, the last boolean operators and the conditionals -----

_TRIG_UNARY = ["sin", "cos", "tan", "asin", "acos", "atan",
               "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"]

#: The header measures the worst case at 4 ulp (float32) and 5 ulp (float64) against the host libm.
#: These are those bounds with a margin, as relative tolerances; every special value (NaN, +/-inf,
#: +/-0 and the domain errors) is compared exactly by the pinned tests instead.
_TRIG_TOL = {"float32": (1e-6, 0.0), "float64": (1e-14, 0.0)}


@op("trig", FLOATING, note="pc.sin ... pc.atanh, to the ulp bound the header states")
def _trig(src, shape):
    x = am.array(src)
    got, expected = [], []
    for name in _TRIG_UNARY:
        got.append(arrow(getattr(x, name)()))
        expected.append(getattr(pc, name)(src))
    other = make_array(type_name_of(src), shape, seed=1)
    got.append(arrow(x.atan2(am.array(other))))
    expected.append(pc.atan2(src, other))
    return got, expected, _TRIG_TOL[type_name_of(src)]


#: (name, the values the checked form accepts). Everything outside raises in both engines.
_TRIG_CHECKED = {
    "sin_checked": lambda a: pc.is_finite(a), "cos_checked": lambda a: pc.is_finite(a),
    "tan_checked": lambda a: pc.is_finite(a),
    "asin_checked": lambda a: pc.less_equal(pc.abs(a), 1.0),
    "acos_checked": lambda a: pc.less_equal(pc.abs(a), 1.0),
    "acosh_checked": lambda a: pc.greater_equal(a, 1.0),
    "atanh_checked": lambda a: pc.less(pc.abs(a), 1.0),
}


@op("trig_checked", FLOATING,
    note="the values inside the domain agree with pyarrow's own _checked kernels; outside it, both "
         "engines have to raise")
def _trig_checked(src, shape):
    x = am.array(src)
    got, expected = [], []
    for name, domain in _TRIG_CHECKED.items():
        filled = pc.fill_null(src, pa.scalar(0.0, src.type))
        inside = pc.or_(pc.is_nan(filled), pc.fill_null(domain(filled), False))
        good = src.filter(inside)
        got.append(arrow(getattr(am.array(good), name)()))
        expected.append(getattr(pc, name)(good))
        # And the raising half: on the rows outside the domain both engines must object.
        bad = src.filter(pc.invert(inside))
        if len(bad) and bad.null_count < len(bad):
            got.append(_raises(lambda: getattr(am.array(bad), name)()))
            expected.append(_raises(lambda: getattr(pc, name)(bad)))
    return got, expected, _TRIG_TOL[type_name_of(src)]


def _raises(call):
    """True when the call reports a domain error, in either engine's way."""
    try:
        call()
        return False
    except (am.ArrowMetalError, pa.ArrowInvalid) as exc:
        if "domain error" not in str(exc):
            raise
        return True


@op("logical_extras", ["bool"], note="pc.xor / and_not / and_not_kleene")
def _logical_extras(src, shape):
    other = make_array("bool", shape, seed=1)
    x, y = am.array(src), am.array(other)
    return ([arrow(x.xor(y)), arrow(x.and_not(y)), arrow(x.and_not_kleene(y))],
            [pc.xor(src, other), pc.and_not(src, other), pc.and_not_kleene(src, other)])


@op("float_class", NUMERIC, note="pc.is_nan / is_inf / is_finite, defined on integers too")
def _float_class(src, shape):
    x = am.array(src)
    return ([arrow(x.is_nan()), arrow(x.is_inf()), arrow(x.is_finite())],
            [pc.is_nan(src), pc.is_inf(src), pc.is_finite(src)])


@op("fill_null_direction", NUMERIC + ["bool"],
    note="pc.fill_null_forward / _backward, on a materialised copy: pyarrow reads a *boolean* "
         "column's values bitmap without its offset")
def _fill_null_direction(src, shape):
    x = am.array(src)
    flat = _materialised(src)
    return ([arrow(x.fill_null_forward()), arrow(x.fill_null_backward())],
            [pc.fill_null_forward(flat), pc.fill_null_backward(flat)])


@op("case_when", NUMERIC + ["bool"], note="pc.case_when over a struct of conditions")
def _case_when(src, shape):
    first = make_array("bool", shape, seed=2)
    second = make_array("bool", shape, seed=6)
    other = make_array(type_name_of(src), shape, seed=1)
    default = make_array(type_name_of(src), shape, seed=7)
    got = arrow(am.case_when([am.array(first), am.array(second)],
                             [am.array(src), am.array(other)], am.array(default)))
    conditions = pa.StructArray.from_arrays([first, second], ["a", "b"])
    return got, pc.case_when(conditions, src, other, default)


@op("choose", NUMERIC + ["bool"], note="pc.choose; a null index gives a null row in both")
def _choose(src, shape):
    other = make_array(type_name_of(src), shape, seed=1)
    n = len(src)
    rng = np.random.default_rng(13)
    raw = rng.integers(0, 2, n).astype(np.int32)
    idx = pa.array(raw, mask=rng.random(n) < 0.1, type=pa.int32()) if n else pa.array([], pa.int32())
    return (arrow(am.choose(am.array(idx), [am.array(src), am.array(other)])),
            pc.choose(idx, src, other))


@op("replace_with_mask", NUMERIC + ["bool"])
def _replace_with_mask(src, shape):
    mask = make_array("bool", shape, seed=2)
    needed = pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0
    replacements = make_array(type_name_of(src), shape, seed=1).slice(0, needed)
    if len(replacements) < needed:
        raise Unsupported("not enough replacement values for this mask")
    # As above: pyarrow's boolean replace_with_mask reads the values bitmap without the offset.
    return (arrow(am.array(src).replace_with_mask(am.array(mask), am.array(replacements))),
            pc.replace_with_mask(_materialised(src), _materialised(mask),
                                 _materialised(replacements)))


@op("indices_nonzero", NUMERIC + ["bool"], note="pc.indices_nonzero: -0.0 is zero, every NaN is not")
def _indices_nonzero(src, shape):
    return arrow(am.array(src).indices_nonzero()), pc.indices_nonzero(src)


@op("coalesce", NUMERIC + ["bool"], note="pc.coalesce over three columns")
def _coalesce(src, shape):
    second = make_array(type_name_of(src), shape, seed=1)
    third = make_array(type_name_of(src), shape, seed=7)
    return (arrow(am.coalesce(am.array(src), am.array(second), am.array(third))),
            pc.coalesce(src, second, third))


@op("nulls_constructor", ["null"],
    note=_no_oracle("nulls_constructor", "am.nulls(n) is a constructor, not a compute function; "
                                         "the reference is pa.nulls(n)"))
def _nulls_constructor(src, shape):
    return arrow(am.nulls(len(src))), pa.nulls(len(src))


# ==================================================================== the Arrow-named surface
#
# The kernels that carry Arrow's own names: the whole `*_checked` family, the remaining element-wise
# math (expm1, log1p, logb, hypot and the two extra rounding forms), the associative transforms
# (unique / value_counts), the selection and permutation kernels, the statistical aggregates Arrow
# spells skew / kurtosis / tdigest / winsorize / rank_quantile / rank_normal, the byte-indexed string
# transforms, the struct-valued regex extractors and the two timezone-metadata calls.
#
# The rules are the ones above: pyarrow is the oracle wherever it has the function, with its options
# spelled out. Two things need the extra machinery in this section:
#
#   * a *checked* kernel is only comparable on the rows where it does not raise, so each case splits
#     its input into the rows both engines answer and the rows both engines have to object to, and
#     compares the answers on one half and the objections on the other -- the shape `trig_checked`
#     already uses for its domain errors;
#   * an *unstable* selection (top_k_unstable, select_k_unstable, partition_nth_indices) returns a
#     permutation neither engine promises, so the comparison is over the multiset of selected VALUES,
#     which is determined even when a tie straddles the cut.

def _raises_checked(call):
    """True when the call reports an overflow, a division by zero or a negative power, in either
    engine's way. Anything else propagates: a checked kernel that fails for another reason is a
    failure, not an expected objection."""
    try:
        call()
        return False
    except (am.ArrowMetalError, pa.ArrowInvalid) as exc:
        text = str(exc).lower()
        if not any(w in text for w in ("overflow", "divide by zero", "negative integer powers",
                                       "shift amount", "domain error", "logarithm", "square root")):
            raise
        return True


#: Below this magnitude a 64-bit accumulator is exact for a sum, a difference and a product of two
#: operands, so the overflow test can be done in numpy rather than in Python ints.
_EXACT_IN_INT64 = 2 ** 31


def _operand_values(src, other, filler):
    """The two columns as numpy arrays with the nulls filled: a null row is never an overflow in
    either engine, so it must not be flagged as one."""
    a = pc.fill_null(src, pa.scalar(filler, src.type)).to_numpy(zero_copy_only=False)
    b = pc.fill_null(other, pa.scalar(filler, other.type)).to_numpy(zero_copy_only=False)
    return a, b


def _integer_overflows(src, other, symbol, name):
    """A boolean numpy array: True exactly where the checked integer op leaves the column's own type.

    Exact throughout -- int64 while both operands are small enough for it to be, Python ints (which
    have no width at all) for the `special` flavor, whose values sit at the type's own extremes."""
    lo, hi = _LIMITS[name]
    a, b = _operand_values(src, other, 1)
    if len(a) == 0:
        return np.zeros(0, dtype=bool)
    small = (int(a.max()) < _EXACT_IN_INT64 and int(b.max()) < _EXACT_IN_INT64 and
             int(a.min()) > -_EXACT_IN_INT64 and int(b.min()) > -_EXACT_IN_INT64)
    if small:
        x, y = a.astype(np.int64), b.astype(np.int64)
        r = x + y if symbol == "+" else (x - y if symbol == "-" else x * y)
        return (r < lo) | (r > hi)
    out = []
    for x, y in zip(a.tolist(), b.tolist()):
        r = x + y if symbol == "+" else (x - y if symbol == "-" else x * y)
        out.append(r < lo or r > hi)
    return np.array(out, dtype=bool)


def _checked_objects(src, other, symbol, name):
    """The rows on which the checked op has to raise, as a pyarrow boolean array."""
    if symbol == "/":
        zero = pa.scalar(0.0 if name in FLOATING else 0, other.type)
        bad = pc.equal(pc.fill_null(other, pa.scalar(1 if name in INTEGER else 1.0, other.type)), zero)
        if name in SIGNED:
            # INT_MIN / -1 is the other divide_checked error, in both engines.
            lo, _ = _LIMITS[name]
            bad = pc.or_(bad, pc.and_(pc.equal(pc.fill_null(src, pa.scalar(0, src.type)),
                                               pa.scalar(lo, src.type)),
                                      pc.equal(pc.fill_null(other, pa.scalar(1, other.type)),
                                               pa.scalar(-1, other.type))))
        return bad
    if name in FLOATING:
        return pa.array(np.zeros(len(src), dtype=bool), pa.bool_())
    return pa.array(_integer_overflows(src, other, symbol, name), pa.bool_())


def _checked_pair(src, other, bad, method, oracle, got, expected):
    """Append the two halves of one checked op: the answers on the rows that do not raise, and the
    objection itself on the rows that do."""
    good = pc.invert(bad)
    s, o = src.filter(good), other.filter(good)
    got.append(arrow(getattr(am.array(s), method)(am.array(o))))
    expected.append(oracle(s, o))
    # A null row never raises, so the objecting half is taken over the valid rows only.
    raising = pc.and_(bad, pc.and_(pc.is_valid(src), pc.is_valid(other)))
    s, o = src.filter(raising), other.filter(raising)
    if len(s):
        got.append(_raises_checked(lambda: getattr(am.array(s), method)(am.array(o))))
        expected.append(_raises_checked(lambda: oracle(s, o)))


_CHECKED_ARITH = [("+", "add_checked", pc.add_checked), ("-", "subtract_checked", pc.subtract_checked),
                  ("*", "multiply_checked", pc.multiply_checked),
                  ("/", "divide_checked", pc.divide_checked)]


@op("arith_checked", NUMERIC,
    note="pc.add_checked / subtract_checked / multiply_checked / divide_checked. The input is split "
         "in two: the rows where the op fits the column's type are compared value by value, and the "
         "rows where it does not (an overflow, a zero divisor, INT_MIN / -1) have to make BOTH "
         "engines raise")
def _arith_checked(src, shape):
    name = type_name_of(src)
    other = make_array(name, shape, seed=1)
    got, expected = [], []
    for symbol, method, oracle in _CHECKED_ARITH:
        bad = _checked_objects(src, other, symbol, name)
        _checked_pair(src, other, bad, method, oracle, got, expected)
    # `binary_checked` is the generic dispatcher every method above goes through; called by name here
    # so the dispatch table itself is under test.
    bad = _checked_objects(src, other, "+", name)
    good = pc.invert(bad)
    s, o = src.filter(good), other.filter(good)
    got.append(arrow(am.array(s).binary_checked("add", am.array(o))))
    expected.append(pc.add_checked(s, o))
    # And the scalar form, which takes a different C entry point (a packed scalar, not a column).
    constant = pa.array(np.full(len(src), scalar_for(name)), type=src.type)
    bad = _checked_objects(src, constant, "+", name)
    s = src.filter(pc.invert(bad))
    got.append(arrow(am.array(s).add_checked(scalar_for(name))))
    expected.append(pc.add_checked(s, pa.scalar(scalar_for(name), src.type)))
    return got, expected


@op("power_checked", INTEGER,
    note="pc.power_checked with the exponents folded into [0, 7], as the unchecked `power` case does; "
         "the rows whose result leaves the type have to raise in both engines")
def _power_checked(src, shape):
    name = type_name_of(src)
    other = make_array(name, shape, seed=1)
    wide = pa.uint64() if name in UNSIGNED else pa.int64()
    e = pc.bit_wise_and(pc.cast(other, wide, safe=False), pa.scalar(7, wide))
    exponent = pc.cast(e, other.type, safe=False)
    lo, hi = _LIMITS[name]
    base, power = _operand_values(src, exponent, 1)
    magnitude = np.abs(base.astype(np.float64)) ** power.astype(np.float64)
    with np.errstate(invalid="ignore"):
        # Conservative on both sides: the rows in between are left out of the case entirely, since a
        # float64 magnitude cannot decide them.
        safe = magnitude <= min(hi, -lo) / 4.0
        objects = magnitude > 4.0 * max(hi, -lo)
    got, expected = [], []
    keep = pa.array(safe, pa.bool_())
    s, o = src.filter(keep), exponent.filter(keep)
    got.append(arrow(am.array(s).power_checked(am.array(o))))
    expected.append(pc.power_checked(s, o))
    raising = pc.and_(pa.array(objects, pa.bool_()),
                      pc.and_(pc.is_valid(src), pc.is_valid(exponent)))
    s, o = src.filter(raising), exponent.filter(raising)
    if len(s):
        got.append(_raises_checked(lambda: am.array(s).power_checked(am.array(o))))
        expected.append(_raises_checked(lambda: pc.power_checked(s, o)))
    return got, expected


@op("shift_checked", INTEGER,
    note="pc.shift_left_checked / shift_right_checked on the counts both engines accept -- [0, digits) "
         "-- plus the count `digits` itself, which both have to refuse")
def _shift_checked(src, shape):
    name = type_name_of(src)
    counts = _shift_counts(make_array(name, shape, seed=1), name)
    x = am.array(src)
    got, expected = [], []
    for method, oracle in [("shift_left_checked", pc.shift_left_checked),
                           ("shift_right_checked", pc.shift_right_checked)]:
        got.append(arrow(getattr(x, method)(am.array(counts))))
        expected.append(oracle(src, counts))
        # The first count outside the accepted range: Arrow's message is "shift amount must be >= 0
        # and less than precision of type".
        valid = src.filter(pc.is_valid(src))
        if len(valid):
            too_far = pa.array(np.full(len(valid), _shift_digits(name)), type=src.type)
            got.append(_raises_checked(
                lambda m=method, v=valid, t=too_far: getattr(am.array(v), m)(am.array(t))))
            expected.append(_raises_checked(lambda o=oracle, v=valid, t=too_far: o(v, t)))
    return got, expected


@op("unary_checked", NUMERIC,
    note="pc.abs_checked and pc.negate_checked. pyarrow has no negate_checked kernel for an unsigned "
         "column at all, so only abs_checked is compared there (ArrowMetal's raises for every "
         "non-zero unsigned value, which the pinned test records)")
def _unary_checked(src, shape):
    name = type_name_of(src)
    lo, _ = _LIMITS.get(name, (None, None))
    x = am.array(src)
    got, expected = [], []
    # INT_MIN is the one value abs_checked and negate_checked object to.
    inside = src if name in FLOATING else \
        src.filter(pc.not_equal(pc.fill_null(src, pa.scalar(0, src.type)), pa.scalar(lo, src.type)))
    methods = ["abs_checked"] + ([] if name in UNSIGNED else ["negate_checked"])
    for method in methods:
        oracle = pc.abs_checked if method == "abs_checked" else pc.negate_checked
        got.append(arrow(getattr(am.array(inside), method)()))
        expected.append(oracle(inside))
    if name in SIGNED:
        extreme = src.filter(pc.equal(pc.fill_null(src, pa.scalar(0, src.type)),
                                      pa.scalar(lo, src.type)))
        if len(extreme):
            for method in methods:
                oracle = pc.abs_checked if method == "abs_checked" else pc.negate_checked
                got.append(_raises_checked(lambda m=method, e=extreme: getattr(am.array(e), m)()))
                expected.append(_raises_checked(lambda o=oracle, e=extreme: o(e)))
    # `unary_checked` is the dispatcher the two methods above go through; called by name so the
    # dispatch table is under test too.
    got.append(arrow(am.array(inside).unary_checked("abs")))
    expected.append(pc.abs_checked(inside))
    return got, expected


#: (method, oracle, the values the checked form accepts). Everything outside raises in both engines.
_CHECKED_DOMAIN = [
    ("sqrt_checked", pc.sqrt_checked, lambda a: pc.greater_equal(a, 0.0)),
    ("ln_checked", pc.ln_checked, lambda a: pc.greater(a, 0.0)),
    ("log10_checked", pc.log10_checked, lambda a: pc.greater(a, 0.0)),
    ("log2_checked", pc.log2_checked, lambda a: pc.greater(a, 0.0)),
    ("log1p_checked", pc.log1p_checked, lambda a: pc.greater(a, -1.0)),
]


@op("log_checked", FLOATING,
    note="pc.sqrt_checked / ln_checked / log10_checked / log2_checked / log1p_checked inside their "
         "domains, and the domain error itself outside them; evaluated in float32 for a float32 "
         "column, as the unchecked forms are")
def _log_checked(src, shape):
    base = _float32_representable(src)
    got, expected = [], []
    for method, oracle, domain in _CHECKED_DOMAIN:
        filled = pc.fill_null(base, pa.scalar(1.0, base.type))
        inside = pc.or_(pc.is_nan(filled), pc.fill_null(domain(filled), False))
        good = base.filter(inside)
        result = oracle(good)
        got.append(arrow(getattr(am.array(good), method)()))
        expected.append(result.cast(good.type) if result.type != good.type else result)
        bad = base.filter(pc.invert(inside))
        if len(bad) and bad.null_count < len(bad):
            got.append(_raises_checked(lambda m=method, b=bad: getattr(am.array(b), m)()))
            expected.append(_raises_checked(lambda o=oracle, b=bad: o(b)))
    tolerance = (1e-6, 1e-6) if pa.types.is_float32(src.type) else (4.0 * _EPS["float64"], 1e-15)
    return got, expected, tolerance


def _trim_running(src, name, symbol):
    """The values on which a running sum or product stays inside the column's own integer type, so
    the checked cumulative kernel never has to raise. A value that would take the accumulator out of
    range is dropped and the running value is left where it was, which is what keeps the trimmed
    column a valid input for the same scan."""
    lo, hi = _LIMITS[name]
    running, keep = (0 if symbol == "+" else 1), []
    for value in src.to_pylist():
        if value is None:
            keep.append(True)
            continue
        step = running + value if symbol == "+" else running * value
        if step < lo or step > hi:
            keep.append(False)
        else:
            running = step
            keep.append(True)
    return src.filter(pa.array(keep, pa.bool_()))


@op("cumulative_checked", INTEGER,
    note="pc.cumulative_sum_checked / cumulative_prod_checked / pairwise_diff_checked with "
         "skip_nulls=True, on the values whose running total stays inside the column's own type; a "
         "float column never raises in either engine and its values are already the unchecked cases'")
def _cumulative_checked(src, shape):
    name = type_name_of(src)
    got, expected = [], []
    for symbol, method, oracle in [("+", "cumulative_sum_checked", pc.cumulative_sum_checked),
                                   ("*", "cumulative_prod_checked", pc.cumulative_prod_checked)]:
        trimmed = _trim_running(src, name, symbol)
        got.append(arrow(getattr(am.array(trimmed), method)()))
        expected.append(oracle(trimmed, skip_nulls=True))
    for period in (1, 2, 5):
        # x[i] - x[i - period] can leave the type wherever the plain difference can.
        shifted = pc.if_else(pa.array(np.arange(len(src)) >= period, pa.bool_()),
                             _materialised(src).take(pa.array(np.maximum(np.arange(len(src)) - period, 0),
                                                              pa.int32())),
                             pa.scalar(0, src.type))
        bad = _checked_objects(src, shifted, "-", name)
        keep = pc.invert(pc.fill_null(bad, False))
        if pc.all(keep).as_py() is not False:
            got.append(arrow(am.array(src).pairwise_diff_checked(period)))
            expected.append(pc.pairwise_diff_checked(_materialised(src), period=period))
    return got, expected


# ---- the remaining element-wise math ---------------------------------

@op("math_extra", FLOATING,
    note="pc.expm1 / log1p / logb / hypot, plus logb_checked on the rows where the value and the base "
         "are both positive and the domain error outside them; the accuracy the header states for "
         "this family is 5 ulp of the host libm")
def _math_extra(src, shape):
    base = _float32_representable(src)
    # The float32 trim drops rows, so the second operand is cut to the same length rather than being
    # filtered by the same mask: which rows pair up does not matter here, only that they line up.
    other = make_array(type_name_of(src), shape, seed=1).slice(0, len(base))
    base = base.slice(0, len(other))
    x, y = am.array(base), am.array(other)
    got = [arrow(x.expm1()), arrow(x.log1p()), arrow(x.logb(y)), arrow(x.logb(2.0)),
           arrow(x.hypot(y)), arrow(x.hypot(2.0)),
           # the generic dispatcher the five methods above go through
           arrow(x.math_extra("expm1"))]
    expected = [pc.expm1(base), pc.log1p(base), pc.logb(base, other),
                pc.logb(base, pa.scalar(2.0, base.type)), pc.hypot(base, other),
                pc.hypot(base, pa.scalar(2.0, base.type)), pc.expm1(base)]
    # logb_checked: both engines refuse a value or a base that is not strictly positive.
    positive = pc.and_(pc.greater(pc.fill_null(base, pa.scalar(1.0, base.type)), 0.0),
                       pc.greater(pc.fill_null(other, pa.scalar(1.0, other.type)), 0.0))
    good_base, good_other = base.filter(positive), other.filter(positive)
    got.append(arrow(am.array(good_base).logb_checked(am.array(good_other))))
    expected.append(pc.logb_checked(good_base, good_other))
    bad_base, bad_other = base.filter(pc.invert(positive)), other.filter(pc.invert(positive))
    if len(bad_base) and bad_base.null_count < len(bad_base) and bad_other.null_count < len(bad_other):
        got.append(_raises_checked(lambda: am.array(bad_base).logb_checked(am.array(bad_other))))
        expected.append(_raises_checked(lambda: pc.logb_checked(bad_base, bad_other)))
    tolerance = (1e-6, 1e-6) if pa.types.is_float32(src.type) else (1e-13, 1e-300)
    return got, expected, tolerance


_ROUND_MODES = ["down", "up", "towards_zero", "towards_infinity", "half_down", "half_up",
                "half_towards_zero", "half_towards_infinity", "half_to_even", "half_to_odd"]


@op("round_extra", FLOATING,
    note="pc.round_to_multiple and pc.round_binary over all ten Arrow round modes, plus round() with "
         "an explicit ndigits (the no-argument form is the `rounding` case)")
def _round_extra(src, shape):
    # A round to a multiple or to a digit count multiplies before it rounds, so a value near the top
    # of the range would compare an overflow rather than a rounding rule; those rows come out.
    filled = pc.fill_null(src, pa.scalar(0.0, src.type))
    limit = pa.scalar(float(np.finfo(NUMPY_TYPE[type_name_of(src)]).max) / 1e4, src.type)
    base = src.filter(pc.or_(pc.is_nan(filled), pc.less_equal(pc.abs(filled), limit)))
    x = am.array(base)
    digits = pa.array(np.tile(np.array([0, 1, 2, -1], np.int32), len(base) // 4 + 1)[:len(base)],
                      pa.int32())
    got, expected = [], []
    for mode in _ROUND_MODES:
        got += [arrow(x.round_to_multiple(0.5, mode)), arrow(x.round_binary(am.array(digits), mode)),
                arrow(x.round(2, mode))]
        expected += [pc.round_to_multiple(base, multiple=0.5, round_mode=mode),
                     pc.round_binary(base, digits, round_mode=mode),
                     pc.round(base, ndigits=2, round_mode=mode)]
    return got, expected


# ---- the associative transforms --------------------------------------

@op("unique", ALL_TYPES,
    note="pc.unique in Arrow's own first-appearance order, and the sorted order ArrowMetal also "
         "offers against the same distinct values sorted")
def _unique(src, shape):
    x = am.array(src)
    first = pc.unique(src)
    ascending = first.drop_null()
    ascending = ascending.take(pc.array_sort_indices(ascending))
    return ([arrow(x.unique()), arrow(x.unique("sorted"))], [first, ascending])


@op("value_counts", ALL_TYPES,
    note="pc.value_counts: a struct<values, counts> with int64 counts, in the same two orders unique "
         "offers")
def _value_counts(src, shape):
    x = am.array(src)
    counted = pc.value_counts(src)
    values, counts = pc.struct_field(counted, "values"), pc.struct_field(counted, "counts")
    valid = pc.is_valid(values)                                 # the sorted order drops the null entry
    values, counts = values.filter(valid), counts.filter(valid)
    order = pc.array_sort_indices(values)
    got = [arrow(x.value_counts()), arrow(x.value_counts("sorted"))]
    expected = [counted,
                pa.StructArray.from_arrays([values.take(order), counts.take(order)],
                                           ["values", "counts"])]
    return got, expected


# ---- the Arrow-named selection and permutation kernels ----------------

@op("sort_indices", NUMERIC + ["utf8"],
    note="pc.array_sort_indices and single-key pc.sort_indices with null_placement spelled out both "
         "ways; array_sort_indices, sort_indices and argsort are one kernel under three Arrow names")
def _sort_indices(src, shape):
    x = am.array(src)
    got, expected = [], []
    for descending in (False, True):
        for placement in ("at_end", "at_start"):
            order = "descending" if descending else "ascending"
            got += [arrow(x.array_sort_indices(descending, placement)),
                    arrow(x.sort_indices(descending, placement))]
            reference = pc.array_sort_indices(src, order=order,
                                              null_placement=placement).cast(pa.int32())
            expected += [reference, reference]
    return got, expected


@op("array_selection", ALL_TYPES,
    note="array_filter and array_take, Arrow's names for filter and take; the same kernels reached "
         "through the second name")
def _array_selection(src, shape):
    mask = make_array("bool", shape, seed=2)
    n = len(src)
    if n == 0:
        idx = pa.array([], pa.int32())
    else:
        rng = np.random.default_rng(11)
        raw = rng.integers(0, n, min(n, 977)).astype(np.int32)
        idx = pa.array(raw, mask=rng.random(len(raw)) < 0.1, type=pa.int32())
    x = am.array(src)
    return ([arrow(x.array_filter(am.array(mask))), arrow(x.array_take(am.array(idx)))],
            [src.filter(mask), src.take(idx)])


@op("invert", ["bool"], note="pc.invert, Arrow's name for the ~ operator the bool_logic case uses")
def _invert(src, shape):
    return arrow(am.array(src).invert()), pc.invert(src)


def _selected_values(src, indices):
    """The values an index list selects, sorted, so an unstable selection compares as a multiset.

    On a float column `-0.0` is folded onto `0.0` first: the two are one tie group in every sort
    order, so which of them an unstable selection picks is not defined by either engine -- the
    multiset of *values* is only determined once the tie is canonicalised."""
    picked = src.take(indices.cast(pa.int32()) if indices.type != pa.int32() else indices)
    if picked.type in _FLOAT_ARROW:
        picked = pc.add(picked, pa.scalar(0.0, picked.type))          # -0.0 + 0.0 == 0.0
    return picked.take(pc.array_sort_indices(picked))


@op("select_k_unstable", NUMERIC,
    note="pc.select_k_unstable. Neither engine promises WHICH of a tied pair it selects, so the "
         "comparison is over the multiset of the selected values, which is determined even when a "
         "tie straddles the cut")
def _select_k_unstable(src, shape):
    x = am.array(src)
    got, expected = [], []
    for k in (0, 1, 17, len(src)):
        if k > len(src):
            continue
        for method, order in [("select_k_unstable", "ascending"),
                              ("bottom_k_unstable", "ascending"),
                              ("top_k_unstable", "descending")]:
            indices = arrow(getattr(x, method)(k))
            reference = pc.select_k_unstable(src, k=k, sort_keys=[("", order)])
            got.append(_selected_values(src, indices))
            expected.append(_selected_values(src, reference))
    return got, expected


@op("partition_nth_indices", NUMERIC,
    note="pc.partition_nth_indices. The permutation itself is not defined by either engine; what is "
         "defined is the SET below the pivot, so both sides are compared as sorted value multisets "
         "and the result is checked to be a permutation of the row numbers")
def _partition_nth_indices(src, shape):
    x = am.array(src)
    n = len(src)
    got, expected = [], []
    for pivot in (0, 1, 7, n // 2, n):
        if pivot > n:
            continue
        indices = arrow(x.partition_nth_indices(pivot))
        if sorted(indices.to_pylist()) != list(range(n)):
            return f"partition_nth_indices({pivot}) is not a permutation of 0..{n - 1}", \
                   "a permutation"
        reference = pc.partition_nth_indices(src, pivot=pivot)
        got.append(_selected_values(src, indices.slice(0, pivot)))
        expected.append(_selected_values(src, reference.slice(0, pivot)))
    return got, expected


def _permutation_indices(src, shape):
    """An index column for inverse_permutation / scatter: the column's own sort order, plus a second
    one carrying nulls and duplicates, which both engines define (a position no index names is null,
    and the last of several wins)."""
    n = len(src)
    order = pc.array_sort_indices(src).cast(pa.int32())
    if n == 0:
        return [order, order]
    rng = np.random.default_rng(29)
    raw = rng.integers(0, n, n).astype(np.int32)
    return [order, pa.array(raw, mask=rng.random(n) < 0.2, type=pa.int32())]


@op("permutation", NUMERIC,
    note="pc.inverse_permutation and pc.scatter, over the column's own sort order and over an index "
         "column with nulls and duplicates in it; max_index spelled out and defaulted")
def _permutation(src, shape):
    n = len(src)
    x = am.array(src)
    got, expected = [], []
    for indices in _permutation_indices(src, shape):
        idx = am.array(indices)
        for max_index in (-1, n + 3):
            kwargs = {} if max_index < 0 else {"max_index": max_index}
            got += [arrow(am.array(indices).inverse_permutation(max_index)),
                    arrow(x.scatter(idx, max_index))]
            expected += [pc.inverse_permutation(indices, **kwargs).cast(pa.int32()),
                         pc.scatter(src, indices, **kwargs)]
    return got, expected


@op("scatter", ALL_TYPES, note="pc.scatter, which is that inverse permutation used as a take, on "
                               "every column type rather than only the numeric ones")
def _scatter(src, shape):
    n = len(src)
    if n == 0:
        indices = pa.array([], pa.int32())
    else:
        rng = np.random.default_rng(31)
        indices = pa.array(rng.permutation(n).astype(np.int32), pa.int32())
    return arrow(am.array(src).scatter(am.array(indices))), pc.scatter(src, indices)


# ---- the statistical aggregates Arrow names ---------------------------

def _moment_input(src):
    """The rows the moment aggregates are compared on: NaN out (both engines treat it the way the
    other reductions do) and, on a float column, the values whose squares stay finite."""
    name = type_name_of(src)
    return _finite_only(_drop_nan(src), name) if name in FLOATING else _drop_nan(src)


#: Measured worst deviation of skew / kurtosis from pyarrow's two-pass answer over the whole matrix
#: is 4.6e-7 of max(|value|, 1); this is that number with 20x margin, as a relative and an absolute
#: bound together (skew and kurtosis are O(1) quantities that pass through zero, where a purely
#: relative bound means nothing).
_MOMENT_TOL = (1e-5, 1e-5)


@op("skew_kurtosis", NUMERIC,
    note="pc.skew / pc.kurtosis, biased (Arrow's default) and unbiased; NaN is dropped from the input "
         "for the reason the other reductions drop it")
def _skew_kurtosis(src, shape):
    clean = _moment_input(src)
    x = am.array(clean)
    got = [x.skew(), x.skew(False), x.kurtosis(), x.kurtosis(False)]
    expected = [pc.skew(clean).as_py(), pc.skew(clean, biased=False).as_py(),
                pc.kurtosis(clean).as_py(), pc.kurtosis(clean, biased=False).as_py()]
    # Arrow answers NaN where the moment is undefined (fewer rows than the moment needs, or a zero
    # variance); ArrowMetal answers null. Both mean "no answer", so they are folded together.
    expected = [None if isinstance(v, float) and math.isnan(v) else v for v in expected]
    return got, expected, _MOMENT_TOL


@op("tdigest", NUMERIC,
    note="pc.tdigest. Both engines answer with a t-digest, which is a SKETCH, so the comparison is "
         "an absolute bound of 5% of the column's own range -- measured worst deviation over the "
         "matrix is 0.6% of the range, at q = 0.75; q = 0 and q = 1 are the exact extremes in both")
def _tdigest(src, shape):
    clean = _moment_input(src)
    x = am.array(clean)
    got, expected = [], []
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = pc.tdigest(clean, q=q)
        got.append(x.tdigest(q))
        expected.append(result[0].as_py() if len(result) else None)
    low, high = pc.min(clean).as_py(), pc.max(clean).as_py()
    spread = 0.0 if low is None else abs(float(high) - float(low))
    return got, expected, (0.0, 0.05 * spread if math.isfinite(spread) else 0.0)


@op("winsorize", NUMERIC,
    note="pc.winsorize on a materialised copy: the values below the lower quantile and above the "
         "upper one are clamped to them, with Arrow's *nearest* (not interpolated) quantiles. "
         "pyarrow 25.0.1 reads a sliced column's validity bitmap without its offset "
         "(test_pyarrow_winsorize_ignores_the_array_offset pins that), so only the oracle is "
         "materialised -- ArrowMetal still gets the slice")
def _winsorize(src, shape):
    x = am.array(src)
    flat = _materialised(src)
    got, expected = [], []
    for lower, upper in [(0.0, 1.0), (0.05, 0.95), (0.25, 0.75), (0.5, 0.5)]:
        got.append(arrow(x.winsorize(lower, upper)))
        expected.append(pc.winsorize(flat, lower_limit=lower, upper_limit=upper))
    return got, expected


@op("rank_quantile_and_normal", NUMERIC, tol=1e-12,
    note="pc.rank_quantile and pc.rank_normal in both sort directions and with the nulls at either "
         "end; rank_normal's inverse CDF is a host implementation of AS 241, hence the 1e-12")
def _rank_quantile_and_normal(src, shape):
    x = am.array(src)
    got, expected = [], []
    for keys in ("ascending", "descending"):
        for placement in ("at_end", "at_start"):
            got += [arrow(x.rank_quantile(keys, placement)),
                    arrow(x.rank_normal(keys, placement))]
            expected += [pc.rank_quantile(src, sort_keys=keys, null_placement=placement),
                         pc.rank_normal(src, sort_keys=keys, null_placement=placement)]
    return got, expected


@op("first_last", NUMERIC + ["bool"],
    note="pc.first_last with skip_nulls both ways: one struct<first, last> row of the column's own type")
def _first_last(src, shape):
    clean = _drop_nan(src)
    x = am.array(clean)
    got, expected = [], []
    for skip in (True, False):
        result = pc.first_last(clean, skip_nulls=skip)
        got.append(arrow(x.first_last(skip)))
        expected.append(pa.StructArray.from_arrays(
            [pa.array([result["first"].as_py()], clean.type),
             pa.array([result["last"].as_py()], clean.type)], ["first", "last"]))
    return got, expected


@op("true_unless_null", ALL_TYPES,
    note="pc.true_unless_null, and count_all against the row count Arrow reports for the same column")
def _true_unless_null(src, shape):
    x = am.array(src)
    return ([arrow(x.true_unless_null()), x.count_all()],
            [pc.true_unless_null(src), len(src)])


@op("list_parent_indices64", LIST_TYPES,
    note="pc.list_parent_indices, which is int64; list_parent_indices() is the int32 form this "
         "package uses everywhere else and has its own case")
def _list_parent_indices64(src, shape):
    return arrow(am.array(src).list_parent_indices64()), pc.list_parent_indices(src)


# ---- the byte-indexed string transforms -------------------------------

@op("ascii_pad", ["utf8"],
    note="pc.ascii_lpad / ascii_rpad / ascii_center, which count BYTES, and pc.utf8_lpad / utf8_rpad, "
         "which count code points -- the pair that separates the two families on accented input")
def _ascii_pad(src, shape):
    x = am.array(src)
    got, expected = [], []
    for width, pad in [(0, " "), (6, " "), (9, "*")]:
        got += [arrow(x.ascii_lpad(width, pad)), arrow(x.ascii_rpad(width, pad)),
                arrow(x.ascii_center(width, pad)),
                arrow(x.utf8_lpad(width, pad)), arrow(x.utf8_rpad(width, pad))]
        expected += [pc.ascii_lpad(src, width, padding=pad), pc.ascii_rpad(src, width, padding=pad),
                     pc.ascii_center(src, width, padding=pad),
                     pc.utf8_lpad(src, width, padding=pad), pc.utf8_rpad(src, width, padding=pad)]
    return got, expected


@op("swapcase_and_zero_fill", ["utf8"],
    note="pc.utf8_swapcase / ascii_swapcase and pc.utf8_zero_fill, which inserts its padding after a "
         "leading sign")
def _swapcase_and_zero_fill(src, shape):
    x = am.array(src)
    got = [arrow(x.utf8_swapcase()), arrow(x.ascii_swapcase())]
    expected = [pc.utf8_swapcase(src), pc.ascii_swapcase(src)]
    for width in (0, 5, 9):
        got.append(arrow(x.utf8_zero_fill(width)))
        expected.append(pc.utf8_zero_fill(src, width))
        got.append(arrow(x.utf8_zero_fill(width, "*")))
        expected.append(pc.utf8_zero_fill(src, width, padding="*"))
    return got, expected


@op("byte_transforms", ["utf8"],
    note="pc.binary_slice and pc.binary_reverse, which index BYTES and return binary, and "
         "pc.ascii_reverse, which both engines refuse on non-ASCII input")
def _byte_transforms(src, shape):
    x = am.array(src)
    binary = src.cast(pa.binary())
    got, expected = [], []
    for start, stop, step in [(0, None, 1), (1, 3, 1), (2, None, 1), (-3, None, 1), (0, -1, 1),
                              (5, 2, 1), (0, None, -1), (1, 6, 2)]:
        got.append(arrow(x.binary_slice(start, stop, step)))
        expected.append(pc.binary_slice(binary, start, stop, step) if stop is not None
                        else pc.binary_slice(binary, start, step=step))
    got.append(arrow(x.binary_reverse()))
    expected.append(pc.binary_reverse(binary))
    # ascii_reverse: both engines refuse a value that is not ASCII, so the rows are split.
    ascii_only = src.filter(pc.fill_null(pc.string_is_ascii(src), True))
    got.append(arrow(am.array(ascii_only).ascii_reverse()))
    expected.append(pc.ascii_reverse(ascii_only))
    wide = src.filter(pc.invert(pc.fill_null(pc.string_is_ascii(src), True)))
    if len(wide):
        got.append(_raises_non_ascii(lambda: am.array(wide).ascii_reverse()))
        expected.append(_raises_non_ascii(lambda: pc.ascii_reverse(wide)))
    return got, expected


def _raises_non_ascii(call):
    try:
        call()
        return False
    except (am.ArrowMetalError, pa.ArrowInvalid) as exc:
        if "non-ascii" not in str(exc).lower():
            raise
        return True


@op("extract_regex_structs", ["utf8"],
    note="pc.extract_regex and pc.extract_regex_span as struct columns -- the shape Arrow returns, "
         "against the dict-of-columns forms the regex_extract cases already compare")
def _extract_regex_structs(src, shape):
    x = am.array(src)
    got, expected = [], []
    for ours, theirs in [(r"(?<head>a)(?<tail>p+)", r"(?P<head>a)(?P<tail>p+)"),
                         (r"(?<digits>[0-9]+)", r"(?P<digits>[0-9]+)"),
                         (r"(?<never>zzz)", r"(?P<never>zzz)")]:
        got.append(arrow(x.extract_regex_struct(ours)))
        expected.append(pc.extract_regex(src, theirs))
        got.append(arrow(x.extract_regex_span_struct(ours)))
        expected.append(pc.extract_regex_span(src, theirs))
    return got, expected


@op("split_pairs", ["utf8"],
    note="split_pattern_pair and split_whitespace_pair: the same splits as the list-valued forms, "
         "handed back as the flat (offsets, values) buffers, rebuilt here into the list Arrow returns")
def _split_pairs(src, shape):
    x = am.array(src)
    got, expected = [], []
    for separator in (" ", "a", "pp"):
        got.append(_as_list_array(*x.split_pattern_pair(separator)))
        expected.append(_split_result(x.split_pattern(separator)))
    for pattern in ("[0-9]", "l+"):
        got.append(_as_list_array(*x.split_pattern_pair(pattern, regex=True)))
        expected.append(_split_result(x.split_pattern(pattern, regex=True)))
    for unicode_class in (False, True):
        got.append(_as_list_array(*x.split_whitespace_pair(unicode_class)))
        expected.append(_split_result(x.split_whitespace(unicode_class)))
    return got, expected


# ---- the timezone metadata calls --------------------------------------

@op("timezone_metadata", TIMESTAMP_TYPES,
    note="to_timezone is Arrow's cast between timestamp timezones (metadata only, so the oracle is "
         "that cast); utc_offset has no pyarrow function and is compared against the difference "
         "pc.local_timestamp leaves behind")
def _timezone_metadata(src, shape):
    x = am.array(src)
    got = [arrow(x.to_timezone("UTC")), arrow(x.to_timezone(TZ_NAME)), arrow(x.to_timezone(None))]
    expected = [src.cast(pa.timestamp(src.type.unit, "UTC")),
                src.cast(pa.timestamp(src.type.unit, TZ_NAME)),
                src.cast(pa.timestamp(src.type.unit))]
    # utc_offset: the seconds pc.local_timestamp moves the instant by, which is the definition.
    naive = pc.local_timestamp(src) if src.type.tz else src
    offset = pc.seconds_between(src.cast(pa.timestamp(src.type.unit)), naive)
    got.append(arrow(x.utc_offset()))
    expected.append(offset.cast(pa.int32()))
    return got, expected


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
    """A divergence this harness has already reported and docs/EVALUATION.md explains. Failures that
    match one are still failures -- the cell reads `known n/N` and pytest xfails it -- but they no
    longer gate: the report exits on *unclassified* divergences, so a new one stands out.

    `data_check` narrows a finding to the datasets whose *values* trigger it, which the (operation,
    type, shape) triple alone cannot express -- a case-mapping gap only bites on the strings that
    need the mapping."""

    def __init__(self, ident, title, ops, types, flavors=None, needs_nulls=False, data_check=None):
        self.id, self.title = ident, title
        self.ops, self.types, self.flavors, self.needs_nulls = set(ops), set(types), flavors, needs_nulls
        self.data_check = data_check

    def matches(self, op_name, type_name, shape):
        if op_name not in self.ops or type_name not in self.types:
            return False
        if self.flavors is not None and shape.flavor not in self.flavors:
            return False
        if self.needs_nulls and shape.null_ratio <= 0:
            return False
        return self.data_check is None or self.data_check(make_array(type_name, shape))


def _contains_negative_zero(src):
    return any(v == 0.0 and math.copysign(1.0, v) < 0 for v in src.to_pylist() if v is not None)


def _prefix_product_leaves_safe_range(src):
    """True when a sequential running product of the values gets within 2^40 of overflow or of the
    smallest normal for the type. A parallel scan multiplies in a different order, so past that point
    which intermediates overflow (inf), underflow (0) or meet (inf * 0 = NaN) depends on the order."""
    values = [v for v in src.to_pylist() if v is not None]
    if not values:
        return False
    if pa.types.is_float32(src.type):
        hi, lo = 3.4028235e38 / 2.0 ** 40, 1.1754944e-38 * 2.0 ** 40
    else:
        hi, lo = sys.float_info.max / 2.0 ** 40, sys.float_info.min * 2.0 ** 40
    running = 1.0
    for v in values:
        running *= v
        if math.isnan(running) or math.isinf(running) or running == 0.0 or abs(running) > hi or abs(running) < lo:
            return True
    return False


def _contains_non_ascii_digit(src):
    """A code point of category Nd outside ASCII, which ICU's `\\d` matches and RE2's does not.
    Taken over the distinct values, which is a handful whatever the row count."""
    import unicodedata
    for value in pc.unique(src).to_pylist():
        for ch in value or "":
            if ord(ch) > 127 and unicodedata.category(ch) == "Nd":
                return True
    return False


def _anchor_can_match_twice(src):
    """A row where an anchored pattern matches more than once under RE2's repeated search: RE2
    re-anchors `^` at the start of each search where ICU anchors it at the start of the input."""
    counts = pc.count_substring_regex(src, "^a")
    return bool(pc.max(counts).as_py() or 0) and (pc.max(counts).as_py() or 0) > 1


def _has_nulls(src):
    return src.null_count > 0


def _float_text_differs(src):
    """Arrow's own rendering of the column, compared with what Swift's `description` produces for the
    same value. Only the two formatters' *shapes* are reproduced here -- an integral value keeping its
    `.0`, a two-digit exponent, and the exponent threshold -- so a column of values both spell the
    same way still has to agree."""
    for value in pc.unique(src).to_pylist():
        if value is None or not math.isfinite(value):
            continue
        if value == int(value) and abs(value) < 1e16:
            return True                                  # Swift writes 1.0, Arrow writes 1
        exponent = math.floor(math.log10(abs(value))) if value else 0
        if abs(value) and (exponent < -4 or exponent >= 16):
            return True                                  # the two disagree on when to go scientific
    return False


def _decimal_rounding_carries(src):
    """A value whose rounding to one of the target scales carries into a digit the narrowed precision
    no longer has, which is where ArrowMetal keeps the value and Arrow's unchecked cast wraps."""
    precision, scale = src.type.precision, src.type.scale
    for target in _decimal_round_targets(precision, scale):
        if target >= scale:
            continue
        width = precision - (scale - target)
        limit = 10 ** width
        for value in src.to_pylist():
            if value is None:
                continue
            unscaled = _unscaled_magnitude(value)
            rounded = (unscaled + 10 ** (scale - target) // 2) // 10 ** (scale - target)
            if rounded >= limit:
                return True
    return False


def _contains_negative_zero_or_subnormal(src):
    tiny = float(np.finfo(NUMPY_TYPE[type_name_of(src)]).tiny)
    for value in src.to_pylist():
        if value is None or math.isnan(value):
            continue
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return True
        if 0.0 < abs(value) < tiny:
            return True
    return False


#: Above this magnitude the software binary64 sin/cos/tan lose their argument reduction; measured at
#: 2^49, exact below it and NaN from about 2^61.
_TRIG_REDUCTION_LIMIT = 2.0 ** 49


def _needs_large_argument_reduction(src):
    """A finite float64 argument at or above the reduction limit. ±inf does not count: both engines
    answer NaN for it, which the pinned tests assert."""
    if type_name_of(src) != "float64":
        return False
    filled = pc.fill_null(src, pa.scalar(0.0, src.type))
    finite = filled.filter(pc.is_finite(filled))
    biggest = pc.max(pc.abs(finite)).as_py() if len(finite) else None
    return biggest is not None and biggest >= _TRIG_REDUCTION_LIMIT


def _moments_leave_the_accumulator(src):
    """The sum of squares of a 64-bit integer column past 2^63, where the variance kernel's own
    accumulator gives up and answers NaN."""
    if type_name_of(src) not in ("int64", "uint64"):
        return False
    for value in src.to_pylist():
        if value is not None and abs(int(value)) > 2 ** 40:
            return True
    return False


def _ends_with_a_whitespace_run(src):
    """A value whose trailing whitespace run is two or more characters long -- the one place the
    Unicode splits still disagree (`" "` gives `['', '']` in both; `"  "` gives three pieces in Arrow's
    utf8_split_whitespace, two in its ascii_split_whitespace and here)."""
    for value in pc.unique(src).to_pylist():
        if value is not None and len(value) >= 2 and value[-1:].isspace() and value[-2:-1].isspace():
            return True
    return False


#: pyarrow's bundled timezone database stops applying a DST rule after the 32-bit epoch runs out.
_TZ_CUTOFF_SECONDS = 2 ** 31 - 1


def _after_the_2038_cutoff(src):
    ticks = _raw_ticks(src)
    per_second = TICKS[src.type.unit] if pa.types.is_timestamp(src.type) else 1
    biggest = pc.max(ticks).as_py()
    return biggest is not None and biggest > _TZ_CUTOFF_SECONDS * per_second


FINDINGS = [
    Finding("float32-subnormal-ftz",
            "Float32 arithmetic flushes subnormal results and operands to zero",
            ["arith_scalar", "arith_array", "pairwise_diff", "trig", "trig_checked",
             "arith_checked", "math_extra", "round_extra"],
            ["float32"], flavors={"special"}),
    Finding("sign-of-negative-zero",
            "sign keeps the sign of -0.0 where pyarrow normalises it to 0.0",
            ["sign"], FLOATING, data_check=_contains_negative_zero),
    Finding("negative-zero-set-lookup",
            "is_in/index_in/count_distinct match -0.0 with 0.0, the total order unique/sort use; "
            "Arrow keeps them apart",
            ["is_in", "index_in", "mode_and_count_distinct"], FLOATING,
            data_check=_contains_negative_zero),
    Finding("cumulative-prod-reassociation",
            "cumulative_prod is a parallel scan; once a running product overflows or underflows, which "
            "intermediates become inf, 0 or NaN depends on the multiplication order",
            ["cumulative_prod"], FLOATING, data_check=_prefix_product_leaves_safe_range),

    # ---- findings 6 onwards: the kernels added after the first round.
    Finding("decimal-to-float64-divides",
            "decimal -> float64 divides the unscaled value by 10^scale (correctly rounded); Arrow "
            "multiplies by the reciprocal, which is a ulp out",
            ["decimal_to_float64"], DECIMAL128_TYPES),
    Finding("decimal-round-carry-past-the-precision",
            "rounding a decimal to a smaller scale narrows the precision by the digits dropped; a "
            "value whose rounding carries keeps its full value here and wraps in Arrow's cast",
            ["decimal_round"], DECIMAL128_TYPES, data_check=_decimal_rounding_carries),
    Finding("regex-icu-unicode-classes",
            "ICU's \\d, \\w and \\s are Unicode-aware and RE2's are ASCII, so a full-width digit "
            "matches here and not in pyarrow",
            ["regex_match", "regex_replace"], ["utf8"], data_check=_contains_non_ascii_digit),
    Finding("regex-anchor-in-a-repeated-search",
            "in a repeated search ICU anchors ^ at the start of the input and RE2 at the start of "
            "each search, so ^a matches once here and twice there on \"aa\"",
            ["regex_match"], ["utf8"], data_check=_anchor_can_match_twice),
    Finding("split-loses-the-null-row",
            "split returns an (offsets, values) pair with nowhere to put a null row, so a null value "
            "splits to an empty list where Arrow's list<utf8> keeps the null",
            ["regex_split", "split_whitespace"], ["utf8"], data_check=_has_nulls),
    Finding("float-text-swift-format",
            "float -> utf8 uses Swift's formatting: an integral value keeps its .0, the exponent has "
            "two digits and the switch to scientific notation happens at a different magnitude",
            ["to_strings_text"], FLOATING, data_check=_float_text_differs),
    Finding("temporal-extract-in-utc",
            "the temporal kernels read a zoned timestamp in UTC; pyarrow reads it in the column's own "
            "timezone",
            ["temporal_calendar", "temporal_clock", "temporal_struct", "temporal_week_options",
             "temporal_round", "temporal_round_calendar", "temporal_ceil_calendar",
             "temporal_round_finer", "temporal_round_unaligned", "temporal_between",
             "temporal_between_clock", "weeks_between", "months_between", "interval_between",
             "interval_layouts", "strftime", "strftime_seconds", "add_interval", "cast_unit"],
            TIMESTAMP_TZ),
    Finding("strftime-seconds-carry-the-fraction",
            "%S prints whole seconds here, as C's strftime does; Arrow appends the sub-second digits",
            ["strftime_seconds"], TIMESTAMP_TYPES),
    Finding("timezone-after-2038",
            "pyarrow's timezone database stops applying a DST rule after the 32-bit epoch; Foundation "
            "keeps applying it, so the two disagree on every summer instant past 2038",
            ["assume_timezone", "temporal_timezone"], TIMESTAMP_TYPES,
            data_check=_after_the_2038_cutoff),
    Finding("trig-argument-reduction",
            "the software binary64 sin/cos/tan lose their argument reduction past 2^49 and return NaN "
            "past about 2^61",
            ["trig", "trig_checked"], ["float64"], data_check=_needs_large_argument_reduction),
    Finding("split-whitespace-trailing-run",
            "a trailing run of two or more whitespace characters yields one empty piece here and two "
            "in Arrow's utf8_split_whitespace (its ascii_split_whitespace gives one, as here)",
            ["split_whitespace"], ["utf8"],
            data_check=_ends_with_a_whitespace_run),
    Finding("variance-accumulator-overflow",
            "variance and stddev accumulate the two moments in a 64-bit accumulator, so a column of "
            "64-bit integers past 2^40 answers with a wrapped number or NaN where pyarrow answers "
            "in double",
            ["variance_and_stddev"], ["int64", "uint64"], data_check=_moments_leave_the_accumulator),
    Finding("rolling-min-max-zero-and-subnormal",
            "the rolling min/max scan compares raw values rather than the canonical sort keys, so a "
            "±0 tie is broken by position and a float32 subnormal reads as zero",
            ["rolling_min_max"], FLOATING, data_check=_contains_negative_zero_or_subnormal),
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

#: MetalArray/GroupBy members that no single pyarrow.compute call corresponds to: the accessors, and
#: the three generic dispatchers whose every op is reached through a named form below.
_NOT_DIFFERENTIABLE = {"from_arrow", "to_arrow", "null_count", "format", "type", "group_by",
                       "unary", "binary", "cumulative",
                       # the later generic dispatchers: every op of each is reached by name below
                       "window", "string_predicate", "string_transform"}

#: Methods with no case in this matrix because the generator makes no array they apply to.
#: Empty since the generator learned the temporal, decimal and nested types: every method on
#: MetalArray now has a case. test_methods_outside_the_matrix_are_reported prints the list, so a
#: method that has to be parked here again stays visible.
_NO_MATRIX_TYPE = {}

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
    "and_kleene": "kleene", "or_kleene": "kleene",
    "floor": "rounding", "ceil": "rounding", "round": "rounding", "trunc": "rounding",
    "ln": "logarithm", "log10": "logarithm", "log2": "logarithm",
    "min_element_wise": "element_wise_min_max", "max_element_wise": "element_wise_min_max",
    "pad_left": "pad", "pad_right": "pad",
    "count_substring": "substring_search", "find_substring": "substring_search",
    "decode": "dictionary_decode",

    # ---- the extended matrix.
    "decimal_add": "decimal_arith", "decimal_sub": "decimal_arith", "decimal_mul": "decimal_arith",
    "to_float64": "decimal_to_float64",
    "to_decimal128": "decimal_widen_narrow", "to_small_decimal": "decimal_widen_narrow",
    "list_value_length": "list_length",
    "child": "struct_field", "child_count": "struct_field",
    "match_substring_regex": "regex_match", "count_substring_regex": "regex_match",
    "find_substring_regex": "regex_match", "replace_substring_regex": "regex_replace",
    "split_pattern": "regex_split", "extract_regex": "regex_extract",
    "extract_regex_span": "regex_extract_span",
    "ascii_is_printable": "string_predicates", "ascii_is_title": "string_predicates",
    "string_is_ascii": "string_predicates", "utf8_is_alnum": "string_predicates",
    "utf8_is_alpha": "string_predicates", "utf8_is_decimal": "string_predicates",
    "utf8_is_digit": "string_predicates", "utf8_is_lower": "string_predicates",
    "utf8_is_numeric": "string_predicates", "utf8_is_printable": "string_predicates",
    "utf8_is_space": "string_predicates", "utf8_is_title": "string_predicates",
    "utf8_is_upper": "string_predicates",
    "ascii_title": "string_case_transforms", "utf8_capitalize": "string_case_transforms",
    "utf8_title": "string_case_transforms",
    "utf8_center": "string_pad_and_slice", "utf8_replace_slice": "string_pad_and_slice",
    "binary_replace_slice": "string_pad_and_slice",
    "utf8_trim": "string_trim", "utf8_ltrim": "string_trim", "utf8_rtrim": "string_trim",
    "utf8_normalize": "string_normalize",
    "parse": "parse_numbers",
    "year_month_day": "temporal_struct", "iso_calendar": "temporal_struct",
    "year": "temporal_calendar", "month": "temporal_calendar", "day": "temporal_calendar",
    "hour": "temporal_clock", "minute": "temporal_clock", "second": "temporal_clock",
    "day_of_week": "temporal_week_options", "cast_unit": "cast_unit",
    "quarter": "temporal_calendar", "day_of_year": "temporal_calendar",
    "iso_week": "temporal_calendar", "iso_year": "temporal_calendar",
    "is_leap_year": "temporal_calendar", "us_week": "temporal_calendar",
    "us_year": "temporal_calendar",
    "millisecond": "temporal_clock", "microsecond": "temporal_clock",
    "nanosecond": "temporal_clock", "subsecond": "temporal_clock",
    "week": "temporal_week_options",
    "is_dst": "temporal_timezone", "local_timestamp": "temporal_timezone",
    "floor_temporal": "temporal_round", "ceil_temporal": "temporal_round",
    "round_temporal": "temporal_round",
    "years_between": "temporal_between", "quarters_between": "temporal_between",
    "days_between": "temporal_between", "hours_between": "temporal_between",
    "minutes_between": "temporal_between", "seconds_between": "temporal_between",
    "milliseconds_between": "temporal_between", "microseconds_between": "temporal_between",
    "nanoseconds_between": "temporal_between",
    "month_day_nano_interval_between": "interval_between",
    "month_interval_between": "interval_layouts",
    "day_time_interval_between": "interval_layouts", "interval_field": "interval_layouts",
    "row_number": "ranking", "rank": "ranking", "dense_rank": "ranking",
    "percent_rank": "percent_rank_and_cume_dist", "cume_dist": "percent_rank_and_cume_dist",
    "rolling_min": "rolling_min_max", "rolling_max": "rolling_min_max",
    "variance": "variance_and_stddev", "stddev": "variance_and_stddev",
    "median": "quantile",
    "mode": "mode_and_count_distinct", "count_distinct": "mode_and_count_distinct",
    "first": "first_last_index_min_max", "last": "first_last_index_min_max",
    "index": "first_last_index_min_max", "min_max": "first_last_index_min_max",
    "any": "any_all", "all": "any_all",
    "run_end_encode": "run_end", "run_end_decode": "run_end",
    "to_float32": "float16_casts", "to_float16": "float16_casts",
    "extension_name": "extension_type", "extension_metadata": "extension_type",
    "extension_storage": "extension_type", "as_extension_type": "extension_type",
    "sin": "trig", "cos": "trig", "tan": "trig", "asin": "trig", "acos": "trig", "atan": "trig",
    "sinh": "trig", "cosh": "trig", "tanh": "trig", "asinh": "trig", "acosh": "trig",
    "atanh": "trig", "atan2": "trig",
    "sin_checked": "trig_checked", "cos_checked": "trig_checked", "tan_checked": "trig_checked",
    "asin_checked": "trig_checked", "acos_checked": "trig_checked",
    "acosh_checked": "trig_checked", "atanh_checked": "trig_checked",
    "xor": "logical_extras", "and_not": "logical_extras", "and_not_kleene": "logical_extras",
    "is_nan": "float_class", "is_inf": "float_class", "is_finite": "float_class",
    "fill_null_forward": "fill_null_direction", "fill_null_backward": "fill_null_direction",

    # ---- the Arrow-named surface: the checked family, the extra math, the associative transforms,
    # the selection and permutation kernels, the statistical aggregates and the byte-indexed strings.
    "add_checked": "arith_checked", "subtract_checked": "arith_checked",
    "multiply_checked": "arith_checked", "divide_checked": "arith_checked",
    "binary_checked": "arith_checked",
    "shift_left_checked": "shift_checked", "shift_right_checked": "shift_checked",
    "abs_checked": "unary_checked", "negate_checked": "unary_checked",
    "sqrt_checked": "log_checked", "ln_checked": "log_checked", "log10_checked": "log_checked",
    "log2_checked": "log_checked", "log1p_checked": "log_checked",
    "cumulative_sum_checked": "cumulative_checked", "cumulative_prod_checked": "cumulative_checked",
    "pairwise_diff_checked": "cumulative_checked",
    "expm1": "math_extra", "log1p": "math_extra", "logb": "math_extra", "hypot": "math_extra",
    "logb_checked": "math_extra",
    "round_to_multiple": "round_extra", "round_binary": "round_extra",
    "array_sort_indices": "sort_indices",
    "array_filter": "array_selection", "array_take": "array_selection",
    "top_k_unstable": "select_k_unstable", "bottom_k_unstable": "select_k_unstable",
    "inverse_permutation": "permutation",
    "skew": "skew_kurtosis", "kurtosis": "skew_kurtosis",
    "rank_quantile": "rank_quantile_and_normal", "rank_normal": "rank_quantile_and_normal",
    "count_all": "true_unless_null",
    "ascii_lpad": "ascii_pad", "ascii_rpad": "ascii_pad", "ascii_center": "ascii_pad",
    "utf8_lpad": "ascii_pad", "utf8_rpad": "ascii_pad",
    "utf8_swapcase": "swapcase_and_zero_fill", "ascii_swapcase": "swapcase_and_zero_fill",
    "utf8_zero_fill": "swapcase_and_zero_fill",
    "binary_slice": "byte_transforms", "binary_reverse": "byte_transforms",
    "ascii_reverse": "byte_transforms",
    "extract_regex_struct": "extract_regex_structs",
    "extract_regex_span_struct": "extract_regex_structs",
    "split_pattern_pair": "split_pairs", "split_whitespace_pair": "split_pairs",
    "to_timezone": "timezone_metadata", "utc_offset": "timezone_metadata",
}


#: Module-level functions that compute something, and the operation that covers each. The rest of the
#: module (`array`, `version`, `device_name`, `batch`, `MetalArray`, `GroupBy`, `ArrowMetalError`) is
#: construction and plumbing rather than a kernel.
_MODULE_LEVEL = {"coalesce": "coalesce", "case_when": "case_when", "choose": "choose",
                 "lexsort_indices": "lexsort", "nulls": "nulls_constructor"}


def public_api():
    return {n for n in dir(am.MetalArray) if not n.startswith("_")} | \
           {n for n in dir(am.GroupBy) if not n.startswith("_")}


def test_every_public_operation_has_a_differential_case():
    """Fails when a method is added to MetalArray or GroupBy without a case here, so the harness
    cannot silently fall behind the library."""
    known = _NOT_DIFFERENTIABLE | set(_COVERED_BY) | set(_NO_MATRIX_TYPE) | \
        {o.name for o in OPS} | {n for n, _ in ABSENT}
    missing = sorted(n for n in public_api() if n not in known)
    assert not missing, (
        "no differential case for: " + ", ".join(missing) +
        " -- add an Op, or an entry in _COVERED_BY / _NO_MATRIX_TYPE / _NOT_DIFFERENTIABLE, "
        "in python/tests/test_differential.py")


def test_every_module_level_function_has_a_differential_case():
    """The same rule for the functions that are not methods: coalesce, case_when, choose,
    lexsort_indices and nulls."""
    registered = {o.name for o in OPS}
    for name, operation in sorted(_MODULE_LEVEL.items()):
        assert hasattr(am, name), f"arrowmetal.{name} no longer exists"
        assert operation in registered, f"{name} claims to be covered by a missing op {operation}"


def test_methods_outside_the_matrix_are_reported():
    """Not a failure: a record of the methods the generator cannot reach, and why."""
    for name, reason in sorted(_NO_MATRIX_TYPE.items()):
        print(f"outside the matrix: {name} (needs a {reason})")
    assert all(hasattr(am.MetalArray, n) for n in _NO_MATRIX_TYPE), \
        "_NO_MATRIX_TYPE lists a method MetalArray no longer has"


def test_no_oracle_operations_are_reported():
    """Not a failure: the operations pyarrow.compute has no function for, each with the reference
    this file compares against instead. Printed on every run so the list cannot grow unnoticed."""
    registered = {o.name for o in OPS}
    for name, why in sorted(_NO_ORACLE.items()):
        print(f"no pyarrow oracle: {name} -- {why}")
        assert name in registered, f"_NO_ORACLE names {name}, which is not a registered operation"
    assert len(_NO_ORACLE) <= 20, "more operations are drifting out of pyarrow's reach than expected"


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


def test_sign_keeps_the_column_type_where_pyarrow_narrows_to_int8():
    """`am_unary` returns the column's own type for every op it defines, `sign` included; `pc.sign`
    narrows an integer column to int8. The values agree, so the matrix casts the oracle back."""
    a = pa.array([-5, 0, 7], pa.int32())
    assert am.array(a).sign().type == pa.int32() and pc.sign(a).type == pa.int8()
    assert pylist(am.array(a).sign()) == pc.sign(a).to_pylist() == [-1, 0, 1]
    f = pa.array([-2.5, 0.0, 3.0], pa.float64())
    assert am.array(f).sign().type == pc.sign(f).type == pa.float64()


@pytest.mark.parametrize("method,fn", [("shift_left", pc.shift_left), ("shift_right", pc.shift_right)])
def test_out_of_range_shift_counts_shift_the_bits_out_where_pyarrow_returns_the_operand(method, fn):
    """A count outside [0, bit width) shifts every bit out on the GPU -- 0, or the sign fill for a
    signed `shift_right`. pyarrow's unchecked kernel returns the operand untouched, and its checked
    variant raises. Arrow's range excludes the sign bit, so int32 already diverges at 31."""
    a = pa.array([-8, -8, 1024, 1024], pa.int32())
    counts = pa.array([-1, 40, 31, 32], pa.int32())
    got = pylist(getattr(am.array(a), method)(am.array(counts)))
    assert got == ([0, 0, 0, 0] if method == "shift_left" else [-1, -1, 0, 0])
    assert fn(a, counts).to_pylist() == [-8, -8, 1024, 1024]
    with pytest.raises(pa.ArrowInvalid):
        getattr(pc, method + "_checked")(a, counts)


def test_cumulative_functions_skip_nulls_where_arrow_propagates_them():
    """ArrowMetal's cumulative kernels carry the running value across a null and leave the output
    null exactly where the input is -- Arrow's `skip_nulls=True`. Arrow's default is the other one:
    the first null poisons every later element. ArrowMetal has no such mode."""
    a = pa.array([1, 2, None, 4], pa.int64())
    assert pylist(am.array(a).cumulative_sum()) == \
        pc.cumulative_sum(a, skip_nulls=True).to_pylist() == [1, 3, None, 7]
    assert pc.cumulative_sum(a).to_pylist() == [1, 3, None, None]
    assert pylist(am.array(a).cumulative_max()) == \
        pc.cumulative_max(a, skip_nulls=True).to_pylist() == [1, 2, None, 4]


def test_arrow_cumulative_max_default_start_clamps_negative_floats():
    """Why the matrix passes `start` explicitly: Arrow's default start for `cumulative_max` is the
    type's `numeric_limits::min()`, which on a float column is the smallest positive normal, so every
    negative running maximum comes back clamped to 1.18e-38."""
    a = pa.array([-1.0, -2.0], pa.float32())
    assert pc.cumulative_max(a).to_pylist() == [float(np.finfo(np.float32).tiny)] * 2
    assert pc.cumulative_max(a, start=pa.scalar(-math.inf, pa.float32())).to_pylist() == [-1.0, -1.0]
    assert pylist(am.array(a).cumulative_max()) == [-1.0, -1.0]


def test_float64_transcendentals_are_true_binary64():
    """`sqrt`/`exp`/`ln`/`log10`/`log2`/`power` on a float64 column run in software IEEE-754 binary64
    on the GPU (Kernels/DoubleTranscendental.swift): the whole double range, sqrt correctly rounded,
    the rest within 1 ulp of libm. Until 2026-09-06 they evaluated in `float` and lost everything
    outside the float32 range; this pins the fix."""
    a = pa.array([1e-300, 1e300, 2.0], pa.float64())
    assert pylist(am.array(a).sqrt()) == pc.sqrt(a).to_pylist() == [1e-150, 1e150, math.sqrt(2.0)]
    b = pa.array([-700.0, 700.0, 1e-10, 1e300], pa.float64())
    for name in ("exp", "ln", "log10", "log2"):
        got = pylist(getattr(am.array(b), name)())
        want = getattr(pc, name)(b).to_pylist()
        for g, w in zip(got, want):
            if math.isfinite(w) and w != 0.0:
                assert abs(g - w) <= 2 * abs(w) * _EPS["float64"], (name, g, w)
            else:
                assert g == w or (math.isnan(g) and math.isnan(w)), (name, g, w)


def test_floor_and_ceil_keep_the_integer_type_where_pyarrow_widens_to_double():
    """`pc.floor`/`ceil`/`trunc` return a double for an integer column -- and refuse the column
    outright past 2^53, where a double can no longer hold it. ArrowMetal's are the identity and keep
    the column's type, so they answer at every magnitude. That is why the matrix compares them on
    integer values within 2^53."""
    small = pa.array([-3, 7], pa.int64())
    assert am.array(small).floor().type == pa.int64() and pc.floor(small).type == pa.float64()
    assert pylist(am.array(small).floor()) == [-3, 7] == \
        [int(v) for v in pc.floor(small).to_pylist()]
    big = pa.array([2 ** 62 + 1], pa.int64())
    assert pylist(am.array(big).floor()) == [2 ** 62 + 1]
    with pytest.raises(pa.ArrowInvalid):
        pc.floor(big)


def test_empty_replace_pattern_is_the_identity():
    """ArrowMetal defines an empty pattern as the identity. `pc.replace_substring` is not compared
    here at all: on an empty pattern Arrow's kernel does not terminate, so the harness must never
    call it with one -- which is also why the matrix's replace patterns are all non-empty."""
    a = pa.array(["abc", "", None], pa.string())
    assert pylist(am.array(a).replace("", "-")) == ["abc", "", None]


def test_empty_pattern_counts_code_points_where_arrow_counts_bytes():
    """`count_substring("")` is a length + 1 in both engines -- of code points here, of bytes in
    Arrow, so the two only differ on multi-byte input."""
    a = pa.array(["héllo", "abc"], pa.string())
    assert pylist(am.array(a).count_substring("")) == [6, 4]
    assert pc.count_substring(a, "").to_pylist() == [7, 4]


def test_element_wise_min_max_break_a_zero_tie_like_fmin_and_fmax_in_both():
    """-0.0 and 0.0 are equal, so which one comes back is a tie-break: both engines answer the way
    `fmin`/`fmax` do, so the result does not depend on the operand order."""
    for ty in (pa.float32(), pa.float64()):
        a, b = pa.array([0.0, -0.0], ty), pa.array([-0.0, 0.0], ty)
        assert pylist(am.array(a).min_element_wise(am.array(b))) == \
            pc.min_element_wise(a, b).to_pylist() == [-0.0, -0.0]
        assert pylist(am.array(a).max_element_wise(am.array(b))) == \
            pc.max_element_wise(a, b).to_pylist() == [0.0, 0.0]
        assert struct.pack("<d", pylist(am.array(a).min_element_wise(am.array(b)))[0]) == \
            struct.pack("<d", -0.0)


def test_is_in_never_matches_a_null_where_arrow_matches_null_to_null():
    """ArrowMetal ignores nulls in the value set and never matches a null element, so `is_in` never
    returns null -- Arrow's `skip_nulls=True`. Arrow's default makes null match null."""
    a = pa.array([1, None], pa.int32())
    values = am.array(pa.array([1, None], pa.int32()))
    assert pylist(am.array(a).is_in(values)) == \
        pc.is_in(a, value_set=pa.array([1, None], pa.int32()), skip_nulls=True).to_pylist() == \
        [True, False]
    assert pc.is_in(a, value_set=pa.array([1, None], pa.int32())).to_pylist() == [True, True]


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


def test_argsort_null_block_is_stable():
    """Was a finding: the null indices used to come back ordered by the bytes under the validity
    bitmap. Two arrays Arrow calls equal must argsort identically, and like Arrow."""
    loud, quiet = _payload_pair()
    assert pylist(am.array(loud).argsort()) == pylist(am.array(quiet).argsort())
    assert pylist(am.array(loud).argsort()) == pc.array_sort_indices(loud).to_pylist()


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


def test_float32_comparison_distinguishes_subnormals_from_zero():
    """Only the *arithmetic* kernels flush: the comparison kernels compare float32 exactly, so a
    subnormal is greater than zero and different from -0.0, as in Arrow."""
    smallest = float(np.finfo(np.float32).tiny) * 2.0 ** -23     # 1.4e-45
    a = pa.array([smallest], pa.float32())
    assert pylist(am.array(a).compare(">", 0.0)) == \
        pc.greater(a, pa.scalar(0.0, pa.float32())).to_pylist() == [True]
    assert pylist(am.array(pa.array([-0.0], pa.float32())).compare("==", am.array(a))) == [False]


def test_sign_of_a_float32_subnormal_is_one():
    """The sign kernel decides on the bit pattern, so it does not inherit the arithmetic flush."""
    smallest = float(np.finfo(np.float32).tiny) * 2.0 ** -23
    a = pa.array([smallest, -smallest], pa.float32())
    assert pylist(am.array(a).sign()) == pc.sign(a).to_pylist() == [1.0, -1.0]


def test_float64_arithmetic_and_comparison_keep_subnormals():
    """The Float64 software path is not affected: only the Float32 kernels flush."""
    smallest = 5e-324
    a = pa.array([smallest, smallest * 4], pa.float64())
    assert pylist(am.array(a).arith("*", 0.5)) == \
        pc.multiply(a, pa.scalar(0.5, pa.float64())).to_pylist()
    assert pylist(am.array(a).compare(">", 0.0)) == [True, True]


@pytest.mark.parametrize("ty", [pa.float32(), pa.float64()])
def test_negative_zero_is_a_sort_tie(ty):
    """IEEE-754 calls -0.0 and 0.0 equal, so both engines leave the tie in input order -- the sort
    keys canonicalise the sign of zero rather than sorting on the raw bit pattern."""
    a = pa.array([0.0, -0.0, 0.0, -0.0], ty)
    assert pylist(am.array(a).argsort()) == pc.array_sort_indices(a).to_pylist() == [0, 1, 2, 3]
    assert pylist(am.array(a).argsort(True)) == \
        pc.array_sort_indices(a, order="descending").to_pylist() == [0, 1, 2, 3]
    assert pylist(am.array(a).top_k(4)) == [0, 1, 2, 3]


@pytest.mark.parametrize("ty", [pa.float32(), pa.float64()])
def test_nan_sorts_last_in_both_directions_in_both(ty):
    """NaN goes after +inf ascending and stays at the end descending, next to the nulls -- Arrow's
    null_placement covers NaN too, so a reversed order does not mirror it to the front."""
    a = pa.array([1.0, math.nan, math.inf, -math.inf, None], ty)
    assert pylist(am.array(a).argsort()) == pc.array_sort_indices(a).to_pylist() == [3, 0, 2, 1, 4]
    assert pylist(am.array(a).argsort(True)) == \
        pc.array_sort_indices(a, order="descending").to_pylist() == [2, 0, 3, 1, 4]
    assert pylist(am.array(a).top_k(3)) == [2, 0, 3]


def test_float32_sum_does_not_overflow_before_arrow_does():
    """Was a finding: the Float32 reductions accumulate in double now, as Arrow's do."""
    big = float(np.finfo(np.float32).max)
    a = pa.array([big, big], pa.float32())
    assert am.array(a).sum() == pc.sum(a).as_py()          # Arrow: 6.805646932770577e+38


def test_group_by_sum_over_uint64_stays_unsigned():
    """Was a finding: the grouped total used to be reported as int64 whatever the value type."""
    keys = pa.array([0, 0], pa.int32())
    values = pa.array([2 ** 63, 2 ** 63 - 5], pa.uint64())
    total = am.array(keys).group_by(1).sum(am.array(values))
    assert total.type == pa.uint64()
    assert pylist(total) == group_oracle(keys, values, 1, "sum")


@pytest.mark.xfail(strict=True, reason="sign-of-negative-zero: ArrowMetal's sign returns -0.0 for "
                                       "-0.0, where pyarrow normalises the result to 0.0")
def test_sign_of_negative_zero_matches_pyarrow():
    a = pa.array([-0.0, 0.0], pa.float64())
    assert struct.pack("<d", pylist(am.array(a).sign())[0]) == \
        struct.pack("<d", pc.sign(a).to_pylist()[0])


def test_case_mapping_covers_all_of_unicode():
    """Once a finding (utf8-case-latin-only): the GPU table covers U+0000-U+017F and every row holding
    a code point above it is mapped on the host, so Greek, Cyrillic and the rest agree with pyarrow."""
    a = pa.array(["Ωμέγα", "ΣΊΣΥΦΟΣ", "ığdır", "𐐀", "ß", "ﬁ"], pa.string())
    assert pylist(am.array(a).lower()) == pc.utf8_lower(a).to_pylist()
    assert pylist(am.array(a).upper()) == pc.utf8_upper(a).to_pylist()


def test_case_mapping_inside_latin_extended_a_matches_pyarrow():
    """The blocks ArrowMetal does cover are expected to agree exactly, byte lengths included."""
    a = pa.array(["héllo", "ÅNGSTRÖM", "naïve", "ſ", "İ", "ı", "Ÿ", "ÿ", "œŒ"], pa.string())
    assert pylist(am.array(a).upper()) == pc.utf8_upper(a).to_pylist()
    assert pylist(am.array(a).lower()) == pc.utf8_lower(a).to_pylist()


@pytest.mark.xfail(strict=True, reason="negative-zero-set-lookup: is_in maps -0.0 onto 0.0, the total "
                                       "order unique() and the sort use; Arrow's hash keeps them apart")
def test_is_in_separates_negative_zero_from_zero():
    a = pa.array([0.0, -0.0], pa.float64())
    values = pa.array([-0.0], pa.float64())
    assert pylist(am.array(a).is_in(am.array(values))) == \
        pc.is_in(a, value_set=values, skip_nulls=True).to_pylist()


@pytest.mark.xfail(strict=True, reason="cumulative-prod-reassociation: the GPU scan multiplies in a "
                                       "different order, so an overflowing and an underflowing "
                                       "intermediate can meet as inf * 0 = NaN where Arrow's sequential "
                                       "product stays inf")
def test_cumulative_prod_matches_arrow_past_overflow():
    a = pa.array([1e30, 1e30, 1e-30, 1e-30] * 8, pa.float32())
    assert pylist(am.array(a).cumulative_prod()) == pc.cumulative_prod(a).to_pylist()


def test_cumulative_prod_matches_arrow_inside_range():
    """Without overflow or underflow the reassociated product agrees to float tolerance."""
    a = pa.array([1.5, -2.0, 0.25, 3.0, None, 2.0, -1.25, 8.0], pa.float64())
    got = pylist(am.array(a).cumulative_prod())
    want = pc.cumulative_prod(a, skip_nulls=True).to_pylist()
    assert all((g is None and w is None) or abs(g - w) <= 1e-12 * max(1.0, abs(w)) for g, w in zip(got, want))


def test_is_in_matches_nan_to_nan_in_both():
    """The other float special case agrees: one NaN is in a set that holds a NaN."""
    a = pa.array([math.nan, 1.0], pa.float64())
    values = pa.array([math.nan], pa.float64())
    assert pylist(am.array(a).is_in(am.array(values))) == \
        pc.is_in(a, value_set=values, skip_nulls=True).to_pylist() == [True, False]


def test_cumulative_sum_propagates_nan_and_reassociates_infinities():
    """NaN poisons the rest of the running sum in both engines. ±inf is where the block scan and a
    left-to-right accumulation part company -- `inf - inf` is NaN, and which of the two an element
    sees depends on the association -- which is why the matrix filters infinities out."""
    a = pa.array([1.0, math.nan, 2.0], pa.float64())
    assert pylist(am.array(a).cumulative_sum())[0] == pc.cumulative_sum(a, skip_nulls=True)[0].as_py()
    assert all(math.isnan(v) for v in pylist(am.array(a).cumulative_sum())[1:])
    assert all(math.isnan(v) for v in pc.cumulative_sum(a, skip_nulls=True).to_pylist()[1:])
    b = pa.array([math.inf, -math.inf], pa.float64())
    assert math.isnan(pylist(am.array(b).cumulative_sum())[1])
    assert math.isnan(pc.cumulative_sum(b, skip_nulls=True).to_pylist()[1])


def test_argsort_is_a_permutation_with_nulls_last_and_values_in_order():
    """The weaker properties that hold whatever the values under the null bits are."""
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


# ------------------------------------------------------------------ the extended matrix's findings
#
# One reproduction per open finding added with the extended matrix, plus the pinned tests that say
# what the two engines *do* agree on around it. The xfail(strict=True) ones start passing -- and the
# suite starts failing -- the moment a kernel changes its mind.


@pytest.mark.xfail(strict=True, reason="decimal-to-float64-divides: ArrowMetal divides the unscaled "
                                       "value by 10^scale, Arrow multiplies by the reciprocal")
def test_decimal_to_float64_matches_arrows_cast():
    a = pa.array([decimal.Decimal("99.99")], pa.decimal128(9, 2))
    assert pylist(am.array(a).to_float64()) == a.cast(pa.float64()).to_pylist()


def test_decimal_to_float64_is_the_correctly_rounded_quotient():
    """The half of it that is not a matter of taste: ArrowMetal's answer is the nearest double to the
    exact decimal value, which Arrow's reciprocal multiply is a ulp away from."""
    a = pa.array([decimal.Decimal("99.99")], pa.decimal128(9, 2))
    assert pylist(am.array(a).to_float64()) == [9999 / 100]
    assert a.cast(pa.float64()).to_pylist() != [9999 / 100]


@pytest.mark.xfail(strict=True, reason="decimal-round-carry-past-the-precision: rounding up past the "
                                       "narrowed precision keeps the value here and wraps in Arrow")
def test_decimal_round_carry_wraps_like_arrows_cast():
    a = pa.array([decimal.Decimal("9999999.99")], pa.decimal128(9, 2))
    rounded = am.array(a).decimal_round(0)
    oracle = pc.round(a, ndigits=0, round_mode="half_towards_infinity")
    assert pylist(rounded) == oracle.cast(rounded.type, safe=False).to_pylist()


def test_decimal_round_narrows_the_precision_by_the_digits_it_drops():
    a = pa.array([decimal.Decimal("9999999.99"), decimal.Decimal("1.25")], pa.decimal128(9, 2))
    assert am.array(a).decimal_round(0).type == pa.decimal128(7, 0)
    assert am.array(a).decimal_round(4).type == pa.decimal128(11, 4)      # scaling up is exact
    assert pylist(am.array(a).decimal_round(4)) == [decimal.Decimal("9999999.9900"),
                                                    decimal.Decimal("1.2500")]


def test_decimal_arithmetic_wraps_modulo_two_to_the_128():
    """Why the matrix keeps the decimal operands inside their precision: past it ArrowMetal wraps in
    the 128-bit storage and Arrow's unchecked cast wraps modulo 10^precision, which is a different
    number."""
    with decimal.localcontext() as ctx:
        ctx.prec = 60
        limit = decimal.Decimal(10 ** 38 - 1).scaleb(-10)
        a = pa.array([limit], pa.decimal128(38, 10))
        total = pylist(am.array(a).decimal_add(am.array(a)))[0]
        assert int(total.scaleb(10)) == 2 * (10 ** 38 - 1) - 2 ** 128
    with pytest.raises(pa.ArrowInvalid):
        pc.add(a, a)                                 # precision 39 is out of decimal128's range


def test_decimal_negate_abs_and_sign_reach_the_decimal_kernels():
    """The dispatch added to `unary()` and `_reduce()`: a decimal column takes am_decimal_op's own
    negate, abs, sign, sum, min and max, which the primitive entry points reject."""
    a = pa.array([decimal.Decimal("-1.25"), decimal.Decimal("0.00")], pa.decimal128(9, 2))
    assert pylist(am.array(a).negate()) == pc.negate(a).to_pylist()
    assert pylist(am.array(a).abs()) == pc.abs(a).to_pylist()
    assert pylist(am.array(a).sign()) == pc.sign(a).cast(pa.int32()).to_pylist()
    assert am.array(a).sum() == pc.sum(a).as_py()
    assert am.array(a).min() == pc.min(a).as_py() and am.array(a).max() == pc.max(a).as_py()


def test_list_element_of_a_short_row_is_null_where_pyarrow_raises():
    a = pa.array([[1, 2], [], None], pa.list_(pa.int64()))
    assert pylist(am.array(a).list_element(1)) == [2, None, None]
    with pytest.raises(pa.ArrowInvalid):
        pc.list_element(a, 1)


def test_list_parent_indices_are_int32_where_pyarrow_returns_int64():
    a = pa.array([[1, 2], [3]], pa.list_(pa.int64()))
    got = am.array(a).list_parent_indices()
    assert got.type == pa.int32() and pc.list_parent_indices(a).type == pa.int64()
    assert pylist(got) == pc.list_parent_indices(a).to_pylist()


def test_run_end_null_count_is_logical():
    """pyarrow reports null_count 0 for a run-end encoded array whatever its values child says;
    ArrowMetal reports the number of null rows the column actually has."""
    encoded = pc.run_end_encode(pa.array([1, None, None], pa.int64()))
    assert encoded.null_count == 0
    assert am.array(encoded).null_count == 2


def test_run_end_selection_decodes():
    """filter, take and slice on a run-end column come back flat -- pyarrow has no kernel for the
    type at all, so there is nothing to re-encode against."""
    encoded = pc.run_end_encode(pa.array([1, 1, 2], pa.int64()))
    kept = am.array(encoded).filter(am.array(pa.array([True, False, True])))
    assert kept.type == pa.int64() and pylist(kept) == [1, 2]
    with pytest.raises(pa.ArrowNotImplementedError):
        pc.filter(encoded, pa.array([True, False, True]))


@pytest.mark.xfail(strict=True, reason="regex-icu-unicode-classes: ICU's backslash-d matches a "
                                       "full-width digit, RE2's does not")
def test_regex_digit_class_is_ascii_only():
    a = pa.array(["２０２４"], pa.string())
    assert pylist(am.array(a).match_substring_regex(r"\d")) == \
        pc.match_substring_regex(a, r"\d").to_pylist()


def test_regex_digit_class_agrees_on_ascii():
    a = pa.array(["2024", "abc", ""], pa.string())
    assert pylist(am.array(a).match_substring_regex(r"\d")) == \
        pc.match_substring_regex(a, r"\d").to_pylist() == [True, False, False]


@pytest.mark.xfail(strict=True, reason="regex-anchor-in-a-repeated-search: ICU anchors ^ at the "
                                       "start of the input, RE2 at the start of each search")
def test_anchored_pattern_counts_once_per_input():
    a = pa.array(["aa"], pa.string())
    assert pylist(am.array(a).count_substring_regex("^a")) == \
        pc.count_substring_regex(a, "^a").cast(pa.int32()).to_pylist()


def test_regex_replacement_template_is_icu_not_re2():
    """ArrowMetal's replacement is an ICU template, so a capture group is `$1`; pyarrow's is RE2's
    `\\1`. Each engine copies the other's spelling through as literal text."""
    a = pa.array(["ap"], pa.string())
    assert pylist(am.array(a).replace_substring_regex("(a)(p)", "$2$1")) == ["pa"]
    assert pc.replace_substring_regex(a, "(a)(p)", r"\2\1").to_pylist() == ["pa"]
    assert pylist(am.array(a).replace_substring_regex("(a)(p)", r"\2\1")) == ["21"]


@pytest.mark.xfail(strict=True, reason="split-loses-the-null-row: the (offsets, values) pair has "
                                       "nowhere to put a null row")
def test_split_keeps_the_null_row():
    a = pa.array(["a b", None], pa.string())
    offsets, values = am.array(a).split_pattern(" ")
    got = pa.ListArray.from_arrays(offsets.to_arrow(), values.to_arrow())
    assert got.to_pylist() == pc.split_pattern(a, " ").to_pylist()


@pytest.mark.xfail(strict=True, reason="split-whitespace-trailing-run: a trailing run of two or more "
                                       "whitespace characters is one empty piece here, two in Arrow's "
                                       "utf8_split_whitespace")
def test_split_whitespace_trailing_run_matches_arrow():
    a = pa.array(["padded  "], pa.string())
    got = _split_result(am.array(a).split_whitespace(unicode=True))
    assert got.to_pylist() == pc.utf8_split_whitespace(a).to_pylist()     # [['padded', '', '']]


def test_ascii_split_whitespace_trailing_run_matches_arrows_ascii_variant():
    """Arrow's two whitespace splits disagree with each other on a trailing run; ArrowMetal gives one
    empty piece in both modes, which is what pc.ascii_split_whitespace does."""
    a = pa.array(["padded  ", "  "], pa.string())
    got = _split_result(am.array(a).split_whitespace())
    assert got.to_pylist() == pc.ascii_split_whitespace(a).to_pylist() == [["padded", ""], ["", ""]]


def test_split_whitespace_agrees_with_arrow_away_from_a_trailing_run():
    """Leading runs, inner runs, a single trailing character, empty and all-blank one-character
    values, and Unicode whitespace under unicode=True all match Arrow piece for piece."""
    a = pa.array(["  padded", "a\t  b", "y ", "", " ", "x\u3000y", "a\xa0b", None], pa.string())
    got = _split_result(am.array(a).split_whitespace(unicode=True))
    assert got.to_pylist() == pc.utf8_split_whitespace(a).to_pylist()
    got = _split_result(am.array(a).split_whitespace())
    assert got.to_pylist() == pc.ascii_split_whitespace(a).to_pylist()


def test_pyarrow_utf8_normalize_ignores_its_form_option():
    """Not our divergence: pyarrow 25.0.1 decomposes whatever `form` says, so the matrix's oracle for
    utf8_normalize is Python's unicodedata. ArrowMetal agrees with Python."""
    import unicodedata
    composed = "héllo"
    a = pa.array([composed], pa.string())
    assert pc.utf8_normalize(a, form="NFC").to_pylist() == ["héllo"]
    assert unicodedata.normalize("NFC", composed) == composed
    assert pylist(am.array(a).utf8_normalize("NFC")) == [composed]


@pytest.mark.xfail(strict=True, reason="float-text-swift-format: Swift keeps the .0 and switches to "
                                       "scientific notation at a different magnitude")
def test_float_to_text_matches_arrows_formatter():
    a = pa.array([1.0, 1.1786107e-06], pa.float64())
    assert pylist(am.array(a).to_strings()) == a.cast(pa.string()).to_pylist()


def test_float_to_text_names_the_same_number():
    """The property the formatting difference must not touch: every digit string ArrowMetal prints
    reads back as exactly the float that went in, `-0.0` and the subnormals included."""
    a = pa.array([1.0, -0.0, 5e-324, 1.7976931348623157e308, 1.1786107e-06], pa.float64())
    assert [float(s) for s in pylist(am.array(a).to_strings())] == a.to_pylist()
    assert struct.pack("<d", float(pylist(am.array(a).to_strings())[1])) == struct.pack("<d", -0.0)


def test_parse_returns_null_where_pyarrow_raises():
    a = pa.array(["1", "x", " 8", ""], pa.string())
    assert pylist(am.array(a).parse("int64")) == [1, None, None, None]
    with pytest.raises(pa.ArrowInvalid):
        a.cast(pa.int64(), safe=False)


@pytest.mark.xfail(strict=True, reason="temporal-extract-in-utc: ArrowMetal reads a zoned timestamp "
                                       "in UTC, pyarrow in the column's own timezone")
def test_temporal_extract_uses_the_columns_timezone():
    a = pa.array([0], pa.timestamp("s", "America/New_York"))
    assert pylist(am.array(a).hour()) == pc.hour(a).cast(pa.int32()).to_pylist()


def test_temporal_extract_is_arrows_answer_for_the_same_naive_instant():
    """The zone is the whole of the difference: on the same instants without a zone the two agree."""
    a = pa.array([0, 1_700_000_000], pa.timestamp("s", "America/New_York"))
    naive = a.cast(pa.timestamp("s"))
    assert pylist(am.array(a).hour()) == pc.hour(naive).cast(pa.int32()).to_pylist()
    assert pylist(am.array(a).year()) == pc.year(naive).cast(pa.int32()).to_pylist()


def test_temporal_extractors_return_int32_where_pyarrow_returns_int64():
    a = pa.array([0], pa.timestamp("s"))
    assert am.array(a).year().type == pa.int32() and pc.year(a).type == pa.int64()
    assert am.array(a).us_week().type == pc.us_week(a).type == pa.int64()
    assert am.array(a).subsecond().type == pc.subsecond(a).type == pa.float64()


def test_ceil_temporal_advances_a_value_on_a_month_boundary():
    """Once a finding (temporal-ceil-on-a-calendar-boundary): a value already on a month boundary is
    now advanced a whole unit, as Arrow's calendar units do."""
    a = pa.array([0, 951782400], pa.timestamp("s"))
    assert pylist(am.array(a).ceil_temporal("month")) == \
        pc.ceil_temporal(a, unit="month").to_pylist()


def test_ceil_temporal_keeps_a_value_on_a_fixed_length_boundary_in_both():
    """Arrow's own calendar and fixed-length units disagree with each other here: a value exactly on
    a day boundary is left alone by both engines, one exactly on a month boundary is not."""
    a = pa.array([0], pa.timestamp("s"))
    assert pylist(am.array(a).ceil_temporal("day")) == pc.ceil_temporal(a, unit="day").to_pylist()
    assert pc.ceil_temporal(a, unit="month").to_pylist() != a.to_pylist()


def test_calendar_multiples_share_an_origin():
    """Once a finding (temporal-calendar-multiple-origin): months and quarters now count from 1970-01,
    as Arrow's do."""
    a = pa.array([0, 1_700_000_000, -86400 * 400], pa.timestamp("s"))
    assert pylist(am.array(a).floor_temporal("month", 7)) == \
        pc.floor_temporal(a, multiple=7, unit="month").to_pylist()


def test_calendar_multiples_agree_wherever_the_two_origins_do():
    """The origins coincide whenever the multiple divides 1970 x 12 months (or 1970 x 4 quarters), so
    every multiple the main rounding operation uses has to agree exactly."""
    a = pa.array([0, 1_700_000_000], pa.timestamp("s"))
    for unit, multiple in [("month", 6), ("quarter", 2), ("year", 1)]:
        assert pylist(am.array(a).floor_temporal(unit, multiple)) == \
            pc.floor_temporal(a, multiple=multiple, unit=unit).to_pylist()
    # And Arrow's year unit counts from year 0, like ArrowMetal's -- which is why its own three
    # calendar units do not agree with each other.
    assert pc.floor_temporal(a, multiple=3, unit="year").to_pylist()[0].year == 1968


def test_rounding_to_a_finer_unit_converts_like_arrow():
    """Once a finding (temporal-round-finer-unit): rounding to a unit below the column's resolution
    now converts, rounds and truncates back, as Arrow does."""
    a = pa.array([1_700_000_000, 0, -1], pa.timestamp("s"))
    assert pylist(am.array(a).floor_temporal("nanosecond", 3)) == \
        pc.floor_temporal(a, multiple=3, unit="nanosecond").to_pylist()


def test_rounding_to_a_finer_unit_is_the_identity_at_multiple_one():
    a = pa.array([1_700_000_000], pa.timestamp("s"))
    assert pylist(am.array(a).floor_temporal("nanosecond")) == \
        pc.floor_temporal(a, unit="nanosecond").to_pylist() == a.to_pylist()


@pytest.mark.xfail(strict=True, reason="strftime-seconds-carry-the-fraction: Arrow's %S appends the "
                                       "sub-second digits, C's does not")
def test_strftime_seconds_carry_the_fraction():
    a = pa.array([1_700_000_000_123_456], pa.timestamp("us"))
    assert pylist(am.array(a).strftime("%S")) == pc.strftime(a, format="%S").to_pylist()


def test_strftime_agrees_on_every_other_field():
    a = pa.array([1_700_000_000_123_456], pa.timestamp("us"))
    for fmt in ("%Y-%m-%d", "%H:%M", "%j", "%Y"):
        assert pylist(am.array(a).strftime(fmt)) == pc.strftime(a, format=fmt).to_pylist()


@pytest.mark.xfail(strict=True, reason="timezone-after-2038: pyarrow's tz database stops applying "
                                       "the DST rule at the 32-bit epoch")
def test_assume_timezone_agrees_past_2038():
    a = pa.array([_seconds_of(2050, 7, 15, 12)], pa.timestamp("s"))
    assert pylist(am.array(a).assume_timezone(TZ_NAME, "earliest", "earliest")) == \
        pc.assume_timezone(a, TZ_NAME, ambiguous="earliest", nonexistent="earliest").to_pylist()


def test_assume_timezone_agrees_before_2038_and_keeps_the_rule_after():
    a = pa.array([_seconds_of(2024, 7, 15, 12)], pa.timestamp("s"))
    assert pylist(am.array(a).assume_timezone(TZ_NAME, "earliest", "earliest")) == \
        pc.assume_timezone(a, TZ_NAME, ambiguous="earliest", nonexistent="earliest").to_pylist()
    later = pa.array([_seconds_of(2050, 7, 15, 12)], pa.timestamp("s"))
    ours = am.array(later).assume_timezone(TZ_NAME, "earliest", "earliest").to_arrow()
    theirs = pc.assume_timezone(later, TZ_NAME, ambiguous="earliest", nonexistent="earliest")
    assert ours.cast(pa.int64())[0].as_py() == theirs.cast(pa.int64())[0].as_py() - 3600


@pytest.mark.xfail(strict=True, reason="trig-argument-reduction: the software binary64 sin/cos/tan "
                                       "lose the reduction past 2^49")
def test_trig_reduces_a_large_argument():
    a = pa.array([2.0 ** 52 + 0.5], pa.float64())
    assert pylist(am.array(a).sin())[0] == pytest.approx(pc.sin(a)[0].as_py(), rel=1e-12)


def test_trig_agrees_below_the_reduction_limit():
    a = pa.array([2.0 ** 48 + 0.5, 1e6, 0.5], pa.float64())
    for got, want in zip(pylist(am.array(a).sin()), pc.sin(a).to_pylist()):
        assert got == pytest.approx(want, rel=1e-14)


def test_trig_special_values_agree_exactly():
    """The tolerance the matrix gives the trigonometry does not reach these: a signed zero keeps its
    sign, an infinity is a NaN and a NaN stays a NaN, in both engines."""
    a = pa.array([0.0, -0.0, math.inf, -math.inf, math.nan], pa.float64())
    got, want = pylist(am.array(a).sin()), pc.sin(a).to_pylist()
    assert struct.pack("<d", got[0]) == struct.pack("<d", want[0]) == struct.pack("<d", 0.0)
    assert struct.pack("<d", got[1]) == struct.pack("<d", want[1]) == struct.pack("<d", -0.0)
    assert all(math.isnan(v) for v in got[2:]) and all(math.isnan(v) for v in want[2:])


def test_checked_trig_raises_in_both_engines_on_a_domain_error():
    a = pa.array([2.0], pa.float64())
    with pytest.raises(am.ArrowMetalError, match="domain error"):
        am.array(a).asin_checked()
    with pytest.raises(pa.ArrowInvalid, match="domain error"):
        pc.asin_checked(a)
    # A NaN is not a domain error in either engine, and a null row is never inspected.
    b = pa.array([math.nan, None], pa.float64())
    assert math.isnan(pylist(am.array(b).asin_checked())[0])
    assert math.isnan(pc.asin_checked(b).to_pylist()[0])


@pytest.mark.xfail(strict=True, reason="variance-accumulator-overflow: the moments leave the 64-bit "
                                       "accumulator, so the answer is a wrapped number or NaN")
def test_variance_of_a_large_int64_column():
    """Three copies of one value have variance zero whatever the value is; past 2^40 the sum of
    squares wraps in the accumulator and the answer stops meaning anything."""
    a = pa.array([2 ** 62] * 3, pa.int64())
    assert am.array(a).variance(0) == pc.variance(a, ddof=0).as_py() == 0.0


def test_variance_is_accurate_inside_the_accumulator():
    a = pa.array([1, 2, 3, 4, 5], pa.int64())
    assert am.array(a).variance(0) == pytest.approx(pc.variance(a, ddof=0).as_py(), rel=1e-12)
    assert am.array(a).stddev(1) == pytest.approx(pc.stddev(a, ddof=1).as_py(), rel=1e-12)


@pytest.mark.xfail(strict=True, reason="rolling-min-max-zero-and-subnormal: the rolling scan breaks "
                                       "a +/-0 tie by position rather than the way fmin does")
def test_rolling_min_breaks_a_zero_tie_like_fmin():
    a = pa.array([0.0, -0.0], pa.float64())
    assert struct.pack("<d", pylist(am.array(a).rolling_min(2))[1]) == \
        struct.pack("<d", pc.min(a).as_py())


def test_rolling_min_of_an_all_nan_window():
    """ArrowMetal treats NaN as missing all the way, so a window holding nothing else answers with
    the scan's identity; pc.min answers NaN. Same disagreement as the scalar min/max."""
    a = pa.array([math.nan, math.nan], pa.float64())
    assert pylist(am.array(a).rolling_min(2))[1] == math.inf
    assert math.isnan(pc.min(a).as_py())


def test_rolling_sum_is_a_prefix_sum_difference():
    """Why the matrix takes the infinities out of a float column first: the rolling sum subtracts two
    prefix sums, so one infinity reaches every later window through `inf - inf`."""
    a = pa.array([1.0, math.inf, 2.0, 3.0], pa.float64())
    got = pylist(am.array(a).rolling_sum(2))
    assert got[1] == math.inf and math.isnan(got[3])     # the window [2.0, 3.0] is 5.0 in truth
    assert pylist(am.array(pa.array([1.0, 2.0, 3.0, 4.0], pa.float64())).rolling_sum(2)) == \
        [None, 3.0, 5.0, 7.0]


def test_rolling_mean_accumulates_in_double():
    """The prefix sums are kept in `double` whatever the column type, so an integer column past 2^53
    cannot recover the window."""
    a = pa.array([2 ** 62, 1, 2 ** 62], pa.int64())
    assert pylist(am.array(a).rolling_mean(1)) == [float(2 ** 62), 0.0, float(2 ** 62)]
    assert pylist(am.array(pa.array([1, 2, 3], pa.int64())).rolling_mean(1)) == [1.0, 2.0, 3.0]


def test_cumulative_mean_accumulates_in_double():
    a = pa.array([1, 2, 3, 4], pa.int64())
    assert pylist(am.array(a).cumulative_mean()) == [1.0, 1.5, 2.0, 2.5]
    big = pa.array([2 ** 63 - 1, 2 ** 63 - 1], pa.int64())
    assert pylist(am.array(big).cumulative_mean())[1] == float(2 ** 63 - 1)


def test_product_reassociates_past_the_normal_range():
    """Why the matrix keeps the product's operands in a band: the reduction is a tree, so once a
    sub-product leaves the range the pairing decides the answer."""
    a = pa.array([1e30, 1e30, 1e-30, 1e-30] * 4, pa.float32())
    total = am.array(a).product()
    assert math.isnan(total) or math.isinf(total) or total == 0.0
    inside = pa.array([1.5, -2.0, 0.25, 4.0], pa.float32())
    assert am.array(inside).product() == pytest.approx(pc.product(inside).as_py(), rel=1e-6)


def test_index_of_a_large_unsigned_value_is_not_found():
    """`MetalArray.index` takes its needle as a C `double`, so a uint64 value above 2^63 does not
    survive the trip and the search reports "absent". Why the matrix picks a needle that round-trips
    exactly and is the only value in the column that does."""
    a = pa.array([2 ** 64 - 1, 2 ** 64 - 2], pa.uint64())
    assert am.array(a).index(2 ** 64 - 1) == -1
    assert pc.index(a, pa.scalar(2 ** 64 - 1, pa.uint64())).as_py() == 0
    # Inside the range a double can hold exactly, the two agree.
    b = pa.array([2 ** 62 + 1, 2 ** 62], pa.int64())
    assert am.array(b).index(2 ** 62) == pc.index(b, pa.scalar(2 ** 62, pa.int64())).as_py() == 1


def test_time_of_day_addition_wraps_where_pyarrow_raises():
    import datetime
    a = pa.array([1000], pa.time32("ms"))
    delta = pa.array([-2000], pa.duration("ms"))
    assert pylist(am.array(a).add_duration(am.array(delta))) == [datetime.time(23, 59, 59)]
    with pytest.raises(pa.ArrowInvalid):
        pc.add(a, delta)


def test_pyarrow_pairwise_diff_ignores_the_array_offset():
    """Not our divergence: pyarrow 25.0.1 reads the values buffer of a sliced column without honouring
    ArrowArray.offset, so the matrix hands its oracle a materialised copy."""
    full = pa.array([10, 20, 30, 40, 50, 7, 3, 99, 1, 60, 2, 5], pa.int32())
    sliced = full.slice(5, 6)
    copy = pa.array(sliced.to_pylist(), pa.int32())
    assert pylist(am.array(sliced).pairwise_diff()) == pc.pairwise_diff(copy).to_pylist()
    assert pc.pairwise_diff(sliced).to_pylist() != pc.pairwise_diff(copy).to_pylist()


def test_pyarrow_boolean_fill_null_forward_ignores_the_array_offset():
    """The same bug on a boolean column, for fill_null_forward / _backward and replace_with_mask."""
    full = pa.array([True, False, None, True, True, False, None, True], pa.bool_())
    sliced = full.slice(3, 5)
    copy = pa.array(sliced.to_pylist(), pa.bool_())
    assert pylist(am.array(sliced).fill_null_forward()) == pc.fill_null_forward(copy).to_pylist()
    assert pc.fill_null_forward(sliced).to_pylist() != pc.fill_null_forward(copy).to_pylist()


def test_hash64_is_deterministic_and_canonicalises_equal_values():
    """The two properties the matrix's hash64 case rests on, spelled out: the same bytes always hash
    the same, and values Arrow calls equal hash equal."""
    a = pa.array([1.0, 2.0, -0.0, 0.0, math.nan, math.nan, None], pa.float64())
    first, second = pylist(am.array(a).hash64()), pylist(am.array(a).hash64())
    assert first == second
    assert first[2] == first[3]                    # -0.0 hashes as 0.0
    assert first[4] == first[5]                    # every NaN hashes alike
    assert first[6] is None
    assert len({first[0], first[1], first[2], first[4]}) == 4


def test_float16_never_computes_in_half_precision():
    a = pa.array(np.array([1.5, 2.5], np.float16), type=pa.float16())
    widened = am.array(a).to_float32()
    assert widened.type == pa.float32()
    assert pylist(widened) == a.cast(pa.float32()).to_pylist()
    assert pylist(widened.to_float16()) == a.to_pylist()


def test_extension_metadata_round_trips_through_pyarrow():
    a = pa.array([1, 2], pa.int64())
    tagged = am.array(a).as_extension_type("arrowmetal.pinned", b"v1")
    assert tagged.extension_name == "arrowmetal.pinned"
    assert tagged.extension_metadata == b"v1"
    assert pylist(tagged.extension_storage()) == a.to_pylist()
