"""Grouped Float64 sum and mean are correctly rounded.

`sum` must equal `math.fsum` of each group's non-null values, bit for bit, and `mean` must equal the
exact mean `Fraction(sum) / count` rounded once (`float(Fraction)` rounds correctly). Checked through
the array API (`am.group_by(...).sum/.mean`) and the plan runner (`am.scan(...).group_by(...).agg`),
over group counts on both sides of the private-table limit (1,024 groups), with null keys and values,
a wide spread of magnitudes, cancellation, signed zeros, subnormals, infinities, NaN and overflow.
Grouped min and max are checked against a host reference too: NaN is skipped and -0.0 is below +0.0.
"""
import math
import struct
from fractions import Fraction

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am


def _py(a):
    return (a.to_arrow() if hasattr(a, "to_arrow") else a).to_pylist()


def _bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return _bits(a) == _bits(b)


def _expected(xs):
    """(sum, mean) of the non-null values, correctly rounded, with IEEE special values."""
    if not xs:
        return None, None
    if any(math.isnan(x) for x in xs):
        return math.nan, math.nan
    pinf, ninf = math.inf in xs, -math.inf in xs
    if pinf and ninf:
        return math.nan, math.nan
    if pinf or ninf:
        v = math.inf if pinf else -math.inf
        return v, v
    if all(x == 0 for x in xs):
        z = -0.0 if all(math.copysign(1, x) < 0 for x in xs) else 0.0
        return z, z
    exact = sum(Fraction(x) for x in xs)

    def rounded(q):
        try:
            return float(q) if q != 0 else 0.0
        except OverflowError:
            return math.inf if q > 0 else -math.inf

    return rounded(exact), rounded(exact / len(xs))


def _reference(keys, vals):
    groups = {}
    for k, v in zip(keys, vals):
        if v is None:
            continue
        groups.setdefault(k, []).append(v)   # a null key is a group of its own
    return groups


def _check(keys, vals, label):
    kt = pa.array(keys, type=pa.int32())
    vt = pa.array(vals, type=pa.float64())
    groups = _reference(keys, vals)

    gb = am.group_by([am.array(kt)])
    gkeys = _py(gb.keys()[0])
    s = _py(gb.sum(am.array(vt)))
    m = _py(gb.mean(am.array(vt)))
    for i, k in enumerate(gkeys):
        es, em = _expected(groups.get(k, []))
        assert _same(s[i], es), f"{label} array sum key {k}: {s[i]!r} want {es!r}"
        assert _same(m[i], em), f"{label} array mean key {k}: {m[i]!r} want {em!r}"

    t = pa.table({"k": kt, "v": vt})
    out = am.scan(t).group_by("k").agg(am.agg.sum("v", "s"), am.agg.mean("v", "m")).collect()
    for k, sv, mv in zip(out.column("k").to_pylist(), out.column("s").to_pylist(), out.column("m").to_pylist()):
        es, em = _expected(groups.get(k, []))
        assert _same(sv, es), f"{label} plan sum key {k}: {sv!r} want {es!r}"
        assert _same(mv, em), f"{label} plan mean key {k}: {mv!r} want {em!r}"


@pytest.mark.parametrize("groups", [1, 7, 1024, 1025, 20000])
@pytest.mark.parametrize("spread", [False, True])
def test_random_groups_correctly_rounded(groups, spread):
    rng = np.random.default_rng(groups * 2 + spread)
    n = max(3000, groups * 3)
    keys = rng.integers(0, groups, n).tolist()
    exps = rng.integers(-300, 300, n) if spread else rng.integers(-3, 20, n)
    vals = (rng.uniform(-1, 1, n) * np.exp2(exps.astype(np.float64))).tolist()
    for i in range(0, n, 17):
        vals[i] = None
    for i in range(3, n, 41):
        keys[i] = None
    _check(keys, vals, f"groups={groups} spread={spread}")


def test_edge_groups():
    tiny = 5e-324
    maxd = 1.7976931348623157e308
    groups = [
        [1e300, 1.0, -1e300], [2.0 ** 53, 1.0, -2.0 ** 53], [1.0, 2.0 ** -60, -1.0],
        [1e16, 1.0, 1.0, 1.0, -1e16], [0.1, 0.2, 0.3, -0.6],
        [-0.0], [-0.0, -0.0], [0.0, -0.0], [0.0], [1.0, -1.0], [-1.0, 1.0, -0.0],
        [tiny], [tiny, tiny, -tiny], [tiny, -tiny], [-tiny, -tiny], [2.0 ** -1022, -tiny],
        [maxd, maxd], [maxd, maxd, -maxd], [-maxd, -maxd], [maxd, 2.0 ** 970], [maxd, 2.0 ** 969],
        [math.nan], [1.0, math.nan], [math.inf, 1.0], [-math.inf, -1.0], [math.inf, -math.inf],
        [2.0 ** 53, 1.0], [2.0 ** 53, 1.0, 2.0 ** -80], [2.0 ** 52 + 1, 0.5],
    ]
    keys, vals = [], []
    for k, g in enumerate(groups):
        for v in g:
            keys.append(k)
            vals.append(v)
        keys.append(k)
        vals.append(None)
    _check(keys, vals, "edge")
    # The same groups spread over many more groups (the device-table path), rows reversed.
    keys2 = [k * 997 + r for r in range(3) for k, g in enumerate(groups) for _ in g]
    vals2 = [v for r in range(3) for g in groups for v in g]
    _check(keys2[::-1], vals2[::-1], "edge-dev")


def test_cancellation_heavy():
    rng = np.random.default_rng(3)
    keys, vals = [], []
    for _ in range(20000):
        k = int(rng.integers(0, 5))
        big = float(rng.uniform(1e15, 1e17))
        keys += [k, k, k]
        vals += [big, float(rng.uniform(-1, 1)), -big]
    _check(keys, vals, "cancel")


@pytest.mark.parametrize("groups", [3, 1024, 1025, 50000])
def test_min_max_against_reference(groups):
    rng = np.random.default_rng(groups)
    n = 60000
    keys = rng.integers(0, groups, n).tolist()
    pick = rng.integers(0, 20, n)
    vals = rng.uniform(-1e6, 1e6, n)
    vals[pick == 0] = np.nan
    vals[pick == 1] = -0.0
    vals[pick == 2] = 0.0
    vals[pick == 3] = np.inf
    vals = vals.tolist()
    for i in range(0, n, 13):
        vals[i] = None
    kt, vt = pa.array(keys, type=pa.int32()), pa.array(vals, type=pa.float64())
    ref = {}
    for k, v in zip(keys, vals):
        if v is None or math.isnan(v):
            ref.setdefault(k, None)
            continue
        lo, hi = ref.get(k) or (v, v)
        key = lambda x: (x, math.copysign(1, x))
        ref[k] = (min(lo, v, key=key), max(hi, v, key=key))
    gb = am.group_by([am.array(kt)])
    gk = _py(gb.keys()[0])
    mn, mx = _py(gb.min(am.array(vt))), _py(gb.max(am.array(vt)))
    t = pa.table({"k": kt, "v": vt})
    out = am.scan(t).group_by("k").agg(am.agg.min("v", "lo"), am.agg.max("v", "hi")).collect()
    plan = {k: (a, b) for k, a, b in zip(out.column("k").to_pylist(), out.column("lo").to_pylist(),
                                        out.column("hi").to_pylist())}
    for i, k in enumerate(gk):
        want = ref.get(k)
        want = (None, None) if want is None else want
        assert _same(mn[i], want[0]) and _same(mx[i], want[1]), f"array key {k}: {(mn[i], mx[i])} want {want}"
        assert _same(plan[k][0], want[0]) and _same(plan[k][1], want[1]), f"plan key {k}: {plan[k]} want {want}"
