"""`sqrt`, `exp`, `ln`, `log2`, `log10` and `power` on float64, against pyarrow.compute.

Apple GPUs have no double-precision hardware, so these six run as software IEEE-754 binary64 on the
GPU (`Kernels/DoubleMath.swift`, `Kernels/DoublePower.swift`). This file is the cross-engine half of
the claim the Swift suite measures: every value is compared with what `pyarrow.compute` returns for
the same input, in **ulp**, and the budget is 2 — with `sqrt` held to bit equality, because a
digit-by-digit extraction is correctly rounded and so is Arrow's.

    PYTHONPATH=python python -m pytest python/tests/test_float64_math.py -q

Needs a real Metal device and libArrowMetalC.dylib (see python/README.md).
"""
import math
import struct

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

#: Sizes the Swift suite also uses: empty, single, a partial threadgroup, several, and one long run.
SIZES = [0, 1, 33, 4097, 1_000_003]

ULP_BUDGET = 2.0


def ulps(a, b):
    """Distance in units of the last place. Two NaNs agree; anything else that differs in finiteness
    does not."""
    if a == b:
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return 0.0 if (math.isnan(a) and math.isnan(b)) else math.inf
    if math.isinf(a) or math.isinf(b):
        return math.inf
    return abs(a - b) / math.ulp(max(abs(a), abs(b)))


def compare(name, got, want, budget=ULP_BUDGET):
    """Worst ulp between two float64 columns, nulls lined up, asserted against `budget`."""
    g, w = got.to_arrow().to_pylist(), want.to_pylist()
    assert len(g) == len(w), f"{name}: length {len(g)} != {len(w)}"
    worst, at = 0.0, -1
    for i, (x, y) in enumerate(zip(g, w)):
        if x is None or y is None:
            assert x is None and y is None, f"{name}: null mismatch at row {i}"
            continue
        e = ulps(x, y)
        if e > worst:
            worst, at = e, i
    assert worst <= budget, f"{name}: worst {worst} ulp at row {at} (got {g[at]}, pyarrow {w[at]})"
    return worst


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def spread(n, seed=1):
    """Positive float64s spread over the whole exponent range, with a null every seventh row.

    A log-uniform draw rather than a uniform one: a uniform sample of [0, 1e300] is, to sixteen
    digits, a sample of one decade, and would never exercise the subnormal or near-1 branches.
    """
    state = seed
    out = []
    for i in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & (2 ** 64 - 1)
        if i % 7 == 6:
            out.append(None)
            continue
        u = (state >> 11) / float(1 << 53)
        out.append(math.exp(u * 1400.0 - 700.0))
    return pa.array(out, pa.float64())


def exponents(n, seed=2):
    """Exponents small enough that `power` usually stays finite, with a null every seventh row."""
    state = seed
    out = []
    for i in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & (2 ** 64 - 1)
        if i % 7 == 6:
            out.append(None)
            continue
        out.append(((state >> 11) / float(1 << 53)) * 60.0 - 30.0)
    return pa.array(out, pa.float64())


@pytest.mark.parametrize("n", SIZES)
def test_unary_against_pyarrow_at_every_size(n):
    a = spread(n)
    g = am.MetalArray.from_arrow(a)
    compare("ln", g.ln(), pc.ln(a))
    compare("log2", g.log2(), pc.log2(a))
    compare("log10", g.log10(), pc.log10(a))
    compare("exp", am.MetalArray.from_arrow(exponents(n)).exp(), pc.exp(exponents(n)))
    # sqrt is correctly rounded on both sides, so nothing less than bit equality would be honest.
    compare("sqrt", g.sqrt(), pc.sqrt(a), budget=0.0)


@pytest.mark.parametrize("n", SIZES)
def test_power_against_pyarrow_at_every_size(n):
    base, expo = spread(n), exponents(n)
    gb, ge = am.MetalArray.from_arrow(base), am.MetalArray.from_arrow(expo)
    compare("power(array, array)", gb.power(ge), pc.power(base, expo))
    compare("power(array, 2.5)", gb.power(2.5), pc.power(base, pa.scalar(2.5)))
    # The result of a two-column op is null wherever either side is; both have nulls in row 6 mod 7,
    # so line them up explicitly as well.
    assert gb.power(ge).null_count == pc.power(base, expo).null_count


def test_checked_twins_match_pyarrow():
    """The checked kernels answer the same values and raise on the same rows as pyarrow's."""
    ok = pa.array([1.0, 2.0, 0.5, 4.0, 1e-320, 1e300], pa.float64())
    g = am.MetalArray.from_arrow(ok)
    compare("ln_checked", g.ln_checked(), pc.ln_checked(ok))
    compare("log2_checked", g.log2_checked(), pc.log2_checked(ok))
    compare("log10_checked", g.log10_checked(), pc.log10_checked(ok))
    compare("sqrt_checked", g.sqrt_checked(), pc.sqrt_checked(ok), budget=0.0)

    for name, ours, theirs in [
        ("ln", lambda c: c.ln_checked(), lambda a: pc.ln_checked(a)),
        ("log2", lambda c: c.log2_checked(), lambda a: pc.log2_checked(a)),
        ("log10", lambda c: c.log10_checked(), lambda a: pc.log10_checked(a)),
    ]:
        for values, word in [([1.0, 0.0], "zero"), ([1.0, -1.0], "negative")]:
            a = pa.array(values, pa.float64())
            with pytest.raises(am.ArrowMetalError) as ours_err:
                ours(am.MetalArray.from_arrow(a))
            with pytest.raises(pa.ArrowInvalid) as their_err:
                theirs(a)
            assert word in str(ours_err.value), f"{name}: {ours_err.value}"
            assert word in str(their_err.value), f"{name}: pyarrow said {their_err.value}"

    a = pa.array([1.0, -1.0], pa.float64())
    with pytest.raises(am.ArrowMetalError):
        am.MetalArray.from_arrow(a).sqrt_checked()
    with pytest.raises(pa.ArrowInvalid):
        pc.sqrt_checked(a)

    # Arrow's float `power_checked` never raises; ours is bit-identical to the unchecked kernel.
    base = pa.array([1e300, -2.0, 0.0, 2.0], pa.float64())
    expo = pa.array([2.0, 3.0, -1.0, 0.5], pa.float64())
    gb = am.MetalArray.from_arrow(base)
    compare("power_checked", gb.power_checked(am.MetalArray.from_arrow(expo)),
            pc.power_checked(base, expo))


def test_edge_table_matches_pyarrow_exactly():
    """The values C99 pins down: x^0, 0^y, 1^y, (-1)^int, and infinity and NaN propagation."""
    inf, nan = math.inf, math.nan
    pairs = [(0.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-0.0, 3.0), (-0.0, 2.0), (-0.0, -3.0),
             (1.0, nan), (1.0, inf), (1.0, -inf), (nan, 0.0), (inf, 0.0), (-inf, 0.0),
             (-1.0, inf), (-1.0, -inf), (2.0, inf), (0.5, inf), (2.0, -inf), (0.5, -inf),
             (inf, 2.0), (inf, -2.0), (-inf, 3.0), (-inf, 2.0), (-inf, -3.0),
             (-2.0, 0.5), (-2.0, 3.0), (-2.0, 2.0), (2.0, 1024.0), (2.0, -1075.0), (2.0, 1023.0),
             (5e-324, 0.5), (1e308, 2.0), (10.0, 308.0), (3.0, 5.0), (nan, 2.0), (2.0, nan)]
    base = pa.array([p[0] for p in pairs], pa.float64())
    expo = pa.array([p[1] for p in pairs], pa.float64())
    got = am.MetalArray.from_arrow(base).power(am.MetalArray.from_arrow(expo)).to_arrow().to_pylist()
    want = pc.power(base, expo).to_pylist()
    for (x, y), g, w in zip(pairs, got, want):
        if math.isnan(w):
            assert math.isnan(g), f"pow({x}, {y}): {g} should be NaN"
        else:
            assert bits(g) == bits(w), f"pow({x}, {y}): got {g!r}, pyarrow {w!r}"

    # The unary functions at their own pinned points.
    xs = pa.array([0.0, -0.0, 1.0, 2.0, 10.0, inf, -inf, nan, -1.0,
                   5e-324, 2.2250738585072014e-308, 1.7976931348623157e308], pa.float64())
    g = am.MetalArray.from_arrow(xs)
    for name, ours, theirs in [("sqrt", g.sqrt(), pc.sqrt(xs)), ("ln", g.ln(), pc.ln(xs)),
                               ("log2", g.log2(), pc.log2(xs)), ("log10", g.log10(), pc.log10(xs)),
                               ("exp", g.exp(), pc.exp(xs))]:
        for x, a, b in zip(xs.to_pylist(), ours.to_arrow().to_pylist(), theirs.to_pylist()):
            if math.isnan(b):
                assert math.isnan(a), f"{name}({x}): {a} should be NaN"
            else:
                assert ulps(a, b) <= ULP_BUDGET, f"{name}({x}): got {a!r}, pyarrow {b!r}"


def test_modulo_is_still_refused_on_float64():
    """The one float64 binary op with no kernel says so rather than answering badly."""
    a = am.MetalArray.from_arrow(pa.array([1.0, 2.0], pa.float64()))
    with pytest.raises(am.ArrowMetalError, match="not implemented for float64"):
        a.modulo(2.0)
