"""ArrowMetal's checked arithmetic and extra element-wise math against pyarrow.compute.

The oracle is Arrow itself: every value below is compared with what `pyarrow.compute` returns for the
same input, and every raise is compared with the `ArrowInvalid` pyarrow raises for the same input. The
handful of places where ArrowMetal deliberately answers differently are collected in
`test_documented_differences_from_pyarrow`, so a divergence that is *not* on that list fails.

    PYTHONPATH=python python -m pytest python/tests/test_checked.py -q

Needs a real Metal device and libArrowMetalC.dylib (see python/README.md).
"""
import math

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am


SIGNED = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
UNSIGNED = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
INTEGER = SIGNED + UNSIGNED
FLOATING = [pa.float32(), pa.float64()]

#: Sizes the Swift suite also uses: empty, single, a partial threadgroup, several, and one long run.
SIZES = [0, 1, 33, 4097, 1_000_003]


def col(values, ty):
    return am.MetalArray.from_arrow(pa.array(values, ty))


def raises(fn):
    """(raised, message) for a call that may raise either engine's error."""
    try:
        fn()
        return False, ""
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, am.ArrowMetalError) as e:
        return True, str(e)


def ints(ty, n, seed=1):
    """Deterministic values that span the type, with a null every seventh row."""
    lo, hi = (-(2 ** (ty.bit_width - 1)), 2 ** (ty.bit_width - 1) - 1) if ty in SIGNED \
        else (0, 2 ** ty.bit_width - 1)
    span = hi - lo
    out = []
    s = seed
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) % (2 ** 64)
        out.append(None if i % 7 == 3 else lo + (s >> 11) % (span + 1))
    return out


# ------------------------------------------------------------------ checked arithmetic


@pytest.mark.parametrize("ty", INTEGER)
@pytest.mark.parametrize("n", SIZES)
def test_checked_matches_pyarrow_on_integers(ty, n):
    """Same values where nothing overflows, same raise where something does."""
    a = pa.array(ints(ty, n, 1), ty)
    b = pa.array([None if v is None else v // 3 or 1 for v in ints(ty, n, 2)], ty)
    ours, theirs = am.MetalArray.from_arrow(a), a
    other = am.MetalArray.from_arrow(b)
    for name, oracle in [("add_checked", pc.add_checked), ("subtract_checked", pc.subtract_checked),
                         ("multiply_checked", pc.multiply_checked), ("divide_checked", pc.divide_checked)]:
        gotRaised, gotMsg = raises(lambda: getattr(ours, name)(other).to_arrow())
        wantRaised, wantMsg = raises(lambda: oracle(theirs, b))
        assert gotRaised == wantRaised, f"{name} {ty} n={n}: ArrowMetal raised={gotRaised} ({gotMsg}), pyarrow raised={wantRaised} ({wantMsg})"
        if gotRaised:
            # The Arrow message is quoted verbatim inside ArrowMetal's, which adds the op and the row.
            assert wantMsg in gotMsg, f"{name}: {gotMsg!r} does not contain {wantMsg!r}"
        else:
            assert getattr(ours, name)(other).to_arrow().equals(oracle(theirs, b))


@pytest.mark.parametrize("ty", FLOATING)
@pytest.mark.parametrize("n", [0, 1, 33, 4097])
def test_checked_floats_never_raise_on_overflow(ty, n):
    """Arrow's checked float kernels treat infinity and NaN as ordinary results, and so do we."""
    vals = [None if i % 7 == 3 else (3.0e38 if ty == pa.float32() else 1.0e308) for i in range(n)]
    a = pa.array(vals, ty)
    ours = am.MetalArray.from_arrow(a)
    assert ours.add_checked(ours).to_arrow().equals(pc.add_checked(a, a))
    assert ours.multiply_checked(ours).to_arrow().equals(pc.multiply_checked(a, a))
    assert ours.subtract_checked(ours).to_arrow().equals(pc.subtract_checked(a, a))


@pytest.mark.parametrize("ty", INTEGER)
def test_checked_boundary_values(ty):
    lo = -(2 ** (ty.bit_width - 1)) if ty in SIGNED else 0
    hi = (2 ** (ty.bit_width - 1) - 1) if ty in SIGNED else 2 ** ty.bit_width - 1
    one = pa.array([1], ty)

    # T.max + 1 and T.min - 1 overflow in both engines.
    for values, op, oracle in [([hi], "add_checked", pc.add_checked), ([lo], "subtract_checked", pc.subtract_checked)]:
        a = pa.array(values, ty)
        got, gotMsg = raises(lambda: getattr(am.MetalArray.from_arrow(a), op)(1).to_arrow())
        want, _ = raises(lambda: oracle(a, one))
        assert got and want, f"{op} at the boundary: ours={got} pyarrow={want} ({gotMsg})"

    # Division by zero, on every type.
    a = pa.array([1], ty)
    got, gotMsg = raises(lambda: am.MetalArray.from_arrow(a).divide_checked(0).to_arrow())
    assert got and "divide by zero" in gotMsg

    if ty in SIGNED:
        # T.min / -1 is the one division that overflows rather than dividing by zero.
        a = pa.array([lo], ty)
        got, gotMsg = raises(lambda: am.MetalArray.from_arrow(a).divide_checked(-1).to_arrow())
        want, wantMsg = raises(lambda: pc.divide_checked(a, pa.array([-1], ty)))
        assert got and want and "overflow" in gotMsg and "overflow" in wantMsg
        # negate/abs of T.min.
        for op, oracle in [("negate_checked", pc.negate_checked), ("abs_checked", pc.abs_checked)]:
            got, _ = raises(lambda: getattr(am.MetalArray.from_arrow(a), op)().to_arrow())
            want, _ = raises(lambda: oracle(a))
            assert got and want, f"{op}({lo}) ours={got} pyarrow={want}"

    # Shift amounts: Arrow's "precision" is the bit width, one less on a signed type.
    precision = ty.bit_width - (1 if ty in SIGNED else 0)
    for k in [0, precision - 1, precision, ty.bit_width, ty.bit_width + 1]:
        if k > (2 ** ty.bit_width - 1):
            continue
        a = pa.array([1], ty)
        amount = pa.array([k], ty)
        got, gotMsg = raises(lambda: am.MetalArray.from_arrow(a).shift_left_checked(k).to_arrow())
        want, _ = raises(lambda: pc.shift_left_checked(a, amount))
        assert got == want, f"shift_left_checked({ty}, {k}): ours={got} ({gotMsg}) pyarrow={want}"


@pytest.mark.parametrize("ty", FLOATING)
def test_domain_checks_match_pyarrow(ty):
    cases = [
        ("sqrt_checked", pc.sqrt_checked, [-1.0, 4.0, float("nan"), float("inf"), -0.0]),
        ("ln_checked", pc.ln_checked, [0.0, -1.0, 1.0, float("nan"), float("inf")]),
        ("log2_checked", pc.log2_checked, [0.0, -1.0, 8.0]),
        ("log10_checked", pc.log10_checked, [0.0, -1.0, 100.0]),
        ("log1p_checked", pc.log1p_checked, [-1.0, -2.0, 0.0, 1.0]),
    ]
    for name, oracle, values in cases:
        for v in values:
            a = pa.array([v], ty)
            got, gotMsg = raises(lambda: getattr(am.MetalArray.from_arrow(a), name)().to_arrow())
            want, wantMsg = raises(lambda: oracle(a))
            assert got == want, f"{name}({v}) {ty}: ours={got} ({gotMsg}) pyarrow={want} ({wantMsg})"
            if got:
                assert wantMsg in gotMsg, f"{name}({v}): {gotMsg!r} does not quote {wantMsg!r}"


@pytest.mark.parametrize("ty", INTEGER)
@pytest.mark.parametrize("n", [1, 33, 4097])
def test_cumulative_and_pairwise_checked(ty, n):
    """Values small enough to stay in range must match; a deliberate overflow must raise."""
    small = [None if i % 7 == 3 else (i % 3) for i in range(n)]
    a = pa.array(small, ty)
    ours = am.MetalArray.from_arrow(a)

    def both(label, ourCall, theirCall):
        """Raise where Arrow raises, and the same values where it does not."""
        got, gotMsg = raises(lambda: ourCall().to_arrow())
        want, wantMsg = raises(theirCall)
        assert got == want, f"{label} {ty} n={n}: ours={got} ({gotMsg}) pyarrow={want} ({wantMsg})"
        if not got:
            assert ourCall().to_arrow().equals(theirCall())

    # ArrowMetal skips nulls in the cumulative functions; pyarrow's default propagates them, so the
    # comparison uses skip_nulls=True, which is the rule this package implements.
    both("cumulative_sum_checked", ours.cumulative_sum_checked,
         lambda: pc.cumulative_sum_checked(a, skip_nulls=True))
    both("cumulative_prod_checked", ours.cumulative_prod_checked,
         lambda: pc.cumulative_prod_checked(a, skip_nulls=True))
    both("pairwise_diff_checked", ours.pairwise_diff_checked, lambda: pc.pairwise_diff_checked(a))

    hi = (2 ** (ty.bit_width - 1) - 1) if ty in SIGNED else 2 ** ty.bit_width - 1
    big = pa.array([hi, 1], ty)
    got, gotMsg = raises(lambda: am.MetalArray.from_arrow(big).cumulative_sum_checked().to_arrow())
    want, _ = raises(lambda: pc.cumulative_sum_checked(big))
    assert got and want, f"cumulative_sum_checked overflow: ours={got} ({gotMsg}) pyarrow={want}"


# ------------------------------------------------------------------ new element-wise math


@pytest.mark.parametrize("ty", FLOATING)
@pytest.mark.parametrize("n", [0, 1, 33, 4097])
def test_expm1_log1p_logb_hypot_match_pyarrow(ty, n):
    tol = 1e-5 if ty == pa.float32() else 1e-13
    xs = [None if i % 7 == 3 else ((i % 400) - 200) * 0.05 for i in range(n)]
    ps = [None if i % 7 == 3 else 0.001 + (i % 1000) for i in range(n)]
    x, p = pa.array(xs, ty), pa.array(ps, ty)
    ox, op = am.MetalArray.from_arrow(x), am.MetalArray.from_arrow(p)

    for got, want in [(ox.expm1(), pc.expm1(x)),
                      (op.log1p(), pc.log1p(p)),
                      (op.logb(2.0), pc.logb(p, pa.scalar(2.0, ty))),
                      (op.logb(op), pc.logb(p, p)),
                      (ox.hypot(3.0), pc.hypot(x, pa.scalar(3.0, ty))),
                      (ox.hypot(ox), pc.hypot(x, x))]:
        g, w = got.to_arrow().to_pylist(), want.to_pylist()
        assert len(g) == len(w)
        for i, (a, b) in enumerate(zip(g, w)):
            assert (a is None) == (b is None), f"null mismatch at {i}"
            if a is None:
                continue
            assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"row {i}: {a} vs {b}"


ROUND_MODES = ["down", "up", "towards_zero", "towards_infinity", "half_down", "half_up",
               "half_towards_zero", "half_towards_infinity", "half_to_even", "half_to_odd"]


@pytest.mark.parametrize("mode", ROUND_MODES)
def test_round_modes_match_pyarrow_on_floats(mode):
    vals = [-2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 3.5, 123.456, -123.456, 1e17, None]
    for ty in FLOATING:
        a = pa.array(vals, ty)
        ours = am.MetalArray.from_arrow(a)
        for nd in [0, 1, 2, -1, -2]:
            got = ours.round(nd, mode).to_arrow()
            want = pc.round(a, ndigits=nd, round_mode=mode)
            assert got.equals(want), f"round ndigits={nd} {mode} {ty}: {got.to_pylist()} vs {want.to_pylist()}"
        for m in [0.5, 1.0, 2.0, 3.0]:
            got = ours.round_to_multiple(m, mode).to_arrow()
            want = pc.round_to_multiple(a, multiple=pa.scalar(m, ty), round_mode=mode)
            assert got.equals(want), f"round_to_multiple {m} {mode} {ty}"


@pytest.mark.parametrize("mode", ROUND_MODES)
@pytest.mark.parametrize("ty", INTEGER)
def test_round_modes_match_pyarrow_on_integers(mode, ty):
    # T.min is left out on purpose: rounding it away from zero leaves the type, which ArrowMetal wraps
    # (like the rest of its unchecked arithmetic) and pyarrow raises on. See
    # test_documented_differences_from_pyarrow.
    lo = -(2 ** (ty.bit_width - 1)) if ty in SIGNED else 0
    vals = [v for v in [-7, -5, -3, 0, 3, 5, 7, 10, 100, None] if v is None or v >= lo]
    a = pa.array(vals, ty)
    ours = am.MetalArray.from_arrow(a)
    for m in [1, 3, 5, 10]:
        if m > 2 ** ty.bit_width - 1:
            continue
        got = ours.round_to_multiple(m, mode).to_arrow()
        want = pc.round_to_multiple(a, multiple=pa.scalar(m, ty), round_mode=mode)
        assert got.equals(want), f"round_to_multiple {m} {mode} {ty}: {got.to_pylist()} vs {want.to_pylist()}"
    for nd in [0, 1, -1]:
        if -nd >= len(str(2 ** ty.bit_width)):
            continue
        got = ours.round(nd, mode).to_arrow()
        want = pc.round(a, ndigits=nd, round_mode=mode)
        assert got.equals(want), f"round ndigits={nd} {mode} {ty}"


def test_round_binary_matches_pyarrow():
    vals = [123.456, -123.456, 1.0, 2.5, None, 0.5]
    nd = [0, 1, 2, -1, 0, None]
    for ty in FLOATING:
        a = pa.array(vals, ty)
        got = am.MetalArray.from_arrow(a).round_binary(nd).to_arrow()
        want = pc.round_binary(a, pa.array(nd, pa.int32()))
        assert got.equals(want), f"round_binary {ty}: {got.to_pylist()} vs {want.to_pylist()}"


def test_round_to_multiple_rejects_a_non_positive_multiple():
    a = am.MetalArray.from_arrow(pa.array([1.0]))
    for bad in (0.0, -1.0):
        with pytest.raises(am.ArrowMetalError):
            a.round_to_multiple(bad)
        with pytest.raises(pa.ArrowInvalid):
            pc.round_to_multiple(pa.array([1.0]), multiple=bad)


def test_round_with_no_arguments_keeps_its_old_meaning():
    """The historical round() rounds halves away from zero; ndigits/mode select Arrow's kernel."""
    a = am.MetalArray.from_arrow(pa.array([-2.5, -0.5, 0.5, 2.5]))
    assert a.round().to_arrow().to_pylist() == [-3.0, -1.0, 1.0, 3.0]
    assert a.round(0).to_arrow().to_pylist() == pc.round(pa.array([-2.5, -0.5, 0.5, 2.5])).to_pylist()


# ------------------------------------------------------------------ the divergences we accept


def test_documented_differences_from_pyarrow():
    """Every place ArrowMetal deliberately answers differently. A change here is a change in contract."""
    # 1. Unsigned negate_checked: pyarrow has no kernel at all; ArrowMetal defines it as "every
    #    non-zero value overflows", the only answer a modular negation could give.
    with pytest.raises(pa.ArrowNotImplementedError):
        pc.negate_checked(pa.array([5], pa.uint8()))
    with pytest.raises(am.ArrowMetalError):
        col([5], pa.uint8()).negate_checked()
    assert col([0], pa.uint8()).negate_checked().to_arrow().to_pylist() == [0]

    # 2. The transcendentals need a float column here; pyarrow promotes an integer one to float64.
    assert pc.sqrt_checked(pa.array([4], pa.int32())).to_pylist() == [2.0]
    for method in ("sqrt_checked", "ln_checked", "log1p_checked", "expm1", "log1p"):
        with pytest.raises(am.ArrowMetalError):
            getattr(col([4], pa.int32()), method)()

    # 3. The cumulative functions skip nulls here; pyarrow's default propagates them.
    a = pa.array([1, None, 3], pa.int64())
    assert col([1, None, 3], pa.int64()).cumulative_sum_checked().to_arrow().to_pylist() == [1, None, 4]
    assert pc.cumulative_sum_checked(a).to_pylist() == [1, None, None]
    assert pc.cumulative_sum_checked(a, skip_nulls=True).to_pylist() == [1, None, 4]

    # 4. A float ndigits past the type's decimal range is the identity here; pyarrow raises above ~1e308.
    assert col([123.456], pa.float64()).round(400).to_arrow().to_pylist() == [123.456]
    with pytest.raises(pa.ArrowInvalid):
        pc.round(pa.array([123.456]), ndigits=400)

    # 5. An integer ndigits whose multiple does not fit the column type gives 0 here; pyarrow raises.
    assert col([125], pa.int32()).round(-10).to_arrow().to_pylist() == [0]
    with pytest.raises(pa.ArrowInvalid):
        pc.round(pa.array([125], pa.int32()), ndigits=-10)

    # 6. Rounding a value away from zero past the type's edge wraps here, like the rest of the
    #    unchecked arithmetic; pyarrow raises "would overflow".
    assert col([-128], pa.int8()).round_to_multiple(3, "down").to_arrow().to_pylist() == [127]
    with pytest.raises(pa.ArrowInvalid):
        pc.round_to_multiple(pa.array([-128], pa.int8()), multiple=pa.scalar(3, pa.int8()), round_mode="down")

    # 7. ArrowMetal's message quotes Arrow's wording and adds the op and the first offending row.
    try:
        col([127, 1], pa.int8()).add_checked(1)
        raise AssertionError("expected a raise")
    except am.ArrowMetalError as e:
        assert str(e) == "add_checked: overflow at index 0"
