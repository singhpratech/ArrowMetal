"""Sort options: `null_placement` and `float_order` on argsort / sort / top_k / lexsort / the lazy sort.

Two float orders, two references:

* ``float_order="ieee"`` (the default) is pyarrow's own order, so it is checked against
  ``pyarrow.compute.array_sort_indices`` / ``sort_indices`` directly. What pyarrow does with floats:
  values in IEEE comparison order, -0.0 and +0.0 equal (a tie that keeps input order), and every NaN
  placed next to the nulls in *both* directions -- after the values with ``null_placement="at_end"``,
  between the nulls and the values with ``"at_start"`` -- in input order.
* ``float_order="total"`` is IEEE 754 totalOrder, the order arrow-rs (``total_cmp``), DataFusion and
  Rust use: -NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN, NaNs by payload, and descending is
  the exact mirror. pyarrow has no such order, so the reference is the definition itself (the bit
  pattern with the sign-flip transform, sorted stably by numpy). Where the two orders agree -- no NaN
  and no -0.0 in the column -- the test also checks it against pyarrow.
"""
import struct

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

SIZES = [0, 1, 2, 33, 8191, 8192, 8193, 70_001]
OPTIONS = [(d, p) for d in (False, True) for p in ("at_end", "at_start")]

_SPECIAL64 = [struct.unpack("<d", struct.pack("<Q", b))[0] for b in (
    0x7FF8000000000000, 0xFFF8000000000000, 0x7FF0000000000001, 0xFFF8000000000042,
    0x7FF8000000000007, 0x0000000000000000, 0x8000000000000000, 0x7FF0000000000000,
    0xFFF0000000000000, 0x0000000000000001, 0x8000000000000003, 0x000FFFFFFFFFFFFF)]


def floats64(n, null_fraction, seed):
    rng = np.random.default_rng(seed)
    v = rng.uniform(-1e6, 1e6, n)
    dup = rng.random(n) < 0.25
    v[dup] = rng.integers(-20, 21, int(dup.sum())) / 4.0
    sp = rng.random(n) < 0.25
    v[sp] = np.array(_SPECIAL64)[rng.integers(0, len(_SPECIAL64), int(sp.sum()))]
    mask = rng.random(n) < null_fraction if null_fraction else None
    return pa.array(v, type=pa.float64(), mask=mask)


def floats32(n, null_fraction, seed):
    with np.errstate(invalid="ignore", over="ignore"):      # the largest doubles become +-inf
        a = floats64(n, 0, seed).to_numpy(zero_copy_only=False).astype(np.float32)
    rng = np.random.default_rng(seed + 1)
    sp = rng.random(n) < 0.1
    specials = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFC00042, 0x80000000, 0],
                        dtype=np.uint32).view(np.float32)
    a[sp] = specials[rng.integers(0, len(specials), int(sp.sum()))]
    mask = rng.random(n) < null_fraction if null_fraction else None
    return pa.array(a, type=pa.float32(), mask=mask)


def total_order_indices(arr, descending, null_placement):
    """arrow-rs's sort_to_indices with total_cmp, from its definition: stable by construction."""
    n = len(arr)
    valid = np.ones(n, bool) if arr.null_count == 0 else ~np.asarray(arr.is_null())
    raw = arr.to_numpy(zero_copy_only=False)
    width = 64 if arr.type == pa.float64() else 32
    bits = raw.astype(np.float64 if width == 64 else np.float32).view(np.uint64 if width == 64 else np.uint32)
    bits = bits.astype(np.uint64)
    sign = np.uint64(1 << (width - 1))
    full = np.uint64((1 << width) - 1) if width < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
    key = np.where(bits & sign, ~bits & full, bits | sign)
    if descending:
        key = ~key & full
    rows = np.nonzero(valid)[0]
    ordered = rows[np.argsort(key[rows], kind="stable")]
    nulls = np.nonzero(~valid)[0]
    out = np.concatenate([nulls, ordered] if null_placement == "at_start" else [ordered, nulls])
    return out.astype(np.int64).tolist()


def _ids(x):
    return x.to_arrow().to_pylist()


def _raw_bits(arr):
    """Validity and the stored bit pattern of every row of a float32 / float64 array."""
    valid = arr.is_valid().to_pylist()
    width = np.uint64 if arr.type == pa.float64() else np.uint32
    vals = arr.fill_null(0).to_numpy(zero_copy_only=False).view(width)
    return [int(b) if ok else None for b, ok in zip(vals, valid)]


def _bits(values):
    out = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            out.append(struct.unpack("<Q", struct.pack("<d", v))[0])
    return out


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("null_fraction", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("descending,placement", OPTIONS)
def test_ieee_order_is_pyarrows(n, null_fraction, descending, placement):
    v = floats64(n, null_fraction, n + 1)
    x = am.MetalArray.from_arrow(v)
    order = "descending" if descending else "ascending"
    want = pc.array_sort_indices(v, order=order, null_placement=placement).to_pylist()
    assert _ids(x.argsort(descending, null_placement=placement)) == want
    assert _ids(x.argsort(descending, null_placement=placement, float_order="ieee")) == want
    for k in (1, 17, 1500):
        assert _ids(x.top_k(k, largest=descending, null_placement=placement)) == want[:k]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("null_fraction", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("descending,placement", OPTIONS)
@pytest.mark.parametrize("make", [floats64, floats32])
def test_total_order_matches_its_definition(n, null_fraction, descending, placement, make):
    v = make(n, null_fraction, 7 * n + 3)
    x = am.MetalArray.from_arrow(v)
    want = total_order_indices(v, descending, placement)
    got = _ids(x.argsort(descending, null_placement=placement, float_order="total"))
    assert got == want
    # The sorted copy is exact to the bit, NaN payloads and zero signs included.
    sorted_ = x.sort(descending, null_placement=placement, float_order="total").to_arrow()
    expect = v.take(pa.array(want, pa.int64()))
    assert _raw_bits(sorted_) == _raw_bits(expect)
    for k in (1, 17, 1500):
        assert _ids(x.top_k(k, largest=descending, null_placement=placement, float_order="total")) == want[:k]


@pytest.mark.parametrize("descending,placement", OPTIONS)
def test_total_order_agrees_with_pyarrow_without_nan_or_negative_zero(descending, placement):
    rng = np.random.default_rng(5)
    v = rng.integers(-50, 50, 100_003) / 4.0
    v[v == 0] = 0.0                                    # +0.0 only
    v[::97] = np.inf
    v[1::89] = -np.inf
    arr = pa.array(v, mask=rng.random(len(v)) < 0.05)
    order = "descending" if descending else "ascending"
    want = pc.array_sort_indices(arr, order=order, null_placement=placement).to_pylist()
    got = _ids(am.MetalArray.from_arrow(arr).argsort(descending, null_placement=placement,
                                                     float_order="total"))
    assert got == want


def test_total_order_special_values():
    nan, inf = float("nan"), float("inf")
    neg_nan = struct.unpack("<d", struct.pack("<Q", 0xFFF8000000000000))[0]
    v = pa.array([1.0, nan, -0.0, None, neg_nan, 0.0, -inf, inf, -0.0, -1.0, None])
    x = am.MetalArray.from_arrow(v)
    assert _ids(x.argsort(float_order="total")) == [4, 6, 9, 2, 8, 5, 0, 7, 1, 3, 10]
    assert _ids(x.argsort(True, null_placement="at_start", float_order="total")) == \
        [3, 10, 1, 7, 0, 5, 2, 8, 9, 6, 4]
    # The default is pyarrow's order: the zeros tie and NaN stays next to the nulls.
    assert _ids(x.argsort(True)) == pc.array_sort_indices(v, order="descending").to_pylist()


def test_integer_and_string_keys_ignore_float_order():
    v = pa.array([3, None, 1, 3, None, 2], pa.int64())
    x = am.MetalArray.from_arrow(v)
    for d, p in OPTIONS:
        order = "descending" if d else "ascending"
        want = pc.array_sort_indices(v, order=order, null_placement=p).to_pylist()
        assert _ids(x.argsort(d, null_placement=p, float_order="total")) == want
        assert _ids(x.top_k(3, largest=d, null_placement=p, float_order="total")) == want[:3]


def test_bad_option_values_raise():
    x = am.MetalArray.from_arrow(pa.array([1.0, 2.0]))
    with pytest.raises(am.ArrowMetalError):
        x.argsort(float_order="bogus")
    with pytest.raises(am.ArrowMetalError):
        x.top_k(1, null_placement="middle")


@pytest.mark.parametrize("n", [8193, 70_001])
def test_lexsort_per_key_options(n):
    rng = np.random.default_rng(n)
    k1 = pa.array(rng.integers(0, 4, n), mask=rng.random(n) < 0.1)
    k2 = floats64(n, 0.1, n)
    for d1, p1 in OPTIONS:
        for d2, p2 in OPTIONS:
            got = _ids(am.lexsort_indices([k1, k2], [d1, d2], null_placement=[p1, p2],
                                          float_order=["ieee", "total"]))
            # Reference: rank rows by each key alone (tied rows share a rank), then a stable lexsort.
            r2 = total_order_indices(k2, d2, p2)
            r1 = pc.array_sort_indices(k1, order="descending" if d1 else "ascending",
                                       null_placement=p1).to_pylist()
            def ranks(order, col):
                vals = col.to_pylist()
                if col.type == pa.float64():
                    vals = _bits(vals)
                rank = [0] * n
                for j, i in enumerate(order):
                    rank[i] = j if j == 0 or vals[order[j - 1]] != vals[i] else rank[order[j - 1]]
                return rank
            a, b = ranks(r1, k1), ranks(r2, k2)
            want = sorted(range(n), key=lambda i: (a[i], b[i], i))
            assert got == want, (d1, p1, d2, p2)


def test_lexsort_single_options_keep_the_old_call():
    k = pa.array([2.0, None, float("nan"), -0.0, 0.0])
    assert _ids(am.lexsort_indices([k], [True], null_placement="at_start")) == \
        pc.array_sort_indices(k, order="descending", null_placement="at_start").to_pylist()


@pytest.mark.parametrize("limit", [None, 100])
def test_lazy_sort_options(limit):
    n = 70_001
    x = floats64(n, 0.05, 11)
    t = pa.table({"x": x, "row": pa.array(np.arange(n, dtype=np.int32))})
    want = total_order_indices(x, True, "at_start")
    q = am.scan(t).sort("x", descending=True, null_placement="at_start", float_order="total")
    if limit:
        q = q.limit(limit)
        want = want[:limit]
    assert q.collect().column("row").to_pylist() == want
    # The default keeps the plain [name, descending] key.
    plain = am.scan(t).sort("x", descending=True).limit(10).collect().column("row").to_pylist()
    assert plain == pc.array_sort_indices(x, order="descending").to_pylist()[:10]


def test_lazy_sort_option_lists_per_key():
    t = pa.table({"a": pa.array([1, None, 1, 2, None], pa.int64()),
                  "b": pa.array([0.0, 1.0, -0.0, float("nan"), None])})
    got = (am.scan(t).sort(["a", "b"], [False, True], null_placement=["at_start", "at_end"],
                           float_order="total").collect())
    # a: nulls first, then 1, 1, 2. Within a == 1: b descending in totalOrder, +0.0 before -0.0.
    # Within the null a's: b = 1.0 before the null b.
    assert got.column("a").to_pylist() == [None, None, 1, 1, 2]
    assert _bits(got.column("b").to_pylist()) == _bits([1.0, None, 0.0, -0.0, float("nan")])
