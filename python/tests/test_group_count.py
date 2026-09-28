"""count(expr) per group is the number of non-null values of expr, for every column type, through
the plan runner (the lazy API) and through the public group-by APIs, against pyarrow's count.

    PYTHONPATH=python python -m pytest python/tests/test_group_count.py -q
"""
import datetime
import decimal

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import arrowmetal as am

N = 2003


def _valid(frac, seed):
    """Deterministic validity with `frac` of the rows null."""
    s = seed * 2654435761 + 17
    out = []
    for _ in range(N):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append((s >> 11) / float(1 << 53) >= frac)
    return out


def _values(kind, valid):
    rows = range(N)
    if kind == "int32":
        return pa.array([i - 7 if v else None for i, v in zip(rows, valid)], pa.int32())
    if kind == "uint64":
        return pa.array([i if v else None for i, v in zip(rows, valid)], pa.uint64())
    if kind == "float64":
        return pa.array([(float("nan") if i % 5 == 0 else i / 8) if v else None for i, v in zip(rows, valid)],
                        pa.float64())
    if kind == "bool":
        return pa.array([i % 3 == 0 if v else None for i, v in zip(rows, valid)], pa.bool_())
    if kind == "utf8":
        return pa.array([f"s{i % 17}" if v else None for i, v in zip(rows, valid)], pa.string())
    if kind == "large_utf8":
        return pa.array([f"s{i % 17}" if v else None for i, v in zip(rows, valid)], pa.large_string())
    if kind == "utf8_view":
        return pa.array([f"a longer string {i}" if v else None for i, v in zip(rows, valid)], pa.string_view())
    if kind == "binary":
        return pa.array([b"b%d" % (i % 3) if v else None for i, v in zip(rows, valid)], pa.binary())
    if kind == "date32":
        return pa.array([datetime.date(2020, 1, 1) + datetime.timedelta(days=i % 900) if v else None
                         for i, v in zip(rows, valid)], pa.date32())
    if kind == "timestamp":
        return pa.array([i * 1_000_003 if v else None for i, v in zip(rows, valid)], pa.timestamp("us"))
    if kind == "decimal128":
        return pa.array([decimal.Decimal(i * 101 - 5000).scaleb(-2) if v else None for i, v in zip(rows, valid)],
                        pa.decimal128(12, 2))
    if kind == "list":
        return pa.array([[i] * (i % 3) if v else None for i, v in zip(rows, valid)], pa.list_(pa.int64()))
    if kind == "dictionary":
        return pa.array([f"d{i % 3}" if v else None for i, v in zip(rows, valid)], pa.string()).dictionary_encode()
    raise AssertionError(kind)


KINDS = ["int32", "uint64", "float64", "bool", "utf8", "large_utf8", "utf8_view", "binary", "date32",
         "timestamp", "decimal128", "list", "dictionary"]
FRACTIONS = [0.0, 0.1, 1.0]


def _table(kind, frac):
    valid = _valid(frac, int(frac * 10) + 3)
    fv = _valid(0.2, 99)
    return pa.table({
        "k": pa.array([None if i % 97 == 50 else (i * 7919) % 13 for i in range(N)], pa.int32()),
        "v": _values(kind, valid),
        "f": pa.array([i / 3 if ok else None for i, ok in zip(range(N), fv)], pa.float64()),
        "i": pa.array([i % 50 for i in range(N)], pa.int32()),
    })


def _want(t):
    """pyarrow's per-key count of non-null values of v, keyed by k (None for the null key)."""
    ks = t.column("k").to_pylist()
    valid = pc.is_valid(t.column("v")).to_pylist()
    out = {}
    for k, ok in zip(ks, valid):
        out[k] = out.get(k, 0) + (1 if ok else 0)
    return out


def _source(t):
    return {n: am.array(t.column(n).combine_chunks()) for n in t.column_names}


CONTEXTS = {
    "alone": [],
    "f64 sum": [lambda: am.agg.sum("f", "x")],
    "f64 mean": [lambda: am.agg.mean("f", "x")],
    "f64 min": [lambda: am.agg.min("f", "x")],
    "f64 max": [lambda: am.agg.max("f", "x")],
    "int sum": [lambda: am.agg.sum("i", "x")],
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("frac", FRACTIONS)
def test_plan_group_by_count_every_context(kind, frac):
    t = _table(kind, frac)
    want = _want(t)
    src = _source(t)
    for label, others in CONTEXTS.items():
        aggs = [f() for f in others] + [am.agg.count("c", "v")]
        got = am.scan(src).group_by("k").agg(*aggs).collect()
        got = dict(zip(got.column("k").to_pylist(), got.column("c").to_pylist()))
        assert got == want, f"{kind} frac={frac} {label}"


def test_minimal_reproductions():
    t = pa.table({"k": pa.array([0, 1, 0, 1], pa.int32()), "v": pa.array([1.0, 2.0, 3.0, 4.0])})
    for other in ("sum", "min", "max", "mean"):
        got = (am.scan(_source(t)).group_by("k")
               .agg(getattr(am.agg, other)("v", "a"), am.agg.count("n", "v")).collect())
        assert sorted(got.column("n").to_pylist()) == [2, 2], other
    t = pa.table({"k": pa.array([1, 1, 1, 1], pa.int64()), "s": pa.array(["a", None, "b", None]),
                  "v": pa.array([1.0, 2.0, 3.0, 4.0])})
    alone = am.scan(_source(t)).group_by("k").agg(am.agg.count("n", "s")).collect()
    assert alone.column("n").to_pylist() == [2]
    with_sum = am.scan(_source(t)).group_by("k").agg(am.agg.count("n", "s"), am.agg.sum("v", "t")).collect()
    assert with_sum.column("n").to_pylist() == [2]
    assert with_sum.column("t").to_pylist() == [10.0]


def test_count_of_a_computed_expression_and_empty_input():
    t = _table("int32", 0.1)
    v, i = t.column("v").to_pylist(), t.column("i").to_pylist()
    ks = t.column("k").to_pylist()
    want = {}
    for k, a, b in zip(ks, v, i):
        want[k] = want.get(k, 0) + (1 if a is not None and a > 100 else 0)
    expr = am.if_else(am.col("v") > 100, am.col("v"), am.null("int32"))
    for label, others in CONTEXTS.items():
        aggs = [f() for f in others] + [am.agg.count("c", expr)]
        got = am.scan(_source(t)).group_by("k").agg(*aggs).collect()
        assert dict(zip(got.column("k").to_pylist(), got.column("c").to_pylist())) == want, label
    empty = _table("utf8", 0.1).slice(0, 0)
    for label, others in CONTEXTS.items():
        aggs = [f() for f in others] + [am.agg.count("c", "v")]
        got = am.scan(_source(empty)).group_by("k").agg(*aggs).collect()
        assert got.num_rows == 0 and "c" in got.column_names, label


@pytest.mark.parametrize("kind", KINDS)
def test_public_group_by_count(kind):
    for frac in FRACTIONS:
        t = _table(kind, frac)
        want = _want(t)
        gb = am.group_by([t.column("k").combine_chunks()])
        got = dict(zip(gb.keys()[0].to_pylist(),
                       gb.count(am.array(t.column("v").combine_chunks())).to_arrow().to_pylist()))
        assert got == want, f"{kind} frac={frac}"


def test_dense_group_by_count_values_takes_any_type():
    keys = am.array(pa.array([i % 5 for i in range(N)], pa.int32()))
    for kind in ("float64", "utf8", "bool", "decimal128", "list"):
        for frac in FRACTIONS:
            valid = _valid(frac, 7)
            v = _values(kind, valid)
            want = [0] * 5
            for j, ok in enumerate(pc.is_valid(v).to_pylist()):
                want[j % 5] += 1 if ok else 0
            got = keys.group_by(5).count_values(am.array(v)).to_arrow().to_pylist()
            assert got == want, f"{kind} frac={frac}"


@pytest.mark.parametrize("kind", ["int32", "float64", "bool", "utf8", "decimal128", "list", "dictionary"])
def test_stream_group_by_count(kind):
    from arrowmetal import stream
    for frac in FRACTIONS:
        # A streaming key column has no nulls here: drop the null-key rows.
        t = _table(kind, frac).select(["k", "v"]).filter(pa.array([i % 97 != 50 for i in range(N)]))
        want = _want(t)
        for dense in (0, 13):
            got = (stream.scan_table(t, batch_rows=700).group_by(["k"], dense_key_count=dense)
                   .agg([("count", "v", "c"), ("count", None, "n")]))
            got = dict(zip(got.column("k").to_pylist(), got.column("c").to_pylist()))
            assert got == want, f"{kind} frac={frac} dense={dense}"
