"""The grouped aggregates, measured where they were losing: a small number of groups.

A narrow stand-in for the group-by family of `Benchmarks/full_matrix.py`. Same seed, the same column
distributions and the same rule (one warm-up, then best of five under a 1.2 s budget), but only the
grouped aggregates, so a before/after can be taken in a few minutes rather than a forty-minute matrix
run. The columns are drawn from full_matrix's seed in this script's own order, so the *values* are not
byte for byte the matrix's (its `Data` draws lazily from one shared generator in whatever order the
whole matrix asks); the distributions, the sizes and the cardinalities are.

    PYTHONPATH=python python Benchmarks/loss_groupby_small.py --tag before --csv out.csv
    PYTHONPATH=python python Benchmarks/loss_groupby_small.py --rows 10000000
    PYTHONPATH=python python Benchmarks/loss_groupby_small.py --cpu          # + pyarrow / Polars / pandas
    PYTHONPATH=python python Benchmarks/loss_groupby_small.py --sweep        # the shape sweep
    PYTHONPATH=python python Benchmarks/loss_groupby_small.py --digest       # answers, not timings

Three modes:

* the default measures the matrix's own group-by rows at a small number of groups;
* `--sweep` walks one axis at a time away from a base shape — row count, group count, key
  distribution, key nulls, value type, value nulls, a sliced input — so a change can be shown not to
  have helped one shape at another's expense;
* `--digest` prints a hash of every answer instead of a timing, so two builds can be compared for
  bit-identical results over the same shapes.

Nothing here is tuned to a benchmark row: the script only measures. Run one process at a time, with
nothing else on the machine, and warm the buffer pool (this script does, with a sort and a sum) before
trusting the first number in a process — see the cold-pool caveat in docs/TESTING.md.
"""
import argparse
import csv
import gc
import hashlib
import os
import sys
import time

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import arrowmetal as am                                                     # noqa: E402

SEED = 20260906            # full_matrix.py's seed
ITERS, BUDGET = 5, 1.2     # full_matrix.py's Bench(5, 1.2)


def best_of(fn):
    """(wall_ms, iterations): best of up to ITERS calls after one warm-up, full_matrix's rule."""
    fn()
    best, total, n = float("inf"), 0.0, 0
    while n < ITERS and (n < 2 or total < BUDGET):
        t0 = time.perf_counter()
        fn()
        w = time.perf_counter() - t0
        best = min(best, w)
        total += w
        n += 1
    return best * 1000.0, n


def fixed_width_utf8(codes, width=12, lead=b"k"):
    """full_matrix.py's utf8 key builder: fixed-width keys straight into Arrow buffers."""
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


# ---------------------------------------------------------------- the matrix rows

class Columns:
    """full_matrix's group-by columns for one row count, built on demand."""

    def __init__(self, n):
        self.n = n
        self.rng = np.random.default_rng(SEED)
        self.i64 = pa.array(self.rng.integers(-(2 ** 62), 2 ** 62, size=n, dtype=np.int64))
        self.f64 = pa.array(self.rng.random(n) * 2e9 - 1e9)
        self.g_i64 = am.array(self.i64)
        self.g_f64 = am.array(self.f64)
        self._keys = {}
        self._str = {}

    def keys(self, distinct):
        if distinct not in self._keys:
            v = self.rng.integers(0, min(distinct, self.n), size=self.n, dtype=np.int32)
            self._keys[distinct] = (pa.array(v), am.array(pa.array(v)))
        return self._keys[distinct]

    def str_keys(self, distinct):
        if distinct not in self._str:
            codes = self.rng.integers(0, min(distinct, self.n), size=self.n, dtype=np.int32)
            a = fixed_width_utf8(codes)
            self._str[distinct] = (a, am.array(a))
        return self._str[distinct]

    def two_keys(self, distinct):
        side = max(2, int(np.ceil(np.sqrt(distinct))))
        a = pa.array(self.rng.integers(0, side, size=self.n, dtype=np.int32))
        b = pa.array(self.rng.integers(0, side, size=self.n, dtype=np.int32))
        return side * side, (a, am.array(a)), (b, am.array(b))

    def warm(self):
        """The buffer pool is cold in a fresh process; give it the big buffers before timing.

        Called *after* every column exists, because building the columns is itself what makes the pool
        park and remap buffers — warming before them left the first timed row reading twice its settled
        value (the cold-pool caveat in docs/TESTING.md).
        """
        for _ in range(2):
            self.g_f64.sort()
            self.g_i64.sum()
            am.group_by([self.keys(1000)[1]]).sum(self.g_i64)
            am.group_by([self.keys(1000)[1]]).min(self.g_f64)


def matrix_cases(c):
    """(op, arrowmetal callable, per-library callables) in the matrix's own order."""
    out = []
    for distinct in (1_000, 100_000, 10_000_000):
        if distinct > c.n:
            continue
        ka, kg = c.keys(distinct)
        for agg in ("sum", "count", "mean", "min", "max"):
            out.append((f"{agg} by int32 key ({distinct} groups)",
                        (lambda a=agg, k=kg: getattr(am.group_by([k]), a)(c.g_i64)),
                        dict(pyarrow=(lambda a=agg, k=ka: pa.table({"k": k, "x": c.i64})
                                      .group_by("k").aggregate([("x", a)])))))
        for agg in ("min", "max"):
            out.append((f"{agg} by key, float64 ({distinct} groups)",
                        (lambda a=agg, k=kg: getattr(am.group_by([k]), a)(c.g_f64)),
                        dict(pyarrow=(lambda a=agg, k=ka: pa.table({"k": k, "x": c.f64})
                                      .group_by("k").aggregate([("x", a)])))))
        for agg in ("variance", "stddev"):
            out.append((f"{agg} by key ({distinct} groups)",
                        (lambda a=agg, k=kg: getattr(am.group_by([k]), a)(c.g_f64, 1)),
                        dict(pyarrow=(lambda a=agg, k=ka: pa.table({"k": k, "x": c.f64})
                                      .group_by("k").aggregate([("x", a)])))))
        if distinct <= 100_000:
            sa, sg = c.str_keys(distinct)
            out.append((f"sum by utf8 key ({distinct} distinct)",
                        (lambda k=sg: am.group_by([k]).sum(c.g_i64)),
                        dict(pyarrow=(lambda k=sa: pa.table({"k": k, "x": c.i64})
                                      .group_by("k").aggregate([("x", "sum")])))))
        groups, (aa, ag), (ba, bg) = c.two_keys(distinct)
        out.append((f"sum by two int32 keys (~{groups} groups)",
                    (lambda x=ag, y=bg: am.group_by([x, y]).sum(c.g_i64)),
                    dict(pyarrow=(lambda x=aa, y=ba: pa.table({"a": x, "b": y, "x": c.i64})
                                  .group_by(["a", "b"]).aggregate([("x", "sum")])))))
    return out


# ---------------------------------------------------------------- the shape sweep

BASE = dict(rows=10_000_000, groups=1_000, keys="uniform", key_nulls=0.0,
            values="int64", value_nulls=0.0, offset=0)

AXES = [
    ("rows", [1_000_000, 3_000_000, 10_000_000, 27_000_000, 50_000_000]),
    ("groups", [1, 2, 10, 100, 1_000, 1_025, 5_000, 20_000, 100_000, 1_000_000, 10_000_000]),
    ("keys", ["uniform", "skewed", "sorted"]),
    ("key_nulls", [0.0, 0.10]),
    ("values", ["int64", "float64"]),
    ("value_nulls", [0.0, 0.10]),
    ("offset", [0, 33]),
]

SWEEP_AGGS = ("sum", "count", "mean", "min", "max", "variance")


def sweep_shapes(base=None):
    """One axis at a time away from BASE, deduplicated, in a stable order.

    A full cross product of the seven axes is 4,620 shapes times six aggregates, which is a day of
    measurement; walking each axis from one base point covers every value of every axis and is what
    catches a change that helped one shape at another's expense.
    """
    base = base or BASE
    seen, out = set(), []
    for axis, values in AXES:
        for v in values:
            shape = dict(base)
            shape[axis] = v
            if shape["groups"] > shape["rows"]:
                continue
            key = tuple(sorted(shape.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(shape)
    return out


def build_shape(shape, rng):
    """(keys MetalArray, values MetalArray, label) for one sweep shape."""
    n, K = shape["rows"], shape["groups"]
    if shape["keys"] == "skewed":
        k = np.where(rng.random(n) < 0.90, 0, rng.integers(0, K, size=n)).astype(np.int32)
    else:
        k = rng.integers(0, K, size=n, dtype=np.int32)
        if shape["keys"] == "sorted":
            k.sort()
    kmask = rng.random(n) < shape["key_nulls"] if shape["key_nulls"] else None
    ka = pa.array(k, mask=kmask)
    if shape["values"] == "int64":
        v = rng.integers(-(2 ** 62), 2 ** 62, size=n, dtype=np.int64)
    else:
        v = rng.random(n) * 2e9 - 1e9
    vmask = rng.random(n) < shape["value_nulls"] if shape["value_nulls"] else None
    va = pa.array(v, mask=vmask)
    off = shape["offset"]
    if off:
        ka, va = ka.slice(off), va.slice(off)
    return am.array(ka), am.array(va), shape


def shape_label(shape):
    return (f"rows={shape['rows']} groups={shape['groups']} keys={shape['keys']} "
            f"knull={shape['key_nulls']} values={shape['values']} vnull={shape['value_nulls']} "
            f"offset={shape['offset']}")


def sweep_call(gb, agg, values):
    if agg == "count":
        return gb.count(values)
    if agg == "variance":
        return gb.variance(values, 1)
    return getattr(gb, agg)(values)


# ---------------------------------------------------------------- answers

def digest(result):
    """A stable hash of an ArrowMetal result column, nulls included."""
    a = result.to_arrow() if hasattr(result, "to_arrow") else result
    h = hashlib.blake2b(digest_size=16)
    h.update(str(a.type).encode())
    h.update(str(len(a)).encode())
    for buf in a.buffers():
        h.update(b"-" if buf is None else memoryview(buf).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- drivers

def run_matrix(args, writer):
    for n in args.rows:
        c = Columns(n)
        cases = matrix_cases(c)          # every column exists before the pool is warmed
        c.warm()
        for op, fn, cpu in cases:
            wall, iters = best_of(fn)
            line(args, writer, "matrix", op, n, "arrowmetal", wall, iters)
            if args.cpu:
                for lib, cfn in cpu.items():
                    w, i = best_of(cfn)
                    line(args, writer, "matrix", op, n, lib, w, i)
            gc.collect()
        del c
        gc.collect()


def run_sweep(args, writer):
    for shape in sweep_shapes(args.base):
        rng = np.random.default_rng(SEED)
        keys, values, _ = build_shape(shape, rng)
        # Warm the pool for this shape's buffer sizes before the first timed call.
        am.group_by([keys]).min(values)
        for agg in SWEEP_AGGS:
            if agg == "variance" and shape["values"] == "int64":
                continue          # the matrix measures variance on float64
            wall, iters = best_of(lambda a=agg, k=keys, v=values: sweep_call(am.group_by([k]), a, v))
            line(args, writer, "sweep", f"{agg} | {shape_label(shape)}", shape["rows"],
                 "arrowmetal", wall, iters)
        del keys, values
        gc.collect()


def run_digest(args, writer):
    for shape in sweep_shapes(args.base):
        rng = np.random.default_rng(SEED)
        keys, values, _ = build_shape(shape, rng)
        gb = am.group_by([keys])
        for agg in SWEEP_AGGS:
            if agg == "variance" and shape["values"] == "int64":
                continue
            d = digest(sweep_call(gb, agg, values))
            row = dict(tag=args.tag, mode="digest", op=f"{agg} | {shape_label(shape)}",
                       rows=shape["rows"], library="arrowmetal", wall_ms="", iters="", digest=d)
            print(f"{args.tag:<8} {row['op'][:96]:<98} {d}", flush=True)
            if writer:
                writer.writerow(row)
        del gb, keys, values
        gc.collect()


def line(args, writer, mode, op, rows, library, wall, iters):
    print(f"{args.tag:<8} {op[:96]:<98} {rows:>11,} {library:<11} {wall:9.3f} ms ({iters})", flush=True)
    if writer:
        writer.writerow(dict(tag=args.tag, mode=mode, op=op, rows=rows, library=library,
                             wall_ms=f"{wall:.4f}", iters=iters, digest=""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[10_000_000, 50_000_000])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--csv")
    ap.add_argument("--cpu", action="store_true", help="also measure pyarrow on the matrix rows")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--values", choices=["int64", "float64"], default="int64",
                    help="element type of the sweep's base shape (float64 is what carries variance)")
    args = ap.parse_args()
    args.base = dict(BASE, values=args.values)

    handle = writer = None
    if args.csv:
        new = not os.path.exists(args.csv)
        handle = open(args.csv, "a", newline="")
        writer = csv.DictWriter(handle, ["tag", "mode", "op", "rows", "library", "wall_ms",
                                         "iters", "digest"])
        if new:
            writer.writeheader()
    try:
        if args.digest:
            run_digest(args, writer)
        elif args.sweep:
            run_sweep(args, writer)
        else:
            run_matrix(args, writer)
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    main()
