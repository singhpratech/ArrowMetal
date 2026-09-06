"""Float64 element-wise math at 50M rows: ArrowMetal's software binary64 against numpy, pyarrow and
Polars, plus the float32 column for scale.

    PYTHONPATH=python python Benchmarks/float64_math_bench.py            # 50M rows, 5 iterations
    PYTHONPATH=python python Benchmarks/float64_math_bench.py 5000000 3  # smaller and faster

Apple GPUs have no double-precision hardware, so every float64 arithmetic and transcendental kernel
here is IEEE-754 binary64 written out in software over `ulong` bit patterns (`Kernels/DoubleMath.swift`,
`Kernels/DoubleTranscendental.swift`, `Kernels/DoublePower.swift`). That is arithmetically honest — the
answers are correctly rounded or within an ulp — and it is *compute* bound rather than memory bound, so
the numbers below are the ones to quote, not a bandwidth figure. The float32 rows are the same kernels
in hardware and show what the software path costs.

Each row reports the best of `iters` runs after one warm-up, in milliseconds and in GB/s over the bytes
actually moved (one read and one write per element, two reads for a two-column op).
"""
import sys, time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import arrowmetal as am

try:
    import polars as pl
except ImportError:                                                   # pragma: no cover
    pl = None

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def best(fn, iters=ITERS):
    """Best wall time of `iters` runs, in seconds. One untimed warm-up first."""
    fn()
    t = float("inf")
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - s)
    return t


def main():
    rng = np.random.default_rng(0x5EED)
    # Positive and well spread, so the logarithms and roots see the whole exponent range rather than
    # one comfortable decade.
    x64 = np.exp(rng.uniform(-30.0, 30.0, ROWS))
    y64 = rng.uniform(0.5, 4.0, ROWS)
    x32, y32 = x64.astype(np.float32), y64.astype(np.float32)

    ax, ay = pa.array(x64), pa.array(y64)
    ax32 = pa.array(x32)
    gx, gy = am.MetalArray.from_arrow(ax), am.MetalArray.from_arrow(ay)
    gx32 = am.MetalArray.from_arrow(ax32)
    px = pl.Series(x64) if pl else None
    py = pl.Series(y64) if pl else None

    B1, B2 = ROWS * 16, ROWS * 24                                     # bytes moved: 1 or 2 reads + 1 write

    rows = [
        ("sqrt   (float64)", B1, lambda: gx.sqrt(), lambda: np.sqrt(x64),
         lambda: pc.sqrt(ax), (lambda: px.sqrt()) if pl else None),
        ("exp    (float64)", B1, lambda: gy.exp(), lambda: np.exp(y64),
         lambda: pc.exp(ay), (lambda: py.exp()) if pl else None),
        ("ln     (float64)", B1, lambda: gx.ln(), lambda: np.log(x64),
         lambda: pc.ln(ax), (lambda: px.log()) if pl else None),
        ("log2   (float64)", B1, lambda: gx.log2(), lambda: np.log2(x64),
         lambda: pc.log2(ax), (lambda: px.log(2.0)) if pl else None),
        ("log10  (float64)", B1, lambda: gx.log10(), lambda: np.log10(x64),
         lambda: pc.log10(ax), (lambda: px.log10()) if pl else None),
        ("power  (float64 ** 2.5)", B1, lambda: gx.power(2.5), lambda: np.power(x64, 2.5),
         lambda: pc.power(ax, pa.scalar(2.5)), (lambda: px ** 2.5) if pl else None),
        ("power  (float64 ** float64)", B2, lambda: gx.power(gy), lambda: np.power(x64, y64),
         lambda: pc.power(ax, ay), (lambda: px ** py) if pl else None),
        ("divide (float64 / float64)", B2, lambda: gx / gy, lambda: x64 / y64,
         lambda: pc.divide(ax, ay), (lambda: px / py) if pl else None),
        ("multiply (float64)", B2, lambda: gx * gy, lambda: x64 * y64,
         lambda: pc.multiply(ax, ay), (lambda: px * py) if pl else None),
        ("add    (float64)", B2, lambda: gx + gy, lambda: x64 + y64,
         lambda: pc.add(ax, ay), (lambda: px + py) if pl else None),
        ("sin    (float64)", B1, lambda: gx.sin(), lambda: np.sin(x64),
         lambda: pc.sin(ax), (lambda: px.sin()) if pl else None),
        # The same shapes in float32, where the GPU has real hardware for them.
        ("sqrt   (float32)", ROWS * 8, lambda: gx32.sqrt(), lambda: np.sqrt(x32),
         lambda: pc.sqrt(ax32), None),
        ("ln     (float32)", ROWS * 8, lambda: gx32.ln(), lambda: np.log(x32),
         lambda: pc.ln(ax32), None),
        ("power  (float32 ** 2.5)", ROWS * 8, lambda: gx32.power(np.float32(2.5)),
         lambda: np.power(x32, np.float32(2.5)),
         lambda: pc.power(ax32, pa.scalar(np.float32(2.5), pa.float32())), None),
    ]

    print(f"float64 element-wise math, {ROWS:,} rows, best of {ITERS}\n")
    head = f"{'op':28s} {'ArrowMetal':>12s} {'GB/s':>7s} {'numpy':>10s} {'pyarrow':>10s} {'polars':>10s} {'ratio':>7s}"
    print(head)
    print("-" * len(head))
    for name, nbytes, gpu, npf, paf, plf in rows:
        try:
            g = best(gpu)
        except Exception as e:                                        # a row the build does not support
            print(f"{name:28s} {'n/a':>12s}   ({type(e).__name__}: {e})")
            continue
        cpu = {}
        for label, fn in (("numpy", npf), ("pyarrow", paf), ("polars", plf)):
            if fn is None:
                continue
            try:
                cpu[label] = best(fn)
            except Exception:
                pass
        fastest = min(cpu.values()) if cpu else float("nan")
        cell = lambda k: f"{cpu[k] * 1e3:9.1f}ms" if k in cpu else f"{'-':>11s}"
        print(f"{name:28s} {g * 1e3:9.1f}ms {nbytes / g / 1e9:7.1f} "
              f"{cell('numpy')} {cell('pyarrow')} {cell('polars')} {fastest / g:6.2f}x")
    print("\nratio = fastest CPU baseline / ArrowMetal (above 1 means the GPU wins)")


if __name__ == "__main__":
    main()
