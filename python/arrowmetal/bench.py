"""CPU against Metal on this Mac, in 30 seconds or less: `python -m arrowmetal.bench`.

One seeded dataset (10,000,000 rows by default), four operations everyone knows (sum, filter, sort,
group-by sum), each measured the way `Benchmarks/full_matrix.py` measures the published matrix: one
warm-up, the best of up to five calls, wall milliseconds with the process CPU milliseconds of that
same call beside them. pyarrow is always measured; Polars is measured when it is installed. Every
ArrowMetal answer is checked against pyarrow's before anything is printed.

The data: int64 `v` uniform in [-1,000,000, 1,000,000) with 10% nulls, float64 `f` standard normal,
int32 `k` uniform over 1,000 keys, all drawn with pyarrow.compute from seeded SplitMix64 streams
(`make_data`), so the bench needs nothing beyond `pip install arrowmetal` (no NumPy).

Nothing is written and nothing is sent: the script prints, and the "Share it" block at the end is
text to paste into a GitHub issue or the Discord channel if you want to.

    python -m arrowmetal.bench                # 10,000,000 rows
    python -m arrowmetal.bench --rows 2000000
    python -m arrowmetal.bench --json         # one JSON object instead of the text report
    python -m arrowmetal.bench --quiet        # the table only
    python -m arrowmetal.bench --no-share     # no "Share it" block
    python -m arrowmetal.bench --no-polars    # skip Polars even when it is installed
    python -m arrowmetal.bench --parquet data.parquet   # your own file instead of generated data

`--parquet` reads the file's integer, floating-point and string columns with pyarrow, Polars and
ArrowMetal, then runs sum and filter on the largest numeric column and group-by sum keyed on the
lowest-cardinality integer or string column, CPU against Metal. It refuses a file whose columns
would take more than a quarter of physical memory (the bench holds them up to four times). The
report and the "Share it" block carry the file's row count, column count, codecs and the timings,
never its path, column names or values.
"""
import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import urllib.parse

import pyarrow as pa
import pyarrow.compute as pc

import arrowmetal as am

SEED = 20260907
KEYS = 1_000
NULL_FRACTION = 0.10
ISSUE_URL = "https://github.com/singhpratech/ArrowMetal/issues/new?template=benchmark_result.yml&title="


# ---- timing, as Benchmarks/full_matrix.py does it

def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Bench:
    """One warm-up, then the best of up to `iters` calls (never fewer than two) within `budget` s."""

    def __init__(self, iters=5, budget=1.2):
        self.iters, self.budget = iters, budget

    def run(self, fn):
        """(wall_ms, cpu_ms, iterations) of the best call; cpu_ms is the process CPU time (all
        threads) of that same call, so a threaded library shows the cores it used."""
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


# ---- the machine

def _sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def machine_info():
    mem = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown CPU",
        "cores": os.cpu_count() or 0,
        "memory_gb": round(int(mem) / 2**30) if mem.isdigit() else None,
        "macos": platform.mac_ver()[0] or platform.release(),
        "gpu": am.device_name(),
    }


def machine_line(m, versions):
    mem = f"{m['memory_gb']} GB" if m["memory_gb"] else "memory unknown"
    libs = f"ArrowMetal {versions['arrowmetal']}, pyarrow {versions['pyarrow']}"
    if versions.get("polars"):
        libs += f", Polars {versions['polars']}"
    return f"{m['chip']}, {m['cores']} cores, {mem}, macOS {m['macos']}, GPU {m['gpu']}; {libs}"


# ---- the data

_M64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


def _mix64(z):
    """SplitMix64's finaliser on a Python int."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _u64(v):
    return pa.scalar(v, pa.uint64())


def _random_u64(rows, seed, stream):
    """`rows` uniform uint64s: SplitMix64 (state_i = start + i * golden, then the finaliser) computed
    column-wise with pyarrow.compute, whose unchecked integer kernels wrap modulo 2^64. Deterministic
    for (seed, stream), and no NumPy."""
    start = _mix64((seed * 0x100 + stream) & _M64)
    z = pc.add(pc.cumulative_sum(pa.repeat(_u64(_GOLDEN), rows)), _u64(start))
    z = pc.multiply(pc.bit_wise_xor(z, pc.shift_right(z, _u64(30))), _u64(0xBF58476D1CE4E5B9))
    z = pc.multiply(pc.bit_wise_xor(z, pc.shift_right(z, _u64(27))), _u64(0x94D049BB133111EB))
    return pc.bit_wise_xor(z, pc.shift_right(z, _u64(31)))


def _below(u, m):
    """`u mod m` for uint64 `u` (the bias is m / 2^64, below 1e-12 for the bounds used here)."""
    return pc.subtract(u, pc.multiply(pc.divide(u, _u64(m)), _u64(m)))


def _unit(u, open_low=False):
    """The top 53 bits of `u` as a float64 in [0, 1), or (0, 1] with `open_low`."""
    top = pc.shift_right(u, _u64(11))
    if open_low:
        top = pc.add(top, _u64(1))
    return pc.multiply(pc.cast(top, pa.float64()), 2.0 ** -53)


def make_data(rows, seed=SEED):
    """int64 `v` uniform in [-1,000,000, 1,000,000) with ~10% nulls, float64 `f` standard normal,
    int32 `k` uniform over 1,000 distinct keys.

    Drawn with pyarrow.compute only, so the bench runs on a plain `pip install arrowmetal`: five
    SplitMix64 streams (`_random_u64`), `v` and `k` by modulo, the null mask by comparing a stream
    against 10% of 2^64, and `f` by the Box-Muller transform of two uniform streams.
    """
    v = pc.cast(pc.subtract(pc.cast(_below(_random_u64(rows, seed, 0), 2_000_000), pa.int64()), 1_000_000),
                pa.int64())
    nulls = pc.less(_random_u64(rows, seed, 1), _u64(int(NULL_FRACTION * 2**64)))
    v = pc.if_else(nulls, pa.scalar(None, pa.int64()), v)
    u1 = _unit(_random_u64(rows, seed, 2), open_low=True)
    u2 = _unit(_random_u64(rows, seed, 3))
    f = pc.multiply(pc.sqrt(pc.multiply(pc.ln(u1), -2.0)), pc.cos(pc.multiply(u2, 2.0 * math.pi)))
    k = pc.cast(_below(_random_u64(rows, seed, 4), KEYS), pa.int32())
    return {"v": v, "f": f, "k": k}


def _polars(enabled):
    if not enabled:
        return None
    try:
        import polars as pl
    except Exception:
        return None
    return pl


# ---- the four operations

OPS = ["sum int64 (10% nulls)", "filter int64 > 0", "sort float64", "sum by int32 key (1000 groups)"]


def _group_dict(keys, sums):
    return dict(zip(keys.to_pylist(), sums.to_pylist()))


def run(rows, use_polars=True, bench=None):
    """Measure everything; returns a dict with the machine, the versions, the timings, the import
    time and the match flag. Raises nothing on a mismatch: `result["match"]` says."""
    bench = bench or Bench()
    t_all = time.perf_counter()
    pl = _polars(use_polars)

    t0 = time.perf_counter()
    d = make_data(rows)
    gen_s = time.perf_counter() - t0
    v, f, k = d["v"], d["f"], d["k"]
    tbl = pa.table({"k": k, "v": v})

    if pl is not None:
        pv, pf = pl.from_arrow(v), pl.from_arrow(f)
        pdf = pl.from_arrow(tbl)

    # The Metal columns are imported once, before any warm-up, and stay resident: the operations
    # are timed on resident columns, as the published matrix times them, and the import is its own
    # line in the report.
    t0 = time.perf_counter()
    gv, gf, gk = am.MetalArray.from_arrow(v), am.MetalArray.from_arrow(f), am.MetalArray.from_arrow(k)
    import_ms = (time.perf_counter() - t0) * 1000.0

    cases = {
        OPS[0]: {
            "pyarrow": lambda: pc.sum(v),
            "polars": (lambda: pv.sum()) if pl is not None else None,
            "arrowmetal": lambda: gv.sum(),
        },
        OPS[1]: {
            "pyarrow": lambda: pc.filter(v, pc.greater(v, 0)),
            "polars": (lambda: pv.filter(pv > 0)) if pl is not None else None,
            "arrowmetal": lambda: gv.filter_where(">", 0),
        },
        OPS[2]: {
            "pyarrow": lambda: pc.take(f, pc.sort_indices(f)),
            "polars": (lambda: pf.sort()) if pl is not None else None,
            "arrowmetal": lambda: gf.sort(),
        },
        OPS[3]: {
            "pyarrow": lambda: tbl.group_by("k").aggregate([("v", "sum")]),
            "polars": (lambda: pdf.group_by("k").agg(pl.sum("v"))) if pl is not None else None,
            "arrowmetal": lambda: am.group_by([gk]).sum(gv),
        },
    }

    timings = {}
    for op, impls in cases.items():
        row = {}
        for lib, fn in impls.items():
            if fn is None:
                continue
            w, c, n = bench.run(fn)
            row[lib] = {"wall_ms": round(w, 3), "cpu_ms": round(c, 3), "iterations": n}
        cpu_best = min(row[lib]["wall_ms"] for lib in ("pyarrow", "polars") if lib in row)
        row["fastest_cpu"] = min((lib for lib in ("pyarrow", "polars") if lib in row),
                                 key=lambda lib: row[lib]["wall_ms"])
        row["speedup"] = round(cpu_best / row["arrowmetal"]["wall_ms"], 2) if row["arrowmetal"]["wall_ms"] > 0 else None
        timings[op] = row

    # Every ArrowMetal answer against pyarrow's, computed once more outside the timing.
    problems = []
    if gv.sum() != pc.sum(v).as_py():
        problems.append(f"{OPS[0]}: ArrowMetal {gv.sum()} vs pyarrow {pc.sum(v).as_py()}")
    got, want = gv.filter_where(">", 0).to_arrow(), pc.filter(v, pc.greater(v, 0))
    if len(got) != len(want):
        problems.append(f"{OPS[1]}: ArrowMetal keeps {len(got)} rows vs pyarrow {len(want)}")
    elif not got.equals(want):
        problems.append(f"{OPS[1]}: same row count, different rows")
    got, want = gf.sort().to_arrow(), pc.take(f, pc.sort_indices(f))
    if not got.equals(want):
        problems.append(f"{OPS[2]}: sorted values differ from pyarrow's sort_indices order")
    gb = am.group_by([gk])
    got = _group_dict(gb.keys()[0], gb.sum(gv).to_arrow())
    pt = tbl.group_by("k").aggregate([("v", "sum")])
    want = _group_dict(pt.column("k"), pt.column("v_sum"))
    if got != want:
        bad = [key for key in want if got.get(key) != want[key]]
        problems.append(f"{OPS[3]}: {len(bad)} of {len(want)} group sums differ, first key {bad[:1]}")

    versions = {"arrowmetal": am.__version__, "pyarrow": pa.__version__,
                "polars": pl.__version__ if pl is not None else None,
                "python": platform.python_version()}
    return {
        "machine": machine_info(),
        "versions": versions,
        "rows": rows,
        "keys": KEYS,
        "import_ms": round(import_ms, 3),
        "generate_s": round(gen_s, 3),
        "timings": timings,
        "match": not problems,
        "problems": problems,
        "protocol": "one warm-up, best of up to 5 calls, wall ms with the process CPU ms of that call; "
                    "ArrowMetal columns resident on the GPU, import timed separately",
        "total_s": round(time.perf_counter() - t_all, 3),
    }


# ---- the report

def _cell(t):
    return f"{t['wall_ms']:.2f} ({t['cpu_ms']:.1f})" if t else "-"


def _speedup(x):
    return f"{x:.2f}x" if x is not None else "-"


def table_rows(result):
    have_polars = result["versions"].get("polars") is not None
    head = ["op", "rows", "pyarrow ms (cpu-ms)"]
    if have_polars:
        head.append("Polars ms (cpu-ms)")
    head += ["ArrowMetal ms (cpu-ms)", "speedup"]
    body = []
    for op, t in result["timings"].items():
        cells = [op, f"{result['rows']:,}", _cell(t.get("pyarrow"))]
        if have_polars:
            cells.append(_cell(t.get("polars")))
        cells += [_cell(t.get("arrowmetal")), _speedup(t["speedup"])]
        body.append(cells)
    return head, body


def text_table(result):
    head, body = table_rows(result)
    widths = [max(len(r[i]) for r in [head] + body) for i in range(len(head))]
    lines = []
    for r in [head] + body:
        cells = [r[0].ljust(widths[0])] + [r[i].rjust(widths[i]) for i in range(1, len(r))]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def markdown_table(result):
    head, body = table_rows(result)
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def issue_url(result, share=None):
    """The new-issue link. `share` prefills the form's `share` field (the Share it block)."""
    if "file" in result:
        f = result["file"]
        title = (f"bench --parquet: {result['machine']['chip']}, {f['rows']:,} rows, {f['columns']} columns, "
                 f"{'/'.join(f['codecs']) or 'no codec'}")
    else:
        title = f"bench: {result['machine']['chip']}, {result['rows']:,} rows"
    url = ISSUE_URL + urllib.parse.quote(title, safe="")
    if share is not None:
        url += "&share=" + urllib.parse.quote("```\n" + share + "\n```", safe="")
    return url


def report(result, quiet=False, share=True):
    m, versions = result["machine"], result["versions"]
    line = machine_line(m, versions)
    out = []
    if not quiet:
        out.append(line)
        out.append("")
    out.append(text_table(result))
    if quiet:
        return "\n".join(out)
    out.append("")
    out.append("results match pyarrow" if result["match"] else "MISMATCH against pyarrow")
    out.append("")
    polars_note = ", Polars its thread pool" if versions.get("polars") else ""
    out.append("Protocol: one warm-up, best of up to 5 calls, wall ms with the process CPU ms of that call in "
               f"parentheses; pyarrow uses its thread pool except for sort_indices, which is single-threaded in pyarrow 25{polars_note}; "
               "speedup is against the fastest CPU number on the row.")
    out.append(f"ArrowMetal columns resident on the GPU; import cost shown separately: importing the three columns "
               f"once took {result['import_ms']:.1f} ms. Data generation {result['generate_s']:.1f} s, "
               f"whole run {result['total_s']:.1f} s. Router mode: {os.environ.get('ARROWMETAL_ROUTER', 'auto')} (the default).")
    if share:
        out.append("")
        out.append("Share it (nothing is sent by this script; copy the block below):")
        out.append("")
        out.append("```")
        out.append(line)
        out.append("")
        out.append(markdown_table(result))
        out.append("```")
        out.append("")
        out.append(f"Open a prefilled issue: {issue_url(result)}")
        out.append("or drop it in the #benchmarks channel of the ArrowMetal Discord (link in the README).")
    return "\n".join(out)


# ---- --parquet: the user's own file
#
# Only the file's shape reaches the report: row count, column count, codecs, row groups, size, the
# types of the columns chosen, the group count and the timings. Never the path, a column name or a
# value (the filter threshold is the column's median and is printed as "median", not as a number).

MEMORY_SHARE = 4    # the bench holds the columns it reads up to four times at once


def physical_memory():
    """Bytes of physical memory, or None when sysctl does not say."""
    mem = _sysctl("hw.memsize")
    return int(mem) if mem.isdigit() else None


def _kind(t):
    """'numeric', 'string' or None: the column types `--parquet` uses."""
    if pa.types.is_dictionary(t):
        t = t.value_type
    if pa.types.is_integer(t) or (pa.types.is_floating(t) and not pa.types.is_float16(t)):
        return "numeric"
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t):
        return "string"
    return None


def _type_name(t):
    if pa.types.is_dictionary(t):
        t = t.value_type
    return {"double": "float64", "float": "float32", "large_string": "string",
            "string_view": "string"}.get(str(t), str(t))


def inspect_parquet(path):
    """The file's footer, read without touching a data page: the row count, the codecs, and per
    top-level column its Arrow type, its kind, its encoded bytes and an in-memory size estimate."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    md, schema = pf.metadata, pf.schema_arrow
    encoded, codecs = {}, set()
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        for j in range(rg.num_columns):
            c = rg.column(j)
            codecs.add(str(c.compression).upper())
            top = c.path_in_schema.split(".")[0]
            encoded[top] = encoded.get(top, 0) + c.total_uncompressed_size
    rows = md.num_rows
    columns = []
    for f in schema:
        kind = _kind(f.type)
        fixed = kind == "numeric" and not pa.types.is_dictionary(f.type)
        # A string column costs at least its offsets in memory, and never less than its encoded pages.
        memory = rows * (f.type.bit_width // 8) if fixed else max(encoded.get(f.name, 0), rows * 8)
        columns.append({"name": f.name, "type": f.type, "kind": kind,
                        "encoded_bytes": encoded.get(f.name, 0), "memory_bytes": memory})
    return {"rows": rows, "columns": columns, "codecs": sorted(codecs),
            "row_groups": md.num_row_groups, "file_bytes": os.path.getsize(path)}


def choose_value_column(info):
    """The largest numeric column: the most bytes in memory, then the most encoded bytes, then the
    first in the file. None when the file has no numeric column."""
    numeric = [(i, c) for i, c in enumerate(info["columns"]) if c["kind"] == "numeric"]
    if not numeric:
        return None
    return max(numeric, key=lambda ic: (ic[1]["memory_bytes"], ic[1]["encoded_bytes"], -ic[0]))[1]["name"]


def choose_key_column(table, info, value):
    """The lowest-cardinality integer or string column other than `value`, with its distinct count
    (a null counts as one group). A column with two or more distinct values wins over a constant one;
    ties go to the first in the file. (None, 0) when there is no candidate."""
    best = None
    for i, c in enumerate(info["columns"]):
        if c["name"] == value or c["name"] not in table.column_names:
            continue
        t = c["type"].value_type if pa.types.is_dictionary(c["type"]) else c["type"]
        if not (pa.types.is_integer(t) or c["kind"] == "string"):
            continue
        n = pc.count_distinct(table.column(c["name"]), mode="all").as_py()
        rank = (n < 2, n, i)
        if best is None or rank < best[0]:
            best = (rank, c["name"], n)
    return (best[1], best[2]) if best else (None, 0)


def size_guard(info, memory_bytes):
    """None when the columns the bench reads fit in memory, else the line that says why not and
    what the limit is."""
    if not memory_bytes:
        return None
    need = sum(c["memory_bytes"] for c in info["columns"] if c["kind"])
    limit = memory_bytes // MEMORY_SHARE
    if need <= limit and info["file_bytes"] <= limit:
        return None

    def gb(b):
        return f"{b / 2**30:.1f} GB"
    return (f"refusing this file: the columns the bench reads take about {gb(need)} in memory "
            f"(estimated from the footer; the file is {gb(info['file_bytes'])} on disk). The bench "
            f"holds them up to {MEMORY_SHARE} times at once (pyarrow, Polars, ArrowMetal's reader and the "
            f"Metal columns), so the limit on this Mac, with {gb(memory_bytes)} of memory, is {gb(limit)}.")


def no_column_line(info):
    types = {}
    for c in info["columns"]:
        name = _type_name(c["type"])
        types[name] = types.get(name, 0) + 1
    listed = ", ".join(f"{t} ({n})" for t, n in sorted(types.items())) or "no columns"
    return ("no supported column in this file: the bench reads integer, floating-point and string "
            f"columns, and the file has {listed}.")


def _close(a, b, scale):
    if a is None or b is None:
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= 1e-9 * max(scale, 1.0)
    return a == b


def _measure(bench, impls):
    row = {}
    for lib, fn in impls.items():
        if fn is None:
            continue
        w, c, n = bench.run(fn)
        row[lib] = {"wall_ms": round(w, 3), "cpu_ms": round(c, 3), "iterations": n}
    cpu = [lib for lib in ("pyarrow", "polars") if lib in row]
    row["fastest_cpu"] = min(cpu, key=lambda lib: row[lib]["wall_ms"])
    am_ms = row.get("arrowmetal", {}).get("wall_ms")
    row["speedup"] = round(row[row["fastest_cpu"]]["wall_ms"] / am_ms, 2) if am_ms else None
    return row


def run_parquet(path, use_polars=True, bench=None, info=None):
    """Measure the read, then sum / filter / group-by, on one Parquet file. The result has the shape
    `run` returns, plus `file` (its shape: no path, no names) and `notes`."""
    import pyarrow.parquet as pq
    bench = bench or Bench()
    t_all = time.perf_counter()
    pl = _polars(use_polars)
    info = info or inspect_parquet(path)
    kinds = {c["name"]: c for c in info["columns"]}
    used = [c["name"] for c in info["columns"] if c["kind"]]
    value = choose_value_column(info)
    notes, problems, timings = [], [], {}

    # The read: the columns the bench can use, each library into its own memory.
    read_op = f"read {len(used)} of {len(info['columns'])} columns"
    am_error = None
    try:
        am.read_parquet(path, columns=used)
    except Exception as e:           # a codec or an encoding ArrowMetal's reader refuses
        am_error = (str(e).splitlines() or [type(e).__name__])[0]
        notes.append(f"ArrowMetal's reader refused the file ({am_error}); the operations ran on columns "
                     "imported from pyarrow's read.")
    timings[read_op] = _measure(bench, {
        "pyarrow": lambda: pq.read_table(path, columns=used),
        "polars": (lambda: pl.read_parquet(path, columns=used)) if pl is not None else None,
        "arrowmetal": (lambda: am.read_parquet(path, columns=used)) if am_error is None else None,
    })
    tbl = pq.read_table(path, columns=used)
    if am_error is None:
        got = am.read_parquet_table(path, columns=used)
        for name in used:
            a, b = tbl.column(name).combine_chunks(), got.column(name).combine_chunks()
            if pa.types.is_dictionary(a.type):
                a = a.dictionary_decode()
            if pa.types.is_dictionary(b.type):
                b = b.dictionary_decode()
            if b.type != a.type:
                b = b.cast(a.type)
            if not a.equals(b):
                problems.append(f"{read_op}: a {_type_name(kinds[name]['type'])} column differs from pyarrow's")

    if value is None:
        notes.append("no numeric column: sum, filter and group-by need one, so only the read was measured.")
    else:
        key, groups = choose_key_column(tbl, info, value)
        vt = _type_name(kinds[value]["type"])
        v = tbl.column(value)
        cols = [value] + ([key] if key else [])
        pdf = pl.read_parquet(path, columns=cols) if pl is not None else None
        pv = pdf[value] if pdf is not None else None
        t0 = time.perf_counter()
        if am_error is None:
            mc = am.read_parquet(path, columns=cols, dictionary=False)
            gv, gk = mc[value], (mc[key] if key else None)
        else:
            gv = am.MetalArray.from_arrow(v.combine_chunks())
            gk = am.MetalArray.from_arrow(tbl.column(key).combine_chunks()) if key else None
        load_ms = (time.perf_counter() - t0) * 1000.0
        integer = pa.types.is_integer(v.type)
        med = pc.approximate_median(v).as_py()
        thr = 0 if med is None else (int(med) if integer else float(med))
        scale = 0.0 if integer else float(pc.sum(pc.abs(v)).as_py() or 0.0)

        sum_op, filter_op = f"sum {vt}", f"filter {vt} > median"
        timings[sum_op] = _measure(bench, {
            "pyarrow": lambda: pc.sum(v),
            "polars": (lambda: pv.sum()) if pv is not None else None,
            "arrowmetal": lambda: gv.sum(),
        })
        timings[filter_op] = _measure(bench, {
            "pyarrow": lambda: pc.filter(v, pc.greater(v, thr)),
            "polars": (lambda: pv.filter(pv > thr)) if pv is not None else None,
            "arrowmetal": lambda: gv.filter_where(">", thr),
        })
        if not _close(gv.sum(), pc.sum(v).as_py(), scale):
            problems.append(f"{sum_op}: ArrowMetal and pyarrow disagree")
        got, want = gv.filter_where(">", thr).to_arrow(), pc.filter(v, pc.greater(v, thr)).combine_chunks()
        if len(got) != len(want) or not got.equals(want):
            problems.append(f"{filter_op}: ArrowMetal keeps {len(got):,} rows, pyarrow {len(want):,}")

        if key is None:
            notes.append("no integer or string column besides the summed one, so group-by was not measured.")
        else:
            kt = _type_name(kinds[key]["type"])
            gb_op = f"sum {vt} by {kt} key ({groups:,} groups)"
            gtbl = tbl.select([key, value])
            timings[gb_op] = _measure(bench, {
                "pyarrow": lambda: gtbl.group_by(key).aggregate([(value, "sum")]),
                "polars": (lambda: pdf.group_by(key).agg(pl.col(value).sum())) if pdf is not None else None,
                "arrowmetal": lambda: am.group_by([gk]).sum(gv),
            })
            g = am.group_by([gk])
            got = _group_dict(g.keys()[0], g.sum(gv).to_arrow())
            pt = gtbl.group_by(key).aggregate([(value, "sum")])
            pkeys = pt.column(key).combine_chunks()
            if pa.types.is_dictionary(pkeys.type):
                pkeys = pkeys.dictionary_decode()
            want = _group_dict(pkeys, pt.column(value + "_sum"))
            if set(got) != set(want):
                problems.append(f"{gb_op}: ArrowMetal has {len(got):,} groups, pyarrow {len(want):,}")
            else:
                bad = sum(1 for k in want if not _close(got[k], want[k], scale))
                if bad:
                    problems.append(f"{gb_op}: {bad:,} of {len(want):,} group sums differ")
        notes.append(("ArrowMetal ran on the columns am.read_parquet returned, already in Metal memory"
                      if am_error is None else "ArrowMetal ran on columns imported from pyarrow's read")
                     + f" (loading them once took {load_ms:.1f} ms).")

    versions = {"arrowmetal": am.__version__, "pyarrow": pa.__version__,
                "polars": pl.__version__ if pl is not None else None,
                "python": platform.python_version()}
    file = {"rows": info["rows"], "columns": len(info["columns"]), "columns_read": len(used),
            "numeric_columns": sum(1 for c in info["columns"] if c["kind"] == "numeric"),
            "string_columns": sum(1 for c in info["columns"] if c["kind"] == "string"),
            "codecs": info["codecs"], "row_groups": info["row_groups"],
            "file_mb": round(info["file_bytes"] / 1e6, 1)}
    return {
        "machine": machine_info(),
        "versions": versions,
        "rows": info["rows"],
        "file": file,
        "timings": timings,
        "match": not problems,
        "problems": problems,
        "notes": notes,
        "protocol": "one warm-up, best of up to 5 calls, wall ms with the process CPU ms of that call; "
                    "the read is warm (the file is in the OS page cache after the warm-up)",
        "total_s": round(time.perf_counter() - t_all, 3),
    }


def file_line(f):
    return (f"file: {f['rows']:,} rows, {f['columns']} columns ({f['columns_read']} read: {f['numeric_columns']} numeric, "
            f"{f['string_columns']} string), {f['row_groups']:,} row groups, {f['file_mb']:,} MB, "
            f"codec {', '.join(f['codecs']) or 'none'}")


def report_parquet(result, quiet=False, share=True):
    versions = result["versions"]
    line, fline = machine_line(result["machine"], versions), file_line(result["file"])
    out = [] if quiet else [line, fline, ""]
    out.append(text_table(result))
    if quiet:
        return "\n".join(out)
    out.append("")
    out.append("results match pyarrow" if result["match"] else "MISMATCH against pyarrow")
    out.append("")
    polars_note = ", Polars its thread pool" if versions.get("polars") else ""
    out.append("Protocol: one warm-up, best of up to 5 calls, wall ms with the process CPU ms of that call in "
               f"parentheses; pyarrow uses its thread pool{polars_note}; the read is warm (the file is in the "
               "OS page cache after the warm-up); speedup is against the fastest CPU number on the row.")
    out.append("Columns: the summed and filtered one is the largest numeric column, the group key the "
               "lowest-cardinality integer or string column.")
    out.extend(result["notes"])
    out.append(f"Whole run {result['total_s']:.1f} s. Router mode: "
               f"{os.environ.get('ARROWMETAL_ROUTER', 'auto')} (the default).")
    if share:
        block = line + "\n" + fline + "\n\n" + markdown_table(result)
        out += ["", "Share it (nothing is sent by this script; the block holds the file's shape and the "
                "timings, never its path, column names or values; copy it below):", "",
                "```", block, "```", "",
                f"Open a prefilled issue: {issue_url(result, share=block)}",
                "or drop it in the #benchmarks channel of the ArrowMetal Discord (link in the README)."]
    return "\n".join(out)


def main_parquet(path, a):
    if not os.path.isfile(path):
        print(f"arrowmetal.bench: no such file: {path}", file=sys.stderr)
        return 2
    try:
        info = inspect_parquet(path)
    except Exception as e:
        print(f"arrowmetal.bench: not a readable Parquet file ({e})", file=sys.stderr)
        return 2
    refusal = size_guard(info, physical_memory())
    if refusal:
        print("arrowmetal.bench: " + refusal, file=sys.stderr)
        return 2
    if not any(c["kind"] for c in info["columns"]):
        print("arrowmetal.bench: " + no_column_line(info), file=sys.stderr)
        return 2
    result = run_parquet(path, use_polars=not a.no_polars, info=info)
    if a.json:
        print(json.dumps(result, indent=2))
    else:
        print(report_parquet(result, quiet=a.quiet, share=not a.no_share))
    if not result["match"]:
        for line in result["problems"]:
            print("MISMATCH: " + line, file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m arrowmetal.bench",
                                description="CPU (pyarrow, Polars if installed) against ArrowMetal on this Mac.",
                                epilog="The generated dataset: int64 v uniform in [-1,000,000, 1,000,000) with 10%% "
                                       "nulls, float64 f standard normal, int32 k uniform over 1,000 keys, drawn "
                                       "with pyarrow.compute from seeded SplitMix64 streams (no NumPy).")
    p.add_argument("--rows", type=int, default=10_000_000, help="rows in the generated dataset (default 10,000,000)")
    p.add_argument("--json", action="store_true", help="print one JSON object instead of the text report")
    p.add_argument("--quiet", action="store_true", help="print the table only")
    p.add_argument("--no-share", action="store_true", help="omit the Share it block")
    p.add_argument("--no-polars", action="store_true", help="do not measure Polars even if it is installed")
    p.add_argument("--parquet", metavar="FILE",
                   help="your own Parquet file instead of generated data: the read, then sum, filter and "
                        "group-by on columns chosen from it")
    a = p.parse_args(argv)
    if a.rows < 1:
        p.error("--rows must be at least 1")
    if a.parquet:
        return main_parquet(a.parquet, a)

    # The router runs as it does for every user, `auto` unless ARROWMETAL_ROUTER says otherwise: at the
    # default row count every routed operation lands on the GPU, and a smaller --rows shows the CPU loop
    # the router chooses instead. The footer says which mode ran.

    result = run(a.rows, use_polars=not a.no_polars)
    if a.json:
        print(json.dumps(result, indent=2))
    else:
        print(report(result, quiet=a.quiet, share=not a.no_share))
    if not result["match"]:
        for line in result["problems"]:
            print("MISMATCH: " + line, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
