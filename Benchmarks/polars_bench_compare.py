"""Two runs of polars_engine_bench.py at 50,000,000 rows on two Polars releases (same code, same
session): Polars' own time on the second against the first, the default's cases and speed-ups on
each, and the cases it leaves where the GPU was faster.

usage: polars_bench_compare.py NEW.csv OLD.csv"""
import csv, io, sys
def load(p):
    out = {}
    for r in csv.DictReader(io.StringIO("".join(l for l in open(p) if not l.startswith("#")))):
        if r["rows"] == "50000000": out.setdefault(r["case"], {})[r["engine"]] = r
    return out
def ms(d, e): return float(d[e]["wall_ms"]) if e in d and d[e]["wall_ms"] else None
def pol(d):
    ts = [ms(d, e) for e in ("polars in-memory", "polars streaming") if ms(d, e)]
    return min(ts) if ts else None
A, B = load(sys.argv[1]), load(sys.argv[2])
engines = sorted({e for d in A.values() for e in d})
print("engines:", engines)
gpu_e = next(e for e in engines if e.startswith("MetalEngine") and "all" in e)
def_e = "MetalEngine default, cold"
rows = []
for c in A:
    a, b = A[c], B.get(c, {})
    pa_, pb = pol(a), pol(b)
    ga = ms(a, gpu_e); gb = ms(b, gpu_e) if b else None
    taken = bool(a.get(def_e, {}).get("taken"))
    da = ms(a, def_e)
    rows.append((c, pb, pa_, gb, ga, taken, da))
ratio = sorted(pa_/pb for c, pb, pa_, *_ in rows if pa_ and pb)
print(f"cases {len(rows)}; Polars NEW time / OLD time (faster engine): median {ratio[len(ratio)//2]:.2f}, min {ratio[0]:.2f}, max {ratio[-1]:.2f}")
taken = [(c, pa_/da) for c, pb, pa_, gb, ga, t, da in rows if t and da and pa_]
xs = sorted(x for _, x in taken)
print(f"default taken on NEW: {len(taken)}; speed-up {xs[0]:.2f}x-{xs[-1]:.2f}x; behind: {[(c[:45], round(x,2)) for c,x in taken if x < 1]}")
missed = [(c, pa_/ga) for c, pb, pa_, gb, ga, t, da in rows if not t and ga and pa_ and pa_/ga > 1.15]
print(f"left to Polars where the GPU was >1.15x faster: {len(missed)}: {[(c[:45], round(x,2)) for c,x in sorted(missed, key=lambda z:-z[1])[:15]]}")
tb = [(c, pol(B[c])/ms(B[c], def_e)) for c in B if B[c].get(def_e, {}).get("taken") and pol(B[c]) and ms(B[c], def_e)]
xb = sorted(x for _, x in tb)
print(f"default taken on OLD (same session): {len(tb)}; {xb[0]:.2f}x-{xb[-1]:.2f}x; behind: {[(c[:45], round(x,2)) for c,x in tb if x < 1]}")
big = sorted(((c, pa_/pb) for c, pb, pa_, *_ in rows if pa_ and pb), key=lambda z: z[1])
print("Polars NEW most changed (NEW/OLD):", [(c[:40], round(x,2)) for c,x in big[:8]], "...", [(c[:40], round(x,2)) for c,x in big[-5:]])
mb = {c: pol(B[c])/ms(B[c], gpu_e) for c in B if not B[c].get(def_e, {}).get("taken") and pol(B[c]) and ms(B[c], gpu_e)}
ma = dict(missed)
print(f"left where GPU >1.15x on OLD (same session): {sum(1 for x in mb.values() if x > 1.15)}")
print("on NEW only:", [(c[:40], round(ma[c],2), round(mb.get(c,0),2)) for c in ma if mb.get(c, 0) <= 1.15])
mx = max(tb, key=lambda z: z[1]); print("OLD max case:", mx[0][:60], round(mx[1],2))
