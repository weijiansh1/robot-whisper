"""Causal-prefix outcome AUROC of denoising-path bending at fixed anchors (same anchors as the topology grid)."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

OUT = Path("/home/swj/data/libero-runtime/samples/flow-path-straightness-20260916")
chunks = list(np.load(OUT / "trap-chunk-rows.npy", allow_pickle=True))
eps = {}
for r in chunks:
    eps.setdefault(r["episode"], {"success": r["success"], "ld": {}, "freeze": {}, "alarm": {}, "one_step": {}, "perp": {}})
    eps[r["episode"]]["ld"][r["q"]] = r["ld"]; eps[r["episode"]]["freeze"][r["q"]] = r["freeze"]
    eps[r["episode"]]["alarm"][r["q"]] = r["alarm"]; eps[r["episode"]]["one_step"][r["q"]] = r["one_step"]; eps[r["episode"]]["perp"][r["q"]] = r["perp"]


def auroc(fail, succ):
    if len(fail) < 3 or len(succ) < 3:
        return float("nan"), 0, 0
    u = mannwhitneyu(fail, succ, alternative="two-sided")
    return float(u.statistic / (len(fail) * len(succ))), len(fail), len(succ)


def feature(e, a, kind):
    qs = [q for q in range(a - 4, a + 1)]
    if a + 1 not in e["ld"]:   # require the anchor to be strictly inside the episode: a full following window exists
        return None
    if kind == "ld_mean5":
        return float(np.mean([e["ld"][q] for q in qs]))
    if kind == "ld_at":
        return float(e["ld"][a])
    if kind == "ld_max_prefix":
        return float(max(e["ld"][q] for q in range(a + 1)))
    if kind == "one_step_mean5":
        return float(np.mean([e["one_step"][q] for q in qs]))
    if kind == "perp_mean5":
        return float(np.mean([e["perp"][q] for q in qs]))
    if kind == "v82_alarm_by":
        return float(max(e["alarm"][q] for q in range(a + 1)))
    if kind == "freeze_at":
        v = e["freeze"][a]
        return None if not np.isfinite(v) else float(v)
    raise KeyError(kind)


results = {}
print("%-16s %-14s" % ("feature", "dataset") + "".join("  a=%d" % a for a in range(16, 21)) + "   (AUROC failure > success; n fail/succ at a=18)")
for kind in ("ld_mean5", "ld_at", "ld_max_prefix", "perp_mean5", "one_step_mean5", "freeze_at", "v82_alarm_by"):
    for ds in ("topo-libero10", "topo-plus", "pooled"):
        row = []
        for a in range(16, 21):
            f, s = [], []
            for name, e in eps.items():
                if ds != "pooled" and not name.startswith(ds + "__"):
                    continue
                v = feature(e, a, kind)
                if v is None:
                    continue
                (f if not e["success"] else s).append(v)
            au, nf, ns = auroc(f, s)
            row.append((au, nf, ns))
        results["%s|%s" % (kind, ds)] = [dict(anchor=a, auroc=r[0], n_fail=r[1], n_succ=r[2]) for a, r in zip(range(16, 21), row)]
        print("%-16s %-14s" % (kind, ds) + "".join("  %.3f" % r[0] for r in row) + "   (%d/%d)" % (row[2][1], row[2][2]))
(OUT / "anchor-outcome-auroc.json").write_text(json.dumps(results, indent=2))
print("saved", OUT / "anchor-outcome-auroc.json")
