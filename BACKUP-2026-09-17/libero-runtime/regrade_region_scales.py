"""Fairer re-grading of the grid's recurrence-region settings: judge direction consistency only over
non-degenerate radius scales (AUROC exactly 0.5 means the feature was constant / no eligible episode)."""
import pandas as pd, numpy as np, json
from pathlib import Path
G = Path("/home/swj/data/moe-capture/topo-20260916/_grid")
df = pd.read_csv(G / "auroc-table.csv")
REGION = ("eligible", "outside_fraction", "exit_confirmed", "final_outside", "returned")
reg = df[df.feature.isin(REGION) & df.scale.notna() & (df.anchor <= 24)].copy()
reg["dir"] = np.sign(reg.task_auroc - 0.5)
reg["nondeg"] = (reg.auroc != 0.5) & (reg.task_auroc != 0.5)
keys = ["dataset", "matrix", "history", "reference", "quantile", "confirm", "anchor", "feature"]
out = {}
for dsname, sub in reg.groupby("dataset"):
    g = sub.groupby(keys[1:])
    rows = []
    for k, grp in g:
        nd = grp[grp.nondeg]
        if len(nd) == 0:
            continue
        consistent = (nd.dir.nunique() == 1)
        rows.append(dict(key=k, n_scales_nondeg=len(nd), consistent=bool(consistent), min_abs=float((0.5 + (nd.task_auroc - 0.5).abs()).min()),
                         mean_abs=float((0.5 + (nd.task_auroc - 0.5).abs()).mean()), scales=list(nd.scale), taus=list(nd.task_auroc.round(3))))
    R = pd.DataFrame(rows)
    summary = {}
    for nmin in (2, 3, 4):
        m = R[(R.n_scales_nondeg >= nmin) & R.consistent]
        summary["nondeg>=%d_consistent" % nmin] = int(len(m))
        summary["nondeg>=%d_consistent_minabs>=0.6" % nmin] = int((m.min_abs >= 0.6).sum())
        summary["nondeg>=%d_consistent_minabs>=0.65" % nmin] = int((m.min_abs >= 0.65).sum())
        summary["nondeg>=%d_total" % nmin] = int((R.n_scales_nondeg >= nmin).sum())
    summary["settings_with_any_nondeg_scale"] = int(len(R))
    top = R[(R.n_scales_nondeg >= 3) & R.consistent].sort_values("min_abs", ascending=False).head(8)
    out[dsname] = dict(summary=summary, top=[dict(key=[str(x) for x in t.key], n=int(t.n_scales_nondeg), min_abs=t.min_abs, scales=t.scales, task_aurocs=t.taus) for t in top.itertuples()])
    print("==", dsname, json.dumps(summary))
    for t in top.itertuples():
        print("   ", t.key, "n_nondeg", t.n_scales_nondeg, "min|tAUROC|", round(t.min_abs, 3), "scales", t.scales, "tAUROC", t.taus)
(G.parent / "_grid" / "region-regrade-q24.json").write_text(json.dumps(out, indent=2))
