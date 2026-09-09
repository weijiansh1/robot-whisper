"""Restate the tightest frozen invariant in its unfitted, conserved-quantity
form, so the claim does not rest on a regression.

The frozen candidate is a fitted relation
    sa@L[s9] = b * sa@L[s0] + a .
If it is really a conservation law then the *unfitted difference*
    D = sa@L[s9] - sa@L[s0]
must already be small compared with the spread of sa itself.  This script
reports std(D)/std(sa) with no coefficient at all, per layer, per cohort, per
suite, and within episode.  It is label-blind and adds no candidate; it only
describes one that phase 1 already froze.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
OUT = BUNDLE / "results"
BANK = OUT / "bank"
LAYERS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
COHORTS = ("development_main", "development_extra", "external_8b")
METRICS = ("state_action_alignment", "conditional_energy", "token_entropy",
           "load_entropy", "conditional_effective_rank", "flow_speed")


def within_group_std(x, g):
    G = int(g.max()) + 1
    cnt = np.bincount(g, minlength=G).astype(float)
    s = np.bincount(g, weights=x, minlength=G)
    m = np.zeros(G)
    m[cnt > 0] = s[cnt > 0] / cnt[cnt > 0]
    return float((x - m[g]).std())


def main() -> None:
    out: dict = {"note": __doc__.strip().splitlines()[0]}
    for coh in COHORTS:
        f = np.load(BANK / f"{coh}_bank.npz", allow_pickle=True)
        names = [str(s) for s in f["var_names"]]
        col = {n: i for i, n in enumerate(names)}
        X = f["X"].astype(np.float64)
        epi = f["row_query"].astype(np.int64)
        suite = f["row_suite"].astype(int)
        rec = {}
        for m in METRICS:
            per_layer = {}
            for L in LAYERS:
                a, b = f"sp.{m}@{L}[s0]", f"sp.{m}@{L}[s9]"
                if a not in col or b not in col:
                    continue
                x0, x9 = X[:, col[a]], X[:, col[b]]
                d = x9 - x0
                scale = 0.5 * (x0.std() + x9.std())
                e = {
                    "mean_level": float(0.5 * (x0.mean() + x9.mean())),
                    "std_of_the_quantity": float(scale),
                    "unfitted_rel_drift_std": float(d.std() / scale),
                    "unfitted_mean_drift_over_std": float(d.mean() / scale),
                    "within_episode_rel_drift_std":
                        float(within_group_std(d, epi) /
                              max(within_group_std(x0, epi), 1e-30)),
                    "p99_abs_drift_over_std":
                        float(np.percentile(np.abs(d), 99) / scale),
                }
                for s, sn in enumerate(SUITES):
                    k = suite == s
                    if k.sum() < 400:
                        continue
                    e[f"unfitted_rel_drift_std|{sn}"] = float(
                        d[k].std() / (0.5 * (x0[k].std() + x9[k].std())))
                per_layer[L] = e
            rec[m] = per_layer
        out[coh] = rec
    (OUT / "top_invariant_description.json").write_text(json.dumps(out, indent=1))

    print("unfitted std(x[s9]-x[s0]) / std(x)   (denoising drift, no fitting)")
    print("  %-28s %-6s %8s %8s %8s" % ("metric", "layer", "dev_main",
                                        "external", "dev_extra"))
    for m in METRICS:
        for L in LAYERS:
            try:
                a = out["development_main"][m][L]["unfitted_rel_drift_std"]
                b = out["external_8b"][m][L]["unfitted_rel_drift_std"]
                c = out["development_extra"][m][L]["unfitted_rel_drift_std"]
            except KeyError:
                continue
            print("  %-28s %-6s %8.4f %8.4f %8.4f" % (m, L, a, b, c))


if __name__ == "__main__":
    main()
