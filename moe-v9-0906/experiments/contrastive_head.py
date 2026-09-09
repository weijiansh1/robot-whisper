"""A late-gated head on the contrastive-PCA direction.

Three label-free projections of the self-baselined 152-dimensional routing
vector were compared.  PC1 (maximum variance) and the survivor-mean drift are
nearly the same direction and share the same failing time profile: strong at
chunks 9 and 14 (AUC 0.61-0.63) and then collapsing, with the sign flipping, by
chunk 18-20.  The contrastive direction - the leading eigenvector of
`cov(late) - cov(early)`, i.e. the direction along which the surviving
population spreads out fastest - behaves oppositely: weak at chunk 9 and
**stable at 0.56-0.64 from chunk 12 through 20 with a consistent sign**.

That late stability is the point.  Every quantity in v7 and v8 weakens or
reverses late, and long-lead detection on `libero_long` needs an alarm by chunk
36.  So this head is gated to the late window and asked one question: does it
add anything at lead >= 16, where v8.3 currently reaches 496/712 on long?

The direction is fitted on `development_main` with **no labels** - only the
covariance of the routing features among episodes still running at two chunks.
The threshold is an order statistic of the unlabeled development projections.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
V8DIR = ROOT / "moe-v8-0906"
sys.path.insert(0, str(V8DIR / "experiments"))
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

import evaluate_full_corpus as V  # noqa: E402
import freeze_v82 as F  # noqa: E402

COHORTS = ("development_main", "external_8b", "legacy_main16x32")
LEADS = (0, 4, 8, 12, 16, 20)
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
V83 = {"baseline": 2, "width": 6, "confirm": 2, "slope": -0.0015}
EARLY_Q, LATE_Q = 8, 20
GATES = {"q12_20": tuple(range(12, 21)), "q12_30": tuple(range(12, 31)),
         "q16_36": tuple(range(16, 37)), "q12_36": tuple(range(12, 37))}
QUANTILES = (0.98, 0.99, 0.995)
CONFIRMS = (1, 2)
SEED = 20260906


def features(cohort: str) -> np.ndarray:
    """Self-baselined log routing features, [n, chunk, 152].

    Self-baselining against the episode's own q1..q4 removes the episode- and
    hence task-level offset without ever reading a task label.
    """
    speed = np.load(V8DIR / "results" / f"{cohort}_flow_speed.npz")["flow_speed"]
    mob = np.load(OUT / f"{cohort}_step_mobility.npz")["mobility"]
    x = np.concatenate([speed.reshape(*speed.shape[:2], -1),
                        mob.reshape(*mob.shape[:2], -1)], axis=2)
    base = np.nanmean(x[:, 1:5], axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log(np.maximum(x, 1e-12)) - np.log(np.maximum(base, 1e-12))
    z[~np.isfinite(z)] = np.nan
    return z


def fit_direction(z, length):
    """Leading eigenvector of cov(late) - cov(early), label-free.

    Only 'is this episode still running' and the routing itself are used; no
    outcome is read.  Standardisation constants come from the same corpus.
    """
    pool = z[np.isfinite(z).all(2) & (np.arange(z.shape[1])[None, :] >= 6)]
    mu, sd = pool.mean(0), pool.std(0) + 1e-9

    def cov_at(q):
        m = np.isfinite(z[:, q]).all(1) & (length > q)
        return np.cov((z[m, q] - mu) / sd, rowvar=False)

    w, vec = np.linalg.eigh(cov_at(LATE_Q) - cov_at(EARLY_Q))
    d = vec[:, int(np.argmax(w))]
    return mu, sd, d / np.linalg.norm(d)


def project(z, mu, sd, d):
    out = np.full(z.shape[:2], np.nan)
    for q in range(z.shape[1]):
        row = (z[:, q] - mu) / sd
        ok = np.isfinite(row).all(1)
        out[ok, q] = np.nan_to_num(row[ok]) @ d
    return out


def gated_first(proj, threshold, gate, confirm):
    hit = np.zeros(proj.shape, dtype=bool)
    for q in gate:
        if q < proj.shape[1]:
            hit[:, q] = np.isfinite(proj[:, q]) & (proj[:, q] >= threshold)
    if confirm > 1:
        held = np.zeros_like(hit)
        run = np.zeros(hit.shape[0], dtype=int)
        for q in range(hit.shape[1]):
            run = np.where(hit[:, q], run + 1, np.where(q in gate, 0, run))
            held[:, q] = hit[:, q] & (run >= confirm)
        hit = held
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def main() -> None:
    data = {c: V.load_cohort(c) for c in COHORTS}
    z = {c: features(c) for c in COHORTS}
    heads = {c: F.build_heads(d["speed"], V83["baseline"], V83["width"])
             for c, d in data.items()}
    thr83 = F.thresholds_from(heads["development_main"])
    base = {}
    for cohort, d in data.items():
        firsts = [F.moving_first(heads[cohort][h], thr83[h], V.DIRECTION[h],
                                 V83["slope"], V83["confirm"], V83["width"])
                  for h in heads[cohort]]
        base[cohort] = V.union(d["v7"], *firsts)

    def profile(alarms):
        prof = {lead: [0, 0] for lead in LEADS}
        for cohort, first in alarms.items():
            d = data[cohort]
            for lead in LEADS:
                s = V.score(first, d["risk"], d["length"], lead)
                prof[lead][0] += s["tp"]
                prof[lead][1] += s["fp"]
        return prof

    p83 = profile(base)
    print("v8.3 " + "  ".join("L%d %d/%d" % (b, *p83[b]) for b in LEADS))

    dev = data["development_main"]
    mu, sd, d = fit_direction(z["development_main"], dev["length"])
    proj = {c: project(z[c], mu, sd, d) for c in COHORTS}

    def long_l16(alarms):
        tp = fp = 0
        for cohort, first in alarms.items():
            dd = data[cohort]
            m = dd["suite"] == "libero_long"
            if not m.any():
                continue
            t = (first >= 0) & ((dd["length"] - first) >= 16)
            tp += int((t & m & dd["risk"]).sum())
            fp += int((t & m & ~dd["risk"]).sum())
        return tp, fp

    d0 = V.score(base["development_main"], dev["risk"], dev["length"], 0)
    rows = []
    for gname, q, confirm in itertools.product(GATES, QUANTILES, CONFIRMS):
        gate = GATES[gname]
        pool = proj["development_main"][:, list(gate)]
        pool = pool[np.isfinite(pool)]
        if pool.size < 1000:
            continue
        threshold = float(np.quantile(pool, q, method="lower"))
        head = gated_first(proj["development_main"], threshold, gate, confirm)
        u = V.union(base["development_main"], head)
        s0 = V.score(u, dev["risk"], dev["length"], 0)
        s16 = V.score(u, dev["risk"], dev["length"], 16)
        rows.append({"gate": gname, "quantile": q, "confirm": confirm,
                     "threshold": threshold, "dev0_tp": s0["tp"],
                     "dev0_fp": s0["fp"], "dev16_tp": s16["tp"]})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "contrastive_grid.csv", index=False)

    d16 = V.score(base["development_main"], dev["risk"], dev["length"], 16)
    ok = grid[(grid.dev0_tp >= d0["tp"]) & (grid.dev0_fp <= d0["fp"] + 6)]
    print("\ndevelopment: lead>=0 %d/487 %dFP, lead>=16 %d/487"
          % (d0["tp"], d0["fp"], d16["tp"]))
    print("网格 %d 个；满足 lead>=0 不退步且新增FP<=6 的有 %d 个" % (len(grid), len(ok)))
    if not len(ok) or ok.dev16_tp.max() <= d16["tp"]:
        print("没有配置在 development 上改善 lead>=16 —— 对比方向头不成立")
        return
    pick = ok.sort_values(["dev16_tp", "dev0_fp"], ascending=[False, True]).iloc[0]
    print("选中: gate %s, 分位 %.3f, K=%d  (dev lead>=16: %d -> %d)"
          % (pick.gate, pick["quantile"], pick.confirm, d16["tp"], pick.dev16_tp))

    alarms = {c: V.union(base[c], gated_first(proj[c], float(pick.threshold),
                                              GATES[pick.gate], int(pick.confirm)))
              for c in COHORTS}
    p9 = profile(alarms)
    np.savez_compressed(OUT / "contrastive_alarms.npz",
                        **{f"{c}|v9k": v for c, v in alarms.items()})
    print("\n%-6s %s" % ("臂", "  ".join("lead>=%-2d" % b for b in LEADS)))
    print("%-6s %s" % ("v8.3", "  ".join("%4d/%-4d" % tuple(p83[b]) for b in LEADS)))
    print("%-6s %s" % ("v9k", "  ".join("%4d/%-4d" % tuple(p9[b]) for b in LEADS)))
    print("\nlibero_long lead>=16:  v8.3 %d/712 %dFP  ->  v9k %d/712 %dFP"
          % (*long_l16(base), *long_l16(alarms)))

    print("\n分 cohort (lead>=4):")
    for cohort in COHORTS:
        dd = data[cohort]
        a = V.score(base[cohort], dd["risk"], dd["length"], 4)
        b = V.score(alarms[cohort], dd["risk"], dd["length"], 4)
        print("  %-18s %3d/%-4d %3dFP -> %3d/%-4d %3dFP"
              % (cohort, a["tp"], int(dd["risk"].sum()), a["fp"],
                 b["tp"], int(dd["risk"].sum()), b["fp"]))

    rng = np.random.default_rng(SEED)
    ntp = nfp = 0
    for cohort in COHORTS:
        p = proj[cohort].copy()
        for q in range(p.shape[1]):
            idx = np.flatnonzero(np.isfinite(p[:, q]))
            p[idx, q] = p[rng.permutation(idx), q]
        head = gated_first(p, float(pick.threshold), GATES[pick.gate],
                           int(pick.confirm))
        s = V.score(V.union(base[cohort], head), data[cohort]["risk"],
                    data[cohort]["length"], 4)
        ntp += s["tp"]
        nfp += s["fp"]
    print("\n零对照(投影逐 chunk 打乱): %d/1358 TP %d FP  —— v9k 是 %d/1358 %d FP"
          % (ntp, nfp, *p9[4]))
    (OUT / "contrastive_summary.json").write_text(json.dumps({
        "gate": pick.gate, "quantile": float(pick["quantile"]),
        "confirm": int(pick.confirm), "threshold": float(pick.threshold),
        "v83": {str(b): list(p83[b]) for b in LEADS},
        "v9k": {str(b): list(p9[b]) for b in LEADS},
        "long_lead16": {"v83": list(long_l16(base)), "v9k": list(long_l16(alarms))},
        "null": {"tp": ntp, "fp": nfp}}, indent=2))


if __name__ == "__main__":
    main()
