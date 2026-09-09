"""Where the remaining headroom is, measured at matched *external* false alarms.

Comparing an honest development-selected combination with an oracle at matched
*development* budget is not a fair test, because the two land at different
external false alarm counts.  So this script builds, for a grid of external
false alarm targets:

  oracle_ceiling   the best external true alarm count reachable with the
                   external outcomes in hand, over single detectors, OR-pairs
                   and greedy OR-of-8, within the 419 detectors that exist in
                   both cohorts and within all 622 published on external
  honest           the same strategies with everything selected on development
                   and external scored once

The gap between them is the cost of honesty.  The gap between the oracle
ceiling and 100% recall is what nothing built so far can see at all.  A null
arm repeats the oracle sweep on rate-matched random alarm vectors, so a ceiling
that a random bank of the same firing rates also reaches is not a ceiling.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ledger_core as lc
import synth_core as sc

SEED = 20260906
FP_TARGETS = (100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000)
NULL_DRAWS = 5


def _rs(c: sc.Cohort, order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    iw = lc.in_window(c, order)
    return iw[:, c.risk], iw[:, ~c.risk]


def oracle(r: np.ndarray, s: np.ndarray, names: list[str], tag: str,
           extra: list[list[int]] | None = None) -> pd.DataFrame:
    """Upper envelope of what is reachable with the external outcomes in hand.

    Singles and OR-pairs are exact.  For k > 2 the search is budget-aware and
    mirrors the strategy an honest arm uses, so that the ceiling cannot be
    beaten by the honest arm (which would make it meaningless): for each total
    false alarm budget F and each per-detector share alpha, the candidate pool
    is restricted to detectors costing at most alpha*F on their own, and
    marginal true alarms are added greedily until F is exhausted.  A
    cost-benefit variant and every honest configuration passed in ``extra`` are
    also thrown into the envelope, which is finally made monotone in F because
    a solution feasible at F is feasible at any larger F.
    """
    rf, sf = r.astype(np.float32), s.astype(np.float32)
    tp, fp = rf.sum(1), sf.sum(1)
    ir, isf = rf @ rf.T, sf @ sf.T
    or_tp = tp[:, None] + tp[None, :] - ir
    or_fp = fp[:, None] + fp[None, :] - isf
    iu = np.triu_indices(len(names), 1)

    cloud: list[tuple[int, int, str]] = [
        (int(fp[i]), int(tp[i]), names[i]) for i in range(len(names))
    ]
    cloud += [(int(or_fp[a, b]), int(or_tp[a, b]), f"{names[a]} OR {names[b]}")
              for a, b in zip(iu[0], iu[1])]
    for combo in (extra or []):
        cov_r = np.zeros(r.shape[1], bool)
        cov_s = np.zeros(s.shape[1], bool)
        for i in combo:
            cov_r |= r[i]
            cov_s |= s[i]
        cloud.append((int(cov_s.sum()), int(cov_r.sum()),
                      " | ".join(names[i] for i in combo)))

    for F in FP_TARGETS:
        for alpha in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
            for rule in ("tp", "ratio"):
                pool = np.flatnonzero(fp <= alpha * F)
                if len(pool) == 0:
                    continue
                cov_r = np.zeros(r.shape[1], bool)
                cov_s = np.zeros(s.shape[1], bool)
                chosen: list[int] = []
                for _ in range(12):
                    base_r, base_s = int(cov_r.sum()), int(cov_s.sum())
                    cand = np.array([i for i in pool if i not in chosen])
                    if len(cand) == 0:
                        break
                    gains = (cov_r[None, :] | r[cand]).sum(1) - base_r
                    costs = (cov_s[None, :] | s[cand]).sum(1) - base_s
                    ok = (costs + base_s <= F) & (gains > 0)
                    if not ok.any():
                        break
                    val = gains.astype(float) if rule == "tp" else (
                        gains / (1.0 + costs))
                    val = np.where(ok, val, -1.0)
                    pick = int(cand[int(np.argmax(val))])
                    chosen.append(pick)
                    cov_r = cov_r | r[pick]
                    cov_s = cov_s | s[pick]
                    cloud.append((int(cov_s.sum()), int(cov_r.sum()),
                                  " | ".join(names[i] for i in chosen)))

    rows = []
    for F in FP_TARGETS:
        rec = {"arm": tag, "ext_fp_target": F}
        ok = fp <= F
        if ok.any():
            i = int(np.flatnonzero(ok)[np.argmax(tp[ok])])
            rec |= {"single_tp": int(tp[i]), "single_fp": int(fp[i]),
                    "single_detector": names[i]}
        m = or_fp[iu] <= F
        if m.any():
            k = int(np.flatnonzero(m)[np.argmax(or_tp[iu][m])])
            a, b = int(iu[0][k]), int(iu[1][k])
            rec |= {"or_pair_tp": int(or_tp[a, b]), "or_pair_fp": int(or_fp[a, b]),
                    "or_pair": f"{names[a]} OR {names[b]}"}
        feas = [c for c in cloud if c[0] <= F]
        if feas:
            best = max(feas, key=lambda c: c[1])
            rec |= {"best_tp": int(best[1]), "best_fp": int(best[0]),
                    "best_members": best[2], "best_k": best[2].count("|") + 1}
        rows.append(rec)
    out = pd.DataFrame(rows)
    if "best_tp" in out:
        out["best_tp"] = out["best_tp"].cummax()
    return out


def honest_points(dev: sc.Cohort, ext: sc.Cohort, names: list[str],
                  labels: np.ndarray) -> pd.DataFrame:
    """Every honest operating point this bundle can construct, with its external
    cost, so the honest curve can be read against the oracle ceiling."""
    rows = []
    iw_dev = lc.in_window(dev, names)
    fpr = (iw_dev & ~dev.risk).sum(1) / (~dev.risk).sum()
    for budget in lc.BUDGETS:
        reps = lc.representatives(dev, names, labels, budget)
        if reps.empty:
            continue
        for tag, order in (
            (f"family_union@{budget}", reps.detector.tolist()),
            (f"dev_greedy8@{budget}",
             _dev_greedy(dev, names, [i for i in range(len(names)) if fpr[i] <= budget], 8)),
        ):
            if not order:
                continue
            cov_e = lc.in_window(ext, order).any(0)
            cov_d = lc.in_window(dev, order).any(0)
            rows.append({
                "arm": tag, "budget": budget, "n_methods": len(order),
                "dev_tp": int((cov_d & dev.risk).sum()),
                "dev_fp": int((cov_d & ~dev.risk).sum()),
                "ext_tp": int((cov_e & ext.risk).sum()),
                "ext_fp": int((cov_e & ~ext.risk).sum()),
                "ext_recall": float((cov_e & ext.risk).sum() / ext.risk.sum()),
                "ext_fpr": float((cov_e & ~ext.risk).sum() / (~ext.risk).sum()),
                "ext_precision": float((cov_e & ext.risk).sum() / max(cov_e.sum(), 1)),
                "members": " | ".join(order),
            })
    return pd.DataFrame(rows)


def _dev_greedy(dev: sc.Cohort, names: list[str], pool: list[int], k: int) -> list[str]:
    iw = lc.in_window(dev, names)
    cov = np.zeros(dev.n, bool)
    chosen: list[str] = []
    pool = list(pool)
    while pool and len(chosen) < k:
        best = max(pool, key=lambda i: (int((iw[i] & dev.risk & ~cov).sum()),
                                        -int((iw[i] & ~dev.risk & ~cov).sum()), names[i]))
        if int((iw[best] & dev.risk & ~cov).sum()) == 0:
            break
        pool.remove(best)
        cov |= iw[best]
        chosen.append(names[best])
    return chosen


def main() -> None:
    rng = np.random.default_rng(SEED)
    dev = sc.load_cohort("development_main")
    ext = sc.load_cohort("external_8b")
    names, labels = lc.load_family_labels()

    r419, s419 = _rs(ext, names)
    all622 = list(ext.detectors)
    r622, s622 = _rs(ext, all622)

    hon = honest_points(dev, ext, names, labels)
    idx419 = {n: i for i, n in enumerate(names)}
    idx622 = {n: i for i, n in enumerate(all622)}
    extra419 = [[idx419[m] for m in row.members.split(" | ")] for _, row in hon.iterrows()]
    extra622 = [[idx622[m] for m in row.members.split(" | ")] for _, row in hon.iterrows()]
    frames = [oracle(r419, s419, names, "ORACLE_in_sample_419", extra419),
              oracle(r622, s622, all622, "ORACLE_in_sample_622", extra622)]
    # null: rate-matched random bank of the same size and per-suite rates
    iw = lc.in_window(ext, names)
    null_rows = []
    for d in range(NULL_DRAWS):
        nf = sc.rate_matched_null(iw, rng, strata=ext.suite)
        o = oracle(nf[:, ext.risk], nf[:, ~ext.risk], names, "NULL_rate_matched_419")
        o["draw"] = d
        null_rows.append(o)
    null = pd.concat(null_rows, ignore_index=True)
    null_agg = (null.groupby("ext_fp_target")[["single_tp", "or_pair_tp", "best_tp"]]
                .agg(["mean", "std"]).reset_index())

    frames.append(pd.concat(null_rows, ignore_index=True)
                  .groupby(["arm", "ext_fp_target"], as_index=False)
                  [["single_tp", "single_fp", "or_pair_tp", "or_pair_fp",
                    "best_tp", "best_fp", "best_k"]].mean())

    ceiling = pd.concat(frames, ignore_index=True)
    ceiling.to_csv(sc.RESULTS / "headroom_ceiling.csv", index=False)
    null_agg.to_csv(sc.RESULTS / "headroom_null.csv", index=False)

    hon.to_csv(sc.RESULTS / "honest_operating_points.csv", index=False)

    # the honesty gap: for each honest point, the oracle ceiling at its own
    # external false alarm count
    o419 = ceiling[ceiling.arm == "ORACLE_in_sample_419"]
    o622 = ceiling[ceiling.arm == "ORACLE_in_sample_622"]
    gap = []
    for _, h in hon.iterrows():
        def bounds(o):
            lo = o[o.ext_fp_target <= h.ext_fp]
            hi = o[o.ext_fp_target >= h.ext_fp]
            return (float(lo.iloc[-1]["best_tp"]) if len(lo) else np.nan,
                    float(hi.iloc[0]["best_tp"]) if len(hi) else np.nan)
        lo419, hi419 = bounds(o419)
        lo622, hi622 = bounds(o622)
        gap.append({"arm": h.arm, "ext_fp": int(h.ext_fp), "ext_tp": int(h.ext_tp),
                    "ext_recall": h.ext_recall,
                    "oracle419_lower": lo419, "oracle419_upper": hi419,
                    "oracle622_lower": lo622, "oracle622_upper": hi622,
                    "n_risks": int(ext.risk.sum())})
    gapdf = pd.DataFrame(gap)
    for tag in ("419", "622"):
        gapdf[f"honesty_gap_{tag}_lower"] = gapdf[f"oracle{tag}_lower"] - gapdf.ext_tp
        gapdf[f"honesty_gap_{tag}_upper"] = gapdf[f"oracle{tag}_upper"] - gapdf.ext_tp
    gapdf.to_csv(sc.RESULTS / "honesty_gap.csv", index=False)

    meta = {
        "schema": "himoe.method_synthesis.headroom.v1",
        "standing_honest_arm": lc.HONEST_ARM,
        "external_risks": int(ext.risk.sum()),
        "external_safe": int((~ext.risk).sum()),
        "note": ("ORACLE rows select on the external outcomes and are an upper "
                 "bound only; they are not achievable honestly."),
    }
    (sc.RESULTS / "headroom.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 260)
    print(ceiling[["arm", "ext_fp_target", "single_tp", "or_pair_tp",
                   "best_k", "best_tp", "best_fp"]].to_string(index=False))
    print()
    print(hon[["arm", "n_methods", "dev_tp", "dev_fp", "ext_tp", "ext_fp",
               "ext_recall", "ext_fpr", "ext_precision"]].to_string(index=False))


if __name__ == "__main__":
    main()
