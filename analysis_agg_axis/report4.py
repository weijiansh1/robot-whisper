"""Final verdict block: recommendation table, corrected phase diagnostic, headline summary."""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

TS = (20, 25, 30)
OUT = []


def det(a):
    return np.maximum(a, 1 - a)


def main():
    df = pd.read_csv("/tmp/moe_agg_divtime.csv")
    df["cell"] = df["divergence"] + "|" + df["temporal"] + "|W" + df["window"].astype(str)
    df["det"] = det(df["auc"]); df["det_resid"] = det(df["auc_resid"])

    # ---- corrected phase diagnostic (skip groups with no variation in T) --------------
    L = ["\n## 21. Phase-confound diagnostic, corrected\n\n```"]
    for tag in ("A", "B"):
        o = pickle.load(open(f"/tmp/moe_agg_series_{tag}.pkl", "rb"))
        thr = {k: tuple(float(np.quantile(d[:, 1:31][np.isfinite(d[:, 1:31])], q))
                        for q in (0.10, 0.25, 0.75, 0.90)) for k, d in o["D"].items()}
        T, groups, y = o["T"].astype(float), o["groups"], o["y"]
        base = lib.aggregate(o["D"]["hell|prev"], 30, "mean", 8, thr=thr["hell|prev"])
        burst = lib.aggregate(o["D"]["hell|prev"], 30, "fracabove_q75", 12, thr=thr["hell|prev"])
        rb = lib.rank_residualise(burst, base, groups)
        acc = {"baseline": 0.0, "burst": 0.0, "burst_resid": 0.0}
        n = 0
        for g in np.unique(groups):
            m = np.where(groups == g)[0]
            rt = lib._avg_rank(T[m])
            if rt.std() < 1e-9 or len(m) < 5:
                continue
            for k, v in (("baseline", base[m]), ("burst", burst[m]), ("burst_resid", rb[m])):
                x = v if k == "burst_resid" else lib._avg_rank(v)
                acc[k] += (0.0 if np.std(x) < 1e-12 else np.corrcoef(x, rt)[0, 1]) * len(m)
            n += len(m)
        L.append(f"  corpus {tag} (n={n} branches in groups where T actually varies): "
                 + "  ".join(f"{k}={v/n:+.3f}" for k, v in acc.items()))
    L.append("```\n\nThe two corpora differ. In A the baseline's within-group correlation with "
             "episode length is small (-0.079). In B it is -0.475: within an initial state, more "
             "route change means an earlier end, and successes end earlier, so a substantial "
             "phase/duration channel is present in the BASELINE itself and is inherited by every "
             "cell in this grid. Same length confound as the 08-28 audit; nothing here fixes it.")
    OUT.append("\n".join(L) + "\n")

    # ---- recommendation table ---------------------------------------------------------
    CAND = [("hell|prev", "mean", 8, "BASELINE (stated)"),
            ("hell|prev", "mean", 12, "longer window"),
            ("hell|prev", "ewmall0.15", 0, "full-history EWM a=0.15"),
            ("hell|prev", "ewm0.15", 12, "windowed EWM a=0.15, W=12"),
            ("hell|prev", "fracabove_q75", 12, "burst count (the near-miss)"),
            ("hell|prev", "median", 3, "short median"),
            ("hell|prev", "max", 3, "short max"),
            ("hell|prev", "std", 8, "window std"),
            ("hell|prev", "slope", 8, "window slope"),
            ("hell|prev", "cum_over_t", 0, "cumulative / t"),
            ("emd|prev", "mean", 8, "EMD instead of Hellinger"),
            ("hell|runmean", "mean", 8, "vs own running mean"),
            ("dent_abs", "mean", 8, "|entropy change|"),
            ("dtop1_abs", "mean", 8, "|top-1 mass change|")]
    L = ["\n## 22. Recommendation table (raw detection AUC; the residual column is w.r.t. the "
         "stated baseline)\n\n```"]
    rows = []
    for dv, tm, W, note in CAND:
        c = f"{dv}|{tm}|W{W}"
        r = {"spec": c, "note": note}
        for t in TS:
            for tag in ("A", "B"):
                s = df[(df.cell == c) & (df.t == t) & (df.corpus == tag)]
                r[f"t{t}_{tag}"] = float(s.det.iloc[0])
                if t == 30:
                    r[f"res30_{tag}"] = float(s.det_resid.iloc[0])
        rows.append(r)
    RT = pd.DataFrame(rows)
    RT.to_csv("/tmp/moe_agg_recommendation.csv", index=False)
    L.append(RT.set_index("spec").round(3).to_string())
    L.append("```")
    OUT.append("\n".join(L) + "\n")

    # ---- headline -------------------------------------------------------------------
    OUT.append("""
## 23. Verdict

**Clean negative on both stages, with one instructive near-miss.**

1. **The divergence functional is a free parameter with no content.** The ten f-divergences
   (Hellinger, Bhattacharyya coefficient and distance, symmetric KL, Jensen-Shannon, total
   variation / L1, L2, cosine, chi-squared) are rank-correlated **0.989-1.000** with each other
   on the same temporal spec in both corpora (section 17). Their best detection AUCs at t=30
   span **0.805-0.810 (A)** and **0.785-0.791 (B)** -- a spread of 0.005, i.e. nothing. Only the
   earth-mover distance separates, and it separates *downwards* (rho 0.89 in B, det 0.775/0.780),
   which is expected: it is the only one of the eleven that depends on the arbitrary numbering of
   the 32 experts. Restricted to this family, the divergence axis explains **0.3-6%** of the
   variance in the grid while the temporal axis explains **65-89%** (section 11).

2. **What each chunk is compared *against* does matter, and the current choice is already the
   right one.** Previous-chunk beats the branch's own running mean and beats the scalar-change
   measures (|entropy change|, |top-1 mass change|) in both corpora at every t
   (t=30 best det: 0.810/0.791 vs 0.758/0.721 vs 0.704/0.694), and its cross-corpus residual sign
   agreement is far better (0.70-0.90 for `|prev` vs 0.36-0.73 for `|runmean`). The apparent
   importance of "the divergence axis" in the unrestricted variance decomposition (36-46% at
   t=30) is entirely this reference contrast, not the choice of metric.

3. **The temporal aggregation has the largest spread of anything measured, but the spread is
   almost all downside.** Marginal detection AUC across temporal specs ranges over 0.24-0.28 AUC
   at t=30 -- larger than any layer/token/denoise effect in the previous sweep, which is why it
   had to be searched. But the window mean is already near the top of that range: the best
   temporal spec in the whole grid beats it by only **+0.011 (A, `cos|prev|fracabove_q75|W12`
   0.8096)** and **+0.040 (B, `symkl|prev|median|W3` 0.7913)** raw. Note the two corpora do not
   even agree on *which direction* the improvement lies in: corpus A prefers longer memory
   (W=12, EWM alpha=0.15), corpus B's raw winner is a **3-step median** while its runner-up is a
   20-step mean -- the classic signature of picking noise. Averaged over divergences, windowed
   max, min, std, slope and the quantile-crossing counts are all worse than the mean.

4. **Nothing clears in both corpora after residualisation.** Family-max on the residual grid:
   **A p = 0.189 (fails), B p = 0.005 (passes)**, and the two corpora peak at different
   statistics. De-duplicating the metric family (section 18) does not change the verdict:
   **A p = 0.144, B p = 0.005**, again with different argmaxes (A `dent_sgn|max|W8` at t=20,
   B `hell|prev|fracabove_q75|W5` at t=30). By the stated rule -- a cell counts only if it
   clears in both corpora -- **zero cells count.**

5. **The near-miss, and why it is not real.** A *burst count* aggregation --
   `fracabove_q75|W12`, the fraction of chunk transitions in the window whose divergence exceeds
   a global 75th-percentile threshold -- was the single most reproducible residual leader:
   residual det-AUC **A 0.646 / B 0.634** at t=30, pointwise p = 0.005 in both corpora, stable
   across all ten f-divergences and across thresholds q0.60-q0.80, and it drove a joint
   cross-corpus clearing count of 128 cells against a null max of 76 (p = 0.005). **It dies under
   a richer baseline.** Residualised on all five window means instead of just W=8, it falls to
   0.547/0.558; residualised additionally on the window std/max/min/median and the full-history
   EWM, it falls to **0.521/0.507 with pointwise p = 0.67 / 0.86** (section 15). The joint
   clearing count falls 128 -> 42 -> **16 (p = 0.085)** along the same ladder (section 20b).
   The burst count was reading nothing but the *memory length* of the same shared factor.

6. **The prior "longer memory" result is reproduced, not overturned.** W=12 and EWM alpha=0.15
   again beat W=8 raw in both corpora at t=30 (A 0.807/0.800 vs 0.799; B 0.769/0.776 vs 0.751),
   again fail residualisation in corpus A (family-max p = 0.189 vs the previous round's 0.110),
   and again the two corpora peak at different statistics. The evidence is *exactly* as strong
   and exactly as insufficient as last time. **Do not promote W=12 over W=8.**

### Recommendation

**Keep the baseline aggregation unchanged: Hellinger to the previous chunk, window mean, W=8.**

The reason is not that it won -- it is that **the grid contains no alternative that wins.** More
precisely:

* Changing the *metric* is free and pointless; use whichever is cheapest. Hellinger is bounded
  in [0,1], needs no epsilon guard (unlike symmetric KL / chi-squared, which need clipping and
  are unbounded), and is already the convention. **Never use EMD** on this axis: expert index has
  no metric meaning, and it measurably costs 0.03 AUC.
* Keep `|prev` as the reference. This is the one choice in the whole grid that is doing real work
  and it is already correct.
* Report W=8 as the headline, but **quote the memory-length sensitivity band explicitly**
  (W in 3..20 and EWM alpha in 0.15..0.5 give A 0.758-0.807 / B 0.751-0.781 at t=30). That band
  is honest uncertainty about a free parameter, not a menu to pick the maximum from.
* Do **not** adopt any thresholded / burst-count / order-statistic aggregation. Section 15 is
  the reason: they look like new information against a single-baseline residual and stop looking
  like it against a two-moment one.

### What this means for the sweep as a whole

The previous round found that varying *which slice* of the tensor to use buys nothing because
chunk-to-chunk route change is a single shared factor across layers, tokens and denoise steps.
This round finds the same thing along the two remaining axes, and sharpens it: the factor is
also single across **how you measure the change** (ten divergences, rho >= 0.99) and across
**how you summarise it in time** (every alternative aggregation collapses onto the family of
window means). The aggregation choice was worth 0.109 AUC in corpus B and therefore had to be
searched -- but that value turns out to be the *cost of choosing badly*, not headroom above the
current choice. **The tensor has one degree of freedom under this reading, and the baseline is
already sitting on it.**
""")

    with open("/tmp/moe_agg_divtime.md", "a") as fh:
        fh.write("\n".join(OUT) + "\n")
    print("appended sections 21-23")


if __name__ == "__main__":
    main()
