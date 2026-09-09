"""Adversarial checks on the one survivor: the burst-count temporal operator.

If `fracabove_q` is just a noisy restatement of the window mean (plus its spread), it must die
when residualised on a RICHER baseline than the single stated one. That is the decisive test.
"""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

MD = "/tmp/moe_agg_divtime.md"
TS = (20, 25, 30)
OUT = []
N_PERM = 200


def det(a):
    return np.maximum(a, 1 - a)


def multi_residualise(scores, basemat, groups):
    """Within group: OLS of rank(scores) on ranks of every column of basemat; return residual."""
    out = np.full(len(scores), np.nan)
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        if len(m) < basemat.shape[1] + 3:
            continue
        ys = lib._avg_rank(scores[m]); ys -= ys.mean()
        X = np.column_stack([lib._avg_rank(basemat[m, k]) for k in range(basemat.shape[1])])
        X = X - X.mean(0)
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        r = ys - X @ beta
        r[np.abs(r) < 1e-8 * len(m)] = 0.0
        out[m] = r
    return out


def main():
    df = pd.read_csv("/tmp/moe_agg_divtime.csv")
    df["cell"] = df["divergence"] + "|" + df["temporal"] + "|W" + df["window"].astype(str)
    df["det"] = det(df["auc"]); df["det_resid"] = det(df["auc_resid"])

    S = {}
    for tag in ("A", "B"):
        o = pickle.load(open(f"/tmp/moe_agg_series_{tag}.pkl", "rb"))
        thr = {k: tuple(float(np.quantile(d[:, 1:31][np.isfinite(d[:, 1:31])], q))
                        for q in (0.10, 0.25, 0.75, 0.90)) for k, d in o["D"].items()}
        S[tag] = (o, thr)

    def score(tag, dv, tm, W, t, thr_override=None):
        o, thr = S[tag]
        th = thr_override if thr_override is not None else thr[dv]
        return lib.aggregate(o["D"][dv], t, tm, W, thr=th)

    # ---------------- 1. raw + residual values of the burst family --------------------
    L = ["\n## 14. The one survivor in close-up: burst-count aggregation "
         "(`fracabove_q75` / its mirror `fracbelow_q25` on a similarity)\n",
         "`fracabove_q75|W` = fraction of the W chunk transitions in the window whose divergence "
         "exceeds a **global, label-free** 75th-percentile threshold (one number per corpus per "
         "divergence, estimated over all branches x steps 1..30). It counts *bursts* of route "
         "change rather than averaging the amount of it.\n\n```"]
    sub = df[(df.t == 30) & (df.temporal.isin(["fracabove_q75", "fracabove_q90", "mean"])) &
             (df.divergence == "hell|prev")]
    L.append(sub.pivot_table(index=["temporal", "window"], columns="corpus",
                             values=["det", "det_resid"]).round(4).to_string())
    L.append("```")
    OUT.append("\n".join(L) + "\n")

    # ---------------- 2. DECISIVE: residualise on a richer baseline -------------------
    setB1 = [("hell|prev", "mean", 8)]
    setB2 = setB1 + [("hell|prev", "mean", W) for W in (3, 5, 12, 20)]
    setB3 = setB2 + [("hell|prev", "std", 8), ("hell|prev", "max", 8), ("hell|prev", "min", 8),
                     ("hell|prev", "median", 8), ("hell|prev", "ewmall0.15", 0)]
    TESTC = [("hell|prev", "fracabove_q75", 12), ("hell|prev", "fracabove_q75", 20),
             ("cos|prev", "fracabove_q75", 12), ("hell|prev", "fracabove_q90", 12),
             ("hell|prev", "fracbelow_q10", 12), ("hell|prev", "ewm0.15", 12),
             ("hell|prev", "median", 3), ("hell|prev", "max", 3), ("hell|prev", "slope", 20),
             ("hell|runmean", "mean", 8), ("dent_abs", "mean", 8), ("dtop1_abs", "mean", 8)]
    rows, nulls = [], {}
    for tag in ("A", "B"):
        o, _ = S[tag]
        groups, y = o["groups"], o["y"]
        rng = np.random.default_rng(4242)
        keep = np.zeros(len(y), bool)
        for g in np.unique(groups):
            m = np.where(groups == g)[0]
            if y[m].sum() and (~y[m]).sum():
                keep[m] = True
        perms = []
        for _ in range(N_PERM):
            v = y.copy()
            for g in np.unique(groups):
                m = np.where(groups == g)[0]
                v[m] = rng.permutation(v[m])
            perms.append(v)
        for t in TS:
            for nm, bset in (("B1_stated", setB1), ("B2_all_window_means", setB2),
                             ("B3_means+moments", setB3)):
                Bm = np.column_stack([score(tag, *b[:1], b[1], b[2], t) if False else
                                      score(tag, b[0], b[1], b[2], t) for b in bset])
                for c in TESTC:
                    s = score(tag, c[0], c[1], c[2], t)
                    r = multi_residualise(s, Bm, groups)
                    a, _ = lib.within_group_auc(r, y, groups)
                    rec = dict(corpus=tag, t=t, baseline_set=nm,
                               cell=f"{c[0]}|{c[1]}|W{c[2]}", auc=a, det=det(a))
                    if nm == "B3_means+moments" and t == 30:
                        nd = np.array([lib.within_group_auc(r, yp, groups)[0] for yp in perms])
                        rec["p_point"] = (1 + (det(nd) >= det(a)).sum()) / (1 + N_PERM)
                        nulls[(tag, rec["cell"])] = det(nd)
                    rows.append(rec)
    RD = pd.DataFrame(rows)
    RD.to_csv("/tmp/moe_agg_richer_baseline.csv", index=False)
    L = ["\n## 15. DECISIVE TEST: does the burst operator survive a *richer* baseline?\n",
         "B1 = the stated baseline only (`hell|prev` window mean W=8).",
         "B2 = every window mean (W=3,5,8,12,20) -- kills anything that is just 'a longer memory'.",
         "B3 = B2 + the window std / max / min / median / full-history EWM at W=8 -- kills anything "
         "that is just 'the mean plus its spread or its extremes'.",
         "All residualisation is label-free rank OLS within group. Entries are detection AUC of the "
         "residual.\n\n```"]
    for t in TS:
        L.append(f"\n  t={t}")
        pv = RD[RD.t == t].pivot_table(index="cell", columns=["baseline_set", "corpus"], values="det")
        L.append("  " + pv.round(3).to_string().replace("\n", "\n  "))
    L.append("\n  pointwise permutation p under B3 at t=30:")
    pp = RD[(RD.t == 30) & (RD.baseline_set == "B3_means+moments")].pivot_table(
        index="cell", columns="corpus", values="p_point")
    L.append("  " + pp.round(4).to_string().replace("\n", "\n  "))
    L.append("```")
    OUT.append("\n".join(L) + "\n")

    # ---------------- 3. threshold sensitivity + branch-internal threshold -------------
    L = ["\n## 16. Is the burst operator an artefact of the particular threshold?\n\n```"]
    o, thr = S["A"]
    rows = []
    for tag in ("A", "B"):
        o, thr = S[tag]
        d = o["D"]["hell|prev"]
        v = d[:, 1:31][np.isfinite(d[:, 1:31])]
        base = score(tag, "hell|prev", "mean", 8, 30)
        for q in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95):
            th = (0, 0, float(np.quantile(v, q)), float(np.quantile(v, q)))
            s = lib.aggregate(d, 30, "fracabove_q75", 12, thr=th)
            r = lib.rank_residualise(s, base, o["groups"])
            a, _ = lib.within_group_auc(s, o["y"], o["groups"])
            ar, _ = lib.within_group_auc(r, o["y"], o["groups"])
            rows.append(dict(corpus=tag, thresh="global_q%.2f" % q, det=det(a), det_resid=det(ar)))
        # branch-internal threshold: each branch's own q75 over its first 30 transitions
        bq = np.nanquantile(d[:, 1:31], 0.75, axis=1)
        seg = d[:, 30 - 12 + 1:31]
        s = (seg > bq[:, None]).mean(1)
        r = lib.rank_residualise(s, base, o["groups"])
        a, _ = lib.within_group_auc(s, o["y"], o["groups"])
        ar, _ = lib.within_group_auc(r, o["y"], o["groups"])
        rows.append(dict(corpus=tag, thresh="branch_own_q75", det=det(a), det_resid=det(ar)))
    TD = pd.DataFrame(rows)
    TD.to_csv("/tmp/moe_agg_threshold_sens.csv", index=False)
    L.append(TD.pivot_table(index="thresh", columns="corpus",
                            values=["det", "det_resid"]).round(4).to_string())
    L.append("```")
    OUT.append("\n".join(L) + "\n")

    # ---------------- 4. redundancy within the metric family --------------------------
    L = ["\n## 17. The 11 f-divergences are one quantity, not eleven\n\n```"]
    for tag in ("A", "B"):
        o, thr = S[tag]
        b = lib._avg_rank(score(tag, "hell|prev", "mean", 8, 30))
        line = []
        for m in lib.PAIRWISE:
            r = lib._avg_rank(score(tag, f"{m}|prev", "mean", 8, 30))
            rho = np.corrcoef(b, r)[0, 1]
            line.append(f"{m}={rho:+.4f}")
        L.append(f"  corpus {tag}, Spearman vs hell|prev|mean|W8 (t=30): " + "  ".join(line))
    L.append("```")
    OUT.append("\n".join(L) + "\n")

    # ---------------- 5. reduced (de-duplicated) family-wise p ------------------------
    G = {t: pickle.load(open(f"/tmp/moe_agg_grid_{t}.pkl", "rb")) for t in ("A", "B")}
    cellnames = np.array([f"{a}|{b}|W{c}" for a, b, c in G["A"]["keys"]])
    keepdiv = {"hell|prev", "hell|runmean", "dent_abs", "dent_sgn", "dtop1_abs", "dtop1_sgn"}
    mask = np.array([c.rsplit("|", 2)[0] in keepdiv for c in cellnames])
    L = [f"\n## 18. Family-wise p on the DE-DUPLICATED family ({int(mask.sum())} cells: one "
         "representative metric `hell` x both references + the 4 scalar-change measures x 69 "
         "temporal specs)\n",
         "Justified by section 17: the 11 f-divergences are rank-correlated 0.97-1.00 with each "
         "other, so counting them as 11 members inflates the family-max null. This family is the "
         "one that would have been pre-specified knowing that.\n"]
    for tag in ("A", "B"):
        for kind, col in (("raw", "det"), ("resid", "det_resid")):
            null = np.abs(np.load(f"/tmp/moe_agg_null_{tag}_{kind}.npy") - .5) + .5
            obs = np.empty((len(cellnames), len(TS)))
            for j, t in enumerate(TS):
                s = df[(df.corpus == tag) & (df.t == t)].set_index("cell")
                obs[:, j] = s.loc[cellnames, col].values
            oo = obs[mask].max()
            nm = null[:, mask, :].max(axis=(1, 2))
            p = (1 + (nm >= oo).sum()) / (1 + len(nm))
            ij = np.unravel_index(obs[mask].argmax(), obs[mask].shape)
            L.append(f"* corpus **{tag}**, **{kind}**: argmax `{cellnames[mask][ij[0]]}` at "
                     f"t={TS[ij[1]]}, observed **{oo:.4f}**, null p95 {np.percentile(nm,95):.4f}, "
                     f"null max {nm.max():.4f} -> **p = {p:.4f}**")
    OUT.append("\n".join(L) + "\n")

    # ---------------- 6. length / phase confound diagnostic ---------------------------
    L = ["\n## 19. Phase-confound diagnostic (a corpus property, not a property of any one "
         "statistic)\n\n```"]
    for tag in ("A", "B"):
        o, thr = S[tag]
        T, groups, y = o["T"], o["groups"], o["y"]
        base = score(tag, "hell|prev", "mean", 8, 30)
        burst = score(tag, "hell|prev", "fracabove_q75", 12, 30)
        rb = lib.rank_residualise(burst, base, groups)
        c1 = c2 = c3 = 0.0; n = 0
        for g in np.unique(groups):
            m = np.where(groups == g)[0]
            if len(m) < 5:
                continue
            c1 += np.corrcoef(lib._avg_rank(base[m]), lib._avg_rank(T[m].astype(float)))[0, 1] * len(m)
            c2 += np.corrcoef(lib._avg_rank(burst[m]), lib._avg_rank(T[m].astype(float)))[0, 1] * len(m)
            c3 += np.corrcoef(rb[m], lib._avg_rank(T[m].astype(float)))[0, 1] * len(m)
            n += len(m)
        L.append(f"  corpus {tag}: within-group Spearman with episode length T -- "
                 f"baseline {c1/n:+.3f}, burst {c2/n:+.3f}, burst-residual {c3/n:+.3f}")
    L.append("```\n\nAt t=30 the successes are much closer to their own termination than the "
             "failures are, so every statistic here reads task phase to some degree. The residual "
             "correlation with T shows how much of the *incremental* burst signal is phase.")
    OUT.append("\n".join(L) + "\n")

    with open(MD, "a") as fh:
        fh.write("\n".join(OUT) + "\n")
    print("appended to", MD)


if __name__ == "__main__":
    main()
