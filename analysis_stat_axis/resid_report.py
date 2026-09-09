import pickle
import numpy as np, pandas as pd
OUT = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
obs_all = pd.read_csv(f"{OUT}/grid_raw.csv")
FAMS = {"ALL": lambda s, k: True,
        "ALL_main": lambda s, k: s == "main",
        "ALL_front": lambda s, k: s == "front",
        "INSTANT_nohist": lambda s, k: k.startswith("I:"),
        "HIST_self": lambda s, k: k.startswith("H:"),
        "REF_otherbranch": lambda s, k: k.startswith("R:")}
BASE = ("main", "H:hel_prev_mean_W8")

cells, famrows = [], []
for tag in ["A", "B"]:
    keys = pickle.load(open(f"{OUT}/keys_{tag}.pkl", "rb"))
    sl = np.array([k[0] for k in keys]); st = np.array([k[1] for k in keys])
    tt = np.array([k[2] for k in keys])
    raw_obs = obs_all[obs_all.corpus == tag].auc_success.values
    raw_nul = np.load(f"/tmp/perm_null_stat_{tag}.npy")
    res_obs = np.load(f"/tmp/resid_obs_{tag}.npy")
    res_nul = np.load(f"/tmp/perm_null_resid_{tag}.npy")
    o = lambda x: np.maximum(x, 1 - x)
    cells.append(pd.DataFrame(dict(corpus=tag, slice=sl, statistic=st, t=tt,
                                   auc_raw=raw_obs, auc_resid=res_obs,
                                   oriented_raw=o(raw_obs), oriented_resid=o(res_obs))))
    for fname, f in FAMS.items():
        m = np.array([f(a, b) for a, b in zip(sl, st)]) & (tt <= 30)
        for lab, ob, nu in [("RAW", raw_obs, raw_nul), ("RESIDUALISED", res_obs, res_nul)]:
            ov, nv = o(ob[m]), o(nu[:, m])
            with np.errstate(invalid="ignore"):
                obs = np.nanmax(ov); nmax = np.nanmax(nv, axis=1)
            i = np.where(m)[0][np.nanargmax(ov)]
            famrows.append(dict(corpus=tag, family=f"{fname}|t<=30|{lab}",
                                obs=round(float(obs), 4),
                                p=round(float((1 + (nmax >= obs).sum()) / 201), 4),
                                null_p95=round(float(np.quantile(nmax, .95)), 4),
                                null_max=round(float(nmax.max()), 4),
                                best_stat=st[i], best_slice=sl[i], best_t=int(tt[i]),
                                n_cells=int(m.sum())))
C = pd.concat(cells, ignore_index=True)
C.to_csv("/tmp/moe_sweep_stat_residual_cells.csv", index=False)
F = pd.DataFrame(famrows)
prev = pd.read_csv("/tmp/moe_sweep_stat_permutation.csv")
pd.concat([prev, F], ignore_index=True).to_csv("/tmp/moe_sweep_stat_permutation.csv", index=False)

# ---- cross-corpus sign agreement of the RESIDUALS (t<=30, baseline cell excluded: identically 0)
w = C[(C.t <= 30)].pivot_table(index=["slice", "statistic", "t"],
                               columns="corpus", values="auc_resid").dropna()
w = w[~((w.index.get_level_values("slice") == BASE[0]) &
        (w.index.get_level_values("statistic") == BASE[1]))]
sgn = (np.sign(w.A - 0.5) == np.sign(w.B - 0.5))
print(f"\nCROSS-CORPUS SIGN AGREEMENT OF RESIDUALS (t<=30, n={len(w)} cells): {sgn.mean():.1%}  (chance 50%)")
for fam, pref in [("instantaneous", "I:"), ("historical_self", "H:"), ("reference", "R:")]:
    m = w.index.get_level_values("statistic").str.startswith(pref)
    print(f"   {fam:16s} n={m.sum():3d}  agreement {sgn[m].mean():.1%}")
# raw for comparison
wr = C[(C.t <= 30)].pivot_table(index=["slice", "statistic", "t"], columns="corpus", values="auc_raw").dropna()
print(f"   (raw, un-residualised, n={len(wr)}): {(np.sign(wr.A-.5)==np.sign(wr.B-.5)).mean():.1%}")

pd.set_option("display.width", 260)
print("\n=== FAMILY-WISE, t<=30, RAW vs RESIDUALISED ===")
print(F.to_string(index=False))
print("\n=== SHRINKAGE OF THE HEADLINE CELLS AT t=30 (oriented AUC) ===")
key = [("front", "H:hel_prev_mean_W12"), ("main", "H:ewm_rate_a0.15"), ("main", "H:hel_prev_mean_W12"),
       ("main", "H:hel_prev_mean_W8"), ("main", "H:cum_speed"), ("main", "H:acf_lag3"),
       ("front", "H:cum_speed"), ("main", "I:tok_heterog"), ("main", "R:succ_minus_fail_ref_W12"),
       ("main", "R:hel_to_succ_ref_W8"), ("main", "R:hel_to_fail_ref"), ("main", "R:hel_to_cohort_LOO_W8")]
k = C[(C.t == 30)].set_index(["slice", "statistic", "corpus"])
rows = []
for s_, st_ in key:
    r = dict(slice=s_, statistic=st_)
    for c in ["A", "B"]:
        r[f"{c}_raw"] = round(float(k.loc[(s_, st_, c), "oriented_raw"]), 3)
        r[f"{c}_resid"] = round(float(k.loc[(s_, st_, c), "oriented_resid"]), 3)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
