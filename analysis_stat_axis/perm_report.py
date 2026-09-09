import pickle, os
import numpy as np, pandas as pd
OUT = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
PCSV = "/tmp/moe_sweep_stat_permutation.csv"
obs_all = pd.read_csv(f"{OUT}/grid_raw.csv")

FAMS = {
    "ALL":            lambda s, k: True,
    "ALL_main":       lambda s, k: s == "main",
    "ALL_front":      lambda s, k: s == "front",
    "INSTANT_nohist": lambda s, k: k.startswith("I:"),
    "HIST_self":      lambda s, k: k.startswith("H:"),
    "REF_otherbranch":lambda s, k: k.startswith("R:"),
}
rows = []
for tag in ["A", "B"]:
    keys = pickle.load(open(f"{OUT}/keys_{tag}.pkl", "rb"))
    null = np.load(f"/tmp/perm_null_stat_{tag}.npy")          # (200, ncell)
    o = obs_all[obs_all.corpus == tag]
    obsv = np.array([r.auc_success for r in o.itertuples()])
    assert [(r.slice, r.stat, int(r.t)) for r in o.itertuples()] == keys
    sl = np.array([k[0] for k in keys]); st = np.array([k[1] for k in keys])
    tt = np.array([k[2] for k in keys])
    for fname, f in FAMS.items():
        base = np.array([f(a, b) for a, b in zip(sl, st)])
        for trng, tlab in [(tt <= 35, "all_t"), (tt <= 30, "t<=30")]:
            m = base & trng
            if not m.any():
                continue
            for orient, fn in [("two_sided", lambda x: np.maximum(x, 1 - x)),
                               ("one_sided", lambda x: x)]:
                ov = fn(obsv[m]); nv = fn(null[:, m])
                with np.errstate(invalid="ignore"):
                    obs = np.nanmax(ov)
                    nmax = np.nanmax(nv, axis=1)
                p = (1 + (nmax >= obs).sum()) / (len(nmax) + 1)
                arg = np.nanargmax(ov)
                idx = np.where(m)[0][arg]
                rows.append(dict(corpus=tag, family=f"{fname}|{tlab}|{orient}",
                                 obs=round(float(obs), 4), p=round(float(p), 4),
                                 null_p95=round(float(np.quantile(nmax, .95)), 4),
                                 null_max=round(float(nmax.max()), 4),
                                 null_med=round(float(np.median(nmax)), 4),
                                 best_stat=st[idx], best_slice=sl[idx], best_t=int(tt[idx]),
                                 best_auc_raw=round(float(obsv[idx]), 4),
                                 n_cells=int(m.sum()), n_draw=len(nmax)))
res = pd.DataFrame(rows)
res.to_csv(PCSV, index=False)

# ---- targeted pricing of the two cells the coordinator flagged (single-cell, no family max)
tg = []
for tag in ["A", "B"]:
    keys = pickle.load(open(f"{OUT}/keys_{tag}.pkl", "rb"))
    null = np.load(f"/tmp/perm_null_stat_{tag}.npy")
    o = obs_all[obs_all.corpus == tag]
    obsv = np.array([r.auc_success for r in o.itertuples()])
    for sl_, st_ in [("front", "H:hel_prev_mean_W12"), ("main", "H:ewm_rate_a0.15"),
                     ("main", "H:hel_prev_mean_W8"), ("main", "R:succ_minus_fail_ref_W12"),
                     ("main", "R:hel_to_succ_ref_W8"), ("main", "I:tok_heterog")]:
        for t_ in [30]:
            i = keys.index((sl_, st_, t_))
            ob, nl = obsv[i], null[:, i]
            two = max(ob, 1 - ob); ntwo = np.maximum(nl, 1 - nl)
            tg.append(dict(corpus=tag, slice=sl_, stat=st_, t=t_, auc_raw=round(ob, 4),
                           auc_oriented=round(two, 4),
                           p_pointwise=round(float((1 + (ntwo >= two).sum()) / 201), 4),
                           null_p95=round(float(np.quantile(ntwo, .95)), 4),
                           null_max=round(float(ntwo.max()), 4)))
tgd = pd.DataFrame(tg)
tgd.to_csv("/tmp/moe_sweep_stat_permutation_pointwise.csv", index=False)
pd.set_option("display.width", 260)
print(res[res.family.str.contains("two_sided")].to_string(index=False))
print()
print(tgd.to_string(index=False))
