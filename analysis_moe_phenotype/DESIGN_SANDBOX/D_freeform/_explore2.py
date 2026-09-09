"""Exploration 2 (not part of deliverable): signal redundancy + template direction sanity.

(1) Spearman correlation among the 8 core signals + min_mobk + mob1_w8, on CLEAN-SUCCESS rows
    only (label-free w.r.t. outcome), to decide whether any pair must be merged before the
    median-vote (a redundant pair would hijack a K=5 median).
(2) Event-aligned mean of the (already sign-flipped) standardized signals, lead -6..+2, to
    confirm on the calibration units that the frozen E2/E3 sign templates point the right way.
"""
import numpy as np, csv
from scipy.stats import rankdata

ROOT = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
UNITS = {"main": "main16x32", "grid": "grid50x8"}
CORE = ["late_flow_volatility", "route_acceleration", "gate_entropy", "top12_margin",
        "token_consensus", "token_dispersion", "layer_disagreement", "flow_com",
        "min_mobk", "mob1_w8"]
SH = ["V", "A", "S_ent", "S_marg", "C", "D_tok", "D_lay", "fcom", "minmk", "mob1w8"]


def load(unit):
    corp = UNITS[unit]
    d = np.load(f"{ROOT}/features/{corp}/{TASK}/rows.npz", allow_pickle=True)
    ep = d["episode_id"].astype(np.int64); cs = d["control_step"].astype(np.int64)
    q = np.zeros_like(cs); seen = {}
    for i in range(len(ep)):
        if ep[i] not in seen: seen[ep[i]] = cs[i]
        q[i] = cs[i] - seen[ep[i]]
    mk = d["mob_k"].astype(np.float64)
    X = np.column_stack([d[s].astype(np.float64) if s != "min_mobk" else np.nanmin(mk[:, 1:], 1)
                         for s in CORE])
    ev = {}
    with open(f"{ROOT}/events/{corp}/{TASK}/events.csv") as f:
        for r in csv.DictReader(f):
            ev[int(r["episode_id"])] = (int(r["scene"]), int(r["success"]), int(r["n_queries"]),
                                        int(r["loop_onset_q"]), int(r["static_onset_q"]))
    return ep, q, X, d["scene"].astype(np.int64), ev


for unit in ["main", "grid"]:
    ep, q, X, scn, ev = load(unit)
    suc = np.array([ev[e][1] for e in ep]); lo = np.array([ev[e][3] for e in ep]); st = np.array([ev[e][4] for e in ep])
    clean = (suc == 1) & (lo < 0) & (st < 0)
    m = clean & (q >= 8) & np.all(np.isfinite(X), 1)
    R = np.column_stack([rankdata(X[m, j]) for j in range(X.shape[1])])
    C = np.corrcoef(R.T)
    print("=" * 90); print(unit, " Spearman on clean-success rows (q>=8), n=", m.sum())
    print("        " + "".join(f"{s:>8s}" for s in SH))
    for i, s in enumerate(SH):
        print(f"{s:>8s}" + "".join(f"{C[i,j]:8.3f}" for j in range(len(SH))))
    hi = [(SH[i], SH[j], C[i, j]) for i in range(len(SH)) for j in range(i + 1, len(SH)) if abs(C[i, j]) > 0.80]
    print("  |rho|>0.80 pairs:", hi)

    # ---- event-aligned check of the frozen sign templates ----
    # standardize each signal by pooled clean-success median/MAD at same q (+-2), no LOGO here (sanity only)
    Z = np.full_like(X, np.nan)
    for qq in range(0, q.max() + 1):
        ref = m & (np.abs(q - qq) <= 2)
        tgt = (q == qq)
        if ref.sum() < 20 or tgt.sum() == 0: continue
        med = np.nanmedian(X[ref], 0)
        mad = 1.4826 * np.nanmedian(np.abs(X[ref] - med), 0)
        mad[mad <= 0] = np.nan
        Z[tgt] = (X[tgt] - med) / mad
    T_LOOP = {"V": +1, "A": +1, "D_tok": +1, "C": -1, "S_ent": -1}
    T_STAT = {"S_ent": +1, "C": +1, "D_tok": -1, "V": -1, "A": -1, "S_marg": -1, "minmk": -1}
    for name, onset_arr, T in [("LOOP", lo, T_LOOP), ("STATIC", st, T_STAT)]:
        rows = []
        for lead in range(-6, 3):
            vals = []
            for e in np.unique(ep):
                o = ev[e][3] if name == "LOOP" else ev[e][4]
                if o < 0 or ev[e][1] == 1: continue   # failing events only
                sel = (ep == e) & (q == o + lead)
                if sel.sum() == 1: vals.append(Z[sel][0])
            if not vals: continue
            V = np.array(vals)
            rows.append((lead, len(V), np.nanmean(V, 0)))
        print(f"  --- {name} failing-event aligned mean z (n_ev, then per-signal) ---")
        print("   lead   n  " + "".join(f"{s:>8s}" for s in SH))
        for lead, n, v in rows:
            print(f"   {lead:+3d} {n:4d}  " + "".join(f"{v[j]:8.2f}" for j in range(len(SH))))
        # template projection (sign-flipped mean)
        idx = {s: i for i, s in enumerate(SH)}
        print("   template-projected (mean of s_i*z_i) per lead:",
              {lead: round(float(np.nanmean([T[s] * v[idx[s]] for s in T])), 2) for lead, n, v in rows})
