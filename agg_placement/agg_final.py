"""Final adjudication: does anything beat the baseline in BOTH corpora, raw and
residualised?  Plus leave-one-group-out stability and the state-token diagnosis."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import agg_core as A

TS = A.TS
NP = 200
BIDX = A.CFG_IDX[A.BASELINE]
F = pd.DataFrame([dict(name=A.cfg_name(c), placement=A.placement_str(c),
                       ls=c[0][0], lp=c[0][1], lo=c[0][2],
                       ds=c[1][0], dp=c[1][1], do=c[1][2],
                       ts=c[2][0], tp=c[2][1], to=c[2][2]) for c in A.CFGS])


def gauc(s, y, g, drop=None):
    num = den = 0.0
    for gg in np.unique(g):
        if drop is not None and gg == drop:
            continue
        m = g == gg
        sf, ss = s[m][y[m] == 0], s[m][y[m] == 1]
        if len(sf) == 0 or len(ss) == 0:
            continue
        r = rankdata(np.concatenate([sf, ss]))
        a = (r[:len(sf)].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sf) * len(ss))
        num += a * len(sf) * len(ss)
        den += len(sf) * len(ss)
    return num / den


def main():
    S = {t: np.load(f"/tmp/agg_stats_{t}.npz") for t in ("A", "B")}
    C = {t: np.load(f"/tmp/agg_cube_{t}.npz") for t in ("A", "B")}
    raw = {tg: np.stack([S[tg][f"auc_raw_{t}"] for t in TS], -1) for tg in ("A", "B")}
    res = {tg: np.stack([S[tg][f"auc_res_{t}"] for t in TS], -1) for tg in ("A", "B")}
    det = {k: {tg: np.maximum(v[tg], 1 - v[tg]) for tg in ("A", "B")}
           for k, v in [("raw", raw), ("res", res)]}
    base = {tg: det["raw"][tg][BIDX] for tg in ("A", "B")}

    print("=" * 96)
    print("(2) DOES ANY AGGREGATION BEAT THE BASELINE IN BOTH CORPORA?")
    same = np.sign(raw["A"] - .5) == np.sign(raw["B"] - .5)
    for j, t in enumerate(TS):
        bA, bB = base["A"][j], base["B"][j]
        gA = det["raw"]["A"][:, j] > bA
        gB = det["raw"]["B"][:, j] > bB
        both = gA & gB & same[:, j]
        print(f"  t={t}: baseline A {bA:.3f} B {bB:.3f}   beats-A {gA.sum():5d}  "
              f"beats-B {gB.sum():5d}  BOTH+same-sign {both.sum():5d} "
              f"({both.sum()/len(gA)*100:.1f}% of {len(gA)})")
        if both.sum():
            m = np.minimum(det["raw"]["A"][:, j], det["raw"]["B"][:, j])
            m = np.where(both, m, 0)
            for k in np.argsort(m)[::-1][:5]:
                print(f"        {F.name[k]:50s} A={det['raw']['A'][k,j]:.3f} "
                      f"B={det['raw']['B'][k,j]:.3f} | resid A={det['res']['A'][k,j]:.3f} "
                      f"B={det['res']['B'][k,j]:.3f}")
    print()
    print("  Residualised: cells with resid det > 0.5 in both AND matching residual sign,")
    print("  ranked by min(A,B), with the joint FWER-95 threshold marked:")
    dA = np.load("/tmp/agg_draws_A.npz")
    dB = np.load("/tmp/agg_draws_B.npz")
    thr = np.percentile(np.minimum(dA["res"], dB["res"]).reshape(NP, -1).max(1), 95)
    sameR = np.sign(res["A"] - .5) == np.sign(res["B"] - .5)
    mn = np.where(sameR, np.minimum(det["res"]["A"], det["res"]["B"]), 0)
    print(f"  joint FWER-95 threshold = {thr:.4f}")
    flat = np.argsort(mn.ravel())[::-1][:12]
    for k in flat:
        i, j = np.unravel_index(k, mn.shape)
        print(f"    {F.name[i]:50s} t={TS[j]} min={mn[i,j]:.4f} "
              f"(A {det['res']['A'][i,j]:.3f} / B {det['res']['B'][i,j]:.3f}) "
              f"{'CLEARS' if mn[i,j]>=thr else ''}")

    print("=" * 96)
    print("STATE-TOKEN DIAGNOSIS of the t=25 survivor family (all use token scope all11)")
    pairs = {
        "all11/B/min  (survivor)": (("all", "B", "mean"), ("all", "A", "max"), ("all11", "B", "min")),
        "act/B/min    (no state)": (("all", "B", "mean"), ("all", "A", "max"), ("act", "B", "min")),
        "state alone           ": (("all", "B", "mean"), ("all", "A", "max"), ("state", "-", "id")),
        "all11/B/mean          ": (("all", "B", "mean"), ("all", "A", "max"), ("all11", "B", "mean")),
        "act/A/mean            ": (("all", "B", "mean"), ("all", "A", "max"), ("act", "A", "mean")),
    }
    idx = {k: A.CFG_IDX[v] for k, v in pairs.items()}
    for tg in ("A", "B"):
        print(f"  corpus {tg}:")
        for k, i in idx.items():
            print(f"    {k}  det t20/25/30 = "
                  f"{det['raw'][tg][i,0]:.3f}/{det['raw'][tg][i,1]:.3f}/{det['raw'][tg][i,2]:.3f}"
                  f"   resid {det['res'][tg][i,0]:.3f}/{det['res'][tg][i,1]:.3f}/{det['res'][tg][i,2]:.3f}")
        cube, y, g = C[tg]["cube"], C[tg]["y"], C[tg]["g"]
        ks = list(idx)
        print("    within-group rank rho between these cells at t=25:")
        M = np.zeros((len(ks), len(ks)))
        for a in range(len(ks)):
            for b in range(len(ks)):
                rr, wt = [], []
                for gg in np.unique(g):
                    m = g == gg
                    if y[m].sum() == 0 or (1 - y[m]).sum() == 0:
                        continue
                    rr.append(np.corrcoef(rankdata(cube[m, idx[ks[a]], 1]),
                                          rankdata(cube[m, idx[ks[b]], 1]))[0, 1])
                    wt.append(m.sum())
                M[a, b] = np.average(rr, weights=wt)
        print(pd.DataFrame(M, index=[k.strip() for k in ks],
                           columns=[k.strip()[:12] for k in ks]).round(2).to_string())

    print("=" * 96)
    print("LEAVE-ONE-GROUP-OUT stability (raw detection AUC) for the leaders")
    LEAD = {
        "baseline": A.BASELINE,
        "tokmin t25": (("all", "B", "mean"), ("all", "A", "max"), ("all11", "B", "min")),
        "tokmin d9 q75 t25": (("all", "B", "q75"), ("d9", "-", "id"), ("all11", "B", "min")),
        "actBstd t30": (("all", "A", "mean"), ("d9", "-", "id"), ("act", "B", "std")),
        "backAmin all11Bstd t30": (("back", "A", "min"), ("d9", "-", "id"), ("all11", "B", "std")),
    }
    for nm, cfg in LEAD.items():
        i = A.CFG_IDX[cfg]
        for tg in ("A", "B"):
            cube, y, g = C[tg]["cube"], C[tg]["y"], C[tg]["g"]
            for j, t in enumerate(TS):
                if t == 20:
                    continue
                vals = []
                for gg in np.unique(g):
                    m = g == gg
                    if y[m].sum() == 0 or (1 - y[m]).sum() == 0:
                        continue
                    a = gauc(cube[:, i, j], y, g, drop=gg)
                    vals.append(max(a, 1 - a))
                full = gauc(cube[:, i, j], y, g)
                print(f"  {nm:24s} {tg} t={t} full={max(full,1-full):.3f} "
                      f"LOGO range {min(vals):.3f}-{max(vals):.3f}")

    print("=" * 96)
    print("(5) VARIANCE: main effects vs interactions (crossed subgrid, det AUC)")
    CROSS = ((F.ds == "all") & (F.ts != "state")).values
    for tg in ("A", "B"):
        for nm in ("raw", "res"):
            v = det[nm][tg][CROSS, 2]
            tot = ((v - v.mean()) ** 2).sum()
            ss = 0.0
            for f in ["ls", "lp", "lo", "dp", "do", "ts", "tp", "to"]:
                lv = F[f].values[CROSS]
                for u in np.unique(lv):
                    m = lv == u
                    ss += m.sum() * (v[m].mean() - v.mean()) ** 2
            print(f"  corpus {tg} {nm} t=30: sum of 8 main-effect eta^2 = {ss/tot:.3f} "
                  f"-> interactions/residual = {1-ss/tot:.3f}")


if __name__ == "__main__":
    main()
