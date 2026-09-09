"""Follow-ups on the surviving cells: per-group breakdown, cross-t behaviour,
null count calibration, the four V1-V4 legacy variants, redundancy with the baseline."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import agg_core as A

TS = A.TS
NP = 200
BIDX = A.CFG_IDX[A.BASELINE]

CAND = {
    "baseline(V2)": (("back", "A", "mean"), ("d9", "-", "id"), ("act", "A", "mean")),
    "V1 prob-avg both": (("back", "B", "mean"), ("d9", "-", "id"), ("act", "B", "mean")),
    "V3 layer-first": (("back", "B", "mean"), ("d9", "-", "id"), ("act", "A", "mean")),
    "V4 token-first": (("back", "A", "mean"), ("d9", "-", "id"), ("act", "B", "mean")),
    "tokminALL/Lallmean": (("all", "B", "mean"), ("all", "A", "max"), ("all11", "B", "min")),
    "tokminALL/Lfrontmed": (("front", "B", "median"), ("d9", "-", "id"), ("all11", "B", "min")),
    "tokminALL/d9/Lallq75": (("all", "B", "q75"), ("d9", "-", "id"), ("all11", "B", "min")),
    "tokstdALL/backAmin": (("back", "A", "min"), ("d9", "-", "id"), ("all11", "B", "std")),
    "tokstdACT/allAmean": (("all", "A", "mean"), ("d9", "-", "id"), ("act", "B", "std")),
    "backBmin/tokAmean": (("back", "B", "min"), ("all", "B", "max"), ("all11", "A", "mean")),
}


def group_auc_detail(s, y, g):
    per, num, den = {}, 0.0, 0.0
    for gg in np.unique(g):
        m = g == gg
        sf, ss = s[m][y[m] == 0], s[m][y[m] == 1]
        if len(sf) == 0 or len(ss) == 0:
            continue
        r = rankdata(np.concatenate([sf, ss]))
        a = (r[:len(sf)].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sf) * len(ss))
        per[int(gg)] = (a, len(sf) * len(ss))
        num += a * len(sf) * len(ss)
        den += len(sf) * len(ss)
    return num / den, per


def main():
    out = []
    D = {}
    for tag in ("A", "B"):
        z = np.load(f"/tmp/agg_cube_{tag}.npz")
        D[tag] = (z["cube"], z["y"], z["g"])

    print("=" * 100)
    print("LEGACY V1-V4 (back block, action tokens, denoise 9, Hellinger, 8-step mean)")
    print("  raw directional AUC / detection AUC")
    for nm in ["baseline(V2)", "V1 prob-avg both", "V3 layer-first", "V4 token-first"]:
        i = A.CFG_IDX[CAND[nm]]
        line = f"  {nm:20s}"
        for tag in ("A", "B"):
            c, y, g = D[tag]
            for j, t in enumerate(TS):
                if t != 30:
                    continue
                a, _ = group_auc_detail(c[:, i, j], y, g)
                line += f"  {tag} t30 raw={a:.4f} det={max(a,1-a):.4f}"
        print(line)
    print("  corpus B spread across V1-V4 at t=30:", end=" ")
    vals = []
    for nm in ["V1 prob-avg both", "baseline(V2)", "V3 layer-first", "V4 token-first"]:
        i = A.CFG_IDX[CAND[nm]]
        c, y, g = D["B"]
        a, _ = group_auc_detail(c[:, i, 2], y, g)
        vals.append(max(a, 1 - a))
    print(f"{min(vals):.3f} .. {max(vals):.3f}  (range {max(vals)-min(vals):.3f})")

    print("=" * 100)
    print("CANDIDATE CELLS: detection AUC at every t, raw and residual, both corpora")
    S = {tag: np.load(f"/tmp/agg_stats_{tag}.npz") for tag in ("A", "B")}
    hdr = f"{'cell':24s}" + "".join(
        f" | {tag} t{t} raw/res" for tag in ("A", "B") for t in TS)
    print(hdr)
    for nm, cfg in CAND.items():
        i = A.CFG_IDX[cfg]
        line = f"{nm:24s}"
        for tag in ("A", "B"):
            for j, t in enumerate(TS):
                r = S[tag][f"auc_raw_{t}"][i]
                e = S[tag][f"auc_res_{t}"][i]
                line += f" | {max(r,1-r):.3f}/{max(e,1-e):.3f}"
        print(line)
    print()
    print("within-group rank |rho| with the baseline (t=30 / t=25):")
    for nm, cfg in CAND.items():
        i = A.CFG_IDX[cfg]
        print(f"  {nm:24s} A {abs(S['A']['rho_30'][i]):.3f}/{abs(S['A']['rho_25'][i]):.3f}"
              f"   B {abs(S['B']['rho_30'][i]):.3f}/{abs(S['B']['rho_25'][i]):.3f}")

    print("=" * 100)
    print("PER-GROUP breakdown of the two joint-residual leaders (raw scores)")
    for nm in ["tokminALL/Lallmean", "tokstdALL/backAmin", "baseline(V2)"]:
        i = A.CFG_IDX[CAND[nm]]
        for tag in ("A", "B"):
            c, y, g = D[tag]
            for j, t in enumerate(TS):
                if (nm.startswith("tokmin") and t != 25) or (not nm.startswith("tokmin") and t != 30):
                    continue
                a, per = group_auc_detail(c[:, i, j], y, g)
                pg = " ".join(f"{k}:{v[0]:.2f}({v[1]})" for k, v in per.items())
                print(f"  {nm:22s} {tag} t={t} pooled={a:.3f} | {pg}")

    print("=" * 100)
    print("NULL CALIBRATION of the 'clears in both corpora' count")
    dA = np.load("/tmp/agg_draws_A.npz")
    dB = np.load("/tmp/agg_draws_B.npz")
    for grid in ("raw", "res"):
        j = np.minimum(dA[grid], dB[grid]).reshape(NP, -1)
        thr = np.percentile(j.max(1), 95)
        cnt = (j >= thr).sum(1)
        obsA = np.stack([S["A"][f"auc_{grid}_{t}"] for t in TS], -1)
        obsB = np.stack([S["B"][f"auc_{grid}_{t}"] for t in TS], -1)
        o = np.where(np.sign(obsA - .5) == np.sign(obsB - .5),
                     np.minimum(np.maximum(obsA, 1 - obsA), np.maximum(obsB, 1 - obsB)), 0.0)
        print(f"  {grid}: threshold {thr:.4f}; observed cells clearing = {(o>=thr).sum()}; "
              f"null draws' cell count: mean {cnt.mean():.0f}, p95 {np.percentile(cnt,95):.0f}, "
              f"max {cnt.max()}  (only {(cnt>0).mean()*100:.0f}% of draws have any)")

    print("=" * 100)
    print("EFFECTIVE INDEPENDENCE: how many distinct cells are behind the survivors?")
    for grid in ("res",):
        j = np.minimum(dA[grid], dB[grid]).reshape(NP, -1)
        thr = np.percentile(j.max(1), 95)
        obsA = np.stack([S["A"][f"auc_{grid}_{t}"] for t in TS], -1)
        obsB = np.stack([S["B"][f"auc_{grid}_{t}"] for t in TS], -1)
        o = np.where(np.sign(obsA - .5) == np.sign(obsB - .5),
                     np.minimum(np.maximum(obsA, 1 - obsA), np.maximum(obsB, 1 - obsB)), 0.0)
        idx = np.argwhere(o >= thr)
        rec = []
        for i, jt in idx:
            (ls, lp, lo), (ds, dp, do), (ts_, tp, to) = A.CFGS[i]
            rec.append(dict(t=TS[jt], L=f"{ls}/{lp}/{lo}", Dn=f"{ds}/{dp}/{do}",
                            T=f"{ts_}/{tp}/{to}"))
        R = pd.DataFrame(rec)
        print(R.groupby(["t", "T"]).size().to_string())
        print()
        print(R.groupby(["t", "L"]).size().to_string())

    print("=" * 100)
    print("LEAD-TIME CHECK: is the t=25 survivor the baseline's t=30 information arriving early?")
    for tag in ("A", "B"):
        c, y, g = D[tag]
        i = A.CFG_IDX[CAND["tokminALL/Lallmean"]]
        for lab, (a1, j1, a2, j2) in {
            "cand@25 vs base@30": (i, 1, BIDX, 2),
            "cand@25 vs base@25": (i, 1, BIDX, 1),
            "cand@30 vs base@30": (i, 2, BIDX, 2),
        }.items():
            rr, wt = [], []
            for gg in np.unique(g):
                m = g == gg
                if y[m].sum() == 0 or (1 - y[m]).sum() == 0:
                    continue
                u = rankdata(c[m, a1, j1])
                v = rankdata(c[m, a2, j2])
                rr.append(np.corrcoef(u, v)[0, 1])
                wt.append(m.sum())
            print(f"  {tag} {lab}: within-group rank rho = "
                  f"{np.average(rr, weights=wt):+.3f}")


if __name__ == "__main__":
    main()
