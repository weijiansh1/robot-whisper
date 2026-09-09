"""(4)(5)(6) Family-wise null, cross-corpus replication, K-stability, verdict inputs."""
import numpy as np
import pandas as pd

import mdp_lib as L

REAL = ("cell1280", "mean32")     # `clock` and `shuf` are controls, not model cells


def load_null(tag):
    z = np.load(f"{L.OUT}/null_{tag}.npz", allow_pickle=True)
    c = pd.DataFrame(z["cells"], columns=["feat", "K", "t", "variant"])
    c["K"] = c["K"].astype(int); c["t"] = c["t"].astype(int)
    return z["raw"], z["res"], z["sgn"], c


def main():
    L.note("\n## 3. DETECTION -- absorption probability vs the scalar baseline\n")
    D = {}
    for tag in ("A", "B"):
        raw, res, sgn, c = load_null(tag)
        D[tag] = dict(raw=raw, res=res, sgn=sgn, c=c)

    # ---------------------------------------------------------- controls
    L.note("### 3a. Two control state spaces\n")
    L.note("| corpus | control | max raw det AUC over its 39x5 cells | max residual det AUC | "
           "reading |")
    L.note("|---|---|---|---|---|")
    for tag in ("A", "B"):
        c = D[tag]["c"]
        for ctl, msg in (("clock", "pure clock, zero routing"),
                         ("shuf", "ACAUSAL within-branch shuffle -- leakage sentinel")):
            m = (c.feat == ctl).values
            L.note(f"| {tag} | {ctl} | {np.nanmax(D[tag]['raw'][0][m]):.3f} | "
                   f"{np.nanmax(D[tag]['res'][0][m]):.3f} | {msg} |")
    L.note("\nThe clock control is *degenerate by construction*: the protocol fixes the query "
           "index t, so every branch alive at t sits in the same clock state and the score is "
           "constant. That is the point -- the fixed-t protocol is already immune to the clock "
           "component of the state, so the NMI(state,t) of 0.32-0.54 cannot be what produces "
           "any AUC below.\n")
    L.note("The acausal shuffle is NOT a control, it is a leakage sentinel: permuting a branch's "
           "chunk order mixes post-t states into the pre-t window and reaches det AUC 0.93 (A) / "
           "0.96 (B). Any pipeline that touches the whole episode gets ~0.95 for free. All model "
           "cells below are strictly causal (window [t-7, t]).\n")

    # ---------------------------------------------------------- family-wise
    L.note("### 3b. Family-wise null over the whole model grid (390 cells: 2 featurisations x "
           "5 K x 3 t x 13 variants), 200 shared within-group branch-label permutations\n")
    L.note("| corpus | statistic | observed | null mean | null q95 | null max | family-wise p |")
    L.note("|---|---|---|---|---|---|---|")
    fam = {}
    for tag in ("A", "B"):
        c = D[tag]["c"]
        m = c.feat.isin(REAL).values
        for nm, arr in (("max RAW det AUC", D[tag]["raw"]), ("max RESIDUAL det AUC", D[tag]["res"])):
            v = np.nanmax(arr[:, m], 1)
            p = (1 + (v[1:] >= v[0]).sum()) / len(v)
            fam[(tag, nm)] = p
            L.note(f"| {tag} | {nm} | {v[0]:.3f} | {v[1:].mean():.3f} | "
                   f"{np.quantile(v[1:],0.95):.3f} | {v[1:].max():.3f} | {p:.4f} |")
            L.emit(dict(corpus=tag, block="familywise", stat=nm, observed=v[0],
                        null_mean=v[1:].mean(), null_q95=np.quantile(v[1:], 0.95),
                        null_max=v[1:].max(), p=p))
        # how many cells clear their own per-cell 5% level
        for nm, arr in (("raw", D[tag]["raw"]), ("res", D[tag]["res"])):
            a = arr[:, m]
            pc = (1 + (a[1:] >= a[0]).sum(0)) / a.shape[0]
            L.note(f"| {tag} | cells with per-cell p_{nm} < 0.05 | "
                   f"{(pc<0.05).sum()}/{m.sum()} = {(pc<0.05).mean():.3f} | 0.05 | | | |")
            L.emit(dict(corpus=tag, block="familywise", stat=f"frac_cells_p{nm}_lt05",
                        observed=float((pc < 0.05).mean())))
    L.flush()

    # ---------------------------------------------------------- cross-corpus
    L.note("\n### 3c. Cross-corpus replication -- the decisive test\n")
    cA, cB = D["A"]["c"], D["B"]["c"]
    key = ["feat", "K", "t", "variant"]
    iA = cA.reset_index().set_index(key)["index"]
    iB = cB.reset_index().set_index(key)["index"]
    common = iA.index.intersection(iB.index)
    common = [k for k in common if k[0] in REAL]
    ja = iA.loc[common].values
    jb = iB.loc[common].values
    sA = D["A"]["sgn"][:, ja] - 0.5
    sB = D["B"]["sgn"][:, jb] - 0.5
    agree = (np.sign(sA) == np.sign(sB)).mean(1)
    p_agree = (1 + (agree[1:] >= agree[0]).sum()) / len(agree)
    L.note(f"Residual sign agreement across corpora, over {len(common)} matched cells: "
           f"**observed {agree[0]:.3f}**, null mean {agree[1:].mean():.3f}, "
           f"null q95 {np.quantile(agree[1:],0.95):.3f}, permutation p = {p_agree:.3f}.\n")
    L.emit(dict(corpus="AB", block="crosscorpus", stat="residual_sign_agreement",
                observed=agree[0], null_mean=agree[1:].mean(),
                null_q95=np.quantile(agree[1:], 0.95), p=p_agree))
    # raw sign agreement for reference
    rA = D["A"]["raw"][:, ja]; rB = D["B"]["raw"][:, jb]
    L.note("Per-t breakdown (residual sign agreement; 0.5 = chance):\n")
    L.note("| t | n cells | observed agreement | null mean | null q95 | p |")
    L.note("|---|---|---|---|---|---|")
    ts = np.array([k[2] for k in common])
    for t in L.TS:
        mm = ts == t
        ag = (np.sign(sA[:, mm]) == np.sign(sB[:, mm])).mean(1)
        pp = (1 + (ag[1:] >= ag[0]).sum()) / len(ag)
        L.note(f"| {t} | {mm.sum()} | {ag[0]:.3f} | {ag[1:].mean():.3f} | "
               f"{np.quantile(ag[1:],0.95):.3f} | {pp:.3f} |")
        L.emit(dict(corpus="AB", block="crosscorpus", stat=f"residual_sign_agreement_t{t}",
                    observed=ag[0], null_mean=ag[1:].mean(), p=pp))

    # joint clearance: p<0.05 in BOTH corpora with matching residual sign
    pcA = (1 + (D["A"]["res"][1:, ja] >= D["A"]["res"][0, ja]).sum(0)) / 201
    pcB = (1 + (D["B"]["res"][1:, jb] >= D["B"]["res"][0, jb]).sum(0)) / 201
    both = (pcA < 0.05) & (pcB < 0.05) & (np.sign(sA[0]) == np.sign(sB[0]))
    exp = len(common) * (pcA < 0.05).mean() * (pcB < 0.05).mean() * 0.5
    L.note(f"\nCells clearing residual p<0.05 in BOTH corpora with matching sign: "
           f"**{both.sum()}** of {len(common)}. Expected if the two corpora were independent "
           f"draws with the same per-corpus clearance rates: **{exp:.1f}**.\n")
    L.emit(dict(corpus="AB", block="crosscorpus", stat="joint_clearance",
                observed=int(both.sum()), null_mean=exp))
    ck = pd.DataFrame(common, columns=key)
    ck["res_A"] = D["A"]["res"][0, ja]; ck["res_B"] = D["B"]["res"][0, jb]
    ck["raw_A"] = D["A"]["raw"][0, ja]; ck["raw_B"] = D["B"]["raw"][0, jb]
    ck["p_res_A"] = pcA; ck["p_res_B"] = pcB; ck["both"] = both
    ck.to_csv(f"{L.OUT}/crosscorpus_cells.csv", index=False)
    L.note("Variants supplying those joint cells:\n")
    vc = ck[both].variant.value_counts()
    L.note("| variant | joint cells | share of grid |")
    L.note("|---|---|---|")
    for v, n in vc.items():
        L.note(f"| {v} | {n} | 1/13 |")

    # ---------------------------------------------------------- per-variant view
    L.note("\n### 3d. Per-variant summary at t=30 (the only t where the baseline is strong)\n")
    L.note("| variant | A raw (best K) | A res | A p_res | B raw (best K) | B res | B p_res | "
           "beats baseline in both? |")
    L.note("|---|---|---|---|---|---|---|---|")
    rows = []
    for v in sorted(set(k[3] for k in common)):
        out = {}
        for tag, jj, pc in (("A", ja, pcA), ("B", jb, pcB)):
            sel = np.array([(k[3] == v and k[2] == 30) for k in common])
            r = D[tag]["raw"][0, (ja if tag == "A" else jb)][sel]
            rs = D[tag]["res"][0, (ja if tag == "A" else jb)][sel]
            p = pc[sel]
            i = int(np.nanargmax(r))
            out[tag] = (r[i], rs[i], p[i], [common[j] for j in np.where(sel)[0]][i][1])
        bA = 0.7988 if True else 0
        beat = (out["A"][0] > 0.7988) and (out["B"][0] > 0.7510)
        L.note(f"| {v} | {out['A'][0]:.3f} (K={out['A'][3]}) | {out['A'][1]:.3f} | "
               f"{out['A'][2]:.3f} | {out['B'][0]:.3f} (K={out['B'][3]}) | {out['B'][1]:.3f} | "
               f"{out['B'][2]:.3f} | {'YES' if beat else 'no'} |")
        rows.append(dict(variant=v, A_raw=out["A"][0], A_res=out["A"][1], A_p=out["A"][2],
                         B_raw=out["B"][0], B_res=out["B"][1], B_p=out["B"][2], beats_both=beat))
    pd.DataFrame(rows).to_csv(f"{L.OUT}/per_variant_t30.csv", index=False)

    # ---------------------------------------------------------- K stability
    L.note("\n### 3e. Does K matter?  det AUC of the two leading variants vs K, t=30, cell1280\n")
    L.note("| corpus | variant | K=4 | K=8 | K=16 | K=32 | K=64 |")
    L.note("|---|---|---|---|---|---|---|")
    for tag in ("A", "B"):
        c = D[tag]["c"]
        for v in ("absorb_w8", "emp_pool_w8"):
            vals = []
            for K in L.KS:
                m = ((c.feat == "cell1280") & (c.K == K) & (c.t == 30) & (c.variant == v)).values
                vals.append(D[tag]["raw"][0, m][0])
            L.note(f"| {tag} | {v} | " + " | ".join(f"{x:.3f}" for x in vals) + " |")
            L.emit(dict(corpus=tag, block="kstability", variant=v,
                        **{f"K{K}": x for K, x in zip(L.KS, vals)}))
    L.flush()

    # ---------------------------------------------------------- memory order
    L.note("\n## 4. Does MEMORY help? first order vs second order vs semi-Markov "
           "(cell1280, t=30)\n")
    o = pd.read_csv(f"{L.OUT}/order_raw.csv")
    o30 = o[o.t == 30]
    L.note("| corpus | variant | order1 best | order2 best | semi best | baseline |")
    L.note("|---|---|---|---|---|---|")
    for tag in ("A", "B"):
        for v in ("absorb_w8", "emp_pool_w8"):
            x = o30[(o30.corpus == tag) & (o30.variant == v)]
            g = {m: x[x["mode"] == m].det_auc.max() for m in ("order1", "order2", "semi")}
            L.note(f"| {tag} | {v} | {g['order1']:.3f} | {g['order2']:.3f} | {g['semi']:.3f} | "
                   f"{x.base_det_auc.iloc[0]:.3f} |")
            L.emit(dict(corpus=tag, block="order", variant=v, **{k: float(vv) for k, vv in g.items()}))
    L.flush()
    L.note("")


if __name__ == "__main__":
    main()
