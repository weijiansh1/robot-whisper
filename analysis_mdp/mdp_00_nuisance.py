"""(1) MANDATORY NUISANCE CHECK + baseline reproduction + length sentinel.

Run first. If the learned routing states are largely a clock (t) or a length code (T),
the detection angle is dead on arrival -- that is how the 2026-08-28 clustering died.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as ami

import mdp_lib as L

FEATS = ("cell1280", "mean32")


def baseline_scores(tag):
    """Hellinger to the previous chunk, per (layer,token) then averaged, 8-step window mean."""
    P, valid, meta = L.load(tag)
    d = L.agglib.divergence_series(P, valid, "hell|prev")
    out = {}
    for t in L.TS:
        out[t] = L.agglib.aggregate(d, t, "mean", 8)
    return out, meta


def main():
    L.note("# MoE routing Markov/MDP model -- running log\n")
    L.note("## 0. Corpus shape, length sentinel, baseline reproduction\n")

    store = {}
    for tag in ("A", "B"):
        P, valid, meta = L.load(tag)
        y = meta["success"].values.astype(bool)
        g = meta["group"].values
        T = meta["T"].values
        cap = T.max()
        atcap = (T == cap)
        B = len(meta)
        L.note(f"**Corpus {tag}**: {B} branches, Tmax={T.max()}, Tmin={T.min()}, "
               f"{y.sum()} success / {(~y).sum()} fail, {meta['group'].nunique()} groups, "
               f"{valid.sum()} chunks.")
        tp = int((atcap & ~y).sum()); fp = int((atcap & y).sum()); fn = int((~atcap & ~y).sum())
        L.note(f"  sentinel `T == cap({cap})`: fires {atcap.sum()}, "
               f"precision(fail)={tp/max(tp+fp,1):.4f}, recall={tp/max(tp+fn,1):.4f}")
        L.emit(dict(corpus=tag, block="sentinel", metric="cap_bit_precision", value=tp / max(tp + fp, 1)))
        L.emit(dict(corpus=tag, block="sentinel", metric="cap_bit_recall", value=tp / max(tp + fn, 1)))

        base, _ = baseline_scores(tag)
        store[tag] = dict(meta=meta, valid=valid, y=y, g=g, T=T, base=base, cap=cap)

        for t in L.TS:
            alive = T > t
            # sentinel AUCs on the same risk set
            a_cap, d_cap, n = L.auc_pair(-atcap[alive].astype(float), y[alive], g[alive])
            a_T, d_T, _ = L.auc_pair(-T[alive].astype(float), y[alive], g[alive])
            a_b, d_b, _ = L.auc_pair(base[t][alive], y[alive], g[alive])
            L.note(f"  t={t}  n_alive={alive.sum()}  npairs={n}  |  "
                   f"BASELINE hell|prev/mean8 det_auc={d_b:.4f} (auc_succ={a_b:.4f})  |  "
                   f"sentinel cap-bit det_auc={d_cap:.4f}  |  oracle length T det_auc={d_T:.4f}")
            for nm, (a, dd) in dict(baseline_hell_mean8=(a_b, d_b), sentinel_capbit=(a_cap, d_cap),
                                    oracle_length=(a_T, d_T)).items():
                L.emit(dict(corpus=tag, block="baseline_sentinel", metric=nm, t=t,
                            auc_succ=a, det_auc=dd, npairs=n, n_alive=int(alive.sum())))
        L.flush()

    L.note("\n> Reproduction target: A 0.795 / B 0.759 at t=30.\n")

    # ------------------------------------------------------------------ NMI
    L.note("## 1. NUISANCE CHECK -- is the routing state a clock or a length code?\n")
    L.note("Label-free global fit (PCA20 + k-means on every chunk). NMI over all valid chunks "
           "between the state assignment and: t (absolute query index), T (branch length), "
           "phase t/T (10 bins), remaining T-t, and the at-cap bit.\n")
    L.note("| corpus | feat | K | NMI(state,t) | NMI(state,T) | NMI(state,phase) | "
           "NMI(state,T-t) | NMI(state,atcap) | AMI(state,T) |")
    L.note("|---|---|---|---|---|---|---|---|---|")

    for tag in ("A", "B"):
        st = store[tag]
        P, valid, meta = L.load(tag)
        T = st["T"]; cap = st["cap"]
        B, Tmax = valid.shape
        tt = np.tile(np.arange(Tmax), (B, 1))
        TT = np.tile(T[:, None], (1, Tmax))
        phase = np.minimum((tt / np.maximum(TT - 1, 1) * 10).astype(int), 9)
        rem = TT - tt
        ac = np.tile((T == cap)[:, None], (1, Tmax))
        v = valid
        for feat in FEATS:
            F = L.features(P, valid, feat)
            for K in L.KS:
                S, _, _ = L.fit_states(F, valid, np.arange(B), K)
                s = S[v]
                r = dict(corpus=tag, block="nmi", feat=feat, K=K,
                         nmi_t=nmi(s, tt[v]), nmi_T=nmi(s, TT[v]), nmi_phase=nmi(s, phase[v]),
                         nmi_rem=nmi(s, rem[v]), nmi_atcap=nmi(s, ac[v]), ami_T=ami(s, TT[v]))
                L.emit(r)
                L.note(f"| {tag} | {feat} | {K} | {r['nmi_t']:.3f} | {r['nmi_T']:.3f} | "
                       f"{r['nmi_phase']:.3f} | {r['nmi_rem']:.3f} | {r['nmi_atcap']:.3f} | "
                       f"{r['ami_T']:.3f} |")
            del F
        L.flush()
    L.note("")


if __name__ == "__main__":
    main()
