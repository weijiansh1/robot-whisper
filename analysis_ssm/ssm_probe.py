"""Sentinels 2-3 and the two falsifiable predictions.

Usage: python ssm_probe.py <A|B>

Sentinel 2 (clock)      : mutual information between the filtered latent and t, T, t/T.
Sentinel 3 (redundancy) : within-group Spearman between the innovation score and the baseline.
Prediction 1            : innovation is large when ||A_k|| is large and routing change is small.
Prediction 2            : the filter's gain concentrates where the best fixed window is not 8.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_ssm")
import ssm_lib as L
from ssm_run import MAIN

TQ = 30
NBIN = 16


def _disc(x, nb):
    q = np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])
    return np.searchsorted(q, x)


def mi_bits(a, b):
    """Mutual information (bits) between two integer-labelled discrete variables."""
    ca = pd.crosstab(a, b).values.astype(float)
    p = ca / ca.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def entropy_bits(a):
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def auc_one(s, y):
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = L._avg_rank(s)
    return (r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    corpus = sys.argv[1]
    r, d = MAIN
    C = np.load(f"{L.CACHE}/prep/{corpus}_real_common.npz")
    base, anorm, valid, y, grp = C["base"], C["anorm"], C["valid"], C["y"], C["grp"]
    S = np.load(f"{L.CACHE}/series_{corpus}_real.npz")
    zf = S["zf_main"]
    innov = S[f"innov__{r}_{d}"]
    nis = S[f"nis__{r}_{d}"]
    Bn, Tmax = valid.shape
    Tb = valid.sum(1)
    kk, bb = np.meshgrid(np.arange(Tmax), np.arange(Bn))
    m = valid & (kk <= TQ)

    # ---------------- sentinel 2: clock ----------------
    kvec, Tvec = kk[m], Tb[bb[m]]
    frac = kvec / Tvec
    ck, cT, cf = _disc(kvec, NBIN), _disc(Tvec, 8), _disc(frac, NBIN)
    Hk, HT, Hf = entropy_bits(ck), entropy_bits(cT), entropy_bits(cf)
    per_dim = []
    for j in range(d):
        zj = _disc(zf[m][:, j], NBIN)
        per_dim.append((mi_bits(zj, ck), mi_bits(zj, cT), mi_bits(zj, cf)))
    per_dim = np.array(per_dim)
    # joint latent state: k-means quantisation of the whole latent vector into 32 states
    from sklearn.cluster import KMeans
    Z = zf[m]
    km = KMeans(n_clusters=32, n_init=4, random_state=L.SEED).fit(
        Z[np.random.default_rng(0).permutation(len(Z))[:20000]])
    lab = km.predict(Z)
    Hz = entropy_bits(lab)
    joint = (mi_bits(lab, ck), mi_bits(lab, cT), mi_bits(lab, cf))
    L.note(f"\n### Sentinel 2 (clock) — {corpus}, latent r={r} d={d}, chunks k<=30")
    L.note(f"H(t)={Hk:.3f} H(T)={HT:.3f} H(t/T)={Hf:.3f} H(latent32)={Hz:.3f} bits")
    L.note(f"per-dim MI bits  vs t: max {per_dim[:,0].max():.3f} mean {per_dim[:,0].mean():.3f}"
           f" | vs T: max {per_dim[:,1].max():.3f} | vs t/T: max {per_dim[:,2].max():.3f}")
    L.note(f"32-state latent MI bits  t {joint[0]:.3f} ({joint[0]/Hk:.3f} of H(t)) | "
           f"T {joint[1]:.3f} ({joint[1]/HT:.3f}) | t/T {joint[2]:.3f} ({joint[2]/Hf:.3f})")
    L.emit(corpus=corpus, stat="clock", metric="mi", Ht=Hk, HT=HT, Hf=Hf, Hz=Hz,
           mi_t=joint[0], mi_T=joint[1], mi_frac=joint[2],
           nmi_t=joint[0] / Hk, nmi_T=joint[1] / HT, nmi_frac=joint[2] / Hf,
           perdim_mi_t_max=per_dim[:, 0].max(), perdim_mi_T_max=per_dim[:, 1].max())

    # score-vs-episode-length diagnostic (T is a future quantity)
    for nm, v in (("nis_w8", nis[:, TQ - 7:TQ + 1].mean(1)),
                  ("base_w8", base[:, TQ - 7:TQ + 1].mean(1))):
        rho, _ = L.within_group_spearman(v, Tb.astype(float), grp)
        L.note(f"within-group Spearman({nm} @t30, episode length T) = {rho:+.3f}")
        L.emit(corpus=corpus, stat=nm, metric="vs_T", rho=rho)

    # ---------------- sentinel 3: redundancy ----------------
    bs = base[:, TQ - 7:TQ + 1].mean(1)
    L.note(f"\n### Sentinel 3 (redundancy vs baseline) — {corpus} @ t=30")
    for nm, v in (("innov_w8", innov[:, TQ - 7:TQ + 1].mean(1)),
                  ("nis_w8", nis[:, TQ - 7:TQ + 1].mean(1)),
                  ("nis_w12", nis[:, TQ - 11:TQ + 1].mean(1)),
                  ("pcaz_w8", S[f"pcaz__{r}_{d}"][:, TQ - 7:TQ + 1].mean(1)),
                  ("dzf_w8", S[f"dzf__{r}_{d}"][:, TQ - 7:TQ + 1].mean(1))):
        rho, rv = L.within_group_spearman(v, bs, grp)
        L.note(f"  {nm:9s} rho={rho:+.3f}  range [{np.nanmin(rv):+.3f},{np.nanmax(rv):+.3f}]"
               f"  {'REDUNDANT (|rho|>0.9)' if abs(rho) > 0.9 else ''}")
        L.emit(corpus=corpus, stat=nm, metric="redundancy", rho=rho,
               rho_min=float(np.nanmin(rv)), rho_max=float(np.nanmax(rv)))

    # ---------------- prediction 1 ----------------
    # within-branch standardisation removes branch-level scale; k in 1..30
    m1 = valid & (kk >= 1) & (kk <= TQ)

    def wz(a):
        x = np.where(m1, a, np.nan)
        mu = np.nanmean(x, 1, keepdims=True)
        sd = np.nanstd(x, 1, keepdims=True) + 1e-12
        return ((x - mu) / sd)[m1]

    ei, an, bd = wz(innov), wz(anorm), wz(np.nan_to_num(base))
    en = wz(nis)
    D = np.column_stack([an, bd, an * bd, np.ones_like(an)])
    L.note(f"\n### Prediction 1 (innovation ~ large ||A_k||, small routing change) — {corpus}")
    for nm, tgt in (("innov", ei), ("nis", en)):
        beta, *_ = np.linalg.lstsq(D, tgt, rcond=None)
        # partial Spearman of target with ||A|| given the routing-change baseline
        ra, rb, rt = L._avg_rank(an), L._avg_rank(bd), L._avg_rank(tgt)
        ra, rb, rt = ra - ra.mean(), rb - rb.mean(), rt - rt.mean()
        res_a = ra - rb * (ra @ rb) / (rb @ rb)
        res_t = rt - rb * (rt @ rb) / (rb @ rb)
        pr = float((res_a @ res_t) / np.sqrt((res_a @ res_a) * (res_t @ res_t)))
        raw = float((ra @ rt) / np.sqrt((ra @ ra) * (rt @ rt)))
        L.note(f"  {nm}: b(||A||)={beta[0]:+.4f} b(route change)={beta[1]:+.4f} "
               f"b(interaction)={beta[2]:+.4f} | rho(.,||A||)={raw:+.3f} "
               f"partial rho given route change = {pr:+.3f}  (n={len(an)})")
        L.emit(corpus=corpus, stat=nm, metric="pred1", b_anorm=beta[0], b_route=beta[1],
               b_inter=beta[2], rho_anorm=raw, partial_rho=pr, n=len(an))
        # quadrant means (within-branch medians)
        hiA, loD = an > 0, bd < 0
        for qa, qd, lbl in ((hiA, loD, "hiA_loD"), (hiA, ~loD, "hiA_hiD"),
                            (~hiA, loD, "loA_loD"), (~hiA, ~loD, "loA_hiD")):
            L.emit(corpus=corpus, stat=nm, metric="pred1_quad", quad=lbl,
                   mean=float(tgt[qa & qd].mean()), n=int((qa & qd).sum()))
        qm = {lbl: float(tgt[qa & qd].mean()) for qa, qd, lbl in
              ((hiA, loD, "hiA_loD"), (hiA, ~loD, "hiA_hiD"),
               (~hiA, loD, "loA_loD"), (~hiA, ~loD, "loA_hiD"))}
        L.note(f"    quadrant means (z): {', '.join(f'{k}={v:+.3f}' for k, v in qm.items())}")

    # ---------------- prediction 2 ----------------
    L.note(f"\n### Prediction 2 (gain concentrates where best window != 8) — {corpus} @ t=30")
    rows = []
    for g in np.unique(grp):
        gm = grp == g
        yg = y[gm]
        if yg.sum() == 0 or (~yg).sum() == 0:
            continue
        aucs = {W: abs(auc_one(base[gm, TQ - W + 1:TQ + 1].mean(1), yg) - 0.5)
                for W in (3, 5, 8, 12, 20)}
        Wstar = max(aucs, key=aucs.get)
        db = abs(auc_one(base[gm, TQ - 7:TQ + 1].mean(1), yg) - 0.5) + 0.5
        dn = abs(auc_one(nis[gm, TQ - 7:TQ + 1].mean(1), yg) - 0.5) + 0.5
        dn12 = abs(auc_one(nis[gm, TQ - 11:TQ + 1].mean(1), yg) - 0.5) + 0.5
        rows.append(dict(group=int(g), n=int(gm.sum()), nsucc=int(yg.sum()), Wstar=Wstar,
                         base_det=db, nis8_det=dn, nis12_det=dn12,
                         gain8=dn - db, gain12=dn12 - db))
    df = pd.DataFrame(rows)
    df.to_csv(f"{L.OUT}/pred2_{corpus}.csv", index=False)
    for _, rr in df.iterrows():
        L.note(f"  g{int(rr['group']):2d} n={int(rr['n'])} succ={int(rr['nsucc'])} "
               f"W*={int(rr['Wstar']):2d} base={rr['base_det']:.3f} "
               f"nisW8={rr['nis8_det']:.3f} gain={rr['gain8']:+.3f}")
    for col in ("gain8", "gain12"):
        a = df.loc[df["Wstar"] == 8, col]
        b = df.loc[df["Wstar"] != 8, col]
        L.note(f"  {col}: W*=8 groups mean {a.mean():+.4f} (n={len(a)}) | "
               f"W*!=8 groups mean {b.mean():+.4f} (n={len(b)})")
        L.emit(corpus=corpus, stat=col, metric="pred2", mean_W8=float(a.mean()) if len(a) else np.nan,
               n_W8=len(a), mean_notW8=float(b.mean()) if len(b) else np.nan, n_notW8=len(b))
    L.flush("moe_ssm_probe.csv")


if __name__ == "__main__":
    main()
