"""Extra blocks: joint cross-corpus clearing test, structured breakdowns, redundancy check."""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

MD = "/tmp/moe_agg_divtime.md"
TS = (20, 25, 30)
BASE = "hell|prev|mean|W8"
OUT = []


def det(a):
    return np.maximum(a, 1 - a)


def main():
    df = pd.read_csv("/tmp/moe_agg_divtime.csv")
    df["cell"] = df["divergence"] + "|" + df["temporal"] + "|W" + df["window"].astype(str)
    df["det"] = det(df["auc"]); df["det_resid"] = det(df["auc_resid"])
    G = {t: pickle.load(open(f"/tmp/moe_agg_grid_{t}.pkl", "rb")) for t in ("A", "B")}
    keys = G["A"]["keys"]
    cellnames = np.array([f"{a}|{b}|W{c}" for a, b, c in keys])
    nc = len(cellnames)

    obs, nullD = {}, {}
    for tag in ("A", "B"):
        for kind, col in (("raw", "auc"), ("resid", "auc_resid")):
            o = np.empty((nc, len(TS)))
            for j, t in enumerate(TS):
                sub = df[(df.corpus == tag) & (df.t == t)].set_index("cell")
                o[:, j] = sub.loc[cellnames, col].values
            obs[(tag, kind)] = o
            nullD[(tag, kind)] = np.load(f"/tmp/moe_agg_null_{tag}_{kind}.npy").astype(np.float64)

    # ---------------- A. joint cross-corpus clearing test -------------------------------
    lines = ["\n## 9. Joint cross-corpus test: how many cells clear in BOTH corpora, "
             "against a paired null?\n",
             "A cell/t 'clears' in a corpus if its detection AUC exceeds that same cell's own "
             "per-cell 95th-percentile permutation threshold (so the per-cell difficulty is "
             "absorbed). It **counts** only if it clears in both corpora *and* the two corpora "
             "agree on the sign of the effect. Under the null, permutation draw i of corpus A is "
             "paired with draw i of corpus B (the two corpora are independent experiments, so any "
             "pairing is valid), giving the null distribution of the joint count.\n"]
    joint_rows = []
    for kind in ("raw", "resid"):
        thr = {tag: np.percentile(det(nullD[(tag, kind)]), 95, axis=0) for tag in ("A", "B")}
        clA = det(obs[("A", kind)]) > thr["A"]
        clB = det(obs[("B", kind)]) > thr["B"]
        sgn = np.sign(obs[("A", kind)] - .5) == np.sign(obs[("B", kind)] - .5)
        obs_ct = int((clA & clB & sgn).sum())
        nA, nB = nullD[("A", kind)], nullD[("B", kind)]
        ncA = det(nA) > thr["A"][None]
        ncB = det(nB) > thr["B"][None]
        nsg = np.sign(nA - .5) == np.sign(nB - .5)
        null_ct = (ncA & ncB & nsg).sum(axis=(1, 2))
        p = (1 + (null_ct >= obs_ct).sum()) / (1 + len(null_ct))
        lines.append(f"* **{kind}** grid ({nc} cells x 3 t = {nc*3} tests): observed joint "
                     f"count **{obs_ct}**; null mean {null_ct.mean():.1f}, p95 "
                     f"{np.percentile(null_ct,95):.0f}, max of 200 {null_ct.max()}; "
                     f"**p = {p:.4f}**")
        joint_rows.append(dict(grid=kind, obs_count=obs_ct, null_mean=null_ct.mean(),
                               null_p95=np.percentile(null_ct, 95), null_max=int(null_ct.max()), p=p))
        if kind == "resid" and obs_ct:
            ii, jj = np.where(clA & clB & sgn)
            lines.append("\n  Cells clearing in both (residual grid):\n\n```")
            rr = sorted(zip(np.minimum(det(obs[("A", kind)])[ii, jj], det(obs[("B", kind)])[ii, jj]),
                            cellnames[ii], [TS[j] for j in jj],
                            det(obs[("A", kind)])[ii, jj], det(obs[("B", kind)])[ii, jj]),
                        reverse=True)
            for mn, c, t, a, b in rr[:40]:
                lines.append(f"    t={t}  {c:34s} min={mn:.4f}  A={a:.4f} B={b:.4f}")
            lines.append("```")
    pd.DataFrame(joint_rows).to_csv("/tmp/moe_agg_joint_test.csv", index=False)
    OUT.append("\n".join(lines) + "\n")

    # ---------------- B. structured breakdown of the divergence axis --------------------
    def fam(d):
        if d.startswith("d") and ("_abs" in d or "_sgn" in d):
            return "C:scalar-change"
        return "A:pairwise|prev" if d.endswith("|prev") else "B:pairwise|runmean"
    df["fam"] = df["divergence"].map(fam)
    lines = ["\n## 10. Structured breakdown of the divergence axis\n",
             "The 26 divergences are not 26 independent choices. They fall into three families:",
             "**A** the 11 pairwise metrics against the previous chunk, **B** the same 11 against "
             "the branch's own running mean, **C** the 4 scalar-change measures (entropy / top-1 "
             "mass, absolute and signed).\n\n```"]
    for t in TS:
        for tag in ("A", "B"):
            s = df[(df.t == t) & (df.corpus == tag)]
            g = s.groupby("fam")[["det", "det_resid"]].agg(["mean", "max"])
            lines.append(f"\n  t={t} corpus {tag}")
            lines.append("  " + g.round(3).to_string().replace("\n", "\n  "))
    lines.append("```")
    lines.append("\n### Spread *within* family A (the 11 f-divergences against the previous "
                 "chunk), t=30, best temporal spec per metric\n\n```")
    s = df[(df.t == 30) & (df.fam == "A:pairwise|prev")]
    pv = s.groupby(["divergence", "corpus"])[["det", "det_resid"]].max().unstack()
    lines.append(pv.round(4).to_string())
    lines.append("```")
    OUT.append("\n".join(lines) + "\n")

    # ---------------- C. variance decomposition restricted to family A ------------------
    vd = []
    for t in TS:
        for tag in ("A", "B"):
            for col in ("det", "det_resid"):
                for scope, sel in (("all_26_divergences", df.fam.notna()),
                                   ("only_pairwise_prev_11", df.fam == "A:pairwise|prev")):
                    sub = df[(df.t == t) & (df.corpus == tag) & sel]
                    M = sub.pivot_table(index="divergence", columns=["temporal", "window"], values=col)
                    v = M.values; gm = v.mean()
                    sst = ((v - gm) ** 2).sum()
                    ssd = v.shape[1] * ((v.mean(1) - gm) ** 2).sum()
                    sst_ = v.shape[0] * ((v.mean(0) - gm) ** 2).sum()
                    vd.append(dict(t=t, corpus=tag, grid=col, scope=scope,
                                   frac_div=ssd / sst, frac_temporal=sst_ / sst,
                                   frac_inter=1 - (ssd + sst_) / sst,
                                   range_div=np.ptp(v.mean(1)), range_temporal=np.ptp(v.mean(0))))
    vdf = pd.DataFrame(vd)
    vdf.to_csv("/tmp/moe_agg_variance_decomp.csv", index=False)
    OUT.append("\n## 11. Divergence vs temporal: variance decomposition, with and without the "
               "near-degenerate metric family\n\n```\n" + vdf.round(4).to_string(index=False) +
               "\n```\n")

    # ---------------- D. redundancy: correlation of each cell with the baseline ---------
    lines = ["\n## 12. Why the raw grid is not 1794 different quantities: rank correlation "
             "with the baseline (t=30, within group, pooled)\n\n```"]
    corr_rows = []
    for tag in ("A", "B"):
        st = G[tag]
        R = st["per_t"][30]["R"]
        bi = list(cellnames).index(BASE)
        Rz = R - R.mean(1, keepdims=True)
        Rz /= np.sqrt((Rz ** 2).sum(1, keepdims=True)) + 1e-12
        rho = np.abs(Rz @ Rz[bi])
        for c, r in zip(cellnames, rho):
            corr_rows.append(dict(corpus=tag, cell=c, abs_rho_to_baseline=r))
    cdf = pd.DataFrame(corr_rows)
    cdf.to_csv("/tmp/moe_agg_baseline_corr.csv", index=False)
    cdf["fam"] = cdf.cell.str.rsplit("|", n=2).str[0].map(fam)
    lines.append(cdf.groupby(["corpus", "fam"])["abs_rho_to_baseline"]
                 .describe()[["mean", "50%", "max"]].round(3).to_string())
    lines.append("\n  |rho| >= 0.9 with the baseline: " + ", ".join(
        f"{tag} {int((cdf[(cdf.corpus==tag)].abs_rho_to_baseline>=.9).sum())}/{nc}"
        for tag in ("A", "B")))
    lines.append("  |rho| >= 0.7 with the baseline: " + ", ".join(
        f"{tag} {int((cdf[(cdf.corpus==tag)].abs_rho_to_baseline>=.7).sum())}/{nc}"
        for tag in ("A", "B")))
    lines.append("```")
    OUT.append("\n".join(lines) + "\n")

    # ---------------- E. stability of the residual leaders across t ---------------------
    lines = ["\n## 13. Do the residual leaders survive at other t? (the previous round's "
             "failure mode)\n\n```"]
    R30 = df[(df.t == 30)].pivot_table(index="cell", columns="corpus", values="auc_resid")
    R30["mn"] = np.minimum(det(R30.A), det(R30.B))
    R30 = R30[np.sign(R30.A - .5) == np.sign(R30.B - .5)]
    top = R30.sort_values("mn", ascending=False).head(12).index
    for c in top:
        row = []
        for t in TS:
            v = df[(df.cell == c) & (df.t == t)].set_index("corpus")
            row.append(f"t{t}: A={v.loc['A','det_resid']:.3f} B={v.loc['B','det_resid']:.3f}")
        lines.append(f"  {c:34s} " + "  ".join(row))
    lines.append("```")
    OUT.append("\n".join(lines) + "\n")

    with open(MD, "a") as fh:
        fh.write("\n".join(OUT) + "\n")
    print("appended to", MD)


if __name__ == "__main__":
    main()
