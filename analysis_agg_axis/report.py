"""Analysis + running markdown summary for the divergence x temporal sweep."""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

MD = "/tmp/moe_agg_divtime.md"
TS = (20, 25, 30)
BASE = ("hell|prev", "mean", 8)
BLOCKS = []


def flush():
    with open(MD, "w") as fh:
        fh.write("\n".join(BLOCKS) + "\n")


def det(a):
    return np.maximum(a, 1 - a)


def load():
    df = pd.read_csv("/tmp/moe_agg_divtime.csv")
    df["cell"] = df["divergence"] + "|" + df["temporal"] + "|W" + df["window"].astype(str)
    df["det"] = det(df["auc"])
    df["det_resid"] = det(df["auc_resid"])
    return df


def main():
    df = load()
    G = {t: pickle.load(open(f"/tmp/moe_agg_grid_{t}.pkl", "rb")) for t in ("A", "B")}
    keys = G["A"]["keys"]
    cellnames = np.array([f"{a}|{b}|W{c}" for a, b, c in keys])
    bi = keys.index(BASE)

    BLOCKS.append(f"""# Divergence function x temporal aggregation sweep (MoE routing tensor)

**Scope.** Only two reduction stages are varied here: (1) the divergence functional on the
32-dim expert axis and what each chunk is compared against, (2) the temporal aggregation
over chunks and its window width. **Everything the sibling agent owns is held fixed:**
placement = **divergence-first** (divergence evaluated per (layer, token) cell, then averaged
over the 40 cells), back-block HB layers **12-15**, action tokens **1-10**, denoise step **9**.
The two sweeps therefore compose: this grid is one column of the sibling's placement axis.

**Grid.** 26 divergences x 69 temporal specs = **1794 cells** per corpus per t; t in {{20,25,30}};
2 corpora; raw and baseline-residualised -> **10764 (cell, t, corpus) evaluations per grid**.

**Protocol.** Target = BDDL `success`. Risk set at query index t = branches with T > t; at
t<=30 that is *every* branch in both corpora (A min T=33, B min T=35), so there is no survival
conditioning anywhere in this report. AUC is computed **within group** (`worker` in A,
`init_state_id` in B) and pooled by pair count; groups that are single-class contribute
nothing (A worker 1 = 0/80; B init states 33 and 36 = 32/32, 49 = 0/32).
Contributing branches: **A {G['A']['n_pos']}S/{G['A']['n_neg']}F ({G['A']['Pairs']} pairs)**,
**B {G['B']['n_pos']}S/{G['B']['n_neg']}F ({G['B']['Pairs']} pairs)**.
No hard top-4 expert IDs are used; probabilities only.

**Sign convention.** `auc` in the CSV is the raw **directional** AUC = P(score | success >
score | failure), so cross-corpus sign flips stay visible. The **detection** AUC that compares
to the baseline is `max(auc, 1-auc)`.

**Baseline reproduction.** `hell|prev` + window mean + W=8 on the fixed slice:
**A {df[(df.corpus=='A')&(df.cell=='hell|prev|mean|W8')&(df.t==30)].det.iloc[0]:.4f} /
B {df[(df.corpus=='B')&(df.cell=='hell|prev|mean|W8')&(df.t==30)].det.iloc[0]:.4f}** at t=30
against the quoted 0.795 / 0.759 -> within 0.004 / 0.008. Reproduced.
Its residual-on-itself is exactly 0.5000 in both corpora (arithmetic check on the residualiser).
""")
    flush()

    # ---------------- 1. baseline trajectory + direction
    b = df[df.cell == "hell|prev|mean|W8"].pivot_table(index="corpus", columns="t", values="auc")
    BLOCKS.append("\n## 1. The baseline itself across t (directional)\n\n```\n" +
                  b.round(4).to_string() +
                  "\n```\n\nDirection at t=30 is `auc > 0.5` in both corpora: chunk-to-chunk route "
                  "change is **higher on successes / lower on failures** (a failing branch's routing "
                  "goes static). At t=20 corpus A flips sign (0.406) and corpus B is 0.644 -- the two "
                  "corpora do not even agree on the direction before t=25, consistent with the "
                  "known absence of lead time.\n")
    flush()

    # ---------------- 2. the grid, marginals
    lines = []
    for t in TS:
        sub = df[df.t == t]
        pv = sub.pivot_table(index="divergence", columns="corpus", values="det", aggfunc=["mean", "max"])
        lines.append(f"\n### t={t}: detection AUC by divergence (over all 69 temporal specs)\n\n```\n"
                     + pv.round(3).to_string() + "\n```")
    BLOCKS.append("\n## 2. Raw grid marginals\n" + "\n".join(lines) + "\n")
    flush()

    lines = []
    for t in TS:
        sub = df[df.t == t]
        pv = sub.pivot_table(index=["temporal", "window"], columns="corpus", values="det",
                             aggfunc=["mean", "max"])
        lines.append(f"\n### t={t}: detection AUC by temporal spec (over all 26 divergences)\n\n```\n"
                     + pv.round(3).to_string() + "\n```")
    BLOCKS.append("\n".join(lines) + "\n")
    flush()

    # ---------------- 3. variance decomposition: divergence vs temporal
    vd = []
    for t in TS:
        for tag in ("A", "B"):
            for col in ("det", "det_resid"):
                sub = df[(df.t == t) & (df.corpus == tag)]
                M = sub.pivot_table(index="divergence", columns=["temporal", "window"], values=col)
                v = M.values
                gm = v.mean()
                ss_tot = ((v - gm) ** 2).sum()
                ss_div = v.shape[1] * ((v.mean(1) - gm) ** 2).sum()
                ss_tmp = v.shape[0] * ((v.mean(0) - gm) ** 2).sum()
                vd.append(dict(t=t, corpus=tag, grid=col,
                               frac_divergence=ss_div / ss_tot,
                               frac_temporal=ss_tmp / ss_tot,
                               frac_interaction=1 - (ss_div + ss_tmp) / ss_tot,
                               sd_total=v.std(), range_div=np.ptp(v.mean(1)),
                               range_tmp=np.ptp(v.mean(0))))
    vdf = pd.DataFrame(vd)
    vdf.to_csv("/tmp/moe_agg_variance_decomp.csv", index=False)
    BLOCKS.append("\n## 3. Which stage matters more? (two-way variance decomposition of "
                  "detection AUC over the 26 x 69 table)\n\n```\n" +
                  vdf.round(4).to_string(index=False) + "\n```\n")
    flush()

    # ---------------- 4. cells that beat the baseline in BOTH corpora
    piv = df.pivot_table(index=["cell", "t"], columns="corpus", values=["det", "det_resid", "auc", "auc_resid"])
    base_det = {(tag, t): float(df[(df.corpus == tag) & (df.cell == "hell|prev|mean|W8") & (df.t == t)].det.iloc[0])
                for tag in ("A", "B") for t in TS}
    rows = []
    for (cell, t), r in piv.iterrows():
        rows.append(dict(cell=cell, t=t,
                         detA=r[("det", "A")], detB=r[("det", "B")],
                         rdetA=r[("det_resid", "A")], rdetB=r[("det_resid", "B")],
                         aucA=r[("auc", "A")], aucB=r[("auc", "B")],
                         raucA=r[("auc_resid", "A")], raucB=r[("auc_resid", "B")],
                         baseA=base_det[("A", t)], baseB=base_det[("B", t)]))
    R = pd.DataFrame(rows)
    R["beats_raw_both"] = (R.detA > R.baseA) & (R.detB > R.baseB)
    R["dirmatch_raw"] = np.sign(R.aucA - .5) == np.sign(R.aucB - .5)
    R["dirmatch_resid"] = np.sign(R.raucA - .5) == np.sign(R.raucB - .5)
    R["min_resid"] = np.minimum(R.rdetA, R.rdetB)
    R.to_csv("/tmp/moe_agg_joint.csv", index=False)

    txt = ["\n## 4. Does anything beat the baseline in BOTH corpora?\n"]
    for t in TS:
        s = R[R.t == t]
        nb = int(s.beats_raw_both.sum())
        nbd = int((s.beats_raw_both & s.dirmatch_raw).sum())
        txt.append(f"* **t={t}** (baseline A {base_det[('A',t)]:.3f} / B {base_det[('B',t)]:.3f}): "
                   f"{nb}/1794 cells beat it in both corpora raw; {nbd} of those also agree on "
                   f"direction between corpora.")
    txt.append("\n### Top 15 by min(det_A, det_B), RAW, direction required to match, t=30\n\n```")
    s = R[(R.t == 30) & R.dirmatch_raw].copy()
    s["mn"] = np.minimum(s.detA, s.detB)
    for _, r in s.sort_values("mn", ascending=False).head(15).iterrows():
        txt.append(f"  {r.cell:34s} min={r.mn:.4f}  A={r.detA:.4f} B={r.detB:.4f}  "
                   f"(raw dir A={r.aucA:.3f} B={r.aucB:.3f})")
    txt.append("```")
    txt.append("\n### Top 15 by min(det_resid_A, det_resid_B), RESIDUALISED, direction required "
               "to match, t=30\n\n```")
    s = R[(R.t == 30) & R.dirmatch_resid].copy()
    for _, r in s.sort_values("min_resid", ascending=False).head(15).iterrows():
        txt.append(f"  {r.cell:34s} min={r.min_resid:.4f}  A={r.rdetA:.4f} B={r.rdetB:.4f}  "
                   f"(raw dir A={r.raucA:.3f} B={r.raucB:.3f})")
    txt.append("```")
    BLOCKS.append("\n".join(txt) + "\n")
    flush()

    # ---------------- 5. permutation family p
    perm = []
    plines = ["\n## 5. Permutation family-wise p (200 draws, labels shuffled WITHIN group at "
              "branch level, one draw reused across all t)\n"]
    for tag in ("A", "B"):
        st = G[tag]
        for kind, col in (("raw", "det"), ("resid", "det_resid")):
            null = np.abs(np.load(f"/tmp/moe_agg_null_{tag}_{kind}.npy") - .5) + .5  # -> det
            obs = np.empty((len(cellnames), len(TS)))
            for j, t in enumerate(TS):
                sub = df[(df.corpus == tag) & (df.t == t)].set_index("cell")
                obs[:, j] = sub.loc[cellnames, col].values
            for famname, mask in (("full_family_incl_baseline", np.ones(len(cellnames), bool)),
                                  ("family_excl_baseline", np.arange(len(cellnames)) != bi)):
                o = obs[mask].max()
                nm = null[:, mask, :].max(axis=(1, 2))
                p = (1 + (nm >= o).sum()) / (1 + len(nm))
                ij = np.unravel_index(obs[mask].argmax(), obs[mask].shape)
                arg = cellnames[mask][ij[0]]
                perm.append(dict(corpus=tag, grid=kind, family=famname, argmax=arg,
                                 t_argmax=TS[ij[1]], obs=o, null_mean=nm.mean(),
                                 null_p95=np.percentile(nm, 95), null_max=nm.max(), p=p))
                plines.append(f"* corpus **{tag}**, **{kind}** grid, {famname}: argmax "
                              f"`{arg}` at t={TS[ij[1]]}, observed max det-AUC **{o:.4f}**; "
                              f"null max mean {nm.mean():.4f}, p95 {np.percentile(nm,95):.4f}, "
                              f"largest of 200 {nm.max():.4f} -> **family-wise p = {p:.4f}**")
    pdf = pd.DataFrame(perm)
    pdf.to_csv("/tmp/moe_agg_permutation.csv", index=False)
    BLOCKS.append("\n".join(plines) + "\n")
    flush()

    # ---------------- 6. cross-corpus sign agreement
    sl = ["\n## 6. Cross-corpus sign agreement\n"]
    for col, lab in (("dirmatch_raw", "raw"), ("dirmatch_resid", "residual")):
        tot = R[col].mean()
        per_t = R.groupby("t")[col].mean()
        sl.append(f"* **{lab}**: overall {tot*100:.1f}% of the 5382 (cell, t) pairs agree on "
                  f"sign; by t: " + ", ".join(f"t={t} {v*100:.1f}%" for t, v in per_t.items()))
    # by divergence family / by temporal
    key = df[df.corpus == "A"][["cell", "divergence", "temporal", "window"]].drop_duplicates()
    R2 = R.merge(key, on="cell", how="left")
    sl.append("\n### Residual sign agreement broken out (t=30)\n\n```")
    s30 = R2[R2.t == 30]
    sl.append("by divergence:")
    sl.append(s30.groupby("divergence")["dirmatch_resid"].mean().round(3).to_string())
    sl.append("\nby temporal operator:")
    sl.append(s30.groupby("temporal")["dirmatch_resid"].mean().round(3).to_string())
    sl.append("\nby window:")
    sl.append(s30.groupby("window")["dirmatch_resid"].mean().round(3).to_string())
    sl.append("```")
    BLOCKS.append("\n".join(sl) + "\n")
    flush()

    # ---------------- 7. per-cell pointwise p for the joint-best residual cells
    pl = ["\n## 7. Pointwise (per-cell) permutation p for the residual leaders\n\n```"]
    nulls = {tag: np.abs(np.load(f"/tmp/moe_agg_null_{tag}_resid.npy") - .5) + .5
              for tag in ("A", "B")}
    cand = R[(R.t == 30) & R.dirmatch_resid].sort_values("min_resid", ascending=False).head(10)
    ci = {c: i for i, c in enumerate(cellnames)}
    for _, r in cand.iterrows():
        i = ci[r.cell]
        ps = {}
        for tag in ("A", "B"):
            o = float(R[(R.cell == r.cell) & (R.t == 30)][f"rdet{tag}"].iloc[0])
            nm = nulls[tag][:, i, 2]
            ps[tag] = (1 + (nm >= o).sum()) / (1 + len(nm))
        pl.append(f"  {r.cell:34s} A {r.rdetA:.4f} p={ps['A']:.4f} | B {r.rdetB:.4f} p={ps['B']:.4f}")
    pl.append("```")
    BLOCKS.append("\n".join(pl) + "\n")
    flush()

    # ---------------- 8. the specific prior claim: long memory
    lm = ["\n## 8. The prior claim under test: does longer memory beat W=8?\n\n```"]
    for t in TS:
        for tag in ("A", "B"):
            row = []
            for W in lib.WINDOWS:
                v = df[(df.corpus == tag) & (df.t == t) & (df.cell == f"hell|prev|mean|W{W}")]
                row.append(f"W{W}={v.det.iloc[0]:.3f}/{v.det_resid.iloc[0]:.3f}")
            for a in ("0.15", "0.30", "0.50"):
                v = df[(df.corpus == tag) & (df.t == t) & (df.cell == f"hell|prev|ewmall{a}|W0")]
                row.append(f"ewmall{a}={v.det.iloc[0]:.3f}/{v.det_resid.iloc[0]:.3f}")
            v = df[(df.corpus == tag) & (df.t == t) & (df.cell == "hell|prev|cum_over_t|W0")]
            row.append(f"cum/t={v.det.iloc[0]:.3f}/{v.det_resid.iloc[0]:.3f}")
            lm.append(f"  t={t} {tag}: " + "  ".join(row))
    lm.append("```\n(each entry is raw / residualised detection AUC)")
    BLOCKS.append("\n".join(lm) + "\n")
    flush()
    print("wrote", MD)


if __name__ == "__main__":
    main()
