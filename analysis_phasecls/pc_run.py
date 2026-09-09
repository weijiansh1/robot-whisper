"""Driver: sentinels -> permutation null -> per-group breakdown -> corpus B ->
sensitivity grid -> what-is-the-classifier-using decomposition.
Everything is written to disk as it is produced."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pc_common import aggregate, phase_bins  # noqa: E402
from pc_eval import (BINS, BLOCKS, CHANNELS, CONTRASTS, DENOISE, OUT, build_features,
                     channel_matrix, load_corpus_a, load_corpus_b, logo_auc,
                     permutation_null)

CSV = Path("/tmp/moe_phasecls.csv")
MD = Path("/tmp/moe_phasecls.md")
MIR = OUT
COLS = ["stage", "corpus", "contrast", "block", "denoise", "channel", "bins",
        "variant", "group", "auc", "n_pos", "n_neg", "p_marg", "p_fwer", "note"]
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEED = 20260829


def emit(rows):
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = ""
    df = df[COLS]
    df.to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
    shutil.copy(CSV, MIR / "moe_phasecls.csv")
    return df


def note(text):
    with open(MD, "a") as f:
        f.write(text + "\n")
    shutil.copy(MD, MIR / "moe_phasecls.md")
    print(text, flush=True)


def main():
    CSV.unlink(missing_ok=True)
    MD.unlink(missing_ok=True)
    note(f"# MoE trap-type phase-classification validation  ({N_PERM} perm draws, seed {SEED})\n")
    note("Groups: corpus A = worker (4), corpus B = init_state_id (16).")
    note("Classes: stagnation = label_stagnation; loop = loop_or_cycling & ~stagnation; "
         "other = neither.  Headline setting = HB back block (layers 12-15), denoise 9, 5 bins.\n")

    corpora = {}
    labA, gA, idA = load_corpus_a()
    corpora["A"] = (labA, gA, build_features(OUT / "hell_A.npz", idA))
    labB, gB, idB = load_corpus_b()
    corpora["B"] = (labB, gB, build_features(OUT / "hell_B.npz", idB))
    labBs, gBs, idBs = load_corpus_b(stag_col="cls_strict_is_stag")
    featBs = build_features(OUT / "hell_B.npz", idBs)

    note("## 0. Corpus composition\n")
    for cn, (lab, g, _) in corpora.items():
        vc = lab["cls"].value_counts().to_dict()
        note(f"- **corpus {cn}**: {len(lab)} failure branches, "
             f"groups={len(np.unique(g))} -> {vc}")
        note("```\n" + pd.crosstab(g, lab["cls"]).to_string() + "\n```")

    # ---------------------------------------------------------- 1. SENTINELS
    note("\n## 1. Sentinels (reported before the headline)\n")
    note("### 1a. Length sentinel\n")
    note("| corpus | contrast | n_pos/n_neg | LOGO length AUC | median len pos | median len neg | MWU p |")
    note("|---|---|---|---|---|---|---|")
    rows = []
    for cn, (lab, g, _) in corpora.items():
        cls, L = lab["cls"].to_numpy(), lab["n_q"].to_numpy().astype(float)
        for ca, cb in CONTRASTS:
            m = np.isin(cls, [ca, cb])
            y = (cls[m] == ca).astype(int)
            a = logo_auc(L[m][:, None], y, g[m])
            uniq = len(np.unique(L[m]))
            p = 1.0 if uniq == 1 else stats.mannwhitneyu(L[m][y == 1], L[m][y == 0]).pvalue
            note(f"| {cn} | {ca} vs {cb} | {y.sum()}/{(1-y).sum()} | {a:.3f} | "
                 f"{np.median(L[m][y==1]):.0f} | {np.median(L[m][y==0]):.0f} | {p:.2g} "
                 f"{'(length CONSTANT)' if uniq==1 else ''}|")
            rows.append(dict(stage="sentinel_length", corpus=cn, contrast=f"{ca}_vs_{cb}",
                             channel="length", auc=a, n_pos=int(y.sum()),
                             n_neg=int((1 - y).sum()), p_marg=p,
                             note=f"n_unique_lengths={uniq}"))
    emit(rows)

    note("\n### 1b. Phase-shuffle sentinel (5 bins permuted within each branch, back/d9, 20 reshuffles)\n")
    note("| corpus | contrast | channel | true AUC | shuffled mean [min,max] | delta |")
    note("|---|---|---|---|---|---|")
    rows = []
    for cn, (lab, g, feats) in corpora.items():
        cls = lab["cls"].to_numpy()
        ids = lab["episode_id"].tolist()
        shuf = [build_features(OUT / f"hell_{cn}.npz", ids,
                               rng_shuffle=np.random.default_rng(SEED + 1000 + k))
                for k in range(20)]
        for ca, cb in CONTRASTS:
            m = np.isin(cls, [ca, cb])
            y = (cls[m] == ca).astype(int)
            for ch in CHANNELS:
                t = logo_auc(channel_matrix(feats, "back", 9, ch, 5)[m], y, g[m])
                s = [logo_auc(channel_matrix(f, "back", 9, ch, 5)[m], y, g[m]) for f in shuf]
                note(f"| {cn} | {ca} vs {cb} | {ch} | {t:.3f} | "
                     f"{np.mean(s):.3f} [{np.min(s):.3f},{np.max(s):.3f}] | {np.mean(s)-t:+.3f} |")
                rows.append(dict(stage="sentinel_phaseshuffle", corpus=cn,
                                 contrast=f"{ca}_vs_{cb}", block="back", denoise=9,
                                 channel=ch, bins=5, variant="shuffled",
                                 auc=float(np.mean(s)), n_pos=int(y.sum()),
                                 n_neg=int((1 - y).sum()),
                                 note=f"true={t:.4f} min={np.min(s):.4f} max={np.max(s):.4f}"))
        del shuf
    emit(rows)

    # ---------------------------------------------------- 2. SENSITIVITY GRID
    note("\n## 2. Observed sensitivity grid (LOGO within-group paired AUC, pooled by pair count)\n")
    obs = {}
    rows = []
    variants = [("A", labA, gA, corpora["A"][2], "main"),
                ("B", labB, gB, corpora["B"][2], "main"),
                ("Bstrict", labBs, gBs, featBs, "strict_stag")]
    for cn, lab, g, feats, var in variants:
        cls = lab["cls"].to_numpy()
        for ca, cb in CONTRASTS:
            m = np.isin(cls, [ca, cb])
            y = (cls[m] == ca).astype(int)
            for blk in BLOCKS:
                for dn in DENOISE:
                    for ch in CHANNELS:
                        for nb in BINS:
                            a = logo_auc(channel_matrix(feats, blk, dn, ch, nb)[m], y, g[m])
                            obs[(cn, ca, cb, blk, dn, ch, nb)] = a
                            rows.append(dict(stage="grid", corpus=cn, variant=var,
                                             contrast=f"{ca}_vs_{cb}", block=blk,
                                             denoise=dn, channel=ch, bins=nb, auc=a,
                                             n_pos=int(y.sum()), n_neg=int((1 - y).sum())))
    grid = emit(rows)
    grid.to_csv(MIR / "grid_observed.csv", index=False)
    for cn in ["A", "B", "Bstrict"]:
        for ca, cb in CONTRASTS:
            sub = grid[(grid.corpus == cn) & (grid.contrast == f"{ca}_vs_{cb}")]
            note(f"\n**corpus {cn} / {ca} vs {cb}** (n={int(sub.n_pos.iloc[0])}/{int(sub.n_neg.iloc[0])})\n")
            note("| block | denoise | channel | " + " | ".join(f"{b} bins" for b in BINS) + " |")
            note("|---|---|---|" + "---|" * len(BINS))
            for blk in BLOCKS:
                for dn in DENOISE:
                    for ch in CHANNELS:
                        v = [sub[(sub.block == blk) & (sub.denoise == dn) &
                                 (sub.channel == ch) & (sub.bins == nb)].auc.iloc[0] for nb in BINS]
                        note(f"| {blk} | {dn} | {ch} | " + " | ".join(f"{x:.3f}" for x in v) + " |")

    # ------------------------------------------------- 3. PER-GROUP BREAKDOWN
    note("\n## 3. Leave-one-group-out breakdown (back / denoise 9 / 5 bins)\n")
    rows = []
    for cn, (lab, g, feats) in corpora.items():
        cls = lab["cls"].to_numpy()
        for ca, cb in CONTRASTS:
            m = np.isin(cls, [ca, cb])
            y = (cls[m] == ca).astype(int)
            for ch in CHANNELS:
                X = channel_matrix(feats, "back", 9, ch, 5)[m]
                pooled, per = logo_auc(X, y, g[m], return_per_group=True)
                tot = sum(n for _, n in per.values())
                note(f"\n- **{cn} / {ca} vs {cb} / {ch}**: pooled = {pooled:.3f} "
                     f"({len(per)} contributing groups, {tot} pairs)")
                for k, (a, n) in sorted(per.items()):
                    rest = {kk: v for kk, v in per.items() if kk != k}
                    dr = (sum(a2 * n2 for a2, n2 in rest.values()) /
                          sum(n2 for _, n2 in rest.values())) if rest else float("nan")
                    note(f"    - g{k}: AUC={a:.3f}  pairs={n} ({100*n/tot:.0f}% of weight)"
                         f"   -> pooled without g{k} = {dr:.3f}")
                    rows.append(dict(stage="per_group", corpus=cn, contrast=f"{ca}_vs_{cb}",
                                     block="back", denoise=9, channel=ch, bins=5,
                                     group=k, auc=a, note=f"pairs={n} pooled_without={dr:.4f}"))
    emit(rows)

    # ------------------------------- 5. WHAT IS THE CLASSIFIER ACTUALLY USING
    note("\n## 5. Decomposition: level vs shape, and the matched physical control\n")
    rows = []
    for cn, (lab, g, _) in corpora.items():
        cls = lab["cls"].to_numpy()
        ids = lab["episode_id"].tolist()
        z = np.load(OUT / f"hell_{cn}.npz")
        ph = np.load(OUT / f"phys_{cn}.npz")
        for fam, toks in (("state", [0]), ("action", list(range(1, 11)))):
            rser = [aggregate(z[str(i)], "back", 9, toks) for i in ids]
            rm = np.array([s.mean() for s in rser])
            rp = np.stack([phase_bins(s, 5) for s in rser])
            pser = [ph[str(i)] for i in ids]
            pm = np.array([s.mean() for s in pser])
            pp = np.stack([phase_bins(s, 5) for s in pser])
            if fam == "state":
                note(f"\n**corpus {cn} -- phase profile, state token, back/d9 (class mean)**\n")
                note("| class | n | bin1 | bin2 | bin3 | bin4 | bin5 | branch mean |")
                note("|---|---|---|---|---|---|---|---|")
                for c in ["stagnation", "loop", "other"]:
                    mm = cls == c
                    note(f"| {c} | {mm.sum()} | " + " | ".join(f"{v:.4f}" for v in rp[mm].mean(0))
                         + f" | {rm[mm].mean():.4f} |")
                note("\n_matched physical control (eef displacement per query, same bins):_\n")
                note("| class | n | bin1 | bin2 | bin3 | bin4 | bin5 | branch mean |")
                note("|---|---|---|---|---|---|---|---|")
                for c in ["stagnation", "loop", "other"]:
                    mm = cls == c
                    note(f"| {c} | {mm.sum()} | " + " | ".join(f"{v:.4f}" for v in pp[mm].mean(0))
                         + f" | {pm[mm].mean():.4f} |")
                note(f"\nSpearman(route change mean, eef step mean) = "
                     f"{stats.spearmanr(rm, pm).statistic:.3f}; "
                     f"late bin only = {stats.spearmanr(rp[:,4], pp[:,4]).statistic:.3f}\n")
            note(f"\n| corpus {cn} / {fam} | 5-bin | mean only (1 feat) | shape only (mean removed) "
                 "| late-minus-early (1 feat) | PHYS 5-bin | PHYS mean | route resid on phys |")
            note("|---|---|---|---|---|---|---|---|")
            for ca, cb in CONTRASTS:
                m = np.isin(cls, [ca, cb])
                y = (cls[m] == ca).astype(int)
                sh = rp - rp.mean(1, keepdims=True)
                res = rm - np.polyval(np.polyfit(pm, rm, 1), pm)
                vals = dict(
                    five=logo_auc(rp[m], y, g[m]),
                    mean=logo_auc(rm[m][:, None], y, g[m]),
                    shape=logo_auc(sh[m], y, g[m]),
                    lme=logo_auc((rp[:, 4] - rp[:, 0])[m][:, None], y, g[m]),
                    phys5=logo_auc(pp[m], y, g[m]),
                    physm=logo_auc(pm[m][:, None], y, g[m]),
                    resid=logo_auc(res[m][:, None], y, g[m]))
                note(f"| {ca} vs {cb} | " + " | ".join(f"{vals[k]:.3f}" for k in
                     ["five", "mean", "shape", "lme", "phys5", "physm", "resid"]) + " |")
                for k, v in vals.items():
                    rows.append(dict(stage="decomp", corpus=cn, contrast=f"{ca}_vs_{cb}",
                                     block="back", denoise=9, channel=f"{fam}_{k}", bins=5,
                                     auc=v, n_pos=int(y.sum()), n_neg=int((1 - y).sum())))
    emit(rows)

    # ---------------------------------------------------- 4. PERMUTATION NULL
    note(f"\n## 4. Permutation null ({N_PERM} draws, class labels shuffled WITHIN group at branch "
         "level, whole LOGO pipeline recomputed each draw)\n")
    cells_full = [(ca, cb, blk, dn, ch, nb) for ca, cb in CONTRASTS for blk in BLOCKS
                  for dn in DENOISE for ch in CHANNELS for nb in BINS]
    head_idx = [i for i, c in enumerate(cells_full) if c[2] == "back" and c[3] == 9]
    for cn, (lab, g, feats) in corpora.items():
        cls = lab["cls"].to_numpy()
        t0 = time.time()
        null = permutation_null(feats, cls, g, N_PERM, SEED + (0 if cn == "A" else 7), cells_full)
        np.save(MIR / f"perm_null_{cn}.npy", null)
        note(f"\n_corpus {cn}: {N_PERM} draws x {len(cells_full)} cells in {time.time()-t0:.0f}s_\n")
        maxT_full = np.nanmax(null, axis=1)
        maxT_head = np.nanmax(null[:, head_idx], axis=1)
        note(f"null max-statistic: family of 36 (back/d9) mean={maxT_head.mean():.3f} "
             f"p95={np.quantile(maxT_head,0.95):.3f} max={maxT_head.max():.3f}; "
             f"family of 144 mean={maxT_full.mean():.3f} p95={np.quantile(maxT_full,0.95):.3f} "
             f"max={maxT_full.max():.3f}\n")
        rows = []
        note("| contrast | block | dn | channel | bins | obs AUC | null mean | null p95 | p_marg | p_FWER(36) | p_FWER(144) |")
        note("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, (ca, cb, blk, dn, ch, nb) in enumerate(cells_full):
            o = obs[(cn, ca, cb, blk, dn, ch, nb)]
            col = null[:, i][~np.isnan(null[:, i])]
            pm_ = (1 + np.sum(col >= o)) / (1 + len(col))
            pf36 = (1 + np.sum(maxT_head >= o)) / (1 + len(maxT_head))
            pf144 = (1 + np.sum(maxT_full >= o)) / (1 + len(maxT_full))
            rows.append(dict(stage="perm", corpus=cn, contrast=f"{ca}_vs_{cb}", block=blk,
                             denoise=dn, channel=ch, bins=nb, auc=o, p_marg=pm_, p_fwer=pf36,
                             note=f"p_fwer144={pf144:.4f} null_mean={col.mean():.4f} "
                                  f"null_p95={np.quantile(col,0.95):.4f} n_null={len(col)}"))
            if blk == "back" and dn == 9:
                note(f"| {ca} vs {cb} | {blk} | {dn} | {ch} | {nb} | {o:.3f} | {col.mean():.3f} | "
                     f"{np.quantile(col,0.95):.3f} | {pm_:.4f} | {pf36:.4f} | {pf144:.4f} |")
        emit(rows)
    note("\nDONE")


if __name__ == "__main__":
    main()
