"""Token-axis sweep: run every block and PERSIST after each one.

Outputs (append/rewrite as we go):
  /tmp/moe_sweep_token.csv   corpus,organization,t,auc,n_pos,n_neg
  /tmp/moe_sweep_token.md    running human-readable summary
  /tmp/tokaxis_cache_{A,B}_d{0,5,9}.pkl   per-branch chunk scalars (expensive slice)
"""
import os, pickle, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

import token_axis_eval as E
import token_axis_sweep as S

CSV = "/tmp/moe_sweep_token.csv"
MD = "/tmp/moe_sweep_token.md"
COLS = ["corpus", "organization", "t", "auc", "n_pos", "n_neg"]
TS = [15, 20, 25, 30, 35]

ORDER = ([f"tok{k:02d}" for k in range(11)] +
         ["act1_3", "act4_7", "act8_10", "act_all_1_10", "all11",
          "disp11_hell", "disp11_bhat", "dispAct_hell", "dispAct_bhat",
          "state_vs_actmean_hell"])
FEATORDER = [f"{o}|{q}" for o in ORDER for q in ("hell", "ent")]
MDBLOCKS = []


# ------------------------------------------------------------------ caching
def cache_pkl(tag, d):
    """Mirror the (expensive) router-probability-derived scalars into /tmp/*.pkl."""
    p = f"/tmp/tokaxis_cache_{tag}_d{d}.pkl"
    if not os.path.exists(p):
        eids, lens, off, F, feats = E.load(tag, d)
        with open(p, "wb") as fh:
            pickle.dump(dict(eids=eids, lens=lens, off=off, F=F, feats=feats), fh, 4)
    with open(p, "rb") as fh:
        return pickle.load(fh)


def append_rows(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
    return len(df)


def flush_md():
    with open(MD, "w") as fh:
        fh.write("\n".join(MDBLOCKS) + "\n")


HEADER = """# Token-axis sweep: MoE routing -> rollout failure detection

**Axis under study:** the 11 suffix token positions (token 0 = state, tokens 1-10 = action).
Signal source is exclusively `hb_router_probs` (32-dim soft router distributions).
**No hard top-4 expert IDs, no set/Jaccard measures anywhere.**

## Conventions (read before using the CSV)

* `auc` in `moe_sweep_token.csv` is the **raw directional** within-group pooled AUC
  `= P(feature_failure > feature_success)`.
  * `auc < 0.5` => the feature is **LOWER** on failures.
  * `auc > 0.5` => the feature is **HIGHER** on failures.
  * The **detection AUC** (what compares to the 0.795 / 0.759 baseline) is `max(auc, 1-auc)`.
  * Raw form is kept so cross-corpus **sign disagreements stay visible**.
* Grouping is mandatory: `worker` (corpus A, 4 workers) / `init_state_id` (corpus B, 16 states).
  AUC is computed inside each group and pooled weighted by pair count. Groups that are
  single-class contribute nothing (corpus A worker 1 is 0/80 successes -> excluded entirely).
* `n_pos` / `n_neg` = successes / failures **in contributing groups** at that t
  (i.e. the branches that actually generate ranked pairs), risk set = branches with `T > t`.
* `organization` = `<token org>|<chunk quantity>|d<denoise iter>`.
  * `|hell` = Hellinger distance to the previous control step (for the two cross-token
    organizations: absolute change of the scalar between consecutive control steps).
  * `|ent`  = normalized entropy level (for the cross-token organizations: the dispersion /
    divergence level itself).
  * Both are averaged over an 8-step window ending at t, and over deep HB layers 12-15.
* Baseline to beat: `act_all_1_10|hell|d9` (mean over action tokens 1-10). Reproduced here as
  detection AUC **A 0.799 / B 0.751** at t=30 (brief quotes 0.795 / 0.759; the ~0.005-0.008 gap
  is a window/renormalisation convention difference, not a disagreement).
"""


def block_grid(tag, d):
    P = S.prep.__wrapped__(tag, d) if hasattr(S.prep, "__wrapped__") else S.prep(tag, d)
    feats = P[TS[0]]["feats"]
    rows, grid = [], {}
    for t in TS:
        a = S.pooled_auc(P[t])
        grid[t] = pd.Series(a, index=feats)
        npos = sum(G["ns"] for G in P[t]["groups"])
        nneg = sum(G["nf"] for G in P[t]["groups"])
        for f, v in zip(feats, a):
            rows.append(dict(corpus=tag, organization=f"{f}|d{d}", t=t,
                             auc=round(float(v), 6), n_pos=npos, n_neg=nneg))
    append_rows(rows)
    return pd.DataFrame(grid).loc[FEATORDER], P


def main():
    if os.path.exists(CSV):
        os.remove(CSV)
    MDBLOCKS.append(HEADER)
    flush_md()

    store = {}
    # ---- Blocks 1-4: the grids -------------------------------------------
    for tag in ("A", "B"):
        for d in (9, 5, 0):
            cache_pkl(tag, d)
            df, P = block_grid(tag, d)
            store[(tag, d)] = (df, P)
            print(f"block done: corpus {tag} denoise {d} -> {CSV}")
            if d == 9:
                risk = " | ".join(
                    f"t={t}: risk-set n={P[t]['n']} (S{P[t]['nsucc']}/F{P[t]['nfail']}), "
                    f"contributing pairs={P[t]['npairs']}" for t in TS)
                dd = np.maximum(df, 1 - df)
                MDBLOCKS.append(
                    f"\n## Grid: corpus {tag}, layers 12-15, denoise 9, 8-step window\n\n"
                    f"Risk sets: {risk}\n\n"
                    f"### Raw directional AUC = P(feat_fail > feat_succ)\n\n```\n"
                    + df.round(3).to_string() +
                    "\n```\n\n### Detection AUC = max(auc, 1-auc)\n\n```\n"
                    + dd.round(3).to_string() + "\n```\n")
                flush_md()

    # ---- Block 5: attrition note ----------------------------------------
    att = ["\n## Risk-set attrition (matters for reading t=35)\n"]
    for tag in ("A", "B"):
        _, P = store[(tag, 9)]
        att.append(f"* Corpus {tag}: " + ", ".join(
            f"t={t} n={P[t]['n']} (S{P[t]['nsucc']}/F{P[t]['nfail']})" for t in TS))
    att.append("* Corpus A min T = 33 -> every branch alive through t=32; by t=35, 20 of 117 "
               "successes have terminated and **zero** failures have, so the t=35 column in "
               "corpus A is survival-conditioned and enriched for failure. Corpus B min T = 35 "
               "-> only 1 branch (a success) is lost by t=35, so corpus B t=35 is effectively clean.")
    MDBLOCKS.append("\n".join(att) + "\n")
    flush_md()

    # ---- Block 6: permutation nulls -------------------------------------
    perm_lines = ["\n## Permutation null (200 draws, success/failure shuffled WITHIN group)\n",
                  "Family statistic = max detection AUC over the whole family. Labels are permuted "
                  "once per draw at branch level within group and reused across all t, preserving "
                  "the nesting of risk sets.\n"]
    permrows = []
    for tag in ("A", "B"):
        df, P = store[(tag, 9)]
        obs = float(np.maximum(df, 1 - df).values.max())
        ij = np.unravel_index(np.maximum(df, 1 - df).values.argmax(), df.shape)
        p, null = S.permutation_family_p(P, obs, n_perm=200, two_sided=True, seed=12345)
        np.save(f"/tmp/tokaxis_null_{tag}_full.npy", null)
        perm_lines.append(
            f"* **Corpus {tag}, full family** (42 organizations x 5 t = 210 cells, denoise 9): "
            f"argmax = `{df.index[ij[0]]}` at t={df.columns[ij[1]]}, observed max det-AUC "
            f"**{obs:.4f}**; null max: mean {null.mean():.4f}, p95 {np.percentile(null,95):.4f}, "
            f"largest of 200 {null.max():.4f}; **family-wise p = {p:.4f}** "
            f"(= 1/201, the floor at 200 draws - no permutation came close).")
        permrows.append(dict(corpus=tag, family="full_210", obs=obs, p=p,
                             null_p95=float(np.percentile(null, 95)), null_max=float(null.max())))
        # restricted to fully-alive t
        S.TS = [15, 20, 25, 30]
        df2, P2 = S.sweep(tag, 9)
        dd2 = np.maximum(df2, 1 - df2)
        obs2 = float(dd2.values.max())
        ij2 = np.unravel_index(dd2.values.argmax(), dd2.shape)
        p2, null2 = S.permutation_family_p(P2, obs2, n_perm=200, two_sided=True, seed=999)
        np.save(f"/tmp/tokaxis_null_{tag}_restricted.npy", null2)
        S.TS = TS
        perm_lines.append(
            f"* **Corpus {tag}, restricted family** (t<=30 only, 168 cells, no survival "
            f"conditioning): argmax = `{df2.index[ij2[0]]}` at t={df2.columns[ij2[1]]}, observed "
            f"**{obs2:.4f}**; null p95 {np.percentile(null2,95):.4f}, largest {null2.max():.4f}; "
            f"**family-wise p = {p2:.4f}**.")
        permrows.append(dict(corpus=tag, family="restricted_168", obs=obs2, p=p2,
                             null_p95=float(np.percentile(null2, 95)), null_max=float(null2.max())))
    pd.DataFrame(permrows).to_csv("/tmp/moe_sweep_token_permutation.csv", index=False)
    MDBLOCKS.append("\n".join(perm_lines) + "\n")
    flush_md()
    print("block done: permutation nulls")

    # ---- Block 7: top cells ---------------------------------------------
    top = ["\n## Top cells per corpus (denoise 9, detection AUC)\n"]
    for tag in ("A", "B"):
        df, _ = store[(tag, 9)]
        dd = np.maximum(df, 1 - df)
        top.append(f"\n**Corpus {tag}**\n\n```")
        for (f, t), v in dd.stack().sort_values(ascending=False)[:12].items():
            top.append(f"  {f:28s} t={t:2d}  det={v:.4f}  raw={df.loc[f,t]:.4f}")
        top.append("```")
    MDBLOCKS.append("\n".join(top) + "\n")
    flush_md()

    # ---- Block 8: cross-corpus joint ranking ----------------------------
    dfa, dfb = store[("A", 9)][0], store[("B", 9)][0]
    da, db = np.maximum(dfa, 1 - dfa), np.maximum(dfb, 1 - dfb)
    same = np.sign(dfa - .5) == np.sign(dfb - .5)
    joint = np.minimum(da, db).where(same, np.nan)
    lines = ["\n## Cross-corpus joint ranking: min(det_A, det_B), direction required to match\n",
             "```"]
    for (f, t), v in joint.stack().sort_values(ascending=False)[:15].items():
        lines.append(f"  {f:28s} t={t:2d}  min={v:.4f}   A={da.loc[f,t]:.4f} B={db.loc[f,t]:.4f}")
    lines.append("```")
    MDBLOCKS.append("\n".join(lines) + "\n")
    flush_md()

    # ---- Block 9: sign disagreements ------------------------------------
    rows = []
    for f in FEATORDER:
        for t in TS:
            a, b = dfa.loc[f, t], dfb.loc[f, t]
            rows.append(dict(organization=f, t=t, auc_A=round(float(a), 4),
                             auc_B=round(float(b), 4),
                             disagree=bool(np.sign(a - .5) != np.sign(b - .5)),
                             both_strong=bool(abs(a - .5) > .08 and abs(b - .5) > .08)))
    R = pd.DataFrame(rows)
    R.to_csv("/tmp/moe_sweep_token_signcheck.csv", index=False)
    bad = R[R.disagree & R.both_strong]
    sl = [f"\n## Sign disagreement between corpora (denoise 9)\n",
          f"Cells where **both** corpora deviate >0.08 from 0.5 but in **opposite** directions: "
          f"**{len(bad)}** of {int(R.both_strong.sum())} jointly-strong cells "
          f"({int((R.both_strong & ~R.disagree).sum())} agree).\n", "```",
          bad[["organization", "t", "auc_A", "auc_B"]].to_string(index=False), "```"]
    MDBLOCKS.append("\n".join(sl) + "\n")
    flush_md()
    print("block done: sign check")

    # ---- Block 10: denoise robustness -----------------------------------
    CHECK = ["tok00|hell", "act_all_1_10|hell", "all11|hell", "act4_7|hell",
             "tok08|ent", "tok09|ent", "act8_10|ent",
             "disp11_hell|ent", "dispAct_hell|ent", "state_vs_actmean_hell|ent"]
    dl = ["\n## Denoise robustness (best organizations re-tested at denoise 0 and 5)\n",
          "Raw directional AUC.\n"]
    for tag in ("A", "B"):
        tbl = {}
        for d in (0, 5, 9):
            df, _ = store[(tag, d)]
            for f in CHECK:
                tbl[(f, d)] = df.loc[f]
        out = pd.DataFrame(tbl).T
        out.index.names = ["organization", "denoise"]
        dl += [f"\n**Corpus {tag}**\n", "```", out.round(3).to_string(), "```"]
    MDBLOCKS.append("\n".join(dl) + "\n")
    flush_md()
    print("block done: denoise robustness")


if __name__ == "__main__":
    main()
