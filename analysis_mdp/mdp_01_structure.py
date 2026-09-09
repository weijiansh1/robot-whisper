"""(2)(3) Chain STRUCTURE: recurrent-class decomposition, trap sets, dwell times.

Label-free global state fit (legitimate: PCA + k-means never see the outcome).
Outcomes are used only to *describe* the structure afterwards.
"""
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, mannwhitneyu

import mdp_lib as L

FEATS = ("cell1280", "mean32")
EPS_GRID = (0.0, 0.01, 0.02, 0.05)


def runs_of(seq):
    """[(state, start, length)] run-length encoding."""
    out = []
    i = 0
    n = len(seq)
    while i < n:
        j = i + 1
        while j < n and seq[j] == seq[i]:
            j += 1
        out.append((int(seq[i]), i, j - i))
        i = j
    return out


def analyse(tag, feat, K, S, valid, meta, md):
    y = meta["success"].values.astype(bool)
    fail = ~y
    B = len(meta)
    allb = np.arange(B)
    C, term = L.chain_counts(S, valid, allb, K)
    rowsum_tr = C.sum(1)
    # terminal outflow: a branch that ends in s contributes one "exit to absorbing" from s
    exit_ct = np.bincount(term[term >= 0], minlength=K).astype(float)
    tot = rowsum_tr + exit_ct
    Pm = np.divide(C, np.maximum(tot, 1)[:, None])          # transient sub-stochastic matrix
    p_exit = np.divide(exit_ct, np.maximum(tot, 1))
    self_p = np.diag(Pm).copy()

    h, Q, Rn, esteps = L.absorption(C, term, allb, fail, K)

    rec = dict(corpus=tag, block="structure", feat=feat, K=K,
               self_p_max=float(self_p.max()), self_p_mean=float(self_p.mean()),
               self_p_median=float(np.median(self_p)),
               p_exit_max=float(p_exit.max()),
               h_min=float(h.min()), h_max=float(h.max()), h_std=float(h.std()),
               esteps_min=float(esteps.min()), esteps_max=float(esteps.max()))
    # --- length-code diagnostic: is the absorption probability just "expected remaining time"?
    rho = spearmanr(h, esteps).statistic
    rec["spearman_h_esteps"] = float(rho)

    # --- recurrent / closed class decomposition at several edge thresholds
    class_lines = []
    for eps in EPS_GRID:
        Adj = (Pm > eps).astype(np.int8)
        ncc, lab = connected_components(csr_matrix(Adj), directed=True, connection="strong")
        # closed class = SCC with no edge leaving it (among transient states)
        closed = []
        for c in range(ncc):
            mem = np.where(lab == c)[0]
            leak = Pm[np.ix_(mem, np.setdiff1d(np.arange(K), mem))].sum()
            if Adj[np.ix_(mem, np.setdiff1d(np.arange(K), mem))].sum() == 0:
                closed.append((c, mem, float(leak)))
        biggest = max((len(np.where(lab == c)[0]) for c in range(ncc)), default=0)
        rec[f"nscc_eps{eps}"] = int(ncc)
        rec[f"maxscc_eps{eps}"] = int(biggest)
        rec[f"nclosed_eps{eps}"] = len(closed)
        rec[f"closed_states_eps{eps}"] = int(sum(len(m) for _, m, _ in closed))
        for c, mem, leak in closed:
            # who visits this closed class, and do they fail?
            vis = np.array([np.isin(S[b][valid[b]], mem).any() for b in range(B)])
            fr = float(fail[vis].mean()) if vis.any() else np.nan
            class_lines.append(dict(corpus=tag, block="closed_class", feat=feat, K=K, eps=eps,
                                    size=len(mem), leak=leak, n_visit=int(vis.sum()),
                                    fail_rate_visitors=fr, base_fail=float(fail.mean()),
                                    states=";".join(map(str, mem))))
    for cl in class_lines:
        L.emit(cl)

    # --- dwell times
    rl_s, rl_f = [], []
    maxrun30_ok, maxrun30 = np.zeros(B, bool), np.zeros(B)
    per_state = {k: [[], []] for k in range(K)}
    for b in range(B):
        seq = S[b][valid[b]]
        rr = runs_of(seq)
        (rl_f if fail[b] else rl_s).extend([l for _, _, l in rr])
        for st, _, l in rr:
            per_state[st][int(fail[b])].append(l)
        if valid[b, 30]:
            maxrun30_ok[b] = True
            maxrun30[b] = max([l for st, s0, l in runs_of(seq[:31])], default=0)
    rl_s, rl_f = np.array(rl_s, float), np.array(rl_f, float)
    u = mannwhitneyu(rl_f, rl_s, alternative="two-sided")
    rec.update(dwell_mean_succ=float(rl_s.mean()), dwell_mean_fail=float(rl_f.mean()),
               dwell_p_ge7_succ=float((rl_s >= 7).mean()), dwell_p_ge7_fail=float((rl_f >= 7).mean()),
               dwell_max_succ=float(rl_s.max()), dwell_max_fail=float(rl_f.max()),
               dwell_mwu_p=float(u.pvalue),
               dwell_auc_runlevel=float(u.statistic / (len(rl_f) * len(rl_s))))
    # branch-level: longest dwell in the first 31 chunks
    m = maxrun30_ok
    a, d, n = L.auc_pair(-maxrun30[m], y[m], meta["group"].values[m])
    rec.update(maxdwell30_auc_succ=a, maxdwell30_det_auc=d)
    # >=7 dwell rule (compendium 22 style)
    fired = m & (maxrun30 >= 7)
    rec["dwell7_fires"] = int(fired.sum())
    rec["dwell7_fail_rate"] = float(fail[fired].mean()) if fired.any() else np.nan
    rec["dwell7_base_fail"] = float(fail[m].mean())
    L.emit(rec)
    md.append(rec)
    return rec


def main():
    L.note("\n## 2. CHAIN STRUCTURE -- is there an absorbing / trap set at all?\n")
    md = []
    for tag in ("A", "B"):
        P, valid, meta = L.load(tag)
        B = len(meta)
        for feat in FEATS:
            F = L.features(P, valid, feat)
            for K in L.KS:
                S, _, _ = L.fit_states(F, valid, np.arange(B), K)
                np.save(f"{L.OUT}/states_{tag}_{feat}_K{K}.npy", S)
                analyse(tag, feat, K, S, valid, meta, md)
            del F
        L.flush()

    d = pd.DataFrame(md)
    L.note("### 2a. Self-transition and closed-class decomposition (eps = min edge prob kept)\n")
    L.note("| corpus | feat | K | max P(s->s) | mean P(s->s) | #SCC eps=0 | maxSCC eps=0 | "
           "#closed eps=0 | #closed eps=.02 | #closed eps=.05 | states in closed eps=.05 |")
    L.note("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        L.note(f"| {r.corpus} | {r.feat} | {r.K} | {r.self_p_max:.3f} | {r.self_p_mean:.3f} | "
               f"{int(r['nscc_eps0.0'])} | {int(r['maxscc_eps0.0'])} | {int(r['nclosed_eps0.0'])} | "
               f"{int(r['nclosed_eps0.02'])} | {int(r['nclosed_eps0.05'])} | "
               f"{int(r['closed_states_eps0.05'])} |")

    L.note("\n### 2b. Absorption probability h(s) = P(eventual failure | state s), and whether it "
           "is just a clock\n")
    L.note("| corpus | feat | K | h range | h std | E[remaining steps] range | "
           "Spearman(h, E[steps]) |")
    L.note("|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        L.note(f"| {r.corpus} | {r.feat} | {r.K} | {r.h_min:.3f}-{r.h_max:.3f} | {r.h_std:.3f} | "
               f"{r.esteps_min:.1f}-{r.esteps_max:.1f} | {r.spearman_h_esteps:+.3f} |")

    L.note("\n### 2c. Dwell-time distributions, eventual success vs eventual failure\n")
    L.note("| corpus | feat | K | mean dwell S | mean dwell F | P(dwell>=7) S | P(dwell>=7) F | "
           "run-level AUC | MWU p | max-dwell-by-t30 det AUC | dwell>=7 rule: fires / fail rate "
           "(base) |")
    L.note("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        L.note(f"| {r.corpus} | {r.feat} | {r.K} | {r.dwell_mean_succ:.2f} | {r.dwell_mean_fail:.2f} | "
               f"{r.dwell_p_ge7_succ:.4f} | {r.dwell_p_ge7_fail:.4f} | {r.dwell_auc_runlevel:.3f} | "
               f"{r.dwell_mwu_p:.2e} | {r.maxdwell30_det_auc:.3f} | {int(r.dwell7_fires)} / "
               f"{r.dwell7_fail_rate:.3f} ({r.dwell7_base_fail:.3f}) |")
    d.to_csv(f"{L.OUT}/structure_summary.csv", index=False)
    d.to_csv("/tmp/mdp_structure_summary.csv", index=False)
    L.note("")


if __name__ == "__main__":
    main()
