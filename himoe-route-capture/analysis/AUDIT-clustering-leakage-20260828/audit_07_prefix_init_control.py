#!/usr/bin/env python3
"""Does the common-absolute-prefix routing structure survive conditioning on
initial state?  Initial state alone gives in-sample failure AUC 0.83-0.95 within
each task, so any 'early prediction' must be tested inside (task, init) cells."""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import hypergeom
from audit_pipeline import (OUT, SEED, load_cache, build_anchor_signatures,
    normalize_trajectory_shape, aligned_landmark_distances, prepare_embedding,
    ward_labels, canonicalize, block_table)

def conditional_test(mask, failure, strat, perms, rng):
    groups = [np.flatnonzero(strat == s) for s in np.unique(strat)]
    obs = int((mask & failure).sum()); exp = var = 0.0
    null = np.zeros(perms, dtype=np.int64)
    for g in groups:
        N, K, n = len(g), int(failure[g].sum()), int(mask[g].sum())
        exp += n * K / N
        if N > 1:
            var += n * (K / N) * (1 - K / N) * (N - n) / (N - 1)
        if n == 0:
            continue
        if n == N:
            null += K; continue
        keys = rng.random((perms, N))
        pick = np.argpartition(keys, n - 1, axis=1)[:, :n]
        null += failure[g][pick].sum(axis=1)
    return dict(observed=obs, expected=round(float(exp), 2),
                sd=round(float(np.sqrt(var)), 3),
                z=(round(float((obs - exp) / np.sqrt(var)), 3) if var > 1e-12 else None),
                p_ge=round(float(((null >= obs).sum() + 1) / (perms + 1)), 5))

def main():
    c = load_cache(); failure = np.asarray(c["meta_failure"])
    task = np.asarray(c["meta_task"]).astype(str); init = np.asarray(c["meta_init_state_id"])
    score = np.asarray(c["diagnostic_mean_soft_speed"])
    length = np.asarray(c["meta_episode_length"])
    t = np.load(OUT / "common_prefix_tapes.npz", allow_pickle=True)
    win = json.loads(str(t["windows"])); full = np.asarray(t["full"])
    rng = np.random.default_rng(SEED)
    out = {}
    for nm, sl in (("common_prefix_full", slice(0, 10)), ("early_half", slice(0, 5))):
        per = {}
        for tk in sorted(set(task)):
            idx = np.flatnonzero(task == tk); nf = int(failure[idx].sum())
            if nf == 0:
                continue
            mat = full[idx][:, sl, :].reshape(len(idx), -1)
            seq = build_anchor_signatures(mat)
            emb = prepare_embedding(
                aligned_landmark_distances(normalize_trajectory_shape(seq), SEED), SEED)
            rows = []
            for k in (2, 3, 4, 6):
                lab = canonicalize(ward_labels(emb, k), score[idx])
                for r in block_table(lab, failure[idx]):
                    m = lab == r["block"]
                    r["k"] = k
                    r["marginal_p"] = float(hypergeom.sf(r["failure"] - 1, len(idx), nf, r["n"]))
                    r["cond_on_init"] = conditional_test(m, failure[idx], init[idx], 20000, rng)
                    rows.append(r)
            best = max(rows, key=lambda r: r["f1"])
            per[tk.split("/")[-1][:38]] = dict(
                abs_query_indices=win[tk]["full"][sl], base_rate=round(nf / len(idx), 4),
                best_f1_block=dict(k=best["k"], n=best["n"], failure=best["failure"],
                                   precision=best["precision"], recall=best["failure_recall"],
                                   marginal_p=f'{best["marginal_p"]:.3g}',
                                   conditional_on_init=best["cond_on_init"]),
                strongest_conditional=min(
                    ({"k": r["k"], "n": r["n"], "failure": r["failure"],
                      "precision": r["precision"], "cond": r["cond_on_init"]}
                     for r in rows if r["n"] >= 20), key=lambda r: r["cond"]["p_ge"]),
            )
            print(nm, tk.split("/")[-1][:34], per[tk.split("/")[-1][:38]]["best_f1_block"], flush=True)
        out[nm] = per
    (OUT / "prefix_init_control.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
