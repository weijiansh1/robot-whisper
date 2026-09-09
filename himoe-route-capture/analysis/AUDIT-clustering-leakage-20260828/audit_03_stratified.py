#!/usr/bin/env python3
"""B + E-adjacent tests: is the consensus core anything beyond (task, length)?

1. LENGTH SURROGATE: build a partition from (task label, per-task length rank)
   with ZERO routing data; compare to the published raw K6 / consensus core.
2. CONDITIONAL PERMUTATION: shuffle outcome within (task, episode_length)
   strata; is core membership still associated with outcome?
3. WITHIN-STRATUM RE-CLUSTERING: re-run the identical aligned-kernel pipeline
   restricted to a single (task, length) stratum, and report the power ceiling.
4. Explained variance of the clustered embedding by (task, length) alone.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import hypergeom
from sklearn.linear_model import LinearRegression

from audit_pipeline import (
    OUT, SEED, CANDIDATE_K, load_cache, build_anchor_signatures,
    normalize_trajectory_shape, aligned_landmark_distances, prepare_embedding,
    ward_labels, canonicalize, summarize, block_table, adjusted_rand_score,
)

STEP_CAP = {
    "libero_goal/open_the_middle_drawer_of_the_cabinet": 30,
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside": 30,
    "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": 52,
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": 22,
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": 22,
}


def main() -> None:
    cache = load_cache()
    failure = np.asarray(cache["meta_failure"])
    task = np.asarray(cache["meta_task"]).astype(str)
    length = np.asarray(cache["meta_episode_length"])
    init = np.asarray(cache["meta_init_state_id"])
    score = np.asarray(cache["diagnostic_mean_soft_speed"])
    geom = np.asarray(cache["feature_primary_geometry"])
    core = np.load(OUT / "core_membership.npz", allow_pickle=False)
    consensus, raw_c1 = core["consensus"], core["raw_c1"]
    result = {}

    # published raw K6 labels (bit-exactly reproduced by audit_pipeline)
    seq = build_anchor_signatures(geom)
    shaped = normalize_trajectory_shape(seq)
    dist = aligned_landmark_distances(shaped, SEED)
    emb = prepare_embedding(dist, SEED)
    raw_k6 = canonicalize(ward_labels(emb, 6), score)
    assert int((raw_k6 == 1).sum()) == 252

    # ---------------------------------------------------------------- 1
    # Surrogate partitions built ONLY from (task, length). No routing at all.
    at_cap = length >= np.array([STEP_CAP[t] for t in task])
    tasks_u = sorted(set(task))
    tid = np.array([tasks_u.index(t) for t in task])
    surrogate_task_cap = tid * 2 + at_cap.astype(int)     # task x {at-cap, not}
    # collapse to 6 groups (the K used by the published fit) by merging the
    # 4 empty/at-cap-free cells
    keys = sorted(set(surrogate_task_cap.tolist()))
    surrogate6 = np.array([keys.index(v) for v in surrogate_task_cap])
    # a length-only surrogate: per-task median split
    med = {t: np.median(length[task == t]) for t in tasks_u}
    surrogate_len = tid * 2 + np.array([length[i] > med[task[i]] for i in range(len(task))]).astype(int)

    result["surrogate"] = dict(
        n_groups_task_cap=int(len(keys)),
        ari_rawK6_vs_task_and_atcap=round(float(adjusted_rand_score(raw_k6, surrogate6)), 4),
        ari_rawK6_vs_task_only=round(float(adjusted_rand_score(raw_k6, tid)), 4),
        ari_rawK6_vs_task_and_medsplit=round(float(adjusted_rand_score(raw_k6, surrogate_len)), 4),
        ari_rawK6_vs_atcap_only=round(float(adjusted_rand_score(raw_k6, at_cap.astype(int))), 4),
        c1_vs_atcap=dict(
            jaccard=round(float((raw_c1 & at_cap).sum() / (raw_c1 | at_cap).sum()), 4),
            atcap_predicts_c1_precision=round(float((raw_c1 & at_cap).sum() / at_cap.sum()), 4),
            atcap_predicts_c1_recall=round(float((raw_c1 & at_cap).sum() / raw_c1.sum()), 4),
        ),
        consensus_vs_atcap=dict(
            jaccard=round(float((consensus & at_cap).sum() / (consensus | at_cap).sum()), 4),
        ),
    )

    # explained variance of the 17-d clustered embedding by (task, length)
    onehot = np.eye(len(tasks_u))[tid]
    for nm, X in (("task_only", onehot),
                  ("task+L+L2", np.c_[onehot, length, length ** 2]),
                  ("task+atcap", np.c_[onehot, at_cap.astype(float)]),
                  ("task+L+L2+atcap", np.c_[onehot, length, length ** 2, at_cap.astype(float)])):
        model = LinearRegression().fit(X, emb)
        pred = model.predict(X)
        ss_res = ((emb - pred) ** 2).sum(axis=0)
        ss_tot = ((emb - emb.mean(axis=0)) ** 2).sum(axis=0)
        result.setdefault("embedding_r2", {})[nm] = dict(
            variance_weighted_r2=round(float(1 - ss_res.sum() / ss_tot.sum()), 4),
            per_pc_r2=[round(float(1 - a / b), 3) for a, b in zip(ss_res[:6], ss_tot[:6])],
        )

    # ---------------------------------------------------------------- 2
    # Conditional permutation of outcome within (task, episode_length).
    strata = np.array([f"{t}|{L}" for t, L in zip(task, length)])

    def conditional_test(mask, strat, perms, rng):
        """Permute outcome within strata; exact conditional moments + p-value."""
        groups = [np.flatnonzero(strat == s) for s in np.unique(strat)]
        obs = int((mask & failure).sum())
        exp = var = 0.0
        for g in groups:
            N, K, n = len(g), int(failure[g].sum()), int(mask[g].sum())
            exp += n * K / N
            if N > 1:
                var += n * (K / N) * (1 - K / N) * (N - n) / (N - 1)
        null = np.zeros(perms, dtype=np.int64)
        for g in groups:
            k = int(mask[g].sum())
            f_g = failure[g]
            if k == 0 or k == len(g):
                null += int(f_g.sum()) if k else 0
                continue
            # draw k without replacement from this stratum, perms times
            keys = rng.random((perms, len(g)))
            pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
            null += f_g[pick].sum(axis=1)
        return dict(observed_failures_in_set=obs,
                    conditional_expected=round(float(exp), 2),
                    conditional_sd=round(float(np.sqrt(var)), 3),
                    z=(round(float((obs - exp) / np.sqrt(var)), 3) if var > 1e-12 else None),
                    permutation_mean=round(float(null.mean()), 2),
                    permutation_p_ge=round(float(((null >= obs).sum() + 1) / (perms + 1)), 4),
                    n_strata=int(len(groups)))

    rng = np.random.default_rng(SEED)
    cond = {}
    core_masks = np.load(OUT / "core_membership.npz", allow_pickle=False)
    for nm, mask in (("consensus", consensus), ("raw_c1", raw_c1),
                     ("event_c0", core_masks["event_c0"]),
                     ("lag_c7", core_masks["lag_c7"]), ("at_cap", at_cap)):
        cond[nm] = conditional_test(mask, strata, 20000, rng)
    result["conditional_permutation_given_task_and_length"] = cond

    # stratum purity: how many strata are informative at all?
    rows = []
    for s in np.unique(strata):
        idx = strata == s
        nf, ns = int(failure[idx].sum()), int((~failure[idx]).sum())
        if nf and ns:
            rows.append(dict(stratum=s.split("/")[-1], n=int(idx.sum()),
                             failure=nf, success=ns))
    result["mixed_strata_task_x_length"] = dict(
        count=len(rows),
        total_episodes_in_mixed_strata=int(sum(r["n"] for r in rows)),
        detail=rows,
    )

    # ---------------------------------------------------------------- 3
    # Within-stratum re-clustering + power ceiling.
    within = {}
    for nm, mask in (
        ("scene8_len52",
         (task == "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove") & (length == 52)),
        ("all_tasks_at_cap", at_cap),
        ("scene8_len47plus",
         (task == "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove") & (length >= 47)),
        ("scene8_len45plus",
         (task == "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove") & (length >= 45)),
    ):
        idx = np.flatnonzero(mask)
        nf, ns = int(failure[idx].sum()), int((~failure[idx]).sum())
        entry = dict(n=len(idx), failure=nf, success=ns)
        if ns == 0 or nf == 0:
            entry["verdict"] = "degenerate stratum: no contrast, zero power"
            within[nm] = entry
            continue
        # power ceiling: best attainable one-sided Fisher p with these margins
        best_p = 1.0
        for m in range(1, len(idx)):
            best_p = min(best_p, float(hypergeom.sf(min(m, nf) - 1, len(idx), nf, m)))
        entry["min_attainable_one_sided_p"] = round(best_p, 6)
        sub_seq = build_anchor_signatures(geom[idx])
        sub_shaped = normalize_trajectory_shape(sub_seq)
        sub_dist = aligned_landmark_distances(sub_shaped, SEED)
        sub_emb = prepare_embedding(sub_dist, SEED)
        blocks = {}
        for k in (2, 3, 4, 6):
            if k >= len(idx):
                continue
            lab = canonicalize(ward_labels(sub_emb, k), score[idx])
            bt = block_table(lab, failure[idx])
            best = max(bt, key=lambda r: r["f1"])
            # Fisher p for that block
            m, x = best["n"], best["failure"]
            p = float(hypergeom.sf(x - 1, len(idx), nf, m))
            blocks[str(k)] = dict(best_block=best, fisher_p_one_sided=round(p, 5),
                                  base_rate=round(nf / len(idx), 4))
        entry["reclustered"] = blocks
        within[nm] = entry
    result["within_stratum"] = within

    # ---------------------------------------------------------------- 3b
    # Within (task, init_state): is core membership associated with outcome
    # beyond length?  Both raw and conditional-on-length versions.
    ti = np.array([f"{t}|{i}" for t, i in zip(task, init)])
    til = np.array([f"{t}|{i}|{L}" for t, i, L in zip(task, init, length)])
    rng2 = np.random.default_rng(SEED + 7)
    for nm, strat in (("task_init", ti), ("task_init_length", til)):
        entry = {}
        for mn, mask in (("consensus", consensus), ("raw_c1", raw_c1)):
            entry[mn] = conditional_test(mask, strat, 5000, rng2)
        mixed = [s for s in np.unique(strat)
                 if 0 < failure[strat == s].sum() < (strat == s).sum()]
        entry["mixed_strata"] = len(mixed)
        entry["episodes_in_mixed_strata"] = int(sum((strat == s).sum() for s in mixed))
        result.setdefault("conditional_permutation_other_strata", {})[nm] = entry

    (OUT / "stratified.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False)[:8000])


if __name__ == "__main__":
    main()
