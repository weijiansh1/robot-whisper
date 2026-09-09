#!/usr/bin/env python3
"""Test D (strong version) + length-surrogate sentinel S6.

D: run the identical aligned-kernel pipeline on router descriptors sampled at
   FIXED INTEGER ABSOLUTE query indices inside each task's common cohort
   (t <= min episode length in that task, where every rollout is still running).
   Episode length cannot enter the descriptor OR the resampling operator.
   Question: does the consensus core survive?

S6: feed the pipeline a (2560, 10, 32) tensor that is a deterministic function
    of (task, episode_length) plus isotropic noise -- zero routing content --
    and compare the resulting partition to the published raw K6 / C1.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import hypergeom

from audit_pipeline import (
    OUT, SEED, CANDIDATE_K, load_cache, build_anchor_signatures, run_pipeline,
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


def jac(a, b):
    return round(float((a & b).sum() / max((a | b).sum(), 1)), 4)


def main() -> None:
    cache = load_cache()
    failure = np.asarray(cache["meta_failure"])
    task = np.asarray(cache["meta_task"]).astype(str)
    length = np.asarray(cache["meta_episode_length"])
    score = np.asarray(cache["diagnostic_mean_soft_speed"])
    core = np.load(OUT / "core_membership.npz", allow_pickle=False)
    raw_c1, consensus = core["raw_c1"], core["consensus"]
    at_cap = length >= np.array([STEP_CAP[t] for t in task])
    result = {}

    # published raw K6 reference (bit-exact)
    seq_full = build_anchor_signatures(np.asarray(cache["feature_primary_geometry"]))
    labs_full = run_pipeline(seq_full, score, SEED)
    assert int((labs_full[6] == 1).sum()) == 252

    # ------------------------------------------------------------------ D
    tapes = np.load(OUT / "common_prefix_tapes.npz", allow_pickle=True)
    windows = json.loads(str(tapes["windows"]))
    result["common_prefix_windows"] = windows
    d_out = {}
    for window in ("full", "late"):
        mat = np.asarray(tapes[window]).reshape(len(length), -1)
        seq = build_anchor_signatures(mat)
        labs = run_pipeline(seq, score, SEED)
        rows = {k: summarize(labs[k], failure, task, length) for k in CANDIDATE_K}
        best_blocks = {}
        for k in CANDIDATE_K:
            bt = block_table(labs[k], failure)
            for r in bt:
                m = labs[k] == r["block"]
                r["jaccard_vs_published_C1"] = jac(m, raw_c1)
                r["jaccard_vs_consensus"] = jac(m, consensus)
                r["fisher_p"] = round(float(hypergeom.sf(
                    r["failure"] - 1, len(length), int(failure.sum()), r["n"])), 6)
            best_blocks[str(k)] = dict(
                blocks=bt,
                best_f1=max(bt, key=lambda r: r["f1"]),
                best_precision_n100=max([r for r in bt if r["n"] >= 100] or bt,
                                        key=lambda r: (r["precision"], r["n"])),
                ari_vs_published_rawK6=round(float(adjusted_rand_score(labs_full[6], labs[k])), 4),
                ami_outcome=rows[k]["ami_outcome"], ami_task=rows[k]["ami_task"],
                ami_length=rows[k]["ami_length"],
            )
        d_out[window] = best_blocks
        np.save(OUT / f"labels_commonprefix_{window}.npy",
                np.stack([labs[k] for k in CANDIDATE_K]))
        print(window, "K6 best-F1", best_blocks["6"]["best_f1"],
              "ARI vs rawK6", best_blocks["6"]["ari_vs_published_rawK6"],
              "AMI len", best_blocks["6"]["ami_length"], flush=True)

    # within-task version of D: does the prefix separate outcome inside a task?
    per_task = {}
    for t in sorted(set(task)):
        idx = np.flatnonzero(task == t)
        nf = int(failure[idx].sum())
        if nf == 0 or nf == len(idx):
            per_task[t] = dict(n=len(idx), failure=nf, verdict="single-outcome task")
            continue
        mat = np.asarray(tapes["full"])[idx].reshape(len(idx), -1)
        seq = build_anchor_signatures(mat)
        shaped = normalize_trajectory_shape(seq)
        emb = prepare_embedding(aligned_landmark_distances(shaped, SEED), SEED)
        entry = dict(n=len(idx), failure=nf, base_rate=round(nf / len(idx), 4), by_k={})
        for k in (2, 3, 4, 6):
            lab = canonicalize(ward_labels(emb, k), score[idx])
            bt = block_table(lab, failure[idx])
            best = max(bt, key=lambda r: r["f1"])
            best["fisher_p"] = round(float(hypergeom.sf(
                best["failure"] - 1, len(idx), nf, best["n"])), 6)
            entry["by_k"][str(k)] = best
        per_task[t] = entry
        print("prefix within", t.split("/")[-1][:38], entry["by_k"]["2"], flush=True)
    result["common_prefix_within_task"] = per_task
    result["common_prefix_pooled"] = d_out

    # ----------------------------------------------------------------- S6
    # Length-only surrogate: (2560,10,32) built from (task, length) + noise.
    rng = np.random.default_rng(SEED)
    tasks_u = sorted(set(task))
    tid = np.array([tasks_u.index(t) for t in task])
    surrogate = {}
    for noise in (0.0, 0.05, 0.2, 0.5):
        base = np.zeros((len(length), 10, 32), dtype=np.float32)
        Ln = (length - length.mean()) / length.std()
        phase = np.linspace(-1, 1, 10)
        for i in range(len(length)):
            # a smooth 10-anchor curve whose shape depends only on T and task
            base[i] = (np.outer(phase ** 1, np.ones(32)) * Ln[i]
                       + np.outer(phase ** 2, np.ones(32)) * (tid[i] - 2.0)
                       + np.outer(np.sin(3 * phase), np.ones(32)) * (Ln[i] ** 2))
        base += rng.standard_normal(base.shape).astype(np.float32) * noise
        labs = run_pipeline(base, score, SEED)
        bt = block_table(labs[6], failure)
        best = max(bt, key=lambda r: r["f1"])
        surrogate[f"noise_{noise}"] = dict(
            ari_vs_published_rawK6=round(float(adjusted_rand_score(labs_full[6], labs[6])), 4),
            best_block_k6=best,
            jaccard_best_vs_C1=jac(labs[6] == best["block"], raw_c1),
            jaccard_best_vs_consensus=jac(labs[6] == best["block"], consensus),
            ami_outcome=summarize(labs[6], failure, task, length)["ami_outcome"],
        )
        print("surrogate", noise, surrogate[f"noise_{noise}"], flush=True)
    result["length_task_surrogate_S6"] = surrogate

    (OUT / "common_prefix.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
