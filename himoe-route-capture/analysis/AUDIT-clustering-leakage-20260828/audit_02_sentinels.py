#!/usr/bin/env python3
"""S1-S5 sentinel baselines, evaluated with the SAME downstream convention as
the published "consensus failure core".

S1  episode length >= task step cap        (1-D, no clustering, no routing)
S2  end-phase EEF kinematics                (physical, same pipeline)
S3  action-chunk statistics                 (policy output, same pipeline)
S4  routing features permuted within (task x relative phase)  -> pipeline null
S5  Gaussian dummy of matched shape         -> pipeline null
R   the real routing geometry               (reference; bit-exact reproduction)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from audit_pipeline import (
    BASE, OUT, SEED, CANDIDATE_K,
    load_cache, build_anchor_signatures, run_pipeline, summarize, block_table,
)

STEP_CAP = {  # meta.json max_steps / 10 action sub-steps per query
    "libero_goal/open_the_middle_drawer_of_the_cabinet": 30,
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside": 30,
    "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": 52,
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": 22,
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": 22,
}


def rule_stats(mask, failure, name):
    n = int(mask.sum())
    f = int((mask & failure).sum())
    prec = f / n if n else float("nan")
    rec = f / int(failure.sum())
    f1 = 0.0 if not n or (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return dict(name=name, n=n, failure=f, success=n - f,
                precision=round(prec, 4), failure_recall=round(rec, 4),
                f1=round(f1, 4))


def main() -> None:
    cache = load_cache()
    failure = np.asarray(cache["meta_failure"])
    task = np.asarray(cache["meta_task"])
    length = np.asarray(cache["meta_episode_length"])
    score = np.asarray(cache["diagnostic_mean_soft_speed"])
    result = {}

    # ---- reference: published consensus core -------------------------------
    core = np.load(OUT / "core_membership.npz", allow_pickle=False)
    assert (core["failure"] == failure).all()
    ref = [rule_stats(core["consensus"], failure, "CONSENSUS core (>=2 of 3)"),
           rule_stats(core["union"], failure, "union (>=1 of 3)"),
           rule_stats(core["raw_c1"], failure, "aligned raw C1"),
           rule_stats(core["event_c0"], failure, "event C0"),
           rule_stats(core["lag_c7"], failure, "lag Louvain C7")]
    result["reference"] = ref

    # ---- S1: length >= step cap -------------------------------------------
    caps = np.array([STEP_CAP[t] for t in task])
    s1 = length >= caps
    per_task_cap = {t: int(length[task == t].max()) for t in np.unique(task)}
    s1b = np.array([length[i] >= per_task_cap[task[i]] for i in range(len(task))])
    result["S1"] = [
        rule_stats(s1, failure, "S1 length >= meta.json step cap"),
        rule_stats(s1b, failure, "S1b length == per-task max observed length"),
        rule_stats(length >= np.array([0.9 * STEP_CAP[t] for t in task]), failure,
                   "S1c length >= 0.9 * step cap"),
    ]
    result["S1_detail"] = {
        t: dict(cap=STEP_CAP[t],
                n_at_cap=int(((task == t) & s1).sum()),
                fail_at_cap=int(((task == t) & s1 & failure).sum()),
                succ_at_cap=int(((task == t) & s1 & ~failure).sum()),
                n_fail=int(((task == t) & failure).sum()),
                max_success_len=int(length[(task == t) & ~failure].max()),
                min_success_len=int(length[(task == t) & ~failure].min()))
        for t in np.unique(task)
    }
    # overlap of S1 with the routing core
    for nm, m in (("consensus", core["consensus"]), ("raw_c1", core["raw_c1"])):
        inter = int((s1 & m).sum()); uni = int((s1 | m).sum())
        result.setdefault("S1_overlap", {})[nm] = dict(
            jaccard=round(inter / uni, 4), core_subset_of_S1=bool((m & ~s1).sum() == 0),
            core_members_not_at_cap=int((m & ~s1).sum()))

    # ---- pipelines --------------------------------------------------------
    tapes = np.load(OUT / "sentinel_tapes.npz", allow_pickle=True)
    assert (tapes["episode_length"] == length).all()

    runs = {}
    geom = np.asarray(cache["feature_primary_geometry"])
    seq_routing = build_anchor_signatures(geom)
    runs["R_routing"] = seq_routing
    runs["S2_physical"] = build_anchor_signatures(
        np.asarray(tapes["physical"]).reshape(len(length), -1))
    runs["S3_action"] = build_anchor_signatures(
        np.asarray(tapes["action"]).reshape(len(length), -1))

    rng = np.random.default_rng(SEED)
    runs["S5_gaussian"] = rng.standard_normal(seq_routing.shape).astype(np.float32)

    pipeline = {}
    for name, seq in runs.items():
        labs = run_pipeline(seq, score, SEED)
        rows = {k: summarize(labs[k], failure, task, length) for k in CANDIDATE_K}
        pipeline[name] = dict(
            dims=list(seq.shape),
            k6=rows[6],
            best_over_k=max(
                (rows[k]["best_f1_block"] | {"k": k} for k in CANDIDATE_K),
                key=lambda r: r["f1"]),
            best_precision_over_k=max(
                (rows[k]["best_precision_block_minN"] | {"k": k} for k in CANDIDATE_K),
                key=lambda r: (r["precision"], r["n"])),
            all_k={str(k): {kk: rows[k][kk] for kk in
                            ("ami_outcome", "ami_task", "ami_length",
                             "best_f1_block", "best_precision_block_minN")}
                   for k in CANDIDATE_K},
        )
        np.save(OUT / f"labels_{name}.npy", np.stack([labs[k] for k in CANDIDATE_K]))
        print(name, "k6 best-F1", pipeline[name]["k6"]["best_f1_block"],
              "AMI(out/task/len)", pipeline[name]["k6"]["ami_outcome"],
              pipeline[name]["k6"]["ami_task"], pipeline[name]["k6"]["ami_length"], flush=True)

    # ---- S4: stratified permutation null ----------------------------------
    n_perm = 20
    perm_rows = []
    for r in range(n_perm):
        prng = np.random.default_rng(SEED + 1000 + r)
        permuted = seq_routing.copy()
        for t in np.unique(task):
            idx = np.flatnonzero(task == t)
            for a in range(permuted.shape[1]):
                permuted[idx, a, :] = permuted[prng.permutation(idx), a, :]
        labs = run_pipeline(permuted, score, SEED)
        rows = {k: summarize(labs[k], failure, task, length) for k in CANDIDATE_K}
        best = max((rows[k]["best_f1_block"] | {"k": k} for k in CANDIDATE_K),
                   key=lambda x: x["f1"])
        bestp = max((rows[k]["best_precision_block_minN"] | {"k": k} for k in CANDIDATE_K),
                    key=lambda x: (x["precision"], x["n"]))
        perm_rows.append(dict(rep=r, k6=rows[6]["best_f1_block"],
                              k6_ami_outcome=rows[6]["ami_outcome"],
                              k6_ami_length=rows[6]["ami_length"],
                              best_f1=best, best_precision=bestp))
        print("perm", r, best, flush=True)
    pipeline["S4_permuted"] = dict(
        n_perm=n_perm,
        best_f1_over_reps={
            "max": max(p["best_f1"]["f1"] for p in perm_rows),
            "median": float(np.median([p["best_f1"]["f1"] for p in perm_rows])),
        },
        best_precision_over_reps={
            "max": max(p["best_precision"]["precision"] for p in perm_rows),
            "median": float(np.median([p["best_precision"]["precision"] for p in perm_rows])),
        },
        k6_ami_outcome_median=float(np.median([p["k6_ami_outcome"] for p in perm_rows])),
        k6_ami_length_median=float(np.median([p["k6_ami_length"] for p in perm_rows])),
        reps=perm_rows,
    )
    result["pipeline"] = pipeline
    (OUT / "sentinels.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "pipeline"},
                     indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
