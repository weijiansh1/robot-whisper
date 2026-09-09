#!/usr/bin/env python3
"""Paired task outcomes and manipulation checks, keeping native controls explicit."""

import argparse
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, digest
from v8_feature_control import ARMS


def ci(values):
    values = np.asarray(values, float)
    if not len(values) or np.all(values == values[0]):
        return None
    rng = np.random.default_rng(20260908)
    means = values[rng.integers(len(values), size=(20000, len(values)))].mean(-1)
    return (100 * np.quantile(means, [.025, .975])).tolist()


def analyze_group(tasks):
    results = {}
    for arm in ARMS:
        failure, healthy, rescue, harm, paired_wins, paired_losses = 0, 0, 0, 0, 0, 0
        successes, total, native_successes, control_queries, model_queries = 0, 0, 0, 0, 0
        parent_delta, random_delta, rescued_ids, gained_ids, lost_ids = [], [], [], [], []
        manipulation, release = [], []
        for task in tasks:
            by_key = {(b["replicate"], b["arm"]): b for b in task["branches"]}
            selected = [b for b in task["branches"] if b["arm"] == arm]
            deltas, random_deltas = [], []
            for b in selected:
                base = by_key[b["replicate"], "native"]
                rnd = by_key[b["replicate"], "random_half" if arm == "combined_half" else "random"]
                s, n = int(b["success"]), int(base["success"])
                total += 1
                successes += s
                native_successes += n
                failure += int(not task["native_success"])
                healthy += int(task["native_success"])
                rescue += int(not task["native_success"] and s)
                harm += int(task["native_success"] and not s)
                paired_wins += int(s > n)
                paired_losses += int(s < n)
                if not task["native_success"] and s:
                    rescued_ids.append(task["main_id"])
                if s > n:
                    gained_ids.append(task["main_id"])
                if s < n:
                    lost_ids.append(task["main_id"])
                deltas.append(s - n)
                random_deltas.append(s - int(rnd["success"]))
                control_queries += b["controlled_queries"]
                model_queries += b["deployment_model_queries"]
                if b["manipulation_mean"]:
                    manipulation.append(b["manipulation_mean"])
                if b["release_after_seven"] and base["release_after_seven"]:
                    release.append((np.asarray(b["release_after_seven"])[1:] - np.asarray(base["release_after_seven"])[1:]).tolist())
            parent_delta.append(np.mean(deltas))
            random_delta.append(np.mean(random_deltas))
        results[arm] = dict(parents=len(tasks), suffixes=total, successes=successes,
            native_randomized_suffix_successes=native_successes,
            original_failure_suffixes=failure, rescues_from_original_failures=rescue,
            original_success_suffixes=healthy, failures_from_original_successes=harm,
            rescued_parent_count=len(set(rescued_ids)), rescued_main_ids=sorted(set(rescued_ids)),
            paired_wins=paired_wins, paired_losses=paired_losses,
            paired_winning_parent_count=len(set(gained_ids)), paired_losing_parent_count=len(set(lost_ids)),
            paired_winning_main_ids=sorted(set(gained_ids)), paired_losing_main_ids=sorted(set(lost_ids)),
            paired_delta_pp=float(np.mean(parent_delta) * 100) if parent_delta else None,
            paired_parent_bootstrap_95ci_pp=ci(parent_delta),
            delta_vs_matched_random_pp=float(np.mean(random_delta) * 100) if random_delta else None,
            delta_vs_random_parent_bootstrap_95ci_pp=ci(random_delta),
            controlled_queries=control_queries, deployment_model_queries=model_queries,
            manipulation_branch_median=np.median(manipulation, axis=0).tolist() if manipulation else None,
            manipulation_branches=len(manipulation),
            manipulation_direction_fractions=(np.mean(np.asarray(manipulation)[:, :3] * np.asarray([1., -1., 1.]) >
                np.asarray([0., -1., 1.]), axis=0).tolist() if manipulation else None),
            release_q11_paired_median=np.median(release, axis=0).tolist() if release else None,
            release_q11_pairs=len(release))
    return results


def run(args):
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "passed":
        raise ValueError("Analysis requires a complete passed audit")
    tasks = audit["tasks"]
    primary = [t for t in tasks if t["analysis_role"] == "perturbation"]
    result = dict(status="completed", audit=str(args.audit.resolve()), audit_sha256=digest(args.audit),
        analyzer_sha256=digest(__file__), stage="exploratory_mechanism_test",
        outcome="full remaining task success; one resampling unit per original main_id",
        scope="alarm-conditioned Long Pro/Plus states; not a population success-rate estimate or held-out validation",
        multiple_comparisons="8 prespecified arms; intervals descriptive, no winner selected as a validated method",
        baseline="same restored state and paired new random streams, no gate bias",
        all=analyze_group(tasks), primary=analyze_group(primary),
        by_benchmark={name: analyze_group([t for t in primary if t["benchmark"] == name]) for name in ("pro", "plus")},
        by_alarm_head={name: analyze_group([t for t in primary if name in t["active_heads"]])
                       for name in ("freeze", "turbulence", "balance", "curvature")},
        by_alarm_time={"q_le_20": analyze_group([t for t in primary if t["first_v8_alarm"] <= 20]),
                       "q_gt_20": analyze_group([t for t in primary if t["first_v8_alarm"] > 20])},
        subgroup_scope="head and time breakdowns are descriptive exploration, not new triggers or confirmatory tests",
        degenerate_bootstrap="identical parent differences return null intervals; an all-zero bootstrap is not evidence of equivalence",
        manipulation_columns=audit["manipulation_columns"],
        release_columns=["frontback_inversion_delta_vs_native", "curvature_delta_vs_native", "freeze_score_delta_vs_native"],
        release_caveat="q11 after five controlled queries, with six fully unforced cross-query transitions; pairs available in both arms only",
        total_parents=len(tasks), primary_parents=len(primary), branches=audit["branches"],
        c0_queries=audit["c0_queries"], actual_model_queries=audit["actual_model_queries"],
        new_main_coverage=0)
    atomic_json(args.output, result)
    print(json.dumps(dict(primary_parents=len(primary),
        arms={arm: {key: values[key] for key in ("rescues_from_original_failures", "failures_from_original_successes",
            "paired_wins", "paired_losses", "paired_delta_pp", "paired_parent_bootstrap_95ci_pp", "rescued_parent_count")}
              for arm, values in result["primary"].items()})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
