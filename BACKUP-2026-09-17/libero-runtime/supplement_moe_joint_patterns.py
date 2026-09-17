"""Post-analysis duration controls and case summaries; no threshold selection."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/data/coding/v8-methods")
from moe_joint_patterns import INDEX, MOTIF_NAMES, motif_scores, prefix_max
from analyze_moe_joint_patterns import read_json, save_csv, save_json, sha256_file

OUT = Path("/data/libero-runtime/samples/moe-joint-patterns-20260915")
SOURCE = Path("/data/libero-runtime/samples/v82-evaluation-20260914T144322Z")


def main():
    protocol = {
        "status": "Post-analysis descriptive supplement after primary results were observed",
        "reason": "Most joint events occur after q18; successful episodes end before failed episodes",
        "comparison": "Same benchmark/task; truncate both members at each successful member's last actual query",
        "uses_outcomes_for_offline_pairing_only": True,
        "changes_to_online_features_or_thresholds": False,
        "independent_blind_test": False,
        "formal_p_value": False,
        "limitations": ["Pairs share episodes", "Success endpoint is outcome-dependent",
                        "This is an observation-opportunity check, not phase alignment or annotated trap onset"],
    }
    save_json(OUT / "duration-control-protocol.json", protocol)
    episodes = read_json(SOURCE / "summary.json")["episodes"]
    with np.load(OUT / "episode-features.npz") as archive:
        features = {k: archive[k] for k in archive.files}
    results = read_json(OUT / "results.json")
    events = {(r["episode"], r["method"]): r["first_query"] for r in results["primary_full_events"]}
    methods = ("freeze_control",) + MOTIF_NAMES + ("joint_union", "v82_reference", "v82_coverage")
    pairs = []
    for success in episodes:
        if not success["source_result"]["success"]:
            continue
        for failure in episodes:
            if failure["source_result"]["success"] or failure["benchmark"] != success["benchmark"]:
                continue
            if failure["base_task_id"] != success["base_task_id"]:
                continue
            last = success["queries"] - 1
            assert failure["queries"] > last
            for method in methods:
                fq, sq = events[failure["name"], method], events[success["name"], method]
                pairs.append(dict(failed_episode=failure["name"], successful_episode=success["name"],
                                  task=success["base_task_id"], last_query=last,
                                  equal_action_steps_before=last * 10, method=method,
                                  failure_event=bool(0 <= fq <= last), success_event=bool(0 <= sq <= last)))
    save_csv(OUT / "duration-matched-pairs.csv", pairs)
    pair_summary = []
    for method in methods:
        rows = [r for r in pairs if r["method"] == method]
        pair_summary.append(dict(method=method, pairs=len(rows),
                                 independent_failed_episodes=len({r["failed_episode"] for r in rows}),
                                 independent_successful_episodes=len({r["successful_episode"] for r in rows}),
                                 failure_only=sum(r["failure_event"] and not r["success_event"] for r in rows),
                                 success_only=sum(r["success_event"] and not r["failure_event"] for r in rows),
                                 both=sum(r["failure_event"] and r["success_event"] for r in rows),
                                 neither=sum(not r["failure_event"] and not r["success_event"] for r in rows)))
    save_csv(OUT / "duration-matched-summary.csv", pair_summary)
    with (OUT / "offline-external-relations.csv").open() as handle:
        ext = list(csv.DictReader(handle))
    external_summary = []
    for method in MOTIF_NAMES:
        eligible = [r for r in ext if r["method"] == method
                    and int(r["active_queries"]) >= 2 and int(r["inactive_queries"]) >= 2]
        ratios = [float(r["active_eef_displacement_m"]) / max(float(r["inactive_eef_displacement_m"]), 1e-9)
                  for r in eligible]
        external_summary.append(dict(method=method, episodes_with_both_states=len(eligible),
                                     median_within_episode_displacement_ratio=float(np.median(ratios)) if ratios else None,
                                     q25=float(np.quantile(ratios, .25)) if ratios else None,
                                     q75=float(np.quantile(ratios, .75)) if ratios else None))
    detected = [r for r in results["primary_full_events"] if r["method"] == "joint_union" and r["first_query"] >= 0]
    missed = [r for r in results["primary_full_events"] if r["method"] == "joint_union"
              and not r["success"] and r["first_query"] < 0]
    coverage_missed = [r["episode"] for r in results["primary_full_events"] if r["method"] == "v82_coverage"
                       and not r["success"] and r["first_query"] < 0]
    additional = [r for r in detected if r["episode"] in coverage_missed]
    elapsed = [r["action_steps_before"] for r in detected]
    out = dict(protocol=protocol, duration_matched_summary=pair_summary,
               offline_external_summary=external_summary,
               joint_detection_action_steps=dict(median=float(np.median(elapsed)),
                                                 q25=float(np.quantile(elapsed, .25)), q75=float(np.quantile(elapsed, .75))),
               joint_missed_failures=[r["episode"] for r in missed], additional_to_coverage=additional,
               hashes={str(p): sha256_file(p) for p in (Path(__file__), OUT / "results.json", OUT / "episode-features.npz")})
    save_json(OUT / "supplement.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
