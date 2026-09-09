#!/usr/bin/env python3
"""Replay original frozen v8.2 on the same audited native Long main cohort."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from collection_protocol import HERE, verify_frozen_alarm
from collection_routes import PROBS_KEY
from collection_storage import atomic_json, atomic_npz, digest, records
from compare_collection_alarms import scalar_v8_first, flow_speed, raw_v8_features
from compare_frozen_alarm_methods import LEGACY, load_profiles, replay_legacy
from v8_feature_control import V8Monitor


def counts(first, failed, cutoff=None):
    fired = first >= 0
    if cutoff is not None:
        fired &= first <= cutoff
    tp, fp = int(np.sum(fired & failed)), int(np.sum(fired & ~failed))
    positives, negatives = int(failed.sum()), int((~failed).sum())
    return dict(tp=tp, fp=fp, fn=positives-tp, tn=negatives-fp,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/positives if positives else None,
        false_positive_rate=fp/negatives if negatives else None)


def run(args):
    started = time.monotonic()
    verify_frozen_alarm()
    source = HERE / "design/frozen_alarm_comparison_20260908"
    parameters, _, _, _ = load_profiles(source / "profiles", source)
    config = parameters["legacy"]
    if config["v82_config"] != dict(baseline=4, width=6, confirm=2, slope=-.0015):
        raise ValueError("Unexpected original v8.2 configuration")
    audited = json.loads(args.audit.read_text())
    plan = json.loads((args.run / "plan.json").read_text())
    if (audited["status"] != "passed" or audited["run"] != str(args.run.resolve()) or
            audited["plan_sha256"] != digest(args.run / "plan.json") or
            plan["benchmark"] != "native_long"):
        raise ValueError("The completed audited original Long cohort is required")
    if args.output.exists():
        raise ValueError("Refusing to overwrite the detector comparison")
    cohort = plan["cohort"]
    by_id = {row["main_id"]: row for row in audited["cohort"]}
    if {t["main_id"] for t in cohort} != set(by_id):
        raise ValueError("Incomplete cohort")
    width = max(t["parent_queries"] for t in cohort)
    cache = dict(valid=np.zeros((len(cohort), width), bool),
        mobility=np.full((len(cohort), width, 8), np.nan, np.float32),
        acceleration=np.full((len(cohort), width), np.nan, np.float32),
        periodicity=np.full((len(cohort), width), np.nan, np.float32))
    raw = np.full((len(cohort), width, 2), np.nan, np.float32)
    scores = np.full((len(cohort), width, 2), np.nan, np.float64)
    checked_queries, rows, input_hashes = 0, [], {}
    for i, task in enumerate(cohort):
        parent = Path(task["parent_directory"])
        for name, key in (("main_complete.json", "parent_commit_sha256"),
                          ("main/manifest.json", "parent_manifest_sha256")):
            if digest(parent / name) != task[key]:
                raise ValueError("Frozen native input changed")
            input_hashes[str(parent / name)] = task[key]
        live = V8Monitor()
        length, final_success = 0, False
        for record in records(parent / "main"):
            q = int(record["query"])
            if q != length or final_success:
                raise ValueError("Invalid native query timeline")
            status = live.update(record[PROBS_KEY])
            np.testing.assert_array_equal(np.array([status[key] for key in
                ("freeze_score", "acceleration_score", "periodicity_score")], np.float32), record["alarm_scores"])
            if status["alarm"] != bool(record["alarm"]):
                raise ValueError("Stored native alarm differs")
            independent_raw = raw_v8_features(flow_speed(record[PROBS_KEY]))
            np.testing.assert_allclose(independent_raw, status["v8_raw"], rtol=2e-5, atol=1e-7)
            cache["valid"][i, q] = True
            cache["mobility"][i, q] = status["layer_mobility"]
            cache["acceleration"][i, q] = status["route_acceleration"]
            cache["periodicity"][i, q] = status["lag_periodicity"]
            raw[i, q] = independent_raw
            scores[i, q] = status["v8_scores"]
            length += 1
            final_success = bool(record["success"])
        if (length != task["parent_queries"] or final_success != task["native_success"] or
                live.first_alarm != by_id[task["main_id"]]["first_v8_alarm"]):
            raise ValueError("Native length, outcome, or previous v8 audit differs")
        rows.append(dict(main_id=task["main_id"], task_index=task["variant"]["registry_index"],
            task_name=task["base_task"], init_index=task["init_index"], queries=length,
            native_success=final_success, v7_online=live.v7.first_alarm_query,
            v8_online=live.first_alarm))
        checked_queries += length
    first = replay_legacy(cache, raw, config)
    np.testing.assert_array_equal(first[0], [r["v7_online"] for r in rows])
    np.testing.assert_array_equal(first[1], [r["v8_online"] for r in rows])
    for i, row in enumerate(rows):
        for version, slope in ((1, 0.), (2, config["v82_config"]["slope"])):
            actual = scalar_v8_first(raw[i, :row["queries"]], int(first[0, i]), config, slope)
            if actual != int(first[version, i]):
                raise ValueError("Scalar confirmation differs from vectorized replay")
        row.update({name: int(first[j, i]) for j, name in enumerate(LEGACY)})
        row["v82_new_alarm"] = row["v82_frozen"] >= 0 and row["v8_frozen"] < 0
        row["v82_advance_queries"] = (row["v8_frozen"]-row["v82_frozen"]
            if row["v8_frozen"] >= 0 and row["v82_frozen"] >= 0 else None)
    fired8 = first[1] >= 0
    if not np.all((first[2, fired8] >= 0) & (first[2, fired8] <= first[1, fired8])):
        raise ValueError("Relaxed v8.2 must contain and never delay frozen v8 alarms")
    for cutoff in (7, 13, 20, 30, 40):
        prefix = {key: value.copy() for key, value in cache.items()}
        prefix_raw = raw.copy()
        for key in prefix:
            prefix[key][:, cutoff+1:] = False if key == "valid" else np.nan
        prefix_raw[:, cutoff+1:] = np.nan
        np.testing.assert_array_equal(replay_legacy(prefix, prefix_raw, config),
            np.where((first >= 0) & (first <= cutoff), first, -1))
    order = np.arange(len(rows))[::-1]
    np.testing.assert_array_equal(replay_legacy({key: value[order] for key, value in cache.items()}, raw[order], config), first[:, order])
    failed = np.array([not row["native_success"] for row in rows])
    lengths = np.array([row["queries"] for row in rows])
    metric = {name: counts(first[j], failed) for j, name in enumerate(LEGACY)}
    effective = {name: counts(np.where(first[j]+1 < lengths, first[j], -1), failed)
        for j, name in enumerate(LEGACY)}
    changed = [r for r in rows if r["v82_frozen"] != r["v8_frozen"]]
    sources = [Path(__file__).resolve(), HERE / "compare_collection_alarms.py",
        HERE / "compare_frozen_alarm_methods.py", HERE / "v8_feature_control.py",
        HERE.parent / "safe&vlaconf/moe_trainfree/temporal_fusion/fusion.py",
        HERE.parent / "safe&vlaconf/moe_trainfree/v82_validation/run_analysis.py",
        HERE.parent / "safe&vlaconf/moe_trainfree/v82_validation/monitor.py"]
    result = dict(status="passed", benchmark="original_libero_long", mains=len(rows),
        audited_native_queries=checked_queries, audit=str(args.audit.resolve()), audit_sha256=digest(args.audit),
        plan_sha256=digest(args.run / "plan.json"), parameters_sha256=digest(source / "profiles/parameters.json"),
        sources={str(p): digest(p) for p in sources}, input_sha256=input_hashes,
        metric_scope="one judgment per complete unintervened main, eventual native failure is the positive label",
        metrics=metric, effective_q_plus_one=effective,
        first_query_cutoffs={str(q): {name: counts(first[j], failed, q) for j, name in enumerate(LEGACY)}
            for q in (13, 20, 30, 40)},
        changes=changed, newly_detected_failures=sum(r["v82_new_alarm"] and not r["native_success"] for r in rows),
        newly_alarmed_successes=sum(r["v82_new_alarm"] and r["native_success"] for r in rows),
        earlier_existing_alarms=sum(r["v82_advance_queries"] is not None and r["v82_advance_queries"] > 0 for r in rows),
        config=config, thresholds="only the two added heads: high-oriented threshold(q) = threshold(0) - 0.0015*q; v7 unchanged",
        verification=dict(native_v7_and_v8_match_previous_audit=True,
            independent_feature_queries=checked_queries, scalar_v8_and_v82_replays=2*len(rows),
            all_mains_prefix_cutoffs=[7, 13, 20, 30, 40], reverse_order_exact=True),
        reference_fit=False, threshold_fit=False, hidden_capture=False, gpu_compute=False,
        actual_model_queries=0, actual_simulator_steps=0, v82_intervention_executed=False,
        notes=["This comparison evaluates detection and trigger timing, not causal recovery under v8.2.",
            "The original slope is frozen globally; query-dependent thresholds do not imply recalibration.",
            "Previous v8 candidate-selection outcomes cannot be relabeled as v8.2 intervention outcomes.",
            "q is the original executed-query index, not a counter incremented by extra candidate evaluations.",
            "Both implementations preserve the original warmup, inclusive comparisons, smoothing and confirmation."])
    args.output.mkdir(parents=True)
    atomic_npz(args.output / "scores.npz", dict(**cache, raw_v8=raw, v8_scores=scores,
        v82_scores=scores-config["v82_config"]["slope"]*np.arange(width)[None, :, None],
        first=first, methods=np.array(LEGACY), main_ids=np.array([r["main_id"] for r in rows])))
    result["scores_sha256"] = digest(args.output / "scores.npz")
    with (args.output / "first_alarms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result.update(first_alarms_sha256=digest(args.output / "first_alarms.csv"), elapsed_seconds=time.monotonic()-started)
    atomic_json(args.output / "summary.json", result)
    print(json.dumps({key: result[key] for key in ("status", "mains", "audited_native_queries", "metrics", "changes",
        "newly_detected_failures", "newly_alarmed_successes", "earlier_existing_alarms", "v82_intervention_executed")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
