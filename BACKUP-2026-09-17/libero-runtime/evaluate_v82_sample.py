"""Evaluate the unchanged v8.2 alarm on the complete saved 60-episode sample."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import zarr

sys.path.insert(0, "/data/coding/v8-methods")
from test_real_signals import (
    REPO, ROUTES, PolicyClient, finite, infer_checked, load_episode_trace,
    request_from_trace, save_json, save_wire, sha256_file,
)

sys.path.insert(0, str(REPO / "moe-trap-control"))
from collection_protocol import verify_frozen_alarm
from v82_closed_loop import V82Monitor, thresholds_at
from v8_closed_loop import SIGNALS, limits, score_status
from himoe_libero_bridge.batch import wilson_interval

COHORT = Path("/data/libero-runtime/samples/report-20260914/episodes.csv")
REQUIRED = ("episode_id", "control_step", "hb_router_probs", "hb_expert_ids", "hb_selected_prob")
FIRST_ID = -1460914000
CONTROL_ID = -1460914999


def frozen_sources():
    result = verify_frozen_alarm()
    for relative in (
        "moe-trap-control/v82_closed_loop.py",
        "moe-trap-control/v8_feature_control.py",
        "moe-trap-control/v8_closed_loop.py",
        "moe-trap-control/collection_protocol.py",
        "moe-trap-control/design/frozen_alarm_comparison_20260908/profiles/parameters.json",
    ):
        result[relative] = sha256_file(REPO / relative)
    return result


def durable_store():
    group = zarr.open_group(str(ROUTES), mode="r")
    durable = min(int(group.attrs["durable_rows"]), *(group[key].shape[0] for key in REQUIRED))
    return group, durable, np.asarray(group["episode_id"][:durable])


def collect(root):
    root.mkdir(parents=True, exist_ok=False)
    with COHORT.open(newline="") as stream:
        cohort = list(csv.DictReader(stream))
    cohort.sort(key=lambda row: (int(row["base_task_id"]), int(row["init_state_id"]), row["benchmark"]))
    if len(cohort) != 60 or sum(row["outcome"] == "succeeded" for row in cohort) != 9:
        raise ValueError("Expected the original complete cohort: 51 failures and 9 successes")
    if len({row["artifact_dir"] for row in cohort}) != len(cohort):
        raise ValueError("Duplicate source episodes")
    group, durable, disk_ids = durable_store()
    assigned = {FIRST_ID - i for i in range(len(cohort))} | {CONTROL_ID}
    if assigned.intersection(np.asarray(group["episode_id"][:]).tolist()):
        raise ValueError("Diagnostic IDs already exist; use a new collection identity")
    thresholds, _ = limits()
    plan = dict(
        schema="local.frozen_v82.benchmark_sample.v1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cohort_csv=str(COHORT), cohort_sha256=sha256_file(COHORT),
        source_commit="5da830a37230f4ecd761f70be4999f4b3a01687e",
        source_sha256=frozen_sources(), frozen_parameters_unchanged=True,
        mode="Exact observation/noise inference replay, followed by prefix-causal shadow scoring",
        thresholds_v8=dict(zip(SIGNALS, thresholds.tolist())),
        v82_new_head_slope_per_query=-0.0015,
        source_store=str(ROUTES), durable_rows_before=durable,
        probability_shape_per_query=[8, 10, 11, 32], probability_dtype="float16",
        episode_alarm_rule="Any latched v8.2 alarm before the source episode ends",
        positive_label="Source episode completed without achieving the task",
        new_environment_actions=0, episodes=[],
    )
    for i, row in enumerate(cohort):
        name = "{benchmark}-task{task:02d}-init{init:03d}".format(
            benchmark=row["benchmark"], task=int(row["base_task_id"]), init=int(row["init_state_id"]))
        manifest, arrays = load_episode_trace(Path(row["artifact_dir"]))
        if not manifest["result"]["trace_complete"] or manifest["result"]["status"] != "completed":
            raise ValueError("Incomplete source trace: " + name)
        if bool(manifest["result"]["success"]) != (row["outcome"] == "succeeded"):
            raise ValueError("Source outcome differs from the cohort")
        if len(arrays["images"]) != int(row["inference_calls"]):
            raise ValueError("Source query count differs from the cohort")
        plan["episodes"].append(dict(row, name=name, episode_id=FIRST_ID-i,
                                     source_trace_sha256=manifest["array_file_sha256"],
                                     policy_identity=manifest["policy_identity"]))
    save_json(root / "plan.json", plan)
    started, completed_calls = time.perf_counter(), 0
    with PolicyClient("127.0.0.1", 9500, inference_timeout=180) as client:
        if client.metadata.get("episode_id_key") != "episode_id":
            raise ValueError("Server does not advertise the required identity field")
        for episode in plan["episodes"]:
            for key, expected in episode["policy_identity"].items():
                if client.metadata.get(key) != expected:
                    raise ValueError("Source/server policy identity mismatch: " + key)
        save_json(root / "server.json", client.metadata)
        for index, episode in enumerate(plan["episodes"]):
            out = root / episode["name"]
            (out / "wire").mkdir(parents=True, exist_ok=False)
            manifest, arrays = load_episode_trace(Path(episode["artifact_dir"]))
            records = []
            with (out / "calls.jsonl").open("x") as stream:
                for q in range(len(arrays["images"])):
                    response, latency = infer_checked(client, request_from_trace(
                        manifest, arrays, q, episode["episode_id"], True))
                    save_wire(out / "wire" / ("query-%03d.npz" % q), response)
                    exact = bool(np.array_equal(response["actions"], arrays["predicted_actions"][q]))
                    record = dict(query=q, latency_ms=latency, actions_exact=exact,
                                  max_action_error=float(np.max(np.abs(response["actions"]-arrays["predicted_actions"][q]))))
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    records.append(record)
                    completed_calls += 1
                    if not exact:
                        raise ValueError("Replay differs from source actions: %s q%d" % (episode["name"], q))
                    if (q+1) % 10 == 0:
                        print(json.dumps(dict(event="queries", episode=episode["name"], queries=q+1,
                                              total_queries=completed_calls, latency_ms=latency)), flush=True)
            save_json(out / "collection.json", dict(
                kind="fixed_observation_and_noise_inference_replay",
                artifact_dir=episode["artifact_dir"], episode_id=episode["episode_id"],
                source_trace_sha256=manifest["array_file_sha256"], inference_calls=len(records),
                all_actions_exact=all(row["actions_exact"] for row in records),
                max_action_error=max(row["max_action_error"] for row in records), new_environment_actions=0))
            print(json.dumps(dict(event="episode_complete", episode=episode["name"], episodes=index+1,
                                  total_queries=completed_calls, elapsed_seconds=time.perf_counter()-started)), flush=True)

        # Separate snapshot pairs flush primary rows through the legacy 32-call buffer.
        controls = root / "capture-controls"
        controls.mkdir(exist_ok=False)
        pairs = []
        for episode in plan["episodes"][:2]:
            manifest, arrays = load_episode_trace(Path(episode["artifact_dir"]))
            for q in np.linspace(0, len(arrays["images"])-1, 8, dtype=int):
                q = int(q)
                on, on_ms = infer_checked(client, request_from_trace(manifest, arrays, q, CONTROL_ID, True))
                off, off_ms = infer_checked(client, request_from_trace(manifest, arrays, q, CONTROL_ID, False))
                if "routing/expert_ids" in off:
                    raise ValueError("Capture-off response unexpectedly contains routing")
                exact = bool(np.array_equal(on["actions"], off["actions"]) and
                             np.array_equal(on["actions"], arrays["predicted_actions"][q]))
                if not exact:
                    raise ValueError("Capture on/off controls differ from source actions")
                np.savez_compressed(controls / ("%s-query-%03d.npz" % (episode["name"], q)),
                                    actions_on=on["actions"], actions_off=off["actions"])
                pairs.append(dict(episode=episode["name"], query=q, actions_exact=True,
                                  capture_on_latency_ms=on_ms, capture_off_latency_ms=off_ms))
        save_json(controls / "summary.json", dict(episode_id=CONTROL_ID, pairs=pairs,
                  inference_calls=2*len(pairs), all_pair_actions_exact=True,
                  purpose="Independent snapshots for capture nonintervention and recorder flush"))
    if frozen_sources() != plan["source_sha256"]:
        raise ValueError("Frozen detector identity changed during collection")
    save_json(root / "collection-summary.json", dict(
        episodes=len(plan["episodes"]), primary_inference_calls=completed_calls,
        control_inference_calls=2*len(pairs), all_actions_exact=True,
        new_environment_actions=0, elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(event="collection_complete", output=str(root), queries=completed_calls)), flush=True)


def rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                value=numerator/denominator if denominator else None,
                wilson_95=wilson_interval(numerator, denominator))


def metrics(episodes, version):
    counts = Counter()
    detected_steps = []
    for episode in episodes:
        failed = not episode["source_result"]["success"]
        alarmed = episode["first_alarm_query_zero_based"][version] >= 0
        counts[(failed, alarmed)] += 1
        if failed and alarmed:
            detected_steps.append(episode["first_alarm_action_step_before"][version])
    tp, fn = counts[True, True], counts[True, False]
    fp, tn = counts[False, True], counts[False, False]
    return dict(episodes=len(episodes), tp=tp, fn=fn, fp=fp, tn=tn,
                recall=rate(tp, tp+fn), false_positive_rate=rate(fp, fp+tn),
                precision=rate(tp, tp+fp), specificity=rate(tn, tn+fp),
                accuracy=rate(tp+tn, len(episodes)),
                median_first_alarm_action_step_on_detected_failures=(
                    float(np.median(detected_steps)) if detected_steps else None))


def score_episode(root, episode, group, durable, indices):
    out = root / episode["name"]
    collection = json.loads((out / "collection.json").read_text())
    manifest, trace = load_episode_trace(Path(episode["artifact_dir"]))
    count = len(trace["images"])
    if (len(indices) != count or count != collection["inference_calls"] or
            manifest["array_file_sha256"] != episode["source_trace_sha256"]):
        raise ValueError("Source or replay identity mismatch")
    controls = np.asarray([group["control_step"][int(i)] for i in indices], dtype=np.int32)
    if not np.all(np.diff(controls) > 0):
        raise ValueError("Non-monotonic inference call order")
    probabilities = np.stack([group["hb_router_probs"][int(i)] for i in indices])
    stored_ids = np.stack([group["hb_expert_ids"][int(i)] for i in indices])
    selected = np.stack([group["hb_selected_prob"][int(i)] for i in indices])
    if probabilities.shape != (count, 8, 10, 11, 32) or probabilities.dtype != np.float16:
        raise ValueError("Incorrect full HB probability format")
    p = probabilities.astype(np.float32)
    sum_error = float(np.max(np.abs(p.sum(-1)-1)))
    if not np.isfinite(p).all() or np.any(p < 0) or sum_error > 0.003:
        raise ValueError("Invalid full HB probabilities")
    if not np.array_equal(np.take_along_axis(probabilities, stored_ids.astype(np.int64), -1), selected):
        raise ValueError("Selected probabilities differ from a full-probability gather")
    weight_error = 0.0
    for q in range(count):
        with np.load(out / "wire" / ("query-%03d.npz" % q), allow_pickle=False) as wire:
            if not np.array_equal(wire["actions"], trace["predicted_actions"][q]):
                raise ValueError("Wire actions differ from source trace")
            if not np.array_equal(wire["expert_ids"], stored_ids[q, :, :, 1:, :].transpose(1, 0, 2, 3)):
                raise ValueError("Wire/disk expert identity mismatch")
            if not np.array_equal(wire["layer_indices"], [2, 3, 4, 5, 12, 13, 14, 15]):
                raise ValueError("Incorrect HB layer order")
            sel = selected[q, :, :, 1:, :].astype(np.float32).transpose(1, 0, 2, 3)
            delta = float(np.max(np.abs(sel/sel.sum(-1, keepdims=True)-wire["expert_weights"])))
            if not np.isfinite(delta) or delta > 0.001:
                raise ValueError("Wire/disk selected weights differ beyond float16 quantization")
            weight_error = max(weight_error, delta)
    route_file = out / "full-hb-routes.npz"
    np.savez_compressed(route_file, hb_router_probs=probabilities, hb_expert_ids=stored_ids,
                        hb_selected_prob=selected, disk_row_indices=indices, control_steps=controls,
                        episode_id=np.full(count, episode["episode_id"], dtype=np.int32))
    thresholds, _ = limits()
    monitor, rows, latencies = V82Monitor(), [], []
    for q, probability in enumerate(probabilities):
        started = time.perf_counter()
        status = monitor.update(probability)
        latencies.append((time.perf_counter()-started)*1000)
        scores = score_status(status)
        current = thresholds_at("iid_v82", q, thresholds)
        row = dict(episode=episode["name"], benchmark=episode["benchmark"],
                   episode_id=episode["episode_id"], query=q,
                   action_step_before=int(trace["replan_start_steps"][q]),
                   executed_actions=int(trace["executed_lengths"][q]),
                   disk_row=int(indices[q]), control_step=int(controls[q]),
                   v7_alarm=bool(status["alarm"]), v8_alarm=bool(status["v8_alarm"]),
                   v82_alarm=bool(status["v82_alarm"]),
                   v7_first_query=int(status["first_alarm_query"]),
                   v8_first_query=int(status["v8_first"]), v82_first_query=int(status["v82_first"]),
                   monitor_latency_ms=latencies[-1])
        for i, signal in enumerate(SIGNALS):
            row[signal+"_score"] = finite(scores[i])
            row[signal+"_threshold_v8"] = float(thresholds[i])
            row[signal+"_threshold_v82"] = float(current[i])
        rows.append(row)
    with (out / "signals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first = dict(v7=int(monitor.v7.first_alarm_query), v8=int(monitor.first_alarm),
                 v82=int(monitor.first_v82_alarm))
    first_steps = {version: int(trace["replan_start_steps"][q]) if q >= 0 else None
                   for version, q in first.items()}
    q = first["v82"]
    reasons = []
    if q >= 0:
        if monitor.v7.first_freeze_query == q:
            reasons.append("freeze")
        if monitor.v7.first_turbulence_query == q:
            reasons.append("turbulence")
        reasons.extend(SIGNALS[3+i] for i, head_query in enumerate(monitor.v82_first) if head_query == q)
    success = bool(manifest["result"]["success"])
    classification = ("FP" if success else "TP") if q >= 0 else ("TN" if success else "FN")
    result = dict(
        name=episode["name"], benchmark=episode["benchmark"], episode_id=episode["episode_id"],
        base_task_id=int(episode["base_task_id"]), init_state_id=int(episode["init_state_id"]),
        difficulty=int(episode["difficulty"]) if episode["difficulty"] else None,
        source_artifact_dir=episode["artifact_dir"], source_result=manifest["result"],
        source_trace_sha256=manifest["array_file_sha256"], policy_identity=manifest["policy_identity"],
        task_suite=manifest["config"]["task_suite"], task_id=manifest["config"]["task_id"],
        prompt=manifest["prompt"], queries=count, action_steps=int(episode["action_steps"]),
        classification_v82=classification, first_alarm_query_zero_based=first,
        first_alarm_action_step_before=first_steps, v82_first_alarm_heads=reasons,
        v82_actions_remaining_to_endpoint=int(episode["action_steps"])-first_steps["v82"] if q >= 0 else None,
        endpoint_note="Distance to episode endpoint, not annotated failure onset or demonstrated recovery time",
        first_head_query=dict(freeze=int(monitor.v7.first_freeze_query),
                              turbulence=int(monitor.v7.first_turbulence_query),
                              v82_inversion=int(monitor.v82_first[0]), v82_curvature=int(monitor.v82_first[1])),
        integrity=dict(all_full_hb_rows_present=True, wire_disk_expert_ids_exact=True,
                       actions_exact_vs_trace=True, max_action_error=0.0,
                       probability_sum_max_error=sum_error, selected_probability_gather_exact=True,
                       selected_weight_max_quantization_error=weight_error,
                       durable_rows_at_read=durable, full_hb_sha256=sha256_file(route_file)),
        monitor_latency_ms=dict(median=float(np.median(latencies)), p95=float(np.quantile(latencies, .95))),
    )
    save_json(out / "signal-summary.json", result)
    print(json.dumps(dict(event="scored", name=episode["name"], classification=classification,
                          first_alarm_query=first, heads=reasons)), flush=True)
    return result


def analyze(root):
    plan = json.loads((root / "plan.json").read_text())
    if frozen_sources() != plan["source_sha256"] or sha256_file(COHORT) != plan["cohort_sha256"]:
        raise ValueError("Frozen detector or original cohort changed")
    group, durable, disk_ids = durable_store()
    results = []
    for episode in plan["episodes"]:
        out = root / episode["name"]
        summary = out / "signal-summary.json"
        if summary.exists():
            results.append(json.loads(summary.read_text()))
            continue
        if not (out / "collection.json").exists():
            continue
        indices = np.flatnonzero(disk_ids == episode["episode_id"])
        count = int(episode["inference_calls"])
        if len(indices) > count:
            raise ValueError("Duplicate diagnostic inference rows")
        if len(indices) < count:
            continue
        results.append(score_episode(root, episode, group, durable, indices))
    complete = len(results) == len(plan["episodes"]) and (root / "collection-summary.json").exists()
    report = dict(plan, generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  complete=complete, evaluated_episodes=len(results), episodes=results,
                  metrics={version: {benchmark: metrics(
                      [row for row in results if benchmark == "all" or row["benchmark"] == benchmark], version)
                      for benchmark in ("all", "plus", "pro")} for version in ("v7", "v8", "v82")},
                  limitations=[
                      "Camera viewpoint and position-swap sample only; no benchmark-wide guarantee",
                      "Only nine successful source episodes, all from Plus; Pro FPR is undefined",
                      "Related tasks and initial states limit independence; Wilson intervals are approximate",
                      "Source outcomes score episode alarms; no physical failure-onset labels are available",
                      "Prefix-causal offline replay, not online control or demonstrated recovery",
                      "Original thresholds unchanged; no calibration or tuning on this sample",
                      "Full HB probabilities are float16 and legacy recorder buffers 32 calls",
                      "The frozen monitor emits latched alarms, not calibrated failure probabilities",
                  ])
    save_json(root / "summary.json", report)
    all_signals = []
    for episode in results:
        with (root / episode["name"] / "signals.csv").open(newline="") as stream:
            all_signals.extend(csv.DictReader(stream))
    if all_signals:
        with (root / "signals.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_signals[0]))
            writer.writeheader()
            writer.writerows(all_signals)
    fields = ("name", "benchmark", "base_task_id", "init_state_id", "success", "classification_v82",
              "queries", "action_steps", "first_v82_query", "first_v82_action_step", "alarm_heads", "source_artifact_dir")
    with (root / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for episode in results:
            row = {key: episode[key] for key in fields if key in episode}
            row.update(success=episode["source_result"]["success"],
                       first_v82_query=episode["first_alarm_query_zero_based"]["v82"],
                       first_v82_action_step=episode["first_alarm_action_step_before"]["v82"],
                       alarm_heads=";".join(episode["v82_first_alarm_heads"]))
            writer.writerow(row)
    combined = report["metrics"]["v82"]["all"]
    print(json.dumps(dict(event="analysis", complete=complete, evaluated=len(results),
                          v82={key: combined[key] for key in ("tp", "fn", "fp", "tn")})), flush=True)


def verify(root):
    from test_real_signals import verify as verify_signals

    data = json.loads((root / "summary.json").read_text())
    if not data["complete"] or len(data["episodes"]) != 60:
        raise ValueError("Final verification requires all 60 completed episodes")
    verify_signals(root)
    plan = json.loads((root / "plan.json").read_text())
    sources = {episode["name"]: episode for episode in plan["episodes"]}
    if set(sources) != {episode["name"] for episode in data["episodes"]}:
        raise ValueError("Evaluation does not cover the exact frozen cohort")
    disk_rows = []
    for episode in data["episodes"]:
        source = sources[episode["name"]]
        if (source["outcome"] == "succeeded") != episode["source_result"]["success"]:
            raise ValueError("Ground-truth outcome changed")
        with np.load(root / episode["name"] / "full-hb-routes.npz", allow_pickle=False) as archive:
            disk_rows.extend(archive["disk_row_indices"].tolist())
        first = episode["first_alarm_query_zero_based"]
        for base in ("v7", "v8"):
            if first[base] >= 0 and not 0 <= first["v82"] <= first[base]:
                raise ValueError("v8.2 failed to retain a base-monitor alarm")
    if len(disk_rows) != 2891 or len(set(disk_rows)) != len(disk_rows):
        raise ValueError("Source queries are missing or share recorded routing rows")
    with (root / "episodes.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    for benchmark in ("all", "plus", "pro"):
        selected = [row for row in table if benchmark == "all" or row["benchmark"] == benchmark]
        labels = Counter(row["classification_v82"] for row in selected)
        item = data["metrics"]["v82"][benchmark]
        for key in ("tp", "fn", "fp", "tn"):
            if item[key] != labels[key.upper()]:
                raise ValueError("Confusion matrix differs from the per-episode CSV")
    verification = json.loads((root / "verification.json").read_text())
    verification.update(exact_frozen_cohort=True, unique_query_rows=len(disk_rows),
                        episode_csv_confusion_counts_match=True,
                        base_alarms_retained_by_v82=True)
    save_json(root / "verification.json", verification)


def original(root):
    data = json.loads((root / "summary.json").read_text())
    if not data["complete"]:
        raise ValueError("Original-route audit requires the complete sample")
    group = zarr.open_group(data["source_store"], mode="r")
    boundary = data["durable_rows_before"]
    probabilities = np.asarray(group["hb_router_probs"][:boundary])
    episode_ids = np.asarray(group["episode_id"][:boundary])
    controls = np.asarray(group["control_step"][:boundary])
    lookup = defaultdict(list)
    for index in np.flatnonzero(episode_ids == 0):
        lookup[hashlib.sha256(probabilities[index].tobytes()).hexdigest()].append(int(index))
    rows = []
    used = set()
    for episode in data["episodes"]:
        previous = -1
        with np.load(root / episode["name"] / "full-hb-routes.npz", allow_pickle=False) as archive:
            for q, probability in enumerate(archive["hb_router_probs"]):
                digest = hashlib.sha256(probability.tobytes()).hexdigest()
                candidates = lookup[digest]
                if len(candidates) != 1:
                    raise ValueError("Full tensor does not have one unique original match: %s q%d" % (episode["name"], q))
                index = candidates[0]
                if index in used or index <= previous or not np.array_equal(probability, probabilities[index]):
                    raise ValueError("Original full tensor identity/order/uniqueness check failed")
                used.add(index)
                previous = index
                rows.append(dict(episode=episode["name"], query=q, original_disk_row=index,
                                 original_control_step=int(controls[index]),
                                 replay_disk_row=int(archive["disk_row_indices"][q]),
                                 full_probability_sha256=digest))
    if len(rows) != 2891:
        raise ValueError("Incomplete original-tensor comparison")
    mapping = root / "original-row-mapping.csv"
    with mapping.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(episodes=len(data["episodes"]), matched_queries=len(rows),
                  all_full_tensors_exact_and_unique=True, original_order_preserved_per_episode=True,
                  original_store=data["source_store"], original_snapshot_rows=boundary,
                  mapping_sha256=sha256_file(mapping),
                  method="Unique SHA-256 match followed by array equality against original untagged rows captured before replay")
    save_json(root / "original-route-verification.json", result)
    print(json.dumps(dict(event="original_route_verification", **result)), flush=True)


def report(root):
    data = json.loads((root / "summary.json").read_text())
    if not data["complete"]:
        raise ValueError("The final report requires the complete cohort")
    verified = json.loads((root / "verification.json").read_text())
    collected = json.loads((root / "collection-summary.json").read_text())
    controls = json.loads((root / "capture-controls/summary.json").read_text())
    original_audit = json.loads((root / "original-route-verification.json").read_text())
    if not verified["original_unit_tests_passed"] or not controls["all_pair_actions_exact"]:
        raise ValueError("Verification has not passed")

    def fraction(value):
        if value["value"] is None:
            return "N/A"
        return "%d/%d (%.1f%%)" % (value["numerator"], value["denominator"], 100*value["value"])

    all_metrics = data["metrics"]["v82"]["all"]
    lines = [
        "# Frozen V8.2 on the 60-Episode LIBERO Sample", "",
        "The source sample contains 51 failed and 9 successful episodes (85% observed failure).",
        "The unchanged v8.2 monitor detects %d of the 51 failures and alarms on %d of the 9 successes."
        % (all_metrics["tp"], all_metrics["fp"]), "",
        "## Episode Results", "",
        "A positive label is failure to complete the task. An alarm is any latched alarm before the episode ends.", "",
        "| Sample | Detected failures | Missed failures | False alarms on successes | Recall | Precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark, label in (("plus", "Plus: camera viewpoints"), ("pro", "Pro: position swap"), ("all", "Combined")):
        item = data["metrics"]["v82"][benchmark]
        lines.append("| %s | %d | %d | %s | %s | %s |" % (
            label, item["tp"], item["fn"], fraction(item["false_positive_rate"]),
            fraction(item["recall"]), fraction(item["precision"])))
    lines += ["", "Pro has no successful episodes in this sample, so its false-positive rate is undefined.",
              "The combined false-positive rate is based on only nine successes, all from Plus.", ""]
    for name, key in (("Recall", "recall"), ("False-positive rate", "false_positive_rate"), ("Precision", "precision")):
        value = all_metrics[key]
        interval = value["wilson_95"]
        if interval["lower"] is not None:
            lines.append("- %s: %s; approximate 95%% Wilson interval %.1f%% to %.1f%%." % (
                name, fraction(value), 100*interval["lower"], 100*interval["upper"]))
    lines += ["", "Related tasks and initial states limit independence; these intervals are not benchmark-wide guarantees.",
              "An always-alarm rule would have 100% recall, 100% false-positive rate, and 85% precision here.",
              "A high precision alone is therefore insufficient evidence of useful discrimination.", "",
              "## Frozen Version Comparison", "",
              "All three versions use the same complete routing histories; no threshold is fitted to these outcomes.", "",
              "| Monitor | Detected / missed failures | False alarms / successful episodes | Recall | Precision |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for version in ("v7", "v8", "v82"):
        item = data["metrics"][version]["all"]
        lines.append("| %s | %d / %d | %d / %d | %s | %s |" % (
            "v8.2" if version == "v82" else version, item["tp"], item["fn"], item["fp"],
            item["fp"]+item["tn"], fraction(item["recall"]), fraction(item["precision"])))
    detected = [row for row in data["episodes"] if row["classification_v82"] == "TP"]
    first_steps = [row["first_alarm_action_step_before"]["v82"] for row in detected]
    endpoint_steps = [row["v82_actions_remaining_to_endpoint"] for row in detected]
    reasons = Counter(head for row in detected for head in row["v82_first_alarm_heads"])
    lines += ["", "## Alarm Timing", ""]
    if first_steps:
        lines += [
            "On detected failures, the first alarm occurs at median action step %.1f (range %d to %d)."
            % (np.median(first_steps), min(first_steps), max(first_steps)),
            "The median remaining distance to the 520-action timeout is %.1f actions." % np.median(endpoint_steps),
            "First-alarm heads on detected failures: " + ", ".join("%s=%d" % pair for pair in sorted(reasons.items())) + ".",
        ]
        lines += ["", "Descriptive timing breakdown using the known episode timeout:", "",
                  "| Minimum actions remaining before timeout | Failures already alarmed / all failures |",
                  "| ---: | ---: |"]
        for remaining in (0, 50, 100, 200):
            count = sum(value >= remaining for value in endpoint_steps)
            lines.append("| %d | %s |" % (remaining, fraction(rate(count, all_metrics["tp"]+all_metrics["fn"]))))
        lines.append("")
    lines += [
        "Query indices start at zero; each query normally executes ten actions.",
        "These distances are to the episode endpoint. Failure onset has not been annotated, so they do not establish a recovery window.", "",
        "## Method and Integrity", "",
        "All 60 original sample episodes were included: 10 base tasks and 3 initial states per benchmark, using the existing LIBERO-10 checkpoint without further training.",
        "Earlier smoke cases and the native control are excluded from the rates above.",
        "Saved observations, instructions, states, and exact flow noises were replayed through the same policy service.",
        "Every episode received a new diagnostic ID; source queries were matched to complete disk rows and wire captures.", "",
        "- %d/%d primary replay action chunks exactly match the original traces. No simulator actions were added."
        % (collected["primary_inference_calls"], collected["primary_inference_calls"]),
        "- All full HB tensors have shape `[8,10,11,32]`, in float16; top-4 approximations were not used.",
        "- All %d replayed full probability tensors exactly and uniquely match the original pre-replay store, in source-query order."
        % original_audit["matched_queries"],
        "- Wire expert IDs, HB layer order, selected-probability gathers, normalization, trace hashes, and model identity all pass checks.",
        "- All %d independent capture-on/off pairs produce the same actions and match their source requests." % len(controls["pairs"]),
        "- Continuous scores and new-head first alarms match the independent batch implementation for all 60 episodes.",
        "- The original upstream unit tests pass; frozen source and parameter hashes remain unchanged.", "",
        "Each episode starts a fresh `V82Monitor`, updated once per query using only its current and preceding routing tensors.",
        "The two additional v8.2 thresholds move by `-0.0015*q`, with the original two-hit confirmation and warmup.",
        "The model service buffers 32 calls, so this is post-collection causal scoring, not a test of real-time delivery or alarm-driven recovery.",
        "The alarm is latched. Its presence on an eventual success is counted as an episode-level false positive; this does not label any transient physical event.",
        "The frozen detector emits alarms rather than calibrated failure probabilities. Historical calibration concerns remain unchanged.", "",
        "## Artifacts and Reproduction", "",
        "- [First-alarm timeline](alarm-timeline.png) and [PDF](alarm-timeline.pdf)",
        "- [Recall and false-positive comparison](detection-rates.png) and [PDF](detection-rates.pdf)",
        "- [Episode outcomes and alarms](episodes.csv)",
        "- [Every query's scores and thresholds](signals.csv)",
        "- [Structured report](summary.json)",
        "- [Frozen cohort and source identities](plan.json)",
        "- [Verification](verification.json) and [upstream tests](upstream-tests.txt)",
        "- [Original routing tensor verification](original-route-verification.json) and [row mapping](original-row-mapping.csv)",
        "- Each episode directory includes wire captures, a standalone full HB archive, and its signal summary.", "",
        "The following uses saved full HB archives and does not make inference requests:", "",
        "```bash", "/data/venv311/bin/python /data/libero-runtime/evaluate_v82_sample.py verify " + str(root), "```", "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(event="report", path=str(root / "REPORT.md"))), flush=True)


def watch(root):
    while True:
        analyze(root)
        if json.loads((root / "summary.json").read_text())["complete"]:
            verify(root)
            original(root)
            report(root)
            return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "analyze", "verify", "original", "report", "watch"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.output.resolve())


if __name__ == "__main__":
    main()
