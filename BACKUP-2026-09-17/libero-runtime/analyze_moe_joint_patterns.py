"""Reproducible offline joint-pattern experiments on archived HiMoE trajectories."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/data/coding/v8-methods")
sys.path.insert(0, "/data/srv/src")
from moe_joint_patterns import (
    INDEX, JOINT_FEATURE_NAMES, JointRoutingEncoder, MOTIF_NAMES,
    first_crossing, motif_scores, prefix_max, sustained_min,
)
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file

SOURCE = Path("/data/libero-runtime/samples/v82-evaluation-20260914T144322Z")
BRANCHES = Path("/data/coding/moe-control-experiments/runs/p3b-head-control-20260915-s39ow9qi")
DEFAULT_OUT = Path("/data/libero-runtime/samples/moe-joint-patterns-20260915")
THRESHOLD = np.log(1.2)
SEED = 20260915


def read_json(path):
    return json.loads(Path(path).read_text())


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "" if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                         else v for k, v in row.items()} for row in rows)


def count_events(episodes, features, threshold):
    rows = []
    all_names = ("freeze_control",) + MOTIF_NAMES + ("joint_union", "v82_reference", "v82_coverage")
    for e in episodes:
        x = features[e["name"]]
        scores = motif_scores(x)
        first = [first_crossing(sustained_min(-x[:, INDEX["mobility_log_ratio"]]), threshold)]
        first.extend(first_crossing(scores[:, k], threshold) for k in range(len(MOTIF_NAMES)))
        detected = [q for q in first[1:] if q >= 0]
        first.extend((min(detected) if detected else -1,
                      e["first_alarm_query_zero_based"]["v82"], e["coverage_first"]))
        for name, q in zip(all_names, first):
            rows.append(dict(episode=e["name"], benchmark=e["benchmark"], task=e["base_task_id"],
                             success=e["source_result"]["success"], method=name,
                             factor=float(np.exp(threshold)), first_query=q,
                             action_steps_before=10 * q if q >= 0 else -1,
                             remaining_actions_to_endpoint=e["action_steps"] - 10 * q if q >= 0 else None))
    return rows


def event_tables(episodes, event_rows):
    rows = []
    by_name = {e["name"]: e for e in episodes}
    for factor in (1.1, 1.2, 1.3):
        for benchmark in ("all", "plus", "pro"):
            for limit in (18, 26, 36, 51):
                for method in sorted({r["method"] for r in event_rows}):
                    subset = [r for r in event_rows if abs(r["factor"] - factor) < 1e-8
                              and r["method"] == method and (benchmark == "all" or r["benchmark"] == benchmark)]
                    tp = sum(not r["success"] and 0 <= r["first_query"] <= limit for r in subset)
                    fp = sum(r["success"] and 0 <= r["first_query"] <= limit for r in subset)
                    failures = sum(not r["success"] for r in subset)
                    successes = len(subset) - failures
                    observed_success = sum(r["success"] and by_name[r["episode"]]["queries"] > limit for r in subset)
                    rows.append(dict(factor=factor, benchmark=benchmark, last_query=limit, method=method,
                                     detected_failures=tp, failures=failures, flagged_successes=fp, successes=successes,
                                     successes_still_observed_at_query=observed_success))
    return rows


def concordance(scores, labels):
    positive, negative = scores[labels], scores[~labels]
    difference = positive[:, None] - negative[None, :]
    return float((difference > 1e-12).sum() + .5 * (np.abs(difference) <= 1e-12).sum())


def exact_matched_association(episodes, features):
    methods = ("freeze_control",) + MOTIF_NAMES
    x = []
    for e in episodes:
        f = features[e["name"]]
        assert len(f) > 18
        scores = motif_scores(f)
        x.append([prefix_max(sustained_min(-f[:, INDEX["mobility_log_ratio"]]), 18)]
                 + [prefix_max(scores[:, k], 18) for k in range(len(MOTIF_NAMES))])
    x = np.asarray(x)
    assert np.isfinite(x).all()
    labels = np.asarray([not e["source_result"]["success"] for e in episodes])
    groups = []
    for key in sorted({(e["benchmark"], e["base_task_id"]) for e in episodes}):
        ids = np.asarray([i for i, e in enumerate(episodes) if (e["benchmark"], e["base_task_id"]) == key])
        if 0 < labels[ids].sum() < len(ids):
            groups.append(ids)
    pair_count = sum(int(labels[g].sum() * (~labels[g]).sum()) for g in groups)
    rows = []
    for k, name in enumerate(methods):
        observed = sum(concordance(x[g, k], labels[g]) for g in groups) / pair_count
        choices = []
        for g in groups:
            totals = []
            for selected in itertools.combinations(range(len(g)), int(labels[g].sum())):
                permuted = np.zeros(len(g), dtype=bool)
                permuted[list(selected)] = True
                totals.append(concordance(x[g, k], permuted))
            choices.append(totals)
        null = np.asarray([sum(v) / pair_count for v in itertools.product(*choices)])
        rows.append(dict(method=name, last_query=18, mixed_groups=len(groups), matched_pairs=pair_count,
                         concordance=observed, permutations=len(null),
                         p_one_sided=float(np.mean(null >= observed - 1e-12)), p_holm=None))
    primary = sorted(rows[1:], key=lambda r: r["p_one_sided"])
    running = 0.
    for i, row in enumerate(primary):
        running = max(running, min(1., (len(primary) - i) * row["p_one_sided"]))
        row["p_holm"] = running
    return rows


def structure_controls(episodes, features, repetitions=200):
    rng = np.random.default_rng(SEED)
    observed = np.zeros((len(episodes), len(MOTIF_NAMES)), bool)
    nulls = {name: np.zeros((repetitions, len(episodes), len(MOTIF_NAMES)), bool)
             for name in ("row_shuffle", "channel_shift")}
    used = [INDEX[k] for k in ("mobility_log_ratio", "acceleration_log_ratio", "diversity_log_ratio",
                              "state_mobility_log_ratio", "front_path_log_ratio", "path_log_ratio",
                              "route_return_log_ratio")]
    for i, e in enumerate(episodes):
        x = features[e["name"]][:19]
        observed[i] = np.any(motif_scores(x) >= THRESHOLD, axis=0)
        for b in range(repetitions):
            shuffled = x.copy()
            shuffled[7:] = x[7:][rng.permutation(len(x) - 7)]
            nulls["row_shuffle"][b, i] = np.any(motif_scores(shuffled) >= THRESHOLD, axis=0)
            shifted = x.copy()
            for k in used:
                shifted[7:, k] = np.roll(x[7:, k], int(rng.integers(len(x) - 7)))
            nulls["channel_shift"][b, i] = np.any(motif_scores(shifted) >= THRESHOLD, axis=0)
    rows = []
    for outcome in (False, True):
        selected = np.asarray([e["source_result"]["success"] == outcome for e in episodes])
        for k, name in enumerate(MOTIF_NAMES):
            for control, values in nulls.items():
                counts = values[:, selected, k].sum(1)
                rows.append(dict(method=name, success=outcome, control=control,
                                 episodes=int(selected.sum()), observed=int(observed[selected, k].sum()),
                                 null_mean=float(counts.mean()), null_q025=float(np.quantile(counts, .025)),
                                 null_q975=float(np.quantile(counts, .975)), repetitions=repetitions))
    return rows


def external_readouts(arrays):
    states = np.vstack((arrays["states"], arrays["final_state"][None]))
    displacement = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=-1)
    actions = arrays["predicted_actions"]
    change = np.full((len(actions), 3), np.nan)
    for q in range(1, len(actions)):
        difference = actions[q] - actions[q - 1]
        change[q] = [np.sqrt(np.mean(difference[:, :3] ** 2)),
                     np.sqrt(np.mean(difference[:, 3:6] ** 2)),
                     np.mean(np.sign(actions[q, :, 6]) != np.sign(actions[q - 1, :, 6]))]
    return np.column_stack((displacement, change))


def external_relations(episodes, features, external):
    rows = []
    for e in episodes:
        name = e["name"]
        x, ext = features[name], external[name]
        scores = motif_scores(x)
        mobility = x[:, INDEX["mobility_log_ratio"]]
        good = np.isfinite(mobility) & np.isfinite(ext[:, 0])
        rho = spearmanr(mobility[good], ext[good, 0]).statistic if good.sum() > 3 else np.nan
        for k, motif in enumerate(MOTIF_NAMES):
            valid = np.isfinite(scores[:, k])
            active = valid & (scores[:, k] >= THRESHOLD)
            inactive = valid & ~active
            rows.append(dict(episode=name, success=e["source_result"]["success"], method=motif,
                             active_queries=int(active.sum()), inactive_queries=int(inactive.sum()),
                             active_eef_displacement_m=float(np.median(ext[active, 0])) if active.any() else np.nan,
                             inactive_eef_displacement_m=float(np.median(ext[inactive, 0])) if inactive.any() else np.nan,
                             active_translation_action_change=float(np.nanmedian(ext[active, 1])) if active.any() else np.nan,
                             active_back_top4_turnover=float(np.nanmedian(x[active, INDEX["back_top4_turnover"]])) if active.any() else np.nan,
                             within_episode_mobility_eef_spearman=rho))
    return rows


def inspect_branches(episodes, out, source_hashes):
    by_name = {e["name"]: e for e in episodes}
    rows, per_query, pool_rows = [], [], []
    new_features = {}
    original = read_json(BRANCHES / "summary.json")
    branch_hashes = {}
    for pool in original["pools"]:
        name = pool["parent"]
        decisions_path = BRANCHES / name / "decisions.json"
        decisions = read_json(decisions_path)
        q = decisions["query"]
        base = JointRoutingEncoder()
        with np.load(SOURCE / name / "full-hb-routes.npz") as archive:
            prefix = [base.update(p, ids)[0] for p, ids in zip(archive["hb_router_probs"][:q], archive["hb_expert_ids"][:q])]
        pool_path = BRANCHES / name / "pool.npz"
        branch_hashes[str(pool_path)] = sha256_file(pool_path)
        branch_hashes[str(decisions_path)] = sha256_file(decisions_path)
        with np.load(pool_path) as archive:
            pool_hb = archive["hb"]
        for candidate in range(4):
            branch = BRANCHES / name / ("candidate-%d" % candidate)
            result = read_json(branch / "result.json")
            queries = [json.loads(line) for line in (branch / "queries.jsonl").read_text().splitlines()]
            branch_hashes[str(branch / "result.json")] = sha256_file(branch / "result.json")
            branch_hashes[str(branch / "queries.jsonl")] = sha256_file(branch / "queries.jsonl")
            encoder = copy.deepcopy(base)
            feature_rows = list(prefix)
            for item in queries:
                assert item["query"] == len(feature_rows)
                path = BRANCHES / item["archive"]
                if str(path) not in branch_hashes:
                    branch_hashes[str(path)] = sha256_file(path)
                assert branch_hashes[str(path)] == item["archive_sha256"]
                if item["origin"] == "pool":
                    assert item["query"] == q and item["candidate"] == candidate
                    p = pool_hb[candidate]
                else:
                    with np.load(path) as archive:
                        p = archive["hb"]
                feature_rows.append(encoder.update(p)[0])
            x = np.asarray(feature_rows)
            assert len(x) == result["queries"]
            scores = motif_scores(x)
            key = "%s__candidate%d" % (name, candidate)
            new_features[key] = x
            for k, motif in enumerate(MOTIF_NAMES):
                pool_rows.append(dict(parent=name, head=decisions["head"], candidate=candidate, method=motif,
                                      intervention_query=q, score_at_intervention=scores[q, k],
                                      mean_next6_score=float(np.nanmean(scores[q + 1:q + 7, k])),
                                      peak_next6_score=float(np.nanmax(scores[q + 1:q + 7, k])),
                                      active_queries_next6=int((scores[q + 1:q + 7, k] >= THRESHOLD).sum()),
                                      success=result["success"], selected_head_low=candidate == decisions["selected"]["head_low"]))
            for j, item in enumerate(queries):
                query = item["query"]
                per_query.append(dict(parent=name, candidate=candidate, query=query, head=decisions["head"],
                                      target_severity=item["head_severity"],
                                      **{motif: scores[query, k] for k, motif in enumerate(MOTIF_NAMES)}))
            rows.append(dict(parent=name, candidate=candidate, head=decisions["head"], intervention_query=q,
                             source_success=by_name[name]["source_result"]["success"], success=result["success"],
                             action_steps=result["action_steps"], suffix_queries=len(queries),
                             target_at_intervention=decisions["target_severity"][candidate],
                             target_next6_mean=float(np.mean([r["head_severity"] for r in queries[1:7]])),
                             joint_active_before=bool((scores[q - 1] >= THRESHOLD).any()),
                             joint_active_at_intervention=bool((scores[q] >= THRESHOLD).any()),
                             joint_active_any_next6=bool((scores[q + 1:q + 7] >= THRESHOLD).any()),
                             joint_active_any_later=bool((scores[q + 1:] >= THRESHOLD).any()),
                             selected_head_low=candidate == decisions["selected"]["head_low"],
                             selected_head_high=candidate == decisions["selected"]["head_high"]))
        print("Reconstructed four saved branches:", name, flush=True)
    np.savez_compressed(out / "branch-features.npz", **new_features)
    save_csv(out / "branch-results.csv", rows)
    save_csv(out / "branch-motifs.csv", pool_rows)
    save_csv(out / "branch-query-scores.csv", per_query)
    source_hashes.update(branch_hashes)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise SystemExit("Results already exist; choose a new output directory")
    started = time.perf_counter()
    summary = read_json(SOURCE / "summary.json")
    episodes = summary["episodes"]
    coverage_path = Path("/data/libero-runtime/samples/v82-alarm-development-20260915/all-candidate-episodes.csv")
    with coverage_path.open() as handle:
        coverage = {row["episode"]: row for row in csv.DictReader(handle)
                    if row["profile"] == "weak0.80_confirm4_curvature2"}
    source_hashes = {str(path): sha256_file(path) for path in (
        SOURCE / "summary.json", coverage_path, out / "PROTOCOL.md",
        Path(__file__), Path("/data/coding/v8-methods/moe_joint_patterns.py"),
        Path("/data/coding/robot-whisper-0909/safe&vlaconf/moe_trainfree/routing_dynamics/encoder.py"))}
    features, layers, external = {}, {}, {}
    query_rows, timings, prefix_checks = [], [], []
    for e in episodes:
        name = e["name"]
        e["coverage_first"] = int(coverage[name]["first_query"])
        route_path = SOURCE / name / "full-hb-routes.npz"
        source_hashes[str(route_path)] = sha256_file(route_path)
        assert source_hashes[str(route_path)] == e["integrity"]["full_hb_sha256"]
        with np.load(route_path) as archive:
            probabilities, ids = archive["hb_router_probs"], archive["hb_expert_ids"]
        encoder = JointRoutingEncoder()
        values, per_layer = [], []
        for p, expert_ids in zip(probabilities, ids):
            begin = time.perf_counter()
            value, layer = encoder.update(p, expert_ids)
            timings.append(1000 * (time.perf_counter() - begin))
            values.append(value)
            per_layer.append(layer)
        x = np.asarray(values)
        features[name], layers[name] = x, np.asarray(per_layer)
        fresh = JointRoutingEncoder()
        prefix = np.asarray([fresh.update(p, expert_ids)[0] for p, expert_ids in zip(probabilities[:19], ids[:19])])
        np.testing.assert_array_equal(x[:19], prefix)
        np.testing.assert_array_equal(motif_scores(x)[:19], motif_scores(prefix))
        prefix_checks.append(name)
        trace_manifest, arrays = load_episode_trace(Path(e["source_artifact_dir"]))
        assert trace_manifest["array_file_sha256"] == e["source_trace_sha256"]
        source_hashes[str(Path(e["source_artifact_dir"]) / "episode-trace.npz")] = e["source_trace_sha256"]
        assert len(arrays["states"]) == len(x) == e["queries"]
        external[name] = external_readouts(arrays)
        scores = motif_scores(x)
        for q in range(len(x)):
            query_rows.append(dict(episode=name, benchmark=e["benchmark"], task=e["base_task_id"],
                                   success=e["source_result"]["success"], query=q,
                                   **dict(zip(JOINT_FEATURE_NAMES, x[q])),
                                   **{"motif_" + motif: scores[q, k] for k, motif in enumerate(MOTIF_NAMES)},
                                   offline_eef_next_displacement_m=external[name][q, 0],
                                   offline_translation_chunk_difference=external[name][q, 1],
                                   offline_rotation_chunk_difference=external[name][q, 2],
                                   offline_gripper_sign_difference=external[name][q, 3]))
        print("Encoded:", name, len(x), flush=True)
    np.savez_compressed(out / "episode-features.npz", **features)
    np.savez_compressed(out / "episode-layer-features.npz", **layers)
    np.savez_compressed(out / "offline-external.npz", **external)
    save_csv(out / "query-features.csv", query_rows)
    events = []
    for factor in (1.1, 1.2, 1.3):
        events.extend(count_events(episodes, features, np.log(factor)))
    save_csv(out / "episode-events.csv", events)
    tables = event_tables(episodes, events)
    save_csv(out / "event-counts.csv", tables)
    association = exact_matched_association(episodes, features)
    save_csv(out / "matched-association.csv", association)
    print("Running within-episode structure controls", flush=True)
    controls = structure_controls(episodes, features)
    save_csv(out / "structure-controls.csv", controls)
    save_csv(out / "offline-external-relations.csv", external_relations(episodes, features, external))
    branch_rows = inspect_branches(episodes, out, source_hashes)
    save_json(out / "source-hashes.json", source_hashes)
    results = dict(schema="local.moe_joint_pattern_exploration.v1", seed=SEED,
                   episodes=len(episodes), failures=sum(not e["source_result"]["success"] for e in episodes),
                   successes=sum(e["source_result"]["success"] for e in episodes), queries=len(query_rows),
                   feature_names=JOINT_FEATURE_NAMES, motif_names=MOTIF_NAMES, primary_factor=1.2,
                   matched_association=association,
                   primary_event_counts=[r for r in tables if r["factor"] == 1.2 and r["benchmark"] == "all"],
                   primary_full_events=[r for r in events if abs(r["factor"] - 1.2) < 1e-8],
                   prefix_causality_checks_passed=len(prefix_checks),
                   source_hashes=len(source_hashes), saved_branches_reanalyzed=len(branch_rows),
                   independent_failed_branch_parents=len({r["parent"] for r in branch_rows if not r["source_success"]}),
                   new_model_forwards=0, new_environment_actions=0, independent_blind_test=False,
                   extractor_cpu_ms=dict(median=float(np.median(timings)), p95=float(np.quantile(timings, .95)),
                                         total=float(np.sum(timings))),
                   analysis_elapsed_seconds=time.perf_counter() - started)
    save_json(out / "results.json", results)
    print(json.dumps({k: v for k, v in results.items() if k not in (
        "primary_event_counts", "primary_full_events", "matched_association", "feature_names")}, indent=2))


if __name__ == "__main__":
    main()
