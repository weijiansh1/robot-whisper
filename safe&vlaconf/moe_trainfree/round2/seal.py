"""Extract paired observations and seal all outcome-free round-two predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import zarr

from paired_core import (BUDGETS, ROOT, SEED, digest, distance_scores, flow_views,
                         grouped_folds, route_features, threshold, write_json)

HERE = Path(__file__).resolve().parent
ANALYSIS = ROOT / "himoe-route-capture/analysis"
CACHE = ROOT / "VLA_MUI_HUB/cache/HiMoE-VLA"
RICH = ROOT / "analysis_moe_execution_signals"
LONG = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
LAYERS = (0, 3, 4, 7)


def raw_query(task, episodes, query, sources):
    run = CACHE / task / "right-16x32"
    metadata_path = run / "client/server_metadata.json"
    sources.add(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    std = np.asarray(metadata["normalization_action_std"], np.float32)
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    route_episode = np.asarray(store["episode_id"][:])
    first = np.r_[0, np.flatnonzero(np.diff(route_episode) != 0) + 1]
    if not np.array_equal(route_episode[first], episodes):
        raise ValueError("raw route episode order mismatch")
    rows = first + query
    if not np.array_equal(route_episode[rows], episodes):
        raise ValueError("query outside corresponding episode")
    p = np.asarray(store["hb_router_probs"].oindex[rows, list(LAYERS), :, 1:, :], np.float32)
    p = np.transpose(p, (0, 2, 1, 3, 4))
    state, actions, lengths = [], [], []
    for episode in episodes:
        path = run / "client" / f"episode_{int(episode):02d}.npz"
        sources.add(path)
        with np.load(path, allow_pickle=False) as archive:
            states = archive["state"]
            state.append(states[query])
            lengths.append(len(states))
            actions.append(archive["actions"][query])
    return p, np.stack(state), np.stack(actions) / std, np.asarray(lengths), metadata


def first_query(path, sources):
    task = path.stem.replace("__", "/", 1)
    sources.add(path)
    matrices, direct = {}, {}
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["feature_version"]) != 1:
            raise ValueError("unknown compact feature version")
        episodes, init, seeds = archive["episodes"], archive["scenes"], archive["seeds"]
        scalar = archive["scalar"].reshape(len(episodes), 10, 4, 11, 7)[:, :, :, 1:]
        ratio = scalar[..., 5]
        functional = np.stack((ratio / (1 + ratio), scalar[..., 2], scalar[..., 3],
                               1 - scalar[..., 6]), axis=-1).mean(axis=(2, 3))
        for branch in ("routed", "shared", "hidden"):
            values = archive[branch].astype(np.float32).reshape(len(episodes), 10, 4, 11, 64)
            values = values[:, :, :, 1:]
            for flow, view in flow_views(values).items():
                matrices[f"{branch}_{flow}"] = view.reshape(len(episodes), -1)
        for flow, view in flow_views(functional).items():
            matrices[f"function4_{flow}"] = view
            for axis, (name, sign) in enumerate((("authority_low", -1), ("disagreement_low", -1),
                                                ("cancellation_low", -1), ("cosine_high", 1))):
                direct[f"{name}__{flow}"] = sign * view[:, axis]
        validation = json.loads(str(archive["validation_json"]))
    p, state, actions, length, metadata = raw_query(task, episodes, 0, sources)
    route, scalar_route = route_features(p)
    for flow, view in flow_views(route).items():
        matrices[f"route_{flow}"] = view.reshape(len(episodes), -1)
    for name, value in scalar_route.items():
        for flow, view in flow_views(value).items():
            direct[f"{name}__{flow}"] = view
    matrices.update(proprio=state, action=actions.reshape(len(episodes), -1))
    direct["clock"] = np.zeros(len(episodes))
    direct["random"] = np.random.default_rng(SEED).random(len(episodes))
    frame = pd.DataFrame({"task": task, "run_id": "right-16x32", "episode": episodes,
                          "init_state_id": init, "flow_noise_seed": seeds, "query": 0,
                          "length": length, "checkpoint_sha256": metadata["checkpoint_sha256"]})
    for _, rows in frame.groupby("init_state_id").groups.items():
        if not np.array_equal(state[rows], np.broadcast_to(state[rows[0]], state[rows].shape)):
            raise ValueError("q0 physical state differs within initial-state group")
    return frame, matrices, direct, validation


def online_query(q0_frame, sources):
    path = ANALYSIS / "t34-closure/hidden_t34.npz"
    sources.add(path)
    base = q0_frame[q0_frame.task == LONG].reset_index(drop=True)
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["episode"], base.episode):
            raise ValueError("q34 hidden and q0 metadata are not aligned")
        hidden = archive["features"]
    p, state, actions, length, metadata = raw_query(LONG, base.episode.to_numpy(), 34, sources)
    route, route_scalar = route_features(p)
    matrices = {"hidden_descriptor": hidden, "proprio": state,
                "action": actions.reshape(len(base), -1)}
    direct = {"clock": np.full(len(base), 34.0),
              "random": np.random.default_rng(SEED).random(len(base))}
    for flow, view in flow_views(route).items():
        matrices[f"route_{flow}"] = view.reshape(len(base), -1)
    for name, value in route_scalar.items():
        for flow, view in flow_views(value).items():
            direct[f"{name}__{flow}"] = view
    if not np.array_equal(length, base.length) or (length <= 34).any():
        raise ValueError("q34 risk set is incomplete")
    base["query"] = 34
    return base, matrices, direct


def crossfit(frame, matrices, direct, grouping, dataset, output):
    names = [f"{name}__{method}" for name in matrices for method in ("deviation", "knn5")]
    names += list(direct)
    score = np.full((len(frame), len(names)), np.nan)
    alarms = np.zeros((len(BUDGETS), len(frame), len(names)), bool)
    profiles, audits = [], []
    for task, task_rows in frame.groupby("task", sort=True).groups.items():
        task_rows = np.asarray(task_rows)
        for fold, (ref, cal, test) in enumerate(grouped_folds(frame.loc[task_rows, grouping])):
            ref, cal, test = task_rows[ref], task_rows[cal], task_rows[test]
            targets = np.r_[cal, test]
            values = {}
            for name, matrix in matrices.items():
                if matrix.shape[0] != len(frame):
                    raise ValueError("matrix and metadata row mismatch")
                for method, value in distance_scores(matrix[ref], matrix[targets]).items():
                    values[f"{name}__{method}"] = value
            values.update({name: value[targets] for name, value in direct.items()})
            thresholds = {}
            for m, name in enumerate(names):
                calibration, prediction = values[name][:len(cal)], values[name][len(cal):]
                score[test, m] = prediction
                thresholds[name] = []
                for b, budget in enumerate(BUDGETS):
                    limit = threshold(calibration, budget)
                    count = int((calibration > limit).sum())
                    if count > int(np.floor(len(cal) * budget)):
                        raise ValueError("calibration alarm budget exceeded")
                    alarms[b, test, m] = prediction > limit
                    thresholds[name].append(limit)
                    audits.append({"dataset": dataset, "grouping": grouping, "task": task,
                                   "fold": fold, "method": name, "budget": budget,
                                   "reference_size": len(ref), "calibration_size": len(cal),
                                   "calibration_alarms": count, "threshold": limit})
            profiles.append({"task": task, "fold": fold, "reference_rows": ref,
                             "calibration_rows": cal, "test_rows": test, "thresholds": thresholds})
    if not np.isfinite(score).all():
        raise ValueError("incomplete or nonfinite out-of-fold predictions")
    name = f"{dataset}_{grouping}"
    np.savez_compressed(output / f"{name}.npz", scores=score, alarms=alarms,
                        method_names=np.asarray(names), budgets=np.asarray(BUDGETS))
    write_json(output / f"{name}_splits.json", profiles)
    pd.DataFrame(audits).to_csv(output / f"{name}_calibration.csv", index=False)
    print(f"Sealed {name}: {len(frame)} observations, {len(names)} methods", flush=True)


def pilot_data(sources):
    plan_path = RICH / "rich_event_plan.json"
    record_path = RICH / "rich_event_functional_32d/records.jsonl"
    inputs_path = RICH / "rich_event_inputs.npz"
    sources.update((plan_path, record_path, inputs_path, RICH / "rich_event_inputs_audit.json"))
    plan = json.loads(plan_path.read_text())
    branch_by_episode = {p[k]["global_episode"]: (p, p[k]) for p in plan["pairs"] for k in ("event", "control")}
    records = sorted((json.loads(line) for line in record_path.open()), key=lambda r: r["row_id"])
    with np.load(inputs_path, allow_pickle=False) as archive:
        state, action = archive["state"], archive["expected_actions"]
        row_ids = archive["row_id"]
    if not np.array_equal(row_ids, [r["row_id"] for r in records]):
        raise ValueError("pilot row alignment failed")
    direct = {k: [] for k in ("authority_low", "disagreement_low", "cancellation_low", "cosine_high",
                              "route_entropy_low", "route_margin_low", "route_token_collapse",
                              "action_translation_low", "action_rotation_low", "eef_motion_low", "clock")}
    frame, functionality = [], []
    for i, record in enumerate(records):
        path = Path(record["snapshot"])
        if digest(path) != record["snapshot_sha256"] or not record["capture_transparent_bitwise"]:
            raise ValueError("pilot snapshot provenance or transparency failed")
        sources.add(path)
        pair, branch = branch_by_episode[record["global_episode"]]
        query = int(record["query_index"])
        original = ROOT / branch["episode_path"]
        sources.add(original)
        with np.load(original, allow_pickle=False) as archive:
            original_state = archive["state"]
            movement = np.linalg.norm(np.diff(original_state[query - 4:query + 1, :3], axis=0), axis=1).mean()
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["episode_id"]) != record["global_episode"] or int(archive["control_step"]) != query:
                raise ValueError("functional snapshot identity mismatch")
            raw = [float(archive[key][:, -4:, :, 1:].astype(np.float64).mean()) for key in
                   ("routed_authority", "expert_disagreement_ratio", "expert_cancellation", "routed_shared_cosine")]
            logits = archive["router_logits_centered"][:, -4:, :, 1:].astype(np.float64)
            p = np.exp(logits - logits.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            _, routing = route_features(np.transpose(p, (0, 2, 1, 3, 4)))
        functionality.append(raw)
        for name, value in zip(("authority_low", "disagreement_low", "cancellation_low", "cosine_high"),
                               (-raw[0], -raw[1], -raw[2], raw[3])):
            direct[name].append(value)
        for key, value in routing.items():
            direct[key].append(float(value.mean()))
        direct["action_translation_low"].append(-float(np.linalg.norm(action[i, :, :3], axis=1).mean()))
        direct["action_rotation_low"].append(-float(np.linalg.norm(action[i, :, 3:6], axis=1).mean()))
        direct["eef_motion_low"].append(-float(movement))
        direct["clock"].append(query)
        frame.append({"row_id": i, "pair_id": record["pair_id"], "init_state_id": pair["init_state_id"],
                      "lead": record["relative_query"], "query": query, "restore_pass": record["restore_pass"],
                      "task": LONG, "run_id": branch["source_run"], "episode": branch["source_episode"],
                      "static_onset": pair["static_onset_query"],
                      "historical_action_max_abs_error": record["original_action_max_abs_error"]})
    direct = {k: np.asarray(v) for k, v in direct.items()}
    matrices = {"function4": np.asarray(functionality), "proprio": state,
                "action": action.reshape(len(action), -1),
                "routing": np.column_stack([direct[k] for k in
                                             ("route_entropy_low", "route_margin_low", "route_token_collapse")])}
    return pd.DataFrame(frame), matrices, direct


def pilot_scores(frame, matrices, direct, output):
    qualified = frame.restore_pass.to_numpy().copy()
    for _, rows in frame.groupby(["pair_id", "lead"]).groups.items():
        if len(rows) != 2 or not qualified[rows].all():
            qualified[rows] = False
    names = list(direct) + [f"{name}__{m}" for name in matrices for m in ("deviation", "knn5")]
    scores = np.full((len(frame), len(names)), np.nan)
    for name, value in direct.items():
        scores[:, names.index(name)] = value
    for i in np.flatnonzero(qualified):
        ref = qualified & (frame.lead.to_numpy() == frame.iloc[i].lead)
        ref &= frame.init_state_id.to_numpy() != frame.iloc[i].init_state_id
        for name, matrix in matrices.items():
            for method, value in distance_scores(matrix[ref], matrix[i:i+1]).items():
                scores[i, names.index(f"{name}__{method}")] = value[0]
    frame["qualified"] = qualified
    if not np.isfinite(scores[qualified]).all():
        raise ValueError("nonfinite qualified pilot score")
    frame.to_csv(output / "pilot_index.csv", index=False)
    np.savez_compressed(output / "pilot_predictions.npz", scores=scores, method_names=np.asarray(names))
    print(f"Sealed pilot: {qualified.sum()} rows, {len(names)} methods", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round2")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = {HERE / "paired_core.py", Path(__file__), HERE / "PROTOCOL_ZH.md"}
    frames, matrix_parts, direct_parts, validation = [], {}, {}, {}
    for path in sorted((ANALYSIS / "moe-state-impact/compact").glob("*.npz")):
        print(f"Extracting {path.stem}", flush=True)
        frame, matrices, direct, audit = first_query(path, sources)
        frames.append(frame)
        for k, value in matrices.items():
            matrix_parts.setdefault(k, []).append(value)
        for k, value in direct.items():
            direct_parts.setdefault(k, []).append(value)
        validation[frame.task.iloc[0]] = audit
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != 2048 or frame.task.nunique() != 4:
        raise ValueError("unexpected first-query corpus")
    matrices = {k: np.concatenate(v) for k, v in matrix_parts.items()}
    direct = {k: np.concatenate(v) for k, v in direct_parts.items()}
    frame.to_csv(output / "q0_index.csv", index=False)
    np.savez_compressed(output / "q0_inputs.npz", **matrices, **direct)
    for grouping in ("init_state_id", "flow_noise_seed"):
        crossfit(frame, matrices, direct, grouping, "q0", output)
    q34_frame, q34_matrices, q34_direct = online_query(frame, sources)
    q34_frame.to_csv(output / "q34_index.csv", index=False)
    np.savez_compressed(output / "q34_inputs.npz", **q34_matrices, **q34_direct)
    crossfit(q34_frame, q34_matrices, q34_direct, "init_state_id", "q34", output)
    pilot_frame, pilot_matrices, pilot_direct = pilot_data(sources)
    np.savez_compressed(output / "pilot_inputs.npz", **pilot_matrices, **pilot_direct)
    pilot_scores(pilot_frame, pilot_matrices, pilot_direct, output)
    write_json(output / "reconstruction_audit.json", validation)
    write_json(output / "sealed_manifest.json", {
        "schema": "himoe.safe_vlaconf.paired_trainfree.v2", "seed": SEED,
        "budgets": BUDGETS, "model_training": False, "outcome_conditioned_calibration": False,
        "historically_explored_data": True, "prospective_confirmation": False,
        "q0_primary": "routed_d9__knn5", "q0_primary_scalar": "cosine_high__flowmean",
        "sources": {str(p.resolve()): digest(p) for p in sorted(sources)},
        "artifacts": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        "seconds": time.perf_counter() - started,
    })
    print(f"SEALED {output}; {time.perf_counter() - started:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
