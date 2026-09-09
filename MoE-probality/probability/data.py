"""Read-only source validation and reproducible group-separated cohorts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import DETECTOR_PATH, FEATURE_NAMES, detector, history_features


RUN_A = "right-50x8-20260903"
RUN_B = "right-50x8b-20260903"
CAPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_long": 520}
SPLIT_SEED = 20260907


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def json_records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def split_for(task: str, init_state: int, run_id: str):
    # Stable task-specific permutations avoid dependence on data order or outcomes.
    seed = int(hashlib.sha256(f"{SPLIT_SEED}:{task}".encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(seed).permutation(50)
    position = int(np.flatnonzero(order == init_state)[0])
    if position >= 40:
        return "test_unseen_init"
    if run_id == RUN_B:
        return "test_new_noise_seen_init"
    return "train" if position < 30 else "calibration"


def validate_record(record, max_steps, replan_steps):
    if not isinstance(record["success"], bool):
        raise ValueError("success must be an explicit boolean")
    steps = int(record["action_steps"])
    length = int(record["inference_calls"])
    if steps < 1 or steps > max_steps or length != (steps + replan_steps - 1) // replan_steps:
        raise ValueError("action count, query count, or deadline mismatch")
    if any(record.get(k, False) for k in ("intervened", "crashed", "aborted", "interrupted")):
        raise ValueError("intervened or incomplete rollout")
    if not record["success"] and steps != max_steps:
        raise ValueError("unsuccessful rollout stopped before its deadline: outcome is censored")


def build_dataset(hub: Path, output: Path):
    cached = hub / "moe-history-only/results"
    source_frame = pd.read_csv(cached / "episode_index.csv")
    frame = source_frame[source_frame.run_id.isin([RUN_A, RUN_B])].copy()
    frame = frame.sort_values(["source_run", "episode"]).reset_index(drop=True)
    if frame.regime.ne("libero_natural").any() or len(frame) != 32000 or frame.task.nunique() != 40:
        raise ValueError("expected two complete 40-task x 50-init x 8-seed natural cohorts")
    frame.insert(0, "episode_row", np.arange(len(frame)))
    audits = {r["source_run"]: r for r in json.loads((cached / "extraction_audit.json").read_text())}
    manifest = json.loads((cached / "manifest.json").read_text())
    paths, policies, verified, blocks, episode_rows, queries = {}, {}, [], [], [], []
    first_alarms, per_episode_steps = [], []
    for source, part in frame.groupby("source_run", sort=False):
        run = hub / source
        summary_path = run / "client/summaries.json"
        meta = json.loads((run / "meta.json").read_text())
        server = json.loads((run / "client/server_metadata.json").read_text())
        records = sorted(json.loads(summary_path.read_text()), key=lambda r: r["episode_index"])
        audit = audits[source]
        if meta.get("status") != "complete" or not meta["sampling"]["complete"]:
            raise ValueError(f"incomplete source run: {source}")
        if meta["wrist_layout"] != "paper-right" or len(records) != 400:
            raise ValueError(f"unsupported collection protocol: {source}")
        suite = part.suite.iloc[0]
        cap, replan = int(meta["max_steps"]), int(server["n_action_steps"])
        if cap != CAPS[suite] or replan != 10 or part.replan_steps.ne(replan).any():
            raise ValueError("execution budget changed")
        policy = {k: server[k] for k in ("checkpoint_sha256", "normalization_stats_sha256",
                  "himoe_upstream_commit", "himoe_working_tree_diff_sha256", "himoe_patch_sha256",
                  "flow_steps", "libero_wrist_layout", "n_action_steps")}
        policy.update(max_steps=cap, environment_seed=7)
        if suite in policies and policies[suite] != policy:
            raise ValueError(f"mixed policies within suite {suite}")
        policies[suite] = policy
        if sha256(summary_path) != audit["summary_sha256"]:
            raise ValueError("source summaries changed after feature extraction")
        cache_path = cached / "features" / audit["cache_file"]
        expected = manifest["results/features/" + audit["cache_file"]]
        if sha256(cache_path) != expected:
            raise ValueError("feature cache checksum mismatch")
        for path in (summary_path, run / "meta.json", run / "client/server_metadata.json", cache_path):
            paths[str(path.resolve())] = sha256(path)
        with np.load(cache_path, allow_pickle=False) as archive:
            features = archive["features"]
            np.testing.assert_array_equal(archive["episode_ids"], part.episode)
            np.testing.assert_array_equal(archive["lengths"], part.length)
        seen = set()
        for local, (row, record) in enumerate(zip(part.itertuples(), records, strict=True)):
            validate_record(record, cap, replan)
            expected_row = (row.episode, row.init_state, row.seed, bool(row.success), row.length)
            actual_row = (record["episode_index"], record["init_state_id"], record["flow_noise_seed"],
                          record["success"], record["inference_calls"])
            if expected_row != actual_row or record["task_name"] != row.task or record["seed"] != 7:
                raise ValueError("index/source metadata mismatch")
            key = (row.init_state, row.seed)
            if key in seen:
                raise ValueError("duplicate initial-state/noise draw")
            seen.add(key)
            x, info = history_features(features[local, :row.length], cap, replan)
            blocks.append(x)
            episode_rows.append(np.full(row.length, row.episode_row, dtype=np.int32))
            queries.append(np.arange(row.length, dtype=np.int16))
            first_alarms.append(info["first_alarm"])
            per_episode_steps.append(record["action_steps"])
        seeds = sorted(part.seed.unique().tolist())
        if seeds != list(range(1000 if part.run_id.iloc[0] == RUN_A else 1008,
                               1008 if part.run_id.iloc[0] == RUN_A else 1016)):
            raise ValueError("unexpected or overlapping flow-noise cohorts")
        verified.append(dict(source_run=source, episodes=len(part), queries=int(part.length.sum()),
                             cache_sha256=expected, seed_values=seeds))
        print(f"validated {len(verified)}/80 {suite}/{part.task.iloc[0]}/{part.run_id.iloc[0]}", flush=True)
    frame["first_alarm_q"] = first_alarms
    frame["action_steps"] = per_episode_steps
    frame["max_steps"] = frame.suite.map(CAPS)
    frame["split"] = [split_for(r.task, r.init_state, r.run_id) for r in frame.itertuples()]
    frame["cluster"] = frame.suite + "/" + frame.task + "/" + frame.init_state.astype(str)
    groups = {s: set(frame.loc[frame.split == s, "cluster"]) for s in frame.split.unique()}
    for a, b in (("train", "calibration"), ("train", "test_unseen_init"), ("calibration", "test_unseen_init")):
        if groups[a] & groups[b]:
            raise AssertionError("initial-state group leakage")
    if frame.duplicated(["suite", "task", "init_state", "seed"]).any():
        raise AssertionError("duplicate natural trajectory design key")
    alarm_source = pd.read_csv(cached / "episode_alarms.csv").set_index("row")
    np.testing.assert_array_equal(frame.first_alarm_q, alarm_source.loc[frame.row, detector.PRIMARY.name])
    for rule in detector.fixed_rules():
        frame["first_q_" + rule.name] = alarm_source.loc[frame.row, rule.name].to_numpy(int)
    x = np.concatenate(blocks)
    ep = np.concatenate(episode_rows)
    q = np.concatenate(queries)
    np.savez_compressed(output / "dataset.npz", x=x, episode_row=ep, query=q,
                        feature_names=np.asarray(FEATURE_NAMES))
    frame.to_csv(output / "episodes.csv", index=False)
    for path in (DETECTOR_PATH, cached / "episode_index.csv", cached / "manifest.json",
                 cached / "extraction_audit.json", cached / "episode_alarms.csv"):
        paths[str(path.resolve())] = sha256(path)
    result = dict(episodes=len(frame), queries=len(x), tasks=int(frame.task.nunique()),
                  successes=int(frame.success.sum()), failures=int((~frame.success).sum()),
                  split_seed=SPLIT_SEED, policy_by_suite=policies, feature_names=FEATURE_NAMES,
                  source_hashes=paths, validated_runs=verified,
                  source_exclusions=source_frame[~source_frame.run_id.isin([RUN_A, RUN_B])]
                    .groupby(["regime", "run_id"]).size().rename("episodes").reset_index().to_dict("records"),
                  rho_status="not_identifiable_without_independent_trap_and_escape_labels",
                  checkpoint_branch_status="no_aligned_midtrajectory_branches_in_hub",
                  primary_split="test_unseen_init", all_first_alarms_match_existing_detector=True,
                  train_calibration_test_initial_state_groups_disjoint=True)
    write_json(output / "data_audit.json", result)
    return frame, x, ep, q, result
