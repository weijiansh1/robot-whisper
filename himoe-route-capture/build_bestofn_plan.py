#!/usr/bin/env python3
"""Build the frozen 300-snapshot plan for the Best-of-N oracle experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from behavior_forks_v2 import EVENT_SCHEMA, atomic_json, load_npz, verify_artifact
from bestofn_protocol import (
    PHASES,
    PLAN_SCHEMA,
    SPLITS,
    BestOfNError,
    load_config,
    rng_from_words,
    seed_words,
    sha256_file,
    validate_plan,
)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _event_path(event_root: Path, task: int, episode: int) -> Path:
    return event_root / f"task_{task:02d}" / f"episode_{episode:04d}" / "artifact.json"


def _profile(path: Path) -> dict[str, Any]:
    metadata = verify_artifact(path, EVENT_SCHEMA)
    arrays = load_npz(path.parent / metadata["files"]["events"]["file"])
    flags = np.asarray(arrays["event_flags"], dtype=np.bool_)
    valid = np.asarray(arrays["valid_steps"], dtype=np.bool_)
    labels = [str(value) for value in np.asarray(arrays["event_labels"])]
    if flags.ndim != 3 or valid.shape != flags.shape[:2]:
        raise BestOfNError(f"malformed source event tape: {path}")
    query_valid = valid.any(axis=1)
    if not np.any(query_valid):
        raise BestOfNError(f"source event tape contains no valid query: {path}")
    stop = int(np.flatnonzero(query_valid)[-1]) + 1
    flags = flags[:stop]
    valid = valid[:stop]

    def count(label: str) -> np.ndarray:
        index = labels.index(label)
        return (flags[..., index] & valid).sum(axis=1).astype(np.float64)

    closing = count("gripper_closing")
    opening = count("gripper_opening")
    motion = count("object_motion")
    success = count("success")
    critical = closing + opening + 2.0 * motion
    # Gripper interpolation often starts changing at query zero even in free
    # space. Object motion is the least task-semantic privileged signal that
    # reliably marks physical interaction across these three tasks. Only use a
    # gripper-change anchor when the source rollout never moves a task object.
    active = np.flatnonzero(motion > 0)
    if not len(active):
        active = np.flatnonzero((closing + opening) > 0)
    anchor = int(active[0]) if len(active) else max(1, stop // 2)
    return {
        "metadata": metadata,
        "stop": stop,
        "valid": query_valid[:stop],
        "closing": closing,
        "opening": opening,
        "motion": motion,
        "success": success,
        "critical": critical,
        "anchor": anchor,
    }


def _phase_entries(
    *, task: int, episode: int, profile: dict[str, Any], master_seed: int
) -> dict[str, list[dict[str, Any]]]:
    stop = int(profile["stop"])
    anchor = int(profile["anchor"])
    critical = np.asarray(profile["critical"])
    motion = np.asarray(profile["motion"])
    closing = np.asarray(profile["closing"])
    opening = np.asarray(profile["opening"])
    result: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}

    for step in range(stop):
        if not bool(profile["valid"][step]):
            continue
        progress = (step + 0.5) / max(1, stop)
        distance = step - anchor
        future_stop = min(stop, step + 4)
        future_manipulation = float(np.sum(critical[step + 1 : future_stop]))
        values: dict[str, float | None] = {
            "free_space": (
                -float(critical[step]) - 0.25 * abs(progress - 0.25)
                if 0.08 <= progress <= 0.45 and step <= anchor
                else None
            ),
            "pre_contact": (
                4.0 - abs(distance) + 0.1 * future_manipulation
                if -4 <= distance <= -1
                else None
            ),
            "contact_manipulation": (
                2.0 * float(motion[step])
                + float(closing[step])
                + 0.25 * float(opening[step])
                - 0.1 * abs(distance)
                if critical[step] > 0 or abs(distance) <= 1
                else None
            ),
            "late_recovery": (
                progress
                + 0.15 * float(motion[step])
                + 0.1 * float(opening[step])
                if progress >= 0.55 and step > anchor
                else None
            ),
        }
        # Short successful episodes can have only one query on one side of the
        # manipulation anchor. These deterministic fallbacks retain phase/time
        # stratification without inspecting any new candidate or outcome.
        fallbacks = {
            "free_space": -abs(progress - 0.20) - 0.1 * float(critical[step]),
            "pre_contact": -abs(progress - 0.42) - 0.05 * max(0, distance),
            "contact_manipulation": -abs(distance) + 0.1 * float(critical[step]),
            "late_recovery": -abs(progress - 0.78) + 0.05 * float(motion[step]),
        }
        for phase_index, phase in enumerate(PHASES):
            score = values[phase]
            fallback = score is None
            if score is None:
                score = fallbacks[phase] - 100.0
            tie = int(
                rng_from_words(
                    seed_words(
                        master_seed,
                        "bestofn/plan/tie",
                        task,
                        episode,
                        step,
                        (phase_index,),
                    )
                ).integers(0, np.iinfo(np.int64).max)
            )
            result[phase].append(
                {
                    "episode": episode,
                    "fork_step": step,
                    "phase_score": float(score),
                    "phase_fallback": bool(fallback),
                    "phase_anchor_step": anchor,
                    "tie": tie,
                }
            )
    return result


def _episode_splits(
    summaries: list[dict[str, Any]], *, task: int, master_seed: int
) -> dict[int, str]:
    result: dict[int, str] = {}
    for outcome in (False, True):
        rows = [row for row in summaries if bool(row["success"]) is outcome]
        words = seed_words(
            master_seed, "bestofn/plan/episode-split", task, int(outcome), 0
        )
        order = rng_from_words(words).permutation(len(rows))
        ordered = [rows[int(index)] for index in order]
        n = len(ordered)
        train_stop = int(round(0.60 * n))
        validation_stop = train_stop + int(round(0.20 * n))
        for index, row in enumerate(ordered):
            split = (
                "train"
                if index < train_stop
                else "validation"
                if index < validation_stop
                else "test"
            )
            result[int(row["episode_index"])] = split
    if set(result) != {int(row["episode_index"]) for row in summaries}:
        raise BestOfNError("episode split did not cover every source episode")
    return result


def _select_phase(
    entries: list[dict[str, Any]],
    *,
    count: int,
    used_states: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    ordered = sorted(entries, key=lambda row: (-row["phase_score"], row["tie"]))
    selected: list[dict[str, Any]] = []
    used_episodes: set[int] = set()
    for row in ordered:
        key = (int(row["episode"]), int(row["fork_step"]))
        episode = int(row["episode"])
        if key in used_states or episode in used_episodes:
            continue
        selected.append(row)
        used_states.add(key)
        used_episodes.add(episode)
        if len(selected) == count:
            return selected
    raise BestOfNError(
        f"phase/split stratum has only {len(selected)} distinct eligible states; needs {count}"
    )


def build_plan(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    master_seed = int(config["master_seed"])
    event_root = _resolve(config_path, str(config["source_event_root"]))
    states: list[dict[str, Any]] = []
    sources = []
    for task_spec in config["tasks"]:
        task = int(task_spec["task_id"])
        source = _resolve(config_path, str(task_spec["source_rollout_dir"]))
        summaries_path = source / "summaries.json"
        summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
        summaries = [row for row in summaries if int(row["task_id"]) == task]
        splits = _episode_splits(summaries, task=task, master_seed=master_seed)
        by_phase_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        episode_metadata: dict[int, dict[str, Any]] = {}
        for row in summaries:
            episode = int(row["episode_index"])
            event_artifact = _event_path(event_root, task, episode)
            profile = _profile(event_artifact)
            episode_metadata[episode] = {
                "summary": row,
                "event_artifact": event_artifact,
                "event_metadata": profile["metadata"],
            }
            phase_entries = _phase_entries(
                task=task,
                episode=episode,
                profile=profile,
                master_seed=master_seed,
            )
            for phase, entries in phase_entries.items():
                for entry in entries:
                    by_phase_split[(phase, splits[episode])].append(entry)

        used_states: set[tuple[int, int]] = set()
        task_rows: list[dict[str, Any]] = []
        split_targets = config["snapshot_sampling"]["split_counts_per_phase"]
        # Narrow manipulation strata select first; broad time strata then avoid
        # exact (episode, step) collisions without moving an episode across splits.
        phase_order = ("pre_contact", "contact_manipulation", "late_recovery", "free_space")
        for phase in phase_order:
            for split in SPLITS:
                selected = _select_phase(
                    by_phase_split[(phase, split)],
                    count=int(split_targets[split]),
                    used_states=used_states,
                )
                for entry in selected:
                    episode = int(entry["episode"])
                    fork_step = int(entry["fork_step"])
                    source_info = episode_metadata[episode]
                    summary = source_info["summary"]
                    source_episode = source / f"episode_{episode:02d}.npz"
                    state_id = f"task{task:02d}-episode{episode:04d}-step{fork_step:04d}"
                    task_rows.append(
                        {
                            "state_id": state_id,
                            "task_id": task,
                            "episode": episode,
                            "fork_step": fork_step,
                            "phase": phase,
                            "split": split,
                            "phase_score": float(entry["phase_score"]),
                            "phase_fallback": bool(entry["phase_fallback"]),
                            "phase_anchor_step": int(entry["phase_anchor_step"]),
                            "source_success": bool(summary["success"]),
                            "task_name": str(summary["task_name"]),
                            "prompt": str(summary["prompt"]),
                            "init_state_id": int(summary["init_state_id"]),
                            "environment_seed": int(summary["seed"]),
                            "settle_steps": int(source_info["event_metadata"]["settle_steps"]),
                            "source_episode_file": str(source_episode),
                            "source_episode_sha256": sha256_file(source_episode),
                            "source_event_artifact": str(source_info["event_artifact"]),
                            "source_event_artifact_sha256": sha256_file(
                                source_info["event_artifact"]
                            ),
                            "candidate_count": int(config["proposal"]["base_candidates"]),
                        }
                    )

        extension_fraction = float(config["proposal"]["extension_snapshot_fraction"])
        for phase in PHASES:
            for split in SPLITS:
                rows = [row for row in task_rows if row["phase"] == phase and row["split"] == split]
                extension_count = int(round(extension_fraction * len(rows)))
                ranked = sorted(
                    rows,
                    key=lambda row: seed_words(
                        master_seed,
                        "bestofn/plan/k16-extension",
                        task,
                        int(row["episode"]),
                        int(row["fork_step"]),
                    ),
                )
                for row in ranked[:extension_count]:
                    row["candidate_count"] = int(config["proposal"]["extension_candidates"])
        if len(task_rows) != 100:
            raise BestOfNError(f"task {task} plan has {len(task_rows)} rows instead of 100")
        states.extend(task_rows)
        sources.append(
            {
                "task_id": task,
                "source_rollout_dir": str(source),
                "summaries_file": str(summaries_path),
                "summaries_sha256": sha256_file(summaries_path),
            }
        )

    states.sort(
        key=lambda row: (
            int(row["task_id"]),
            PHASES.index(str(row["phase"])),
            SPLITS.index(str(row["split"])),
            int(row["episode"]),
            int(row["fork_step"]),
        )
    )
    for index, row in enumerate(states):
        row["snapshot_index"] = index
    plan = {
        "schema": PLAN_SCHEMA,
        "confirmatory": False,
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "master_seed": master_seed,
        "sources": sources,
        "selection_contract": (
            "source physical event/time strata only; no new candidate, routing, or outcome data"
        ),
        "states": states,
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)
    plan["plan_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    validate_plan(plan, config)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("bestofn_experiment_config.json"))
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    plan = build_plan(config_path)
    atomic_json(args.out.expanduser().resolve(), plan)
    k16 = sum(int(row["candidate_count"]) == 16 for row in plan["states"])
    print(f"wrote {len(plan['states'])} snapshots ({k16} with K=16) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
