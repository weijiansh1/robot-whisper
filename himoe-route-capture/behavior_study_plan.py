"""Build the preregistered 18-state HiMoE behavior-study plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from behavior_forks_v2 import (
    EVENT_LABELS,
    EVENT_SCHEMA,
    PLAN_SCHEMA,
    ROOT_SEED,
    SEED_DOMAINS,
    atomic_json,
    load_npz,
    rng_from_words,
    seed_words,
    sha256_file,
    state_id,
    verify_artifact,
)


TASK_IDS = (0, 1, 3)
STATES_PER_TASK = 6
OUTCOMES_PER_TASK = 3
EVENT_CRITICAL_PER_OUTCOME = 2
AUDIT_PER_OUTCOME = 1
CRITICAL_EVENT_LABELS = (
    "contact_onset",
    "contact_release",
    "gripper_closing",
    "gripper_opening",
    "object_motion",
    "success_onset",
)


def _parse_task_sources(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        task_raw, separator, path_raw = value.partition("=")
        if not separator:
            raise ValueError("--task-source must use TASK=PATH")
        task = int(task_raw)
        if task in result:
            raise ValueError(f"duplicate source for task {task}")
        result[task] = Path(path_raw).expanduser().resolve()
    if set(result) != set(TASK_IDS):
        raise ValueError(f"task sources must be exactly {TASK_IDS}")
    return result


def event_artifact_path(event_root: Path, task: int, episode: int) -> Path:
    return event_root / f"task_{task:02d}" / f"episode_{episode:04d}" / "artifact.json"


def _stable_order(items: list[dict[str, Any]], words: tuple[int, ...]) -> list[dict[str, Any]]:
    rng = rng_from_words(words)
    permutation = rng.permutation(len(items))
    return [items[int(index)] for index in permutation]


def _event_profile(event_artifact: Path) -> dict[str, Any]:
    descriptor = verify_artifact(event_artifact, EVENT_SCHEMA)
    if descriptor.get("replay_fidelity_passed") is not True:
        raise ValueError(f"source-event replay fidelity did not pass: {event_artifact}")
    fidelity_max = float(descriptor["replay_fidelity_max_abs"])
    fidelity_tolerance = float(descriptor["replay_fidelity_tolerance"])
    if (
        not np.isfinite(fidelity_max)
        or not np.isfinite(fidelity_tolerance)
        or fidelity_tolerance <= 0.0
        or fidelity_max > fidelity_tolerance
    ):
        raise ValueError(f"invalid source-event replay fidelity: {event_artifact}")
    arrays = load_npz(event_artifact.parent / descriptor["files"]["events"]["file"])
    flags = np.asarray(arrays["event_flags"], dtype=np.bool_)
    valid = np.asarray(arrays["valid_steps"], dtype=np.bool_)
    labels = [str(value) for value in np.asarray(arrays["event_labels"])]
    if flags.ndim != 3 or valid.shape != flags.shape[:2] or labels != list(EVENT_LABELS):
        raise ValueError(f"invalid source-event tape: {event_artifact}")
    critical_indices = [labels.index(label) for label in CRITICAL_EVENT_LABELS]
    score = (flags[..., critical_indices] & valid[..., None]).sum(axis=(1, 2)).astype(np.int64)
    if len(score) == 0:
        raise ValueError(f"source-event tape contains no control steps: {event_artifact}")
    return {
        "descriptor": descriptor,
        "artifact": event_artifact,
        "score": score,
        "valid_control_steps": np.flatnonzero(valid.any(axis=1)),
    }


def _window_bounds(count: int, lower: float, upper: float) -> tuple[int, int]:
    if count <= 0:
        raise ValueError("event profile must contain at least one control step")
    lo = min(count - 1, int(np.floor(lower * count)))
    hi = min(count, max(lo + 1, int(np.ceil(upper * count))))
    return lo, hi


def _window_choice(profile: dict[str, Any], lower: float, upper: float) -> tuple[int, int]:
    score = np.asarray(profile["score"], dtype=np.int64)
    lo, hi = _window_bounds(len(score), lower, upper)
    window = score[lo:hi]
    best = int(window.max())
    if best == 0:
        # A zero-event cell remains in the preregistered frame; it is an explicit
        # midpoint fallback, not an endpoint proxy or a reason to cherry-pick.
        return int((lo + hi - 1) // 2), 0
    return int(lo + np.flatnonzero(window == best)[0]), best


def _assign_critical(group: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str, int, int]]:
    """Jointly maximize early+late event score with distinct episodes."""

    if len(group) < 3:
        raise ValueError("each source outcome needs at least three episodes")
    candidates = []
    for row in group:
        early_step, early_score = _window_choice(row["profile"], 0.20, 0.50)
        late_step, late_score = _window_choice(row["profile"], 0.50, 0.80)
        candidates.append((row, early_step, early_score, late_step, late_score))
    assignments = []
    for early in candidates:
        for late in candidates:
            if int(early[0]["episode_index"]) == int(late[0]["episode_index"]):
                continue
            # Maximum total score; ties follow earliest step then lowest episode
            # independently in early-before-late cell order.
            key = (
                -(early[2] + late[4]),
                early[1],
                int(early[0]["episode_index"]),
                late[3],
                int(late[0]["episode_index"]),
            )
            assignments.append((key, early, late))
    if not assignments:
        raise ValueError("could not assign distinct early/late episodes")
    _key, early, late = min(assignments, key=lambda item: item[0])
    return [
        (early[0], "event_critical_early", early[1], early[2]),
        (late[0], "event_critical_late", late[3], late[4]),
    ]


def _audit_step(profile: dict[str, Any], task: int, episode: int) -> int:
    count = len(np.asarray(profile["score"]))
    lo, hi = _window_bounds(count, 0.20, 0.80)
    eligible = np.asarray(
        [step for step in profile["valid_control_steps"] if lo <= int(step) < hi],
        dtype=np.int64,
    )
    if len(eligible) == 0:
        raise ValueError(f"task {task} episode {episode} has no valid central control step")
    # The draw is uniform over source control queries and independent of event score.
    words = seed_words(SEED_DOMAINS["audit_inclusion"], task, episode, 0, (1,))
    return int(eligible[int(rng_from_words(words).integers(len(eligible)))])


def build_plan(task_sources: dict[int, Path], event_root: Path) -> dict[str, Any]:
    if set(task_sources) != set(TASK_IDS):
        raise ValueError(f"task_sources must be exactly {TASK_IDS}")
    states: list[dict[str, Any]] = []
    source_records = []
    for task in TASK_IDS:
        source = task_sources[task]
        summaries_path = source / "summaries.json"
        summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
        rows = [row for row in summaries if int(row["task_id"]) == task]
        if not rows:
            raise ValueError(f"source {source} has no task {task} episodes")
        enriched = []
        for row in rows:
            episode = int(row["episode_index"])
            artifact = event_artifact_path(event_root, task, episode)
            profile = _event_profile(artifact)
            enriched.append({**row, "profile": profile})
        for outcome in (True, False):
            group = [row for row in enriched if bool(row["success"]) is outcome]
            if len(group) < OUTCOMES_PER_TASK:
                raise ValueError(f"task {task} needs at least three source-{outcome} episodes")
            critical = _assign_critical(group)
            used = {int(row["episode_index"]) for row, _kind, _step, _score in critical}
            audit_pool = [row for row in group if int(row["episode_index"]) not in used]
            audit_pool = _stable_order(
                audit_pool,
                seed_words(
                    SEED_DOMAINS["audit_inclusion"], task, int(outcome), 0, (0,)
                ),
            )
            if len(audit_pool) < AUDIT_PER_OUTCOME:
                raise ValueError(f"task {task} source-{outcome} has no independent audit episode")
            selected = list(critical)
            selected += [(audit_pool[0], "uniform_audit", None, None)]
            for row, selection, selected_step, selected_score in selected:
                episode = int(row["episode_index"])
                profile = row["profile"]
                if selection.startswith("event_critical"):
                    fork_step, event_score = int(selected_step), int(selected_score)
                else:
                    fork_step, event_score = _audit_step(profile, task, episode), None
                source_episode = source / f"episode_{episode:02d}.npz"
                identifier = state_id(task, episode, fork_step)
                states.append(
                    {
                        "state_id": identifier,
                        "task_id": task,
                        "episode": episode,
                        "fork_step": fork_step,
                        "source_success": outcome,
                        "selection": selection,
                        "event_score": event_score,
                        "task_name": str(row["task_name"]),
                        "prompt": str(row["prompt"]),
                        "init_state_id": int(row["init_state_id"]),
                        "environment_seed": int(row["seed"]),
                        "source_dir": str(source),
                        "source_episode_file": str(source_episode),
                        "source_episode_sha256": sha256_file(source_episode),
                        "source_event_artifact": str(profile["artifact"]),
                        "source_event_artifact_sha256": sha256_file(profile["artifact"]),
                        "source_replay_fidelity_max_abs": float(
                            profile["descriptor"]["replay_fidelity_max_abs"]
                        ),
                        "source_replay_fidelity_tolerance": float(
                            profile["descriptor"]["replay_fidelity_tolerance"]
                        ),
                        "source_replay_fidelity_passed": bool(
                            profile["descriptor"]["replay_fidelity_passed"]
                        ),
                        "settle_steps": int(profile["descriptor"]["settle_steps"]),
                    }
                )
        source_records.append(
            {
                "task_id": task,
                "source_dir": str(source),
                "summaries_file": str(summaries_path),
                "summaries_sha256": sha256_file(summaries_path),
            }
        )
    validate_plan_states(states)
    ordered_states = sorted(
        states,
        key=lambda row: (
            row["task_id"],
            row["source_success"],
            row["selection"],
            row["episode"],
        ),
    )
    for snapshot_index, row in enumerate(ordered_states):
        row["snapshot_index"] = snapshot_index
    validate_plan_states(ordered_states)
    payload = {
        "schema": PLAN_SCHEMA,
        "root_seed": ROOT_SEED,
        "seed_contract": (
            "SeedSequence([20260824, SHA256(domain)_low32, task, episode, "
            "fork_step, *coordinates])"
        ),
        "tasks": list(TASK_IDS),
        "sources": source_records,
        "pools": {
            "screen": {"candidate_count": 32, "continuation_repeats": 8},
            "formal": {
                "candidate_count": 32,
                "continuation_repeats_initial": 48,
                "continuation_repeats_upgrade": 96,
            },
        },
        "continuation_control_steps": 5,
        "action_chunk_steps": 10,
        "states": ordered_states,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["plan_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_plan_states(states: list[dict[str, Any]]) -> None:
    if len(states) != len(TASK_IDS) * STATES_PER_TASK:
        raise ValueError("the micro-pilot plan must contain exactly 18 states")
    identifiers = [row["state_id"] for row in states]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("state ids must be unique")
    indices = [row.get("snapshot_index") for row in states if "snapshot_index" in row]
    if indices and sorted(indices) != list(range(len(states))):
        raise ValueError("snapshot_index must be the contiguous plan axis")
    for task in TASK_IDS:
        rows = [row for row in states if int(row["task_id"]) == task]
        if len(rows) != STATES_PER_TASK:
            raise ValueError(f"task {task} must contain exactly six states")
        if len({int(row["episode"]) for row in rows}) != STATES_PER_TASK:
            raise ValueError(f"task {task} states must use six different episodes")
        for outcome in (True, False):
            outcome_rows = [row for row in rows if bool(row["source_success"]) is outcome]
            if len(outcome_rows) != OUTCOMES_PER_TASK:
                raise ValueError(f"task {task} must have three source-{outcome} states")
            if sum(row["selection"].startswith("event_critical") for row in outcome_rows) != 2:
                raise ValueError(f"task {task} source-{outcome} must have two critical states")
            if {row["selection"] for row in outcome_rows if row["selection"].startswith("event_critical")} != {
                "event_critical_early",
                "event_critical_late",
            }:
                raise ValueError(f"task {task} source-{outcome} must cover early and late")
            if sum(row["selection"] == "uniform_audit" for row in outcome_rows) != 1:
                raise ValueError(f"task {task} source-{outcome} must have one audit state")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--task-source", action="append", required=True, metavar="TASK=PATH")
    build.add_argument("--event-root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command != "build":
        raise AssertionError(args.command)
    plan = build_plan(_parse_task_sources(args.task_source), args.event_root.resolve())
    atomic_json(args.out, plan)
    print(f"wrote {len(plan['states'])} states to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
