"""Shared contracts for the MoE-aware chunk Best-of-N experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CONFIG_SCHEMA = "himoe.bestofn.config.v1"
PLAN_SCHEMA = "himoe.bestofn.plan.v1"
CANDIDATE_SCHEMA = "himoe.bestofn.candidates.v1"
CONTINUATION_SCHEMA = "himoe.bestofn.terminal_continuation.v1"
SUMMARY_SCHEMA = "himoe.bestofn.oracle_summary.v1"

PHASES = ("free_space", "pre_contact", "contact_manipulation", "late_recovery")
SPLITS = ("train", "validation", "test")
FLOW_NOISE_SHAPE = (10, 24)


class BestOfNError(RuntimeError):
    """The Best-of-N protocol or one of its artifacts is invalid."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BestOfNError(f"cannot read JSON {path}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def domain_low32(domain: str) -> int:
    if not domain:
        raise ValueError("seed domain must be non-empty")
    return int.from_bytes(hashlib.sha256(domain.encode("utf-8")).digest()[-4:], "big")


def seed_words(
    master_seed: int,
    domain: str,
    task: int,
    episode: int,
    fork_step: int,
    coordinates: Sequence[int] = (),
) -> tuple[int, ...]:
    words = (
        int(master_seed),
        domain_low32(domain),
        int(task),
        int(episode),
        int(fork_step),
        *(int(value) for value in coordinates),
    )
    if any(value < 0 or value > 0xFFFFFFFF for value in words):
        raise ValueError(f"seed coordinate does not fit uint32: {words}")
    return words


def rng_from_words(words: Sequence[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(value) for value in words]))


def flow_noise(words: Sequence[int]) -> np.ndarray:
    return rng_from_words(words).standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)


def candidate_query_id(snapshot_index: int, candidate: int) -> int:
    if snapshot_index < 0 or candidate < 0 or candidate >= 32:
        raise ValueError("candidate query coordinates are invalid")
    value = 1_000_000 + int(snapshot_index) * 32 + int(candidate)
    if value > np.iinfo(np.int32).max:
        raise ValueError("candidate query id does not fit int32")
    return value


def continuation_query_id(
    snapshot_index: int, candidate: int, repeat: int, future_query: int
) -> int:
    if min(snapshot_index, candidate, repeat, future_query) < 0:
        raise ValueError("continuation query coordinates must be non-negative")
    value = (
        100_000_000
        + int(snapshot_index) * 1_000_000
        + int(candidate) * 10_000
        + int(repeat) * 100
        + int(future_query)
    )
    if value > np.iinfo(np.int32).max:
        raise ValueError("continuation query id does not fit int32")
    return value


def candidate_dir(root: Path, state_id: str) -> Path:
    return root / "capture" / state_id / "candidates"


def continuation_dir(root: Path, state_id: str) -> Path:
    return root / "capture" / state_id / "terminal_continuation"


def continuation_shard_dir(root: Path, state_id: str, start: int, stop: int) -> Path:
    if start < 0 or stop <= start:
        raise ValueError("continuation shard repeat interval is invalid")
    return continuation_dir(root, state_id) / f"repeats_{start:03d}_{stop:03d}"


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise BestOfNError("unexpected Best-of-N config schema")
    if config.get("confirmatory") is not False:
        raise BestOfNError("the first Best-of-N experiment must be exploratory")
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise BestOfNError("exactly three feasibility tasks are required")
    if len({int(row["task_id"]) for row in tasks}) != 3:
        raise BestOfNError("task ids must be unique")
    sampling = config["snapshot_sampling"]
    if int(sampling["snapshots_per_task"]) != 100:
        raise BestOfNError("the frozen feasibility design requires 100 snapshots per task")
    if set(sampling["phase_counts"]) != set(PHASES):
        raise BestOfNError("phase_counts do not match the four frozen strata")
    if sum(int(value) for value in sampling["phase_counts"].values()) != 100:
        raise BestOfNError("phase counts must sum to 100")
    split_counts = sampling["split_counts_per_phase"]
    if set(split_counts) != set(SPLITS):
        raise BestOfNError("split counts must define train/validation/test")
    if sum(int(value) for value in split_counts.values()) != 25:
        raise BestOfNError("split counts must sum to 25 snapshots per phase")
    proposal = config["proposal"]
    if int(proposal["base_candidates"]) != 8:
        raise BestOfNError("Gate 1 must use K=8")
    if int(proposal["extension_candidates"]) != 16:
        raise BestOfNError("the headroom extension must use K=16")
    fraction = float(proposal["extension_snapshot_fraction"])
    if not 0.0 <= fraction <= 1.0:
        raise BestOfNError("extension_snapshot_fraction must be in [0,1]")
    if int(proposal["action_chunk_horizon"]) != 10:
        raise BestOfNError("HiMoE LIBERO action chunks must have H=10")
    required_capture = {
        "full_hb_router_probabilities",
        "router_hidden",
        "normalized_flow_trajectory",
        "pooled_frozen_vlm_feature",
        "raw_observation_images",
        "proprioception",
        "candidate_action_chunk",
    }
    if not required_capture <= set(proposal["capture"]):
        raise BestOfNError("proposal capture omits a frozen critic input")
    continuation = config["continuation"]
    initial = int(continuation["initial_repeats"])
    topup = int(continuation["topup_repeats"])
    if initial < 4 or initial % 2 or topup < initial or topup % 2:
        raise BestOfNError("continuation repeats must be even with initial>=4 and topup>=initial")
    if int(continuation["terminal_environment_steps"]) != 300:
        raise BestOfNError("terminal continuation must use the LIBERO 300-step budget")
    if continuation.get("common_random_numbers") is not True:
        raise BestOfNError("continuation candidates must share future noise columns")
    if continuation.get("seed_excludes_candidate") is not True:
        raise BestOfNError("continuation seed derivation must exclude candidate id")
    values = [int(value) for value in config["oracle"]["candidate_counts"]]
    if values != [1, 2, 4, 8, 16]:
        raise BestOfNError("oracle candidate counts must remain [1,2,4,8,16]")
    critic = config["critic"]
    if critic.get("authorized_only_after_oracle_gate") is not True:
        raise BestOfNError("critic training must remain gated by terminal oracle headroom")
    expected_baselines = {
        "state_action",
        "state_route",
        "state_action_route",
        "state_action_router_hidden",
    }
    if set(critic["baselines"]) != expected_baselines:
        raise BestOfNError("the four frozen critic baselines changed")
    seed_groups = ("ensemble_seeds", "control_seeds", "loto_seeds")
    for name in seed_groups:
        seeds = [int(value) for value in critic[name]]
        if not seeds or len(set(seeds)) != len(seeds) or any(
            value < 0 or value > 0xFFFFFFFF for value in seeds
        ):
            raise BestOfNError(f"critic {name} must contain unique uint32 seeds")
    if len(critic["ensemble_seeds"]) < 3:
        raise BestOfNError("primary critics require an ensemble of at least three")
    training = critic["training"]
    if int(training["max_epochs"]) <= 0 or int(training["patience"]) <= 0:
        raise BestOfNError("critic epoch and patience limits must be positive")
    if int(training["batch_snapshots"]) <= 0:
        raise BestOfNError("critic batches must contain snapshots, not flat candidates")
    if int(critic["evaluation"]["bootstrap_draws"]) < 1000:
        raise BestOfNError("critic hierarchical bootstrap needs at least 1000 draws")


def load_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise BestOfNError("Best-of-N config must be one JSON object")
    validate_config(value)
    return value


def validate_plan(plan: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise BestOfNError("unexpected Best-of-N plan schema")
    states = plan.get("states")
    expected = len(config["tasks"]) * int(config["snapshot_sampling"]["snapshots_per_task"])
    if not isinstance(states, list) or len(states) != expected:
        raise BestOfNError(f"plan must contain exactly {expected} snapshots")
    indices = [int(row["snapshot_index"]) for row in states]
    if indices != list(range(expected)):
        raise BestOfNError("snapshot_index must be the contiguous plan axis")
    ids = [str(row["state_id"]) for row in states]
    if len(set(ids)) != len(ids):
        raise BestOfNError("state ids must be unique")
    task_ids = [int(row["task_id"]) for row in config["tasks"]]
    for task in task_ids:
        task_rows = [row for row in states if int(row["task_id"]) == task]
        if len(task_rows) != 100:
            raise BestOfNError(f"task {task} does not have 100 snapshots")
        for phase in PHASES:
            phase_rows = [row for row in task_rows if row["phase"] == phase]
            if len(phase_rows) != 25:
                raise BestOfNError(f"task {task} phase {phase} does not have 25 snapshots")
            for split, count in config["snapshot_sampling"]["split_counts_per_phase"].items():
                actual = sum(row["split"] == split for row in phase_rows)
                if actual != int(count):
                    raise BestOfNError(
                        f"task {task} phase {phase} split {split}: {actual} != {count}"
                    )
        episode_splits: dict[int, str] = {}
        for row in task_rows:
            episode = int(row["episode"])
            previous = episode_splits.setdefault(episode, str(row["split"]))
            if previous != row["split"]:
                raise BestOfNError("one source episode crosses dataset splits")
            if int(row["candidate_count"]) not in {8, 16}:
                raise BestOfNError("candidate_count must be 8 or 16")
