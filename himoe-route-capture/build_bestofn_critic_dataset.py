#!/usr/bin/env python3
"""Assemble audited state/action/routing/hidden features after Gate 1 passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from analyze_bestofn_oracle import _load_outcomes
from behavior_forks_v2 import load_npz, verify_artifact
from bestofn_artifacts import validate_candidate
from bestofn_critic_features import (
    as_route_features,
    hidden_candidate_features,
    hidden_state_features,
    route_candidate_features,
    route_state_features,
)
from bestofn_protocol import (
    CANDIDATE_SCHEMA,
    candidate_dir,
    continuation_shard_dir,
    load_config,
    load_json,
    sha256_file,
    validate_plan,
)


DATASET_SCHEMA = "himoe.bestofn.critic_dataset.v1"
FLOW_VARIANTS = {
    "all": tuple(range(10)),
    "early": (0, 1, 2),
    "middle": (3, 4, 5, 6),
    "late": (7, 8, 9),
    "final": (9,),
    "reversed": tuple(reversed(range(10))),
}
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


class DatasetError(RuntimeError):
    """The gated labels or aligned telemetry are incomplete/inconsistent."""


def _memmap(root: Path, name: str, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)


def _store(root: Path, name: str) -> Any:
    path = root / name
    if not path.is_dir():
        raise DatasetError(f"candidate telemetry store is missing: {path}")
    return zarr.open_group(str(path), mode="r")


def _audit_stores(root: Path, expected_rows: int) -> tuple[Any, Any, Any, str]:
    routes = _store(root, "routes.zarr")
    hidden = _store(root, "hidden.zarr")
    flow = _store(root, "flow_trajectory.zarr")
    groups = (routes, hidden, flow)
    ids = [group.attrs.get("capture_run_id") for group in groups]
    if len(set(ids)) != 1 or not isinstance(ids[0], str) or not ids[0]:
        raise DatasetError("route/hidden/flow stores do not share one identity")
    lengths = (
        int(routes["episode_id"].shape[0]),
        int(hidden["episode_id"].shape[0]),
        int(flow["query_id"].shape[0]),
    )
    if lengths != (expected_rows, expected_rows, expected_rows):
        raise DatasetError(f"telemetry row counts differ from {expected_rows}: {lengths}")
    for group in groups:
        if int(group.attrs.get("common_durable_rows", -1)) != expected_rows:
            raise DatasetError("a telemetry store is not durable through the candidate panel")
    if "hb_router_probs" not in set(routes.array_keys()) or "hb_hidden" not in set(
        hidden.array_keys()
    ):
        raise DatasetError("full routing probabilities and router hidden states are mandatory")
    return routes, hidden, flow, ids[0]


def _validate_gate(config: Mapping[str, Any], run_root: Path) -> tuple[dict[str, Any], set[str]]:
    gate_path = run_root / "analysis" / "gate1_r4" / "summary.json"
    topup_path = run_root / "analysis" / "gate1_r4" / "topup_plan.json"
    if not gate_path.is_file() or not topup_path.is_file():
        raise DatasetError("Gate 1 summary/top-up plan is missing")
    gate = load_json(gate_path)
    if gate.get("gate_1", {}).get("passed") is not True:
        raise DatasetError("Gate 1 did not authorize critic training")
    topup = load_json(topup_path)
    states = {str(value) for value in topup["states"]}
    initial = int(config["continuation"]["initial_repeats"])
    target = int(config["continuation"]["topup_repeats"])
    for state_id in states:
        if not (
            continuation_shard_dir(run_root, state_id, initial, target) / "artifact.json"
        ).is_file():
            raise DatasetError(f"adaptive terminal-label top-up is missing for {state_id}")
    return gate, states


def _existing(target: Path, config_sha: str, plan_sha: str) -> bool:
    manifest_path = target / "dataset_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = load_json(manifest_path)
    if manifest.get("schema") != DATASET_SCHEMA:
        raise DatasetError("existing critic dataset has another schema")
    if manifest.get("config_sha256") != config_sha or manifest.get("plan_sha256") != plan_sha:
        raise DatasetError("existing critic dataset belongs to another protocol")
    for name, expected in manifest["files"].items():
        path = target / name
        if not path.is_file() or sha256_file(path) != expected:
            raise DatasetError(f"critic dataset file failed integrity check: {path}")
    return True


def build(config_path: Path, plan_path: Path, run_root: Path, target: Path) -> dict[str, Any]:
    config = load_config(config_path)
    plan = load_json(plan_path)
    validate_plan(plan, config)
    config_sha, plan_sha = sha256_file(config_path), sha256_file(plan_path)
    if plan.get("config_sha256") != config_sha:
        raise DatasetError("snapshot plan belongs to another config")
    if _existing(target, config_sha, plan_sha):
        return load_json(target / "dataset_manifest.json")
    if target.exists():
        raise DatasetError(f"uncommitted critic dataset directory exists: {target}")
    gate, topup_states = _validate_gate(config, run_root)
    total = sum(int(row["candidate_count"]) for row in plan["states"])
    routes, hidden, flow, capture_run_id = _audit_stores(
        run_root / "candidate_stores", total
    )

    first_state = plan["states"][0]
    first_descriptor = candidate_dir(run_root, str(first_state["state_id"])) / "artifact.json"
    first_meta = verify_artifact(first_descriptor, CANDIDATE_SCHEMA)
    first_arrays = load_npz(first_descriptor.parent / first_meta["files"]["data"]["file"])
    validate_candidate(first_arrays, first_meta)
    first_k = int(first_meta["candidate_count"])
    first_start = int(first_meta["server_row_start"])
    first_probs = np.asarray(
        routes["hb_router_probs"][first_start : first_start + first_k], dtype=np.float32
    )
    route_candidate_dim = route_candidate_features(first_probs).shape[1]
    route_state_dim = route_state_features(first_probs).shape[0]
    first_hidden = np.asarray(
        hidden["hb_hidden"][first_start : first_start + first_k], dtype=np.float32
    )
    hidden_candidate_shape = hidden_candidate_features(first_hidden).shape[1:]
    hidden_state_shape = hidden_state_features(first_hidden).shape
    as_dim = as_route_features(
        np.asarray(routes["as_probs"][first_start : first_start + first_k])
    ).shape[1]
    vlm_dim = int(first_meta["frozen_vlm_feature_dim"])

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    n_states = len(plan["states"])
    arrays: dict[str, np.memmap] = {
        "state_offsets": _memmap(stage, "state_offsets", np.int64, (n_states + 1,)),
        "task_id": _memmap(stage, "task_id", np.int16, (n_states,)),
        "episode": _memmap(stage, "episode", np.int16, (n_states,)),
        "fork_step": _memmap(stage, "fork_step", np.int16, (n_states,)),
        "split": _memmap(stage, "split", np.int8, (n_states,)),
        "candidate_count": _memmap(stage, "candidate_count", np.int8, (n_states,)),
        "state_vlm": _memmap(stage, "state_vlm", np.float16, (n_states, vlm_dim)),
        "state_proprio": _memmap(stage, "state_proprio", np.float32, (n_states, 8)),
        "hidden_state": _memmap(stage, "hidden_state", np.float16, (n_states, *hidden_state_shape)),
        "action": _memmap(stage, "action", np.float32, (total, 70)),
        "candidate_to_state": _memmap(stage, "candidate_to_state", np.int16, (total,)),
        "candidate_id": _memmap(stage, "candidate_id", np.int8, (total,)),
        "query_id": _memmap(stage, "query_id", np.int32, (total,)),
        "success_count": _memmap(stage, "success_count", np.int8, (total,)),
        "repeat_count": _memmap(stage, "repeat_count", np.int8, (total,)),
        "chunk_success": _memmap(stage, "chunk_success", np.bool_, (total,)),
        "hidden_candidate": _memmap(
            stage, "hidden_candidate", np.float16, (total, *hidden_candidate_shape)
        ),
        "as_route_candidate": _memmap(stage, "as_route_candidate", np.float16, (total, as_dim)),
    }
    for variant in (*FLOW_VARIANTS, "expert_permuted"):
        arrays[f"route_candidate_{variant}"] = _memmap(
            stage, f"route_candidate_{variant}", np.float16, (total, route_candidate_dim)
        )
        arrays[f"route_state_{variant}"] = _memmap(
            stage, f"route_state_{variant}", np.float16, (n_states, route_state_dim)
        )

    expert_rng = np.random.default_rng(
        np.random.SeedSequence([int(config["master_seed"]), 0x45585054])
    )
    expert_permutation = expert_rng.permutation(first_probs.shape[-1])
    cursor = 0
    repeat_histogram: Counter[int] = Counter()
    state_rows = []
    for state_index, state in enumerate(plan["states"]):
        state_id = str(state["state_id"])
        descriptor = candidate_dir(run_root, state_id) / "artifact.json"
        metadata = verify_artifact(descriptor, CANDIDATE_SCHEMA)
        candidate = load_npz(descriptor.parent / metadata["files"]["data"]["file"])
        validate_candidate(candidate, metadata)
        k = int(state["candidate_count"])
        start, stop = int(metadata["server_row_start"]), int(metadata["server_row_stop"])
        if (start, stop) != (cursor, cursor + k):
            raise DatasetError(f"candidate rows do not follow the plan at {state_id}")
        if str(metadata["route_store_id"]) != capture_run_id:
            raise DatasetError(f"candidate artifact uses another route store at {state_id}")
        expected_queries = np.asarray(candidate["query_ids"], dtype=np.int32)
        route_queries = np.asarray(routes["episode_id"][start:stop], dtype=np.int32)
        hidden_queries = np.asarray(hidden["episode_id"][start:stop], dtype=np.int32)
        flow_queries = np.asarray(flow["query_id"][start:stop], dtype=np.int32)
        if not (
            np.array_equal(expected_queries, route_queries)
            and np.array_equal(expected_queries, hidden_queries)
            and np.array_equal(expected_queries, flow_queries)
        ):
            raise DatasetError(f"telemetry query identity differs at {state_id}")
        if not np.array_equal(np.asarray(flow["capture_row"][start:stop]), np.arange(start, stop)):
            raise DatasetError(f"flow capture rows differ at {state_id}")
        if not np.all(np.asarray(flow["snapshot_index"][start:stop]) == int(state["snapshot_index"])):
            raise DatasetError(f"flow snapshot identity differs at {state_id}")
        if not np.array_equal(np.asarray(flow["candidate_id"][start:stop]), np.arange(k)):
            raise DatasetError(f"flow candidate identity differs at {state_id}")
        for local in range(k):
            digest = hashlib.sha256(
                np.ascontiguousarray(candidate["actions"][local]).tobytes()
            ).digest()
            stored = bytes(np.asarray(flow["actions_sha256"][start + local], dtype=np.uint8))
            if digest != stored:
                raise DatasetError(f"action response hash differs at {state_id}/{local}")

        outcomes, outcome_meta = _load_outcomes(run_root, state)
        if int(outcome_meta["server_row_start"]) != start:
            raise DatasetError(f"outcome/candidate linkage differs at {state_id}")
        repeats = int(outcomes.shape[1])
        expected_repeats = (
            int(config["continuation"]["topup_repeats"])
            if state_id in topup_states
            else int(config["continuation"]["initial_repeats"])
        )
        if repeats != expected_repeats:
            raise DatasetError(f"terminal repeat panel differs at {state_id}: {repeats}")
        repeat_histogram[repeats] += 1

        probs = np.asarray(routes["hb_router_probs"][start:stop], dtype=np.float32)
        hb_hidden = np.asarray(hidden["hb_hidden"][start:stop], dtype=np.float32)
        for variant, indices in FLOW_VARIANTS.items():
            arrays[f"route_candidate_{variant}"][start:stop] = route_candidate_features(
                probs, indices
            ).astype(np.float16)
            arrays[f"route_state_{variant}"][state_index] = route_state_features(
                probs, indices
            ).astype(np.float16)
        permuted = probs[..., expert_permutation]
        arrays["route_candidate_expert_permuted"][start:stop] = route_candidate_features(
            permuted
        ).astype(np.float16)
        arrays["route_state_expert_permuted"][state_index] = route_state_features(
            permuted
        ).astype(np.float16)
        arrays["hidden_candidate"][start:stop] = hidden_candidate_features(
            hb_hidden
        ).astype(np.float16)
        arrays["hidden_state"][state_index] = hidden_state_features(hb_hidden).astype(
            np.float16
        )
        arrays["as_route_candidate"][start:stop] = as_route_features(
            np.asarray(routes["as_probs"][start:stop])
        ).astype(np.float16)
        arrays["state_vlm"][state_index] = np.asarray(
            candidate["frozen_vlm_feature"], dtype=np.float32
        ).mean(axis=0).astype(np.float16)
        arrays["state_proprio"][state_index] = candidate["observation_state"]
        arrays["action"][start:stop] = np.asarray(candidate["actions"]).reshape(k, -1)
        arrays["candidate_to_state"][start:stop] = state_index
        arrays["candidate_id"][start:stop] = np.arange(k)
        arrays["query_id"][start:stop] = expected_queries
        arrays["success_count"][start:stop] = outcomes.sum(axis=1).astype(np.int8)
        arrays["repeat_count"][start:stop] = repeats
        arrays["chunk_success"][start:stop] = candidate["chunk_terminal_success"]
        arrays["state_offsets"][state_index] = start
        arrays["task_id"][state_index] = int(state["task_id"])
        arrays["episode"][state_index] = int(state["episode"])
        arrays["fork_step"][state_index] = int(state["fork_step"])
        arrays["split"][state_index] = SPLIT_CODE[str(state["split"])]
        arrays["candidate_count"][state_index] = k
        state_rows.append(
            {
                "state_index": state_index,
                "state_id": state_id,
                "task_id": int(state["task_id"]),
                "episode": int(state["episode"]),
                "fork_step": int(state["fork_step"]),
                "phase": state["phase"],
                "split": state["split"],
                "candidate_start": start,
                "candidate_stop": stop,
                "repeats": repeats,
            }
        )
        cursor = stop
    arrays["state_offsets"][-1] = cursor
    if cursor != total:
        raise DatasetError("critic dataset did not consume the complete candidate panel")
    for array in arrays.values():
        array.flush()
    (stage / "states.json").write_text(
        json.dumps({"states": state_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    file_names = sorted(path.name for path in stage.iterdir() if path.is_file())
    files = {name: sha256_file(stage / name) for name in file_names}
    manifest = {
        "schema": DATASET_SCHEMA,
        "config_file": str(config_path),
        "config_sha256": config_sha,
        "plan_file": str(plan_path),
        "plan_sha256": plan_sha,
        "gate1_summary": str(run_root / "analysis" / "gate1_r4" / "summary.json"),
        "gate1_summary_sha256": sha256_file(
            run_root / "analysis" / "gate1_r4" / "summary.json"
        ),
        "gate1_cross_fitted_headroom": gate["aggregates"]["8"][
            "task_macro_cross_fitted_headroom"
        ],
        "capture_run_id": capture_run_id,
        "states": n_states,
        "candidates": total,
        "repeat_histogram": {str(key): value for key, value in sorted(repeat_histogram.items())},
        "topup_states": len(topup_states),
        "split_codes": SPLIT_CODE,
        "flow_variants": {key: list(value) for key, value in FLOW_VARIANTS.items()},
        "expert_permutation": expert_permutation.tolist(),
        "dimensions": {
            "vlm": vlm_dim,
            "route_candidate": route_candidate_dim,
            "route_state": route_state_dim,
            "hidden_candidate": list(hidden_candidate_shape),
            "hidden_state": list(hidden_state_shape),
            "as_route_candidate": as_dim,
        },
        "feature_contract": {
            "state": "candidate-invariant pooled frozen-PaliGemma prefix plus proprioception",
            "route": (
                "full HB probabilities summarized by layer/expert moments, entropy, margin, "
                "adjacent-flow JS/L1/top1 churn, token disagreement, and state-action JS"
            ),
            "hidden": "six flow/token aggregates retaining every layer and all 1024 channels",
            "labels": "terminal success counts under common-random-number frozen-policy continuations",
            "outcome_blind_transform": True,
        },
        "files": files,
    }
    (stage / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(stage, target)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build(
        args.config.expanduser().resolve(),
        args.plan.expanduser().resolve(),
        args.run_root.expanduser().resolve(),
        args.out.expanduser().resolve(),
    )
    print(
        "assembled critic dataset: %d states, %d candidates"
        % (manifest["states"], manifest["candidates"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
