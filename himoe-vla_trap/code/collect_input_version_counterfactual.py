#!/usr/bin/env python3
"""Collect same-noise counterfactual MoE routes for adjacent input versions.

Each record restores two adjacent archived simulator states, reconstructs their
camera observations, inserts the exact archived 8-D policy states, and asks the
same HiMoE checkpoint to infer both inputs with identical flow-noise tensors.
No trajectory outcome or physical quantity is used by inference or selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = pathlib.Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
ROUTE_CAPTURE = WORKSPACE_ROOT / "himoe-route-capture"
sys.path.insert(0, str(WORKSPACE_ROOT / "himoe-libero-wrist-fix/src"))
sys.path.insert(0, str(ROUTE_CAPTURE))

from himoe_libero_bridge.client import PolicyClient  # noqa: E402
from himoe_libero_bridge.libero_runtime import (  # noqa: E402
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation  # noqa: E402
from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    FLOW_NOISE_SHAPE,
    STATE_KEY,
    validate_action_response,
)
from rolling_star_collect import (  # noqa: E402
    _array_sha256,
    _atomic_json,
    _atomic_npz,
    _sha256_file,
)


SCHEMA = "himoe.input_version_counterfactual.capture.v1"
FULL_PROBS_KEY = "recorder/hb_router_probs"
EXPECTED_ROUTE_SHAPE = (8, 10, 11, 32)


def file_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "himoe.input_version_counterfactual.config.v1":
        raise ValueError("unsupported config schema")
    if bool(value.get("training", True)):
        raise ValueError("counterfactual experiment must declare training=false")
    if int(value["noise_draws"]) < 2:
        raise ValueError("noise_draws must be at least two")
    return value


def load_alignment(path: pathlib.Path) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            candidate = int(row["candidate"])
            records[candidate] = {
                "candidate": candidate,
                "episode_id": int(row["episode_id"]),
                "outcome": str(row["outcome"]),
                "closure_query": int(row["closure_query"]),
            }
    return records


def load_branch(path: pathlib.Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as archive:
        required = ("flow_noise", "policy_state", "sim_state", "action_chunks")
        missing = [name for name in required if name not in archive]
        if missing:
            raise RuntimeError("%s misses %s" % (path, missing))
        return {name: np.asarray(archive[name]) for name in required}


def alternate_noise(seed: int, candidate: int, query: int, draw: int) -> np.ndarray:
    sequence = np.random.SeedSequence(
        [int(seed), int(candidate), int(query), int(draw), 0x564552]
    )
    return np.random.default_rng(sequence).standard_normal(FLOW_NOISE_SHAPE).astype(
        np.float32
    )


def request(
    client: PolicyClient,
    observation: Mapping[str, Any],
    noise: np.ndarray,
    episode_id: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    payload = dict(observation)
    payload[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, dtype=np.float32)
    payload["episode_id"] = int(episode_id)
    response = validate_action_response(client.infer(payload))
    if response.get(FLOW_NOISE_SHA256_KEY) != _array_sha256(noise):
        raise RuntimeError("server did not acknowledge the exact flow-noise tensor")
    if FULL_PROBS_KEY not in response:
        raise RuntimeError("server does not return full HB router probabilities")
    action = np.asarray(response[ACTION_KEY], dtype=np.float32)
    route = np.asarray(response[FULL_PROBS_KEY], dtype=np.float32)
    if action.shape != (10, 7):
        raise RuntimeError("unexpected action shape %s" % (action.shape,))
    if route.shape != EXPECTED_ROUTE_SHAPE:
        raise RuntimeError("unexpected route shape %s" % (route.shape,))
    if not np.isfinite(route).all():
        raise RuntimeError("route contains NaN or infinity")
    if not np.allclose(route.sum(axis=-1), 1.0, atol=2e-3, rtol=2e-3):
        raise RuntimeError("router probabilities do not sum to one")
    return action, route, float(response.get("server/inference_ms", np.nan))


def reconstruct_observation(
    environment: Any,
    prompt: str,
    sim_state: np.ndarray,
    archived_policy_state: np.ndarray,
) -> Tuple[Dict[str, Any], float]:
    raw = environment.regenerate_obs_from_state(sim_state)
    policy = build_policy_observation(raw, prompt)
    reconstructed = np.asarray(policy[STATE_KEY], dtype=np.float32)
    archived = np.asarray(archived_policy_state, dtype=np.float32)
    state_error = float(np.max(np.abs(reconstructed - archived)))
    # The exact state vector was archived at collection time.  Rendering must be
    # regenerated, but there is no reason to retain quaternion/float32 roundoff.
    policy[STATE_KEY] = np.ascontiguousarray(archived)
    return policy, state_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PACKAGE_ROOT / "configs/input_version_counterfactual.json",
    )
    parser.add_argument(
        "--alignment-csv",
        type=pathlib.Path,
        default=PACKAGE_ROOT
        / "results/failed_grasp_moe_dynamics/tables/physical_event_alignment.csv",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--libero-root",
        type=pathlib.Path,
        default=WORKSPACE_ROOT
        / "himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--inference-timeout", type=float, default=180.0)
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    parser.add_argument("--episode-id-base", type=int, default=1900000000)
    parser.add_argument(
        "--candidates",
        default=None,
        help="optional comma-separated subset; default is failed plus all controls",
    )
    parser.add_argument(
        "--relatives",
        default=None,
        help="optional comma-separated subset; default comes from config",
    )
    parser.add_argument("--noise-draws", type=int, default=None)
    return parser.parse_args()


def parse_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    alignment_path = args.alignment_csv.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite output: %s" % output)
    output.mkdir(parents=True)

    config = load_config(config_path)
    raw_root = (WORKSPACE_ROOT / config["raw_run_root"]).resolve()
    snapshot_dir = raw_root / (
        "formal/worker%d/snapshot_%03d"
        % (int(config["worker"]), int(config["snapshot"]))
    )
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt = str(manifest["prompt"])
    alignment = load_alignment(alignment_path)
    default_candidates = [
        int(config["failed_candidate"]),
        *[int(value) for value in config["control_candidates"]],
    ]
    candidates = default_candidates if args.candidates is None else parse_ints(args.candidates)
    relatives = (
        [int(value) for value in config["relative_queries"]]
        if args.relatives is None
        else parse_ints(args.relatives)
    )
    noise_draws = int(config["noise_draws"] if args.noise_draws is None else args.noise_draws)
    if noise_draws < 2:
        raise ValueError("noise_draws must be at least two")
    if len(set(candidates)) != len(candidates) or len(set(relatives)) != len(relatives):
        raise ValueError("candidate and relative lists must not contain duplicates")
    for candidate in candidates:
        if candidate not in alignment:
            raise ValueError("candidate %d has no event alignment" % candidate)

    run_config = {
        "schema": SCHEMA,
        "status": "collecting",
        "training": False,
        "selection_uses_outcome": False,
        "selection_uses_physics": False,
        "normal_trajectory_bank_used": False,
        "task_identity_used_by_score": False,
        "counterfactual": "same flow noise, adjacent input versions",
        "config": str(config_path),
        "config_sha256": file_digest(config_path),
        "alignment_csv": str(alignment_path),
        "alignment_csv_sha256": file_digest(alignment_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_digest(manifest_path),
        "raw_run_root": str(raw_root),
        "candidates": candidates,
        "relative_queries": relatives,
        "noise_draws": noise_draws,
        "prompt": prompt,
        "benchmark": str(config["benchmark"]),
        "task_id": int(config["task_id"]),
        "init_state_id": int(config["init_state_id"]),
        "counterfactual_seed": int(config["counterfactual_seed"]),
    }
    _atomic_json(output / "experiment_config.json", run_config)

    episode_config = EpisodeConfig(
        libero_root=str(args.libero_root.expanduser().resolve()),
        task_suite=str(config["benchmark"]),
        task_id=int(config["task_id"]),
        init_state_id=int(config["init_state_id"]),
        seed=int(config["environment_seed"]),
        settle_steps=10,
        max_steps=520,
        render_size=512,
    )
    environment, _initial, task, runtime_prompt = _load_task(episode_config)
    if str(runtime_prompt) != prompt:
        environment.close()
        raise RuntimeError("runtime/source prompt mismatch")

    records: List[Dict[str, Any]] = []
    routes_previous: List[np.ndarray] = []
    routes_current: List[np.ndarray] = []
    actions_previous: List[np.ndarray] = []
    actions_current: List[np.ndarray] = []
    noises_all: List[np.ndarray] = []
    inference_ms_previous: List[np.ndarray] = []
    inference_ms_current: List[np.ndarray] = []
    source_actions_current: List[np.ndarray] = []
    request_id = int(args.episode_id_base)
    started = time.time()

    try:
        with PolicyClient(
            host=args.host,
            port=args.port,
            connect_timeout=args.connect_timeout,
            inference_timeout=args.inference_timeout,
        ) as client:
            validate_policy_suite(client.metadata, str(config["benchmark"]))
            if client.metadata.get("full_router_probs_response_key") != FULL_PROBS_KEY:
                raise RuntimeError("server lacks full-probability response contract")
            _atomic_json(output / "server_metadata.json", dict(client.metadata))

            for candidate in candidates:
                source_path = snapshot_dir / ("candidate_%02d.npz" % candidate)
                arrays = load_branch(source_path)
                event = alignment[candidate]
                closure = int(event["closure_query"])
                for relative in relatives:
                    current_query = closure + int(relative)
                    previous_query = current_query - 1
                    if previous_query < 0 or current_query >= len(arrays["sim_state"]):
                        continue
                    previous_observation, previous_state_error = reconstruct_observation(
                        environment,
                        prompt,
                        arrays["sim_state"][previous_query],
                        arrays["policy_state"][previous_query],
                    )
                    current_observation, current_state_error = reconstruct_observation(
                        environment,
                        prompt,
                        arrays["sim_state"][current_query],
                        arrays["policy_state"][current_query],
                    )
                    noises = [
                        np.asarray(arrays["flow_noise"][current_query], dtype=np.float32)
                    ]
                    noises.extend(
                        alternate_noise(
                            int(config["counterfactual_seed"]),
                            candidate,
                            current_query,
                            draw,
                        )
                        for draw in range(1, noise_draws)
                    )

                    pair_previous_routes: List[np.ndarray] = []
                    pair_current_routes: List[np.ndarray] = []
                    pair_previous_actions: List[np.ndarray] = []
                    pair_current_actions: List[np.ndarray] = []
                    pair_previous_ms: List[float] = []
                    pair_current_ms: List[float] = []
                    for draw, noise in enumerate(noises):
                        previous_action, previous_route, previous_ms = request(
                            client, previous_observation, noise, request_id
                        )
                        request_id += 1
                        current_action, current_route, current_ms = request(
                            client, current_observation, noise, request_id
                        )
                        request_id += 1
                        pair_previous_actions.append(previous_action)
                        pair_current_actions.append(current_action)
                        pair_previous_routes.append(previous_route.astype(np.float16))
                        pair_current_routes.append(current_route.astype(np.float16))
                        pair_previous_ms.append(previous_ms)
                        pair_current_ms.append(current_ms)

                    record_index = len(records)
                    records.append(
                        {
                            "record_index": record_index,
                            "candidate": candidate,
                            "episode_id": int(event["episode_id"]),
                            "outcome": str(event["outcome"]),
                            "closure_query": closure,
                            "relative_query": int(relative),
                            "previous_query": previous_query,
                            "current_query": current_query,
                            "source_npz": str(source_path),
                            "source_npz_sha256": _sha256_file(source_path),
                            "previous_reconstructed_state_max_abs_error": previous_state_error,
                            "current_reconstructed_state_max_abs_error": current_state_error,
                            "previous_image_sha256": array_digest(
                                previous_observation["observation/image"]
                            ),
                            "current_image_sha256": array_digest(
                                current_observation["observation/image"]
                            ),
                            "previous_wrist_image_sha256": array_digest(
                                previous_observation["observation/wrist_image"]
                            ),
                            "current_wrist_image_sha256": array_digest(
                                current_observation["observation/wrist_image"]
                            ),
                            "archived_noise_sha256": _array_sha256(noises[0]),
                        }
                    )
                    routes_previous.append(np.stack(pair_previous_routes))
                    routes_current.append(np.stack(pair_current_routes))
                    actions_previous.append(np.stack(pair_previous_actions))
                    actions_current.append(np.stack(pair_current_actions))
                    noises_all.append(np.stack(noises))
                    inference_ms_previous.append(np.asarray(pair_previous_ms, np.float32))
                    inference_ms_current.append(np.asarray(pair_current_ms, np.float32))
                    source_actions_current.append(
                        np.asarray(arrays["action_chunks"][current_query], np.float32)
                    )
                    print(
                        "record=%03d candidate=%02d outcome=%s rel=%+d q=%d calls=%d"
                        % (
                            record_index,
                            candidate,
                            event["outcome"],
                            relative,
                            current_query,
                            2 * noise_draws,
                        ),
                        flush=True,
                    )
                    _atomic_json(
                        output / "progress.json",
                        {
                            "schema": SCHEMA,
                            "status": "collecting",
                            "completed_records": len(records),
                            "completed_inference_calls": 2 * noise_draws * len(records),
                            "last_record": records[-1],
                            "elapsed_s": time.time() - started,
                        },
                    )
    finally:
        environment.close()

    arrays_path = output / "paired_counterfactual_routes.npz"
    _atomic_npz(
        arrays_path,
        {
            "schema": np.asarray(SCHEMA),
            "training": np.asarray(False),
            "routes_previous": np.stack(routes_previous),
            "routes_current": np.stack(routes_current),
            "actions_previous": np.stack(actions_previous),
            "actions_current": np.stack(actions_current),
            "source_actions_current": np.stack(source_actions_current),
            "flow_noise": np.stack(noises_all),
            "inference_ms_previous": np.stack(inference_ms_previous),
            "inference_ms_current": np.stack(inference_ms_current),
            "candidate": np.asarray([row["candidate"] for row in records], np.int16),
            "episode_id": np.asarray([row["episode_id"] for row in records], np.int32),
            "relative_query": np.asarray(
                [row["relative_query"] for row in records], np.int16
            ),
            "previous_query": np.asarray(
                [row["previous_query"] for row in records], np.int16
            ),
            "current_query": np.asarray(
                [row["current_query"] for row in records], np.int16
            ),
            "failed_grasp": np.asarray(
                [row["outcome"] == "failed_grasp" for row in records], np.bool_
            ),
        },
    )
    _atomic_json(output / "records.json", {"schema": SCHEMA, "records": records})
    manifest_out = {
        **run_config,
        "status": "complete",
        "records": len(records),
        "inference_calls": 2 * noise_draws * len(records),
        "elapsed_s": time.time() - started,
        "paired_routes": str(arrays_path),
        "paired_routes_sha256": _sha256_file(arrays_path),
        "records_json_sha256": _sha256_file(output / "records.json"),
        "max_reconstructed_state_error_before_exact_override": float(
            max(
                max(row["previous_reconstructed_state_max_abs_error"], row["current_reconstructed_state_max_abs_error"])
                for row in records
            )
        ),
        "exact_archived_policy_state_inserted": True,
    }
    _atomic_json(output / "manifest.json", manifest_out)
    print(json.dumps(manifest_out, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
