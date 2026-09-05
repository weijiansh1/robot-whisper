#!/usr/bin/env python3
"""Complete a 2x2 image/state counterfactual for selected input-version pairs.

The existing paired capture supplies V00 (old image, old state), V11 (new
image, new state), and the exact noise tensors.  This collector adds V10 (new
image, old state) and V01 (old image, new state) under those same noises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Mapping, Tuple

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
    STATE_KEY,
    validate_action_response,
)
from rolling_star_collect import (  # noqa: E402
    _array_sha256,
    _atomic_json,
    _atomic_npz,
    _sha256_file,
)


SCHEMA = "himoe.input_modality_counterfactual.capture.v1"
FULL_PROBS_KEY = "recorder/hb_router_probs"
EXPECTED_ROUTE_SHAPE = (8, 10, 11, 32)


def file_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise RuntimeError("server did not acknowledge exact noise")
    route = np.asarray(response[FULL_PROBS_KEY], dtype=np.float32)
    action = np.asarray(response[ACTION_KEY], dtype=np.float32)
    if route.shape != EXPECTED_ROUTE_SHAPE or action.shape != (10, 7):
        raise RuntimeError("unexpected response shapes: %s %s" % (route.shape, action.shape))
    return action, route, float(response.get("server/inference_ms", np.nan))


def exact_observation(
    environment: Any,
    prompt: str,
    sim_state: np.ndarray,
    policy_state: np.ndarray,
) -> Dict[str, Any]:
    observation = build_policy_observation(
        environment.regenerate_obs_from_state(sim_state), prompt
    )
    observation[STATE_KEY] = np.ascontiguousarray(policy_state, dtype=np.float32)
    return observation


def hybrid(
    image_source: Mapping[str, Any], state_source: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "observation/image": np.ascontiguousarray(image_source["observation/image"]),
        "observation/wrist_image": np.ascontiguousarray(
            image_source["observation/wrist_image"]
        ),
        STATE_KEY: np.ascontiguousarray(state_source[STATE_KEY], dtype=np.float32),
        "prompt": str(image_source["prompt"]),
    }


def parse_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PACKAGE_ROOT / "configs/input_version_counterfactual.json",
    )
    parser.add_argument(
        "--paired-capture",
        type=pathlib.Path,
        default=PACKAGE_ROOT / "results/input_version_counterfactual/formal_capture",
    )
    parser.add_argument(
        "--libero-root",
        type=pathlib.Path,
        default=WORKSPACE_ROOT
        / "himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO",
    )
    parser.add_argument("--relatives", default="-2,-1,0,1,2")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--episode-id-base", type=int, default=1950000000)
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    parser.add_argument("--inference-timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite output: %s" % output)
    output.mkdir(parents=True)
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paired_root = args.paired_capture.expanduser().resolve()
    paired_manifest_path = paired_root / "manifest.json"
    paired_manifest = json.loads(paired_manifest_path.read_text(encoding="utf-8"))
    records_path = paired_root / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))["records"]
    paired_path = paired_root / "paired_counterfactual_routes.npz"
    with np.load(str(paired_path), allow_pickle=False) as source:
        paired = {name: np.asarray(source[name]) for name in source.files}
    relatives = parse_ints(args.relatives)
    selected = [
        index
        for index, record in enumerate(records)
        if int(record["relative_query"]) in relatives
    ]
    if not selected:
        raise RuntimeError("no paired records match requested relatives")

    raw_root = pathlib.Path(paired_manifest["raw_run_root"])
    snapshot_dir = raw_root / (
        "formal/worker%d/snapshot_%03d"
        % (int(config["worker"]), int(config["snapshot"]))
    )
    source_manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    prompt = str(source_manifest["prompt"])
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
    environment, _initial, _task, runtime_prompt = _load_task(episode_config)
    if str(runtime_prompt) != prompt:
        environment.close()
        raise RuntimeError("prompt mismatch")

    run_config = {
        "schema": SCHEMA,
        "status": "collecting",
        "training": False,
        "selection_uses_outcome": False,
        "selection_uses_physics": False,
        "normal_trajectory_bank_used": False,
        "task_identity_used_by_score": False,
        "factorial_inputs": {
            "V00": "old images + old state (reused from paired capture)",
            "V10": "new images + old state (collected here)",
            "V01": "old images + new state (collected here)",
            "V11": "new images + new state (reused from paired capture)",
        },
        "paired_capture": str(paired_root),
        "paired_capture_sha256": file_digest(paired_path),
        "selected_record_indices": selected,
        "relative_queries": relatives,
        "noise_draws": int(paired["flow_noise"].shape[1]),
    }
    _atomic_json(output / "experiment_config.json", run_config)

    routes_vision_only: List[np.ndarray] = []
    routes_state_only: List[np.ndarray] = []
    actions_vision_only: List[np.ndarray] = []
    actions_state_only: List[np.ndarray] = []
    inference_ms_vision_only: List[np.ndarray] = []
    inference_ms_state_only: List[np.ndarray] = []
    branch_cache: Dict[int, Dict[str, np.ndarray]] = {}
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
                raise RuntimeError("server lacks full route response")
            _atomic_json(output / "server_metadata.json", dict(client.metadata))
            for ordinal, record_index in enumerate(selected):
                record = records[record_index]
                candidate = int(record["candidate"])
                if candidate not in branch_cache:
                    path = snapshot_dir / ("candidate_%02d.npz" % candidate)
                    with np.load(str(path), allow_pickle=False) as source:
                        branch_cache[candidate] = {
                            "sim_state": np.asarray(source["sim_state"]),
                            "policy_state": np.asarray(source["policy_state"]),
                        }
                branch = branch_cache[candidate]
                previous_query = int(record["previous_query"])
                current_query = int(record["current_query"])
                old = exact_observation(
                    environment,
                    prompt,
                    branch["sim_state"][previous_query],
                    branch["policy_state"][previous_query],
                )
                new = exact_observation(
                    environment,
                    prompt,
                    branch["sim_state"][current_query],
                    branch["policy_state"][current_query],
                )
                vision_only = hybrid(new, old)
                state_only = hybrid(old, new)
                vision_routes: List[np.ndarray] = []
                state_routes: List[np.ndarray] = []
                vision_actions: List[np.ndarray] = []
                state_actions: List[np.ndarray] = []
                vision_ms: List[float] = []
                state_ms: List[float] = []
                for noise in paired["flow_noise"][record_index]:
                    action, route, elapsed = request(
                        client, vision_only, noise, request_id
                    )
                    request_id += 1
                    vision_actions.append(action)
                    vision_routes.append(route.astype(np.float16))
                    vision_ms.append(elapsed)
                    action, route, elapsed = request(
                        client, state_only, noise, request_id
                    )
                    request_id += 1
                    state_actions.append(action)
                    state_routes.append(route.astype(np.float16))
                    state_ms.append(elapsed)
                routes_vision_only.append(np.stack(vision_routes))
                routes_state_only.append(np.stack(state_routes))
                actions_vision_only.append(np.stack(vision_actions))
                actions_state_only.append(np.stack(state_actions))
                inference_ms_vision_only.append(np.asarray(vision_ms, np.float32))
                inference_ms_state_only.append(np.asarray(state_ms, np.float32))
                print(
                    "factorial=%03d/%03d candidate=%02d rel=%+d paired_record=%d"
                    % (
                        ordinal + 1,
                        len(selected),
                        candidate,
                        int(record["relative_query"]),
                        record_index,
                    ),
                    flush=True,
                )
                _atomic_json(
                    output / "progress.json",
                    {
                        "schema": SCHEMA,
                        "status": "collecting",
                        "completed_records": ordinal + 1,
                        "inference_calls": 2
                        * int(paired["flow_noise"].shape[1])
                        * (ordinal + 1),
                        "elapsed_s": time.time() - started,
                    },
                )
    finally:
        environment.close()

    artifact_path = output / "modality_counterfactual_routes.npz"
    _atomic_npz(
        artifact_path,
        {
            "schema": np.asarray(SCHEMA),
            "training": np.asarray(False),
            "paired_record_index": np.asarray(selected, np.int16),
            "routes_vision_only": np.stack(routes_vision_only),
            "routes_state_only": np.stack(routes_state_only),
            "actions_vision_only": np.stack(actions_vision_only),
            "actions_state_only": np.stack(actions_state_only),
            "inference_ms_vision_only": np.stack(inference_ms_vision_only),
            "inference_ms_state_only": np.stack(inference_ms_state_only),
        },
    )
    manifest = {
        **run_config,
        "status": "complete",
        "records": len(selected),
        "inference_calls": 2 * int(paired["flow_noise"].shape[1]) * len(selected),
        "elapsed_s": time.time() - started,
        "artifact": str(artifact_path),
        "artifact_sha256": _sha256_file(artifact_path),
    }
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
