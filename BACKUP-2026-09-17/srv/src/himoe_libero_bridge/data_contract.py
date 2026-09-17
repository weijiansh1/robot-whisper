"""Auditable LIBERO tensor-contract checks against the released HiMoE transforms."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from himoe_libero_bridge.preprocess import resize_with_pad
from himoe_libero_bridge.protocol import ACTION_CHUNK_STEPS, ACTION_DIM, MODEL_ACTION_DIM, STATE_DIM
from himoe_libero_bridge.suites import SUITE_NAMES, get_suite


LIBERO_DATA_MASK = tuple([True] * ACTION_DIM + [False] * (MODEL_ACTION_DIM - ACTION_DIM))
STATE_WITH_VALIDITY_DIM = MODEL_ACTION_DIM * 2
PAPER_URL = "https://arxiv.org/html/2512.05693v2"


@dataclass(frozen=True)
class ArrayNormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_array(value: np.ndarray, first_values: int = 20) -> Dict[str, Any]:
    array = np.asarray(value)
    flat = array.reshape(-1)
    result = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": array_sha256(array),
        "first_values": [_json_scalar(item) for item in flat[:first_values]],
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        numeric = array.astype(np.float64, copy=False)
        result.update(
            {
                "min": float(np.min(numeric)),
                "max": float(np.max(numeric)),
                "mean": float(np.mean(numeric)),
                "std": float(np.std(numeric)),
            }
        )
    return result


def load_norm_stats(path: pathlib.Path) -> Dict[str, ArrayNormStats]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Could not parse normalization stats %s: %s" % (path, error))
    if not isinstance(document, Mapping):
        raise RuntimeError("Normalization stats root must be a mapping: %s" % path)

    result = {}
    for key, expected_dim in (("state", STATE_DIM), ("actions", ACTION_DIM)):
        raw = document.get(key)
        if not isinstance(raw, Mapping):
            raise RuntimeError("Normalization stats are missing %s: %s" % (key, path))
        values = {}
        for field in ("mean", "std", "q01", "q99"):
            array = np.asarray(raw.get(field), dtype=np.float64)
            if array.shape != (expected_dim,) or not np.all(np.isfinite(array)):
                raise RuntimeError(
                    "%s.%s must contain %d finite values: %s"
                    % (key, field, expected_dim, path)
                )
            values[field] = array
        if np.any(values["std"] <= 0.0):
            raise RuntimeError("%s.std must be strictly positive: %s" % (key, path))
        result[key] = ArrayNormStats(**values)
    return result


def normalize_raw(value: np.ndarray, stats: ArrayNormStats) -> np.ndarray:
    clipped = np.clip(np.asarray(value), stats.q01, stats.q99)
    return (clipped - stats.mean) / (stats.std + 1e-6)


def denormalize_raw(value: np.ndarray, stats: ArrayNormStats) -> np.ndarray:
    return np.asarray(value) * (stats.std + 1e-6) + stats.mean


def pad_libero_action(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[-1] != ACTION_DIM:
        raise ValueError("LIBERO action must end in dimension %d, got %s" % (ACTION_DIM, array.shape))
    result = np.zeros(array.shape[:-1] + (MODEL_ACTION_DIM,), dtype=np.float64)
    result[..., np.asarray(LIBERO_DATA_MASK, dtype=bool)] = array
    return result


def unpad_libero_action(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError(
            "Padded action must end in dimension %d, got %s" % (MODEL_ACTION_DIM, array.shape)
        )
    return array[..., np.asarray(LIBERO_DATA_MASK, dtype=bool)]


def pad_libero_state(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[-1] != STATE_DIM:
        raise ValueError("LIBERO state must end in dimension %d, got %s" % (STATE_DIM, array.shape))
    result = np.zeros(array.shape[:-1] + (STATE_WITH_VALIDITY_DIM,), dtype=np.float32)
    result[..., :STATE_DIM] = array
    result[..., MODEL_ACTION_DIM:][..., np.asarray(LIBERO_DATA_MASK, dtype=bool)] = 1.0
    return result


def released_camera_layout(base_image: np.ndarray, wrist_image: np.ndarray) -> Dict[str, Any]:
    base = np.asarray(base_image)
    wrist = np.asarray(wrist_image)
    if base.shape != wrist.shape:
        raise ValueError("Base and wrist images must have equal shapes")
    return {
        "image": {
            "base_0_rgb": base,
            "left_wrist_0_rgb": wrist,
            "right_wrist_0_rgb": np.zeros_like(base),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        },
    }


def make_synthetic_sample(stats: Mapping[str, ArrayNormStats]) -> Dict[str, Any]:
    rows, columns = np.indices((256, 256), dtype=np.uint16)
    base = np.stack(
        (columns % 256, rows % 256, (3 * rows + 5 * columns) % 256), axis=-1
    ).astype(np.uint8)
    wrist = np.stack(
        ((7 * rows + columns) % 256, (rows + 11 * columns) % 256, (rows ^ columns) % 256),
        axis=-1,
    ).astype(np.uint8)
    state_offsets = np.linspace(-0.25, 0.25, STATE_DIM, dtype=np.float64)
    state = stats["state"].mean + state_offsets * stats["state"].std
    time_offsets = np.linspace(-0.3, 0.3, ACTION_CHUNK_STEPS, dtype=np.float64)[:, None]
    dimension_offsets = np.linspace(0.5, 1.1, ACTION_DIM, dtype=np.float64)[None, :]
    actions = stats["actions"].mean + time_offsets * dimension_offsets * stats["actions"].std
    return {
        "source_base": base,
        "source_wrist": wrist,
        "post_rotation_base": np.ascontiguousarray(base[::-1, ::-1]),
        "post_rotation_wrist": np.ascontiguousarray(wrist[::-1, ::-1]),
        "state": state.astype(np.float32),
        "actions": actions.astype(np.float32),
        "prompt": "open the middle drawer of the cabinet",
    }


def _check(identifier: str, status: str, evidence: Any) -> Dict[str, Any]:
    if status not in ("pass", "fail", "needs_review", "blocked"):
        raise ValueError("Unknown check status: %s" % status)
    return {"id": identifier, "status": status, "evidence": evidence}


def _overall_status(checks: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(check["status"]) for check in checks}
    if "fail" in statuses:
        return "fail"
    if "needs_review" in statuses:
        return "needs_review"
    if "blocked" in statuses:
        return "blocked"
    return "pass"


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _new_artifact_dir(root: pathlib.Path, suite: str) -> pathlib.Path:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root.expanduser().resolve() / ("data-contract-%s-%s" % (suite, timestamp))
    path.mkdir(parents=True, exist_ok=False)
    return path


def _import_upstream(upstream_root: pathlib.Path) -> Tuple[Any, ...]:
    source_root = upstream_root.expanduser().resolve() / "src"
    if not source_root.is_dir():
        raise FileNotFoundError("Invalid HiMoE upstream root: %s" % upstream_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from moevla import transforms
    from moevla.models.tokenizer import PaligemmaTokenizer
    from moevla.policies.libero_policy import LiberoInputs, LiberoOutputs
    from moevla.shared import normalize as upstream_normalize

    return transforms, PaligemmaTokenizer, LiberoInputs, LiberoOutputs, upstream_normalize


def run_upstream_contract_audit(
    suite: str,
    cache_root: pathlib.Path,
    upstream_root: pathlib.Path,
    moevla_data_home: pathlib.Path,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    spec = get_suite(suite)
    checkpoint_dir = cache_root.expanduser().resolve() / "checkpoints" / spec.checkpoint_name
    stats_path = checkpoint_dir / spec.normalization_asset / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError("Missing normalization stats: %s" % stats_path)
    os.environ["MOEVLA_DATA_HOME"] = str(moevla_data_home.expanduser().resolve())

    stats = load_norm_stats(stats_path)
    sample = make_synthetic_sample(stats)
    transforms, tokenizer_class, libero_inputs_class, libero_outputs_class, upstream_normalize = (
        _import_upstream(upstream_root)
    )
    upstream_stats = upstream_normalize.load(str(checkpoint_dir / spec.normalization_asset))
    data_mask = list(int(value) for value in LIBERO_DATA_MASK)

    tokenizer = tokenizer_class(48)
    input_transform = libero_inputs_class(action_dim=MODEL_ACTION_DIM, data_mask=data_mask)
    normalizer = transforms.Normalize(data_mask=data_mask, norm_stats=upstream_stats)
    image_resizer = transforms.ResizeImages(224, 224)
    prompt_tokenizer = transforms.TokenizePrompt(tokenizer)
    padder = transforms.PadStatesAndActions(
        MODEL_ACTION_DIM, data_mask, is_libero_or_oxe=True
    )

    train_input = {
        "observation/image": sample["post_rotation_base"],
        "observation/wrist_image": sample["post_rotation_wrist"],
        "observation/state": sample["state"],
        "actions": sample["actions"],
        "prompt": sample["prompt"],
    }
    evaluation_base = resize_with_pad(sample["post_rotation_base"])
    evaluation_wrist = resize_with_pad(sample["post_rotation_wrist"])
    evaluation_input = {
        "observation/image": evaluation_base,
        "observation/wrist_image": evaluation_wrist,
        "observation/state": sample["state"],
        "prompt": sample["prompt"],
    }

    def apply_inputs(value: Dict[str, Any]) -> Dict[str, Any]:
        transformed = input_transform(value)
        transformed = normalizer(transformed)
        transformed = image_resizer(transformed)
        transformed = prompt_tokenizer(transformed)
        return padder(transformed)

    transformed_train = apply_inputs(train_input)
    transformed_evaluation = apply_inputs(evaluation_input)

    # Policy.infer carries the transformed state alongside sampled actions because the
    # released output transform unconditionally unpads both fields.
    output = {
        "state": np.array(transformed_train["state"], copy=True),
        "actions": np.array(transformed_train["actions"], copy=True),
    }
    output = transforms.UnpadStatesAndActions(
        MODEL_ACTION_DIM, data_mask, is_libero_or_oxe=True
    )(output)
    output = transforms.Unnormalize(data_mask=data_mask, norm_stats=upstream_stats)(output)
    output = libero_outputs_class(data_mask=data_mask)(output)
    round_trip = np.asarray(output["actions"])

    expected_action_norm = normalize_raw(sample["actions"], stats["actions"])
    expected_state_norm = normalize_raw(sample["state"], stats["state"])
    expected_action_padded = pad_libero_action(expected_action_norm)
    expected_state_padded = pad_libero_state(expected_state_norm)
    expected_camera = released_camera_layout(evaluation_base, evaluation_wrist)
    action_round_trip_error = float(np.max(np.abs(round_trip - sample["actions"])))

    comparisons = {
        "train_eval_base_image_exact": np.array_equal(
            transformed_train["image"]["base_0_rgb"],
            transformed_evaluation["image"]["base_0_rgb"],
        ),
        "train_eval_left_wrist_exact": np.array_equal(
            transformed_train["image"]["left_wrist_0_rgb"],
            transformed_evaluation["image"]["left_wrist_0_rgb"],
        ),
        "train_eval_right_wrist_exact": np.array_equal(
            transformed_train["image"]["right_wrist_0_rgb"],
            transformed_evaluation["image"]["right_wrist_0_rgb"],
        ),
        "train_eval_state_exact": np.array_equal(
            transformed_train["state"], transformed_evaluation["state"]
        ),
        "train_eval_token_ids_exact": np.array_equal(
            transformed_train["tokenized_prompt"],
            transformed_evaluation["tokenized_prompt"],
        ),
        "train_eval_token_mask_exact": np.array_equal(
            transformed_train["tokenized_prompt_mask"],
            transformed_evaluation["tokenized_prompt_mask"],
        ),
        "reference_action_padding_exact": np.array_equal(
            transformed_train["actions"], expected_action_padded
        ),
        "reference_state_padding_exact": np.array_equal(
            transformed_train["state"], expected_state_padded
        ),
        "reference_left_wrist_exact": np.array_equal(
            transformed_evaluation["image"]["left_wrist_0_rgb"],
            expected_camera["image"]["left_wrist_0_rgb"],
        ),
        "reference_right_wrist_exact": np.array_equal(
            transformed_evaluation["image"]["right_wrist_0_rgb"],
            expected_camera["image"]["right_wrist_0_rgb"],
        ),
    }
    common_pipeline_exact = all(
        value for key, value in comparisons.items() if key.startswith("train_eval_")
    )
    reference_exact = all(
        value for key, value in comparisons.items() if key.startswith("reference_")
    )
    invalid_action_values = transformed_train["actions"][..., ~np.asarray(LIBERO_DATA_MASK)]
    state_validity = transformed_train["state"][..., MODEL_ACTION_DIM:]
    state_valid_count = int(np.count_nonzero(state_validity))

    source_modules = (
        transforms,
        sys.modules[libero_inputs_class.__module__],
        sys.modules[tokenizer_class.__module__],
    )
    runtime_sources = {}
    for module in source_modules:
        path = pathlib.Path(inspect.getfile(module)).resolve()
        runtime_sources[module.__name__] = {"path": str(path), "sha256": file_sha256(path)}

    checks = [
        _check(
            "synthetic_train_eval_common_inputs",
            "pass" if common_pipeline_exact else "fail",
            comparisons,
        ),
        _check(
            "released_transform_reference_equivalence",
            "pass" if reference_exact else "fail",
            comparisons,
        ),
        _check(
            "action_round_trip",
            "pass" if action_round_trip_error < 1e-6 else "fail",
            {"max_abs_error": action_round_trip_error, "threshold": 1e-6},
        ),
        _check(
            "invalid_action_dimensions_zero",
            "pass" if np.count_nonzero(invalid_action_values) == 0 else "fail",
            {"invalid_dimension_nonzero_count": int(np.count_nonzero(invalid_action_values))},
        ),
        _check(
            "state_value_validity_width",
            "needs_review" if state_valid_count != STATE_DIM else "pass",
            {
                "raw_state_dimension": STATE_DIM,
                "state_value_slots": MODEL_ACTION_DIM,
                "validity_slots": MODEL_ACTION_DIM,
                "validity_true_count": state_valid_count,
                "released_data_mask_true_count": int(sum(LIBERO_DATA_MASK)),
                "observation": (
                    "The released transform stores all 8 raw state values in slots 0-7 but marks "
                    "only 7 corresponding validity positions true."
                ),
            },
        ),
        _check(
            "single_arm_wrist_slot",
            "needs_review",
            {
                "paper_contract": "single-arm visual input mapped to right-arm channel; left zero-padded",
                "released_transform": "real wrist in left_wrist_0_rgb; right_wrist_0_rgb zero and mask false",
                "paper_url": PAPER_URL,
                "interpretation": "Could be naming-only or a training/inference version mismatch; paired A/B is required.",
            },
        ),
        _check(
            "raw_demonstration_train_eval_equivalence",
            "blocked",
            {
                "reason": "No LIBERO demonstration dataset is present in the current cache.",
                "required_asset": "openvla/modified_libero_rlds plus an immutable conversion manifest",
            },
        ),
        _check(
            "normalization_reconstruction",
            "blocked",
            {
                "reason": "Published stats are present, but source demonstrations are absent.",
                "published_stats_sha256": file_sha256(stats_path),
            },
        ),
    ]

    arrays = {
        "source_base": sample["source_base"],
        "source_wrist": sample["source_wrist"],
        "post_rotation_base": sample["post_rotation_base"],
        "post_rotation_wrist": sample["post_rotation_wrist"],
        "evaluation_base": evaluation_base,
        "evaluation_wrist": evaluation_wrist,
        "raw_state": sample["state"],
        "normalized_state": expected_state_norm,
        "model_state_with_validity": transformed_train["state"],
        "raw_actions": sample["actions"],
        "normalized_actions": expected_action_norm,
        "model_actions": transformed_train["actions"],
        "round_trip_actions": round_trip,
        "token_ids": transformed_train["tokenized_prompt"],
        "token_mask": transformed_train["tokenized_prompt_mask"],
        "data_mask": transformed_train["data_mask"],
    }
    report = {
        "schema": "himoe-libero-data-contract-audit-v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Synthetic code-level train/eval equivalence; not demonstration-level equivalence",
        "suite": suite,
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "weights_expected_sha256": spec.weights_sha256,
            "normalization_stats_path": str(stats_path),
            "normalization_stats_sha256": file_sha256(stats_path),
        },
        "contract": {
            "raw_state_dimension": STATE_DIM,
            "raw_action_dimension": ACTION_DIM,
            "model_action_dimension": MODEL_ACTION_DIM,
            "model_state_dimension": STATE_WITH_VALIDITY_DIM,
            "data_mask": [int(value) for value in LIBERO_DATA_MASK],
            "actual_input_order": [
                "repack raw sample",
                "LiberoInputs camera/state/action mapping",
                "normalize raw 8D state and raw 7D action",
                "resize images and tokenize prompt",
                "pad state to 24D+24D validity and action to 24D",
            ],
            "actual_output_order": [
                "unpad 24D action to 7D",
                "unnormalize 7D action",
                "LiberoOutputs",
            ],
        },
        "comparisons": comparisons,
        "tensor_fingerprints": {key: summarize_array(value) for key, value in arrays.items()},
        "runtime_sources": runtime_sources,
        "checks": checks,
        "overall_status": _overall_status(checks),
    }
    return report, arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITE_NAMES, default="goal")
    parser.add_argument(
        "--cache-root", default="/home/jovyan/.cache/himoe-libero-bridge"
    )
    parser.add_argument(
        "--upstream-root",
        default="/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA",
    )
    parser.add_argument(
        "--moevla-data-home",
        default="/home/jovyan/.cache/himoe-libero-bridge/moevla-data",
    )
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit zero after writing the report; default exit 2 means unresolved review/blockers.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report, arrays = run_upstream_contract_audit(
        args.suite,
        pathlib.Path(args.cache_root),
        pathlib.Path(args.upstream_root),
        pathlib.Path(args.moevla_data_home),
    )
    artifact_dir = _new_artifact_dir(pathlib.Path(args.output_root), args.suite)
    arrays_path = artifact_dir / "synthetic-golden-contract.npz"
    np.savez_compressed(str(arrays_path), **arrays)
    report["artifacts"] = {
        "tensor_archive": str(arrays_path),
        "tensor_archive_sha256": file_sha256(arrays_path),
    }
    report_path = artifact_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    print("overall_status=%s" % report["overall_status"])
    if args.report_only:
        return 0
    if report["overall_status"] == "pass":
        return 0
    if report["overall_status"] == "fail":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
