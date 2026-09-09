#!/usr/bin/env python3
"""HB5/d0 pre-down expert-MLP activation audit on the runtime v3 grid.

Three quantities are intentionally kept distinct:

* router weights select and mix experts;
* ``m_e = SiLU(gate_proj_e(h)) * up_proj_e(h)`` is the expert's internal
  activation before ``down_proj``;
* ``E_e(h) = down_proj_e(m_e)`` is the post-down expert output stored by v3.

The pre-down coordinates of separately learned experts are not an aligned
basis.  Consequently this analysis never averages pre-down vectors, takes
cross-expert angles, or defines cancellation between them.  It computes scalar
summaries within each expert and only then forms permutation-invariant weighted
means and scalar dispersions across selected experts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr

import analyze_runtime_vector_v3 as runtime


HERE = Path(__file__).resolve().parent
DEFAULT_GRID = HERE / "runs" / "runtime-vector-grid-v3"
DEFAULT_OUT = (
    HERE / "analysis" / "raw-internal-activation-correction" / "observational-hb5-d0"
)
FEATURE_SCHEMA = "himoe-internal-mlp-features-v2"
SUMMARY_SCHEMA = "himoe-internal-mlp-audit-v2"
LAYER = 5
ACTIVE_EPSILON = 1e-3
GATE_CLOSED_THRESHOLD = -4.0
GATE_OPEN_THRESHOLD = 4.0
INTERNAL_SLOT_FEATURES = (
    "m_raw_l2",
    "m_raw_l1",
    "m_raw_signed_mean",
    "m_positive_fraction",
    "m_raw_linf",
)
DESCRIPTIVE_SLOT_FEATURES = (
    "m_raw_rms_fixed_width_rescaling",
    "m_active_fraction_abs_gt_1e-3",
    "gate_pre_closed_fraction_le_neg4",
    "gate_pre_open_fraction_ge_pos4",
)
ALL_INTERNAL_SLOT_FEATURES = INTERNAL_SLOT_FEATURES + DESCRIPTIVE_SLOT_FEATURES
POSTDOWN_SLOT_FEATURES = (
    "e_raw_l2",
    "e_raw_l1",
    "e_raw_signed_mean",
    "e_positive_fraction",
    "e_raw_linf",
)
AGGREGATIONS = ("weighted_mean", "weighted_scalar_dispersion")
INTERNAL_FAMILY = "hb5_predown_sorted_raw_l2_exact40"
CONTROL_FAMILY = "hb5_postdown_sorted_raw_l2_exact40"
FEATURE_WIDTH = runtime.N_ACTION_TOKENS * 4
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260823
EXACT_B1_FAMILIES = dict(runtime.BASELINE_TIERS)["b1_operational"]


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def source_signature(data: runtime.RuntimeVectorData) -> dict[str, Any]:
    """Fingerprint every stored field used by reconstruction or inference."""
    return {
        "rows": len(data.x_traj),
        "sample_id_sha256": _sha256_array(data.sample_id.astype("U")),
        "input_hidden_sha256": _sha256_array(data.input_hidden),
        "selected_ids_sha256": _sha256_array(data.selected_ids),
        "selected_weights_sha256": _sha256_array(data.selected_weights),
        "selected_raw_sha256": _sha256_array(data.selected_raw),
        "x_traj_sha256": _sha256_array(data.x_traj),
        "checkpoint_ids": sorted(map(str, np.unique(data.checkpoint_group))),
    }


def discover_checkpoints(grid_root: Path) -> dict[str, Path]:
    """Resolve checkpoint hashes from immutable capture metadata."""
    result: dict[str, Path] = {}
    sizes: dict[str, int] = {}
    for metadata_path in sorted(
        grid_root.glob("client/*/state-*/server_metadata.json")
    ):
        metadata = json.loads(metadata_path.read_text())
        digest = str(metadata["checkpoint_sha256"])
        checkpoint = Path(metadata["checkpoint"]) / "pytorch_model.pth"
        expected_size = int(metadata["checkpoint_bytes"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint.stat().st_size != expected_size:
            raise RuntimeError(
                f"checkpoint size disagrees with capture metadata: {checkpoint}"
            )
        previous = result.setdefault(digest, checkpoint.resolve())
        if previous != checkpoint.resolve():
            raise RuntimeError(f"checkpoint hash {digest} maps to multiple files")
        previous_size = sizes.setdefault(digest, expected_size)
        if previous_size != expected_size:
            raise RuntimeError(f"checkpoint hash {digest} maps to multiple sizes")
    if not result:
        raise RuntimeError(f"no checkpoint metadata found below {grid_root}")
    return result


def vector_statistics(value: np.ndarray) -> np.ndarray:
    """Five matched post-down summaries, preserving raw checkpoint units."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError("expert vectors need a non-empty channel axis")
    l2 = np.sqrt(np.sum(np.square(array), axis=-1))
    l1 = np.sum(np.abs(array), axis=-1)
    mean = np.mean(array, axis=-1)
    positive = np.mean(array > 0.0, axis=-1)
    linf = np.max(np.abs(array), axis=-1)
    return np.stack((l2, l1, mean, positive, linf), axis=-1)


def internal_statistics(
    gate_pre: np.ndarray,
    intermediate: np.ndarray,
) -> np.ndarray:
    """Frozen raw-L2 family plus descriptive saturation/active summaries."""
    gate = np.asarray(gate_pre, dtype=np.float64)
    value = np.asarray(intermediate, dtype=np.float64)
    if gate.shape != value.shape or value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError("gate and intermediate tensors must share a channel axis")
    l2 = np.sqrt(np.sum(np.square(value), axis=-1))
    l1 = np.sum(np.abs(value), axis=-1)
    mean = np.mean(value, axis=-1)
    positive = np.mean(value > 0.0, axis=-1)
    linf = np.max(np.abs(value), axis=-1)
    rms = l2 / np.sqrt(value.shape[-1])
    active = np.mean(np.abs(value) > ACTIVE_EPSILON, axis=-1)
    closed = np.mean(gate <= GATE_CLOSED_THRESHOLD, axis=-1)
    opened = np.mean(gate >= GATE_OPEN_THRESHOLD, axis=-1)
    return np.stack(
        (l2, l1, mean, positive, linf, rms, active, closed, opened), axis=-1
    )


def aggregate_selected_scalars(
    slot_statistics: np.ndarray,
    selected_weights: np.ndarray,
) -> np.ndarray:
    """Aggregate expert-local scalars without depending on top-k slot order."""
    stats = np.asarray(slot_statistics, dtype=np.float64)
    weight = np.asarray(selected_weights, dtype=np.float64)
    if stats.ndim != 4 or weight.shape != stats.shape[:3]:
        raise ValueError("statistics must be [sample,token,slot,metric]")
    mass = weight.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("selected weights need positive total mass")
    probability = weight / mass
    mean = np.sum(probability[..., None] * stats, axis=2)
    variance = np.sum(
        probability[..., None] * np.square(stats - mean[:, :, None]), axis=2
    )
    dispersion = np.sqrt(np.maximum(variance, 0.0))
    return np.concatenate((mean, dispersion), axis=-1)


def sorted_raw_l2_feature(slot_statistics: np.ndarray) -> np.ndarray:
    """Keep all four raw L2 values while removing expert/slot labels."""
    stats = np.asarray(slot_statistics)
    if stats.ndim != 4 or stats.shape[2] != 4 or stats.shape[3] < 1:
        raise ValueError("slot statistics must be [sample,token,4,metric]")
    return np.sort(stats[..., 0], axis=2)


def _weight_key(expert: int, projection: str) -> str:
    return (
        f"paligemma_with_expert.gemma_expert.layers.{LAYER}.mlp."
        f"experts.{expert}.{projection}.weight"
    )


def reconstruct_checkpoint_internal(
    data: runtime.RuntimeVectorData,
    rows: np.ndarray,
    checkpoint: Path,
    threads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct selected ``gate``, ``up``, and ``m`` from stored HB5 input."""
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(threads)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    hidden = torch.from_numpy(
        np.ascontiguousarray(data.input_hidden[rows], dtype=np.float32)
    ).reshape(-1, data.input_hidden.shape[-1])
    ids = torch.from_numpy(
        np.ascontiguousarray(data.selected_ids[rows], dtype=np.int64)
    ).reshape(-1, data.selected_ids.shape[-1])
    reference = torch.from_numpy(
        np.ascontiguousarray(data.selected_raw[rows], dtype=np.float32)
    ).reshape(-1, data.selected_ids.shape[-1], data.selected_raw.shape[-1])
    slot_stats = torch.full(
        (*ids.shape, len(ALL_INTERNAL_SLOT_FEATURES)),
        float("nan"),
        dtype=torch.float32,
    )
    squared_error = 0.0
    absolute_error = 0.0
    reference_squared = 0.0
    elements = 0
    max_abs_error = 0.0
    selected_counts: dict[str, int] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for expert in range(runtime.N_EXPERTS):
            selected = torch.nonzero(ids == expert, as_tuple=False)
            if selected.numel() == 0:
                selected_counts[str(expert)] = 0
                continue
            token, slot = selected[:, 0], selected[:, 1]
            selected_counts[str(expert)] = int(len(token))
            x = hidden[token]
            gate_weight = state[_weight_key(expert, "gate_proj")].float()
            up_weight = state[_weight_key(expert, "up_proj")].float()
            down_weight = state[_weight_key(expert, "down_proj")].float()
            gate_pre = functional.linear(x, gate_weight)
            up = functional.linear(x, up_weight)
            intermediate = functional.silu(gate_pre) * up
            output = functional.linear(intermediate, down_weight)
            computed = torch.stack(
                (
                    intermediate.square().sum(-1).sqrt(),
                    intermediate.abs().sum(-1),
                    intermediate.mean(-1),
                    (intermediate > 0.0).float().mean(-1),
                    intermediate.abs().amax(-1),
                    intermediate.square().mean(-1).sqrt(),
                    (intermediate.abs() > ACTIVE_EPSILON).float().mean(-1),
                    (gate_pre <= GATE_CLOSED_THRESHOLD).float().mean(-1),
                    (gate_pre >= GATE_OPEN_THRESHOLD).float().mean(-1),
                ),
                dim=-1,
            )
            slot_stats[token, slot] = computed
            error = output - reference[token, slot]
            squared_error += float(error.square().sum())
            absolute_error += float(error.abs().sum())
            reference_squared += float(reference[token, slot].square().sum())
            elements += int(error.numel())
            max_abs_error = max(max_abs_error, float(error.abs().max()))
            del x, gate_weight, up_weight, down_weight
            del gate_pre, up, intermediate, output, computed, error
    if not bool(torch.isfinite(slot_stats).all()):
        raise RuntimeError("not every selected expert slot was reconstructed")
    if elements == 0:
        raise RuntimeError("checkpoint subset has no selected expert slots")
    validation = {
        "checkpoint": str(checkpoint),
        "rows": int(len(rows)),
        "selected_slots": int(sum(selected_counts.values())),
        "selected_slots_by_expert": selected_counts,
        "postdown_vs_v3_fp16_max_abs_error": max_abs_error,
        "postdown_vs_v3_fp16_rmse": float(np.sqrt(squared_error / elements)),
        "postdown_vs_v3_fp16_mae": float(absolute_error / elements),
        "postdown_reference_rms": float(np.sqrt(reference_squared / elements)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "precision_note": (
            "float32 recomputation from fp16-stored HB input and bf16 checkpoint; "
            "v3 reference is fp16, so this validates reconstruction identity but "
            "cannot be bit-exact to the original bf16 runtime input"
        ),
    }
    result = slot_stats.numpy().reshape(
        len(rows),
        runtime.N_ACTION_TOKENS,
        data.selected_ids.shape[-1],
        len(ALL_INTERNAL_SLOT_FEATURES),
    )
    del state, hidden, ids, reference, slot_stats
    gc.collect()
    return result, validation


def reconstruct_features(
    data: runtime.RuntimeVectorData,
    checkpoints: dict[str, Path],
    threads: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Build low-dimensional features without altering the v3 source store."""
    internal_slot = np.full(
        (
            len(data.x_traj),
            runtime.N_ACTION_TOKENS,
            data.selected_ids.shape[-1],
            len(ALL_INTERNAL_SLOT_FEATURES),
        ),
        np.nan,
        dtype=np.float32,
    )
    validation: dict[str, Any] = {}
    for checkpoint_id in np.unique(data.checkpoint_group.astype(str)):
        if checkpoint_id not in checkpoints:
            raise RuntimeError(f"no checkpoint path for captured hash {checkpoint_id}")
        rows = np.flatnonzero(data.checkpoint_group.astype(str) == checkpoint_id)
        reconstructed, row = reconstruct_checkpoint_internal(
            data, rows, checkpoints[checkpoint_id], threads
        )
        internal_slot[rows] = reconstructed
        validation[checkpoint_id] = row
    if not np.all(np.isfinite(internal_slot)):
        raise RuntimeError("internal reconstruction left non-finite values")
    postdown_slot = vector_statistics(data.selected_raw).astype(np.float32)
    # Primary features are absolute magnitudes only. Sorting retains all four
    # selected values while removing top-k slot and expert-ID labels; router
    # weights do not enter this feature.
    internal = sorted_raw_l2_feature(internal_slot).astype(np.float32)
    control = sorted_raw_l2_feature(postdown_slot).astype(np.float32)
    if internal.shape != control.shape or internal.shape[1:] != (
        runtime.N_ACTION_TOKENS,
        data.selected_ids.shape[-1],
    ):
        raise RuntimeError(
            f"internal/control exact-dimension contract failed: {internal.shape}, "
            f"{control.shape}"
        )
    return internal, control, validation, internal_slot


def save_feature_cache(
    out_dir: Path,
    data: runtime.RuntimeVectorData,
    internal: np.ndarray,
    control: np.ndarray,
    slot: np.ndarray,
    validation: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "features.npz",
        sample_id=data.sample_id.astype("U"),
        internal_exact=internal,
        matched_postdown_exact=control,
        internal_slot_statistics=slot,
    )
    metadata = {
        "schema": FEATURE_SCHEMA,
        "source_signature": source_signature(data),
        "internal_slot_features": list(INTERNAL_SLOT_FEATURES),
        "descriptive_slot_features": list(DESCRIPTIVE_SLOT_FEATURES),
        "postdown_slot_features": list(POSTDOWN_SLOT_FEATURES),
        "primary_aggregation": "sort four raw L2 values per token; no router weights",
        "secondary_aggregations": [
            "unweighted mean/std",
            "router-weighted mean/scalar dispersion",
        ],
        "feature_width": FEATURE_WIDTH,
        "reconstruction_validation": validation,
    }
    (out_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def load_feature_cache(
    out_dir: Path,
    data: runtime.RuntimeVectorData,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray] | None:
    archive_path = out_dir / "features.npz"
    metadata_path = out_dir / "feature_metadata.json"
    if not archive_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != FEATURE_SCHEMA:
        return None
    if metadata.get("source_signature") != source_signature(data):
        return None
    with np.load(archive_path) as archive:
        if not np.array_equal(
            archive["sample_id"].astype(str), data.sample_id.astype(str)
        ):
            return None
        internal = np.asarray(archive["internal_exact"], dtype=np.float32)
        control = np.asarray(archive["matched_postdown_exact"], dtype=np.float32)
        slot = np.asarray(archive["internal_slot_statistics"], dtype=np.float32)
    return internal, control, metadata["reconstruction_validation"], slot


def _within_state_spearman(
    value: np.ndarray,
    target: np.ndarray,
    states: np.ndarray,
) -> float | None:
    rows = []
    state_axis = np.asarray(states).astype(str)
    scalar = np.asarray(value, dtype=np.float64)
    response = np.asarray(target, dtype=np.float64).reshape(len(target), -1)[:, 0]
    for state in np.unique(state_axis):
        index = state_axis == state
        if np.std(scalar[index]) <= 1e-15 or np.std(response[index]) <= 1e-15:
            continue
        correlation = spearmanr(scalar[index], response[index]).statistic
        if np.isfinite(correlation):
            rows.append(float(correlation))
    return None if not rows else float(np.mean(rows))


def raw_scale_table(
    data: runtime.RuntimeVectorData,
    internal_slot: np.ndarray,
) -> dict[str, Any]:
    """Report unstandardised checkpoint-unit amplitudes before any readout."""
    result: dict[str, Any] = {}
    for task in np.unique(data.task_group.astype(str)):
        index = data.task_group.astype(str) == task
        slot = internal_slot[index]
        names = ALL_INTERNAL_SLOT_FEATURES
        task_row: dict[str, Any] = {}
        for feature_index, name in enumerate(names):
            flattened = slot[..., feature_index].reshape(-1)
            task_row[name] = {
                "mean": float(np.mean(flattened)),
                "std": float(np.std(flattened)),
                "q05": float(np.quantile(flattened, 0.05)),
                "q50": float(np.quantile(flattened, 0.50)),
                "q95": float(np.quantile(flattened, 0.95)),
            }
        target_net_rms = np.sqrt(
            np.mean(np.square(data.secondary_target[index]), axis=1)
        )
        task_row["within_state_spearman_vs_remaining_correction_rms"] = {
            name: _within_state_spearman(
                slot[..., feature_index].mean(axis=(1, 2)),
                target_net_rms[:, None],
                data.state_group[index],
            )
            for feature_index, name in enumerate(names)
        }
        task_row["rows"] = int(np.sum(index))
        result[str(task)] = task_row
    return result


def exact_b1_feature_families(
    data: runtime.RuntimeVectorData,
) -> dict[str, np.ndarray]:
    """Return the eight unprojected B1 families in their original coordinates."""
    selected_one = runtime.scatter_selected_by_expert(
        data.selected_ids, np.ones_like(data.selected_weights)
    )
    selected_weight = runtime.scatter_selected_by_expert(
        data.selected_ids, data.selected_weights
    )
    routed = np.sum(
        data.selected_raw.astype(np.float64)
        * data.selected_weights.astype(np.float64)[..., None],
        axis=2,
    )
    values = {
        "x0": data.x_traj[:, 0],
        "x1": data.x_traj[:, 1],
        "hb5_input_hidden": data.input_hidden,
        "hb5_shared_output": data.shared_output,
        "hb5_router_probs": data.router_probs,
        "hb5_selected_id_one_hot": selected_one,
        "hb5_selected_weight_by_id": selected_weight,
        "hb5_actual_merged_routed": routed,
    }
    if tuple(values) != EXACT_B1_FAMILIES:
        raise RuntimeError("exact B1 family order differs from runtime-v3 B1")
    return {
        name: np.asarray(value, dtype=np.float64).reshape(len(data.x_traj), -1)
        for name, value in values.items()
    }


def _one_standardized_linear_kernel(
    value: np.ndarray,
    fit: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    mean = array[fit].mean(axis=0)
    scale = array[fit].std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    left_value = (array[left] - mean) / scale
    right_value = (array[right] - mean) / scale
    return left_value @ right_value.T / array.shape[1]


def exact_additive_kernel(
    families: dict[str, np.ndarray],
    fit: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Equal-family additive linear kernel, with train-only standardisation."""
    kernels = [
        _one_standardized_linear_kernel(value, fit, left, right)
        for value in families.values()
    ]
    return np.mean(kernels, axis=0)


def add_one_equal_family_kernel(
    b1_kernel: np.ndarray,
    feature_kernel: np.ndarray,
) -> np.ndarray:
    """Append one family at the same weight as each of B1's eight families."""
    family_count = len(EXACT_B1_FAMILIES)
    return (family_count * b1_kernel + feature_kernel) / (family_count + 1)


def _kernel_prediction(
    train_kernel: np.ndarray,
    apply_kernel: np.ndarray,
    train_target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    target = np.asarray(train_target, dtype=np.float64)
    mean = target.mean(axis=0)
    dual = np.linalg.solve(
        train_kernel + float(alpha) * np.eye(len(train_kernel)), target - mean
    )
    return apply_kernel @ dual + mean


def cross_validated_exact_arms(
    b1_families: dict[str, np.ndarray],
    additions: dict[str, np.ndarray | None],
    target: np.ndarray,
    state_group: np.ndarray,
    seed_group: np.ndarray,
    outer_splits: int,
    split_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]], np.ndarray]:
    """Nested state+candidate-disjoint KRR for B1 and every exact addition."""
    n = len(target)
    if any(value is not None and len(value) != n for value in additions.values()):
        raise ValueError("every exact addition must share the target row axis")
    outer = runtime.state_seed_disjoint_folds(
        state_group, seed_group, outer_splits, split_seed
    )
    predictions = {
        name: np.full_like(target, np.nan, dtype=np.float64) for name in additions
    }
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in additions}
    fold_id = np.full(n, -1, dtype=np.int64)
    for outer_index, fold in enumerate(outer):
        train_states = np.asarray(state_group)[fold.train]
        train_seeds = np.asarray(seed_group)[fold.train]
        inner_count = min(
            max(2, outer_splits - 1),
            len(np.unique(train_states)),
            len(np.unique(train_seeds)),
        )
        inner = runtime.state_seed_disjoint_folds(
            train_states,
            train_seeds,
            inner_count,
            split_seed + 1009 + outer_index,
        )
        losses = {
            name: np.zeros(len(runtime.RIDGE_ALPHAS), dtype=np.float64)
            for name in additions
        }
        elements = 0
        for inner_fold in inner:
            inner_train = fold.train[inner_fold.train]
            inner_test = fold.train[inner_fold.test]
            b1_train = exact_additive_kernel(
                b1_families, inner_train, inner_train, inner_train
            )
            b1_test = exact_additive_kernel(
                b1_families, inner_train, inner_test, inner_train
            )
            for name, addition in additions.items():
                train_kernel, test_kernel = b1_train, b1_test
                if addition is not None:
                    train_kernel = add_one_equal_family_kernel(
                        train_kernel,
                        _one_standardized_linear_kernel(
                            addition, inner_train, inner_train, inner_train
                        ),
                    )
                    test_kernel = add_one_equal_family_kernel(
                        test_kernel,
                        _one_standardized_linear_kernel(
                            addition, inner_train, inner_test, inner_train
                        ),
                    )
                for alpha_index, alpha in enumerate(runtime.RIDGE_ALPHAS):
                    prediction = _kernel_prediction(
                        train_kernel,
                        test_kernel,
                        target[inner_train],
                        alpha,
                    )
                    losses[name][alpha_index] += float(
                        np.sum(np.square(prediction - target[inner_test]))
                    )
            elements += int(np.prod(target[inner_test].shape))
        b1_train = exact_additive_kernel(
            b1_families, fold.train, fold.train, fold.train
        )
        b1_test = exact_additive_kernel(b1_families, fold.train, fold.test, fold.train)
        for name, addition in additions.items():
            selected_alpha = float(
                runtime.RIDGE_ALPHAS[int(np.argmin(losses[name] / elements))]
            )
            train_kernel, test_kernel = b1_train, b1_test
            if addition is not None:
                train_kernel = add_one_equal_family_kernel(
                    train_kernel,
                    _one_standardized_linear_kernel(
                        addition, fold.train, fold.train, fold.train
                    ),
                )
                test_kernel = add_one_equal_family_kernel(
                    test_kernel,
                    _one_standardized_linear_kernel(
                        addition, fold.train, fold.test, fold.train
                    ),
                )
            predictions[name][fold.test] = _kernel_prediction(
                train_kernel,
                test_kernel,
                target[fold.train],
                selected_alpha,
            )
            rows[name].append(
                {
                    "fold": outer_index,
                    "train_rows": len(fold.train),
                    "test_rows": len(fold.test),
                    "held_states": list(fold.held_states),
                    "held_seeds": list(fold.held_seeds),
                    "selected_alpha": selected_alpha,
                }
            )
        fold_id[fold.test] = outer_index
    if any(not np.all(np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("exact-kernel cross-fitting left non-finite predictions")
    if np.any(fold_id < 0):
        raise RuntimeError("exact-kernel cross-fitting did not cover every row")
    return predictions, rows, fold_id


def _evaluate_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    data: runtime.RuntimeVectorData,
    folds: list[dict[str, Any]],
    fold_id: np.ndarray,
) -> dict[str, Any]:
    row_mse = np.mean(np.square(prediction - target), axis=1)
    states = data.state_group.astype(str)
    seeds = data.seed_group.astype(str)
    error_grid = {
        state: {
            seed: float(np.mean(row_mse[(states == state) & (seeds == seed)]))
            for seed in np.unique(seeds)
        }
        for state in np.unique(states)
    }
    correction_geometry = runtime.geometry_spearman(prediction, target, states, fold_id)
    predicted_final = (
        data.x_traj[:, 1, :, : runtime.N_LIVE_DIMS].reshape(len(target), -1)
        + prediction
    )
    actual_final = data.x_traj[:, 10, :, : runtime.N_LIVE_DIMS].reshape(len(target), -1)
    return {
        "held_out_rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "secondary_correction_geometry_spearman": correction_geometry,
        "held_out_geometry_spearman": runtime.geometry_spearman(
            predicted_final, actual_final, states, fold_id
        ),
        "secondary_net_rms_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.sqrt(np.mean(np.square(prediction), axis=1))
                        - np.sqrt(np.mean(np.square(target), axis=1))
                    )
                )
            )
        ),
        "paired_squared_error_state_by_candidate": error_grid,
        "folds": folds,
        "outer_fold_id_by_sample": fold_id.tolist(),
    }


def _paired_grids(metric: dict[str, Any]) -> np.ndarray:
    grid = metric["paired_squared_error_state_by_candidate"]
    states = sorted(grid)
    candidates = sorted(grid[states[0]])
    return np.asarray(
        [[grid[state][candidate] for candidate in candidates] for state in states],
        dtype=np.float64,
    )


def _exploratory_commonality(task_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    triplets = []
    base_relative, control_relative = [], []
    for row in task_rows:
        base = _paired_grids(row["b1_exact_additive"])
        control = _paired_grids(row["b1_plus_matched_postdown_sorted_l2_exact40"])
        internal = _paired_grids(row["b1_plus_predown_sorted_l2_exact40"])
        triplets.append((base, control, internal))
        base_rmse = np.sqrt(np.mean(base))
        control_rmse = np.sqrt(np.mean(control))
        internal_rmse = np.sqrt(np.mean(internal))
        base_relative.append(float((base_rmse - internal_rmse) / base_rmse))
        control_relative.append(float((control_rmse - internal_rmse) / control_rmse))
    state_draw, candidate_draw = runtime.bootstrap_axis_indices(
        len(triplets),
        runtime.N_STATES_PER_TASK,
        runtime.N_SEEDS_PER_TASK,
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_SEED,
    )
    bootstrap = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        effects = []
        for task_index, (base, _control, internal) in enumerate(triplets):
            sampled = np.ix_(state_draw[draw, task_index], candidate_draw[draw])
            base_rmse = np.sqrt(np.mean(base[sampled]))
            internal_rmse = np.sqrt(np.mean(internal[sampled]))
            effects.append((base_rmse - internal_rmse) / base_rmse)
        bootstrap[draw] = np.mean(effects)
    ci = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "status": "exploratory screen; not a preregistered pruning gate",
        "interval_scope": (
            "conditional two-axis resampling interval for the fixed cross-fit "
            "predictions; not a population-level confidence interval"
        ),
        "per_task_relative_rmse_improvement_vs_b1": base_relative,
        "equal_task_macro_relative_rmse_improvement_vs_b1": float(
            np.mean(base_relative)
        ),
        "per_task_relative_rmse_improvement_vs_matched_control": control_relative,
        "equal_task_macro_relative_rmse_improvement_vs_matched_control": float(
            np.mean(control_relative)
        ),
        "paired_state_candidate_bootstrap_ci95_vs_b1": list(map(float, ci)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "all_five_tasks_better_than_b1": bool(np.all(np.asarray(base_relative) > 0.0)),
        "all_five_tasks_better_than_matched_control": bool(
            np.all(np.asarray(control_relative) > 0.0)
        ),
        "bootstrap_ci_lower_above_zero": bool(float(ci[0]) > 0.0),
    }


def analyze(
    data: runtime.RuntimeVectorData,
    internal: np.ndarray,
    control: np.ndarray,
    internal_slot: np.ndarray,
    validation: dict[str, Any],
    outer_splits: int,
    split_seed: int,
) -> dict[str, Any]:
    expected = (len(data.x_traj), runtime.N_ACTION_TOKENS, 4)
    if internal.shape != expected or control.shape != expected:
        raise ValueError(f"primary exact40 features must have shape {expected}")
    task_names = sorted(map(str, np.unique(data.task_group.astype(str))))
    if set(task_names) != set(runtime.EXPECTED_TASKS):
        raise runtime.AdmissionError("the audit requires the frozen five-task grid")
    unweighted_mean = internal_slot.mean(axis=2)
    unweighted_std = internal_slot.std(axis=2)
    unweighted_secondary = np.concatenate((unweighted_mean, unweighted_std), axis=-1)
    router_weighted_secondary = aggregate_selected_scalars(
        internal_slot, data.selected_weights
    )
    per_task: dict[str, Any] = {}
    for task in task_names:
        index = np.flatnonzero(data.task_group.astype(str) == task)
        task_data = runtime.subset_data(data, index)
        admission = runtime.validate_admission(task_data, outer_splits)
        b1 = exact_b1_feature_families(task_data)
        additions = {
            "b1_exact_additive": None,
            "b1_plus_matched_postdown_sorted_l2_exact40": control[index].reshape(
                len(index), -1
            ),
            "b1_plus_predown_sorted_l2_exact40": internal[index].reshape(
                len(index), -1
            ),
            "secondary_b1_plus_unweighted_companions": unweighted_secondary[
                index
            ].reshape(len(index), -1),
            "secondary_b1_plus_router_weighted_companions": (
                router_weighted_secondary[index].reshape(len(index), -1)
            ),
        }
        target = task_data.secondary_target
        predictions, fold_rows, fold_id = cross_validated_exact_arms(
            b1,
            additions,
            target,
            task_data.state_group,
            task_data.seed_group,
            outer_splits,
            split_seed,
        )
        result = {
            name: _evaluate_prediction(
                prediction, target, task_data, fold_rows[name], fold_id
            )
            for name, prediction in predictions.items()
        }
        per_task[task] = {
            "admission": admission,
            "checkpoint": str(np.unique(task_data.checkpoint_group.astype(str))[0]),
            **result,
        }
    task_rows = [per_task[task] for task in task_names]
    commonality = _exploratory_commonality(task_rows)
    commonality["task_order"] = task_names
    commonality["b1_rmse"] = [
        row["b1_exact_additive"]["held_out_rmse"] for row in task_rows
    ]
    commonality["matched_control_rmse"] = [
        row["b1_plus_matched_postdown_sorted_l2_exact40"]["held_out_rmse"]
        for row in task_rows
    ]
    commonality["internal_rmse"] = [
        row["b1_plus_predown_sorted_l2_exact40"]["held_out_rmse"] for row in task_rows
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "scientific_status": "exploratory held-out association; no causal claim",
        "protocol_authority": str(
            (
                HERE
                / "analysis"
                / "raw-internal-activation-correction"
                / "PREREGISTRATION.md"
            ).resolve()
        ),
        "clock": (
            "HB5/d0 only. This audit cannot test whether confidence increases "
            "across denoising rounds."
        ),
        "quantity_distinction": {
            "router": "w_e: selection/mixing probability, not activation",
            "pre_down_internal": "m_e = SiLU(gate_proj_e(h)) * up_proj_e(h)",
            "post_down_output": "v_e = down_proj_e(m_e), stored by runtime v3",
            "routed_contribution": "c_e = w_e * v_e",
        },
        "coordinate_restriction": (
            "pre-down coordinates are separately learned per expert; no cross-expert "
            "vector angle, vector mean, or cancellation is computed"
        ),
        "raw_unit_policy": (
            "primary size is absolute L2(m_e) in checkpoint units; no hidden, "
            "shared, routed, post-MoE, or router-weight normalization. RMS is only "
            "the fixed-width identity L2/sqrt(1024)."
        ),
        "target": {
            "primary": "flatten10x7(live7(x10-x1))",
            "metrics": "held-out vector RMSE and correction/final geometry Spearman",
            "future_path_energy": (
                "excluded from the claim gate because it is nearly determined by "
                "the already observed d0 velocity"
            ),
        },
        "feature_contract": {
            "primary": "sort four absolute L2(m_e) values at each token",
            "matched_control": "sort four absolute L2(v_e) values at each token",
            "exact_feature_width": FEATURE_WIDTH,
            "uses_router_weight": False,
            "slot_order_invariant": True,
            "expert_id_permutation_invariant": True,
            "internal_slot_features": list(INTERNAL_SLOT_FEATURES),
            "descriptive_only_slot_features": list(DESCRIPTIVE_SLOT_FEATURES),
            "secondary_only": [
                "unweighted mean/std of all expert-local scalar summaries",
                "router-weighted mean/scalar dispersion",
            ],
        },
        "baseline": {
            "name": "exact additive-kernel B1",
            "families": list(EXACT_B1_FAMILIES),
            "family_kernel": (
                "train-fold-standardized exact linear kernel divided by family width"
            ),
            "aggregation": "equal mean of eight family kernels",
            "feature_addition": "(8 * K_B1 + K_feature) / 9",
            "random_projection": None,
        },
        "split": {
            "scheme": "nested Cartesian state+candidate group-disjoint CV",
            "outer_splits_per_axis": outer_splits,
            "split_seed": split_seed,
            "target_centering": "training fold only",
            "kernel_standardization": "training fold only",
        },
        "reconstruction_validation": validation,
        "raw_checkpoint_unit_scale_by_task": raw_scale_table(data, internal_slot),
        "per_task": per_task,
        "exploratory_commonality": commonality,
        "pruning_claim": None,
    }


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def write_report(summary: dict[str, Any], out_dir: Path) -> None:
    result = summary["exploratory_commonality"]
    tasks = result["task_order"]
    scale = summary["raw_checkpoint_unit_scale_by_task"]
    lines = [
        "# HB5/d0 expert-MLP 内部激活审计",
        "",
        "## 先区分三层",
        "",
        "1. **Router weight**：选择和混合专家的权重，不是专家激活。",
        "2. **`down_proj` 前内部激活**：`m_e = SiLU(gate_proj_e(h)) * up_proj_e(h)`；这是本实验新增测量的量。",
        "3. **`down_proj` 后专家输出**：`E_e(h) = down_proj_e(m_e)`；v3 原先保存的是这个量，不能再把它简称为 `m`。",
        "",
        "主幅值是 `||m_e||₂` 的绝对 checkpoint 单位，不乘也不除 router weight，也不除 hidden、shared、routed 或 post-MoE。RMS 仅是固定宽度恒等换算 `L2/sqrt(1024)`。训练折标准化只用于核岭回归的数值条件，不改变下表原始量级。",
        "",
        "不同专家的 pre-down 神经元没有可识别的共同坐标轴，因此不能对 `m_e` 跨专家算向量夹角或 cancellation。主块只保留每个专家内部的 raw L2，并对四个值排序；weighted mean/dispersion 只属于 secondary 描述，不进入主判定。",
        "",
        "## 原始量级",
        "",
        "| task | raw L2 | raw L1 | RMS=L2/sqrt(1024) | active fraction | gate closed | gate open |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        row = scale[task]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                task,
                _fmt(row["m_raw_l2"]["mean"]),
                _fmt(row["m_raw_l1"]["mean"]),
                _fmt(row["m_raw_rms_fixed_width_rescaling"]["mean"]),
                _fmt(row["m_active_fraction_abs_gt_1e-3"]["mean"]),
                _fmt(row["gate_pre_closed_fraction_le_neg4"]["mean"]),
                _fmt(row["gate_pre_open_fraction_ge_pos4"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## 主结果",
            "",
            "主目标是 d0 之后的完整 `flatten10x7(x10-x1)` remaining correction。每个任务都用 state 与 candidate 双重不相交的 nested CV。B1 的八个原始 family 用完整坐标构造等权 additive linear kernel，没有 CountSketch。",
            "",
            "主块是每 token 四个排序后的 raw `||m_e||₂`，共 exact40；它完全不使用 router weight。matched control 对 post-down `||E_e(h)||₂` 做同样的 exact40 构造。",
            "",
            "| task | B1 RMSE | + post-down exact40 | + pre-down exact40 | rel. gain vs B1 | correction geometry rho |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    relative = result["per_task_relative_rmse_improvement_vs_b1"]
    for task, base, control, internal, gain in zip(
        tasks,
        result["b1_rmse"],
        result["matched_control_rmse"],
        result["internal_rmse"],
        relative,
        strict=True,
    ):
        rho = summary["per_task"][task]["b1_plus_predown_sorted_l2_exact40"][
            "secondary_correction_geometry_spearman"
        ]["pooled_within_state"]
        lines.append(
            f"| {task} | {base:.6f} | {control:.6f} | {internal:.6f} | {100 * gain:.3f}% | {_fmt(rho)} |"
        )
    ci = result["paired_state_candidate_bootstrap_ci95_vs_b1"]
    gain_count = sum(value > 0 for value in relative)
    control_count = sum(
        value > 0
        for value in result["per_task_relative_rmse_improvement_vs_matched_control"]
    )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            (
                "equal-task macro relative RMSE gain vs exact B1: "
                f"{100 * result['equal_task_macro_relative_rmse_improvement_vs_b1']:.3f}% "
                f"(fixed-crossfit conditional two-axis 95% interval [{100 * ci[0]:.3f}%, "
                f"{100 * ci[1]:.3f}%]); task direction {gain_count}/5; "
                f"better than matched post-down control in {control_count}/5 tasks."
            ),
            "",
            "这是探索性 observational screen，不是 pruning gate。future path energy 没有进入主判定，因为它几乎由已观察到的 d0 velocity 决定。这里只测 HB5/d0，不能据此声称模型在去噪中‘越来越确定’。",
            "",
            "重构使用 v3 保存的 fp16 HB input 和 bf16 checkpoint，在 float32 中计算。报告中的 post-down 重构误差验证专家 ID、权重文件和算子路径匹配；它不可能与原始 bf16 runtime input bit-exact。",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=20260823)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    data = runtime.load_grid_root(args.grid_root)
    cached = None if args.rebuild_features else load_feature_cache(args.out_dir, data)
    if cached is None:
        checkpoints = discover_checkpoints(args.grid_root)
        internal, control, validation, slot = reconstruct_features(
            data, checkpoints, args.threads
        )
        save_feature_cache(args.out_dir, data, internal, control, slot, validation)
    else:
        internal, control, validation, slot = cached
    if args.features_only:
        print(args.out_dir / "features.npz")
        return 0
    summary = analyze(
        data,
        internal,
        control,
        slot,
        validation,
        args.outer_splits,
        args.split_seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_report(summary, args.out_dir)
    print(args.out_dir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
