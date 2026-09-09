#!/usr/bin/env python3
"""Fixed HB5/d0 offline vector screen against final-action geometry.

This is a retrospective screening experiment before a runtime remaining-
correction study.  It reconstructs HB5 expert and shared outputs from stored
fp16 hidden states and checkpoint weights, then uses only state-and-seed-
disjoint predictions to evaluate normalized final ``actions[0]`` geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from analyze_early_action_head import (
    center_targets,
    distance_correlation,
    exact_stratified_random_coverage,
    pairwise_rms,
    pam,
    tune_and_fit_head,
)
from analyze_expert_activation_proxy import (
    load_primary_cell,
    reconstruct_primary_cell,
)
from analyze_expert_activation_future import _layer_weights, _mlp


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = HERE / "analysis" / "expert-activation-hidden-matched"
DEFAULT_OUT = HERE / "analysis" / "offline-vector-screen"
FEATURE_CACHE_SCHEMA = "himoe-offline-vector-screen-features-v4"
SUMMARY_SCHEMA = "himoe-offline-vector-screen-v2"
FIXED_LAYER = 5
FIXED_DENOISE = 0
N_TASKS = 5
N_STATES = 16
N_CANDIDATES = 32
N_ACTION_TOKENS = 10
N_ACTION_DIMS = 7
N_SEED_FOLDS = 4
K_KEEP = 8
METHODS = (
    "noise24",
    "hidden",
    "shared",
    "routed",
    "router",
    "expert_scalars",
    "raw_equal",
    "expert_channel_std",
    "cancellation_gap",
    "base",
    "base_plus_routed",
)
BASE_BLOCKS = ("noise24", "hidden", "shared", "router")
STANDALONE_BASELINES = ("noise24", "hidden", "shared", "router")
DECOMPOSITION_CONTROLS = ("raw_equal", "expert_channel_std", "cancellation_gap")
RESIDUAL_RIDGE = 1.0
MIN_TASK_SPEARMAN_GAIN = 0.02
MIN_TASK_COVERAGE_GAIN = 0.01
MIN_POOL_WIN_RATE = 0.60
MIN_MACRO_RANDOM_COVERAGE_GAIN = 0.01
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260823


@dataclass
class TaskFeatures:
    name: str
    run: str
    checkpoint: str
    checkpoint_sha256: str | None
    checkpoint_content_sha256: str | None
    scene_ids: np.ndarray
    seed_ids: np.ndarray
    actions: np.ndarray
    blocks: dict[str, np.ndarray]
    reconstruction_validation: dict[str, float]
    source_archive_validation: dict[str, float]


def finite(value: float | np.floating) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite result: {result}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_manifest(root: Path, paths: list[Path]) -> dict[str, Any]:
    """Fingerprint immutable source trees without rereading multi-GB chunk contents."""
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    files = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise FileNotFoundError(path)
    for path in sorted(set(files)):
        stat = path.stat()
        relative = str(path.relative_to(root))
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(b"\n")
        file_count += 1
        total_bytes += stat.st_size
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "fields": "relative_path,size,mtime_ns",
    }


def source_signature(summary_path: Path) -> dict[str, Any]:
    source = json.loads(summary_path.read_text())
    checkpoint = Path(source["checkpoint"]).resolve()
    checkpoint_stat = checkpoint.stat()
    run = Path(source["run"]).resolve()
    run_manifest = stat_manifest(
        run,
        [
            run / "meta.json",
            run / "server" / "hidden.zarr",
            run / "server" / "routes.zarr",
            run / "client",
        ],
    )
    return {
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
        "proxy_archive_sha256": sha256(
            summary_path.parent / "candidate_proxy_values.npz"
        ),
        "checkpoint": str(checkpoint),
        "declared_checkpoint_sha256": source.get("checkpoint_sha256"),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "run": str(run),
        "run_input_stat_manifest": run_manifest,
    }


def seed_folds(candidates: int = N_CANDIDATES) -> list[np.ndarray]:
    if candidates % N_SEED_FOLDS:
        raise ValueError("candidate count must divide into four equal seed folds")
    return [
        np.arange(candidates, dtype=np.int64)[fold::N_SEED_FOLDS]
        for fold in range(N_SEED_FOLDS)
    ]


def equal_block_kernel_features(blocks: list[np.ndarray]) -> np.ndarray:
    """Concatenate standardized blocks so their linear kernels are equiponderant."""
    if not blocks:
        raise ValueError("at least one feature block is required")
    arrays = [np.asarray(block, dtype=np.float64) for block in blocks]
    prefix = arrays[0].shape[:-1]
    if any(array.ndim < 2 or array.shape[:-1] != prefix for array in arrays):
        raise ValueError("feature blocks must share all non-width axes")
    if any(array.shape[-1] <= 0 for array in arrays):
        raise ValueError("feature blocks must have positive width")
    total_width = sum(array.shape[-1] for array in arrays)
    n_blocks = len(arrays)
    scaled = [
        array * math.sqrt(total_width / (n_blocks * array.shape[-1]))
        for array in arrays
    ]
    return np.concatenate(scaled, axis=-1)


def _source_summaries(source_root: Path) -> list[Path]:
    paths = sorted(source_root.glob("*/summary.json"))
    if len(paths) != N_TASKS:
        raise RuntimeError(f"expected five source summaries below {source_root}, found {len(paths)}")
    if len({path.parent.name for path in paths}) != N_TASKS:
        raise RuntimeError("source task names are not unique")
    return paths


def _ordered_token_mean(value: np.ndarray, order: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    ordered = array[order]
    if ordered.shape[:3] != (N_STATES, N_CANDIDATES, N_ACTION_TOKENS):
        raise RuntimeError(f"unexpected ordered token tensor: {ordered.shape}")
    return ordered.mean(axis=2, dtype=np.float32)


def _validate_source_archive(
    archive_path: Path,
    scene_ids: np.ndarray,
    seed_ids: np.ndarray,
    scalar_blocks: dict[str, np.ndarray],
) -> dict[str, float]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"missing source proxy archive: {archive_path}")
    with np.load(archive_path) as archive:
        if not np.array_equal(archive["scene_ids"], scene_ids):
            raise RuntimeError(f"scene ids disagree with {archive_path}")
        if not np.array_equal(archive["seed_ids"], seed_ids):
            raise RuntimeError(f"seed ids disagree with {archive_path}")
        errors = {}
        for name in ("disagreement", "cancellation", "conflict"):
            reference = np.asarray(archive[name], dtype=np.float32)
            candidate = scalar_blocks[name]
            if reference.shape != candidate.shape:
                raise RuntimeError(f"{name} shape disagrees with {archive_path}")
            errors[f"{name}_max_abs_error"] = finite(
                np.max(np.abs(reference.astype(np.float64) - candidate))
            )
    if max(errors.values(), default=0.0) > 1e-6:
        raise RuntimeError(f"reconstructed proxy scalars disagree with {archive_path}: {errors}")
    return errors


def reconstruct_raw_decomposition(
    data: dict[str, Any],
    checkpoint: Path,
    threads: int,
    routed_reference: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Re-evaluate selected experts and retain fixed raw-vector decompositions."""
    import torch

    torch.set_num_threads(threads)
    checkpoint_state = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    experts, _shared, _gate = _layer_weights(
        checkpoint_state, FIXED_LAYER, "cpu", torch
    )
    experts = [tuple(weight.float() for weight in triplet) for triplet in experts]
    hidden = torch.from_numpy(np.ascontiguousarray(data["hidden"])).reshape(-1, 1024)
    ids = torch.from_numpy(np.ascontiguousarray(data["route_ids"])).reshape(-1, 4)
    alpha = torch.from_numpy(np.ascontiguousarray(data["route_weight"])).reshape(-1, 4)
    raw = torch.full(
        (len(hidden), 4, hidden.shape[-1]),
        float("nan"),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        for expert_id, weights in enumerate(experts):
            matches = torch.nonzero(ids == expert_id, as_tuple=False)
            if matches.numel() == 0:
                continue
            token, slot = matches[:, 0], matches[:, 1]
            raw[token, slot] = _mlp(torch, hidden[token], weights)
        if not bool(torch.isfinite(raw).all()):
            raise RuntimeError("raw expert decomposition has missing or non-finite slots")
        routed = (raw * alpha[:, :, None]).sum(dim=1)
        raw_equal = raw.mean(dim=1)
        channel_std = (
            (raw - routed[:, None]).square() * alpha[:, :, None]
        ).sum(dim=1).clamp_min(0.0).sqrt()
        cancellation_gap = (
            (raw.abs() * alpha[:, :, None]).sum(dim=1) - routed.abs()
        ).clamp_min(0.0)

    candidates = len(data["episodes"])

    def shaped(value) -> np.ndarray:
        return value.numpy().reshape(candidates, N_ACTION_TOKENS, 1024)

    return (
        {
            "raw_equal": shaped(raw_equal),
            "expert_channel_std": shaped(channel_std),
            "cancellation_gap": shaped(cancellation_gap),
        },
        {
            "routed_recompute_max_abs_error": finite(
                np.max(
                    np.abs(
                        shaped(routed).astype(np.float64)
                        - np.asarray(routed_reference, dtype=np.float64)
                    )
                )
            )
        },
    )


def reconstruct_task(source_summary: Path, threads: int) -> TaskFeatures:
    source = json.loads(source_summary.read_text())
    if int(source.get("layer", -1)) != FIXED_LAYER or int(source.get("denoise", -1)) != FIXED_DENOISE:
        raise RuntimeError(f"source is not the fixed HB5/d0 cell: {source_summary}")
    run = Path(source["run"]).resolve()
    checkpoint = Path(source["checkpoint"]).resolve()
    data = load_primary_cell(run, FIXED_LAYER, FIXED_DENOISE)
    reconstructed = reconstruct_primary_cell(data, checkpoint, FIXED_LAYER, threads)
    decomposition, decomposition_validation = reconstruct_raw_decomposition(
        data, checkpoint, threads, reconstructed["routed"]
    )
    order = np.asarray(data["order"], dtype=np.int64)
    if order.shape != (N_STATES, N_CANDIDATES):
        raise RuntimeError(f"source is not 16x32: {source_summary}")

    actions = np.asarray(data["actions"][order], dtype=np.float32)
    if actions.shape != (N_STATES, N_CANDIDATES, N_ACTION_TOKENS, N_ACTION_DIMS):
        raise RuntimeError(f"unexpected normalized final action tensor: {actions.shape}")
    scalar_parts = {
        name: _ordered_token_mean(reconstructed[name], order)
        for name in ("expert_mass", "disagreement", "cancellation", "conflict")
    }
    scalars = np.stack(
        [
            scalar_parts["expert_mass"],
            scalar_parts["disagreement"],
            scalar_parts["cancellation"],
            scalar_parts["conflict"],
        ],
        axis=-1,
    )
    blocks = {
        "noise24": np.asarray(data["noise"][order], dtype=np.float32).reshape(
            N_STATES, N_CANDIDATES, -1
        ),
        "hidden": _ordered_token_mean(data["hidden"], order),
        "shared": _ordered_token_mean(reconstructed["shared"], order),
        "routed": _ordered_token_mean(reconstructed["routed"], order),
        "router": _ordered_token_mean(data["route_probs"], order),
        "expert_scalars": scalars,
        "raw_equal": _ordered_token_mean(decomposition["raw_equal"], order),
        "expert_channel_std": _ordered_token_mean(
            decomposition["expert_channel_std"], order
        ),
        "cancellation_gap": _ordered_token_mean(
            decomposition["cancellation_gap"], order
        ),
    }
    expected_widths = {
        "noise24": 240,
        "hidden": 1024,
        "shared": 1024,
        "routed": 1024,
        "router": 32,
        "expert_scalars": 4,
        "raw_equal": 1024,
        "expert_channel_std": 1024,
        "cancellation_gap": 1024,
    }
    for name, width in expected_widths.items():
        if blocks[name].shape != (N_STATES, N_CANDIDATES, width):
            raise RuntimeError(f"unexpected {name} feature shape: {blocks[name].shape}")
        if not np.all(np.isfinite(blocks[name])):
            raise RuntimeError(f"non-finite {name} feature in {source_summary}")

    archive_validation = _validate_source_archive(
        source_summary.parent / "candidate_proxy_values.npz",
        np.asarray(data["scene_ids"], dtype=np.int64),
        np.asarray(data["seed_ids"], dtype=np.int64),
        scalar_parts,
    )
    source_validation = source.get("offline_reconstruction_validation", {})
    current_validation = {
        key: finite(value) for key, value in reconstructed["validation"].items()
    }
    current_validation.update(decomposition_validation)
    for key, value in source_validation.items():
        if key not in current_validation or not np.isclose(
            current_validation[key], float(value), atol=1e-9, rtol=1e-7
        ):
            raise RuntimeError(f"reconstruction validation changed for {source_summary}: {key}")
    if source.get("checkpoint_sha256") != data.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint identity changed for {source_summary}")
    return TaskFeatures(
        name=source_summary.parent.name,
        run=str(run),
        checkpoint=str(checkpoint),
        checkpoint_sha256=data.get("checkpoint_sha256"),
        checkpoint_content_sha256=None,
        scene_ids=np.asarray(data["scene_ids"], dtype=np.int64),
        seed_ids=np.asarray(data["seed_ids"], dtype=np.int64),
        actions=actions,
        blocks=blocks,
        reconstruction_validation=current_validation,
        source_archive_validation=archive_validation,
    )


def _cache_metadata(tasks: list[TaskFeatures], sources: list[Path]) -> dict[str, Any]:
    return {
        "schema": FEATURE_CACHE_SCHEMA,
        "fixed_layer": FIXED_LAYER,
        "fixed_denoise": FIXED_DENOISE,
        "source_signatures": [source_signature(path) for path in sources],
        "tasks": [
            {
                "name": task.name,
                "run": task.run,
                "checkpoint": task.checkpoint,
                "checkpoint_sha256": task.checkpoint_sha256,
                "checkpoint_content_sha256": task.checkpoint_content_sha256,
                "reconstruction_validation": task.reconstruction_validation,
                "source_archive_validation": task.source_archive_validation,
            }
            for task in tasks
        ],
    }


def save_feature_cache(path: Path, tasks: list[TaskFeatures], sources: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_cache_metadata(tasks, sources))),
        "task_names": np.asarray([task.name for task in tasks]),
        "scene_ids": np.stack([task.scene_ids for task in tasks]),
        "seed_ids": np.stack([task.seed_ids for task in tasks]),
        "actions": np.stack([task.actions for task in tasks]),
    }
    for name in (
        "noise24",
        "hidden",
        "shared",
        "routed",
        "router",
        "expert_scalars",
        *DECOMPOSITION_CONTROLS,
    ):
        arrays[name] = np.stack([task.blocks[name] for task in tasks])
    np.savez_compressed(path, **arrays)


def load_feature_cache(path: Path, sources: list[Path]) -> list[TaskFeatures]:
    with np.load(path) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        expected_sources = [source_signature(source) for source in sources]
        if metadata.get("schema") != FEATURE_CACHE_SCHEMA:
            raise RuntimeError(f"unsupported feature cache schema in {path}")
        if metadata.get("fixed_layer") != FIXED_LAYER or metadata.get("fixed_denoise") != FIXED_DENOISE:
            raise RuntimeError(f"feature cache is not fixed HB5/d0: {path}")
        if metadata.get("source_signatures") != expected_sources:
            raise RuntimeError(f"feature cache sources changed: {path}")
        names = [str(value) for value in cache["task_names"]]
        if len(names) != N_TASKS or names != [row["name"] for row in metadata["tasks"]]:
            raise RuntimeError(f"feature cache task identity is invalid: {path}")
        tasks = []
        for axis, row in enumerate(metadata["tasks"]):
            blocks = {
                name: np.asarray(cache[name][axis], dtype=np.float32)
                for name in (
                    "noise24",
                    "hidden",
                    "shared",
                    "routed",
                    "router",
                    "expert_scalars",
                    *DECOMPOSITION_CONTROLS,
                )
            }
            tasks.append(
                TaskFeatures(
                    name=names[axis],
                    run=row["run"],
                    checkpoint=row["checkpoint"],
                    checkpoint_sha256=row.get("checkpoint_sha256"),
                    checkpoint_content_sha256=row.get("checkpoint_content_sha256"),
                    scene_ids=np.asarray(cache["scene_ids"][axis], dtype=np.int64),
                    seed_ids=np.asarray(cache["seed_ids"][axis], dtype=np.int64),
                    actions=np.asarray(cache["actions"][axis], dtype=np.float32),
                    blocks=blocks,
                    reconstruction_validation=row["reconstruction_validation"],
                    source_archive_validation=row["source_archive_validation"],
                )
            )
    return tasks


def verify_checkpoint_content(tasks: list[TaskFeatures]) -> None:
    verified = {}
    for task in tasks:
        expected = task.checkpoint_sha256
        if expected is None:
            raise RuntimeError(f"source has no declared checkpoint SHA-256: {task.name}")
        if task.checkpoint not in verified:
            print(f"verifying checkpoint SHA-256 {task.checkpoint}", flush=True)
            verified[task.checkpoint] = sha256(Path(task.checkpoint))
        actual = verified[task.checkpoint]
        if actual != expected:
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch for {task.name}: {actual} != {expected}"
            )
        task.checkpoint_content_sha256 = actual


def load_or_reconstruct_features(
    source_root: Path,
    cache_path: Path,
    threads: int,
    force: bool,
) -> list[TaskFeatures]:
    sources = _source_summaries(source_root)
    if cache_path.is_file() and not force:
        print(f"loading feature cache {cache_path}", flush=True)
        return load_feature_cache(cache_path, sources)
    tasks = []
    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{N_TASKS}] reconstructing {source.parent.name}", flush=True)
        tasks.append(reconstruct_task(source, threads))
    canonical_seeds = tuple(map(int, tasks[0].seed_ids))
    if any(tuple(map(int, task.seed_ids)) != canonical_seeds for task in tasks[1:]):
        raise RuntimeError("the five tasks do not share the same sorted seed grid")
    verify_checkpoint_content(tasks)
    save_feature_cache(cache_path, tasks, sources)
    print(f"wrote {cache_path}", flush=True)
    return tasks


def fit_train_fold_transform(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit one block transform on outer-train samples and apply it to held8."""
    train_array = np.asarray(train, dtype=np.float64)
    test_array = np.asarray(test, dtype=np.float64)
    if train_array.ndim != 3 or test_array.ndim != 2:
        raise ValueError("train/test features must be [state,seed,width] and [seed,width]")
    if train_array.shape[-1] != test_array.shape[-1]:
        raise ValueError("train/test feature widths differ")
    flat = train_array.reshape(-1, train_array.shape[-1])
    center = flat.mean(axis=0)
    centered_train = train_array - center
    centered_test = test_array - center
    scale = finite(np.sqrt(np.mean(np.square(centered_train))))
    if scale <= 1e-12:
        raise RuntimeError("outer-train feature block has zero RMS")
    transformed_train = centered_train / scale
    transformed_test = centered_test / scale
    return (
        transformed_train,
        transformed_test,
        {
            "train_scalar_rms_before_scaling": scale,
            "train_kernel_trace_after_scaling": finite(
                np.mean(np.square(transformed_train))
            ),
        },
    )


def residualized_nested_features(
    train_base: np.ndarray,
    test_base: np.ndarray,
    train_routed: np.ndarray,
    test_routed: np.ndarray,
    ridge: float = RESIDUAL_RIDGE,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Append train-only base-residual routed features without downweighting base."""
    if ridge <= 0.0:
        raise ValueError("residual ridge must be positive")
    base = np.asarray(train_base, dtype=np.float64)
    held_base = np.asarray(test_base, dtype=np.float64)
    routed = np.asarray(train_routed, dtype=np.float64)
    held_routed = np.asarray(test_routed, dtype=np.float64)
    groups, candidates, base_width = base.shape
    routed_width = routed.shape[-1]
    flat_base = base.reshape(-1, base_width)
    flat_routed = routed.reshape(-1, routed_width)
    base_kernel = (flat_base @ flat_base.T) / base_width
    held_kernel = (held_base @ flat_base.T) / base_width
    dual = np.linalg.solve(
        base_kernel + ridge * np.eye(len(base_kernel)), flat_routed
    )
    residual_train = flat_routed - base_kernel @ dual
    residual_test = held_routed - held_kernel @ dual

    total_width = base_width + routed_width
    base_scale = math.sqrt(total_width / base_width)
    residual_scale = math.sqrt(total_width / routed_width)
    nested_train = np.concatenate(
        [flat_base * base_scale, residual_train * residual_scale], axis=-1
    ).reshape(groups, candidates, total_width)
    nested_test = np.concatenate(
        [held_base * base_scale, residual_test * residual_scale], axis=-1
    )
    base_trace = finite(np.mean(np.einsum("nd,nd->n", flat_base, flat_base)) / base_width)
    routed_trace = finite(
        np.mean(np.einsum("nd,nd->n", flat_routed, flat_routed)) / routed_width
    )
    residual_trace = finite(
        np.mean(np.einsum("nd,nd->n", residual_train, residual_train))
        / routed_width
    )
    nested_trace = finite(
        np.mean(
            np.einsum(
                "nd,nd->n",
                nested_train.reshape(-1, total_width),
                nested_train.reshape(-1, total_width),
            )
        )
        / total_width
    )
    if not np.isclose(nested_trace, base_trace + residual_trace, atol=1e-10, rtol=1e-10):
        raise RuntimeError("nested feature map does not preserve base plus residual kernel")
    return (
        nested_train,
        nested_test,
        {
            "base_kernel_trace": base_trace,
            "routed_kernel_trace": routed_trace,
            "residual_routed_kernel_trace": residual_trace,
            "nested_kernel_trace": nested_trace,
        },
    )


def fold_method_features(
    blocks: dict[str, np.ndarray],
    train_states: np.ndarray,
    allowed_seeds: np.ndarray,
    held_state: int,
    held_seeds: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    transformed = {}
    transform_diagnostics = {}
    for name, value in blocks.items():
        train, test, diagnostics = fit_train_fold_transform(
            value[train_states][:, allowed_seeds], value[held_state, held_seeds]
        )
        transformed[name] = (train, test)
        transform_diagnostics[name] = diagnostics

    result = {name: transformed[name] for name in transformed}
    base_train = equal_block_kernel_features(
        [transformed[name][0] for name in BASE_BLOCKS]
    )
    base_test = equal_block_kernel_features(
        [transformed[name][1] for name in BASE_BLOCKS]
    )
    result["base"] = (base_train, base_test)
    nested_train, nested_test, nested_diagnostics = residualized_nested_features(
        base_train,
        base_test,
        transformed["routed"][0],
        transformed["routed"][1],
    )
    result["base_plus_routed"] = (nested_train, nested_test)
    if tuple(result) != METHODS:
        raise RuntimeError(f"method ordering changed: {tuple(result)}")
    return result, {
        "block_transforms": transform_diagnostics,
        "nested_residual": nested_diagnostics,
    }


def strict_crossfit_predictions(
    blocks: dict[str, np.ndarray],
    raw_targets: np.ndarray,
    folds: list[np.ndarray],
    inner_folds: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]], list[dict[str, Any]]]:
    """Cross-fit all methods with outer-train-only feature preprocessing."""
    targets = np.asarray(raw_targets, dtype=np.float64)
    if targets.shape[:2] != (N_STATES, N_CANDIDATES) or targets.ndim != 3:
        raise ValueError("raw targets must be [16,32,width]")
    predictions = {
        name: np.zeros_like(targets, dtype=np.float64) for name in METHODS
    }
    selected_alphas = {name: [] for name in METHODS}
    diagnostics = []
    all_seeds = np.arange(N_CANDIDATES, dtype=np.int64)
    for outer in range(N_STATES):
        train_states = np.flatnonzero(np.arange(N_STATES) != outer)
        for fold_axis, held in enumerate(folds):
            allowed = np.setdiff1d(all_seeds, held, assume_unique=True)
            feature_pairs, fold_diagnostics = fold_method_features(
                blocks, train_states, allowed, outer, held
            )
            train_targets = center_targets(targets[train_states][:, allowed])
            for name in METHODS:
                train_features, test_features = feature_pairs[name]
                prediction, alpha = tune_and_fit_head(
                    train_features, train_targets, test_features, inner_folds
                )
                predictions[name][outer, held] = prediction
                selected_alphas[name].append(alpha)
            diagnostics.append(
                {
                    "held_state_axis": outer,
                    "held_seed_fold": fold_axis,
                    **fold_diagnostics,
                }
            )
    if any(not np.all(np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("strict cross-fit produced non-finite predictions")
    return predictions, selected_alphas, diagnostics


def evaluate_pool(
    predicted: np.ndarray,
    actions: np.ndarray,
    folds: list[np.ndarray],
) -> dict[str, Any]:
    predicted_distance = pairwise_rms(predicted)
    action_distance = pairwise_rms(actions)
    keep_per_fold = K_KEEP // len(folds)
    correlations = []
    selected_parts = []
    for fold in folds:
        local_prediction = predicted_distance[np.ix_(fold, fold)]
        local_action = action_distance[np.ix_(fold, fold)]
        correlations.append(distance_correlation(local_prediction, local_action)["spearman"])
        selected_parts.append(fold[pam(local_prediction, keep_per_fold)])
    selected = np.sort(np.concatenate(selected_parts))
    if selected.shape != (K_KEEP,) or len(np.unique(selected)) != K_KEEP:
        raise RuntimeError("stratified selection did not produce unique K8")
    coverage = finite(np.min(action_distance[:, selected], axis=1).mean())
    random_coverage = exact_stratified_random_coverage(
        action_distance, folds, keep_per_fold
    )
    ratio = finite(coverage / random_coverage)
    return {
        "within_held8_spearman": correlations,
        "pool_spearman_mean": finite(np.mean(correlations)),
        "coverage": coverage,
        "exact_stratified_random_coverage": random_coverage,
        "coverage_over_exact_stratified_random": ratio,
        "relative_coverage_improvement": finite(1.0 - ratio),
        "selected_seed_positions": selected.tolist(),
    }


def evaluate_task(task: TaskFeatures, inner_folds: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if inner_folds < 2 or inner_folds > N_STATES - 1:
        raise ValueError("inner_folds must be between 2 and 15")
    folds = seed_folds()
    print(f"  strict cross-fit {task.name}: {len(METHODS)} methods", flush=True)
    targets = task.actions.reshape(N_STATES, N_CANDIDATES, -1)
    predictions, selected_alphas, fold_diagnostics = strict_crossfit_predictions(
        task.blocks, targets, folds, inner_folds
    )
    alpha_counts = {
        name: dict(sorted(Counter(map(str, values)).items()))
        for name, values in selected_alphas.items()
    }

    rows = []
    for state_axis, state_id in enumerate(task.scene_ids):
        methods = {
            name: evaluate_pool(
                predictions[name][state_axis], task.actions[state_axis], folds
            )
            for name in METHODS
        }
        baseline_values = {
            methods[name]["exact_stratified_random_coverage"] for name in METHODS
        }
        if len(baseline_values) != 1:
            raise RuntimeError("method-independent random coverage changed across methods")
        rows.append(
            {
                "task": task.name,
                "state_id": int(state_id),
                "methods": methods,
            }
        )
    diagnostics = {
        "task": task.name,
        "run": task.run,
        "checkpoint": task.checkpoint,
        "checkpoint_sha256": task.checkpoint_sha256,
        "checkpoint_content_sha256": task.checkpoint_content_sha256,
        "state_ids": task.scene_ids.tolist(),
        "seed_ids": task.seed_ids.tolist(),
        "seed_folds": [[int(task.seed_ids[index]) for index in fold] for fold in folds],
        "reconstruction_validation": task.reconstruction_validation,
        "source_archive_validation": task.source_archive_validation,
        "alpha_counts": alpha_counts,
        "cross_fit_fold_diagnostics": fold_diagnostics,
    }
    return rows, diagnostics


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values = [row["methods"][method] for row in rows]
    block_rhos = [rho for value in values for rho in value["within_held8_spearman"]]
    ratios = [value["coverage_over_exact_stratified_random"] for value in values]
    return {
        "pool_count": len(values),
        "within_held8_spearman_mean": finite(np.mean(block_rhos)),
        "within_held8_spearman_median": finite(np.median(block_rhos)),
        "pool_spearman_mean": finite(np.mean([value["pool_spearman_mean"] for value in values])),
        "coverage_mean": finite(np.mean([value["coverage"] for value in values])),
        "exact_stratified_random_coverage_mean": finite(
            np.mean([value["exact_stratified_random_coverage"] for value in values])
        ),
        "coverage_over_exact_stratified_random_mean": finite(np.mean(ratios)),
        "relative_coverage_improvement_mean": finite(np.mean([1.0 - ratio for ratio in ratios])),
        "pools_with_coverage_improvement": int(np.sum(np.asarray(ratios) < 1.0)),
    }


def _bootstrap_means(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("bootstrap values must be one nonempty vector")
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    draws = array[indices].mean(axis=1)
    return draws, [
        finite(np.percentile(draws, 2.5)),
        finite(np.percentile(draws, 97.5)),
    ]


def _paired_comparison(
    rows: list[dict[str, Any]],
    task_names: list[str],
    left_method: str,
    right_by_task: dict[str, str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    per_task = {}
    task_spearman_draws = []
    task_coverage_draws = []
    all_spearman = []
    all_coverage = []
    for task in task_names:
        task_rows = [row for row in rows if row["task"] == task]
        right_method = right_by_task[task]
        spearman = np.asarray(
            [
                row["methods"][left_method]["pool_spearman_mean"]
                - row["methods"][right_method]["pool_spearman_mean"]
                for row in task_rows
            ],
            dtype=np.float64,
        )
        coverage = np.asarray(
            [
                1.0
                - row["methods"][left_method]["coverage"]
                / row["methods"][right_method]["coverage"]
                for row in task_rows
            ],
            dtype=np.float64,
        )
        spearman_draws, spearman_ci = _bootstrap_means(spearman, rng)
        coverage_draws, coverage_ci = _bootstrap_means(coverage, rng)
        task_spearman_draws.append(spearman_draws)
        task_coverage_draws.append(coverage_draws)
        all_spearman.append(spearman)
        all_coverage.append(coverage)
        per_task[task] = {
            "right_method": right_method,
            "spearman_gain_mean": finite(spearman.mean()),
            "spearman_gain_paired_state_bootstrap_ci95": spearman_ci,
            "spearman_pool_win_rate": finite(np.mean(spearman > 0.0)),
            "relative_coverage_improvement_mean": finite(coverage.mean()),
            "coverage_improvement_paired_state_bootstrap_ci95": coverage_ci,
            "coverage_pool_win_rate": finite(np.mean(coverage > 0.0)),
        }
    macro_spearman_draws = np.stack(task_spearman_draws).mean(axis=0)
    macro_coverage_draws = np.stack(task_coverage_draws).mean(axis=0)
    all_spearman_array = np.concatenate(all_spearman)
    all_coverage_array = np.concatenate(all_coverage)
    return {
        "left_method": left_method,
        "per_task": per_task,
        "macro": {
            "spearman_gain_mean": finite(
                np.mean([value["spearman_gain_mean"] for value in per_task.values()])
            ),
            "spearman_gain_stratified_state_bootstrap_ci95": [
                finite(np.percentile(macro_spearman_draws, 2.5)),
                finite(np.percentile(macro_spearman_draws, 97.5)),
            ],
            "spearman_pool_win_rate": finite(np.mean(all_spearman_array > 0.0)),
            "relative_coverage_improvement_mean": finite(
                np.mean(
                    [
                        value["relative_coverage_improvement_mean"]
                        for value in per_task.values()
                    ]
                )
            ),
            "coverage_improvement_stratified_state_bootstrap_ci95": [
                finite(np.percentile(macro_coverage_draws, 2.5)),
                finite(np.percentile(macro_coverage_draws, 97.5)),
            ],
            "coverage_pool_win_rate": finite(np.mean(all_coverage_array > 0.0)),
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired state pool within each fixed task",
            "interpretation": "descriptive interval, not a significance test",
        },
    }


def summarize(rows: list[dict[str, Any]], task_names: list[str]) -> dict[str, Any]:
    per_task = {
        task: {
            method: _method_summary([row for row in rows if row["task"] == task], method)
            for method in METHODS
        }
        for task in task_names
    }
    macro = {method: _method_summary(rows, method) for method in METHODS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    nested = _paired_comparison(
        rows,
        task_names,
        "base_plus_routed",
        {task: "base" for task in task_names},
        rng,
    )
    best_standalone = {
        task: max(
            STANDALONE_BASELINES,
            key=lambda method: per_task[task][method]["within_held8_spearman_mean"],
        )
        for task in task_names
    }
    standalone = _paired_comparison(
        rows, task_names, "routed", best_standalone, rng
    )
    direct_controls = {
        control: {
            task: finite(
                per_task[task]["routed"]["within_held8_spearman_mean"]
                - per_task[task][control]["within_held8_spearman_mean"]
            )
            for task in task_names
        }
        for control in ("hidden", "shared")
    }
    task_nested_spearman = [
        value["spearman_gain_mean"] for value in nested["per_task"].values()
    ]
    task_nested_coverage = [
        value["relative_coverage_improvement_mean"]
        for value in nested["per_task"].values()
    ]
    random_coverage_gain = finite(
        macro["base_plus_routed"]["relative_coverage_improvement_mean"]
    )
    decision = {
        "primary": "base_plus_routed_residualized_minus_base",
        "all_tasks_meet_min_spearman_gain": all(
            value >= MIN_TASK_SPEARMAN_GAIN for value in task_nested_spearman
        ),
        "all_tasks_meet_min_coverage_gain": all(
            value >= MIN_TASK_COVERAGE_GAIN for value in task_nested_coverage
        ),
        "spearman_pool_win_rate_meets_minimum": (
            nested["macro"]["spearman_pool_win_rate"] >= MIN_POOL_WIN_RATE
        ),
        "coverage_pool_win_rate_meets_minimum": (
            nested["macro"]["coverage_pool_win_rate"] >= MIN_POOL_WIN_RATE
        ),
        "macro_random_coverage_gain": random_coverage_gain,
        "macro_random_coverage_gain_meets_minimum": (
            random_coverage_gain >= MIN_MACRO_RANDOM_COVERAGE_GAIN
        ),
        "thresholds": {
            "min_task_spearman_gain": MIN_TASK_SPEARMAN_GAIN,
            "min_task_relative_coverage_improvement": MIN_TASK_COVERAGE_GAIN,
            "min_paired_pool_win_rate": MIN_POOL_WIN_RATE,
            "min_macro_relative_coverage_improvement_over_exact_random": (
                MIN_MACRO_RANDOM_COVERAGE_GAIN
            ),
        },
        "rule": (
            "positive only if the train-fold residualized routed addition clears the fixed "
            "Spearman and coverage engineering thresholds in every task, clears both paired "
            "pool win-rate thresholds, and improves macro coverage over exact random"
        ),
        "threshold_interpretation": (
            "conservative engineering effect-size gates fixed before the corrected run; "
            "not null-hypothesis significance tests"
        ),
    }
    decision["screen_positive"] = all(
        (
            decision["all_tasks_meet_min_spearman_gain"],
            decision["all_tasks_meet_min_coverage_gain"],
            decision["spearman_pool_win_rate_meets_minimum"],
            decision["coverage_pool_win_rate_meets_minimum"],
            decision["macro_random_coverage_gain_meets_minimum"],
        )
    )
    return {
        "macro": macro,
        "per_task": per_task,
        "paired_comparisons": {
            "nested_residualized_routed_minus_base": nested,
            "standalone_routed_minus_best_baseline_diagnostic": standalone,
            "standalone_routed_direct_control_deltas": direct_controls,
        },
        "decision": decision,
    }


def build(tasks: list[TaskFeatures], inner_folds: int) -> dict[str, Any]:
    if len(tasks) != N_TASKS:
        raise RuntimeError(f"expected five tasks, got {len(tasks)}")
    names = [task.name for task in tasks]
    if len(set(names)) != N_TASKS:
        raise RuntimeError("task names are not unique")
    canonical_seeds = tuple(map(int, tasks[0].seed_ids))
    if any(tuple(map(int, task.seed_ids)) != canonical_seeds for task in tasks[1:]):
        raise RuntimeError("task seed grids differ")
    rows = []
    diagnostics = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{N_TASKS}] evaluating {task.name}", flush=True)
        task_rows, task_diagnostics = evaluate_task(task, inner_folds)
        rows.extend(task_rows)
        diagnostics.append(task_diagnostics)
    if len(rows) != N_TASKS * N_STATES:
        raise RuntimeError(f"expected 80 pools, got {len(rows)}")
    summary = summarize(rows, names)
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "retrospective_offline_screen_not_runtime_causal",
        "protocol": {
            "source": "five right-16x32 sources referenced by expert-activation-hidden-matched summaries",
            "fixed_cell": {"hb_layer": FIXED_LAYER, "denoise": FIXED_DENOISE},
            "flow_site": "HB5/d0 features are evaluated on the initial x0 forward",
            "tasks": N_TASKS,
            "states_per_task": N_STATES,
            "candidates_per_state": N_CANDIDATES,
            "common_flow_noise_seeds": list(canonical_seeds),
            "target": "normalized final client actions[0], complete 10x7 geometry",
            "normalization": "official checkpoint action standard deviation, applied by load_primary_cell",
            "features": {
                "noise24": "exact initial 10x24 PCG64 flow noise, flattened",
                "hidden": "HB5/d0 pre-MoE hidden, mean over ten action tokens",
                "shared": "offline reconstructed HB5/d0 shared vector, mean over ten action tokens",
                "routed": "offline reconstructed HB5/d0 routed expert vector, mean over ten action tokens",
                "router": "recorded full HB5/d0 router probabilities, mean over ten action tokens",
                "expert_scalars": "token means of s1 expert mass, D disagreement, C cancellation, Q routed/shared conflict",
                "raw_equal": "descriptive equal-weight mean of four selected pre-gate expert vectors, then token mean",
                "expert_channel_std": "descriptive combine-weighted per-channel standard deviation across selected experts, then token mean",
                "cancellation_gap": "descriptive per-channel sum_k(alpha_k*abs(E_k))-abs(routed), then token mean",
                "base": "noise24 + hidden + shared + router, arithmetic mean of four train-normalized linear kernels",
                "base_plus_routed": "K_base plus the train-fold base-residualized routed kernel; K_base is not downweighted",
            },
            "feature_preprocessing": (
                "for every outer state x held8 fold, each primitive block's per-channel mean "
                "and scalar RMS are fitted on the other 15 states x allowed24 seeds only and "
                "then applied unchanged to held8; no held-state or held-seed feature enters "
                "the transform fit"
            ),
            "cross_fit": (
                "within each task, sorted seed positions modulo four form four held8 blocks; "
                "each held state and held seed block are absent from head fitting and target "
                "centering; target centers use only the other 24 seeds; each model independently "
                "selects ridge alpha from the fixed grid using inner state-group CV"
            ),
            "nested_increment": (
                "within each outer train fold, ridge=1 predicts routed features from the "
                "trace-normalized base kernel; base+routed appends the routed residual with a "
                "feature map whose kernel is exactly K_base + K_routed_residual"
            ),
            "inner_folds": inner_folds,
            "evaluation": (
                "Spearman uses pair distances only within each held8; select PAM K2 within "
                "each held8 and union K8"
            ),
            "coverage": (
                "mean normalized-final-action distance to nearest selected candidate; baseline "
                "is the exact expectation for two uniformly selected candidates per fixed held8"
            ),
            "primary_comparison": (
                "train-fold residualized base+routed versus the unchanged fixed base"
            ),
            "standalone_diagnostic": (
                "standalone routed versus hidden/shared and the best standalone nonexpert "
                "baseline is reported but cannot make the primary screen positive"
            ),
            "engineering_effect_thresholds": {
                "min_task_spearman_gain": MIN_TASK_SPEARMAN_GAIN,
                "min_task_relative_coverage_improvement": MIN_TASK_COVERAGE_GAIN,
                "min_paired_pool_win_rate": MIN_POOL_WIN_RATE,
                "min_macro_relative_coverage_improvement_over_exact_random": (
                    MIN_MACRO_RANDOM_COVERAGE_GAIN
                ),
                "interpretation": "fixed conservative engineering gates, not significance tests",
            },
            "paired_state_bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "task_stratified": True,
                "used_for_gate": False,
            },
            "descriptive_raw_decomposition_controls": (
                "raw_equal, expert_channel_std, and cancellation_gap are excluded from the "
                "primary rule, base, and selector"
            ),
            "offline_boundary": (
                "routed/shared are fp32 checkpoint reconstructions from stored fp16 hidden; "
                "recorded selected expert ids and weights are authoritative; expert identity "
                "is never aligned or shared across checkpoint suites"
            ),
        },
        "summary": summary,
        "task_diagnostics": diagnostics,
        "pools": rows,
        "limitations": [
            "The endpoint is normalized final-action geometry, not runtime remaining correction.",
            "Offline vectors start from stored fp16 hidden states and are not runtime-exact.",
            "All vector features are ten-token means; per-token raw Gram and S_LOO structure are not tested.",
            "Expert IDs are interpreted only within each checkpoint and are never pooled across suites.",
            "This retrospective screen is associative and does not establish a causal expert intervention.",
        ],
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _fmt_ci(value: list[float]) -> str:
    return f"[{value[0]:.3f}, {value[1]:.3f}]"


def _fmt_percent_ci(value: list[float]) -> str:
    return f"[{100.0 * value[0]:+.1f}%, {100.0 * value[1]:+.1f}%]"


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    macro = summary["macro"]
    per_task = summary["per_task"]
    decision = summary["decision"]
    nested = summary["paired_comparisons"]["nested_residualized_routed_minus_base"]
    standalone = summary["paired_comparisons"][
        "standalone_routed_minus_best_baseline_diagnostic"
    ]
    direct = summary["paired_comparisons"][
        "standalone_routed_direct_control_deltas"
    ]
    task_names = list(per_task)
    result = "positive" if decision["screen_positive"] else "negative"
    nested_macro = nested["macro"]
    lines = [
        "# Fixed HB5/d0 offline vector screen",
        "",
        "## Result",
        "",
        f"The pre-runtime screen is **{result}** under the frozen decision rule. "
        f"Adding the train-fold base-residualized routed vector changes macro held8 "
        f"Spearman by `{nested_macro['spearman_gain_mean']:+.3f}` "
        f"({_fmt_ci(nested_macro['spearman_gain_stratified_state_bootstrap_ci95'])}) and "
        f"relative K8 coverage by `{100.0 * nested_macro['relative_coverage_improvement_mean']:+.1f}%` "
        f"({_fmt_percent_ci(nested_macro['coverage_improvement_stratified_state_bootstrap_ci95'])}).",
        "",
        "This target is the normalized final `actions[0]` 10x7 geometry. It is an offline "
        "screening target and is not the runtime remaining-correction target.",
        "",
        "The thresholds below are conservative engineering effect-size gates fixed before "
        "the corrected run. They are not null-hypothesis significance tests; paired bootstrap "
        "intervals are descriptive.",
        "",
        "## Primary nested increment",
        "",
        "`base` is the equal-kernel combination of noise24, hidden, shared, and router. "
        "Within every outer train fold, routed is ridge-residualized against that base. The "
        "augmented kernel is exactly `K_base + K_routed_residual`; `K_base` is unchanged.",
        "",
        "| task | base rho | base+residual routed rho | rho gain (CI95) | coverage gain (CI95) | rho pool wins | coverage pool wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in task_names:
        row = per_task[task]
        comparison = nested["per_task"][task]
        lines.append(
            f"| {task} | {_fmt(row['base']['within_held8_spearman_mean'])} | "
            f"{_fmt(row['base_plus_routed']['within_held8_spearman_mean'])} | "
            f"{comparison['spearman_gain_mean']:+.3f} "
            f"{_fmt_ci(comparison['spearman_gain_paired_state_bootstrap_ci95'])} | "
            f"{100.0 * comparison['relative_coverage_improvement_mean']:+.1f}% "
            f"{_fmt_percent_ci(comparison['coverage_improvement_paired_state_bootstrap_ci95'])} | "
            f"{comparison['spearman_pool_win_rate']:.3f} | "
            f"{comparison['coverage_pool_win_rate']:.3f} |"
        )

    gate_rows = (
        (
            "every task rho gain >= +0.02",
            decision["all_tasks_meet_min_spearman_gain"],
        ),
        (
            "every task relative coverage gain >= +1%",
            decision["all_tasks_meet_min_coverage_gain"],
        ),
        (
            "paired rho pool win rate >= 60%",
            decision["spearman_pool_win_rate_meets_minimum"],
        ),
        (
            "paired coverage pool win rate >= 60%",
            decision["coverage_pool_win_rate_meets_minimum"],
        ),
        (
            "macro coverage over exact random improves >= 1%",
            decision["macro_random_coverage_gain_meets_minimum"],
        ),
    )
    lines += [
        "",
        "| engineering gate | pass |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {label} | {str(value).lower()} |" for label, value in gate_rows
    )

    lines += [
        "",
        "## Macro over 80 state pools",
        "",
        "Spearman is computed only within each held-eight seed block. Coverage ratios below "
        "one are better than the exact two-per-held8 random expectation.",
        "",
        "| method | within-held8 Spearman | K8 coverage / exact random | improved pools |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "noise24": "noise24",
        "hidden": "HB5 hidden token-mean",
        "shared": "shared token-mean",
        "routed": "routed expert token-mean",
        "router": "router probability token-mean",
        "expert_scalars": "expert scalars (s1,D,C,Q)",
        "raw_equal": "unweighted selected raw-vector mean",
        "expert_channel_std": "weighted expert channel std",
        "cancellation_gap": "per-channel cancellation gap",
        "base": "fixed base (4 equal blocks)",
        "base_plus_routed": "base + residual routed",
    }
    for method in METHODS:
        value = macro[method]
        lines.append(
            f"| {labels[method]} | {_fmt(value['within_held8_spearman_mean'])} | "
            f"{_fmt(value['coverage_over_exact_stratified_random_mean'])} | "
            f"{value['pools_with_coverage_improvement']}/80 |"
        )

    lines += [
        "",
        "## Standalone routed diagnostic",
        "",
        "Standalone routed is compared with the best observed standalone nonexpert baseline "
        "among noise24, hidden, shared, and router. That baseline choice is descriptive and "
        "data-dependent, so this table cannot make the primary screen positive.",
        "",
        "| task | best baseline | baseline rho | routed rho | routed-best | routed-hidden | routed-shared |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in task_names:
        row = per_task[task]
        comparison = standalone["per_task"][task]
        best = comparison["right_method"]
        lines.append(
            f"| {task} | {best} | "
            f"{_fmt(row[best]['within_held8_spearman_mean'])} | "
            f"{_fmt(row['routed']['within_held8_spearman_mean'])} | "
            f"{comparison['spearman_gain_mean']:+.3f} | "
            f"{direct['hidden'][task]:+.3f} | {direct['shared'][task]:+.3f} |"
        )

    hidden_wins = sum(direct["hidden"][task] > 0.0 for task in task_names)
    shared_wins = sum(direct["shared"][task] > 0.0 for task in task_names)
    lines += [
        "",
        f"Standalone routed exceeds hidden in {hidden_wins}/5 tasks and shared in "
        f"{shared_wins}/5 tasks. It therefore does not satisfy the original five-task "
        "control-consistency question either.",
    ]

    lines += [
        "",
        "## Descriptive raw decomposition controls",
        "",
        "These three controls were added to check whether the weighted routed merge erases "
        "a useful raw-expert signal. They are not part of the primary gate or fixed base; "
        "their own K8 metrics are descriptive only.",
        "",
        "| task | raw equal mean | weighted channel std | cancellation gap |",
        "|---|---:|---:|---:|",
    ]
    for task in task_names:
        row = per_task[task]
        lines.append(
            f"| {task} | {_fmt(row['raw_equal']['within_held8_spearman_mean'])} | "
            f"{_fmt(row['expert_channel_std']['within_held8_spearman_mean'])} | "
            f"{_fmt(row['cancellation_gap']['within_held8_spearman_mean'])} |"
        )

    lines += [
        "",
        "## Per-task method Spearman",
        "",
        "| task | noise24 | hidden | shared | routed | router | scalars |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in task_names:
        row = per_task[task]
        values = " | ".join(
            _fmt(row[method]["within_held8_spearman_mean"])
            for method in ("noise24", "hidden", "shared", "routed", "router", "expert_scalars")
        )
        lines.append(f"| {task} | {values} |")

    lines += [
        "",
        "## Per-pool paired results",
        "",
        "Each pool value is the mean of its four held8 Spearman correlations.",
        "",
        "| task | state | base rho | base+residual rho | rho gain | relative coverage gain | hidden | shared | routed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["pools"]:
        methods = row["methods"]
        coverage_gain = 1.0 - (
            methods["base_plus_routed"]["coverage"] / methods["base"]["coverage"]
        )
        lines.append(
            f"| {row['task']} | {row['state_id']} | "
            f"{_fmt(methods['base']['pool_spearman_mean'])} | "
            f"{_fmt(methods['base_plus_routed']['pool_spearman_mean'])} | "
            f"{methods['base_plus_routed']['pool_spearman_mean'] - methods['base']['pool_spearman_mean']:+.3f} | "
            f"{100.0 * coverage_gain:+.1f}% | "
            f"{_fmt(methods['hidden']['pool_spearman_mean'])} | "
            f"{_fmt(methods['shared']['pool_spearman_mean'])} | "
            f"{_fmt(methods['routed']['pool_spearman_mean'])} |"
        )

    lines += [
        "",
        "## Protocol and boundary",
        "",
        "- Cell selection was frozen at HB5/d0. No layer or denoise round was searched.",
        "- HB5/d0 is evaluated on the initial x0 forward. The endpoint remains the normalized "
        "final action chunk; it is not relabeled as a remaining-correction target.",
        "- The five tasks are the complete right-16x32 sources already referenced by the "
        "expert-activation proxy analysis. Every task has 16 state pools and the same 32 seeds.",
        "- For every held state and held-eight seed block, training uses only the other 15 "
        "states and other 24 seeds. Feature centering/RMS is fitted only on that outer-train "
        "fold and applied to held8. Target centering is used only for outer-train targets; "
        "held8 targets are untouched and pair distances are invariant to a common offset.",
        "- The base kernel is the arithmetic mean of four outer-train trace-normalized block "
        "kernels. Routed residualization uses fixed ridge 1.0 fitted on the same outer train "
        "fold; the augmented kernel preserves the full base kernel and appends only the residual.",
        "- Final-action geometry is the full client `actions[0]` 10x7 chunk divided by the "
        "official checkpoint action standard deviations.",
        "- Routed/shared vectors are offline fp32 checkpoint reconstructions from stored fp16 "
        "hidden states. Recorded selected expert IDs and probabilities remain authoritative; "
        "this is not a runtime-exact or causal intervention.",
        "- Expert IDs are used only within their own checkpoint suite. No expert identity is "
        "aligned or pooled across the Goal, Spatial, and Libero-10 checkpoints.",
        "- The vector features are ten-token means. This screen does not test per-token raw "
        "Gram structure or S_LOO, so the negative result does not exclude those interfaces.",
        "- K8 selection is PAM K2 independently within each held8. The coverage baseline is "
        "the exact combinatorial expectation for two uniform candidates from every held8.",
        "- Paired bootstrap intervals resample the 16 state pools within each fixed task; the "
        "macro interval averages the five task-stratified draws. They are descriptive and are "
        "not used as significance gates.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    out_dir = args.out_dir.resolve()
    cache_path = (
        args.feature_cache.resolve()
        if args.feature_cache is not None
        else out_dir / "features.npz"
    )
    started = time.perf_counter()
    tasks = load_or_reconstruct_features(
        args.source_root.resolve(), cache_path, args.threads, args.force_features
    )
    with threadpool_limits(limits=1):
        payload = build(tasks, args.inner_folds)
    payload["elapsed_seconds"] = finite(time.perf_counter() - started)
    payload["feature_cache"] = str(cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "REPORT.md"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    report_path.write_text(render_report(payload))
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
