#!/usr/bin/env python3
"""Train and evaluate the frozen four-baseline MoE-aware chunk critic panel."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from behavior_forks_v2 import atomic_json
from bestofn_critic import (
    DuelingChunkCritic,
    aggregate_metrics,
    critic_loss,
    hierarchical_delta_ci,
    snapshot_metrics,
)
from bestofn_protocol import load_config, load_json, rng_from_words, seed_words, sha256_file
from build_bestofn_critic_dataset import DATASET_SCHEMA, SPLIT_CODE


RESULT_SCHEMA = "himoe.bestofn.critic_results.v1"
PRIMARY_MODES = (
    "state_action",
    "state_route",
    "state_action_route",
    "state_action_router_hidden",
)


class TrainingError(RuntimeError):
    """Critic data, config, or a completed member is inconsistent."""


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_std(value: np.ndarray, axis: int | tuple[int, ...] = 0) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(value, dtype=np.float32).mean(axis=axis)
    std = np.asarray(value, dtype=np.float32).std(axis=axis)
    return mean, np.maximum(std, 1e-4)


class CriticDataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = load_json(root / "dataset_manifest.json")
        if self.manifest.get("schema") != DATASET_SCHEMA:
            raise TrainingError("critic dataset schema mismatch")
        for name, expected in self.manifest.get("files", {}).items():
            path = root / name
            if not path.is_file() or sha256_file(path) != expected:
                raise TrainingError(f"critic dataset integrity check failed: {path}")
        self.dataset_sha = sha256_file(root / "dataset_manifest.json")
        self.arrays = {
            path.stem: np.load(path, mmap_mode="r", allow_pickle=False)
            for path in sorted(root.glob("*.npy"))
        }
        required = {
            "state_offsets",
            "task_id",
            "split",
            "state_vlm",
            "state_proprio",
            "action",
            "success_count",
            "repeat_count",
            "hidden_candidate",
            "hidden_state",
            "route_candidate_all",
            "route_state_all",
        }
        if not required <= set(self.arrays):
            raise TrainingError(f"critic dataset is missing arrays: {sorted(required - set(self.arrays))}")
        self.offsets = np.asarray(self.arrays["state_offsets"])
        self.n_states = len(self.offsets) - 1
        self.n_candidates = int(self.offsets[-1])

    def states(self, *, split: str | None = None, include_tasks=None, exclude_task=None) -> list[int]:
        result = []
        for index in range(self.n_states):
            if split is not None and int(self.arrays["split"][index]) != SPLIT_CODE[split]:
                continue
            task = int(self.arrays["task_id"][index])
            if include_tasks is not None and task not in set(include_tasks):
                continue
            if exclude_task is not None and task == int(exclude_task):
                continue
            result.append(index)
        return result

    def candidate_indices(self, states: Sequence[int]) -> np.ndarray:
        return np.concatenate(
            [np.arange(self.offsets[index], self.offsets[index + 1]) for index in states]
        )

    def normalization(
        self, states: Sequence[int], *, mode: str, route_variant: str
    ) -> dict[str, np.ndarray]:
        candidates = self.candidate_indices(states)
        result = {}
        result["proprio_mean"], result["proprio_std"] = _safe_std(
            self.arrays["state_proprio"][states]
        )
        if mode in {"state_action", "state_action_route", "state_action_router_hidden"}:
            result["action_mean"], result["action_std"] = _safe_std(
                self.arrays["action"][candidates]
            )
        if mode in {"state_route", "state_action_route"}:
            result["route_candidate_mean"], result["route_candidate_std"] = _safe_std(
                self.arrays[f"route_candidate_{route_variant}"][candidates]
            )
            result["route_state_mean"], result["route_state_std"] = _safe_std(
                self.arrays[f"route_state_{route_variant}"][states]
            )
        if mode == "state_as_route":
            raw = np.asarray(self.arrays["as_route_candidate"][candidates], dtype=np.float32)
            result["route_candidate_mean"], result["route_candidate_std"] = _safe_std(raw)
            state_values = []
            for index in states:
                start, stop = self.offsets[index : index + 2]
                state_values.append(np.asarray(self.arrays["as_route_candidate"][start:stop]).mean(axis=0))
            result["route_state_mean"], result["route_state_std"] = _safe_std(
                np.stack(state_values)
            )
        return result

    def dimensions(self, *, mode: str, route_variant: str) -> dict[str, Any]:
        if mode == "state_as_route":
            route_candidate = route_state = int(self.arrays["as_route_candidate"].shape[1])
        else:
            route_candidate = int(self.arrays[f"route_candidate_{route_variant}"].shape[1])
            route_state = int(self.arrays[f"route_state_{route_variant}"].shape[1])
        return {
            "vlm_dim": int(self.arrays["state_vlm"].shape[1]),
            "action_dim": int(self.arrays["action"].shape[1]),
            "route_candidate_dim": route_candidate,
            "route_state_dim": route_state,
            "hidden_candidate_shape": tuple(self.arrays["hidden_candidate"].shape[1:]),
            "hidden_state_shape": tuple(self.arrays["hidden_state"].shape[1:]),
        }

    def batch(
        self,
        states: Sequence[int],
        *,
        mode: str,
        route_variant: str,
        normalization: Mapping[str, np.ndarray],
        device: torch.device,
        route_transform: str = "none",
        route_source: Mapping[int, int] | None = None,
        transform_seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        b = len(states)
        counts = [int(self.offsets[index + 1] - self.offsets[index]) for index in states]
        kmax = max(counts)
        mask = np.zeros((b, kmax), dtype=np.bool_)
        success = np.zeros((b, kmax), dtype=np.float32)
        repeats = np.ones((b, kmax), dtype=np.float32)
        result: dict[str, np.ndarray] = {
            "state_vlm": np.asarray(self.arrays["state_vlm"][states], dtype=np.float32),
            "state_proprio": (
                np.asarray(self.arrays["state_proprio"][states], dtype=np.float32)
                - normalization["proprio_mean"]
            )
            / normalization["proprio_std"],
        }
        use_action = mode in {"state_action", "state_action_route", "state_action_router_hidden"}
        use_route = mode in {"state_route", "state_action_route", "state_as_route"}
        use_hidden = mode == "state_action_router_hidden"
        if use_action:
            result["action"] = np.zeros((b, kmax, self.arrays["action"].shape[1]), dtype=np.float32)
        if use_route:
            route_dim = self.dimensions(mode=mode, route_variant=route_variant)["route_candidate_dim"]
            state_dim = self.dimensions(mode=mode, route_variant=route_variant)["route_state_dim"]
            result["route_candidate"] = np.zeros((b, kmax, route_dim), dtype=np.float32)
            result["route_state"] = np.zeros((b, state_dim), dtype=np.float32)
        if use_hidden:
            result["hidden_candidate"] = np.zeros(
                (b, kmax, *self.arrays["hidden_candidate"].shape[1:]), dtype=np.float32
            )
            result["hidden_state"] = np.asarray(self.arrays["hidden_state"][states], dtype=np.float32)
        for local, state_index in enumerate(states):
            start, stop = map(int, self.offsets[state_index : state_index + 2])
            k = stop - start
            mask[local, :k] = True
            success[local, :k] = self.arrays["success_count"][start:stop]
            repeats[local, :k] = self.arrays["repeat_count"][start:stop]
            if use_action:
                result["action"][local, :k] = (
                    np.asarray(self.arrays["action"][start:stop], dtype=np.float32)
                    - normalization["action_mean"]
                ) / normalization["action_std"]
            if use_hidden:
                result["hidden_candidate"][local, :k] = self.arrays["hidden_candidate"][start:stop]
            if use_route:
                source_index = state_index if route_source is None else int(route_source[state_index])
                source_start, source_stop = map(int, self.offsets[source_index : source_index + 2])
                if source_stop - source_start != k:
                    raise TrainingError("cross-snapshot routing source has another K")
                if mode == "state_as_route":
                    candidate_route = np.asarray(
                        self.arrays["as_route_candidate"][source_start:source_stop], dtype=np.float32
                    )
                    state_route = candidate_route.mean(axis=0)
                else:
                    candidate_route = np.asarray(
                        self.arrays[f"route_candidate_{route_variant}"][source_start:source_stop],
                        dtype=np.float32,
                    )
                    state_route = np.asarray(
                        self.arrays[f"route_state_{route_variant}"][source_index], dtype=np.float32
                    )
                if route_transform == "same_snapshot_shuffle":
                    rng = np.random.default_rng(
                        np.random.SeedSequence([transform_seed, int(state_index)])
                    )
                    candidate_route = candidate_route[rng.permutation(k)]
                elif route_transform == "candidate_mean":
                    candidate_route = np.broadcast_to(candidate_route.mean(axis=0), candidate_route.shape)
                result["route_candidate"][local, :k] = (
                    candidate_route - normalization["route_candidate_mean"]
                ) / normalization["route_candidate_std"]
                result["route_state"][local] = (
                    state_route - normalization["route_state_mean"]
                ) / normalization["route_state_std"]
                if route_transform == "state_removed":
                    result["route_state"][local] = 0.0
        result["mask"] = mask
        result["success_count"] = success
        result["repeat_count"] = repeats
        return {
            key: torch.as_tensor(value, device=device)
            for key, value in result.items()
        }


@dataclass
class MemberResult:
    model: DuelingChunkCritic
    normalization: dict[str, np.ndarray]
    best_epoch: int
    validation: dict[str, float]
    history: list[dict[str, float]]


def _make_model(config: Mapping[str, Any], dataset: CriticDataset, mode: str, variant: str) -> DuelingChunkCritic:
    parameters = config["critic"]["model"]
    return DuelingChunkCritic(
        mode=mode,
        **dataset.dimensions(mode=mode, route_variant=variant),
        width=int(parameters["width"]),
        latent=int(parameters["latent"]),
        hidden_projection=int(parameters["hidden_projection"]),
    )


def _predict(
    model: DuelingChunkCritic,
    dataset: CriticDataset,
    states: Sequence[int],
    normalization: Mapping[str, np.ndarray],
    *,
    mode: str,
    variant: str,
    device: torch.device,
    batch_states: int,
    route_transform: str = "none",
    route_source: Mapping[int, int] | None = None,
    transform_seed: int = 0,
) -> np.ndarray:
    output = np.full(dataset.n_candidates, np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(states), batch_states):
            group = list(states[offset : offset + batch_states])
            batch = dataset.batch(
                group,
                mode=mode,
                route_variant=variant,
                normalization=normalization,
                device=device,
                route_transform=route_transform,
                route_source=route_source,
                transform_seed=transform_seed,
            )
            logits = model(batch).cpu().numpy()
            for local, state_index in enumerate(group):
                start, stop = map(int, dataset.offsets[state_index : state_index + 2])
                output[start:stop] = logits[local, : stop - start]
    return output


def _metric_rows(dataset: CriticDataset, logits: np.ndarray, states: Sequence[int]):
    return snapshot_metrics(
        logits,
        dataset.arrays["success_count"],
        dataset.arrays["repeat_count"],
        dataset.offsets,
        dataset.arrays["task_id"],
        states,
    )


def train_member(
    config: Mapping[str, Any],
    dataset: CriticDataset,
    *,
    mode: str,
    variant: str,
    seed: int,
    train_states: Sequence[int],
    validation_states: Sequence[int],
    device: torch.device,
) -> MemberResult:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(config["critic"]["training"]["deterministic"])
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    normalization = dataset.normalization(train_states, mode=mode, route_variant=variant)
    model = _make_model(config, dataset, mode, variant).to(device)
    training = config["critic"]["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    batch_states = int(training["batch_snapshots"])
    rng = np.random.default_rng(seed)
    best_score = -math.inf
    best_epoch = -1
    best_state = None
    best_validation = None
    history = []
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        order = np.asarray(train_states)[rng.permutation(len(train_states))]
        losses = []
        for offset in range(0, len(order), batch_states):
            group = order[offset : offset + batch_states].tolist()
            batch = dataset.batch(
                group,
                mode=mode,
                route_variant=variant,
                normalization=normalization,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss, _parts = critic_loss(
                logits,
                batch["success_count"],
                batch["repeat_count"],
                batch["mask"],
                rank_weight=float(training["rank_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_logits = _predict(
            model,
            dataset,
            validation_states,
            normalization,
            mode=mode,
            variant=variant,
            device=device,
            batch_states=batch_states,
        )
        validation = aggregate_metrics(
            _metric_rows(dataset, validation_logits, validation_states)
        )
        pair = validation["pairwise_accuracy"]
        score = pair if np.isfinite(pair) else -validation["binomial_proportion_nll"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "validation_pairwise": float(pair),
                "validation_nll": float(validation["binomial_proportion_nll"]),
            }
        )
        if score > best_score + float(training["minimum_delta"]):
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_validation = validation
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None or best_validation is None:
        raise TrainingError("critic training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return MemberResult(model, normalization, best_epoch, best_validation, history)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _member_paths(root: Path, label: str, member: int) -> tuple[Path, Path]:
    directory = root / "members" / label
    return directory / f"member_{member:02d}.pt", directory / f"member_{member:02d}.json"


def fit_ensemble(
    config: Mapping[str, Any],
    dataset: CriticDataset,
    output: Path,
    *,
    label: str,
    mode: str,
    variant: str,
    seeds: Sequence[int],
    train_states: Sequence[int],
    validation_states: Sequence[int],
    prediction_states: Sequence[int],
    device: torch.device,
) -> tuple[np.ndarray, list[DuelingChunkCritic], list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    predictions = []
    models = []
    normalizations = []
    summaries = []
    batch_states = int(config["critic"]["training"]["batch_snapshots"])
    for member, seed in enumerate(seeds):
        checkpoint_path, metadata_path = _member_paths(output, label, member)
        if checkpoint_path.is_file() and metadata_path.is_file():
            metadata = load_json(metadata_path)
            if (
                metadata.get("dataset_sha256") != dataset.dataset_sha
                or metadata.get("config_fingerprint") != _config_fingerprint(config)
                or metadata.get("label") != label
                or metadata.get("mode") != mode
                or metadata.get("route_variant") != variant
                or int(metadata["seed"]) != int(seed)
            ):
                raise TrainingError(f"completed member belongs to another dataset/seed: {metadata_path}")
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = _make_model(config, dataset, mode, variant).to(device)
            model.load_state_dict(payload["model"])
            normalization = {key: np.asarray(value) for key, value in payload["normalization"].items()}
            summary = metadata
        else:
            result = train_member(
                config,
                dataset,
                mode=mode,
                variant=variant,
                seed=int(seed),
                train_states=train_states,
                validation_states=validation_states,
                device=device,
            )
            model, normalization = result.model, result.normalization
            summary = {
                "dataset_sha256": dataset.dataset_sha,
                "config_fingerprint": _config_fingerprint(config),
                "label": label,
                "mode": mode,
                "route_variant": variant,
                "member": member,
                "seed": int(seed),
                "best_epoch": result.best_epoch,
                "validation": result.validation,
                "history": result.history,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model": model.state_dict(),
                    "normalization": {key: np.asarray(value) for key, value in normalization.items()},
                },
            )
            atomic_json(metadata_path, summary)
        prediction = _predict(
            model,
            dataset,
            prediction_states,
            normalization,
            mode=mode,
            variant=variant,
            device=device,
            batch_states=batch_states,
        )
        predictions.append(prediction)
        models.append(model)
        normalizations.append(normalization)
        summaries.append(summary)
    return np.stack(predictions), models, normalizations, summaries


def _cross_snapshot_sources(dataset: CriticDataset, states: Sequence[int], seed: int) -> dict[int, int]:
    groups: dict[tuple[int, int], list[int]] = {}
    for state in states:
        task = int(dataset.arrays["task_id"][state])
        k = int(dataset.offsets[state + 1] - dataset.offsets[state])
        groups.setdefault((task, k), []).append(state)
    mapping = {}
    rng = np.random.default_rng(seed)
    for values in groups.values():
        values = sorted(values)
        if len(values) < 2:
            mapping.update({value: value for value in values})
            continue
        shift = int(rng.integers(1, len(values)))
        mapping.update({value: values[(index + shift) % len(values)] for index, value in enumerate(values)})
    return mapping


def run(config_path: Path, dataset_root: Path, output: Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    dataset = CriticDataset(dataset_root)
    if dataset.manifest["config_sha256"] != sha256_file(config_path):
        raise TrainingError("critic dataset was assembled under another config")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    train_states = dataset.states(split="train")
    validation_states = dataset.states(split="validation")
    test_states = dataset.states(split="test")
    seeds = [int(value) for value in config["critic"]["ensemble_seeds"]]
    predictions = {}
    model_panels = {}
    normalization_panels = {}
    members = {}
    primary = {}
    for mode in PRIMARY_MODES:
        panel, models, norms, summaries = fit_ensemble(
            config,
            dataset,
            output,
            label=mode,
            mode=mode,
            variant="all",
            seeds=seeds,
            train_states=train_states,
            validation_states=validation_states,
            prediction_states=list(range(dataset.n_states)),
            device=device,
        )
        mean = np.nanmean(panel, axis=0)
        predictions[mode] = mean
        model_panels[mode], normalization_panels[mode], members[mode] = models, norms, summaries
        primary[mode] = {
            "validation": aggregate_metrics(_metric_rows(dataset, mean, validation_states)),
            "test": aggregate_metrics(_metric_rows(dataset, mean, test_states)),
            "ensemble_members": len(seeds),
            "mean_parameters": float(np.mean([row["parameters"] for row in summaries])),
        }

    action_rows = _metric_rows(dataset, predictions["state_action"], test_states)
    joint_rows = _metric_rows(dataset, predictions["state_action_route"], test_states)
    bootstrap_draws = int(config["critic"]["evaluation"]["bootstrap_draws"])
    rng = rng_from_words(
        seed_words(int(config["master_seed"]), "bestofn/critic/bootstrap", 0, 0, 0)
    )
    selected_delta, selected_ci = hierarchical_delta_ci(
        joint_rows,
        action_rows,
        field="selected_q",
        draws=bootstrap_draws,
        rng=rng,
    )
    regret_delta, regret_ci = hierarchical_delta_ci(
        joint_rows,
        action_rows,
        field="top1_regret",
        draws=bootstrap_draws,
        rng=rng,
    )
    pair_delta, pair_ci = hierarchical_delta_ci(
        joint_rows,
        action_rows,
        field="pairwise_accuracy",
        draws=bootstrap_draws,
        rng=rng,
    )
    gate2 = {
        "selected_q_delta_joint_minus_action": selected_delta,
        "selected_q_delta_ci95": selected_ci,
        "top1_regret_delta_joint_minus_action": regret_delta,
        "top1_regret_delta_ci95": regret_ci,
        "pairwise_accuracy_delta_joint_minus_action": pair_delta,
        "pairwise_accuracy_delta_ci95": pair_ci,
    }
    gate2["passed"] = bool(
        selected_delta > 0
        and selected_ci[0] > 0
        and pair_delta > 0
        and regret_delta < 0
    )

    joint_models = model_panels["state_action_route"]
    joint_norms = normalization_panels["state_action_route"]
    batch_states = int(config["critic"]["training"]["batch_snapshots"])
    ablations = {}
    transforms = {
        "same_snapshot_route_shuffle": ("all", "same_snapshot_shuffle", None),
        "cross_snapshot_route_shuffle": (
            "all",
            "none",
            _cross_snapshot_sources(dataset, test_states, int(config["master_seed"])),
        ),
        "candidate_route_removed": ("all", "candidate_mean", None),
        "state_route_removed": ("all", "state_removed", None),
        "flow_order_reversed": ("reversed", "none", None),
        "early_flow_only": ("early", "none", None),
        "middle_flow_only": ("middle", "none", None),
        "late_flow_only": ("late", "none", None),
        "final_flow_only": ("final", "none", None),
    }
    for name, (variant, transform, sources) in transforms.items():
        panel = []
        for member, (model, normalization) in enumerate(zip(joint_models, joint_norms)):
            panel.append(
                _predict(
                    model,
                    dataset,
                    test_states,
                    normalization,
                    mode="state_action_route",
                    variant=variant,
                    device=device,
                    batch_states=batch_states,
                    route_transform=transform,
                    route_source=sources,
                    transform_seed=int(config["master_seed"]) + member,
                )
            )
        mean = np.nanmean(np.stack(panel), axis=0)
        ablations[name] = aggregate_metrics(_metric_rows(dataset, mean, test_states))

    control_seeds = [int(value) for value in config["critic"]["control_seeds"]]
    controls = {}
    for label, mode, variant in (
        ("as_moe_negative_control", "state_as_route", "all"),
        ("expert_id_permutation_retrained", "state_action_route", "expert_permuted"),
    ):
        panel, _models, _norms, summaries = fit_ensemble(
            config,
            dataset,
            output,
            label=label,
            mode=mode,
            variant=variant,
            seeds=control_seeds,
            train_states=train_states,
            validation_states=validation_states,
            prediction_states=list(range(dataset.n_states)),
            device=device,
        )
        mean = np.nanmean(panel, axis=0)
        controls[label] = {
            "test": aggregate_metrics(_metric_rows(dataset, mean, test_states)),
            "ensemble_members": len(control_seeds),
            "mean_parameters": float(np.mean([row["parameters"] for row in summaries])),
        }

    loto = {}
    loto_seeds = [int(value) for value in config["critic"]["loto_seeds"]]
    for held_task in sorted(set(np.asarray(dataset.arrays["task_id"]).tolist())):
        fold_train = dataset.states(split="train", exclude_task=held_task)
        fold_validation = dataset.states(split="validation", exclude_task=held_task)
        fold_test = dataset.states(include_tasks={held_task})
        fold = {}
        for mode in ("state_action", "state_action_route"):
            label = f"loto_task{held_task}_{mode}"
            panel, _models, _norms, _summaries = fit_ensemble(
                config,
                dataset,
                output,
                label=label,
                mode=mode,
                variant="all",
                seeds=loto_seeds,
                train_states=fold_train,
                validation_states=fold_validation,
                prediction_states=fold_test,
                device=device,
            )
            mean = np.nanmean(panel, axis=0)
            fold[mode] = aggregate_metrics(_metric_rows(dataset, mean, fold_test))
        loto[str(held_task)] = fold

    result = {
        "schema": RESULT_SCHEMA,
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset": str(dataset_root),
        "dataset_manifest_sha256": dataset.dataset_sha,
        "device": str(device),
        "evaluation_unit": "task-then-snapshot; candidates and pairs are never independent units",
        "primary_parent_episode_grouped": primary,
        "gate_2": gate2,
        "route_exclusion_ablations": ablations,
        "retrained_controls": controls,
        "leave_one_task_out": loto,
        "closed_loop_gate_3_authorized": bool(gate2["passed"]),
    }
    atomic_json(output / "results.json", result)
    return result


def render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# MoE-aware chunk critic",
        "",
        "> New counterfactual terminal-Q labels only; no legacy candidate/outcome data.",
        "",
        "| critic | pair accuracy | top-1 regret | selected q | recovery | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["primary_parent_episode_grouped"].items():
        row = value["test"]
        lines.append(
            "| %s | %.4f | %.4f | %.4f | %.4f | %.4f |"
            % (
                name,
                row["pairwise_accuracy"],
                row["top1_regret"],
                row["selected_q"],
                row["oracle_recovery_ratio"],
                row["brier"],
            )
        )
    gate = result["gate_2"]
    lines.extend(
        [
            "",
            "## Gate 2",
            "",
            "- Decision: **%s**." % ("PASS" if gate["passed"] else "STOP"),
            "- Joint minus action selected-q: %+.4f, 95%% CI [%+.4f, %+.4f]."
            % (
                gate["selected_q_delta_joint_minus_action"],
                gate["selected_q_delta_ci95"][0],
                gate["selected_q_delta_ci95"][1],
            ),
            "- Closed-loop Gate 3 authorized: `%s`."
            % str(result["closed_loop_gate_3_authorized"]).lower(),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        args.config.expanduser().resolve(),
        args.dataset.expanduser().resolve(),
        args.out.expanduser().resolve(),
        args.device,
    )
    (args.out.expanduser().resolve() / "REPORT.md").write_text(
        render_report(result), encoding="utf-8"
    )
    print("Gate 2 %s" % ("PASS" if result["gate_2"]["passed"] else "STOP"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
