from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

N_LAYERS = 8
N_FLOW = 10
N_TOKENS = 11
N_EXPERTS = 32
TOP_K = 4

LAYER_METRICS = (
    "entropy",
    "margin",
    "top1_mass",
    "top4_mass",
    "soft_token_consensus",
    "top4_token_consensus",
    "effective_rank",
    "flow_velocity",
    "flow_top4_switch",
    "flow_acceleration",
)
GLOBAL_METRICS = ("layer_profile_disagreement",)

LEVEL_METRICS = (
    "entropy",
    "margin",
    "top1_mass",
    "top4_mass",
    "soft_token_consensus",
    "top4_token_consensus",
    "effective_rank",
)
NATIVE_DYNAMICS_METRICS = (
    "flow_velocity",
    "flow_top4_switch",
    "flow_acceleration",
)


LEGACY_GLOBAL_METRIC = "layer_disagreement"


def step_feature_names() -> tuple[str, ...]:
    names = [f"{metric}|layer_{layer}" for metric in LAYER_METRICS for layer in range(N_LAYERS)]
    names.extend(GLOBAL_METRICS)
    return tuple(names)


def global_metric_column(names: Sequence[str]) -> int:
    """Resolve the cross-layer disagreement column in corrected or legacy artifacts.

    Legacy artifacts store the permutation-dependent ``layer_disagreement`` in the same
    slot. Callers that only need the column position may accept either, but numeric
    comparisons across the two schemas are not meaningful.
    """
    names = list(names)
    for candidate in (GLOBAL_METRICS[0], LEGACY_GLOBAL_METRIC):
        if candidate in names:
            return names.index(candidate)
    raise ValueError("feature names contain no cross-layer disagreement column")


def descriptor_feature_names(names: Sequence[str] | None = None) -> tuple[str, ...]:
    names = tuple(names or step_feature_names())
    output: list[str] = []
    for derivative, start in (("level", 0), ("delta", 1), ("accel", 2)):
        for flow in range(start, N_FLOW):
            output.extend(f"{derivative}|flow_{flow}|{name}" for name in names)
    return tuple(output)


def build_query_descriptors(features: np.ndarray) -> np.ndarray:
    """Reproduce the original 2,187-D descriptor used by the GMM/PST audit.

    This legacy representation differentiates native velocity and acceleration
    tracks again. New experiments should use ``build_clean_query_descriptors``.
    """
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (N_FLOW, len(step_feature_names())):
        raise ValueError(f"expected [N,{N_FLOW},{len(step_feature_names())}], got {values.shape}")
    delta = np.diff(values, axis=1)
    acceleration = np.diff(values, n=2, axis=1)
    return np.concatenate(
        [
            values.reshape(len(values), -1),
            delta.reshape(len(values), -1),
            acceleration.reshape(len(values), -1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def clean_descriptor_feature_names(
    names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    names = tuple(names or step_feature_names())
    output: list[str] = []
    bins = ("early", "middle", "late")
    for name in names:
        metric = name.split("|", 1)[0]
        output.extend(f"{flow_bin}|{name}" for flow_bin in bins)
        if metric in LEVEL_METRICS or metric == GLOBAL_METRICS[0]:
            output.append(f"flow_slope|{name}")
    return tuple(output)


def build_clean_query_descriptors(
    features: np.ndarray,
    names: Sequence[str] | None = None,
) -> np.ndarray:
    """Summarize flow curves without applying derivatives to native dynamics."""

    values = np.asarray(features, dtype=np.float32)
    names = tuple(names or step_feature_names())
    if values.ndim != 3 or values.shape[1:] != (N_FLOW, len(names)):
        raise ValueError(f"expected [N,{N_FLOW},{len(names)}], got {values.shape}")
    time = np.arange(N_FLOW, dtype=np.float32)
    time -= time.mean()
    denominator = np.square(time).sum()
    parts: list[np.ndarray] = []
    for column, name in enumerate(names):
        curve = values[:, :, column]
        parts.extend(
            [
                curve[:, :3].mean(axis=1),
                curve[:, 3:7].mean(axis=1),
                curve[:, 7:].mean(axis=1),
            ]
        )
        metric = name.split("|", 1)[0]
        if metric in LEVEL_METRICS or metric == GLOBAL_METRICS[0]:
            parts.append((curve * time[None, :]).sum(axis=1) / denominator)
    result = np.column_stack(parts).astype(np.float32, copy=False)
    if result.shape[1] != len(clean_descriptor_feature_names(names)):
        raise AssertionError("clean descriptor name and value shapes differ")
    return result


def deterministic_flow_permutations(count: int, seed: int, row_offset: int = 0) -> np.ndarray:
    """Generate stable, non-identity per-query permutations independent of batching."""
    output = np.empty((count, N_FLOW), dtype=np.int64)
    for local_index in range(count):
        row = row_offset + local_index
        rng = np.random.default_rng(np.random.SeedSequence([seed, row]))
        permutation = rng.permutation(N_FLOW)
        if np.array_equal(permutation, np.arange(N_FLOW)):
            permutation = np.roll(permutation, 1)
        output[local_index] = permutation
    return output


def permute_flow_tensor(values: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    if values.shape[0] != permutation.shape[0] or values.shape[2] != N_FLOW:
        raise ValueError("flow permutation shape does not match route tensor")
    index_shape = [values.shape[0], 1, N_FLOW] + [1] * (values.ndim - 3)
    expand_shape = list(values.shape)
    index = permutation.view(index_shape).expand(expand_shape)
    return torch.gather(values, dim=2, index=index)


def _mean_upper_triangle(values: torch.Tensor) -> torch.Tensor:
    size = values.shape[-1]
    row, column = torch.triu_indices(size, size, offset=1, device=values.device)
    return values[..., row, column].mean(dim=-1)


def _hellinger_from_bhattacharyya(similarity: torch.Tensor) -> torch.Tensor:
    similarity = torch.where(similarity > 1.0 - 1e-6, torch.ones_like(similarity), similarity)
    return torch.sqrt(torch.clamp(1.0 - similarity, min=0.0, max=1.0))


@torch.inference_mode()
def extract_multitrack_features(
    probability: torch.Tensor, expert_ids: torch.Tensor
) -> torch.Tensor:
    """Extract [batch, flow, 81] permutation-invariant routing phenotypes."""
    if tuple(probability.shape[1:]) != (N_LAYERS, N_FLOW, N_TOKENS, N_EXPERTS):
        raise ValueError(f"unexpected probability shape {tuple(probability.shape)}")
    if tuple(expert_ids.shape[1:]) != (N_LAYERS, N_FLOW, N_TOKENS, TOP_K):
        raise ValueError(f"unexpected expert-id shape {tuple(expert_ids.shape)}")

    probability = torch.clamp(probability.float(), min=0.0)
    probability = probability / torch.clamp(probability.sum(dim=-1, keepdim=True), min=1e-12)
    expert_ids = expert_ids.long()
    if torch.any((expert_ids < 0) | (expert_ids >= N_EXPERTS)):
        raise ValueError("expert id outside valid range")

    entropy = -(probability * torch.log(torch.clamp(probability, min=1e-12))).sum(dim=-1)
    entropy = entropy.mean(dim=-1) / math.log(N_EXPERTS)
    top = torch.topk(probability, k=TOP_K, dim=-1, sorted=True).values
    margin = (top[..., 0] - top[..., 1]).mean(dim=-1)
    top1_mass = top[..., 0].mean(dim=-1)
    top4_mass = top.sum(dim=-1).mean(dim=-1)

    sqrt_probability = torch.sqrt(probability)
    token_bc = sqrt_probability @ sqrt_probability.transpose(-1, -2)
    token_distance = _hellinger_from_bhattacharyya(token_bc)
    soft_consensus = 1.0 - _mean_upper_triangle(token_distance)

    support = torch.zeros_like(probability)
    support.scatter_(-1, expert_ids, 1.0)
    intersection = support @ support.transpose(-1, -2)
    union = 2.0 * TOP_K - intersection
    support_jaccard = intersection / torch.clamp(union, min=1.0)
    top4_consensus = _mean_upper_triangle(support_jaccard)

    gram = probability @ probability.transpose(-1, -2)
    eigenvalue = torch.clamp(torch.linalg.eigvalsh(gram), min=0.0)
    singular_value = torch.sqrt(eigenvalue)
    singular_weight = singular_value / torch.clamp(
        singular_value.sum(dim=-1, keepdim=True), min=1e-12
    )
    effective_rank = torch.exp(
        -(singular_weight * torch.log(torch.clamp(singular_weight, min=1e-12))).sum(dim=-1)
    ) / min(N_TOKENS, N_EXPERTS)

    flow_bc = (sqrt_probability[:, :, 1:] * sqrt_probability[:, :, :-1]).sum(dim=-1)
    flow_distance = _hellinger_from_bhattacharyya(flow_bc).mean(dim=-1)
    flow_velocity = torch.zeros_like(entropy)
    flow_velocity[:, :, 1:] = flow_distance

    current_ids = expert_ids[:, :, 1:, :, :, None]
    previous_ids = expert_ids[:, :, :-1, :, None, :]
    flow_intersection = (current_ids == previous_ids).any(dim=-1).sum(dim=-1).float()
    flow_jaccard = flow_intersection / torch.clamp(2.0 * TOP_K - flow_intersection, min=1.0)
    flow_support_switch = torch.zeros_like(entropy)
    flow_support_switch[:, :, 1:] = (1.0 - flow_jaccard).mean(dim=-1)

    second_difference = (
        sqrt_probability[:, :, 2:]
        - 2.0 * sqrt_probability[:, :, 1:-1]
        + sqrt_probability[:, :, :-2]
    )
    acceleration = torch.zeros_like(entropy)
    acceleration[:, :, 1:-1] = torch.linalg.vector_norm(second_difference, dim=-1).mean(dim=-1)

    layer_metrics = (
        entropy,
        margin,
        top1_mass,
        top4_mass,
        soft_consensus,
        top4_consensus,
        effective_rank,
        flow_velocity,
        flow_support_switch,
        acceleration,
    )
    # Expert identities are independently permutable in every layer. Compare layers
    # through invariant scalar profiles rather than aligning equal numeric IDs.
    invariant_profile = torch.stack(layer_metrics[:7], dim=-1)
    layer_profile_disagreement = invariant_profile.std(dim=1, unbiased=False).mean(dim=-1)
    flattened = [metric.permute(0, 2, 1) for metric in layer_metrics]
    flattened.append(layer_profile_disagreement.unsqueeze(-1))
    result = torch.cat(flattened, dim=-1)
    if result.shape[1:] != (N_FLOW, len(step_feature_names())):
        raise AssertionError(f"internal feature shape mismatch: {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise ValueError("non-finite routing feature")
    return result
