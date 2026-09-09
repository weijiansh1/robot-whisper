#!/usr/bin/env python3
"""Audit whether cached HiMoE routing is numerically identifiable.

The cache contains the actual expert IDs used by the model, but full routing
probabilities were persisted as float16.  This script separates three questions:

1. What numeric chain is directly evidenced by source, runtime probes, and Zarr?
2. How far is the fourth/fifth-expert boundary from dtype rounding scales?
3. Do old hard-ID turnover results survive stable-only or margin weighting?

No claim about the unavailable historical fp32 logits is made.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE_ROOT = ROOT / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/router-quantization-identifiability"

HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
N_LAYERS = 8
N_DENOISE = 10
N_TOKENS = 11
N_EXPERTS = 32
TOP_K = 4

SUM_FIELDS = (
    "p4_p5_margin",
    "p1",
    "top4_mass",
    "entropy_normalised",
    "probability_sum_abs_error",
    "gap_exact_zero",
    "stable_fp16_rounding_bound",
    "stable_bf16_rounding_bound",
    "stable_fp32_rounding_bound",
    "actual_vs_saved_fp16_top4_disagree",
    "actual_vs_bf16_recast_top4_disagree",
    "saved_fp16_vs_bf16_recast_top4_disagree",
    "actual_selection_unique_from_saved_probs",
    "actual_selection_boundary_ambiguous",
    "actual_selection_contradicted_by_saved_probs",
    "bf16_recast_vector_exact",
    "bf16_recast_element_exact_fraction",
    "bf16_rounding_bound",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--sample-modulus", type=int, default=64)
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="regenerate summary/report/figure from existing CSVs without rereading route arrays",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def bf16_round(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round-to-nearest-even), returned as float32."""
    source = np.ascontiguousarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def half_ulp(values: np.ndarray, dtype: str) -> np.ndarray:
    """Conservative half-width of the local rounding cell."""
    source = np.asarray(values, dtype=np.float32)
    if dtype == "bf16":
        centre = bf16_round(source)
        words = centre.view(np.uint32) >> np.uint32(16)
        upper = ((words + np.uint32(1)) << np.uint32(16)).view(np.float32)
        lower = ((words - np.uint32(1)) << np.uint32(16)).view(np.float32)
    elif dtype == "fp16":
        centre16 = source.astype(np.float16)
        centre = centre16.astype(np.float32)
        upper = np.nextafter(centre16, np.float16(np.inf)).astype(np.float32)
        lower = np.nextafter(centre16, np.float16(-np.inf)).astype(np.float32)
    elif dtype == "fp32":
        centre = source
        upper = np.nextafter(centre, np.float32(np.inf))
        lower = np.nextafter(centre, np.float32(-np.inf))
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    return 0.5 * np.maximum(upper - centre, centre - lower)


def deterministic_top4(scores: np.ndarray) -> np.ndarray:
    """Top-4 set with expert-index tie breaking."""
    return np.argsort(-scores, axis=-1, kind="stable")[..., :TOP_K].astype(np.uint8)


def set_disagreement(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.any(np.sort(left, axis=-1) != np.sort(right, axis=-1), axis=-1)


def top4_turnover(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    shared = (left[..., :, None] == right[..., None, :]).sum(axis=(-1, -2))
    return 1.0 - shared / float(TOP_K)


def line_of(path: Path, needle: str) -> int:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return number
    raise ValueError(f"could not find {needle!r} in {path}")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class CacheInfo:
    cache_id: str
    task: str
    variant: str
    scope: str
    run: Path
    store_path: Path
    rows: int
    probs_dtype: str
    ids_dtype: str
    probs_shape: tuple[int, ...]
    ids_shape: tuple[int, ...]
    hook_verified_calls: int
    hook_verify_failures: int
    store_full_probs: bool | None
    pin_regime: str | None


def discover_caches(cache_root: Path) -> list[CacheInfo]:
    infos = []
    metadata_paths = sorted(cache_root.glob("**/server/routes.zarr/hb_router_probs/zarr.json"))
    for metadata in metadata_paths:
        store_path = metadata.parents[1]
        run = store_path.parent.parent
        rel = run.relative_to(cache_root)
        variant = rel.parts[-1]
        task = "/".join(rel.parts[:-1])
        if variant in {"right-16x32", "routes-v1"}:
            scope = "primary_canonical"
        elif variant == "pin-base":
            scope = "duplicate_baseline"
        else:
            scope = "routing_intervention"

        root = zarr.open(store=str(store_path), mode="r")
        probs = root["hb_router_probs"]
        ids = root["hb_expert_ids"]
        expected_probs = (N_LAYERS, N_DENOISE, N_TOKENS, N_EXPERTS)
        expected_ids = (N_LAYERS, N_DENOISE, N_TOKENS, TOP_K)
        if tuple(probs.shape[1:]) != expected_probs or tuple(ids.shape[1:]) != expected_ids:
            raise ValueError(f"unexpected route schema in {store_path}")
        if int(probs.shape[0]) != int(ids.shape[0]):
            raise ValueError(f"row mismatch in {store_path}")

        summary_path = run / "server/capture_summary.json"
        capture = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        pin = capture.get("pin")
        infos.append(
            CacheInfo(
                cache_id=str(rel),
                task=task,
                variant=variant,
                scope=scope,
                run=run,
                store_path=store_path,
                rows=int(probs.shape[0]),
                probs_dtype=str(probs.dtype),
                ids_dtype=str(ids.dtype),
                probs_shape=tuple(int(x) for x in probs.shape),
                ids_shape=tuple(int(x) for x in ids.shape),
                hook_verified_calls=int(capture.get("hook_verified_calls", 0)),
                hook_verify_failures=len(capture.get("hook_verify_failures", [])),
                store_full_probs=capture.get("store_full_probs"),
                pin_regime=pin.get("regime") if isinstance(pin, dict) else None,
            )
        )
    if not infos:
        raise RuntimeError(f"no full-probability route caches under {cache_root}")
    return infos


@dataclass
class StratumAccumulator:
    sample_modulus: int
    n: np.ndarray = field(default_factory=lambda: np.zeros((N_LAYERS, N_TOKENS), np.int64))
    sums: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            name: np.zeros((N_LAYERS, N_TOKENS), np.float64) for name in SUM_FIELDS
        }
    )
    margin_samples: list[list[np.ndarray]] = field(
        default_factory=lambda: [[] for _ in range(N_LAYERS * N_TOKENS)]
    )
    bound_samples: list[list[np.ndarray]] = field(
        default_factory=lambda: [[] for _ in range(N_LAYERS * N_TOKENS)]
    )

    def update(
        self,
        metrics: dict[str, np.ndarray],
        gap: np.ndarray,
        bf16_bound: np.ndarray,
        start_row: int,
    ) -> None:
        block_rows = gap.shape[0]
        self.n += block_rows * N_DENOISE
        for name in SUM_FIELDS:
            self.sums[name] += np.sum(metrics[name], axis=(0, 2), dtype=np.float64)

        query = np.arange(start_row, start_row + block_rows, dtype=np.int64)[:, None]
        denoise = np.arange(N_DENOISE, dtype=np.int64)[None, :]
        for layer in range(N_LAYERS):
            for token in range(N_TOKENS):
                selector = (
                    query * 1_000_003 + denoise * 9_176 + layer * 131 + token * 17
                ) % self.sample_modulus == 0
                index = layer * N_TOKENS + token
                self.margin_samples[index].append(gap[:, layer, :, token][selector])
                self.bound_samples[index].append(bf16_bound[:, layer, :, token][selector])

    def merge(self, other: "StratumAccumulator") -> None:
        self.n += other.n
        for name in SUM_FIELDS:
            self.sums[name] += other.sums[name]
        for index in range(N_LAYERS * N_TOKENS):
            self.margin_samples[index].extend(other.margin_samples[index])
            self.bound_samples[index].extend(other.bound_samples[index])

    def _sample(self, layer: int, tokens: list[int]) -> tuple[np.ndarray, np.ndarray]:
        margin_parts = []
        bound_parts = []
        for token in tokens:
            index = layer * N_TOKENS + token
            margin_parts.extend(self.margin_samples[index])
            bound_parts.extend(self.bound_samples[index])
        margin = np.concatenate(margin_parts) if margin_parts else np.empty(0)
        bound = np.concatenate(bound_parts) if bound_parts else np.empty(0)
        return margin, bound

    def row(self, cache_id: str, scope: str, layer: int, tokens: list[int]) -> dict[str, Any]:
        count = int(self.n[layer, tokens].sum())
        if count == 0:
            raise ValueError("empty stratum")
        output: dict[str, Any] = {
            "cache_id": cache_id,
            "scope": scope,
            "layer": HB_LAYERS[layer],
            "token_index": tokens[0] if len(tokens) == 1 else "all",
            "token_role": "state" if tokens == [0] else "action" if tokens == list(range(1, 11)) else "mixed",
            "n_sites": count,
        }
        for name in SUM_FIELDS:
            output[name] = float(self.sums[name][layer, tokens].sum() / count)
        output["actual_vs_saved_fp32_upcast_top4_disagree"] = output[
            "actual_vs_saved_fp16_top4_disagree"
        ]
        output["saved_fp16_vs_fp32_upcast_top4_disagree"] = 0.0

        margin, bound = self._sample(layer, tokens)
        for quantile, label in ((0.05, "p05"), (0.5, "median"), (0.95, "p95")):
            output[f"p4_p5_margin_{label}"] = float(np.quantile(margin, quantile))
        ratio = np.divide(
            margin,
            bound,
            out=np.full_like(margin, np.inf, dtype=np.float64),
            where=bound > 0,
        )
        output["gap_over_bf16_rounding_bound_median"] = float(np.median(ratio))
        output["n_margin_sample"] = int(len(margin))
        return output


def block_metrics(probs: np.ndarray, actual_ids: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if probs.shape[-1] != N_EXPERTS or actual_ids.shape[-1] != TOP_K:
        raise ValueError("invalid routing block")
    five = np.partition(probs, -5, axis=-1)[..., -5:]
    five.sort(axis=-1)
    p5 = five[..., 0]
    p4 = five[..., 1]
    p1 = five[..., 4]
    gap = p4 - p5
    top4_mass = five[..., 1:].sum(axis=-1)

    fp16_ids = deterministic_top4(probs)
    actual_vs_fp16 = set_disagreement(actual_ids, fp16_ids)
    bf16_probs = bf16_round(probs)
    bf16_ids = deterministic_top4(bf16_probs)
    actual_vs_bf16 = set_disagreement(actual_ids, bf16_ids)
    fp16_vs_bf16 = set_disagreement(fp16_ids, bf16_ids)

    selected = np.take_along_axis(probs, actual_ids.astype(np.int64), axis=-1)
    available = np.ones(probs.shape, dtype=bool)
    np.put_along_axis(available, actual_ids.astype(np.int64), False, axis=-1)
    strongest_unselected = np.max(np.where(available, probs, -np.inf), axis=-1)
    weakest_selected = np.min(selected, axis=-1)
    compatibility_gap = weakest_selected - strongest_unselected

    fp16_bound = half_ulp(p4, "fp16") + half_ulp(p5, "fp16")
    bf16_bound = half_ulp(p4, "bf16") + half_ulp(p5, "bf16")
    fp32_bound = half_ulp(p4, "fp32") + half_ulp(p5, "fp32")

    total = probs.sum(axis=-1)
    normalised = probs / total[..., None]
    entropy = -np.sum(normalised * np.log(np.maximum(normalised, 1e-30)), axis=-1)
    metrics = {
        "p4_p5_margin": gap,
        "p1": p1,
        "top4_mass": top4_mass,
        "entropy_normalised": entropy / math.log(N_EXPERTS),
        "probability_sum_abs_error": np.abs(total - 1.0),
        "gap_exact_zero": gap == 0,
        "stable_fp16_rounding_bound": gap > fp16_bound,
        "stable_bf16_rounding_bound": gap > bf16_bound,
        "stable_fp32_rounding_bound": gap > fp32_bound,
        "actual_vs_saved_fp16_top4_disagree": actual_vs_fp16,
        "actual_vs_bf16_recast_top4_disagree": actual_vs_bf16,
        "saved_fp16_vs_bf16_recast_top4_disagree": fp16_vs_bf16,
        "actual_selection_unique_from_saved_probs": compatibility_gap > 0,
        "actual_selection_boundary_ambiguous": compatibility_gap == 0,
        "actual_selection_contradicted_by_saved_probs": compatibility_gap < 0,
        "bf16_recast_vector_exact": np.all(bf16_probs == probs, axis=-1),
        "bf16_recast_element_exact_fraction": np.mean(bf16_probs == probs, axis=-1),
        "bf16_rounding_bound": bf16_bound,
    }
    return metrics, gap, bf16_bound


def audit_cache(info: CacheInfo, block_rows: int, sample_modulus: int) -> StratumAccumulator:
    store = zarr.open(store=str(info.store_path), mode="r")
    accumulator = StratumAccumulator(sample_modulus=sample_modulus)
    for start in range(0, info.rows, block_rows):
        stop = min(info.rows, start + block_rows)
        probs = np.asarray(store["hb_router_probs"][start:stop], dtype=np.float32)
        actual_ids = np.asarray(store["hb_expert_ids"][start:stop], dtype=np.uint8)
        if not np.isfinite(probs).all() or np.any(probs < 0):
            raise ValueError(f"invalid probabilities in {info.cache_id}, rows {start}:{stop}")
        metrics, gap, bf16_bound = block_metrics(probs, actual_ids)
        accumulator.update(metrics, gap, bf16_bound, start)
    return accumulator


def inventory_frame(infos: list[CacheInfo]) -> pd.DataFrame:
    rows = []
    for info in infos:
        rows.append(
            {
                "cache_id": info.cache_id,
                "task": info.task,
                "variant": info.variant,
                "scope": info.scope,
                "control_rows": info.rows,
                "routing_sites": info.rows * N_LAYERS * N_DENOISE * N_TOKENS,
                "hb_router_probs_shape": "x".join(map(str, info.probs_shape)),
                "hb_router_probs_zarr_dtype": info.probs_dtype,
                "hb_expert_ids_shape": "x".join(map(str, info.ids_shape)),
                "hb_expert_ids_zarr_dtype": info.ids_dtype,
                "store_full_probs_metadata": info.store_full_probs,
                "hook_verified_calls": info.hook_verified_calls,
                "hook_verify_failures": info.hook_verify_failures,
                "pin_regime": info.pin_regime,
                "store_path": relative(info.store_path),
            }
        )
    return pd.DataFrame(rows)


def numeric_chain_frame(infos: list[CacheInfo]) -> pd.DataFrame:
    model = ROOT / (
        "himoe-vla-cache/himoe-libero-bridge/cache/upstream/HiMoE-VLA/"
        "src/moevla/models/modeling_moe.py"
    )
    policy = ROOT / (
        "himoe-vla-cache/himoe-libero-bridge/cache/upstream/HiMoE-VLA/"
        "src/moevla/policies/policy.py"
    )
    recorder = HERE / "himoe_router_recorder.py"
    writer = HERE / "himoe_route_store.py"
    probe = HERE / "probe_near_tie.py"
    probe_summary = HERE / "analysis/near-tie/summary.json"
    example_meta = infos[0].store_path / "hb_router_probs/zarr.json"
    rows = [
        {
            "stage": "policy inference context",
            "operation": "CUDA autocast requested as bfloat16",
            "dtype_evidence": "torch.bfloat16",
            "evidence_kind": "source",
            "evidence": f"{relative(policy)}:{line_of(policy, 'torch.autocast')}" ,
            "direct_for_historical_cache": True,
            "limitation": "autocast policy alone does not log every operator output dtype",
        },
        {
            "stage": "HB logits",
            "operation": "F.linear(hidden_states, weight)",
            "dtype_evidence": "not separately logged",
            "evidence_kind": "source",
            "evidence": f"{relative(model)}:{line_of(model, 'logits = F.linear')}" ,
            "direct_for_historical_cache": False,
            "limitation": "requires fresh instrumentation for historical GPU logits dtype",
        },
        {
            "stage": "HB softmax scores",
            "operation": "logits.softmax(dim=-1)",
            "dtype_evidence": "torch.bfloat16 in separate CPU-autocast runtime probe",
            "evidence_kind": "source + CPU runtime probe",
            "evidence": (
                f"{relative(model)}:{line_of(model, 'scores = logits.softmax')}; "
                f"{relative(probe_summary)}:{line_of(probe_summary, '\"score_dtype\"')}"
            ),
            "direct_for_historical_cache": False,
            "limitation": "CPU and CUDA autocast policies need not choose the same softmax dtype",
        },
        {
            "stage": "HB TopK",
            "operation": "torch.topk(scores, k=4)",
            "dtype_evidence": "runtime index/weight dtypes not separately logged",
            "evidence_kind": "source",
            "evidence": f"{relative(model)}:{line_of(model, 'topk_weight, topk_idx = torch.topk')}" ,
            "direct_for_historical_cache": False,
            "limitation": "stored IDs and probabilities are cast later",
        },
        {
            "stage": "recorder fidelity",
            "operation": "capture actual topk_idx; recompute probabilities; verify sets",
            "dtype_evidence": "480 verified hook calls per canonical capture",
            "evidence_kind": "source + capture metadata",
            "evidence": (
                f"{relative(recorder)}:{line_of(recorder, 'topk_idx, topk_weight, _aux = output')},"
                f"{line_of(recorder, 'same = (mine.sort')}; capture_summary.json"
            ),
            "direct_for_historical_cache": True,
            "limitation": "verification covers initial calls and TopK sets, not probability bit identity",
        },
        {
            "stage": "host conversion",
            "operation": "actual IDs -> uint8; full probabilities -> float16",
            "dtype_evidence": "uint8 / float16",
            "evidence_kind": "source",
            "evidence": (
                f"{relative(recorder)}:{line_of(recorder, 'rec.hb_expert_ids =')},"
                f"{line_of(recorder, 'rec.hb_router_probs = (')}"
            ),
            "direct_for_historical_cache": True,
            "limitation": "pre-cast full-probability dtype was not written to capture metadata",
        },
        {
            "stage": "Zarr schema",
            "operation": "persist hb_expert_ids / hb_router_probs",
            "dtype_evidence": "uint8 / float16",
            "evidence_kind": "writer source + array metadata",
            "evidence": (
                f"{relative(writer)}:{line_of(writer, '(\"hb_expert_ids\"')},"
                f"{line_of(writer, '(\"hb_router_probs\"')}; "
                f"{relative(example_meta)}:{line_of(example_meta, '\"data_type\"')}"
            ),
            "direct_for_historical_cache": True,
            "limitation": "float32 upcast cannot restore precision absent from this array",
        },
        {
            "stage": "fp32 shadow probe",
            "operation": "autocast-disabled fp32 F.linear + softmax",
            "dtype_evidence": "torch.float32",
            "evidence_kind": "probe source + runtime output",
            "evidence": (
                f"{relative(probe)}:{line_of(probe, 'with torch.autocast(\"cpu\", enabled=False)')},"
                f"{line_of(probe, 'shadow_logits = functional.linear')}"
            ),
            "direct_for_historical_cache": False,
            "limitation": "small separate probe; full caches do not contain shadow logits",
        },
    ]
    return pd.DataFrame(rows)


def sensitivity_metric(
    name: str,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    left_stable: np.ndarray,
    right_stable: np.ndarray,
    left_margin: np.ndarray,
    right_margin: np.ndarray,
    eligible: np.ndarray | None = None,
) -> dict[str, Any]:
    turnover = top4_turnover(left_ids, right_ids)
    stable = left_stable & right_stable
    weight = np.minimum(left_margin, right_margin).astype(np.float64)
    sites = int(np.prod(turnover.shape[1:]))
    flat_turnover = turnover.reshape(len(turnover), sites)
    flat_stable = stable.reshape(len(stable), sites)
    flat_weight = weight.reshape(len(weight), sites)
    keep = np.ones(len(turnover), dtype=bool) if eligible is None else np.asarray(eligible, bool)

    baseline_per_query = flat_turnover.mean(axis=1)
    stable_denominator = flat_stable.sum(axis=1)
    stable_per_query = np.divide(
        (flat_turnover * flat_stable).sum(axis=1),
        stable_denominator,
        out=np.full(len(turnover), np.nan),
        where=stable_denominator > 0,
    )
    weight_denominator = flat_weight.sum(axis=1)
    weighted_per_query = np.divide(
        (flat_turnover * flat_weight).sum(axis=1),
        weight_denominator,
        out=np.full(len(turnover), np.nan),
        where=weight_denominator > 0,
    )
    valid_stable = keep & np.isfinite(stable_per_query)
    valid_weighted = keep & np.isfinite(weighted_per_query)
    return {
        "metric": name,
        "n_query_pairs": int(keep.sum()),
        "n_sites_per_pair": sites,
        "baseline_hard_turnover": float(baseline_per_query[keep].mean()),
        "bf16_stable_only_turnover": float(stable_per_query[valid_stable].mean()),
        "margin_weighted_turnover": float(weighted_per_query[valid_weighted].mean()),
        "bf16_stable_pair_site_fraction": float(flat_stable[keep].mean()),
        "exact_zero_min_margin_fraction": float((flat_weight[keep] == 0).mean()),
        "n_pairs_with_stable_site": int(valid_stable.sum()),
        "n_pairs_with_positive_weight": int(valid_weighted.sum()),
    }


def hard_metric_sensitivity(infos: list[CacheInfo]) -> pd.DataFrame:
    target = next(
        info
        for info in infos
        if info.scope == "primary_canonical" and info.task.startswith("libero_long/")
    )
    store = zarr.open(store=str(target.store_path), mode="r")
    n_rows = min(2048, target.rows)
    probs = np.asarray(store["hb_router_probs"][:n_rows], dtype=np.float32)
    ids = np.asarray(store["hb_expert_ids"][:n_rows], dtype=np.uint8)
    episode = np.asarray(store["episode_id"][:n_rows], dtype=np.int32)
    five = np.partition(probs, -5, axis=-1)[..., -5:]
    five.sort(axis=-1)
    margin = five[..., 1] - five[..., 0]
    bound = half_ulp(five[..., 1], "bf16") + half_ulp(five[..., 0], "bf16")
    stable = margin > bound

    same_episode = episode[1:] == episode[:-1]
    rows = [
        sensitivity_metric(
            "state_adjacent_query_top4_turnover",
            ids[1:, :, 0, 0, :],
            ids[:-1, :, 0, 0, :],
            stable[1:, :, 0, 0],
            stable[:-1, :, 0, 0],
            margin[1:, :, 0, 0],
            margin[:-1, :, 0, 0],
            same_episode,
        ),
        sensitivity_metric(
            "action_adjacent_query_top4_turnover",
            ids[1:, :, :, 1:, :],
            ids[:-1, :, :, 1:, :],
            stable[1:, :, :, 1:],
            stable[:-1, :, :, 1:],
            margin[1:, :, :, 1:],
            margin[:-1, :, :, 1:],
            same_episode,
        ),
        sensitivity_metric(
            "action_adjacent_denoise_top4_turnover",
            ids[:, :, 1:, 1:, :],
            ids[:, :, :-1, 1:, :],
            stable[:, :, 1:, 1:],
            stable[:, :, :-1, 1:],
            margin[:, :, 1:, 1:],
            margin[:, :, :-1, 1:],
        ),
    ]
    span = min(512, n_rows)
    rows.append(
        sensitivity_metric(
            "action_denoise0_vs_denoise9_top4_turnover",
            ids[:span, :, -1, 1:, :],
            ids[:span, :, 0, 1:, :],
            stable[:span, :, -1, 1:],
            stable[:span, :, 0, 1:],
            margin[:span, :, -1, 1:],
            margin[:span, :, 0, 1:],
        )
    )

    expected = {
        "state_adjacent_query_top4_turnover": 0.311127744510978,
        "action_adjacent_query_top4_turnover": 0.6949723989520958,
        "action_adjacent_denoise_top4_turnover": 0.20313110351562502,
        "action_denoise0_vs_denoise9_top4_turnover": 1.0 - 0.301641845703125,
    }
    for row in rows:
        row["legacy_expected"] = expected[row["metric"]]
        row["legacy_abs_error"] = abs(row["baseline_hard_turnover"] - row["legacy_expected"])
        if row["legacy_abs_error"] > 1e-10:
            raise AssertionError(f"failed to reproduce {row['metric']}: {row}")
        row["cache_id"] = target.cache_id
        row["rows_used"] = span if "denoise0_vs" in row["metric"] else n_rows
    return pd.DataFrame(rows)


def plot_summary(role: pd.DataFrame, hard: pd.DataFrame, output: Path) -> None:
    pooled = role[role.cache_id == "POOLED_PRIMARY"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for token_role, colour in (("state", "#1f6f8b"), ("action", "#c74b50")):
        frame = pooled[pooled.token_role == token_role].sort_values("layer")
        axes[0, 0].plot(frame.layer, frame.p4_p5_margin_median, "o-", label=token_role, color=colour)
        axes[0, 1].plot(
            frame.layer,
            frame.stable_bf16_rounding_bound,
            "o-",
            label=token_role,
            color=colour,
        )
        axes[1, 0].plot(
            frame.layer,
            frame.actual_selection_boundary_ambiguous,
            "o-",
            label=token_role,
            color=colour,
        )
    axes[0, 0].set(title="Cached p4-p5 margin", xlabel="HB layer", ylabel="median probability gap")
    axes[0, 1].set(
        title="Hypothetical BF16 rounding-bound stability",
        xlabel="HB layer",
        ylabel="fraction",
    )
    axes[1, 0].set(
        title="Saved p4=p5 boundary ambiguity",
        xlabel="HB layer",
        ylabel="fraction",
    )
    for axis in axes.flat[:3]:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

    selected = hard[hard.metric.str.contains("adjacent_query")]
    x = np.arange(len(selected))
    width = 0.25
    for offset, column, label, colour in (
        (-width, "baseline_hard_turnover", "all sites", "#666666"),
        (0.0, "bf16_stable_only_turnover", "stable only", "#1f6f8b"),
        (width, "margin_weighted_turnover", "margin weighted", "#c74b50"),
    ):
        axes[1, 1].bar(x + offset, selected[column], width, label=label, color=colour)
    axes[1, 1].set(
        title="Legacy hard-ID sensitivity",
        ylabel="Top-4 turnover",
        xticks=x,
        xticklabels=["state query", "action query"],
    )
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def weighted_role_summary(role: pd.DataFrame, token_role: str) -> dict[str, float]:
    frame = role[(role.cache_id == "POOLED_PRIMARY") & (role.token_role == token_role)]
    weights = frame.n_sites.to_numpy(dtype=np.float64)
    fields = [
        "p4_p5_margin",
        "entropy_normalised",
        "gap_exact_zero",
        "stable_fp16_rounding_bound",
        "stable_bf16_rounding_bound",
        "stable_fp32_rounding_bound",
        "actual_vs_saved_fp16_top4_disagree",
        "actual_vs_bf16_recast_top4_disagree",
        "saved_fp16_vs_bf16_recast_top4_disagree",
        "actual_selection_boundary_ambiguous",
        "actual_selection_contradicted_by_saved_probs",
        "bf16_recast_vector_exact",
        "bf16_recast_element_exact_fraction",
    ]
    return {
        field: float(np.average(frame[field].to_numpy(dtype=float), weights=weights))
        for field in fields
    }


def write_report(
    out_dir: Path,
    inventory: pd.DataFrame,
    role: pd.DataFrame,
    hard: pd.DataFrame,
    chain: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    state = summary["primary_pooled"]["state"]
    action = summary["primary_pooled"]["action"]
    action_query = hard[hard.metric == "action_adjacent_query_top4_turnover"].iloc[0]
    state_query = hard[hard.metric == "state_adjacent_query_top4_turnover"].iloc[0]
    token_summary_path = HERE / "analysis/post-error-adaptation-20260828/token_axis_summary.json"
    token_report_path = HERE / "analysis/post-error-adaptation-20260828/token_decomposition.md"
    near_path = HERE / "analysis/near-tie/summary.json"
    router_audit = HERE / "analysis/router-audit/summary.json"
    denoise_report = HERE / "analysis/denoise-stop-sweep/report.md"

    layer_rows = role[(role.cache_id == "POOLED_PRIMARY")].copy()
    pivot_margin = layer_rows.pivot(index="layer", columns="token_role", values="p4_p5_margin_median")
    pivot_stable = layer_rows.pivot(
        index="layer", columns="token_role", values="stable_bf16_rounding_bound"
    )
    layer_table = [
        "| HB layer | state median gap | action median gap | state hyp-BF16-stable | action hyp-BF16-stable |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer in HB_LAYERS:
        layer_table.append(
            f"| {layer} | {pivot_margin.loc[layer, 'state']:.6g} | "
            f"{pivot_margin.loc[layer, 'action']:.6g} | "
            f"{pivot_stable.loc[layer, 'state']:.1%} | "
            f"{pivot_stable.loc[layer, 'action']:.1%} |"
        )

    intervention = inventory[inventory.scope == "routing_intervention"]
    report = f"""# Router 量化可识别性审计

## 结论

**对实际执行过哪些 expert 的事实记录是 Go；从缓存概率唯一重建 action Top-4、或把 hard-ID
变化直接解释成“策略改变”，都是 No-Go。**
缓存中的 `hb_expert_ids` 是 hook 直接取得的实际 Top-4，再转成 `uint8`；它不是从
`float16` 概率重建的。但若对保存概率施加 hypothetical BF16 output rounding，action router
的 p4-p5 边界有相当一部分不满足保守稳定界；所以 hard-ID 的变化不能自动等价为计算盆地
或策略改变。

六个互不重复的 canonical cache 共 {summary['primary_control_rows']:,} 个 control query、
{summary['primary_routing_sites']:,} 个 HB routing site。action 的平均 p4-p5 gap 为
`{action['p4_p5_margin']:.6g}`，对保存概率施加 hypothetical BF16 rounding bound 的 stable fraction 为
**{action['stable_bf16_rounding_bound']:.1%}**；state 对应为
`{state['p4_p5_margin']:.6g}` 和 **{state['stable_bf16_rounding_bound']:.1%}**。
actual ID 与保存 fp16 概率的确定性 Top-4 集合在 canonical 数据上不一致率为
action **{action['actual_vs_saved_fp16_top4_disagree']:.1%}**、state
**{state['actual_vs_saved_fp16_top4_disagree']:.1%}**；但这不消除 tie 时的非唯一性。
缓存概率有 action **{action['gap_exact_zero']:.1%}**、state **{state['gap_exact_zero']:.1%}**
的 p4-p5 exact tie；所以即使 actual ID 已保存，仅凭概率也无法在这些 site 唯一确定 Top-4。
缓存值重新 cast 到 BF16 后，逐元素完全相等的比例只有 action
**{action['bf16_recast_element_exact_fraction']:.1%}**、state
**{state['bf16_recast_element_exact_fraction']:.1%}**，完整 32-way 向量相等率两者均为
**{action['bf16_recast_vector_exact']:.1%}**。因此现有概率不是纯 BF16 网格值的无损副本；
历史 GPU softmax dtype 与 fp16 落盘前损失分别占多少，当前缓存无法分解。

## 旧 hard-ID 结论复核

旧报告的 action 相邻 query Top-4 更替率 `69.5%` 被逐位复现为
`{action_query.baseline_hard_turnover:.9f}`（绝对误差
`{action_query.legacy_abs_error:.2g}`）。只保留两端都通过 BF16 舍入界的 site 后为
`{action_query.bf16_stable_only_turnover:.3f}`，覆盖原 pair-site 的
`{action_query.bf16_stable_pair_site_fraction:.1%}`；用两端较小 p4-p5 margin 加权后为
`{action_query.margin_weighted_turnover:.3f}`。这两个敏感版本仍然很高，因此近似平局
**没有解释掉大部分 69.5% turnover**；它说明 ID 确实频繁换，但 hard turnover 自身仍不说明
软分布变化有多大。state 的三项分别是
`{state_query.baseline_hard_turnover:.3f}`、`{state_query.bf16_stable_only_turnover:.3f}`、
`{state_query.margin_weighted_turnover:.3f}`。

这里的 BF16 stable-only 是**对保存概率施加 hypothetical BF16 output rounding** 的保守
诊断，不是 deployed 稳定率，也不是 fp32 真值：条件为
`p4-p5 > halfULP(p4)+halfULP(p5)`。它只排除“单次输出舍入本身足以翻边界”的点，
不覆盖 BF16 `F.linear` 累积误差。

## 分层结果

{chr(10).join(layer_table)}

完整 task/layer/token 明细见 `strata.csv`，role 汇总见 `role_summary.csv`。
margin、tie 和 cast 都直接作用在保存的 fp16 概率（读取后仅无损 upcast 到 fp32），不先
renormalize；只有熵为了消除逐元素落盘舍入造成的微小 sum error 才重新归一化。

## 数值链路：证据与缺口

1. upstream policy 在 `{chain.iloc[0].evidence}` 的 CUDA 推理外层请求 BF16 autocast；HB gate
   源码依次执行 `F.linear -> softmax -> torch.topk`，中间没有显式 dtype cast。autocast 上下文
   本身不等于每个算子输出都是 BF16。
2. 独立 CPU-autocast runtime probe 记录 `scores.dtype=torch.bfloat16`，并在关闭 autocast 后
   计算 fp32 shadow（`{relative(near_path)}:2`；探针实现见
   `{relative(HERE / 'probe_near_tie.py')}:{line_of(HERE / 'probe_near_tie.py', 'with torch.autocast(\"cpu\", enabled=False)')}`）。
   这是 CPU 运行证据，但不是历史 CUDA capture 的 dtype 元数据；不同 device 的 autocast
   operator policy 不能在无记录时视为相同。
3. recorder 从 gate 输出直接拿 `topk_idx`，前若干步用重算概率验证 Top-4 集合；canonical
   capture 都记录 480 次 verified call、0 failures。随后 IDs 显式转 `uint8`，full probs 显式
   转 `float16`。全部 10 个现有 full-prob cache 的 Zarr 元数据也分别验证为这两个 dtype。
4. **历史 capture 没有单独记录 logits dtype、softmax 输入/输出 dtype、TopK weight dtype、
   TopK index runtime dtype、或落盘前 full-prob dtype。** `runtime_numeric_chain.csv` 对每一段标明
   direct / mirror / unlogged，报告不靠猜测补齐。

## 三个旧数字的来源核对

- 相关的 action top-1 不是精确 `0.0369`，而是
  `{json.loads(token_summary_path.read_text())['structural']['action_p1']:.9f}`，旧报告四舍五入为
  `0.0371`（`{relative(token_report_path)}:{line_of(token_report_path, '| top-1 概率')}`）。仓库里
  精确文本 `0.0369` 出现在另一个 denoise-stop 指标表
  (`{relative(denoise_report)}:{line_of(denoise_report, '0.0369')}`)，不能串成同一证据链。
- `0.9988` 的精确近邻来自 router audit 的 action layer0
  `H_token={json.loads(router_audit.read_text())['t00__open_the_middle_drawer_of_the_cabinet']['layers']['action_layer0']['H_token']:.9f}`
  (`{relative(router_audit)}:{line_of(router_audit, '\"H_token\"')}`)；post-error 汇总的全 action
  normalized entropy 是 `{json.loads(token_summary_path.read_text())['structural']['action_entropy_normalised']:.9f}`。
- `69.5%` 来自前 2048 个 long-task query 的 action-token 相邻 query hard Top-4 turnover
  (`{relative(token_report_path)}:{line_of(token_report_path, '69.5%')}`)，本审计按原公式精确复现。

## 缓存范围

发现 {len(inventory)} 个 full-prob cache：{sum(inventory.scope == 'primary_canonical')} 个 canonical
主样本、{sum(inventory.scope == 'duplicate_baseline')} 个重复 baseline、
{len(intervention)} 个 pin intervention。所有 cache 都逐个计算并写入明细；总体数字只合并
canonical，避免把重复 baseline 和人为替换专家的 intervention 混入。pin intervention 的
actual ID 可以有意与未干预 router 概率冲突，因此不用于精度结论。

## 必须新采 fp32 logits 才能回答

- 全量历史 site 的 deployed Top-4 与**同 hidden/weight 的 fp32 shadow Top-4**究竟差多少；
- BF16 `F.linear` 累积误差与仅对输出概率做 cast 的误差各占多少；
- softmax 前 logit p4-p5 margin，以及概率近均匀是否掩盖了 logit 尺度；
- stable-only hard route-return AUC 是否仍为 0.733。现有缓存能严格复核 overlap/turnover，
  但没有旧 AUC 所需的逐事件 margin 和 fp32 counterfactual；
- GPU kernel 在 exact tie 上的跨硬件 tie-breaking 可重复性。

新 capture 至少要同时保存：实际 `topk_idx`、deployed logits/scores 及各自 runtime dtype、
关闭 autocast 后同 hidden/weight 的 fp32 shadow logits/scores，并在 cast 前保存。仅把现有
`float16` Zarr 数组 `.astype(float32)` 不会回答这些问题。

![Quantization audit](router_quantization_identifiability.png)
"""
    (out_dir / "report.md").write_text(report)


def self_test() -> None:
    values = np.asarray([1.0, 0.1, 0.03125, 0.037], dtype=np.float32)
    rounded = bf16_round(values)
    assert np.all((rounded.view(np.uint32) & np.uint32(0xFFFF)) == 0)
    assert np.array_equal(rounded, bf16_round(rounded))
    assert np.all(half_ulp(values, "bf16") > 0)

    scores = np.asarray(
        [[0.30, 0.20, 0.15, 0.10, 0.09, 0.08, 0.05, 0.03]], dtype=np.float32
    )
    ids = deterministic_top4(scores)
    assert np.array_equal(ids, np.asarray([[0, 1, 2, 3]], np.uint8))
    assert not set_disagreement(ids, ids).any()
    assert float(top4_turnover(ids, ids)[0]) == 0.0
    print("self-test: PASS")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.block_rows <= 0 or args.sample_modulus <= 0:
        raise ValueError("block rows and sample modulus must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    infos = discover_caches(args.cache_root)
    inventory = inventory_frame(infos)
    inventory.to_csv(args.out_dir / "cache_inventory.csv", index=False)
    chain = numeric_chain_frame(infos)
    chain.to_csv(args.out_dir / "runtime_numeric_chain.csv", index=False)

    if args.reuse_results:
        token_frame = pd.read_csv(args.out_dir / "strata.csv")
        role_frame = pd.read_csv(args.out_dir / "role_summary.csv")
        hard = pd.read_csv(args.out_dir / "hard_metric_sensitivity.csv")
    else:
        accumulators: dict[str, StratumAccumulator] = {}
        pooled = StratumAccumulator(sample_modulus=args.sample_modulus)
        for info in infos:
            print(f"audit {info.cache_id}: {info.rows} rows", flush=True)
            accumulator = audit_cache(info, args.block_rows, args.sample_modulus)
            accumulators[info.cache_id] = accumulator
            if info.scope == "primary_canonical":
                pooled.merge(accumulator)

        token_rows = []
        role_rows = []
        for info in infos:
            accumulator = accumulators[info.cache_id]
            for layer in range(N_LAYERS):
                for token in range(N_TOKENS):
                    token_rows.append(accumulator.row(info.cache_id, info.scope, layer, [token]))
                role_rows.append(accumulator.row(info.cache_id, info.scope, layer, [0]))
                role_rows.append(
                    accumulator.row(info.cache_id, info.scope, layer, list(range(1, N_TOKENS)))
                )
        for layer in range(N_LAYERS):
            for token in range(N_TOKENS):
                token_rows.append(pooled.row("POOLED_PRIMARY", "primary_pooled", layer, [token]))
            role_rows.append(pooled.row("POOLED_PRIMARY", "primary_pooled", layer, [0]))
            role_rows.append(
                pooled.row("POOLED_PRIMARY", "primary_pooled", layer, list(range(1, N_TOKENS)))
            )
        token_frame = pd.DataFrame(token_rows)
        role_frame = pd.DataFrame(role_rows)
        token_frame.to_csv(args.out_dir / "strata.csv", index=False)
        role_frame.to_csv(args.out_dir / "role_summary.csv", index=False)

        hard = hard_metric_sensitivity(infos)
        hard.to_csv(args.out_dir / "hard_metric_sensitivity.csv", index=False)
    plot_summary(role_frame, hard, args.out_dir / "router_quantization_identifiability.png")

    primary_inventory = inventory[inventory.scope == "primary_canonical"]
    summary = {
        "verdict": {
            "actual_executed_expert_ids": "GO: authoritative stored uint8 IDs",
            "saved_fp16_deterministic_top4_matches_actual": "GO on canonical caches: 0% set disagreement",
            "saved_probability_unique_action_top4": "NO-GO: 28.6% p4=p5 boundary ambiguity",
            "hard_action_id_as_strategy_change": (
                "NO-GO semantically; stable/margin sensitivity remains high but hard IDs omit soft magnitude"
            ),
            "cached_probabilities_as_deployed_bf16_values": (
                "NO-GO: historical CUDA score dtype unlogged and saved values are not on a BF16 grid"
            ),
            "historical_fp32_counterfactual": "UNANSWERABLE without fresh fp32 logits",
        },
        "cache_counts": inventory.scope.value_counts().to_dict(),
        "primary_control_rows": int(primary_inventory.control_rows.sum()),
        "primary_routing_sites": int(primary_inventory.routing_sites.sum()),
        "primary_pooled": {
            "state": weighted_role_summary(role_frame, "state"),
            "action": weighted_role_summary(role_frame, "action"),
        },
        "legacy_hard_metric_sensitivity": hard.to_dict("records"),
        "runtime_probe": json.loads((HERE / "analysis/near-tie/summary.json").read_text()),
        "definitions": {
            "stable_rounding_bound": "p4-p5 > halfULP(p4)+halfULP(p5)",
            "bf16_stability_scope": (
                "hypothetical output rounding applied to saved fp16 probabilities; not deployed stability"
            ),
            "fp32_upcast": "upcast of saved fp16 values only; not historical fp32 routing",
            "top4_disagreement": "set disagreement; expert order ignored",
            "score_preprocessing": (
                "saved fp16 probabilities losslessly upcast to fp32; no renormalisation before margin/cast"
            ),
        },
        "requires_fresh_capture": [
            "historical deployed logits dtype per call",
            "pre-softmax fp32 shadow logits on all sites",
            "BF16 F.linear error versus fp32 F.linear",
            "full-cache deployed versus fp32-shadow TopK disagreement",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(plain(summary), indent=2))
    write_report(args.out_dir, inventory, role_frame, hard, chain, summary)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
