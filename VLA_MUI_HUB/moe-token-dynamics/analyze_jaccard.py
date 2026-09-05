#!/usr/bin/env python3
"""Compare aligned HB routes with cosine and Jaccard distances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr

import analyze as core


HERE = Path(__file__).resolve().parent
METRICS = (
    "top4_ja",
    "top4_cos",
    "soft_wj",
    "soft_cos",
    "selected_wj",
    "selected_cos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=core.RUN)
    parser.add_argument("--discovery-state", type=int, default=0)
    parser.add_argument("--confirmation-state", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    intersection = np.logical_and(a, b).sum(axis=-1)
    union = np.logical_or(a, b).sum(axis=-1)
    return 1.0 - intersection / np.maximum(union, 1)


def weighted_jaccard_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    numerator = np.minimum(a, b).sum(axis=-1)
    denominator = np.maximum(a, b).sum(axis=-1)
    return 1.0 - numerator / np.maximum(denominator, 1e-12)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    numerator = (a * b).sum(axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.maximum(1.0 - numerator / np.maximum(denominator, 1e-12), 0.0)


def selected_weight_vectors(ids: np.ndarray, raw_weight: np.ndarray) -> np.ndarray:
    weight = np.maximum(np.asarray(raw_weight, dtype=np.float32), 0.0)
    weight /= np.maximum(weight.sum(axis=-1, keepdims=True), 1e-12)
    dense = np.zeros(ids.shape[:-1] + (core.N_EXPERTS,), dtype=np.float32)
    np.put_along_axis(dense, ids.astype(np.int64), weight, axis=-1)
    if np.max(np.abs(dense.sum(axis=-1) - 1.0)) > 1e-5:
        raise ValueError("selected Top-4 weights do not sum to one")
    return dense


def register(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    key: str,
    value: float,
    **fields: Any,
) -> None:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite value for {key}")
    output[key] = value
    previous = metadata.setdefault(key, fields)
    if previous != fields:
        raise ValueError(f"metadata mismatch for {key}")


def add_cross_chunk(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    metric: str,
    values: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    for horizon in core.HORIZONS:
        start = max(0, horizon - core.TRAILING_WINDOW + 1)
        lag_scores = []
        for lag in core.LAGS:
            measured = distance(
                values[start + lag : horizon + 1],
                values[start : horizon + 1 - lag],
            )
            reduced = measured.mean(axis=(0, 1, 2))
            if reduced.shape != (11,):
                raise ValueError(f"bad aligned-token distance {reduced.shape}")
            lag_scores.append(reduced)
            for token, value in enumerate(reduced):
                key = f"{metric}|cross|h{horizon:02d}|lag{lag}|T{token}"
                register(
                    output,
                    metadata,
                    key,
                    value,
                    metric=metric,
                    mode="cross",
                    horizon=horizon,
                    statistic=f"lag{lag}",
                    comparison=f"T{token}",
                )
            key = f"{metric}|cross|h{horizon:02d}|lag{lag}|action_all"
            register(
                output,
                metadata,
                key,
                reduced[1:11].mean(),
                metric=metric,
                mode="cross",
                horizon=horizon,
                statistic=f"lag{lag}",
                comparison="action_all",
            )

        lag_scores = np.stack(lag_scores)
        recurrences = {
            2: 0.5 * (lag_scores[0] + lag_scores[2]) - lag_scores[1],
            3: 0.5 * (lag_scores[1] + lag_scores[3]) - lag_scores[2],
            4: 0.5 * (lag_scores[2] + lag_scores[4]) - lag_scores[3],
            5: lag_scores[3] - lag_scores[4],
        }
        for period, reduced in recurrences.items():
            for token, value in enumerate(reduced):
                key = f"{metric}|cross|h{horizon:02d}|period{period}|T{token}"
                register(
                    output,
                    metadata,
                    key,
                    value,
                    metric=metric,
                    mode="cross",
                    horizon=horizon,
                    statistic=f"period{period}",
                    comparison=f"T{token}",
                )
            key = f"{metric}|cross|h{horizon:02d}|period{period}|action_all"
            register(
                output,
                metadata,
                key,
                reduced[1:11].mean(),
                metric=metric,
                mode="cross",
                horizon=horizon,
                statistic=f"period{period}",
                comparison="action_all",
            )


def add_within_chunk(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    metric: str,
    values: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    for horizon in core.HORIZONS:
        start = max(0, horizon - core.TRAILING_WINDOW + 1)
        segment = values[start : horizon + 1]
        action_pair_values = []
        adjacent_values = []
        for first in range(1, 11):
            for second in range(first + 1, 11):
                value = float(
                    distance(segment[..., first, :], segment[..., second, :]).mean()
                )
                action_pair_values.append(value)
                if second == first + 1:
                    adjacent_values.append(value)
                key = (
                    f"{metric}|within|h{horizon:02d}|level|T{first}_T{second}"
                )
                register(
                    output,
                    metadata,
                    key,
                    value,
                    metric=metric,
                    mode="within",
                    horizon=horizon,
                    statistic="level",
                    comparison=f"T{first}_T{second}",
                )
        state_values = []
        for token in range(1, 11):
            value = float(
                distance(segment[..., 0, :], segment[..., token, :]).mean()
            )
            state_values.append(value)
            key = f"{metric}|within|h{horizon:02d}|level|T0_T{token}"
            register(
                output,
                metadata,
                key,
                value,
                metric=metric,
                mode="within",
                horizon=horizon,
                statistic="level",
                comparison=f"T0_T{token}",
            )
        aggregates = {
            "all_action_pairs": float(np.mean(action_pair_values)),
            "adjacent_action_pairs": float(np.mean(adjacent_values)),
            "state_action_pairs": float(np.mean(state_values)),
        }
        for comparison, value in aggregates.items():
            key = f"{metric}|within|h{horizon:02d}|level|{comparison}"
            register(
                output,
                metadata,
                key,
                value,
                metric=metric,
                mode="within",
                horizon=horizon,
                statistic="level",
                comparison=comparison,
            )


def extract(
    run: Path,
    episodes: list[core.Episode],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str], dict[str, dict[str, Any]]]:
    store = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    prefix = max(core.HORIZONS) + 1
    metadata: dict[str, dict[str, Any]] = {}
    rows_by_state: dict[int, list[dict[str, float]]] = {}
    labels_by_state: dict[int, np.ndarray] = {}
    for state in sorted({episode.state for episode in episodes}):
        group = sorted(
            [episode for episode in episodes if episode.state == state],
            key=lambda episode: episode.noise_seed,
        )
        rows = []
        for position, episode in enumerate(group, start=1):
            start = episode.offset
            stop = start + prefix
            soft = core.normalize_probability(
                np.asarray(store["hb_router_probs"][start:stop], dtype=np.float32)
            )
            ids = np.asarray(store["hb_expert_ids"][start:stop], dtype=np.uint8)
            selected = selected_weight_vectors(
                ids,
                np.asarray(store["hb_selected_prob"][start:stop], dtype=np.float32),
            )
            hard = core.top4_probabilities(ids) > 0
            representations = (
                ("top4_ja", hard, jaccard_distance),
                ("top4_cos", hard, cosine_distance),
                ("soft_wj", soft, weighted_jaccard_distance),
                ("soft_cos", soft, cosine_distance),
                ("selected_wj", selected, weighted_jaccard_distance),
                ("selected_cos", selected, cosine_distance),
            )
            features: dict[str, float] = {}
            for metric, values, distance in representations:
                add_cross_chunk(features, metadata, metric, values, distance)
                add_within_chunk(features, metadata, metric, values, distance)
            rows.append(features)
            print(f"Route distances state {state}: {position:02d}/32", flush=True)
        rows_by_state[state] = rows
        labels_by_state[state] = np.asarray(
            [episode.failure for episode in group], dtype=np.int8
        )

    keys = sorted(metadata)
    matrices = {
        state: np.asarray([[row[key] for key in keys] for row in rows], dtype=np.float64)
        for state, rows in rows_by_state.items()
    }
    return matrices, labels_by_state, keys, metadata


def map_scan(local: dict[str, np.ndarray], indexes: np.ndarray, width: int) -> dict[str, np.ndarray]:
    output = {
        name: np.full(width, np.nan, dtype=np.float64)
        for name in ("auc", "p_unadjusted", "p_fwer")
    }
    for name, values in local.items():
        output[name][indexes] = values
    return output


def feature_result(
    index: int,
    keys: list[str],
    metadata: dict[str, dict[str, Any]],
    scan: dict[str, np.ndarray],
    discovery: np.ndarray,
    confirmation: np.ndarray,
    discovery_labels: np.ndarray,
    confirmation_labels: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    auc = float(scan["auc"][index])
    direction = 1.0 if auc >= 0.5 else -1.0
    selected = discovery[:, index]
    confirmed = confirmation[:, index]
    return {
        "key": keys[index],
        **metadata[keys[index]],
        "discovery_auc_failure": auc,
        "discovery_direction": "larger_in_failure" if direction > 0 else "smaller_in_failure",
        "discovery_p_unadjusted": float(scan["p_unadjusted"][index]),
        "discovery_p_fwer": float(scan["p_fwer"][index]),
        "discovery_means": {
            "success": float(selected[discovery_labels == 0].mean()),
            "timeout_failure": float(selected[discovery_labels == 1].mean()),
        },
        "confirmation_means": {
            "success": float(confirmed[confirmation_labels == 0].mean()),
            "timeout_failure": float(confirmed[confirmation_labels == 1].mean()),
        },
        "confirmation": core.confirmation_test(
            confirmed,
            confirmation_labels,
            direction,
            permutations,
            rng,
        ),
    }


def render_report(summary: dict[str, Any]) -> str:
    metric_tests = len(summary["metric_mode_horizon_leads"])
    fixed_tests = len(summary["fixed_aggregate_tests"])
    lines = [
        "# HB-MoE route distance comparisons",
        "",
        "## Definitions",
        "",
        "- `cross`: compare the same token slot between control chunks t and t-k.",
        "- `within`: compare two token slots inside one control chunk; no control-step distance is used.",
        "- `top4_ja`: set Jaccard distance over authoritative Top-4 IDs.",
        "- `top4_cos`: cosine distance over the authoritative Top-4 indicator.",
        "- `soft_wj`: weighted-Jaccard distance over all 32 router probabilities.",
        "- `soft_cos`: cosine distance over all 32 router probabilities.",
        "- `selected_wj`: weighted-Jaccard distance over normalized actual Top-4 combine weights.",
        "- `selected_cos`: cosine distance over normalized actual Top-4 combine weights.",
        "",
        "## Fixed aggregate tests (no token or lag selection)",
        "",
        "- `cross`: lag 1, average the aligned T1-T10 distances.",
        "- `within`: average all 45 action-token pair distances in the same chunk.",
        "",
        f"| metric | mode | horizon | discovery AUC | discovery maxT p | confirmation oriented AUC | p | Bonferroni over {fixed_tests} |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["fixed_aggregate_tests"]:
        lines.append(
            "| %s | %s | t%d | %.4f | %.6f | %.4f | %.6f | %.6f |"
            % (
                item["metric"],
                item["mode"],
                item["horizon"],
                item["discovery_auc_failure"],
                item["discovery_p_fwer"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_fixed"],
            )
        )
    lines.extend(
        [
        "",
        "## Mode by horizon",
        "",
        "| mode | horizon | selected feature | discovery maxT p | confirmation oriented AUC | p | Bonferroni over 6 |",
        "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["mode_horizon_leads"]:
        lines.append(
            "| %s | t%d | `%s` | %.6f | %.4f | %.6f | %.6f |"
            % (
                item["mode"],
                item["horizon"],
                item["key"],
                item["discovery_p_fwer"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_6"],
            )
        )
    lines.extend(
        [
            "",
            "## Metric, mode, and horizon",
            "",
            f"| metric | mode | horizon | selected comparison | discovery AUC | confirmation oriented AUC | p | Bonferroni over {metric_tests} |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["metric_mode_horizon_leads"]:
        lines.append(
            "| %s | %s | t%d | `%s` | %.4f | %.4f | %.6f | %.6f |"
            % (
                item["metric"],
                item["mode"],
                item["horizon"],
                item["comparison"],
                item["discovery_auc_failure"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_metrics"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    states = (args.discovery_state, args.confirmation_state)
    episodes, _ = core.load_episodes(run, states)
    matrices, labels, keys, metadata = extract(run, episodes)
    discovery = matrices[args.discovery_state]
    confirmation = matrices[args.confirmation_state]
    discovery_labels = labels[args.discovery_state]
    confirmation_labels = labels[args.confirmation_state]
    variable = np.ptp(discovery, axis=0) > 1e-12
    route_indexes = np.flatnonzero(variable)
    rng = np.random.default_rng(args.seed)

    global_local = core.permutation_scan(
        discovery[:, route_indexes], discovery_labels, args.permutations, rng
    )
    global_scan = map_scan(global_local, route_indexes, len(keys))

    fixed_indexes = []
    for mode in ("cross", "within"):
        for metric in METRICS:
            for horizon in core.HORIZONS:
                key = (
                    f"{metric}|cross|h{horizon:02d}|lag1|action_all"
                    if mode == "cross"
                    else f"{metric}|within|h{horizon:02d}|level|all_action_pairs"
                )
                fixed_indexes.append(keys.index(key))
    fixed_indexes = np.asarray(fixed_indexes, dtype=np.int64)
    fixed_local = core.permutation_scan(
        discovery[:, fixed_indexes], discovery_labels, args.permutations, rng
    )
    fixed_scan = map_scan(fixed_local, fixed_indexes, len(keys))
    fixed_aggregate_tests = []
    for index in fixed_indexes:
        item = feature_result(
            int(index),
            keys,
            metadata,
            fixed_scan,
            discovery,
            confirmation,
            discovery_labels,
            confirmation_labels,
            args.permutations,
            rng,
        )
        item["confirmation_p_bonferroni_fixed"] = min(
            1.0,
            len(fixed_indexes)
            * item["confirmation"]["permutation_p_one_sided"],
        )
        fixed_aggregate_tests.append(item)

    mode_horizon_leads = []
    for mode in ("cross", "within"):
        for horizon in core.HORIZONS:
            indexes = np.asarray(
                [
                    index
                    for index in route_indexes
                    if metadata[keys[index]]["mode"] == mode
                    and metadata[keys[index]]["horizon"] == horizon
                ],
                dtype=np.int64,
            )
            local = core.permutation_scan(
                discovery[:, indexes], discovery_labels, args.permutations, rng
            )
            scan = map_scan(local, indexes, len(keys))
            selected = int(indexes[np.argmax(np.abs(scan["auc"][indexes] - 0.5))])
            item = feature_result(
                selected,
                keys,
                metadata,
                scan,
                discovery,
                confirmation,
                discovery_labels,
                confirmation_labels,
                args.permutations,
                rng,
            )
            item["discovery_p_all_features_fwer"] = float(
                global_scan["p_fwer"][selected]
            )
            item["confirmation_p_bonferroni_6"] = min(
                1.0, 6.0 * item["confirmation"]["permutation_p_one_sided"]
            )
            mode_horizon_leads.append(item)

    metric_mode_horizon_leads = []
    metric_tests = len(METRICS) * 2 * len(core.HORIZONS)
    for metric in METRICS:
        for mode in ("cross", "within"):
            for horizon in core.HORIZONS:
                indexes = np.asarray(
                    [
                        index
                        for index in route_indexes
                        if metadata[keys[index]]["metric"] == metric
                        and metadata[keys[index]]["mode"] == mode
                        and metadata[keys[index]]["horizon"] == horizon
                    ],
                    dtype=np.int64,
                )
                local = core.permutation_scan(
                    discovery[:, indexes], discovery_labels, args.permutations, rng
                )
                scan = map_scan(local, indexes, len(keys))
                selected = int(
                    indexes[np.argmax(np.abs(scan["auc"][indexes] - 0.5))]
                )
                item = feature_result(
                    selected,
                    keys,
                    metadata,
                    scan,
                    discovery,
                    confirmation,
                    discovery_labels,
                    confirmation_labels,
                    args.permutations,
                    rng,
                )
                item["confirmation_p_bonferroni_metrics"] = min(
                    1.0,
                    metric_tests
                    * item["confirmation"]["permutation_p_one_sided"],
                )
                metric_mode_horizon_leads.append(item)

    summary = {
        "experiment": "himoe_route_distance_comparison_v2",
        "integrity": {
            "run": str(run),
            "features": len(keys),
            "variable_features": int(variable.sum()),
            "episodes_per_state": 32,
            "discovery_state": args.discovery_state,
            "confirmation_state": args.confirmation_state,
        },
        "protocol": {
            "horizons": list(core.HORIZONS),
            "lags": list(core.LAGS),
            "trailing_window": core.TRAILING_WINDOW,
            "sample_unit": "rollout",
            "permutation_draws": args.permutations,
            "metrics": list(METRICS),
        },
        "fixed_aggregate_tests": fixed_aggregate_tests,
        "mode_horizon_leads": mode_horizon_leads,
        "metric_mode_horizon_leads": metric_mode_horizon_leads,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(core.finite_json(summary), indent=2, ensure_ascii=True) + "\n"
    report = render_report(summary)
    arrays = {
        "feature_keys": np.asarray(keys),
        "discovery_scores": discovery,
        "confirmation_scores": confirmation,
        "discovery_failure": discovery_labels,
        "confirmation_failure": confirmation_labels,
    }
    for stem in ("route_distance", "jaccard"):
        (output / f"{stem}_summary.json").write_text(serialized)
        (output / f"{stem}_report.md").write_text(report)
        np.savez_compressed(output / f"{stem}_scores.npz", **arrays)
    print(f"wrote {output / 'route_distance_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
