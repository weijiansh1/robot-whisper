#!/usr/bin/env python3
"""Ablate the number of terminal aligned-token distance pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr
from scipy.stats import rankdata

import analyze as core
import analyze_jaccard as route


HERE = Path(__file__).resolve().parent
WINDOWS = tuple(range(1, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=core.RUN)
    parser.add_argument("--discovery-state", type=int, default=0)
    parser.add_argument("--confirmation-state", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


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


def add_window_features(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    metric: str,
    values: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    for horizon in core.HORIZONS:
        for lag in core.LAGS:
            for window in WINDOWS:
                current_start = horizon - window + 1
                previous_start = current_start - lag
                if previous_start < 0:
                    continue
                current = values[current_start : horizon + 1]
                previous = values[previous_start : horizon + 1 - lag]
                if current.shape != previous.shape or current.shape[0] != window:
                    raise ValueError(
                        f"bad window h={horizon} lag={lag} W={window}: "
                        f"{current.shape} vs {previous.shape}"
                    )
                measured = distance(current, previous)
                reduced = measured.mean(axis=(0, 1, 2))
                if reduced.shape != (11,):
                    raise ValueError(f"bad aligned-token distance {reduced.shape}")
                for token, value in enumerate(reduced):
                    key = (
                        f"{metric}|h{horizon:02d}|W{window}|lag{lag}|T{token}"
                    )
                    register(
                        output,
                        metadata,
                        key,
                        value,
                        metric=metric,
                        horizon=horizon,
                        window=window,
                        lag=lag,
                        comparison=f"T{token}",
                    )
                key = (
                    f"{metric}|h{horizon:02d}|W{window}|lag{lag}|action_all"
                )
                register(
                    output,
                    metadata,
                    key,
                    reduced[1:11].mean(),
                    metric=metric,
                    horizon=horizon,
                    window=window,
                    lag=lag,
                    comparison="action_all",
                )


def extract(
    run: Path,
    episodes: list[core.Episode],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[str],
    dict[str, dict[str, Any]],
]:
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
            selected = route.selected_weight_vectors(
                ids,
                np.asarray(store["hb_selected_prob"][start:stop], dtype=np.float32),
            )
            hard = core.top4_probabilities(ids) > 0
            representations = (
                ("top4_ja", hard, route.jaccard_distance),
                ("top4_cos", hard, route.cosine_distance),
                ("soft_wj", soft, route.weighted_jaccard_distance),
                ("soft_cos", soft, route.cosine_distance),
                ("selected_wj", selected, route.weighted_jaccard_distance),
                ("selected_cos", selected, route.cosine_distance),
            )
            features: dict[str, float] = {}
            for metric, values, distance in representations:
                add_window_features(features, metadata, metric, values, distance)
            rows.append(features)
            print(f"Window state {state}: {position:02d}/32", flush=True)
        rows_by_state[state] = rows
        labels_by_state[state] = np.asarray(
            [episode.failure for episode in group], dtype=np.int8
        )

    keys = sorted(metadata)
    matrices = {
        state: np.asarray(
            [[row[key] for key in keys] for row in rows], dtype=np.float64
        )
        for state, rows in rows_by_state.items()
    }
    return matrices, labels_by_state, keys, metadata


def confirmation_many(
    matrix: np.ndarray,
    labels: np.ndarray,
    directions: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks = rankdata(matrix, axis=0, method="average")
    raw_auc = core.auc_from_ranks(ranks, labels)
    oriented_auc = 0.5 + directions * (raw_auc - 0.5)
    exceed = np.zeros(matrix.shape[1], dtype=np.int64)
    for start in range(0, permutations, 500):
        count = min(500, permutations - start)
        permuted = np.stack([rng.permutation(labels) for _ in range(count)])
        positive = int(labels.sum())
        negative = len(labels) - positive
        null = (
            permuted @ ranks - positive * (positive + 1) / 2.0
        ) / (positive * negative)
        null_oriented = 0.5 + directions[None, :] * (null - 0.5)
        exceed += (
            null_oriented >= oriented_auc[None, :] - 1e-15
        ).sum(axis=0)
    p_value = (exceed + 1.0) / (permutations + 1.0)
    return raw_auc, oriented_auc, p_value


def fixed_result(
    index: int,
    keys: list[str],
    metadata: dict[str, dict[str, Any]],
    discovery: np.ndarray,
    confirmation: np.ndarray,
    discovery_labels: np.ndarray,
    confirmation_labels: np.ndarray,
    discovery_scan: dict[str, np.ndarray],
    confirmation_raw_auc: float,
    confirmation_oriented_auc: float,
    confirmation_p: float,
    fixed_family_size: int,
) -> dict[str, Any]:
    discovery_auc = float(discovery_scan["auc"][index])
    direction = 1.0 if discovery_auc >= 0.5 else -1.0
    return {
        "key": keys[index],
        **metadata[keys[index]],
        "discovery_auc_failure": discovery_auc,
        "discovery_direction": (
            "larger_in_failure" if direction > 0 else "smaller_in_failure"
        ),
        "discovery_p_unadjusted": float(
            discovery_scan["p_unadjusted"][index]
        ),
        "discovery_p_fwer_fixed_family": float(
            discovery_scan["p_fwer"][index]
        ),
        "discovery_means": {
            "success": float(discovery[discovery_labels == 0, index].mean()),
            "timeout_failure": float(
                discovery[discovery_labels == 1, index].mean()
            ),
        },
        "confirmation_means": {
            "success": float(confirmation[confirmation_labels == 0, index].mean()),
            "timeout_failure": float(
                confirmation[confirmation_labels == 1, index].mean()
            ),
        },
        "confirmation": {
            "auc_failure_raw": float(confirmation_raw_auc),
            "auc_discovery_oriented": float(confirmation_oriented_auc),
            "permutation_p_one_sided": float(confirmation_p),
            "p_bonferroni_all_fixed": min(
                1.0, fixed_family_size * float(confirmation_p)
            ),
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    by_key = {item["key"]: item for item in summary["fixed_lag1_action"]}
    lines = [
        "# Terminal window ablation",
        "",
        "`W` is the number of terminal distance pairs, not the number of raw control rows.",
        "For lag 1, W7 exactly matches the prior eight-control-row window.",
        "Every fixed curve uses aligned T1-T10 routes and averages over token, layer, and flow.",
        "",
        "## t34 fixed lag-1 action-token curves",
        "",
        "Each cell is confirmation oriented AUC (one-sided permutation p). Direction is fixed in state 0 and evaluated in state 42.",
        "",
        "| metric | W1 | W2 | W3 | W4 | W5 | W6 | W7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in route.METRICS:
        cells = []
        for window in WINDOWS:
            key = f"{metric}|h34|W{window}|lag1|action_all"
            item = by_key[key]
            cells.append(
                "%.3f (%.4f)"
                % (
                    item["confirmation"]["auc_discovery_oriented"],
                    item["confirmation"]["permutation_p_one_sided"],
                )
            )
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Discovery-selected window per metric and horizon",
            "",
            "Only W is selected in state 0; lag is fixed at 1 and comparison at action_all.",
            "",
            "| metric | horizon | selected W | discovery AUC | confirmation AUC | p | Bonferroni over 18 metric-horizon families |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["window_leads"]:
        lines.append(
            "| %s | t%d | W%d | %.4f | %.4f | %.6f | %.6f |"
            % (
                item["metric"],
                item["horizon"],
                item["window"],
                item["discovery_auc_failure"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_18"],
            )
        )

    lines.extend(
        [
            "",
            "## Discovery-selected window and lag (all action tokens)",
            "",
            "W and lag are selected in state 0; T1-T10 are always averaged.",
            "",
            "| metric | horizon | selected W/lag | discovery maxT p | confirmation AUC | p | Bonferroni over 18 |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["action_lag_leads"]:
        lines.append(
            "| %s | t%d | W%d / lag%d | %.6f | %.4f | %.6f | %.6f |"
            % (
                item["metric"],
                item["horizon"],
                item["window"],
                item["lag"],
                item["discovery_p_fwer"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_18"],
            )
        )

    lines.extend(
        [
            "",
            "## Joint window-lag-token scan",
            "",
            "W1-W7, lag1-lag5, and T0-T10/action_all are selected in state 0 and tested in state 42.",
            "",
            "| metric | horizon | selected feature | discovery maxT p | confirmation AUC | p | Bonferroni over 18 |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["joint_leads"]:
        lines.append(
            "| %s | t%d | `%s` | %.6f | %.4f | %.6f | %.6f |"
            % (
                item["metric"],
                item["horizon"],
                item["key"],
                item["discovery_p_fwer"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_18"],
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
    key_index = {key: index for index, key in enumerate(keys)}
    rng = np.random.default_rng(args.seed)

    fixed_indexes = np.asarray(
        [
            key_index[
                f"{metric}|h{horizon:02d}|W{window}|lag1|action_all"
            ]
            for metric in route.METRICS
            for horizon in core.HORIZONS
            for window in WINDOWS
        ],
        dtype=np.int64,
    )
    fixed_local = core.permutation_scan(
        discovery[:, fixed_indexes],
        discovery_labels,
        args.permutations,
        rng,
    )
    fixed_scan = route.map_scan(fixed_local, fixed_indexes, len(keys))
    fixed_directions = np.where(fixed_local["auc"] >= 0.5, 1.0, -1.0)
    fixed_raw, fixed_oriented, fixed_p = confirmation_many(
        confirmation[:, fixed_indexes],
        confirmation_labels,
        fixed_directions,
        args.permutations,
        rng,
    )
    fixed_lag1_action = [
        fixed_result(
            int(index),
            keys,
            metadata,
            discovery,
            confirmation,
            discovery_labels,
            confirmation_labels,
            fixed_scan,
            fixed_raw[position],
            fixed_oriented[position],
            fixed_p[position],
            len(fixed_indexes),
        )
        for position, index in enumerate(fixed_indexes)
    ]
    fixed_by_key = {item["key"]: item for item in fixed_lag1_action}

    window_leads = []
    for metric in route.METRICS:
        for horizon in core.HORIZONS:
            candidates = [
                fixed_by_key[
                    f"{metric}|h{horizon:02d}|W{window}|lag1|action_all"
                ]
                for window in WINDOWS
            ]
            item = max(
                candidates,
                key=lambda candidate: abs(
                    candidate["discovery_auc_failure"] - 0.5
                ),
            ).copy()
            item["confirmation_p_bonferroni_18"] = min(
                1.0,
                18.0 * item["confirmation"]["permutation_p_one_sided"],
            )
            window_leads.append(item)

    action_lag_leads = []
    for metric in route.METRICS:
        for horizon in core.HORIZONS:
            indexes = np.asarray(
                [
                    index
                    for index, key in enumerate(keys)
                    if metadata[key]["metric"] == metric
                    and metadata[key]["horizon"] == horizon
                    and metadata[key]["comparison"] == "action_all"
                ],
                dtype=np.int64,
            )
            local = core.permutation_scan(
                discovery[:, indexes],
                discovery_labels,
                args.permutations,
                rng,
            )
            scan = route.map_scan(local, indexes, len(keys))
            selected = int(
                indexes[np.argmax(np.abs(scan["auc"][indexes] - 0.5))]
            )
            item = route.feature_result(
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
            item["confirmation_p_bonferroni_18"] = min(
                1.0,
                18.0 * item["confirmation"]["permutation_p_one_sided"],
            )
            action_lag_leads.append(item)

    joint_leads = []
    for metric in route.METRICS:
        for horizon in core.HORIZONS:
            indexes = np.asarray(
                [
                    index
                    for index, key in enumerate(keys)
                    if metadata[key]["metric"] == metric
                    and metadata[key]["horizon"] == horizon
                ],
                dtype=np.int64,
            )
            local = core.permutation_scan(
                discovery[:, indexes],
                discovery_labels,
                args.permutations,
                rng,
            )
            scan = route.map_scan(local, indexes, len(keys))
            selected = int(
                indexes[np.argmax(np.abs(scan["auc"][indexes] - 0.5))]
            )
            item = route.feature_result(
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
            item["confirmation_p_bonferroni_18"] = min(
                1.0,
                18.0 * item["confirmation"]["permutation_p_one_sided"],
            )
            joint_leads.append(item)

    summary = {
        "experiment": "terminal_route_window_ablation_v1",
        "integrity": {
            "run": str(run),
            "features": len(keys),
            "episodes_per_state": 32,
            "discovery_state": args.discovery_state,
            "confirmation_state": args.confirmation_state,
        },
        "protocol": {
            "windows_are_distance_pair_counts": True,
            "windows": list(WINDOWS),
            "horizons": list(core.HORIZONS),
            "lags": list(core.LAGS),
            "metrics": list(route.METRICS),
            "sample_unit": "rollout",
            "permutation_draws": args.permutations,
        },
        "fixed_lag1_action": fixed_lag1_action,
        "window_leads": window_leads,
        "action_lag_leads": action_lag_leads,
        "joint_leads": joint_leads,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "window_summary.json").write_text(
        json.dumps(core.finite_json(summary), indent=2, ensure_ascii=True) + "\n"
    )
    (output / "window_report.md").write_text(render_report(summary))
    np.savez_compressed(
        output / "window_scores.npz",
        feature_keys=np.asarray(keys),
        discovery_scores=discovery,
        confirmation_scores=confirmation,
        discovery_failure=discovery_labels,
        confirmation_failure=confirmation_labels,
    )
    print(f"wrote {output / 'window_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
