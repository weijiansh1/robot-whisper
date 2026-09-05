#!/usr/bin/env python3
"""Test first- and second-order temporal information in HB-MoE routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

import analyze as core
import analyze_jaccard as route


HERE = Path(__file__).resolve().parent
WINDOWS = tuple(range(1, 8))
SOURCES = ("hard_route", "soft_route", "selected_route")
FIRST_ORDER = ("speed", "active_fraction")
DIRECTION_ONLY = ("alignment", "straightness")
HIGHER_ORDER = (
    "acceleration",
    "alignment",
    "speed_delta",
    "straightness",
    "ema05_magnitude",
    "ema08_magnitude",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=core.RUN)
    parser.add_argument("--discovery-state", type=int, default=0)
    parser.add_argument("--confirmation-state", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def register(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    source: str,
    horizon: int,
    window: int,
    statistic: str,
    reduced: np.ndarray,
) -> None:
    reduced = np.asarray(reduced, dtype=np.float64)
    if reduced.shape != (11,) or not np.all(np.isfinite(reduced)):
        raise ValueError(
            f"bad {source}/{statistic}/h{horizon}/W{window}: {reduced.shape}"
        )
    for token, value in enumerate(reduced):
        key = f"{source}|h{horizon:02d}|W{window}|{statistic}|T{token}"
        fields = {
            "source": source,
            "horizon": horizon,
            "window": window,
            "statistic": statistic,
            "comparison": f"T{token}",
        }
        output[key] = float(value)
        previous = metadata.setdefault(key, fields)
        if previous != fields:
            raise ValueError(f"metadata mismatch for {key}")
    key = f"{source}|h{horizon:02d}|W{window}|{statistic}|action_all"
    fields = {
        "source": source,
        "horizon": horizon,
        "window": window,
        "statistic": statistic,
        "comparison": "action_all",
    }
    output[key] = float(reduced[1:11].mean())
    previous = metadata.setdefault(key, fields)
    if previous != fields:
        raise ValueError(f"metadata mismatch for {key}")


def masked_alignment(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = (first * second).sum(axis=-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    valid = denominator > 1e-8
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=valid,
    )
    count = valid.sum(axis=(0, 1, 2))
    return np.divide(
        (values * valid).sum(axis=(0, 1, 2)),
        count,
        out=np.zeros(11, dtype=np.float64),
        where=count > 0,
    )


def ema_magnitude(velocity: np.ndarray, beta: float) -> np.ndarray:
    weights = beta ** np.arange(velocity.shape[0] - 1, -1, -1, dtype=np.float32)
    weights /= weights.sum()
    momentum = np.einsum("t,tlfue->lfue", weights, velocity, optimize=True)
    return np.linalg.norm(momentum, axis=-1).mean(axis=(0, 1))


def add_momentum_features(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    source: str,
    values: np.ndarray,
) -> None:
    values = np.asarray(values, dtype=np.float32)
    for horizon in core.HORIZONS:
        for window in WINDOWS:
            current_start = horizon - window + 1
            if current_start < 1:
                continue
            current = values[current_start : horizon + 1]
            previous = values[current_start - 1 : horizon]
            velocity = current - previous
            speed = np.linalg.norm(velocity, axis=-1)
            register(
                output,
                metadata,
                source,
                horizon,
                window,
                "speed",
                speed.mean(axis=(0, 1, 2)),
            )
            register(
                output,
                metadata,
                source,
                horizon,
                window,
                "active_fraction",
                (speed > 1e-8).mean(axis=(0, 1, 2)),
            )

            path_length = speed.sum(axis=0)
            net_displacement = np.linalg.norm(velocity.sum(axis=0), axis=-1)
            straightness = np.divide(
                net_displacement,
                path_length,
                out=np.zeros_like(net_displacement),
                where=path_length > 1e-8,
            )
            register(
                output,
                metadata,
                source,
                horizon,
                window,
                "straightness",
                straightness.mean(axis=(0, 1)),
            )
            register(
                output,
                metadata,
                source,
                horizon,
                window,
                "ema05_magnitude",
                ema_magnitude(velocity, 0.5),
            )
            register(
                output,
                metadata,
                source,
                horizon,
                window,
                "ema08_magnitude",
                ema_magnitude(velocity, 0.8),
            )

            acceleration_start = max(current_start, 2)
            acceleration_current = values[acceleration_start : horizon + 1]
            acceleration_previous = values[acceleration_start - 1 : horizon]
            velocity_current = acceleration_current - acceleration_previous
            velocity_previous = (
                acceleration_previous
                - values[acceleration_start - 2 : horizon - 1]
            )
            acceleration = velocity_current - velocity_previous
            register(
                output,
                metadata,
                source,
                horizon,
                window,
                "acceleration",
                np.linalg.norm(acceleration, axis=-1).mean(axis=(0, 1, 2)),
            )
            register(
                output,
                metadata,
                source,
                horizon,
                window,
                "alignment",
                masked_alignment(velocity_current, velocity_previous),
            )
            speed_delta = np.linalg.norm(velocity_current, axis=-1) - np.linalg.norm(
                velocity_previous, axis=-1
            )
            register(
                output,
                metadata,
                source,
                horizon,
                window,
                "speed_delta",
                speed_delta.mean(axis=(0, 1, 2)),
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
            hard = core.top4_probabilities(ids)
            selected = route.selected_weight_vectors(
                ids,
                np.asarray(store["hb_selected_prob"][start:stop], dtype=np.float32),
            )
            features: dict[str, float] = {}
            add_momentum_features(features, metadata, "hard_route", hard)
            add_momentum_features(features, metadata, "soft_route", soft)
            add_momentum_features(features, metadata, "selected_route", selected)
            rows.append(features)
            print(f"Momentum state {state}: {position:02d}/32", flush=True)
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


def select_leads(
    discovery: np.ndarray,
    confirmation: np.ndarray,
    discovery_labels: np.ndarray,
    confirmation_labels: np.ndarray,
    keys: list[str],
    metadata: dict[str, dict[str, Any]],
    permutations: int,
    rng: np.random.Generator,
    *,
    comparisons: tuple[str, ...] | None,
    statistics: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    leads = []
    for source in SOURCES:
        for horizon in core.HORIZONS:
            indexes = np.asarray(
                [
                    index
                    for index, key in enumerate(keys)
                    if metadata[key]["source"] == source
                    and metadata[key]["horizon"] == horizon
                    and (
                        comparisons is None
                        or metadata[key]["comparison"] in comparisons
                    )
                    and (
                        statistics is None
                        or metadata[key]["statistic"] in statistics
                    )
                    and np.ptp(discovery[:, index]) > 1e-12
                ],
                dtype=np.int64,
            )
            local = core.permutation_scan(
                discovery[:, indexes], discovery_labels, permutations, rng
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
                permutations,
                rng,
            )
            item["confirmation_p_bonferroni_9"] = min(
                1.0,
                9.0 * item["confirmation"]["permutation_p_one_sided"],
            )
            leads.append(item)
    return leads


def add_lead_table(
    lines: list[str], title: str, description: str, rows: list[dict[str, Any]]
) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            description,
            "",
            "| source | horizon | selected feature | discovery maxT p | confirmation AUC | p | Bonferroni over 9 |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in rows:
        lines.append(
            "| %s | t%d | `%s` | %.6f | %.4f | %.6f | %.6f |"
            % (
                item["source"],
                item["horizon"],
                item["key"],
                item["discovery_p_fwer"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_9"],
            )
        )
    lines.append("")


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# HB-MoE route momentum analysis",
        "",
        "Velocity is v_t=x_t-x_(t-1); acceleration is a_t=v_t-v_(t-1).",
        "W is the number of terminal velocity samples. State 0 selects a feature and state 42 confirms it.",
        "",
    ]
    add_lead_table(
        lines,
        "All aggregate motion features",
        "Only action_all features are eligible; window and statistic are selected in discovery.",
        summary["aggregate_leads"],
    )
    add_lead_table(
        lines,
        "Direction-only features",
        "Only alignment and straightness are eligible, excluding speed magnitude.",
        summary["direction_leads"],
    )
    add_lead_table(
        lines,
        "Full token scan",
        "Window, statistic, and T0-T10/action_all are selected in discovery.",
        summary["full_leads"],
    )
    return "\n".join(lines)


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
    rng = np.random.default_rng(args.seed)

    aggregate_leads = select_leads(
        discovery,
        confirmation,
        discovery_labels,
        confirmation_labels,
        keys,
        metadata,
        args.permutations,
        rng,
        comparisons=("action_all",),
        statistics=None,
    )
    direction_leads = select_leads(
        discovery,
        confirmation,
        discovery_labels,
        confirmation_labels,
        keys,
        metadata,
        args.permutations,
        rng,
        comparisons=("action_all",),
        statistics=DIRECTION_ONLY,
    )
    full_leads = select_leads(
        discovery,
        confirmation,
        discovery_labels,
        confirmation_labels,
        keys,
        metadata,
        args.permutations,
        rng,
        comparisons=None,
        statistics=None,
    )
    summary = {
        "experiment": "himoe_route_momentum_v1",
        "integrity": {
            "run": str(run),
            "features": len(keys),
            "episodes_per_state": 32,
            "discovery_state": args.discovery_state,
            "confirmation_state": args.confirmation_state,
        },
        "protocol": {
            "windows": list(WINDOWS),
            "horizons": list(core.HORIZONS),
            "sources": list(SOURCES),
            "first_order_statistics": list(FIRST_ORDER),
            "higher_order_statistics": list(HIGHER_ORDER),
            "sample_unit": "rollout",
            "permutation_draws": args.permutations,
        },
        "aggregate_leads": aggregate_leads,
        "direction_leads": direction_leads,
        "full_leads": full_leads,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "momentum_summary.json").write_text(
        json.dumps(core.finite_json(summary), indent=2, ensure_ascii=True) + "\n"
    )
    (output / "momentum_report.md").write_text(render_report(summary))
    np.savez_compressed(
        output / "momentum_scores.npz",
        feature_keys=np.asarray(keys),
        discovery_scores=discovery,
        confirmation_scores=confirmation,
        discovery_failure=discovery_labels,
        confirmation_failure=confirmation_labels,
    )
    print(f"wrote {output / 'momentum_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
