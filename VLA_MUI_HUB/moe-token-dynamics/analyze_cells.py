#!/usr/bin/env python3
"""Post-discovery localization without averaging HB layer or flow cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import zarr
from scipy.stats import rankdata

import analyze as core


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=core.RUN)
    parser.add_argument("--discovery-state", type=int, default=0)
    parser.add_argument("--confirmation-state", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def route_series(probabilities: np.ndarray) -> list[tuple[str, np.ndarray, Callable]]:
    probabilities = core.normalize_probability(probabilities)
    embedding = np.sqrt(probabilities)
    action = probabilities[..., core.ACTION_TOKENS, :]
    action_embedding = embedding[..., core.ACTION_TOKENS, :]
    action_mean = core.normalize_probability(action.mean(axis=-2))
    action_std = action.std(axis=-2)
    endpoint_delta = action[..., -1, :] - action[..., 0, :]
    spectrum, _ = core.spectrum_and_gram(action_embedding)
    return [
        ("tokens", embedding, core.hellinger_embedding_distance),
        ("action_mean", np.sqrt(action_mean), core.hellinger_embedding_distance),
        ("action_std", action_std, core.euclidean_distance),
        ("endpoint_delta", endpoint_delta, core.euclidean_distance),
        ("gram_spectrum", spectrum, core.euclidean_distance),
    ]


def hidden_series(hidden: np.ndarray) -> list[tuple[str, np.ndarray, Callable]]:
    unit = core.normalize_vector(hidden)
    action = unit[..., core.ACTION_TOKENS, :]
    action_mean = core.normalize_vector(action.mean(axis=-2))
    action_std = action.std(axis=-2)
    endpoint_delta = action[..., -1, :] - action[..., 0, :]
    spectrum, _ = core.spectrum_and_gram(action)
    return [
        ("tokens", unit, core.cosine_distance),
        ("action_mean", action_mean, core.cosine_distance),
        ("action_std", action_std, core.euclidean_distance),
        ("endpoint_delta", endpoint_delta, core.euclidean_distance),
        ("gram_spectrum", spectrum, core.euclidean_distance),
    ]


def append_cells(
    parts: list[np.ndarray],
    keys: list[str] | None,
    source: str,
    horizon: int,
    statistic: str,
    series: str,
    values: np.ndarray,
) -> None:
    if series == "tokens":
        if values.shape != (8, 10, 11):
            raise ValueError(f"bad token cell values {values.shape}")
        parts.append(values[:, 0, 0].reshape(-1))
        parts.append(values[:, :, 1:11].reshape(-1))
        if keys is not None:
            for layer in core.LAYERS:
                keys.append(
                    f"{source}|h{horizon:02d}|{statistic}|T0|L{layer}|f0"
                )
            for layer in core.LAYERS:
                for flow in range(10):
                    for token in range(1, 11):
                        keys.append(
                            f"{source}|h{horizon:02d}|{statistic}|T{token}|L{layer}|f{flow}"
                        )
    else:
        if values.shape != (8, 10):
            raise ValueError(f"bad descriptor cell values {values.shape}")
        parts.append(values.reshape(-1))
        if keys is not None:
            for layer in core.LAYERS:
                for flow in range(10):
                    keys.append(
                        f"{source}|h{horizon:02d}|{statistic}|{series}|L{layer}|f{flow}"
                    )


def source_cell_features(
    source: str,
    specs: list[tuple[str, np.ndarray, Callable]],
    build_keys: bool,
) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    keys: list[str] | None = [] if build_keys else None
    for horizon in core.HORIZONS:
        start = max(0, horizon - core.TRAILING_WINDOW + 1)
        for series, values, metric in specs:
            lag_values = []
            for lag in core.LAGS:
                measured = metric(
                    values[start + lag : horizon + 1],
                    values[start : horizon + 1 - lag],
                ).mean(axis=0)
                lag_values.append(measured)
                append_cells(
                    parts,
                    keys,
                    source,
                    horizon,
                    f"lag{lag}",
                    series,
                    measured,
                )
            lag_values = np.stack(lag_values)
            recurrences = {
                2: 0.5 * (lag_values[0] + lag_values[2]) - lag_values[1],
                3: 0.5 * (lag_values[1] + lag_values[3]) - lag_values[2],
                4: 0.5 * (lag_values[2] + lag_values[4]) - lag_values[3],
                5: lag_values[3] - lag_values[4],
            }
            for period, measured in recurrences.items():
                append_cells(
                    parts,
                    keys,
                    source,
                    horizon,
                    f"period{period}",
                    series,
                    measured,
                )
    return np.concatenate(parts).astype(np.float32), keys or []


def extract(
    run: Path,
    episodes: list[core.Episode],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str]]:
    routes = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    hidden = zarr.open(str(run / "server" / "hidden.zarr"), mode="r")
    prefix = max(core.HORIZONS) + 1
    keys: list[str] = []
    matrices: dict[int, np.ndarray] = {}
    labels: dict[int, np.ndarray] = {}
    first = True
    for state in sorted({episode.state for episode in episodes}):
        group = sorted(
            [episode for episode in episodes if episode.state == state],
            key=lambda episode: episode.noise_seed,
        )
        rows = []
        for position, episode in enumerate(group, start=1):
            start = episode.offset
            stop = start + prefix
            soft = np.asarray(routes["hb_router_probs"][start:stop], dtype=np.float32)
            ids = np.asarray(routes["hb_expert_ids"][start:stop], dtype=np.uint8)
            hidden_values = np.asarray(hidden["hb_hidden"][start:stop], dtype=np.float32)
            pieces = []
            local_keys = []
            for source, specs in (
                ("route_soft", route_series(soft)),
                ("route_top4", route_series(core.top4_probabilities(ids))),
                ("hidden", hidden_series(hidden_values)),
            ):
                vector, source_keys = source_cell_features(source, specs, first)
                pieces.append(vector)
                local_keys.extend(source_keys)
            row = np.concatenate(pieces)
            if first:
                keys = local_keys
                if len(keys) != len(row):
                    raise ValueError("cell feature keys do not align")
                first = False
            elif len(row) != len(keys):
                raise ValueError("cell feature width changed")
            rows.append(row)
            print(f"cell state {state}: {position:02d}/32", flush=True)
        matrices[state] = np.stack(rows)
        labels[state] = np.asarray([episode.failure for episode in group], dtype=np.int8)
    return matrices, labels, keys


def auc_vector(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return core.auc_from_ranks(rankdata(matrix, axis=0, method="average"), labels)


def selected_result(
    index: int,
    keys: list[str],
    discovery_auc: np.ndarray,
    discovery: np.ndarray,
    confirmation: np.ndarray,
    discovery_labels: np.ndarray,
    confirmation_labels: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    auc = float(discovery_auc[index])
    direction = 1.0 if auc >= 0.5 else -1.0
    result = core.confirmation_test(
        confirmation[:, index],
        confirmation_labels,
        direction,
        permutations,
        rng,
    )
    key = keys[index]
    fields = key.split("|")
    selected = discovery[:, index]
    confirmed = confirmation[:, index]
    return {
        "key": key,
        "source": fields[0],
        "horizon": int(fields[1][1:]),
        "statistic": fields[2],
        "series": fields[3],
        "layer": int(fields[4][1:]),
        "flow": int(fields[5][1:]),
        "discovery_auc_failure": auc,
        "discovery_direction": "larger_in_failure" if direction > 0 else "smaller_in_failure",
        "discovery_means": {
            "success": float(selected[discovery_labels == 0].mean()),
            "timeout_failure": float(selected[discovery_labels == 1].mean()),
        },
        "confirmation_means": {
            "success": float(confirmed[confirmation_labels == 0].mean()),
            "timeout_failure": float(confirmed[confirmation_labels == 1].mean()),
        },
        "confirmation": result,
    }


def render_report(summary: dict) -> str:
    lines = [
        "# Layer-flow cell localization",
        "",
        "The discovery scan is intentionally treated only as feature selection. Its p-values are not used. Each selected cell is tested with a fixed direction in state 42.",
        "",
        "## Horizon leads",
        "",
        "| horizon | selected cell | discovery AUC | confirmation oriented AUC | confirmation p | Bonferroni over 3 horizons |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in summary["horizon_leads"]:
        lines.append(
            "| t%d | `%s` | %.4f | %.4f | %.6f | %.6f |"
            % (
                item["horizon"],
                item["key"],
                item["discovery_auc_failure"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_3_horizons"],
            )
        )
    lines.extend(
        [
            "",
            "## Source by horizon",
            "",
            "| horizon | source | selected cell | discovery AUC | confirmation oriented AUC | p |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for item in summary["source_horizon_leads"]:
        lines.append(
            "| t%d | %s | `%s` | %.4f | %.4f | %.6f |"
            % (
                item["horizon"],
                item["source"],
                item["key"],
                item["discovery_auc_failure"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
            )
        )
    lines.extend(
        [
            "",
            "## Category by horizon",
            "",
            "| horizon | category | selected cell | discovery AUC | confirmation oriented AUC | p | Bonferroni over 9 category-horizon tests |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["category_horizon_leads"]:
        lines.append(
            "| t%d | %s | `%s` | %.4f | %.4f | %.6f | %.6f |"
            % (
                item["horizon"],
                item["category"],
                item["key"],
                item["discovery_auc_failure"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_9"],
            )
        )
    lines.extend(
        [
            "",
            "The same 32 noise seed IDs occur in discovery and confirmation states. A replicated cell is cross-state evidence, not unseen-seed evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    states = (args.discovery_state, args.confirmation_state)
    episodes, _ = core.load_episodes(run, states)
    matrices, labels, keys = extract(run, episodes)
    discovery = matrices[args.discovery_state]
    confirmation = matrices[args.confirmation_state]
    discovery_labels = labels[args.discovery_state]
    confirmation_labels = labels[args.confirmation_state]
    variable = np.ptp(discovery, axis=0) > 1e-12
    discovery_auc = np.full(len(keys), np.nan)
    discovery_auc[variable] = auc_vector(discovery[:, variable], discovery_labels)
    rng = np.random.default_rng(args.seed)

    horizon_leads = []
    for horizon in core.HORIZONS:
        candidates = np.asarray(
            [
                index
                for index, key in enumerate(keys)
                if variable[index] and f"|h{horizon:02d}|" in key
            ],
            dtype=np.int64,
        )
        selected = int(
            candidates[np.argmax(np.abs(discovery_auc[candidates] - 0.5))]
        )
        item = selected_result(
            selected,
            keys,
            discovery_auc,
            discovery,
            confirmation,
            discovery_labels,
            confirmation_labels,
            args.permutations,
            rng,
        )
        item["confirmation_p_bonferroni_3_horizons"] = min(
            1.0, 3.0 * item["confirmation"]["permutation_p_one_sided"]
        )
        horizon_leads.append(item)

    source_horizon_leads = []
    for horizon in core.HORIZONS:
        for source in ("route_soft", "route_top4", "hidden"):
            candidates = np.asarray(
                [
                    index
                    for index, key in enumerate(keys)
                    if variable[index]
                    and key.startswith(source + "|")
                    and f"|h{horizon:02d}|" in key
                ],
                dtype=np.int64,
            )
            selected = int(
                candidates[np.argmax(np.abs(discovery_auc[candidates] - 0.5))]
            )
            source_horizon_leads.append(
                selected_result(
                    selected,
                    keys,
                    discovery_auc,
                    discovery,
                    confirmation,
                    discovery_labels,
                    confirmation_labels,
                    args.permutations,
                    rng,
                )
            )

    category_horizon_leads = []
    for horizon in core.HORIZONS:
        for category in ("token_lag", "descriptor_lag", "periodicity"):
            candidates = []
            for index, key in enumerate(keys):
                if not variable[index] or f"|h{horizon:02d}|" not in key:
                    continue
                fields = key.split("|")
                statistic = fields[2]
                series = fields[3]
                item_category = (
                    "periodicity"
                    if statistic.startswith("period")
                    else "token_lag"
                    if series.startswith("T")
                    else "descriptor_lag"
                )
                if item_category == category:
                    candidates.append(index)
            candidates_array = np.asarray(candidates, dtype=np.int64)
            selected = int(
                candidates_array[
                    np.argmax(np.abs(discovery_auc[candidates_array] - 0.5))
                ]
            )
            item = selected_result(
                selected,
                keys,
                discovery_auc,
                discovery,
                confirmation,
                discovery_labels,
                confirmation_labels,
                args.permutations,
                rng,
            )
            item["category"] = category
            item["confirmation_p_bonferroni_9"] = min(
                1.0, 9.0 * item["confirmation"]["permutation_p_one_sided"]
            )
            category_horizon_leads.append(item)

    summary = {
        "experiment": "himoe_hb_layer_flow_cell_localization_v1",
        "integrity": {
            "run": str(run),
            "features": len(keys),
            "variable_discovery_features": int(variable.sum()),
            "episodes_per_state": 32,
            "discovery_state": args.discovery_state,
            "confirmation_state": args.confirmation_state,
        },
        "protocol": {
            "sample_unit": "rollout",
            "selection": "maximum abs failure AUC in state 0; no discovery p-value interpreted",
            "confirmation": "fixed direction in state 42",
            "permutation_draws": args.permutations,
            "state_token_flow_handling": "T0 stored once per layer at f0 because it is denoise invariant",
        },
        "horizon_leads": horizon_leads,
        "source_horizon_leads": source_horizon_leads,
        "category_horizon_leads": category_horizon_leads,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "cell_summary.json").write_text(
        json.dumps(core.finite_json(summary), indent=2, ensure_ascii=True) + "\n"
    )
    (output / "cell_report.md").write_text(render_report(summary))
    np.savez_compressed(
        output / "cell_scores.npz",
        feature_keys=np.asarray(keys),
        discovery_scores=discovery,
        confirmation_scores=confirmation,
        discovery_failure=discovery_labels,
        confirmation_failure=confirmation_labels,
    )
    print(f"wrote {output / 'cell_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
