"""Analyze a paired early-route commitment grid.

Primary endpoint: for a fixed first-step route, does changing only the future
flow-noise stream ever change the final success label?  A mixed row is a direct
counterexample to deterministic commitment by that early route.  Permutations
are performed independently within each future-noise column, preserving common
future-stream difficulty while testing whether row identity concentrates the
outcomes.

The per-denoising-step route decoder is deliberately secondary and exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import zarr


@dataclass
class Cell:
    summary: dict
    route0: np.ndarray
    as_probs0: np.ndarray
    ids0: np.ndarray
    entropy0: np.ndarray
    flow_noises: np.ndarray
    action0: np.ndarray


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability))


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _knn_weight_matrix(features: np.ndarray, k: int = 3) -> np.ndarray:
    """Label-free leave-one-row-out neighbor weights."""
    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]
    if n < 3:
        raise ValueError("at least three rows are needed for route signal analysis")
    k = min(k, n - 1)
    weights = np.zeros((n, n), dtype=np.float64)
    for held_out in range(n):
        train = np.arange(n) != held_out
        mean = features[train].mean(axis=0)
        scale = features[train].std(axis=0)
        scale[scale < 1e-7] = 1.0
        delta = (features[train] - features[held_out]) / scale
        distances = np.sqrt(np.mean(delta * delta, axis=1))
        candidates = np.flatnonzero(train)
        nearest_local = np.argsort(distances)[:k]
        nearest = candidates[nearest_local]
        inverse = 1.0 / np.maximum(distances[nearest_local], 1e-9)
        weights[held_out, nearest] = inverse / inverse.sum()
    return weights


def _signal_score(weights: np.ndarray, target: np.ndarray) -> dict:
    prediction = weights @ target
    baseline = (target.sum() - target) / (len(target) - 1)
    mse = float(np.mean((prediction - target) ** 2))
    baseline_mse = float(np.mean((baseline - target) ** 2))
    return {
        "correlation": _safe_corr(prediction, target),
        "mse": mse,
        "loo_mean_baseline_mse": baseline_mse,
        "mse_skill": 1.0 - mse / baseline_mse if baseline_mse > 0 else 0.0,
        "prediction": prediction.tolist(),
    }


def _commitment_statistics(outcomes: np.ndarray, n_perm: int, seed: int) -> dict:
    outcomes = np.asarray(outcomes, dtype=np.float64)
    n_rows, n_cols = outcomes.shape
    row_rates = outcomes.mean(axis=1)
    col_rates = outcomes.mean(axis=0)
    overall = float(outcomes.mean())
    conditional_entropy = float(np.mean([_binary_entropy(v) for v in row_rates]))
    mutual_information = _binary_entropy(overall) - conditional_entropy
    stable_rows = int(np.sum((row_rates == 0) | (row_rates == 1)))
    row_variance = float(np.var(row_rates))
    column_variance = float(np.var(col_rates))

    rng = np.random.default_rng(seed)
    future_rng = np.random.default_rng(seed + 1)
    null_entropy = np.empty(n_perm, dtype=np.float64)
    null_mi = np.empty(n_perm, dtype=np.float64)
    null_stable = np.empty(n_perm, dtype=np.int16)
    null_variance = np.empty(n_perm, dtype=np.float64)
    null_column_variance = np.empty(n_perm, dtype=np.float64)
    for iteration in range(n_perm):
        # Preserve each future stream's success count and destroy row identity.
        permuted = np.stack([rng.permutation(outcomes[:, col]) for col in range(n_cols)], axis=1)
        rates = permuted.mean(axis=1)
        entropy = float(np.mean([_binary_entropy(v) for v in rates]))
        null_entropy[iteration] = entropy
        null_mi[iteration] = _binary_entropy(float(permuted.mean())) - entropy
        null_stable[iteration] = np.sum((rates == 0) | (rates == 1))
        null_variance[iteration] = np.var(rates)

        # Symmetric check: preserve each route's success count and destroy the
        # identity of the common future stream.
        future_permuted = np.stack(
            [future_rng.permutation(outcomes[row]) for row in range(n_rows)], axis=0
        )
        null_column_variance[iteration] = np.var(future_permuted.mean(axis=0))

    def upper_p(null, observed):
        return float((1 + np.sum(null >= observed - 1e-15)) / (len(null) + 1))

    def lower_p(null, observed):
        return float((1 + np.sum(null <= observed + 1e-15)) / (len(null) + 1))

    return {
        "n_first_routes": n_rows,
        "n_future_streams": n_cols,
        "overall_success_rate": overall,
        "row_success_rates": row_rates.tolist(),
        "future_stream_success_rates": col_rates.tolist(),
        "mixed_rows": int(np.sum((row_rates > 0) & (row_rates < 1))),
        "all_failure_rows": int(np.sum(row_rates == 0)),
        "all_success_rows": int(np.sum(row_rates == 1)),
        "stable_rows": stable_rows,
        "conditional_entropy_bits": conditional_entropy,
        "outcome_entropy_bits": _binary_entropy(overall),
        "route_outcome_mutual_information_bits_plugin": mutual_information,
        "row_rate_variance": row_variance,
        "future_stream_effect": {
            "column_rate_variance": column_variance,
            "permutation": {
                "method": "independent column permutation within every fixed-first-route row",
                "n": n_perm,
                "column_variance_upper_p": upper_p(null_column_variance, column_variance),
                "null_column_variance_mean": float(null_column_variance.mean()),
            },
        },
        "permutation": {
            "method": "independent row permutation within every common-future column",
            "n": n_perm,
            "conditional_entropy_lower_p": lower_p(null_entropy, conditional_entropy),
            "mutual_information_upper_p": upper_p(null_mi, mutual_information),
            "stable_rows_upper_p": upper_p(null_stable, stable_rows),
            "row_variance_upper_p": upper_p(null_variance, row_variance),
            "null_conditional_entropy_mean": float(null_entropy.mean()),
            "null_mutual_information_mean": float(null_mi.mean()),
            "null_stable_rows_mean": float(null_stable.mean()),
            "null_row_variance_mean": float(null_variance.mean()),
        },
    }


def _route_signal(
    routes: np.ndarray,
    first_noises: np.ndarray,
    first_actions: np.ndarray,
    target: np.ndarray,
    n_perm: int,
    seed: int,
) -> dict:
    """Exploratory leave-one-route-out geometry test for each denoising round."""
    n_rows, _layers, n_denoise, _tokens, _experts = routes.shape
    if n_rows < 5:
        return {"available": False, "reason": "fewer than five unique first routes"}

    route_weights = []
    observed = []
    for denoise in range(n_denoise):
        # Keep all ten action tokens; exclude the state token.
        features = routes[:, :, denoise, 1:, :].reshape(n_rows, -1)
        weights = _knn_weight_matrix(features)
        route_weights.append(weights)
        observed.append(_signal_score(weights, target))

    noise_weights = _knn_weight_matrix(first_noises.reshape(n_rows, -1))
    action_weights = _knn_weight_matrix(first_actions.reshape(n_rows, -1))
    noise_score = _signal_score(noise_weights, target)
    action_score = _signal_score(action_weights, target)

    rng = np.random.default_rng(seed)
    null_max_corr = np.empty(n_perm, dtype=np.float64)
    null_noise_corr = np.empty(n_perm, dtype=np.float64)
    null_action_corr = np.empty(n_perm, dtype=np.float64)
    null_per_denoise = np.empty((n_perm, n_denoise), dtype=np.float64)
    for iteration in range(n_perm):
        permuted = rng.permutation(target)
        for denoise, weights in enumerate(route_weights):
            null_per_denoise[iteration, denoise] = _safe_corr(weights @ permuted, permuted)
        null_max_corr[iteration] = null_per_denoise[iteration].max()
        null_noise_corr[iteration] = _safe_corr(noise_weights @ permuted, permuted)
        null_action_corr[iteration] = _safe_corr(action_weights @ permuted, permuted)

    for denoise, score in enumerate(observed):
        corr = score["correlation"]
        score["denoise"] = denoise
        score["permutation_p_uncorrected"] = float(
            (1 + np.sum(null_per_denoise[:, denoise] >= corr - 1e-15)) / (n_perm + 1)
        )
        score["permutation_p_fwer_10_denoise"] = float(
            (1 + np.sum(null_max_corr >= corr - 1e-15)) / (n_perm + 1)
        )

    noise_score["permutation_p"] = float(
        (1 + np.sum(null_noise_corr >= noise_score["correlation"] - 1e-15)) / (n_perm + 1)
    )
    action_score["permutation_p"] = float(
        (1 + np.sum(null_action_corr >= action_score["correlation"] - 1e-15)) / (n_perm + 1)
    )
    return {
        "available": True,
        "method": "3-nearest-neighbor leave-one-first-route-out regression",
        "target": "empirical P(success | fixed first route) over common future streams",
        "per_denoise": observed,
        "first_noise_baseline": noise_score,
        "first_action_chunk_baseline": action_score,
        "n_permutations": n_perm,
    }


def _load_cells(server_dir: pathlib.Path, client_dir: pathlib.Path) -> tuple[list[Cell], dict]:
    summaries = json.loads((client_dir / "summaries.json").read_text())
    group = zarr.open(str(server_dir / "routes.zarr"), mode="r")
    required = (
        "as_probs",
        "hb_router_probs",
        "hb_expert_ids",
        "hb_entropy",
        "episode_id",
        "control_step",
    )
    missing = [name for name in required if name not in group]
    if missing:
        raise RuntimeError("route store is missing arrays: %s" % missing)

    total = int(sum(int(summary["inference_calls"]) for summary in summaries))
    available = int(group["hb_router_probs"].shape[0])
    if total != available:
        raise RuntimeError(
            "client has %d inference calls but route store has %d; flush the server first"
            % (total, available)
        )

    control_steps = np.asarray(group["control_step"][:])
    if len(control_steps) > 1 and not np.all(np.diff(control_steps) == 1):
        raise RuntimeError("server control_step axis is not contiguous")

    cells = []
    cursor = 0
    for summary in summaries:
        count = int(summary["inference_calls"])
        if count <= 0:
            raise RuntimeError("episode %s contains no inference calls" % summary["episode_index"])
        episode_ids = np.asarray(group["episode_id"][cursor : cursor + count])
        if not np.all(episode_ids == int(summary["episode_index"])):
            raise RuntimeError("server/client episode ids disagree at trace offset %d" % cursor)
        archive = np.load(client_dir / ("episode_%04d.npz" % int(summary["episode_index"])))
        cells.append(
            Cell(
                summary=summary,
                route0=np.asarray(group["hb_router_probs"][cursor], dtype=np.float32),
                as_probs0=np.asarray(group["as_probs"][cursor], dtype=np.float32),
                ids0=np.asarray(group["hb_expert_ids"][cursor]),
                entropy0=np.asarray(group["hb_entropy"][cursor], dtype=np.float32),
                flow_noises=np.asarray(archive["flow_noises"], dtype=np.float32),
                action0=np.asarray(archive["action_chunks"][0], dtype=np.float32),
            )
        )
        cursor += count
    return cells, dict(group.attrs)


def _integrity(cells: list[Cell]) -> dict:
    observation_hashes = {cell.summary["initial_observation_sha256"] for cell in cells}
    rows = sorted({int(cell.summary["grid_row"]) for cell in cells})
    cols = sorted({int(cell.summary["grid_col"]) for cell in cells})

    max_hb_route_diff = 0.0
    max_as_route_diff = 0.0
    max_action_diff = 0.0
    first_noise_mismatches = 0
    compared_row_pairs = 0
    for row in rows:
        group = [cell for cell in cells if int(cell.summary["grid_row"]) == row]
        reference = group[0]
        for cell in group[1:]:
            max_hb_route_diff = max(
                max_hb_route_diff, float(np.max(np.abs(cell.route0 - reference.route0)))
            )
            max_as_route_diff = max(
                max_as_route_diff, float(np.max(np.abs(cell.as_probs0 - reference.as_probs0)))
            )
            max_action_diff = max(max_action_diff, float(np.max(np.abs(cell.action0 - reference.action0))))
            first_noise_mismatches += int(
                _array_sha256(cell.flow_noises[0]) != _array_sha256(reference.flow_noises[0])
            )
            compared_row_pairs += 1

    future_noise_mismatches = 0
    compared_future_tensors = 0
    for col in cols:
        group = [cell for cell in cells if int(cell.summary["grid_col"]) == col]
        reference = group[0]
        for cell in group[1:]:
            common = min(len(reference.flow_noises), len(cell.flow_noises)) - 1
            for offset in range(max(0, common)):
                compared_future_tensors += 1
                future_noise_mismatches += int(
                    _array_sha256(reference.flow_noises[offset + 1])
                    != _array_sha256(cell.flow_noises[offset + 1])
                )

    return {
        "n_cells": len(cells),
        "n_initial_observation_hashes": len(observation_hashes),
        "initial_observation_sha256": next(iter(observation_hashes)) if len(observation_hashes) == 1 else None,
        "same_row_pairs_compared": compared_row_pairs,
        "same_row_first_noise_mismatches": first_noise_mismatches,
        # Keep the old aggregate key for readers created before AS was audited.
        "same_row_max_abs_route0_difference": max(max_as_route_diff, max_hb_route_diff),
        "same_row_max_abs_as_route0_difference": max_as_route_diff,
        "same_row_max_abs_hb_route0_difference": max_hb_route_diff,
        "same_row_max_abs_action0_difference": max_action_diff,
        "same_future_tensors_compared": compared_future_tensors,
        "same_column_future_noise_mismatches": future_noise_mismatches,
        "valid": (
            len(observation_hashes) == 1
            and first_noise_mismatches == 0
            and future_noise_mismatches == 0
            and max_as_route_diff == 0.0
            and max_hb_route_diff == 0.0
            and max_action_diff == 0.0
        ),
    }


def _grid(cells: list[Cell], n_first: int, n_future: int):
    by_key = {}
    for cell in cells:
        key = (int(cell.summary["grid_row"]), int(cell.summary["grid_col"]))
        if key in by_key:
            raise RuntimeError("duplicate grid cell %s" % (key,))
        by_key[key] = cell
    complete = len(by_key) == n_first * n_future
    if complete:
        outcomes = np.empty((n_first, n_future), dtype=np.float64)
        for row in range(n_first):
            for col in range(n_future):
                outcomes[row, col] = float(by_key[(row, col)].summary["success"])
    else:
        outcomes = None
    return by_key, outcomes


def _write_report(path: pathlib.Path, analysis: dict, outcomes: np.ndarray | None) -> None:
    integrity = analysis["integrity"]
    lines = [
        "# Early MoE Route Commitment Grid",
        "",
        "## Question",
        "",
        "Fix the complete control-step-0 routing trajectory, vary only future flow noise, and ask whether the final outcome changes.",
        "",
        "## Integrity",
        "",
        "- Cells captured: `%d`" % integrity["n_cells"],
        "- Initial observation hashes: `%d`" % integrity["n_initial_observation_hashes"],
        "- Same AS-route maximum absolute difference: `%.9g`"
        % integrity["same_row_max_abs_as_route0_difference"],
        "- Same HB-route maximum absolute difference: `%.9g`"
        % integrity["same_row_max_abs_hb_route0_difference"],
        "- Same first-action maximum absolute difference: `%.9g`" % integrity["same_row_max_abs_action0_difference"],
        "- Future-noise mismatches: `%d / %d`"
        % (integrity["same_column_future_noise_mismatches"], integrity["same_future_tensors_compared"]),
        "- Integrity valid: `%s`" % integrity["valid"],
        "",
    ]
    if outcomes is None:
        lines.extend(["The grid is incomplete; commitment statistics were not computed.", ""])
    else:
        stats = analysis["commitment"]
        lines.extend(
            [
                "## Outcome Matrix",
                "",
                "Rows fix the first route; columns fix the common future-noise stream. `S` is success and `F` is failure.",
                "",
                "```text",
            ]
        )
        for row, values in enumerate(outcomes.astype(int)):
            pattern = " ".join("S" if value else "F" for value in values)
            lines.append("route %02d: %s  p=%.3f" % (row, pattern, values.mean()))
        lines.extend(
            [
                "```",
                "",
                "## Primary Result",
                "",
                "- Mixed rows: `%d / %d`" % (stats["mixed_rows"], stats["n_first_routes"]),
                "- All-failure rows: `%d`" % stats["all_failure_rows"],
                "- All-success rows: `%d`" % stats["all_success_rows"],
                "- Conditional entropy H(outcome | first route): `%.4f bits`"
                % stats["conditional_entropy_bits"],
                "- Plug-in mutual information: `%.4f bits`" % stats["route_outcome_mutual_information_bits_plugin"],
                "- Permutation p (stable rows): `%.6f`" % stats["permutation"]["stable_rows_upper_p"],
                "- Permutation p (conditional entropy): `%.6f`"
                % stats["permutation"]["conditional_entropy_lower_p"],
                "- Permutation p (common-future column variance): `%.6f`"
                % stats["future_stream_effect"]["permutation"]["column_variance_upper_p"],
                "",
            ]
        )
        if stats["mixed_rows"]:
            lines.append(
                "At least one identical early route produced both outcomes under different future noise. The early route alone therefore did not determine the final outcome in this grid."
            )
        else:
            lines.append(
                "Every tested early route kept one outcome across the sampled future streams. This supports early commitment over the tested streams, but does not prove invariance to all possible future noise."
            )
        lines.append("")
    path.write_text("\n".join(lines))


def _plot(path: pathlib.Path, outcomes: np.ndarray, analysis: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    stats = analysis["commitment"]
    signal = analysis.get("route_signal", {})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)

    axes[0].imshow(outcomes, aspect="auto", vmin=0, vmax=1, cmap=ListedColormap(["#d75a4a", "#2f8f62"]))
    axes[0].set_xlabel("future-noise column")
    axes[0].set_ylabel("fixed first-route row")
    axes[0].set_title("Outcome matrix (red=fail, green=success)")
    axes[0].set_xticks(np.arange(outcomes.shape[1]))
    axes[0].set_yticks(np.arange(outcomes.shape[0]))

    row_rates = np.asarray(stats["row_success_rates"])
    axes[1].bar(np.arange(len(row_rates)), row_rates, color="#4b70a7")
    axes[1].axhline(stats["overall_success_rate"], color="black", ls="--", lw=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("fixed first-route row")
    axes[1].set_ylabel("P(success | first route)")
    axes[1].set_title("Sensitivity to future noise")

    if signal.get("available"):
        entries = signal["per_denoise"]
        axes[2].plot(
            [entry["denoise"] for entry in entries],
            [entry["correlation"] for entry in entries],
            marker="o",
            color="#7c4f9e",
            label="route kNN",
        )
        axes[2].axhline(signal["first_noise_baseline"]["correlation"], color="#777777", ls="--", label="noise")
        axes[2].axhline(signal["first_action_chunk_baseline"]["correlation"], color="#d28b26", ls=":", label="action")
        axes[2].set_xlabel("denoising round")
        axes[2].set_ylabel("LOO correlation with row success rate")
        axes[2].legend(frameon=False, fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "route signal unavailable", ha="center", va="center")
        axes[2].set_axis_off()
    axes[2].set_title("Exploratory route signal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    server_dir = pathlib.Path(args.server_dir)
    client_dir = pathlib.Path(args.client_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads((client_dir / "experiment_config.json").read_text())
    n_first = int(config["n_first"])
    n_future = int(config["n_future"])
    cells, route_attrs = _load_cells(server_dir, client_dir)
    integrity = _integrity(cells)
    by_key, outcomes = _grid(cells, n_first, n_future)

    if not integrity["valid"]:
        raise RuntimeError("paired-grid integrity check failed: %s" % integrity)
    if outcomes is None and not args.allow_partial:
        raise RuntimeError("grid is incomplete; pass --allow-partial for integrity-only analysis")

    analysis = {
        "protocol": "early-route-commitment-analysis-v1",
        "experiment_config": config,
        "route_store_attrs": route_attrs,
        "integrity": integrity,
        "complete": outcomes is not None,
    }

    if outcomes is not None:
        commitment = _commitment_statistics(outcomes, args.permutations, args.seed)
        analysis["commitment"] = commitment

        routes = np.stack(
            [np.mean([by_key[(row, col)].route0 for col in range(n_future)], axis=0) for row in range(n_first)]
        )
        first_noises = np.stack([by_key[(row, 0)].flow_noises[0] for row in range(n_first)])
        first_actions = np.stack([by_key[(row, 0)].action0 for row in range(n_first)])
        analysis["route_signal"] = _route_signal(
            routes,
            first_noises,
            first_actions,
            np.asarray(commitment["row_success_rates"]),
            args.permutations,
            args.seed + 1,
        )

    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True))
    _write_report(out / "REPORT.md", analysis, outcomes)
    if outcomes is not None:
        _plot(out / "commitment_grid.png", outcomes, analysis)

    print(json.dumps({
        "complete": analysis["complete"],
        "integrity": integrity,
        "commitment": analysis.get("commitment"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
