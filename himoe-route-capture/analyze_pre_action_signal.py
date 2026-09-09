"""Pre-registered analysis of pre-action MoE denoising signals.

The independent sample is the first-noise / first-route row.  Each row's target
is its success rate over common future-noise continuations.  No feature uses a
control step after the first policy inference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib

import numpy as np

from analyze_commitment_grid import (
    _commitment_statistics,
    _integrity,
    _knn_weight_matrix,
    _load_cells,
    _safe_corr,
    _signal_score,
)


PRIMARY_DENOISE = 6
N_NEIGHBORS = 5
EXPECTED_GPU = "MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a"


def _row_arrays(cells, n_rows: int, n_cols: int):
    by_key = {}
    for cell in cells:
        key = (int(cell.summary["grid_row"]), int(cell.summary["grid_col"]))
        if key in by_key:
            raise RuntimeError("duplicate grid cell %s" % (key,))
        by_key[key] = cell
    if len(by_key) != n_rows * n_cols:
        raise RuntimeError("outcome grid is incomplete")

    outcomes = np.empty((n_rows, n_cols), dtype=np.float64)
    routes = []
    as_probs = []
    first_noises = []
    first_actions = []
    for row in range(n_rows):
        reference = by_key[(row, 0)]
        routes.append(reference.route0)
        as_probs.append(reference.as_probs0)
        first_noises.append(reference.flow_noises[0])
        first_actions.append(reference.action0)
        for col in range(n_cols):
            outcomes[row, col] = float(by_key[(row, col)].summary["success"])
    return (
        outcomes,
        np.stack(routes),
        np.stack(as_probs),
        np.stack(first_noises),
        np.stack(first_actions),
    )


def _feature_families(routes: np.ndarray) -> tuple[list[dict[str, np.ndarray]], dict]:
    """Build the frozen cumulative Z_j representation and its components."""
    routes = np.asarray(routes, dtype=np.float64)
    if routes.ndim != 5:
        raise ValueError("expected routes [row, layer, denoise, token, expert]")
    n_rows, n_layers, n_denoise, n_tokens, n_experts = routes.shape
    if n_tokens < 2:
        raise ValueError("route tensor has no action tokens")

    probabilities = routes[:, :, :, 1:, :]
    top1 = np.argmax(probabilities, axis=-1)
    rounds = []
    for denoise in range(n_denoise):
        prefix = probabilities[:, :, : denoise + 1]
        prefix_top1 = top1[:, :, : denoise + 1]

        probability = prefix.mean(axis=(2, 3)).reshape(n_rows, -1)
        entropy = (
            -np.sum(prefix * np.log(np.maximum(prefix, 1e-12)), axis=-1) / math.log(n_experts)
        ).mean(axis=(2, 3))
        ordered = np.sort(prefix, axis=-1)
        margin = (ordered[..., -1] - ordered[..., -2]).mean(axis=(2, 3))

        one_hot = prefix_top1[..., None] == np.arange(n_experts)
        token_fractions = one_hot.mean(axis=3)
        token_consensus = token_fractions.max(axis=-1).mean(axis=2)

        if denoise == 0:
            churn = np.zeros((n_rows, n_layers), dtype=np.float64)
        else:
            churn = np.mean(
                prefix_top1[:, :, 1:] != prefix_top1[:, :, :-1], axis=(2, 3)
            )

        displacement = 0.5 * np.sum(
            np.abs(probabilities[:, :, denoise] - probabilities[:, :, 0]), axis=-1
        ).mean(axis=2)

        families = {
            "probability": probability,
            "entropy": entropy,
            "margin": margin,
            "token_consensus": token_consensus,
            "churn": churn,
            "displacement": displacement,
        }
        combined = np.concatenate([families[name] for name in families], axis=1)
        rounds.append({"combined": combined, **families})

    manifest = {
        "n_rows": n_rows,
        "n_layers": n_layers,
        "n_denoise": n_denoise,
        "n_suffix_tokens": n_tokens,
        "n_action_tokens": n_tokens - 1,
        "n_experts": n_experts,
        "combined_dimensions": int(rounds[0]["combined"].shape[1]),
        "component_dimensions": {
            name: int(rounds[0][name].shape[1])
            for name in (
                "probability",
                "entropy",
                "margin",
                "token_consensus",
                "churn",
                "displacement",
            )
        },
    }
    return rounds, manifest


def _rowwise_correlation(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    left = prediction - prediction.mean(axis=1, keepdims=True)
    right = target - target.mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-15,
    )


def _rowwise_skill(weights: np.ndarray, targets: np.ndarray) -> np.ndarray:
    predictions = targets @ weights.T
    baselines = (targets.sum(axis=1, keepdims=True) - targets) / (targets.shape[1] - 1)
    mse = np.mean((predictions - targets) ** 2, axis=1)
    baseline_mse = np.mean((baselines - targets) ** 2, axis=1)
    return np.divide(
        baseline_mse - mse,
        baseline_mse,
        out=np.zeros_like(mse),
        where=baseline_mse > 1e-15,
    )


def _permuted_targets(
    outcomes: np.ndarray, n_permutations: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_rows, n_cols = outcomes.shape
    targets = np.empty((n_permutations, n_rows), dtype=np.float64)
    permuted = np.empty_like(outcomes)
    for iteration in range(n_permutations):
        for col in range(n_cols):
            permuted[:, col] = rng.permutation(outcomes[:, col])
        targets[iteration] = permuted.mean(axis=1)
    return targets


def _upper_p(null: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null >= observed - 1e-15)) / (len(null) + 1))


def _score_with_null(
    weights: np.ndarray, target: np.ndarray, permuted_targets: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray]:
    score = _signal_score(weights, target)
    null_prediction = permuted_targets @ weights.T
    null_correlation = _rowwise_correlation(null_prediction, permuted_targets)
    null_skill = _rowwise_skill(weights, permuted_targets)
    score["permutation_p_correlation"] = _upper_p(null_correlation, score["correlation"])
    score["permutation_p_mse_skill"] = _upper_p(null_skill, score["mse_skill"])
    return score, null_correlation, null_skill


def _signal_analysis(
    feature_rounds: list[dict[str, np.ndarray]],
    outcomes: np.ndarray,
    first_noises: np.ndarray,
    first_actions: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict:
    target = outcomes.mean(axis=1)
    if len(feature_rounds) <= PRIMARY_DENOISE:
        raise RuntimeError("route tensor does not contain pre-registered denoise round 6")

    combined_weights = [
        _knn_weight_matrix(features["combined"], k=N_NEIGHBORS)
        for features in feature_rounds
    ]
    combined_scores = [_signal_score(weights, target) for weights in combined_weights]

    component_names = [
        "probability",
        "entropy",
        "margin",
        "token_consensus",
        "churn",
        "displacement",
    ]
    component_weights = {
        name: _knn_weight_matrix(
            feature_rounds[PRIMARY_DENOISE][name], k=N_NEIGHBORS
        )
        for name in component_names
    }
    component_scores = {
        name: _signal_score(weights, target) for name, weights in component_weights.items()
    }

    noise_weights = _knn_weight_matrix(first_noises.reshape(len(target), -1), k=N_NEIGHBORS)
    action_weights = _knn_weight_matrix(first_actions.reshape(len(target), -1), k=N_NEIGHBORS)

    permuted_targets = _permuted_targets(outcomes, n_permutations, seed)
    null_combined = np.empty((n_permutations, len(feature_rounds)), dtype=np.float64)
    null_combined_skill = np.empty_like(null_combined)
    for denoise, weights in enumerate(combined_weights):
        prediction = permuted_targets @ weights.T
        null_combined[:, denoise] = _rowwise_correlation(prediction, permuted_targets)
        null_combined_skill[:, denoise] = _rowwise_skill(weights, permuted_targets)

    null_round_max = null_combined.max(axis=1)
    null_component = np.empty((n_permutations, len(component_names)), dtype=np.float64)
    for index, name in enumerate(component_names):
        prediction = permuted_targets @ component_weights[name].T
        null_component[:, index] = _rowwise_correlation(prediction, permuted_targets)
    null_component_max = null_component.max(axis=1)

    for denoise, score in enumerate(combined_scores):
        score["denoise"] = denoise
        score["permutation_p_uncorrected"] = _upper_p(
            null_combined[:, denoise], score["correlation"]
        )
        score["permutation_p_fwer_10_denoise"] = _upper_p(
            null_round_max, score["correlation"]
        )
        score["permutation_p_mse_skill_uncorrected"] = _upper_p(
            null_combined_skill[:, denoise], score["mse_skill"]
        )

    for index, name in enumerate(component_names):
        score = component_scores[name]
        score["permutation_p_uncorrected"] = _upper_p(
            null_component[:, index], score["correlation"]
        )
        score["permutation_p_fwer_6_components"] = _upper_p(
            null_component_max, score["correlation"]
        )

    noise_score, _null_noise, _ = _score_with_null(
        noise_weights, target, permuted_targets
    )
    action_score, _null_action, _ = _score_with_null(
        action_weights, target, permuted_targets
    )

    primary = dict(combined_scores[PRIMARY_DENOISE])
    primary["pre_registered"] = True
    primary["denoise"] = PRIMARY_DENOISE
    primary["permutation_p_primary"] = primary["permutation_p_uncorrected"]

    return {
        "target": "empirical continuation success probability over 8 common future streams",
        "method": "distance-weighted 5-NN leave-one-initial-route-out regression",
        "primary": primary,
        "combined_by_denoise": combined_scores,
        "components_at_denoise_6": component_scores,
        "controls": {
            "first_flow_noise": noise_score,
            "first_action_chunk": action_score,
        },
        "null": {
            "method": "independent row permutation within each common-future column",
            "n_permutations": n_permutations,
            "round_max_correlation_p95": float(np.quantile(null_round_max, 0.95)),
            "component_max_correlation_p95": float(
                np.quantile(null_component_max, 0.95)
            ),
        },
    }


def _target_reliability(outcomes: np.ndarray) -> dict:
    n_cols = outcomes.shape[1]
    if n_cols != 8:
        raise ValueError("pre-registered reliability analysis requires 8 future columns")
    correlations = []
    corrected = []
    for remainder in itertools.combinations(range(1, n_cols), 3):
        left_columns = (0,) + remainder
        right_columns = tuple(col for col in range(n_cols) if col not in left_columns)
        left = outcomes[:, left_columns].mean(axis=1)
        right = outcomes[:, right_columns].mean(axis=1)
        correlation = _safe_corr(left, right)
        correlations.append(correlation)
        corrected.append(
            2 * correlation / (1 + correlation) if correlation > -1 + 1e-12 else -1.0
        )
    return {
        "method": "all 35 unique 4-vs-4 common-future column splits",
        "n_splits": len(correlations),
        "half_grid_correlations": correlations,
        "half_grid_correlation_median": float(np.median(correlations)),
        "half_grid_correlation_mean": float(np.mean(correlations)),
        "spearman_brown_median": float(np.median(corrected)),
        "spearman_brown_mean": float(np.mean(corrected)),
    }


def _write_report(path: pathlib.Path, analysis: dict) -> None:
    commitment = analysis["commitment"]
    signal = analysis["signal"]
    primary = signal["primary"]
    lines = [
        "# Pre-action MoE denoising signal",
        "",
        "## Paired target",
        "",
        "- Initial-route rows: `%d`" % commitment["n_first_routes"],
        "- Common future streams per row: `%d`" % commitment["n_future_streams"],
        "- Overall success: `%.4f`" % commitment["overall_success_rate"],
        "- Mixed rows: `%d / %d`"
        % (commitment["mixed_rows"], commitment["n_first_routes"]),
        "- Target split-half median: `%.4f`"
        % analysis["target_reliability"]["half_grid_correlation_median"],
        "- Spearman-Brown median: `%.4f`"
        % analysis["target_reliability"]["spearman_brown_median"],
        "",
        "## Pre-registered primary result",
        "",
        "- Feature: cumulative `Z_6` before the first action",
        "- LOO correlation: `%.4f`" % primary["correlation"],
        "- MSE skill: `%.4f`" % primary["mse_skill"],
        "- Column-preserving permutation p: `%.6f`"
        % primary["permutation_p_primary"],
        "",
        "## Denoising curve",
        "",
        "| d | correlation | MSE skill | p raw | p FWER-10 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for entry in signal["combined_by_denoise"]:
        lines.append(
            "| %d | %.4f | %.4f | %.6f | %.6f |"
            % (
                entry["denoise"],
                entry["correlation"],
                entry["mse_skill"],
                entry["permutation_p_uncorrected"],
                entry["permutation_p_fwer_10_denoise"],
            )
        )
    lines.extend(
        [
            "",
            "## Components at d=6",
            "",
            "| feature | correlation | p raw | p FWER-6 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, entry in signal["components_at_denoise_6"].items():
        lines.append(
            "| %s | %.4f | %.6f | %.6f |"
            % (
                name,
                entry["correlation"],
                entry["permutation_p_uncorrected"],
                entry["permutation_p_fwer_6_components"],
            )
        )
    lines.extend(["", "## Controls", ""])
    for name, entry in signal["controls"].items():
        lines.append(
            "- `%s`: r=`%.4f`, skill=`%.4f`, p=`%.6f`"
            % (
                name,
                entry["correlation"],
                entry["mse_skill"],
                entry["permutation_p_correlation"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines))


def _plot(path: pathlib.Path, analysis: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    outcomes = np.asarray(analysis["outcomes"])
    signal = analysis["signal"]
    rounds = signal["combined_by_denoise"]
    target = outcomes.mean(axis=1)
    primary_prediction = np.asarray(signal["primary"]["prediction"])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    axes[0, 0].imshow(
        outcomes,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap=ListedColormap(["#d75a4a", "#2f8f62"]),
    )
    axes[0, 0].set_xlabel("common future-noise stream")
    axes[0, 0].set_ylabel("fixed pre-action route")
    axes[0, 0].set_title("Continuation outcomes")

    denoise = [entry["denoise"] for entry in rounds]
    correlation = [entry["correlation"] for entry in rounds]
    axes[0, 1].plot(denoise, correlation, marker="o", color="#4b70a7")
    axes[0, 1].axvline(PRIMARY_DENOISE, color="#d28b26", ls="--", lw=1)
    axes[0, 1].axhline(
        signal["null"]["round_max_correlation_p95"], color="#777777", ls=":", lw=1
    )
    axes[0, 1].set_xlabel("internal denoising round d")
    axes[0, 1].set_ylabel("LOO correlation with continuation success")
    axes[0, 1].set_title("Cumulative pre-action signal Z_d")

    axes[1, 0].scatter(target, primary_prediction, color="#7c4f9e", alpha=0.8)
    axes[1, 0].plot([0, 1], [0, 1], color="#777777", ls=":", lw=1)
    axes[1, 0].set_xlim(-0.03, 1.03)
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_xlabel("observed continuation success rate")
    axes[1, 0].set_ylabel("LOO Z_6 prediction")
    axes[1, 0].set_title("Pre-registered held-out prediction")

    names = list(signal["components_at_denoise_6"])
    names.extend(["flow_noise", "action_chunk"])
    values = [
        signal["components_at_denoise_6"][name]["correlation"]
        for name in signal["components_at_denoise_6"]
    ]
    values.extend(
        [
            signal["controls"]["first_flow_noise"]["correlation"],
            signal["controls"]["first_action_chunk"]["correlation"],
        ]
    )
    axes[1, 1].bar(np.arange(len(names)), values, color="#4f8a6f")
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    axes[1, 1].set_ylabel("LOO correlation")
    axes[1, 1].set_title("Components and controls")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    server_dir = pathlib.Path(args.server_dir)
    client_dir = pathlib.Path(args.client_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads((client_dir / "experiment_config.json").read_text())
    metadata = json.loads((client_dir / "server_metadata.json").read_text())
    expected_config = {
        "n_first": 64,
        "n_future": 8,
        "first_seed_base": 3000,
        "future_seed_base": 10000,
        "task_id": 0,
        "init_state_id": 24,
        "environment_seed": 7,
        "settle_steps": 10,
        "benchmark": "libero_goal",
        "replan_steps": 10,
        "max_steps": 300,
    }
    mismatched = {
        key: (config.get(key), value)
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatched:
        raise RuntimeError("run does not match frozen protocol: %s" % mismatched)
    if metadata.get("cuda_visible_devices") != EXPECTED_GPU:
        raise RuntimeError(
            "run used %r, expected frozen GPU %r"
            % (metadata.get("cuda_visible_devices"), EXPECTED_GPU)
        )
    expected_metadata = {
        "checkpoint_sha256": "98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953",
        "libero_wrist_layout": "released-left",
        "flow_steps": 10,
    }
    metadata_mismatch = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if metadata_mismatch:
        raise RuntimeError("server metadata violates frozen protocol: %s" % metadata_mismatch)

    cells, route_attrs = _load_cells(server_dir, client_dir)
    integrity = _integrity(cells)
    if not integrity["valid"]:
        raise RuntimeError("paired-grid integrity failed: %s" % integrity)

    outcomes, routes, as_probs, first_noises, first_actions = _row_arrays(
        cells, 64, 8
    )
    feature_rounds, feature_manifest = _feature_families(routes)
    commitment = _commitment_statistics(outcomes, args.permutations, args.seed)
    signal = _signal_analysis(
        feature_rounds,
        outcomes,
        first_noises,
        first_actions,
        args.permutations,
        args.seed + 1,
    )
    reliability = _target_reliability(outcomes)

    analysis = {
        "protocol": "pre-action-denoise-signal-v1",
        "protocol_document": "PRE_ACTION_DENOISE_PROTOCOL.md",
        "experiment_config": config,
        "server_metadata": {
            "cuda_visible_devices": metadata.get("cuda_visible_devices"),
            "gpu": metadata.get("gpu"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "libero_wrist_layout": metadata.get("libero_wrist_layout"),
        },
        "route_store_attrs": route_attrs,
        "integrity": integrity,
        "feature_manifest": feature_manifest,
        "as_route_cross_row_max_std": float(np.max(np.std(as_probs, axis=0))),
        "outcomes": outcomes.astype(int).tolist(),
        "commitment": commitment,
        "target_reliability": reliability,
        "signal": signal,
    }

    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True))
    _write_report(out / "REPORT.md", analysis)
    _plot(out / "pre_action_signal.png", analysis)
    print(
        json.dumps(
            {
                "integrity": integrity,
                "commitment": {
                    "success_rate": commitment["overall_success_rate"],
                    "mixed_rows": commitment["mixed_rows"],
                    "stable_rows": commitment["stable_rows"],
                },
                "target_reliability": reliability,
                "primary": signal["primary"],
                "controls": signal["controls"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
