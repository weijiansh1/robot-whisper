#!/usr/bin/env python3
"""Audit whether route momentum adds information beyond route speed."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analyze as core


HERE = Path(__file__).resolve().parent
STATS = (
    "speed",
    "active_fraction",
    "acceleration",
    "alignment",
    "speed_delta",
    "straightness",
    "ema05_magnitude",
    "ema08_magnitude",
)
SOURCES = ("hard_route", "soft_route", "selected_route")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    return parser.parse_args()


def auc(labels: np.ndarray, score: np.ndarray) -> float:
    ranks = rankdata(score, method="average")[:, None]
    return float(core.auc_from_ranks(ranks, labels)[0])


def residual_audit(
    target_key: str,
    source: str,
    horizon: int,
    window: int,
    keys: list[str],
    discovery: np.ndarray,
    confirmation: np.ndarray,
    discovery_labels: np.ndarray,
    confirmation_labels: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    key_index = {key: index for index, key in enumerate(keys)}
    target_index = key_index[target_key]
    control_keys = [f"{source}|h{horizon:02d}|W{window}|speed|action_all"]
    active_key = f"{source}|h{horizon:02d}|W{window}|active_fraction|action_all"
    if np.ptp(discovery[:, key_index[active_key]]) > 1e-12:
        control_keys.append(active_key)
    control_indexes = [key_index[key] for key in control_keys]
    residuals = []
    explained = []
    for matrix in (discovery, confirmation):
        target = matrix[:, target_index]
        design = np.column_stack(
            [np.ones(len(target)), matrix[:, control_indexes]]
        )
        fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - fitted
        residuals.append(residual)
        explained.append(
            float(
                1.0
                - np.square(residual).sum()
                / np.square(target - target.mean()).sum()
            )
        )
    scan = core.permutation_scan(
        residuals[0][:, None], discovery_labels, permutations, rng
    )
    discovery_auc = float(scan["auc"][0])
    direction = 1.0 if discovery_auc >= 0.5 else -1.0
    return {
        "target": target_key,
        "controls": control_keys,
        "variance_explained": {
            "discovery": explained[0],
            "confirmation": explained[1],
        },
        "residual_discovery_auc": discovery_auc,
        "residual_discovery_p": float(scan["p_unadjusted"][0]),
        "residual_confirmation": core.confirmation_test(
            residuals[1],
            confirmation_labels,
            direction,
            permutations,
            rng,
        ),
    }


def fit_classifier(
    feature_keys: list[str],
    key_index: dict[str, int],
    discovery: np.ndarray,
    confirmation: np.ndarray,
    discovery_labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    indexes = [key_index[key] for key in feature_keys]
    cv = StratifiedKFold(4, shuffle=True, random_state=20260905)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 6),
            cv=cv,
            scoring="roc_auc",
            max_iter=10_000,
            solver="liblinear",
            random_state=20260905,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(discovery[:, indexes], discovery_labels)
    score = model.predict_proba(confirmation[:, indexes])[:, 1]
    return score, float(model[-1].C_[0])


def paired_increment(
    baseline: np.ndarray,
    augmented: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    baseline_auc = auc(labels, baseline)
    augmented_auc = auc(labels, augmented)
    observed = augmented_auc - baseline_auc
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        null_delta = auc(permuted, augmented) - auc(permuted, baseline)
        exceed += int(null_delta >= observed - 1e-15)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    bootstrap_delta = np.empty(bootstrap, dtype=np.float64)
    for index in range(bootstrap):
        sample = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        bootstrap_delta[index] = auc(
            labels[sample], augmented[sample]
        ) - auc(labels[sample], baseline[sample])
    return {
        "baseline_auc": baseline_auc,
        "augmented_auc": augmented_auc,
        "auc_delta": observed,
        "permutation_p_delta": (exceed + 1.0) / (permutations + 1.0),
        "bootstrap_delta_95ci": np.quantile(
            bootstrap_delta, [0.025, 0.975]
        ).tolist(),
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Momentum incremental-information audit",
        "",
        "## Direction residuals after route speed",
        "",
        "| source | target | variance explained discovery / confirmation | residual confirmation AUC | p |",
        "|---|---|---:|---:|---:|",
    ]
    for item in summary["direction_residuals"]:
        lines.append(
            "| %s | `%s` | %.3f / %.3f | %.4f | %.6f |"
            % (
                item["source"],
                item["target"],
                item["variance_explained"]["discovery"],
                item["variance_explained"]["confirmation"],
                item["residual_confirmation"]["auc_discovery_oriented"],
                item["residual_confirmation"]["permutation_p_one_sided"],
            )
        )
    lines.extend(
        [
            "",
            "## Fixed-W7 cross-state classifiers",
            "",
            "Regularization is selected by four-fold CV in state 0; AUC is evaluated only in state 42.",
            "",
            "| horizon | model | dimensions | C | confirmation AUC | p | Bonferroni over 15 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["classifiers"]:
        lines.append(
            "| t%d | %s | %d | %g | %.4f | %.6f | %.6f |"
            % (
                item["horizon"],
                item["model"],
                item["dimensions"],
                item["regularization_C"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
                item["confirmation_p_bonferroni_15"],
            )
        )
    lines.extend(
        [
            "",
            "## All-momentum gain over hard-route speed",
            "",
            "| horizon | speed AUC | all-momentum AUC | delta | paired p | bootstrap delta 95% CI |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["paired_increments"]:
        low, high = item["bootstrap_delta_95ci"]
        lines.append(
            "| t%d | %.4f | %.4f | %.4f | %.6f | [%.4f, %.4f] |"
            % (
                item["horizon"],
                item["baseline_auc"],
                item["augmented_auc"],
                item["auc_delta"],
                item["permutation_p_delta"],
                low,
                high,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    results = args.results.resolve()
    scores = np.load(results / "momentum_scores.npz")
    momentum = json.loads((results / "momentum_summary.json").read_text())
    keys = scores["feature_keys"].tolist()
    key_index = {key: index for index, key in enumerate(keys)}
    discovery = scores["discovery_scores"]
    confirmation = scores["confirmation_scores"]
    discovery_labels = scores["discovery_failure"]
    confirmation_labels = scores["confirmation_failure"]
    rng = np.random.default_rng(args.seed)

    direction_residuals = []
    for lead in momentum["direction_leads"]:
        if lead["horizon"] != 34:
            continue
        item = residual_audit(
            lead["key"],
            lead["source"],
            lead["horizon"],
            lead["window"],
            keys,
            discovery,
            confirmation,
            discovery_labels,
            confirmation_labels,
            args.permutations,
            rng,
        )
        item["source"] = lead["source"]
        direction_residuals.append(item)

    classifiers = []
    prediction: dict[tuple[int, str], np.ndarray] = {}
    for horizon in core.HORIZONS:
        groups = {
            "hard_speed": [
                f"hard_route|h{horizon:02d}|W7|speed|action_all"
            ],
            "hard_momentum": [
                f"hard_route|h{horizon:02d}|W7|{stat}|action_all"
                for stat in STATS
            ],
            "soft_momentum": [
                f"soft_route|h{horizon:02d}|W7|{stat}|action_all"
                for stat in STATS
            ],
            "selected_momentum": [
                f"selected_route|h{horizon:02d}|W7|{stat}|action_all"
                for stat in STATS
            ],
            "all_momentum": [
                f"{source}|h{horizon:02d}|W7|{stat}|action_all"
                for source in SOURCES
                for stat in STATS
            ],
        }
        for name, feature_keys in groups.items():
            score, regularization = fit_classifier(
                feature_keys,
                key_index,
                discovery,
                confirmation,
                discovery_labels,
            )
            prediction[(horizon, name)] = score
            tested = core.confirmation_test(
                score,
                confirmation_labels,
                1.0,
                args.permutations,
                rng,
            )
            classifiers.append(
                {
                    "horizon": horizon,
                    "model": name,
                    "dimensions": len(feature_keys),
                    "regularization_C": regularization,
                    "confirmation": tested,
                    "confirmation_p_bonferroni_15": min(
                        1.0, 15.0 * tested["permutation_p_one_sided"]
                    ),
                }
            )

    paired_increments = []
    for horizon in core.HORIZONS:
        item = paired_increment(
            prediction[(horizon, "hard_speed")],
            prediction[(horizon, "all_momentum")],
            confirmation_labels,
            args.permutations,
            args.bootstrap,
            rng,
        )
        item["horizon"] = horizon
        paired_increments.append(item)

    summary = {
        "experiment": "momentum_increment_audit_v1",
        "protocol": {
            "training_state": momentum["integrity"]["discovery_state"],
            "confirmation_state": momentum["integrity"]["confirmation_state"],
            "window": 7,
            "permutations": args.permutations,
            "bootstrap": args.bootstrap,
        },
        "direction_residuals": direction_residuals,
        "classifiers": classifiers,
        "paired_increments": paired_increments,
    }
    (results / "momentum_increment_summary.json").write_text(
        json.dumps(core.finite_json(summary), indent=2, ensure_ascii=True) + "\n"
    )
    (results / "momentum_increment_report.md").write_text(
        render_report(summary)
    )
    print(f"wrote {results / 'momentum_increment_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
