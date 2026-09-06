#!/usr/bin/env python3
"""How much risk information does the geometry carry, once survival is removed?

The alarm machinery in ``detect_deformation.py`` answers a decision question and
is heavily constrained by its own false-alarm cap.  This script answers the
prior scientific question directly:

    among the rollouts that are still running at chunk q, does the token
    geometry at chunk q separate the ones that will miss the horizon cap?

Conditioning on "still running at q" removes the survival base rate exactly,
because every episode in the comparison has already survived to q.  Three
stratifications are reported:

    pooled        all suites together; can be inflated by suite composition
    within_suite  AUC computed inside each suite, then pooled by pair count
    within_task   AUC computed inside each task; no task-side information can
                  contribute, so this is the pure MoE-side effect

Labels are used only to evaluate, never to build a score.  ``mobility`` from the
moe-v4-0904 cache is carried along as the anchor, since the published best
global detector is built on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import geometry as G

sys.path.insert(0, str(G.PROJECT / "moe-v4-0904/experiments"))
import evaluate_layerwise_alarm_development as dev  # noqa: E402


DEFAULT_OUTPUT = G.BUNDLE / "results/effect"
FEATURES = G.FEATURE_ROOT
LABEL_ROOT = G.PROJECT / "double-selete/trainfree/results/timeout_extension_plus10"
LABEL_PATHS = {
    "development_main": LABEL_ROOT / "development_main_clean_labels.csv",
    "external_8b": LABEL_ROOT / "external_8b_clean_labels.csv",
}
MOBILITY_PATHS = {
    "development_main": G.PROJECT / "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
    "external_8b": G.PROJECT / "moe-v4-0904/results/layerwise_mobility/external_8b.npz",
}
DYNAMIC_NAMES = ("procrustes_prev", "procrustes_self")
QUANTITIES = (
    "shape_pr", "shape_erank", "lam1_share", "lam12_share", "bandedness", "d_cv",
    "size", "kernel_trace", "centroid_norm2", "kernel_erank",
    "procrustes_layer", "procrustes_prev", "procrustes_self", "mobility",
)
CHUNKS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32)
WIDTH = 4
MIN_GROUP = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def dense_quantity(cohort: str, quantity: str, valid: np.ndarray) -> np.ndarray:
    if quantity == "mobility":
        with np.load(MOBILITY_PATHS[cohort], allow_pickle=False) as archive:
            values = np.asarray(archive["mobility"], dtype=np.float32).copy()
        values[~valid] = np.nan
        return values
    if quantity in DYNAMIC_NAMES:
        packed = np.load(FEATURES / f"{cohort}_dynamic.npy")
        column = DYNAMIC_NAMES.index(quantity)
    else:
        packed = np.load(FEATURES / f"{cohort}_features.npy")
        column = G.FEATURE_NAMES.index(quantity)
    return G.expand(packed[:, :, column], valid)


def auc(scores: np.ndarray, positive: np.ndarray) -> tuple[float, int]:
    """Mann-Whitney AUC with average ranks; returns (auc, discordant pairs)."""
    good = np.isfinite(scores)
    scores, positive = scores[good], positive[good]
    n_positive = int(positive.sum())
    n_negative = int(len(positive) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan"), 0
    ranks = rankdata(scores, method="average")
    statistic = ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0
    return float(statistic / (n_positive * n_negative)), n_positive * n_negative


def stratified(
    scores: np.ndarray, positive: np.ndarray, strata: np.ndarray
) -> tuple[float, int]:
    total, weight = 0.0, 0
    for level in np.unique(strata):
        take = strata == level
        if int(take.sum()) < MIN_GROUP:
            continue
        value, pairs = auc(scores[take], positive[take])
        if pairs == 0 or not np.isfinite(value):
            continue
        total += value * pairs
        weight += pairs
    return (total / weight if weight else float("nan")), weight


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for cohort in ("development_main", "external_8b"):
        index = G.load_index(cohort)
        valid = index["valid"].astype(bool)
        length = index["length"].astype(int)
        labels = pd.read_csv(LABEL_PATHS[cohort])
        task = index["task_names"].astype(str)[index["task_index"].astype(int)]
        if not np.array_equal(labels["task"].astype(str).to_numpy(), task):
            raise ValueError(f"{cohort}: labels are not aligned with the kernel index")
        risk = labels["original_failure"].to_numpy(bool)
        suite = np.asarray([name.split("/", 1)[0] for name in task])
        task_index = index["task_index"].astype(int)

        for quantity in QUANTITIES:
            values = dense_quantity(cohort, quantity, valid)
            for layer in range(8):
                causal = dev.trailing_mean(values[:, :, layer], WIDTH)
                for chunk in CHUNKS:
                    if chunk >= causal.shape[1]:
                        continue
                    alive = length > chunk
                    score = causal[alive, chunk]
                    positive = risk[alive]
                    if positive.sum() < MIN_GROUP or (~positive).sum() < MIN_GROUP:
                        continue
                    pooled, _ = auc(score, positive)
                    by_suite, _ = stratified(score, positive, suite[alive])
                    by_task, _ = stratified(score, positive, task_index[alive])
                    rows.append(
                        {
                            "cohort": cohort,
                            "quantity": quantity,
                            "layer": G.LAYER_NAMES[layer],
                            "group": "front" if layer < 4 else "back",
                            "chunk": chunk,
                            "alive": int(alive.sum()),
                            "alive_risk": int(positive.sum()),
                            "prior": float(positive.mean()),
                            "auc_pooled": pooled,
                            "auc_within_suite": by_suite,
                            "auc_within_task": by_task,
                        }
                    )
            print(f"  {cohort} {quantity}: done", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "survival_conditioned_auc.csv", index=False)

    table["deviation"] = (table["auc_within_task"] - 0.5).abs()
    best = (
        table.sort_values("deviation", ascending=False, kind="stable")
        .groupby(["cohort", "quantity"], as_index=False)
        .first()
    )
    best.to_csv(args.output / "best_within_task_auc.csv", index=False)

    (args.output / "effect.json").write_text(
        json.dumps(
            {
                "schema": "himoe.token_geometry.effect.v1",
                "conditioning": "episodes still running at chunk q",
                "causal_input": f"trailing mean of width {WIDTH} ending at q",
                "labels_used_for": "evaluation only",
                "min_stratum_size": MIN_GROUP,
                "chunks": list(CHUNKS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 220)
    for cohort in ("development_main", "external_8b"):
        block = best[best["cohort"] == cohort].sort_values("deviation", ascending=False)
        print(f"\n=== {cohort}: strongest survival-conditioned AUC per quantity ===")
        print(
            block[
                ["quantity", "layer", "chunk", "alive", "alive_risk",
                 "auc_pooled", "auc_within_suite", "auc_within_task"]
            ].to_string(index=False, float_format="%.4f")
        )


if __name__ == "__main__":
    main()
