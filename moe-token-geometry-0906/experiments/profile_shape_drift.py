#!/usr/bin/env python3
"""Fix the layer on development, then read the deformation profile on external.

``stratified_effect.py`` reports every layer at every chunk, so its per-chunk
best is a maximum over eight layers and is optimistic.  Here each quantity gets
one layer, chosen on development by mean |AUC - 0.5| over chunks 8..24, and the
external profile at that single layer is reported without further choices.

Two redundancy checks accompany it: how far the shape-spectrum signal overlaps
the published mobility signal at the same chunk inside the same task, and which
suites are still alive at each chunk, because a late chunk is only one suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import geometry as G
from stratified_effect import (
    CHUNKS,
    LABEL_PATHS,
    MIN_GROUP,
    WIDTH,
    dense_quantity,
    dev,
)


DEFAULT_OUTPUT = G.BUNDLE / "results/effect"
SELECTION_CHUNKS = (8, 10, 12, 14, 16, 18, 20, 24)
PAIRED = ("shape_pr", "shape_erank", "lam1_share", "bandedness", "procrustes_layer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.output / "survival_conditioned_auc.csv")

    development = table[
        (table["cohort"] == "development_main") & table["chunk"].isin(SELECTION_CHUNKS)
    ].copy()
    development["deviation"] = (development["auc_within_task"] - 0.5).abs()
    picked = (
        development.groupby(["quantity", "layer"], as_index=False)["deviation"].mean()
        .sort_values("deviation", ascending=False, kind="stable")
        .groupby("quantity", as_index=False)
        .first()
        .rename(columns={"deviation": "development_mean_deviation"})
    )
    profile = table.merge(picked[["quantity", "layer"]], on=["quantity", "layer"])
    profile = profile.merge(picked, on=["quantity", "layer"])
    profile.to_csv(args.output / "fixed_layer_profile.csv", index=False)

    # suite composition at each chunk
    composition = []
    for cohort in ("development_main", "external_8b"):
        index = G.load_index(cohort)
        task = index["task_names"].astype(str)[index["task_index"].astype(int)]
        suite = np.asarray([name.split("/", 1)[0] for name in task])
        length = index["length"].astype(int)
        for chunk in CHUNKS:
            alive = length > chunk
            record = {"cohort": cohort, "chunk": chunk, "alive": int(alive.sum())}
            for name in np.unique(suite):
                record[name] = int((alive & (suite == name)).sum())
            composition.append(record)
    pd.DataFrame(composition).to_csv(args.output / "alive_composition.csv", index=False)

    # redundancy against the published mobility signal, same chunk, same task
    redundancy = []
    for cohort in ("development_main", "external_8b"):
        index = G.load_index(cohort)
        valid = index["valid"].astype(bool)
        length = index["length"].astype(int)
        task_index = index["task_index"].astype(int)
        mobility = dense_quantity(cohort, "mobility", valid)
        for quantity in PAIRED:
            layer_name = picked.loc[picked["quantity"] == quantity, "layer"].iloc[0]
            layer = G.LAYER_NAMES.index(layer_name)
            mobility_layer = dev.trailing_mean(mobility[:, :, layer], WIDTH)
            values = dev.trailing_mean(
                dense_quantity(cohort, quantity, valid)[:, :, layer], WIDTH
            )
            for chunk in CHUNKS:
                alive = length > chunk
                if alive.sum() < MIN_GROUP:
                    continue
                total, weight = 0.0, 0
                for level in np.unique(task_index[alive]):
                    take = alive & (task_index == level)
                    if int(take.sum()) < MIN_GROUP:
                        continue
                    a, b = values[take, chunk], mobility_layer[take, chunk]
                    good = np.isfinite(a) & np.isfinite(b)
                    if good.sum() < MIN_GROUP:
                        continue
                    rho = spearmanr(a[good], b[good]).statistic
                    if np.isfinite(rho):
                        total += rho * int(good.sum())
                        weight += int(good.sum())
                redundancy.append(
                    {
                        "cohort": cohort,
                        "quantity": quantity,
                        "layer": layer_name,
                        "chunk": chunk,
                        "within_task_spearman_vs_mobility": total / weight
                        if weight
                        else float("nan"),
                        "episodes": int(weight and alive.sum()),
                    }
                )
    pd.DataFrame(redundancy).to_csv(
        args.output / "redundancy_vs_mobility.csv", index=False
    )

    (args.output / "fixed_layer_selection.json").write_text(
        json.dumps(
            {
                "schema": "himoe.token_geometry.fixed_layer.v1",
                "layer_selected_on": "development_main",
                "selection_statistic": "mean |auc_within_task - 0.5| over chunks 8..24",
                "selection_chunks": list(SELECTION_CHUNKS),
                "layers": {
                    row["quantity"]: row["layer"] for _, row in picked.iterrows()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 240)
    for cohort in ("development_main", "external_8b"):
        block = profile[profile["cohort"] == cohort]
        wide = block.pivot(index=["quantity", "layer"], columns="chunk",
                           values="auc_within_task")
        print(f"\n=== {cohort}: within-task AUC at the development-selected layer ===")
        print(wide.to_string(float_format="%.3f"))
    print("\n=== within-task Spearman against mobility at the same layer/chunk ===")
    red = pd.DataFrame(redundancy)
    print(
        red.pivot_table(
            index=["cohort", "quantity"], columns="chunk",
            values="within_task_spearman_vs_mobility",
        ).to_string(float_format="%.3f")
    )


if __name__ == "__main__":
    main()
