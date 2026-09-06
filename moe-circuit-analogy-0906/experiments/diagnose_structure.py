#!/usr/bin/env python3
"""Label-free diagnostics of the routing graph and of feature redundancy.

Answers, with numbers and without ever touching an outcome label:
  1. How close is the action-token conductance network to a uniform clique?
  2. Does Foster's identity hold, and where does it fail?
  3. Which circuit quantities are new relative to the eleven already-published
     HB layer-graph metrics plus layerwise mobility?
  4. Front vs back layer contrast for each circuit quantity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import circuit_lib as cl


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
FEATURES = BUNDLE / "results/circuit_features"
GRAPHS = PROJECT / "moe-hb-front-back-0905/results/layer_graphs"
MOBILITY = PROJECT / "moe-v4-0904/results/layerwise_mobility"
MOBILITY_FILE = {
    "development_main": "main_reference.npz",
    "development_extra": "extra_reference.npz",
    "external_8b": "external_8b.npz",
}
DEFAULT_OUTPUT = BUNDLE / "results/diagnostics"
RNG_SEED = 20260906
SUBSAMPLE = 200_000  # rank correlations use this many (layer, query) cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort", default="development_main")
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    circuit = load(FEATURES / f"{args.cohort}.npz")
    graph = load(GRAPHS / f"{args.cohort}.npz")
    mob = load(MOBILITY / MOBILITY_FILE[args.cohort])

    names = list(circuit["feature_names"].astype(str))
    values = circuit["features"]            # [n_valid, 8, F]
    row = circuit["flat_row"].astype(int)
    query = circuit["flat_query"].astype(int)
    n_valid = len(row)

    # ---- 1. how close to a uniform clique -------------------------------
    structure = {}
    for key in ("gap_ratio", "spectral_erank", "kirchhoff_efficiency", "res_cv",
                "th_cv", "res_shortcut", "norm_fiedler", "n_near_zero",
                "dd_deficit", "ground_leak_mean", "vol"):
        v = values[:, :, names.index(key)].ravel().astype(np.float64)
        structure[key] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "q001": float(np.quantile(v, 0.001)),
            "q01": float(np.quantile(v, 0.01)),
            "median": float(np.median(v)),
            "q99": float(np.quantile(v, 0.99)),
            "q999": float(np.quantile(v, 0.999)),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    # ---- 2. Foster / connectivity ---------------------------------------
    foster = values[:, :, names.index("foster_residual")].astype(np.float64)
    comps = values[:, :, names.index("n_near_zero")].astype(np.float64)
    connected = comps <= 1.0
    foster_check = {
        "layer_query_cells": int(foster.size),
        "connected_cells": int(connected.sum()),
        "disconnected_cells": int((~connected).sum()),
        "max_abs_residual_connected": float(np.abs(foster[connected]).max()),
        "mean_abs_residual_connected": float(np.abs(foster[connected]).mean()),
        "residual_equals_components_minus_one_on_disconnected": bool(
            np.allclose(foster[~connected], comps[~connected] - 1.0, atol=1e-9)
        ) if (~connected).any() else None,
        "identity": "sum_{i<j} w_ij R_ij == n - k, n = 10, k = number of components",
    }

    # ---- 3. redundancy vs published quantities --------------------------
    existing_names = list(graph["metric_names"].astype(str))
    existing = graph["metrics"][row, query]                     # [n_valid, 8, 11]
    mobility = mob["mobility"][row, query][:, :, None]          # [n_valid, 8, 1]
    reference = np.concatenate([existing, mobility], axis=2)
    reference_names = existing_names + ["mobility"]

    rng = np.random.default_rng(RNG_SEED)
    take = rng.choice(n_valid, size=min(SUBSAMPLE, n_valid), replace=False)
    take.sort()
    xs = values[take].reshape(-1, len(names)).astype(np.float64)
    ys = reference[take].reshape(-1, len(reference_names)).astype(np.float64)
    del reference
    good = np.isfinite(xs).all(1) & np.isfinite(ys).all(1)
    xs, ys = xs[good], ys[good]
    xr = np.column_stack([stats.rankdata(xs[:, i]) for i in range(xs.shape[1])])
    yr = np.column_stack([stats.rankdata(ys[:, i]) for i in range(ys.shape[1])])
    cross = np.corrcoef(np.column_stack([xr, yr]), rowvar=False)
    n_x = xs.shape[1]
    rows = []
    for fi, fname in enumerate(names):
        if xs[:, fi].std() == 0:
            rows.append({"feature": fname, "best_reference": None,
                         "best_abs_spearman": 0.0, "degenerate": True})
            continue
        per_ref = {rname: float(cross[fi, n_x + ri])
                   for ri, rname in enumerate(reference_names)}
        best_ref = max(per_ref, key=lambda k: abs(per_ref[k]))
        rows.append({"feature": fname, "best_reference": best_ref,
                     "best_abs_spearman": abs(per_ref[best_ref]), "degenerate": False,
                     **{f"rho_{k}": v for k, v in per_ref.items()}})
    redundancy = pd.DataFrame(rows)
    redundancy.to_csv(args.output / "redundancy_vs_published.csv", index=False)
    pd.DataFrame(cross[:n_x, :n_x], index=names, columns=names).to_csv(
        args.output / "circuit_internal_spearman.csv"
    )
    print(f"rank correlations on {len(xs)} finite (query, layer) cells", flush=True)

    # ---- 4. front vs back ------------------------------------------------
    front = values[:, :4, :].mean(axis=1)
    back = values[:, 4:, :].mean(axis=1)
    contrast = []
    for fi, fname in enumerate(names):
        f, b = front[:, fi].astype(np.float64), back[:, fi].astype(np.float64)
        pooled = np.std(np.concatenate([f, b]))
        contrast.append({
            "feature": fname,
            "front_mean": float(f.mean()),
            "back_mean": float(b.mean()),
            "back_minus_front": float(b.mean() - f.mean()),
            "standardised_gap": float((b.mean() - f.mean()) / pooled) if pooled > 0 else 0.0,
            "front_std": float(f.std()),
            "back_std": float(b.std()),
        })
    pd.DataFrame(contrast).to_csv(args.output / "front_back_contrast.csv", index=False)

    summary = {
        "schema": "himoe.circuit_analogy.diagnostics.v1",
        "cohort": args.cohort,
        "labels_used": False,
        "valid_queries": int(n_valid),
        "subsample_for_rank_correlation": int(len(take)),
        "subsample_seed": RNG_SEED,
        "clique_structure": structure,
        "foster_check": foster_check,
        "most_redundant": redundancy.nlargest(8, "best_abs_spearman")[
            ["feature", "best_reference", "best_abs_spearman"]
        ].to_dict("records"),
        "least_redundant": redundancy.nsmallest(12, "best_abs_spearman")[
            ["feature", "best_reference", "best_abs_spearman"]
        ].to_dict("records"),
    }
    (args.output / "diagnostics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
