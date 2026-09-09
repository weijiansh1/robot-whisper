"""Post-hoc check of peripheral endpoints and original-space reference radii."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from analyze import BATCH, ROOT, SEED, digest, dynamics, load_suite, write_json

HERE = Path(__file__).resolve().parent


def projection_endpoints(root, inputs):
    records = []
    for suite in sorted(pd.read_csv(root / "projection_metrics.csv").suite.unique()):
        meta_path = root / "projections" / f"{suite}_matched_points.csv"
        meta = pd.read_csv(meta_path)
        inputs[str(meta_path.relative_to(ROOT))] = digest(meta_path)
        endpoints = meta.groupby("global_row")["query"].idxmax().to_numpy()
        np.testing.assert_array_equal(meta.iloc[endpoints]["query"], meta.iloc[endpoints].length - 1)
        for representation in ("routing", "dynamics"):
            for seed in (SEED, SEED + 1, SEED + 2):
                folder = "projections" if seed == SEED else "random_init"
                path = root / folder / f"{suite}_{representation}_matched_seed{seed}.npz"
                inputs[str(path.relative_to(ROOT))] = digest(path)
                with np.load(path, allow_pickle=False) as z:
                    xy = z["xy"].astype(np.float64)
                    assert xy.shape == (len(meta), 2) and np.isfinite(xy).all()
                    if seed == SEED:
                        np.testing.assert_array_equal(z["point_global_rows"], meta.global_row)
                        np.testing.assert_array_equal(z["point_queries"], meta["query"])
                # A rotation-invariant, label-blind proxy for the outer part of a plot.
                radius = np.linalg.norm(xy - xy.mean(0), axis=1)
                threshold = np.quantile(radius, .8)
                percentile = rankdata(radius, method="average") / len(radius)
                for failure in (False, True):
                    rows = endpoints[meta.iloc[endpoints].failure.to_numpy() == failure]
                    records.append({"suite": suite, "representation": representation,
                        "seed": seed, "initialization": "pca" if seed == SEED else "random",
                        "failure": failure, "episodes": len(rows),
                        "endpoint_in_outer_20_fraction": float((radius[rows] >= threshold).mean()),
                        "endpoint_median_radius_percentile": float(np.median(percentile[rows])),
                        "all_points_in_outer_20_fraction": float((radius >= threshold).mean())})
    return pd.DataFrame(records)


def original_radii(root, parent, inputs):
    paths = [parent / "outcome_alignment.csv", parent / "extraction_audit.json",
             parent / "v7/v7_inputs.npz", root / "scaling.json"]
    inputs.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
    frame = pd.read_csv(parent / "outcome_alignment.csv")
    scaling = json.loads((root / "scaling.json").read_text())
    with np.load(parent / "v7/v7_inputs.npz", allow_pickle=False) as z:
        mobility, acceleration, periodicity, valid = (z[k] for k in ("mobility", "acceleration", "periodicity", "valid"))
    points, summaries = [], []
    for suite in sorted(frame.suite.unique()):
        part, arrays = load_suite(parent, frame, suite, inputs)
        g = part.global_row.to_numpy()
        d, _ = dynamics(mobility[g], acceleration[g], periodicity[g], scaling[suite]["periodicity_scale"])
        d[~valid[g]] = np.nan
        arrays["dynamics"] = d
        for q in (7, 14, 21):
            eligible = (part.run_id == BATCH).to_numpy() & (part.length.to_numpy() > q)
            eligible &= np.isfinite(d[:, q]).all(-1)
            rows = np.flatnonzero(eligible)
            current = part.iloc[rows]
            y = current.failure.to_numpy()
            for representation in ("routing", "dynamics"):
                normalization = scaling[suite][representation]
                x = ((arrays[representation][rows, q].astype(np.float64) - normalization["center"]) /
                     normalization["scale"])
                radius = np.linalg.norm(x, axis=1)
                assert np.isfinite(radius).all()
                task_aucs = []
                for task in sorted(current.task.unique()):
                    task_rows = current.task.to_numpy() == task
                    if np.unique(y[task_rows]).size == 2:
                        task_aucs.append(roc_auc_score(y[task_rows], radius[task_rows]))
                summaries.append({"suite": suite, "representation": representation, "query": q,
                    "episodes": len(rows), "failure_episodes": int(y.sum()),
                    "failure_median_radius": float(np.median(radius[y])),
                    "success_median_radius": float(np.median(radius[~y])),
                    "pooled_auroc": float(roc_auc_score(y, radius)),
                    "within_task_macro_auroc": float(np.mean(task_aucs)),
                    "tasks_with_both_outcomes": len(task_aucs)})
                for i, row in enumerate(current.itertuples()):
                    points.append({"suite": suite, "representation": representation, "query": q,
                        "global_row": int(row.global_row), "task": row.task, "failure": bool(row.failure),
                        "reference_radius": float(radius[i]), "eef_motion_m": float(-arrays["direct"][rows[i], q, 5])})
        print(f"CHECKED original reference radii: {suite}", flush=True)
    return pd.DataFrame(points), pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round4_geometry")
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    root, parent = args.input.resolve(), args.parent.resolve()
    destination = root / "edge_probe"
    destination.mkdir(exist_ok=False)
    inputs = {}
    endpoints = projection_endpoints(root, inputs)
    endpoints.to_csv(destination / "projection_endpoints.csv", index=False)
    points, summary = original_radii(root, parent, inputs)
    points.to_csv(destination / "original_radius_points.csv", index=False)
    summary.to_csv(destination / "original_radius_summary.csv", index=False)
    write_json(destination / "audit.json", {"source_sha256": digest(Path(__file__)),
        "inputs": inputs, "artifacts": {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
        "post_hoc_exploration": True, "new_model_training": False,
        "two_dimensional_proxy": "distance to the mean of all plotted points; outer 20 percent; endpoint per episode",
        "original_space_proxy": "L2 distance from the frozen A reference center after existing scaling",
        "no_threshold_or_direction_tuning": True,
        "evaluation_queries": [7, 14, 21], "initializations": ["pca", "random", "random"]})
    print(destination, flush=True)


if __name__ == "__main__":
    main()
