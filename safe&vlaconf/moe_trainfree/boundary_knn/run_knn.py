"""Seal train-free boundary and kNN predictions before evaluating B outcomes."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from knn import (HERE, ROOT, SEEDS, ALPHAS, METHODS, PRIMARY, dynamics, reference_scaling,
                 reference_pairs, score_streams, calibrate_constant)
from core import digest, write_json
from run import load_suite, make_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe")
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round5_knn")
    args = parser.parse_args()
    parent, output = args.input.resolve(), args.output.resolve()
    output.mkdir(exist_ok=False)
    for name in ("predictions", "profiles", "calibration"):
        (output / name).mkdir()
    shutil.copyfile(parent / "index.csv", output / "index.csv")
    paths = [Path(__file__), HERE / "knn.py", HERE / "PROTOCOL_ZH.md",
             HERE.parent / "feature_geometry/analyze.py", HERE.parent / "safe_protocol/core.py",
             HERE.parent / "safe_protocol/run.py", ROOT / "moe-v7-0905/method/intrinsic_guard_monitor.py"]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    input_paths = [parent / name for name in ("index.csv", "extraction_audit.json", "v7/v7_inputs.npz")]
    input_paths.extend(sorted((parent / "features").glob("*.npz")))
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in input_paths}
    write_json(output / "scoring_contract.json", {"sources": sources, "inputs": inputs, "methods": METHODS,
        "primary": PRIMARY, "primary_calibration": "task_init", "primary_alpha": .05,
        "new_model_training": False, "reference_and_calibration_use_historical_labels": True})
    full = pd.read_csv(output / "index.csv")
    with np.load(parent / "v7/v7_inputs.npz", allow_pickle=False) as z:
        mobility, acceleration, periodicity, valid = (z[k] for k in ("mobility", "acceleration", "periodicity", "valid"))
    folds = []
    for suite_i, suite in enumerate(sorted(full.suite.unique())):
        frame, arrays, labels = load_suite(parent, full, suite)
        global_rows = frame.global_row.to_numpy()
        m, a, p = mobility[global_rows], acceleration[global_rows], periodicity[global_rows]
        for source in frame.source.unique():
            path = ROOT / "VLA_MUI_HUB" / source / "client/summaries.json"
            if str(source).split("/")[-1] == "right-50x8-20260903":
                inputs[str(path.relative_to(ROOT))] = digest(path)
        for seed in SEEDS:
            started = time.perf_counter()
            fold = f"{suite}_{seed}"
            reference, calibration, test, seen, unseen = make_split(frame, seed, suite_i)
            assert (labels[test] == -1).all()
            periodicity_scale = float(np.quantile(np.abs(p[reference][np.isfinite(p[reference])]), .75))
            dynamic, _ = dynamics(m, a, p, periodicity_scale)
            dynamic[~valid[global_rows]] = np.nan
            normalized_d, d_scale = reference_scaling(dynamic, reference)
            normalized_r, r_scale = reference_scaling(arrays["load"], reference, global_scale=True)
            success_pairs = reference_pairs(dynamic, reference[labels[reference] == 0], seed=seed)
            mixture_pairs = reference_pairs(dynamic, reference, seed=seed + 1)
            dr, dq = success_pairs.T
            mr, mq = mixture_pairs.T
            success_d = normalized_d[dr, dq].astype(np.float64)
            mixture_d = normalized_d[mr, mq].astype(np.float64)
            pca = PCA(n_components=2, svd_solver="full").fit(mixture_d)
            profile = {"dynamic_center": np.asarray(d_scale["center"]), "dynamic_scale": np.asarray(d_scale["scale"]),
                "routing_center": np.asarray(r_scale["center"]), "routing_scale": np.asarray(r_scale["scale"]),
                "success_dynamic": success_d, "success_routing": normalized_r[dr, dq].astype(np.float64),
                "mixture_dynamic": mixture_d, "mixture_failure": labels[mr].astype(np.int8),
                "success_global_rows": global_rows[dr], "success_queries": dq,
                "mixture_global_rows": global_rows[mr], "mixture_queries": mq,
                "pca_mean": pca.mean_, "pca_components": pca.components_,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_,
                "success_pca2": pca.transform(success_d), "periodicity_scale": np.asarray(periodicity_scale),
                "methods": np.asarray(METHODS), "alphas": np.asarray(ALPHAS),
                "checkpoint": np.asarray(frame.checkpoint.iloc[0])}
            del normalized_d, normalized_r
            needed = np.concatenate((calibration, test))
            scoring_started = time.perf_counter()
            scores = score_streams(profile, dynamic[needed], arrays["load"][needed], arrays["direct"][needed])
            scoring_seconds = time.perf_counter() - scoring_started
            ncal = len(calibration)
            thresholds, first, records = calibrate_constant(scores[:, :ncal], labels[calibration],
                frame.iloc[calibration].reset_index(drop=True), scores[:, ncal:])
            profile["thresholds"] = thresholds
            profile_path = output / "profiles" / f"{fold}.npz"
            np.savez_compressed(profile_path, **profile)
            prediction = output / "predictions" / f"{fold}.npz"
            np.savez_compressed(prediction, scores=scores[:, ncal:], calibration_scores=scores[:, :ncal],
                methods=np.asarray(METHODS), thresholds=thresholds, first=first, alphas=np.asarray(ALPHAS),
                reference_rows=global_rows[reference], calibration_rows=global_rows[calibration],
                calibration_labels=labels[calibration], test_rows=global_rows[test],
                test_unseen=frame.iloc[test].task.isin(unseen).to_numpy(),
                checkpoint=np.asarray(frame.checkpoint.iloc[0]), seed=np.asarray(seed))
            for record in records:
                record["fold"] = fold
            pd.DataFrame(records).to_csv(output / "calibration" / f"{fold}.csv", index=False)
            info = {"fold": fold, "suite": suite, "seed": seed, "seen_tasks": seen, "unseen_tasks": unseen,
                "reference_episodes": len(reference), "calibration_episodes": len(calibration),
                "test_episodes": len(test), "reference_failures": int((labels[reference] == 1).sum()),
                "success_points": len(success_pairs), "mixture_points": len(mixture_pairs),
                "mixture_failure_points": int(profile["mixture_failure"].sum()),
                "pca_variance_2d": float(pca.explained_variance_ratio_.sum()),
                "score_queries": int(np.isfinite(dynamic[needed]).all(-1).sum()),
                "scoring_seconds": scoring_seconds, "seconds": time.perf_counter() - started,
                "prediction_sha256": digest(prediction), "profile_sha256": digest(profile_path)}
            folds.append(info)
            write_json(output / "profiles" / f"{fold}.json", info)
            print(f"SEALED {fold}: {info['score_queries']} queries, {info['seconds']:.1f}s", flush=True)
        del arrays
    artifacts = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "sealed_manifest.json", {"sources": sources, "inputs": inputs, "artifacts": artifacts,
        "folds": folds, "primary": PRIMARY, "test_outcomes_used_for_scoring": False,
        "new_model_training": False, "label_free": False, "historically_explored_data": True})
    print("ALL BOUNDARY/KNN PREDICTIONS SEALED", flush=True)


if __name__ == "__main__":
    main()
