"""Explicit post-result check of nonnegative accumulation and success-only reference."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from core import ROOT, ALPHAS, SEEDS, ReferenceDistance, aggregate, digest, write_json
from run import load_suite, make_split, calibrate

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    parent = args.input.resolve()
    original = json.loads((parent / "sealed_manifest.json").read_text())
    output = parent / "followup"
    output.mkdir(exist_ok=False)
    for name in ("predictions", "profiles", "calibration"):
        (output / name).mkdir()
    (output / "features").symlink_to("../features", target_is_directory=True)
    for name in ("index.csv", "extraction_audit.json"):
        shutil.copyfile(parent / name, output / name)
    full = pd.read_csv(output / "index.csv")
    methods = tuple(f"{base}__{mode}" for base in ("stats_positive", "stats_ratio", "success_only") for mode in ("current", "cumsum"))
    records = []
    for suite_i, suite in enumerate(sorted(full.suite.unique())):
        frame, arrays, labels = load_suite(parent, full, suite)
        for seed in SEEDS:
            begin = time.perf_counter()
            name = f"{suite}_{seed}"
            reference, calibration, test, seen, unseen = make_split(frame, seed, suite_i)
            needed = np.unique(np.concatenate((reference, calibration, test)))
            remap = np.full(len(frame), -1, int)
            remap[needed] = np.arange(len(needed))
            with np.load(parent / "profiles" / f"{name}.npz", allow_pickle=False) as profile:
                success, failure = profile["stats_success"], profile["stats_failure"]
            raw = ReferenceDistance(success, failure).score(arrays["stats"][needed])
            ds, df = raw[..., 1], np.maximum(raw[..., 1] - raw[..., 0], 0)
            success_distance = ReferenceDistance(success, success).score(arrays["stats"][needed])[..., 1]
            bases = (np.maximum(raw[..., 0], 0), (ds + 1e-8) / (ds + df + 2e-8), success_distance)
            scores = np.stack([aggregate(base, mode) for base in bases for mode in ("current", "cumsum")])
            center, scale, counts, thresholds, first, calibration_records = calibrate(
                scores, labels[needed], frame.iloc[needed].reset_index(drop=True), remap[reference], remap[calibration], remap[test])
            for row in calibration_records:
                row["method"] = methods[row.pop("method_index")]
                row["fold"] = name
            global_rows = frame.global_row.to_numpy()
            path = output / "predictions" / f"{name}.npz"
            np.savez_compressed(path, scores=scores[:, remap[test]], methods=np.asarray(methods),
                test_rows=global_rows[test], test_unseen=frame.iloc[test].task.isin(unseen).to_numpy(),
                reference_rows=global_rows[reference], calibration_rows=global_rows[calibration],
                center=center, scale=scale, reference_counts=counts, thresholds=thresholds, first=first,
                alphas=np.asarray(ALPHAS), checkpoint=np.asarray(frame.checkpoint.iloc[0]), seed=np.asarray(seed), reference_cap=np.asarray(4096))
            pd.DataFrame(calibration_records).to_csv(output / "calibration" / f"{name}.csv", index=False)
            info = {"fold": name, "suite": suite, "seed": seed, "reference_cap": 4096,
                "seen_tasks": seen, "unseen_tasks": unseen, "reference_episodes": len(reference),
                "calibration_episodes": len(calibration), "test_episodes": len(test),
                "reference_failures": int((labels[reference] == 1).sum()),
                "seconds": time.perf_counter() - begin, "prediction_sha256": digest(path)}
            write_json(output / "profiles" / f"{name}.json", info)
            records.append(info)
            print(f"FOLLOWUP SEALED {name}: {info['seconds']:.1f}s", flush=True)
        del arrays
    artifacts = {str(path.relative_to(output)): digest(path) for folder in ("predictions", "profiles", "calibration")
                 for path in sorted((output / folder).iterdir())}
    artifacts.update({name: digest(output / name) for name in ("index.csv", "extraction_audit.json")})
    write_json(output / "sealed_manifest.json", {"folds": records, "artifacts": artifacts,
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), HERE / "FOLLOWUP_ZH.md", HERE / "core.py", HERE / "run.py")},
        "parent_manifest_sha256": digest(parent / "sealed_manifest.json"),
        "original_primary": original["primary"], "primary": None, "post_result_exploration": True,
        "new_model_training": False, "test_labels_used_in_formulas_or_calibration": False})


if __name__ == "__main__":
    main()
