"""Freeze the declared temporal/fusion candidates on the existing A/B splits."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from fusion import (HERE, ROOT, METHODS, PRIMARY, V7_HEADS, ALPHAS, v8_heads,
                    reference_scaling, base_streams, combine_streams, calibrate)
from core import digest, write_json


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round7_temporal_fusion")
    args = parser.parse_args()
    output = args.output.resolve()
    old = HERE.parent / "results/round5_knn"
    v7_root = HERE.parent / "results/round3_safe/v7"
    manifest = json.loads((old / "sealed_manifest.json").read_text())
    v7_manifest = json.loads((v7_root / "sealed_manifest.json").read_text())
    for directory in ("predictions", "profiles", "calibration"):
        (output / directory).mkdir(exist_ok=False)
    frame = pd.read_csv(output / "index.csv")
    pd.testing.assert_frame_equal(frame, pd.read_csv(old / "index.csv"))
    cached = load_npz(v7_root / "v7_inputs.npz")
    extra = load_npz(output / "v8_inputs.npz")
    np.testing.assert_array_equal(cached["valid"], extra["valid"])
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in
              (v7_root / "v7_inputs.npz", output / "v8_inputs.npz", output / "input_audit.json", old / "sealed_manifest.json")}
    sources = {str(path.relative_to(ROOT)): digest(path) for path in
               (Path(__file__), HERE / "fusion.py", HERE / "PROTOCOL_ZH.md", HERE / "monitor.py")}
    write_json(output / "scoring_contract.json", dict(methods=METHODS, primary=PRIMARY,
        alpha=.05, calibration="task_init", sources=sources, inputs=inputs,
        new_model_training=False, test_outcomes_used_for_scoring=False, historically_explored_data=True))
    folds = []
    for info in manifest["folds"]:
        start = time.perf_counter()
        fold = info["fold"]
        for name in (f"predictions/{fold}.npz", f"profiles/{fold}.npz"):
            assert digest(old / name) == manifest["artifacts"][name]
            inputs[str((old / name).relative_to(ROOT))] = digest(old / name)
        profile_name = f"profiles/{fold}.json"
        assert digest(v7_root / profile_name) == v7_manifest["artifacts"][profile_name]
        inputs[str((v7_root / profile_name).relative_to(ROOT))] = digest(v7_root / profile_name)
        original = load_npz(old / "predictions" / f"{fold}.npz")
        profile = load_npz(old / "profiles" / f"{fold}.npz")
        previous = json.loads((v7_root / profile_name).read_text())
        reference, calibration, tests = (original[key] for key in ("reference_rows", "calibration_rows", "test_rows"))
        rows = np.concatenate((reference, calibration, tests))
        assert len(rows) == len(set(rows.tolist()))
        assert frame.iloc[np.r_[reference, calibration]].run_id.eq("right-50x8-20260903").all()
        assert frame.iloc[tests].run_id.eq("right-50x8b-20260903").all()
        nr, nc = len(reference), len(calibration)
        needed = frame.iloc[rows].reset_index(drop=True)
        raw = extra["raw"][rows]
        heads = v8_heads(raw)
        heads[:, :7] = np.nan
        _, stats = reference_scaling(heads, np.arange(nr))
        profile["v8_center"] = np.asarray(stats["center"])
        profile["v8_scale"] = np.asarray(stats["scale"])
        profile["v7_center"] = np.asarray([previous["head_stats"][key]["center"] for key in V7_HEADS])
        profile["v7_scale"] = np.asarray([previous["head_stats"][key]["scale"] for key in V7_HEADS])
        positions = pd.Index(rows).get_indexer(profile["success_global_rows"])
        assert (positions >= 0).all() and (positions < nr).all()
        additions = (heads[positions, profile["success_queries"]].astype(np.float64) - profile["v8_center"]) / profile["v8_scale"]
        assert np.isfinite(additions).all()
        profile["success_augmented"] = np.concatenate((profile["success_dynamic"], additions), axis=1)
        base = base_streams(profile, cached["mobility"][rows], cached["acceleration"][rows], cached["periodicity"][rows], raw)
        scale_values = np.stack([base[name] for name in ("knn10", "v7_guard", "v8_guard")], axis=-1)
        _, stats = reference_scaling(scale_values, np.arange(nr))
        profile["fusion_center"] = np.asarray(stats["center"])
        profile["fusion_scale"] = np.asarray(stats["scale"])
        scores = combine_streams(base, profile)
        valid = cached["valid"][rows]
        assert not np.isfinite(scores[:, ~valid]).any()
        cal_scores, test_scores = scores[:, nr:nr+nc], scores[:, nr+nc:]
        thresholds, first, records = calibrate(cal_scores, original["calibration_labels"], needed.iloc[nr:nr+nc].reset_index(drop=True), test_scores)
        for current, old_name in (("knn10", "dyn_success_knn_k20"), ("knn10_persist3", "dyn_success_knn_k20_persist3")):
            mi, oi = METHODS.index(current), list(original["methods"]).index(old_name)
            np.testing.assert_allclose(test_scores[mi], original["scores"][oi], rtol=1e-6, atol=2e-7, equal_nan=True)
            np.testing.assert_array_equal(first[:, :, mi], original["first"][:, :, oi])
        old_v7_path = v7_root / "predictions" / f"{fold}.npz"
        assert digest(old_v7_path) == v7_manifest["artifacts"][f"predictions/{fold}.npz"]
        inputs[str(old_v7_path.relative_to(ROOT))] = digest(old_v7_path)
        old_v7 = load_npz(old_v7_path)
        oi = list(old_v7["methods"]).index("v7_guard_constant")
        mi = METHODS.index("v7_guard")
        np.testing.assert_allclose(test_scores[mi], old_v7["scores"][oi], rtol=1e-6, atol=2e-7, equal_nan=True)
        np.testing.assert_array_equal(first[:, :, mi], old_v7["first"][:, :, oi])
        profile.update(methods=np.asarray(METHODS), thresholds=thresholds, alphas=np.asarray(ALPHAS))
        np.savez_compressed(output / "profiles" / f"{fold}.npz", **profile)
        np.savez_compressed(output / "predictions" / f"{fold}.npz", scores=test_scores,
            calibration_scores=cal_scores, methods=np.asarray(METHODS), alphas=np.asarray(ALPHAS),
            thresholds=thresholds, first=first, reference_rows=reference, calibration_rows=calibration,
            calibration_labels=original["calibration_labels"], test_rows=tests,
            test_unseen=original["test_unseen"], checkpoint=original["checkpoint"], seed=original["seed"])
        for record in records:
            record["fold"] = fold
        pd.DataFrame(records).to_csv(output / "calibration" / f"{fold}.csv", index=False)
        result = dict(info, seconds=time.perf_counter()-start, baseline_score_and_alarm_reproductions=3)
        folds.append(result)
        print(f"FUSION SEALED {fold}: {result['seconds']:.1f}s", flush=True)
    artifacts = {str(path.relative_to(output)): digest(path) for directory in ("profiles", "predictions", "calibration")
                 for path in sorted((output / directory).iterdir())}
    artifacts.update({name: digest(output / name) for name in ("index.csv", "v8_inputs.npz", "input_audit.json", "scoring_contract.json")})
    write_json(output / "sealed_manifest.json", dict(folds=folds, artifacts=artifacts, inputs=inputs, sources=sources,
        primary=PRIMARY, new_model_training=False, test_outcomes_used_for_scoring=False, historically_explored_data=True))
    print("ALL FUSION CANDIDATES SEALED", flush=True)


if __name__ == "__main__":
    main()
