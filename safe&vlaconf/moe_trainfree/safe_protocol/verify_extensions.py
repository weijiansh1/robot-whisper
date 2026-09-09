"""Check subsequent distance readouts and v7 adaptations on held-out raw prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from core import ROOT, ReferenceDistance, route_features, aggregate, digest, write_json
from v7_adapter import raw_episode
from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor

HERE = Path(__file__).resolve().parent


def independent_distances(values, positive, negative):
    both = np.concatenate((positive, negative))
    center = np.median(both, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(both - center), axis=0), 1e-6)
    x = (values - center) / scale
    result = []
    for bank in (positive, negative):
        distances = cdist(x, (bank - center) / scale) / np.sqrt(values.shape[-1])
        k = min(5, len(bank))
        result.append(np.partition(distances, k - 1, axis=1)[:, :k].mean(-1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    parent = args.input.resolve()
    frame = pd.read_csv(parent / "outcome_alignment.csv")
    index = pd.read_csv(parent / "index.csv")
    pd.testing.assert_frame_equal(frame[index.columns], index)
    audits = json.loads((parent / "extraction_audit.json").read_text())
    stats = np.full((len(frame), 52, 32), np.nan, np.float32)
    for audit in audits:
        rows = np.flatnonzero(frame.source == audit["source"])
        with np.load(parent / "features" / audit["output"], allow_pickle=False) as stored:
            stats[rows] = stored["stats"]
    verified_hashes, rank_checks, followup_checks, v7_checks = 0, 0, [], []
    for stage in ("followup", "v7"):
        output = parent / stage
        manifest = json.loads((output / "sealed_manifest.json").read_text())
        for name, expected in manifest["sources"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"extension source changed {name}")
            verified_hashes += 1
        for name, expected in manifest["artifacts"].items():
            if digest(output / name) != expected:
                raise ValueError(f"extension artifact changed {name}")
            verified_hashes += 1
        for file in (output / "calibration").glob("*.csv"):
            calibration = pd.read_csv(file)
            if (calibration.exceedances > calibration.calibration_units + 1 - calibration["rank"]).any():
                raise AssertionError("extension calibration rank failed")
            rank_checks += len(calibration)
    threshold_checks = 0
    main_manifest = json.loads((parent / "sealed_manifest.json").read_text())
    for info in main_manifest["folds"]:
        if info["reference_cap"] != 4096:
            continue
        name = info["fold"]
        with np.load(parent / "profiles" / f"{name}.npz", allow_pickle=False) as stored:
            success, failure = stored["stats_success"], stored["stats_failure"]
        with np.load(parent / "followup/predictions" / f"{name}.npz", allow_pickle=False) as stored:
            followup = {k: stored[k] for k in stored.files}
        with np.load(parent / "v7/predictions" / f"{name}.npz", allow_pickle=False) as stored:
            v7 = {k: stored[k] for k in stored.files}
        profile = json.loads((parent / "v7/profiles" / f"{name}.json").read_text())
        cal = followup["calibration_rows"]
        cal = cal[~frame.iloc[cal].failure.to_numpy()]
        original_distance = ReferenceDistance(success, failure)
        success_distance = ReferenceDistance(success, success)
        raw = original_distance.score(stats[cal])
        ds, df = raw[..., 1], np.maximum(raw[..., 1] - raw[..., 0], 0)
        bases = (np.maximum(raw[..., 0], 0), (ds + 1e-8) / (ds + df + 2e-8), success_distance.score(stats[cal])[..., 1])
        scores = np.stack([aggregate(base, mode) for base in bases for mode in ("current", "cumsum")])
        ids = (frame.iloc[cal].task + "|" + frame.iloc[cal].init_state_id.astype(str)).to_numpy()
        for method_i, score in enumerate(scores):
            normalized = (score - followup["center"][method_i]) / followup["scale"][method_i]
            peaks = np.nanmax(normalized, axis=1)
            grouped = np.asarray([peaks[ids == group].max() for group in sorted(set(ids))])
            for kind, units in enumerate((peaks, grouped)):
                for alpha_i, alpha in enumerate(followup["alphas"]):
                    rank = int(np.ceil((len(units) + 1) * (1 - alpha)))
                    expected = np.sort(units)[rank - 1] if rank <= len(units) else np.inf
                    np.testing.assert_allclose(expected, followup["thresholds"][kind, alpha_i, method_i], rtol=1e-6, atol=1e-5)
                    threshold_checks += 1
        for local in (0, len(v7["test_rows"]) // 2, len(v7["test_rows"]) - 1):
            global_row = int(v7["test_rows"][local])
            record = frame.loc[global_row]
            raw_routes = raw_episode(record)
            vectors = route_features(raw_routes[:, :, 9])["stats"]
            a, b = independent_distances(vectors, success, failure)
            exact = original_distance.score(vectors)
            np.testing.assert_allclose(exact[:, 0], a - b, rtol=3e-5, atol=1e-5)
            np.testing.assert_allclose(exact[:, 1], a, rtol=3e-5, atol=1e-5)
            dummy = GlobalIntrinsicProfile(1e9, 1e9, 1e9, profile["periodicity_scale"])
            monitor = IntrinsicGuardMonitor(dummy)
            maximum_a, maximum_p = -np.inf, -np.inf
            computed = []
            for q, routes in enumerate(raw_routes):
                result = monitor.update(routes)
                heads = []
                for head, field in (("freeze", "freeze_score"), ("acceleration_persistent", "acceleration_score"), ("periodicity_persistent", "periodicity_score")):
                    parameters = profile["head_stats"][head]
                    heads.append((result[field] - parameters["center"]) / parameters["scale"])
                f, a, p = heads
                if np.isfinite(a): maximum_a = max(maximum_a, a)
                if np.isfinite(p): maximum_p = max(maximum_p, p)
                turbulence = min(maximum_a, maximum_p)
                if not np.isfinite(turbulence): turbulence = np.nan
                guard = np.fmax(f, turbulence)
                gscale, dscale = profile["fusion_guard_stats"], profile["fusion_distance_stats"]
                d = float(success_distance.score(vectors[q:q+1])[0, 1])
                fusion = np.fmax((guard - gscale["center"]) / gscale["scale"], (d - dscale["center"]) / dscale["scale"])
                computed.append([guard, guard, f, turbulence, fusion])
            computed = np.asarray(computed, np.float32).T
            np.testing.assert_allclose(computed, v7["scores"][:, local, :record.length], rtol=3e-5, atol=2e-5, equal_nan=True)
            for method_i, score in enumerate(computed):
                for kind in range(2):
                    alpha_i = list(v7["alphas"]).index(.05)
                    standardized = (score - v7["center"][method_i, :record.length]) / v7["scale"][method_i, :record.length]
                    crossings = np.flatnonzero(standardized > v7["thresholds"][kind, alpha_i, method_i])
                    first = int(crossings[0]) if len(crossings) else -1
                    if first != int(v7["first"][kind, alpha_i, method_i, local]):
                        raise AssertionError("v7 adaptation online first-alarm mismatch")
            v7_checks.append({"fold": name, "global_row": global_row, "methods": 5, "queries": int(record.length)})
            ds, df = exact[:, 1], np.maximum(exact[:, 1] - exact[:, 0], 0)
            bases = (np.maximum(exact[:, 0], 0), (ds + 1e-8) / (ds + df + 2e-8), success_distance.score(vectors)[:, 1])
            prediction = np.stack([aggregate(base[None], mode)[0] for base in bases for mode in ("current", "cumsum")])
            np.testing.assert_allclose(prediction, followup["scores"][:, local, :record.length], rtol=3e-5, atol=2e-5)
            followup_checks.append({"fold": name, "global_row": global_row, "methods": 6, "queries": int(record.length)})
        print(f"EXTENSIONS VERIFIED {name}", flush=True)
    write_json(parent / "extension_verification.json", {"hash_checks": verified_hashes, "calibration_rank_checks": rank_checks,
        "recomputed_distance_thresholds": threshold_checks, "distance_raw_replays": followup_checks,
        "v7_adaptation_raw_replays": v7_checks, "test_labels_used_only_for_evaluation": True,
        "verifier_sha256": digest(Path(__file__))})


if __name__ == "__main__":
    main()
