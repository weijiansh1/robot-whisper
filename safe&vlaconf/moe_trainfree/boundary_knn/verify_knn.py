"""Independently check reference membership, calibration, neighbors, and raw replay."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from knn import HERE, ROOT, METHODS, PRIMARY, ALPHAS, dynamics
from knn_monitor import BoundaryKNNMonitor
from core import digest, write_json
from run import make_split
from v7_adapter import raw_episode


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round5_knn")
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    output, parent = args.input.resolve(), args.parent.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    checks = 0
    for category, prefix in (("sources", ROOT), ("inputs", ROOT), ("artifacts", output)):
        for name, expected in manifest[category].items():
            assert digest(prefix / name) == expected, name
            checks += 1
    frame = pd.read_csv(parent / "outcome_alignment.csv")
    index = pd.read_csv(output / "index.csv")
    pd.testing.assert_frame_equal(frame[index.columns], index)
    cached = load_npz(parent / "v7/v7_inputs.npz")
    mobility, acceleration, periodicity = (cached[k] for k in ("mobility", "acceleration", "periodicity"))
    geometry_inputs = {a["source"]: a["output"] for a in json.loads((parent / "extraction_audit.json").read_text())}
    calibration_checks, neighbor_checks, stream_replays = 0, [], []
    primary_update_seconds = []
    for info in manifest["folds"]:
        fold, suite = info["fold"], info["suite"]
        data = load_npz(output / "predictions" / f"{fold}.npz")
        profile_path = output / "profiles" / f"{fold}.npz"
        profile = load_npz(profile_path)
        part = frame.loc[frame.suite == suite].copy()
        part["global_row"] = part.index
        part = part.reset_index(drop=True)
        ref, cal, test, _, unseen = make_split(part, info["seed"], sorted(frame.suite.unique()).index(suite))
        for name, rows in (("reference", ref), ("calibration", cal), ("test", test)):
            np.testing.assert_array_equal(data[f"{name}_rows"], part.iloc[rows].global_row)
        np.testing.assert_array_equal(data["test_unseen"], part.iloc[test].task.isin(unseen))
        np.testing.assert_array_equal(data["calibration_labels"], frame.iloc[data["calibration_rows"]].failure.astype(int))
        period = periodicity[data["reference_rows"]]
        scale = np.quantile(np.abs(period[np.isfinite(period)]), .75)
        np.testing.assert_allclose(profile["periodicity_scale"], scale, rtol=0, atol=0)
        for bank in ("success", "mixture"):
            rows, queries = profile[f"{bank}_global_rows"], profile[f"{bank}_queries"]
            assert np.isin(rows, data["reference_rows"]).all()
            assert not np.isin(rows, np.r_[data["calibration_rows"], data["test_rows"]]).any()
            assert len(rows) <= 4096 and (queries >= 7).all()
            assert np.unique(np.stack((rows, queries), axis=1), axis=0).shape[0] == len(rows)
            assert np.unique(rows, return_counts=True)[1].max() <= 8
            if bank == "success":
                assert not frame.iloc[rows].failure.any()
            else:
                np.testing.assert_array_equal(profile["mixture_failure"], frame.iloc[rows].failure.astype(int))
        np.testing.assert_allclose(profile["pca_components"] @ profile["pca_components"].T, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(profile["pca_mean"], profile["mixture_dynamic"].mean(0), atol=1e-12)
        np.testing.assert_allclose(profile["success_pca2"],
            (profile["success_dynamic"] - profile["pca_mean"]) @ profile["pca_components"].T, atol=1e-12)
        success = data["calibration_labels"] == 0
        cal_frame = frame.iloc[data["calibration_rows"]].loc[success]
        groups = (cal_frame.task + "|" + cal_frame.init_state_id.astype(str)).to_numpy()
        for method_i, method in enumerate(METHODS):
            values = data["calibration_scores"][method_i, success]
            peaks = np.where(np.isfinite(values), values, -np.inf).max(1)
            units_by_kind = (peaks, np.asarray([peaks[groups == key].max() for key in sorted(set(groups))]))
            valid = np.arange(52)[None] < frame.iloc[data["test_rows"]].length.to_numpy()[:, None]
            raw = data["scores"][method_i]
            assert not np.isfinite(raw[~valid]).any()
            earliest = 9 if method.endswith("persist3") else 7
            assert not np.isfinite(raw[:, :earliest]).any()
            for kind, units in enumerate(units_by_kind):
                for alpha_i, alpha in enumerate(ALPHAS):
                    rank = int(np.ceil((len(units) + 1) * (1 - alpha)))
                    threshold = np.sort(units)[rank - 1] if rank <= len(units) else np.inf
                    np.testing.assert_equal(data["thresholds"][kind, alpha_i, method_i], threshold)
                    crossed = np.isfinite(raw) & (raw > threshold)
                    expected_first = np.where(crossed.any(1), crossed.argmax(1), -1)
                    np.testing.assert_array_equal(expected_first, data["first"][kind, alpha_i, method_i])
                    calibration_checks += 1
        needed = []
        for kind in ("calibration", "test"):
            rows = data[f"{kind}_rows"]
            for local in (0, len(rows) // 2, len(rows) - 1):
                needed.append((kind, local, int(rows[local])))
        for kind, local, global_row in needed:
            row = frame.iloc[global_row]
            d, _ = dynamics(mobility[global_row:global_row + 1], acceleration[global_row:global_row + 1],
                periodicity[global_row:global_row + 1], float(scale))
            path = parent / "features" / geometry_inputs[row.source]
            with np.load(path, allow_pickle=False) as z:
                stored_index = pd.DataFrame(json.loads(str(z["index"])))
                location = int(np.flatnonzero(stored_index.episode.to_numpy() == row.episode)[0])
                routing = z["load"][location]
            for q in (7, 14, 21):
                if q >= row.length:
                    continue
                x = (d[0, q].astype(np.float64) - profile["dynamic_center"]) / profile["dynamic_scale"]
                r = (routing[q].astype(np.float64) - profile["routing_center"]) / profile["routing_scale"]
                ds = np.sort(np.linalg.norm(profile["success_dynamic"] - x, axis=1))
                dm = np.linalg.norm(profile["mixture_dynamic"] - x, axis=1)
                nearest = np.argsort(dm)[:20]
                xy = (x - profile["pca_mean"]) @ profile["pca_components"].T
                expected = {"dyn_radius": np.linalg.norm(x),
                    **{f"dyn_success_knn_k{k}": ds[:k].mean() for k in (1, 5, 20)},
                    "dyn_vote_knn_k20": profile["mixture_failure"][nearest].mean(),
                    "dyn_pca2_success_knn_k20": np.sort(np.linalg.norm(profile["success_pca2"] - xy, axis=1))[:20].mean(),
                    "route_success_knn_k20": np.sort(np.linalg.norm(profile["success_routing"] - r, axis=1))[:20].mean()}
                scores = data["scores"] if kind == "test" else data["calibration_scores"]
                for method, value in expected.items():
                    np.testing.assert_allclose(scores[METHODS.index(method), local, q], value, rtol=2e-5, atol=2e-6,
                        err_msg=f"{fold} {global_row} {q} {method}")
                neighbor_checks.append({"fold": fold, "global_row": global_row, "query": q, "kind": kind})
            if kind != "test":
                continue
            raw = raw_episode(row)
            for method in (PRIMARY, "dyn_radius_outward", "dyn_success_knn_k20_persist3"):
                monitor = BoundaryKNNMonitor(profile_path, str(profile["checkpoint"]), method=method)
                observed = []
                for query in raw:
                    started = time.perf_counter()
                    result = monitor.update(query)
                    if method == PRIMARY and result["query"] >= 7:
                        primary_update_seconds.append(time.perf_counter() - started)
                    observed.append(result["score"])
                method_i = METHODS.index(method)
                expected = data["scores"][method_i, local, :row.length]
                np.testing.assert_allclose(observed, expected, rtol=5e-4, atol=3e-4, equal_nan=True,
                    err_msg=f"raw replay {fold} {global_row} {method}")
                expected_first = int(data["first"][1, list(ALPHAS).index(.05), method_i, local])
                assert result["first_alarm_query"] == expected_first, (fold, global_row, method, result, expected_first)
                stream_replays.append({"fold": fold, "global_row": global_row, "method": method, "queries": len(raw),
                                       "first_alarm": expected_first})
        print(f"VERIFIED {fold}", flush=True)
    write_json(output / "verification.json", {"hash_checks": checks, "calibration_rank_checks": calibration_checks,
        "direct_neighbor_checks": neighbor_checks, "raw_stream_replays": stream_replays,
        "primary_update_ms_median": float(np.median(primary_update_seconds) * 1000),
        "primary_update_ms_p95": float(np.quantile(primary_update_seconds, .95) * 1000),
        "primary_update_count": len(primary_update_seconds),
        "latency_scope": "CPU, raw routing query already in memory; includes MoE feature extraction and lookup, not VLA inference or disk",
        "verifier_sha256": digest(Path(__file__)), "monitor_sha256": digest(HERE / "knn_monitor.py"),
        "all_stream_alarm_times_match": True})
    print(f"PASS: {calibration_checks} calibration ranks, {len(neighbor_checks)} neighbor oracles, {len(stream_replays)} raw stream replays", flush=True)


if __name__ == "__main__":
    main()
