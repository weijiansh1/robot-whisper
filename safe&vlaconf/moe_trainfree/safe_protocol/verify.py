"""Replay the deployable monitor and independently recompute primary calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import zarr

from core import ROOT, PRIMARY, ALPHAS, digest, write_json, route_features
from monitor import TrainFreeMoEMonitor

HERE = Path(__file__).resolve().parent


def oracle_distance(values, success, failure):
    both = np.concatenate((success, failure))
    center = np.median(both, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(both - center), axis=0), 1e-6)
    normalized = (values.reshape(-1, values.shape[-1]) - center) / scale
    finite = np.isfinite(normalized).all(-1)
    result = np.full(len(normalized), np.nan)
    targets = normalized[finite]
    estimates = np.empty(len(targets))
    pos, neg = (success - center) / scale, (failure - center) / scale
    for start in range(0, len(targets), 256):
        batch = targets[start:start + 256]
        distances = []
        for bank in (pos, neg):
            d = cdist(batch, bank, metric="euclidean") / np.sqrt(values.shape[-1])
            k = min(5, len(bank))
            distances.append(np.partition(d, k - 1, axis=1)[:, :k].mean(-1))
        estimates[start:start + len(batch)] = distances[0] - distances[1]
    result[finite] = estimates.astype(np.float32)
    return result.reshape(values.shape[:-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    output = args.input.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for relative, expected in manifest["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("scoring source changed")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"sealed output changed {relative}")
    frame = pd.read_csv(output / "index.csv")
    audits = json.loads((output / "extraction_audit.json").read_text())
    stats = np.full((len(frame), 52, 32), np.nan, np.float32)
    source_audits = {a["source"]: a for a in audits}
    for source, rows in frame.groupby("source").groups.items():
        with np.load(output / "features" / source_audits[source]["output"], allow_pickle=False) as archive:
            stats[rows] = archive["stats"]
    checks, timings = [], []
    for info in manifest["folds"]:
        if info["reference_cap"] != 4096:
            continue
        profile = output / "profiles" / f"{info['fold']}.npz"
        with np.load(output / "predictions" / f"{info['fold']}.npz", allow_pickle=False) as archive:
            stored = {k: archive[k] for k in archive.files}
        method = list(stored["methods"].astype(str)).index(PRIMARY)
        with np.load(profile, allow_pickle=False) as archive:
            success, failure = archive["stats_success"], archive["stats_failure"]
        rows = stored["calibration_rows"]
        subset = frame.iloc[rows].copy()
        outcomes = {}
        for source in subset.source.unique():
            records = json.loads((ROOT / "VLA_MUI_HUB" / source / "client/summaries.json").read_text())
            outcomes.update({(source, r["episode_index"]): r["success"] for r in records})
        success_rows = np.asarray([r for r in rows if outcomes[(frame.loc[r, "source"], int(frame.loc[r, "episode"]))]])
        raw = oracle_distance(stats[success_rows], success, failure)
        score = np.where(np.isfinite(raw), np.nancumsum(raw, axis=1), np.nan).astype(np.float32)
        standard = (score - stored["center"][method]) / stored["scale"][method]
        peaks = np.nanmax(standard, axis=1)
        group = (frame.loc[success_rows, "task"] + "|" + frame.loc[success_rows, "init_state_id"].astype(str)).to_numpy()
        grouped = np.asarray([peaks[group == key].max() for key in sorted(set(group))])
        for kind, units in enumerate((peaks, grouped)):
            for alpha_i, alpha in enumerate(ALPHAS):
                rank = int(np.ceil((len(units) + 1) * (1 - alpha)))
                expected = np.sort(units)[rank - 1] if rank <= len(units) else np.inf
                np.testing.assert_allclose(expected, stored["thresholds"][kind, alpha_i, method], rtol=3e-5, atol=3e-4)
        for local in (0, len(stored["test_rows"]) // 2, len(stored["test_rows"]) - 1):
            global_row = int(stored["test_rows"][local])
            row = frame.loc[global_row]
            source = ROOT / "VLA_MUI_HUB" / row.source
            summaries = sorted(json.loads((source / "client/summaries.json").read_text()), key=lambda r: r["episode_index"])
            position = next(i for i, r in enumerate(summaries) if r["episode_index"] == row.episode)
            start = sum(r["inference_calls"] for r in summaries[:position])
            store = zarr.open_group(str(source / "server/routes.zarr"), mode="r")
            raw_routes = np.asarray(store["hb_router_probs"][start:start + row.length, :, 9])
            extracted = route_features(raw_routes)["stats"]
            np.testing.assert_array_equal(extracted, stats[global_row, :row.length])
            monitor = TrainFreeMoEMonitor(profile, row.checkpoint)
            outputs = []
            begin = time.perf_counter()
            for probability in raw_routes:
                outputs.append(monitor.update(probability))
            timings.append((time.perf_counter() - begin) / row.length)
            actual = np.asarray([r["score"] for r in outputs])
            np.testing.assert_allclose(actual, stored["scores"][method, local, :row.length], rtol=2e-5, atol=2e-5)
            alpha_i = list(ALPHAS).index(0.05)
            if outputs[-1]["first_alarm_query"] != int(stored["first"][0, alpha_i, method, local]):
                raise AssertionError("streaming and batch first alarm differ")
            monitor.reset()
            if monitor.query != 0 or monitor.first_alarm_query != -1:
                raise AssertionError("monitor state not reset")
            checks.append({"fold": info["fold"], "global_row": global_row, "queries": int(row.length)})
        print(f"VERIFIED {info['fold']}: independent calibration and 3 raw-route replays", flush=True)
    calibration = pd.concat([pd.read_csv(p) for p in (output / "calibration").glob("*.csv")], ignore_index=True)
    if (calibration.exceedances > calibration.calibration_units + 1 - calibration["rank"]).any():
        raise AssertionError("calibration budget exceeded")
    write_json(output / "verification.json", {"source_hashes_checked": len(manifest["sources"]),
        "artifact_hashes_checked": len(manifest["artifacts"]), "calibration_rank_checks": len(calibration),
        "independent_primary_threshold_checks": 12 * 2 * len(ALPHAS), "raw_replays": checks,
        "extraction_prefix_checks": sum(len(a["prefix_checks"]) for a in audits),
        "median_primary_monitor_ms_per_query": float(np.median(timings) * 1000),
        "monitor_seconds_exclude_loading_and_disk": True,
        "verifier_sha256": digest(Path(__file__)), "monitor_sha256": digest(HERE / "monitor.py")})


if __name__ == "__main__":
    main()
