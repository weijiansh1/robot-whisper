#!/usr/bin/env python3
"""Replay real route tensors and time CPU feature/readout components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from compare_alarm_methods import HERE, ROOT, SAFE, RESULTS, audit
import numpy as np
import pandas as pd
import zarr
from threadpoolctl import threadpool_limits
from intrinsic_guard_monitor import IntrinsicGuardMonitor
from knn import ReferenceScorer, dynamics
from kmeans_reference import score_clusters
sys.path.insert(0, str(SAFE / "temporal_fusion"))
from fusion import flow_speed, raw_v8_features


def timed(function, repeats=200):
    for _ in range(5):
        function()
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        elapsed.append((time.perf_counter_ns() - started) / 1e6)
    return dict(repeats=repeats, median_ms=float(np.median(elapsed)), p95_ms=float(np.quantile(elapsed, .95)))


def run(output):
    frame = pd.read_csv(RESULTS / "round9_full_corpus/index.csv")
    cache = audit.archive(RESULTS / "round3_safe/v7/v7_inputs.npz")
    raw_cache = audit.archive(RESULTS / "round7_temporal_fusion/v8_inputs.npz")["raw"]
    manifest = json.loads((RESULTS / "round9_full_corpus/sealed_manifest.json").read_text())
    results = []
    for suite, part in frame.groupby("suite", sort=True):
        fold = next(item for item in manifest["folds"] if item["suite"] == suite)
        row = part.loc[part.task.isin(fold["test_tasks"]) & (part.length >= 14)].iloc[0]
        directory = ROOT / "VLA_MUI_HUB" / row.source
        summaries = sorted(json.loads((directory / "client/summaries.json").read_text()), key=lambda item: item["episode_index"])
        offset = 0
        for summary in summaries:
            if summary["episode_index"] == row.episode:
                break
            offset += summary["inference_calls"]
        assert summary["episode_index"] == row.episode
        store = zarr.open_group(str(directory / "server/routes.zarr"), mode="r")
        probability = np.asarray(store["hb_router_probs"][offset:offset + row.length])
        np.testing.assert_array_equal(store["episode_id"][offset:offset + row.length], row.episode)
        assert probability.shape == (row.length, 8, 10, 11, 32)
        profile = audit.archive(RESULTS / "round9_full_corpus/profiles" / (fold["fold"] + ".npz"))
        assert str(profile["checkpoint"]) == row.checkpoint
        cluster_profiles = {
            count: audit.archive(RESULTS / "round11_kmeans/profiles" / (fold["fold"] + "_c%d.npz" % count))
            for count in (4, 32)
        }
        previous, history = None, []
        extracted = {key: [] for key in ("mobility", "acceleration", "periodicity")}
        extra = []
        probe = None
        for query, current in enumerate(probability):
            if query == 12:
                probe = (current.copy(), previous.copy(), [value.copy() for value in history])
            mobility, acceleration, periodicity, final = IntrinsicGuardMonitor._query_features(current, previous, history)
            for name, value in zip(extracted, (mobility, acceleration, periodicity)):
                extracted[name].append(value)
            extra.append(raw_v8_features(flow_speed(current)))
            previous = final
            history.append(final[4:].reshape(40, 32))
        errors = {}
        for name, values in extracted.items():
            expected = cache[name][row.name, :row.length]
            actual = np.asarray(values)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-6, equal_nan=True)
            errors[name] = float(np.nanmax(np.abs(actual - expected)))
        np.testing.assert_allclose(extra, raw_cache[row.name, :row.length], rtol=2e-4, atol=3e-6)
        errors["v8_raw"] = float(np.max(np.abs(extra - raw_cache[row.name, :row.length])))
        vector, _ = dynamics(cache["mobility"][row.name:row.name + 1, :row.length],
            cache["acceleration"][row.name:row.name + 1, :row.length],
            cache["periodicity"][row.name:row.name + 1, :row.length], float(profile["periodicity_scale"]))
        current = (vector[:, 12].astype(np.float64) - profile["dynamic_center"]) / profile["dynamic_scale"]
        scorer = ReferenceScorer(profile)
        scorer.neighbors("success_dynamic", current)

        def base_features():
            IntrinsicGuardMonitor._query_features(*probe)
            raw_v8_features(flow_speed(probe[0]))

        results.append(dict(suite=suite, source=row.source, episode=int(row.episode), queries=int(row.length),
            profile=fold["fold"], feature_max_absolute_errors=errors,
            raw_feature_derivation_cpu=timed(base_features),
            knn20_search_cpu=timed(lambda: scorer.neighbors("success_dynamic", current)),
            kmeans_c4_three_scores_cpu=timed(lambda: score_clusters(current, cluster_profiles[4])),
            kmeans_c32_three_scores_cpu=timed(lambda: score_clusters(current, cluster_profiles[32])),
            knn_reference_array_bytes=int(profile["success_dynamic"].nbytes),
            kmeans_c4_centers_radii_bytes=int(cluster_profiles[4]["centers"].nbytes + cluster_profiles[4]["radii"].nbytes),
            kmeans_c32_centers_radii_bytes=int(cluster_profiles[32]["centers"].nbytes + cluster_profiles[32]["radii"].nbytes)))
        print("VERIFIED MoE features: %s, %d real queries" % (suite, row.length), flush=True)
    payload_bytes = {
        "hb_native_probs_float16_8_10_11_32": 8 * 10 * 11 * 32 * 2,
        "hb_native_ids_uint8_8_10_11_4": 8 * 10 * 11 * 4,
        "hb_effective_ids_uint8_8_10_11_4": 8 * 10 * 11 * 4,
        "hb_native_weights_float32_8_10_11_4": 8 * 10 * 11 * 4 * 4,
        "hb_effective_weights_float32_8_10_11_4": 8 * 10 * 11 * 4 * 4,
        "base_features_float32_mobility8_acceleration1_periodicity1_flow_speed72": 82 * 4,
    }
    audit.write_json(output, dict(model_queries=0, hidden_capture=False, gpu_compute=False,
        real_cached_queries=sum(row["queries"] for row in results), suites=results,
        measured_scope="warm single-thread CPU feature derivation and scorer calls; excludes GPU gate hook, transfer, temporal smoothing, RPC, VLA inference and environment",
        payload_bytes_per_query=payload_bytes, payload_total_bytes_per_query=sum(payload_bytes.values()),
        payload_excludes="AS, actions, flow noise, physical state, metadata, alarm snapshots and compression",
        byte_saving_when_native_equals_effective="explicit schema alias can avoid duplicate IDs/weights on unmodified gates",
        source_sha256=audit.digest(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "design/moe_readout_probe.json")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output.resolve())
