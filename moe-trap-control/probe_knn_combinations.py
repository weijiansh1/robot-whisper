#!/usr/bin/env python3
"""Compare KNN combinations using existing global references and thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from compare_frozen_alarm_methods import (
    HERE, ROOT, RESULTS, audit, guard, metrics, load_profiles,
    ReferenceScorer, CosineReference, normalized_vectors,
)

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

METHODS = ("euclidean", "cosine", "euclidean_OR_cosine", "euclidean_AND_cosine_latched",
           "cosine_neighbors_euclidean_score", "euclidean_neighbors_cosine_score")


def hybrids(x, reference):
    finite = np.isfinite(x).all(-1)
    values = x[finite]
    out = np.full((4, *finite.shape), np.nan, np.float32)
    if not finite.any():
        return out
    bank = reference["success_dynamic"]
    edist, eids = ReferenceScorer(reference).neighbors("success_dynamic", values)
    cdist, cids = CosineReference(bank).neighbors(values)
    out[0, finite], out[1, finite] = edist.mean(axis=1), cdist.mean(axis=1)
    out[2, finite] = np.linalg.norm(bank[cids] - values[:, None], axis=-1).mean(axis=1)
    neighbors = bank[eids]
    cosine = np.clip(1 - np.einsum("nkd,nd->nk", neighbors, values) /
                     (np.linalg.norm(neighbors, axis=-1) * np.linalg.norm(values, axis=-1)[:, None]), 0, 2)
    out[3, finite] = cosine.mean(axis=1)
    return out


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parameters, reference, _, manifest = load_profiles(source / "profiles", source)
    cache_path = RESULTS / "round3_safe/v7/v7_inputs.npz"
    inputs = json.loads((source / "input_verification.json").read_text())
    assert audit.digest(cache_path) == inputs["inputs"][str(cache_path.relative_to(ROOT))]
    sealed = json.loads((source / "verification.json").read_text())
    for name in ("index.csv", "scores_and_alarms.npz", "first_alarms.csv"):
        assert audit.digest(source / name) == sealed["artifacts"][name]
    frame = pd.read_csv(source / "index.csv")
    cache = audit.archive(cache_path)
    original = audit.archive(source / "scores_and_alarms.npz")
    valid = np.arange(52)[None] < frame.length.to_numpy()[:, None]
    np.testing.assert_array_equal(valid, cache["valid"])
    np.testing.assert_array_equal(valid, original["valid"])
    e_index = list(original["geometry_methods"]).index("knn20")
    c_index = list(original["geometry_methods"]).index("cosine_knn20")
    threshold = parameters["geometry"]["thresholds"]
    e_tau, c_tau = (threshold[name]["threshold"] for name in ("knn20", "cosine_knn20"))
    contract = dict(methods=METHODS, source=str(source.resolve()),
        parameters_sha256=audit.digest(source / "profiles/parameters.json"),
        profile_manifest_sha256=audit.digest(source / "profiles/manifest.json"),
        neighbor_count=20, bank_points=len(reference["success_dynamic"]),
        euclidean_threshold=e_tau, cosine_threshold=c_tau,
        hybrid_thresholds="reuse the threshold of the scoring metric; no new calibration",
        score_changes_do_not_preserve_original_false_alarm_budget=True,
        refit_reference=False, refit_scaling=False, refit_threshold=False,
        hidden_capture=False, model_queries=0, gpu_compute=False, pro_plus_results=False,
        a_contains_reference_and_calibration=True, new_blind_test=False)
    audit.write_json(output / "contract.json", contract)
    first = np.full((len(METHODS), len(frame)), -1, np.int16)
    first[0] = original["first"][list(original["methods"]).index("knn20")]
    first[1] = original["first"][list(original["methods"]).index("cosine_knn20")]
    first[2] = guard.union(first[0], first[1])
    first[3] = guard.first_and(first[0], first[1])
    scores = np.full((2, len(frame), 52), np.nan, np.float32)
    checked_queries = 0
    for start in range(0, len(frame), 2000):
        rows = np.arange(start, min(start + 2000, len(frame)))
        x = normalized_vectors(cache, rows, reference)
        values = hybrids(x, reference)
        np.testing.assert_array_equal(values[0], original["scores"][e_index, rows])
        np.testing.assert_array_equal(values[1], original["scores"][c_index, rows])
        scores[:, rows] = values[2:]
        for mi, tau in enumerate((e_tau, c_tau)):
            first[4 + mi, rows] = guard.first_alarm(scores[mi, rows], valid[rows], tau)
        finite = np.isfinite(x).all(-1)
        assert np.all(values[2, finite] + 1e-6 >= values[0, finite])
        assert np.all(values[3, finite] + 1e-6 >= values[1, finite])
        checked_queries += int(finite.sum())
        print(f"FIXED COMBINATIONS {rows[-1] + 1}/{len(frame)}; {time.perf_counter() - started:.1f}s", flush=True)

    rows = np.linspace(0, len(frame) - 1, 32, dtype=int)[::-1]
    x = normalized_vectors(cache, rows, reference)
    np.testing.assert_array_equal(hybrids(x, reference)[2:], scores[:, rows])
    bank = reference["success_dynamic"]
    points = np.argwhere(np.isfinite(x).all(-1))
    for episode, query in points[np.linspace(0, len(points) - 1, 32, dtype=int)]:
        value = x[episode, query]
        e_distance = np.linalg.norm(bank - value, axis=1)
        c_distance = np.clip(1 - (bank @ value) / (np.linalg.norm(bank, axis=1) * np.linalg.norm(value)), 0, 2)
        expected = (e_distance[np.argsort(c_distance)[:20]].mean(), c_distance[np.argsort(e_distance)[:20]].mean())
        np.testing.assert_allclose(scores[:, rows[episode], query], expected, rtol=2e-6, atol=1e-7)
    for mi, tau in enumerate((e_tau, c_tau)):
        crossing = np.isfinite(scores[mi]) & valid & (scores[mi] > tau)
        expected = np.where(crossing.any(axis=1), crossing.argmax(axis=1), -1)
        np.testing.assert_array_equal(first[4 + mi], expected)
    for values in first:
        assert ((values == -1) | ((values >= 0) & (values < frame.length.to_numpy()))).all()

    metrics_rows, overlap_rows, changed_rows = [], [], []
    for cohort, part in (("all", frame), ("A", frame.loc[frame.run_id.eq(audit.RUN_A)]),
                         ("B", frame.loc[frame.run_id.eq(audit.RUN_B)])):
        selected = first[:, part.index]
        failure = part.failure.to_numpy(bool)
        for mi, method in enumerate(METHODS):
            for cutoff in (7, 10, 13, 20, 51):
                metrics_rows.append(dict(cohort=cohort, method=method, query_cutoff=cutoff,
                    **metrics(selected[mi], part, cutoff)))
        e, c = selected[:2]
        for category, mask in (
            ("cosine_only", (c >= 0) & (e < 0)),
            ("euclidean_only", (e >= 0) & (c < 0)),
            ("both", (e >= 0) & (c >= 0)),
            ("both_cosine_earlier", (c >= 0) & (e >= 0) & (c < e)),
        ):
            overlap_rows.append(dict(cohort=cohort, category=category,
                failures=int((mask & failure).sum()), successes=int((mask & ~failure).sum())))
    for mi in (2, 4):
        changed = first[mi] != first[0]
        for row in np.flatnonzero(changed):
            changed_rows.append(dict(global_row=int(row), source=frame.iloc[row].source,
                episode=int(frame.iloc[row].episode), run_id=frame.iloc[row].run_id,
                task=frame.iloc[row].task, failure=bool(frame.iloc[row].failure), method=METHODS[mi],
                euclidean_first=int(first[0, row]), combined_first=int(first[mi, row]),
                newly_detected=bool(first[0, row] < 0)))
    pd.DataFrame(metrics_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output / "overlap.csv", index=False)
    pd.DataFrame(changed_rows).to_csv(output / "changed_alarms.csv", index=False)
    np.savez_compressed(output / "scores_and_alarms.npz", methods=np.asarray(METHODS),
        first=first, hybrid_methods=np.asarray(METHODS[4:]), hybrid_scores=scores,
        thresholds=np.asarray((e_tau, c_tau)), valid=valid)
    for name, digest in manifest["files"].items():
        assert audit.digest(source / "profiles" / name) == digest
    verification = dict(passed=True, trajectory_decisions=int(first.size),
        baseline_query_scores_exactly_reproduced=2 * checked_queries,
        independent_hybrid_distance_points=32, batch_order_episodes=32,
        source_profiles_unchanged=True, elapsed_seconds=time.perf_counter() - started,
        source_code_sha256=audit.digest(Path(__file__)),
        artifacts={p.name: audit.digest(p) for p in output.iterdir() if p.is_file()})
    audit.write_json(output / "verification.json", verification)
    print("VERIFIED: source reference, scaling, k and thresholds all unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "design/frozen_alarm_comparison_20260908")
    parser.add_argument("--output", type=Path, default=HERE / "design/knn_combinations_20260908")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.output)
