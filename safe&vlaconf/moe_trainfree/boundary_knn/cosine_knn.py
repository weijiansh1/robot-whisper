"""Frozen-reference cosine ablation, with sealed scores and independent checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from knn import HERE, ROOT, PRIMARY, ReferenceScorer, dynamics
from core import conformal_threshold, digest, first_alarm, trajectory_peak, write_json

METHODS = ("euclidean", "cosine")
NORM_FLOOR = 1e-12
CASE_ROW = 15023
CASE_FOLD = "libero_long_20260907"
DEFAULT_OUTPUT = HERE.parent / "results/round8_cosine_knn"


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class CosineReference:
    def __init__(self, bank):
        self.bank = np.asarray(bank, dtype=np.float64)
        self.bank_norms = self.check_norms(self.bank)
        self.index = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute", n_jobs=1).fit(self.bank)

    @staticmethod
    def check_norms(values):
        if values.ndim != 2 or values.shape[1] != 10 or not np.isfinite(values).all():
            raise ValueError("expected finite, standardized 10D vectors")
        norms = np.linalg.norm(values, axis=1)
        if (norms <= NORM_FLOOR).any():
            raise ValueError("cosine direction is undefined for a near-zero vector")
        return norms

    def neighbors(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.check_norms(values)
        distances, identities = [], []
        for start in range(0, len(values), 512):
            distance, identity = self.index.kneighbors(values[start:start + 512])
            distances.append(distance)
            identities.append(identity)
        return np.concatenate(distances), np.concatenate(identities)

    def score(self, vectors):
        finite = np.isfinite(vectors).all(-1)
        scores = np.full(finite.shape, np.nan, np.float32)
        if finite.any():
            distance, _ = self.neighbors(vectors[finite])
            scores[finite] = distance.mean(1)
        return scores


def normalized_vectors(cache, rows, profile):
    dynamic, _ = dynamics(cache["mobility"][rows], cache["acceleration"][rows],
        cache["periodicity"][rows], float(profile["periodicity_scale"]))
    dynamic[~cache["valid"][rows]] = np.nan
    return (dynamic.astype(np.float64) - profile["dynamic_center"]) / profile["dynamic_scale"]


def calibrate(cal_scores, test_scores, labels, frame, alphas):
    success = labels == 0
    group_ids = (frame.loc[success, "task"] + "|" + frame.loc[success, "init_state_id"].astype(str)).to_numpy()
    thresholds = np.empty((2, len(alphas), len(METHODS)), np.float64)
    first = np.empty((*thresholds.shape, test_scores.shape[1]), np.int16)
    audit = []
    for method_i, method in enumerate(METHODS):
        peaks = trajectory_peak(cal_scores[method_i, success])
        grouped = np.asarray([peaks[group_ids == key].max() for key in sorted(set(group_ids))])
        for kind_i, (kind, units) in enumerate((("episode", peaks), ("task_init", grouped))):
            for alpha_i, alpha in enumerate(alphas):
                tau, rank = conformal_threshold(units, float(alpha))
                thresholds[kind_i, alpha_i, method_i] = tau
                first[kind_i, alpha_i, method_i] = first_alarm(test_scores[method_i], tau)
                audit.append(dict(method=method, calibration=kind, alpha=float(alpha),
                    threshold=tau, rank=rank, units=len(units), exceedances=int((units > tau).sum())))
    return thresholds, first, audit


def save_case(output, index, profile, data, vectors):
    local = int(np.flatnonzero(data["test_rows"] == CASE_ROW)[0])
    length = int(index.loc[CASE_ROW, "length"])
    x = vectors[local, :length]
    valid = np.isfinite(x).all(1)
    alpha_i = int(np.flatnonzero(np.isclose(data["alphas"], .05))[0])
    records = pd.DataFrame({"query": np.arange(length), "executed_actions_before_query": 10 * np.arange(length),
                            "standardized_vector_norm": np.linalg.norm(x, axis=1)})
    neighbors = []
    for method_i, method in enumerate(METHODS):
        score = data["scores"][method_i, local, :length]
        tau = float(data["thresholds"][1, alpha_i, method_i])
        records[f"{method}_score"] = score
        records[f"{method}_threshold"] = tau
        records[f"{method}_ratio"] = score / tau
        records[f"{method}_current_exceeds"] = np.isfinite(score) & (score > tau)
        first = int(data["first"][1, alpha_i, method_i, local])
        records[f"{method}_alarm_latched"] = (np.arange(length) >= first) & (first >= 0)
        if method == "cosine":
            distances, identities = CosineReference(profile["success_dynamic"]).neighbors(x[valid])
        else:
            distances, identities = ReferenceScorer(profile).neighbors("success_dynamic", x[valid])
        np.testing.assert_allclose(distances.mean(1), score[valid], rtol=2e-6, atol=2e-7)
        for q, ds, ids in zip(np.flatnonzero(valid), distances, identities):
            for rank, (distance, identity) in enumerate(zip(ds, ids), start=1):
                ref_row = int(profile["success_global_rows"][identity])
                neighbors.append(dict(query=int(q), method=method, rank=rank, distance=float(distance),
                    reference_global_row=ref_row, reference_query=int(profile["success_queries"][identity]),
                    reference_task=index.loc[ref_row, "task"], reference_episode=int(index.loc[ref_row, "episode"])))
    records.to_csv(output / "long_episode_223_queries.csv", index=False)
    pd.DataFrame(neighbors).to_csv(output / "long_episode_223_neighbors.csv", index=False)


def score_all(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "profiles", "calibration"):
        (output / name).mkdir()
    original_manifest = json.loads((parent / "sealed_manifest.json").read_text())
    cache_path = HERE.parent / "results/round3_safe/v7/v7_inputs.npz"
    inputs = {str(path.relative_to(ROOT)): digest(path)
              for path in (parent / "sealed_manifest.json", parent / "index.csv", cache_path)}
    for path in (parent / "index.csv", cache_path):
        expected = (original_manifest["artifacts"]["index.csv"] if path.name == "index.csv"
                    else original_manifest["inputs"][str(path.relative_to(ROOT))])
        assert digest(path) == expected, path
    source_paths = (Path(__file__), HERE / "COSINE_PROTOCOL_ZH.md", HERE / "knn.py",
                    HERE.parent / "feature_geometry/analyze.py", HERE.parent / "safe_protocol/core.py",
                    ROOT / "moe-v7-0905/method/intrinsic_guard_monitor.py")
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    for path, expected in original_manifest["sources"].items():
        assert digest(ROOT / path) == expected, path
    write_json(output / "scoring_contract.json", dict(methods=METHODS, sources=sources,
        reference="identical Round 5 successful chunk identities", k=20, norm_floor=NORM_FLOOR,
        cosine_origin="frozen reference median, followed by frozen MAD scaling",
        new_training=False, historical_outcome_labels_for_reference_and_calibration=True,
        primary_calibration="task_init", primary_alpha=.05, test_outcomes_used_for_scoring=False,
        historically_explored_data=True, fixed_case_global_row=CASE_ROW, fixed_case_fold=CASE_FOLD))
    index = pd.read_csv(parent / "index.csv")
    cache = load_npz(cache_path)
    audits, folds = [], []
    for info in original_manifest["folds"]:
        started = time.perf_counter()
        fold = info["fold"]
        old_paths = [parent / "profiles" / f"{fold}.npz", parent / "predictions" / f"{fold}.npz"]
        for path in old_paths:
            actual = digest(path)
            assert actual == original_manifest["artifacts"][str(path.relative_to(parent))], path
            inputs[str(path.relative_to(ROOT))] = actual
        profile, old = (load_npz(path) for path in old_paths)
        rows = np.r_[old["calibration_rows"], old["test_rows"]]
        x = normalized_vectors(cache, rows, profile)
        finite = np.isfinite(x).all(-1)
        scorer = CosineReference(profile["success_dynamic"])
        cosine = scorer.score(x)
        method_i = list(old["methods"].astype(str)).index(PRIMARY)
        euclidean = np.concatenate((old["calibration_scores"][method_i], old["scores"][method_i]))
        np.testing.assert_array_equal(np.isfinite(euclidean), finite)
        ncal = len(old["calibration_rows"])
        scores = np.stack((euclidean[ncal:], cosine[ncal:]))
        calibration_scores = np.stack((euclidean[:ncal], cosine[:ncal]))
        thresholds, first, cal_audit = calibrate(calibration_scores, scores, old["calibration_labels"],
            index.iloc[old["calibration_rows"]].reset_index(drop=True), old["alphas"])
        np.testing.assert_array_equal(thresholds[:, :, 0], old["thresholds"][:, :, method_i])
        np.testing.assert_array_equal(first[:, :, 0], old["first"][:, :, method_i])
        data = {key: old[key] for key in ("reference_rows", "calibration_rows", "calibration_labels",
                                        "test_rows", "test_unseen", "alphas", "checkpoint", "seed")}
        data.update(methods=np.asarray(METHODS), scores=scores, calibration_scores=calibration_scores,
                    thresholds=thresholds, first=first)
        np.savez_compressed(output / "predictions" / f"{fold}.npz", **data)
        saved_profile = {key: profile[key] for key in ("dynamic_center", "dynamic_scale", "success_dynamic",
            "success_global_rows", "success_queries", "periodicity_scale", "checkpoint")}
        np.savez_compressed(output / "profiles" / f"{fold}.npz", **saved_profile,
                            thresholds=thresholds, methods=np.asarray(METHODS), alphas=old["alphas"])
        pd.DataFrame(cal_audit).to_csv(output / "calibration" / f"{fold}.csv", index=False)
        norms = np.linalg.norm(x[finite], axis=1)
        audit = dict(fold=fold, suite=info["suite"], seed=int(info["seed"]),
            score_queries=int(finite.sum()), bank_points=len(scorer.bank),
            bank_norm_min=float(scorer.bank_norms.min()), query_norm_min=float(norms.min()),
            query_norm_p01=float(np.quantile(norms, .01)), query_norm_p50=float(np.quantile(norms, .5)),
            query_norm_p99=float(np.quantile(norms, .99)), zero_vectors=0,
            seconds=time.perf_counter() - started)
        audits.append(audit)
        folds.append({key: info[key] for key in ("fold", "suite", "seed")})
        if fold == CASE_FOLD:
            save_case(output, index, profile, data, x[ncal:])
        print(f"SEALED {fold}: {audit['score_queries']} queries, {audit['seconds']:.1f}s", flush=True)
    pd.DataFrame(audits).to_csv(output / "norm_audit.csv", index=False)
    artifacts = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "sealed_manifest.json", dict(sources=sources, inputs=inputs, artifacts=artifacts,
        folds=folds, test_outcomes_used_for_scoring=False, historically_explored_data=True))
    print("ALL COSINE PREDICTIONS SEALED", flush=True)


def verify_all(parent, output):
    from v7_adapter import raw_episode, extract_episode
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    hash_checks = 0
    for category, prefix in (("sources", ROOT), ("inputs", ROOT), ("artifacts", output)):
        for name, expected in manifest[category].items():
            assert digest(prefix / name) == expected, name
            hash_checks += 1
    frame = pd.read_csv(parent / "outcome_alignment.csv")
    cache = load_npz(HERE.parent / "results/round3_safe/v7/v7_inputs.npz")
    oracles, raw_checks = [], []
    calibration_checks = 0
    cases_path = HERE.parent / "results/round4_geometry/cases.json"
    cases = [case for case in json.loads(cases_path.read_text()) if case["role"] == "failure"]
    pending = {case["suite"]: case for case in cases}
    for info in manifest["folds"]:
        fold = info["fold"]
        profile = load_npz(output / "profiles" / f"{fold}.npz")
        data = load_npz(output / "predictions" / f"{fold}.npz")
        old = load_npz(parent / "predictions" / f"{fold}.npz")
        for key in ("reference_rows", "calibration_rows", "test_rows", "test_unseen", "calibration_labels", "alphas"):
            np.testing.assert_array_equal(data[key], old[key])
        bank = profile["success_dynamic"]
        reference = profile["success_global_rows"]
        assert np.isin(reference, data["reference_rows"]).all()
        assert not np.isin(reference, np.r_[data["calibration_rows"], data["test_rows"]]).any()
        assert not frame.iloc[reference].failure.any()
        assert np.unique(reference, return_counts=True)[1].max() <= 8
        assert (profile["success_queries"] >= 7).all()
        assert (profile["success_queries"] < frame.iloc[reference].length.to_numpy()).all()
        for key in ("success_dynamic", "success_global_rows", "success_queries", "dynamic_center", "dynamic_scale"):
            original_profile = load_npz(parent / "profiles" / f"{fold}.npz") if key == "success_dynamic" else original_profile
            np.testing.assert_array_equal(profile[key], original_profile[key])
        success = data["calibration_labels"] == 0
        cal_frame = frame.iloc[data["calibration_rows"]].loc[success]
        groups = (cal_frame.task + "|" + cal_frame.init_state_id.astype(str)).to_numpy()
        valid = np.arange(52)[None] < frame.iloc[data["test_rows"]].length.to_numpy()[:, None]
        for method_i in range(2):
            raw = data["scores"][method_i]
            assert not np.isfinite(raw[:, :7]).any() and not np.isfinite(raw[~valid]).any()
            peaks = np.where(np.isfinite(data["calibration_scores"][method_i, success]),
                             data["calibration_scores"][method_i, success], -np.inf).max(1)
            grouped = np.asarray([peaks[groups == group].max() for group in sorted(set(groups))])
            for kind_i, units in enumerate((peaks, grouped)):
                for alpha_i, alpha in enumerate(data["alphas"]):
                    rank = int(np.ceil((len(units) + 1) * (1 - alpha)))
                    tau = np.sort(units)[rank - 1] if rank <= len(units) else np.inf
                    np.testing.assert_equal(tau, data["thresholds"][kind_i, alpha_i, method_i])
                    hit = np.isfinite(raw) & (raw > tau)
                    first = np.where(hit.any(1), hit.argmax(1), -1)
                    np.testing.assert_array_equal(first, data["first"][kind_i, alpha_i, method_i])
                    calibration_checks += 1
        for split, score_key in (("calibration", "calibration_scores"), ("test", "scores")):
            rows = data[f"{split}_rows"]
            local = np.asarray([0, len(rows) // 2, len(rows) - 1])
            x = normalized_vectors(cache, rows[local], profile)
            for j, row in enumerate(rows[local]):
                for q in sorted({7, min(14, int(frame.loc[row, "length"]) - 1), int(frame.loc[row, "length"]) - 1}):
                    if q < 7 or q >= frame.loc[row, "length"]:
                        continue
                    current = x[j, q]
                    euclidean = np.sort(np.linalg.norm(bank - current, axis=1))[:20].mean()
                    cosine = np.clip(1 - np.sum(bank * current, axis=1) /
                        (np.linalg.norm(bank, axis=1) * np.linalg.norm(current)), 0, 2)
                    expected = np.asarray([euclidean, np.sort(cosine)[:20].mean()])
                    actual = data[score_key][:, local[j], q]
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
                    oracles.append(dict(fold=fold, split=split, global_row=int(row), query=q,
                                        max_abs_error=float(np.max(np.abs(actual - expected)))))
        case = pending.get(info["suite"])
        if case and case["global_row"] in data["test_rows"]:
            row = frame.loc[case["global_row"]]
            raw = raw_episode(row)
            m, a, p = extract_episode(raw)
            dynamic, _ = dynamics(m[None], a[None], p[None], float(profile["periodicity_scale"]))
            x = (dynamic.astype(float) - profile["dynamic_center"]) / profile["dynamic_scale"]
            replay = CosineReference(bank).score(x)[0]
            position = int(np.flatnonzero(data["test_rows"] == case["global_row"])[0])
            expected = data["scores"][1, position, :row.length]
            np.testing.assert_allclose(replay, expected, rtol=5e-4, atol=3e-5, equal_nan=True)
            alpha_i = int(np.flatnonzero(np.isclose(data["alphas"], .05))[0])
            tau = data["thresholds"][1, alpha_i, 1]
            first = int(first_alarm(replay[None], tau)[0])
            assert first == data["first"][1, alpha_i, 1, position]
            for q in sorted({7, max(7, first), len(raw) - 1}):
                prefix, _ = dynamics(m[None, :q + 1], a[None, :q + 1], p[None, :q + 1], float(profile["periodicity_scale"]))
                np.testing.assert_allclose(prefix[0, -1], dynamic[0, q], rtol=0, atol=0, equal_nan=True)
            raw_checks.append(dict(fold=fold, global_row=int(case["global_row"]), queries=len(raw),
                source=str(row.source), raw_slice_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                first_alarm=first, max_abs_error=float(np.nanmax(np.abs(replay - expected)))))
            del pending[info["suite"]]
        print(f"VERIFIED {fold}", flush=True)
    assert not pending, pending
    # Degenerate directions must be rejected instead of silently becoming normal.
    for values in (np.zeros((20, 10)), np.full((20, 10), 1e-15)):
        try:
            CosineReference(values)
        except ValueError:
            pass
        else:
            raise AssertionError("near-zero bank accepted")
    write_json(output / "verification.json", dict(hash_checks=hash_checks, calibration_checks=calibration_checks,
        direct_distance_checks=oracles, raw_stream_replays=raw_checks, all_alarm_positions_match=True,
        zero_norm_rejected=True, case_selection_sha256=digest(cases_path),
        outcome_alignment_sha256=digest(parent / "outcome_alignment.csv"), verifier_sha256=digest(Path(__file__))))
    print(f"PASS: {hash_checks} hashes, {calibration_checks} calibrations, {len(oracles)} direct oracles, {len(raw_checks)} raw replays", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("score", "verify"))
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round5_knn")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (score_all if args.mode == "score" else verify_all)(args.parent.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
