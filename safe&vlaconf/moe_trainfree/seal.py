"""Build all fixed experiment predictions before opening outcome labels."""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core import (
    ROOT, SEED, BUDGETS, META_FIELDS, METHOD_NAMES, ONLINE_NAMES, PRIMARY,
    build_scores, budget_threshold, digest, first_from_score, fit_stats,
    subset, v7_alarms, write_json,
)

HERE = Path(__file__).resolve().parent
STEP = ROOT / "moe-flow-semantics-0906/results/step_profiles"
LAYER = ROOT / "moe-v4-0904/results/layerwise_mobility"
ROUTE = ROOT / "double-selete/trainfree/results"
SOURCES = {
    "development_main": ("main_reference.npz", "online_multihead_hub"),
    "development_extra": ("extra_reference.npz", "online_multihead_hub_external"),
    "external_8b": ("external_8b.npz", "online_precision_cascade_external"),
}


def load_archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_data(cohort: str, source_paths: set[Path]) -> dict:
    layer_name, route_dir = SOURCES[cohort]
    paths = (LAYER / layer_name, ROUTE / route_dir / "unlabeled_query_features.npz",
             STEP / f"{cohort}_index.npz", STEP / f"{cohort}_metrics.npy")
    source_paths.update(paths)
    layer, route, index = [load_archive(p) for p in paths[:3]]
    for field in ("task_names", "task_index", "episode", "init_state_id",
                  "flow_noise_seed", "length", "valid"):
        if not np.array_equal(layer[field], route[field]) or not np.array_equal(layer[field], index[field]):
            raise ValueError(f"cache alignment failure: {cohort}/{field}")
    valid = layer["valid"].astype(bool)
    if not np.array_equal(valid, np.arange(valid.shape[1])[None] < layer["length"][:, None]):
        raise ValueError("non-prefix validity mask")
    metrics = np.load(paths[3], mmap_mode="r")
    if metrics.shape != (*valid.shape, 8, 10, 8):
        raise ValueError(f"unexpected step metric shape: {metrics.shape}")
    names = index["metric_names"].astype(str).tolist()
    route_names = route["feature_names"].astype(str).tolist()

    def metric(name, layers=slice(4, 8), flow=9):
        return np.asarray(metrics[:, :, layers, flow, names.index(name)]).mean(axis=2)

    def route_feature(name):
        return route["features"][:, :, route_names.index(name)]

    path = np.asarray(metrics[:, :, :, 1:, names.index("flow_speed")]).sum(axis=3)
    back_path = path[:, :, 4:].mean(axis=2)
    front_path = path[:, :, :4].mean(axis=2)
    raw = np.stack((metric("token_entropy"), route_feature("top12_margin"),
                    metric("token_differentiation"), back_path,
                    np.log((back_path + 1e-6) / (front_path + 1e-6)),
                    metric("action_consensus"), metric("token_entropy", slice(0, 4)),
                    metric("token_entropy", flow=0),
                    metric("token_differentiation", slice(0, 4))), axis=-1)
    if not np.isfinite(raw[valid]).all():
        raise ValueError(f"missing raw metrics on valid queries: {cohort}")
    data = {k: layer[k] for k in ("episode", "init_state_id", "flow_noise_seed", "length")}
    data["task"] = layer["task_names"].astype(str)[layer["task_index"]]
    data["run_id"] = np.repeat(str(layer["run_id"]), len(valid))
    data.update(raw=np.where(valid[:, :, None], raw, np.nan).astype(np.float32),
                mobility=layer["mobility"], acceleration=route_feature("route_acceleration"),
                periodicity=route_feature("lag_periodicity"), valid=valid)
    print(f"Loaded {cohort}: {len(valid)} episodes, {valid.sum()} queries", flush=True)
    return data


def splits(reference, test):
    nr, nt = len(reference["valid"]), len(test["valid"])
    all_ref, all_test = np.arange(nr), np.arange(nt)
    yield "cohort_transfer", [("all", all_ref, all_test)]
    yield "state_heldout", [
        (str(f), all_ref[reference["init_state_id"] % 5 != f],
         all_test[test["init_state_id"] % 5 == f]) for f in range(5)]
    tasks = np.union1d(reference["task"], test["task"])
    task_fold = {}
    suites = np.unique([t.split("/", 1)[0] for t in tasks])
    for suite in suites:
        for i, task in enumerate(t for t in tasks if t.startswith(suite + "/")):
            task_fold[task] = i % 5
    rf = np.asarray([task_fold[t] for t in reference["task"]])
    tf = np.asarray([task_fold[t] for t in test["task"]])
    yield "task_heldout", [(str(f), all_ref[rf != f], all_test[tf == f]) for f in range(5)]
    rs = np.asarray([t.split("/", 1)[0] for t in reference["task"]])
    ts = np.asarray([t.split("/", 1)[0] for t in test["task"]])
    yield "suite_heldout", [(s, all_ref[rs != s], all_test[ts == s]) for s in suites]
    order = np.random.default_rng(SEED).permutation(nr)
    for size in (64, 256, 1024, 4096):
        yield f"reference_{size}", [(str(size), np.sort(order[:size]), all_test)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/round1")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_paths: set[Path] = set()
    main_data = load_data("development_main", source_paths)
    extra = load_data("development_extra", source_paths)
    reference = {k: np.concatenate((v, extra[k]), axis=0) for k, v in main_data.items()}
    test = load_data("external_8b", source_paths)
    del main_data, extra
    for offset, data in enumerate((reference, test)):
        rng = np.random.default_rng(SEED + offset)
        data["random"] = np.where(data["valid"], rng.random(data["valid"].shape), np.nan).astype(np.float32)
    if len(reference["valid"]) != 16000 or len(test["valid"]) != 15600:
        raise ValueError("unexpected corpus sizes")
    for name, data in (("reference", reference), ("test", test)):
        np.savez_compressed(output / f"{name}_index.npz", **{k: data[k] for k in META_FIELDS},
                            valid=data["valid"])
    # Retain the compact inputs for timing and independent replay checks.
    np.savez_compressed(output / "test_inputs.npz", **test)
    all_alarms, profile_rows, timings, summaries = {}, {}, [], []
    for setting, folds in splits(reference, test):
        print(f"Starting {setting}: {len(folds)} calibration folds", flush=True)
        scores = np.full((len(METHOD_NAMES), *test["valid"].shape), np.nan, dtype=np.float32)
        first = np.full((len(BUDGETS), len(ONLINE_NAMES) + 1, len(test["valid"])), -2, dtype=np.int16)
        fold_ids = np.full(len(test["valid"]), -1, dtype=np.int16)
        profiles = []
        for fold_index, (fold_name, ref_rows, test_rows) in enumerate(folds):
            if np.any(fold_ids[test_rows] >= 0):
                raise ValueError("test fold overlap")
            ref, target = subset(reference, ref_rows), subset(test, test_rows)
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                stats = fit_stats(ref)
            reference_scores = build_scores(ref, stats)
            begin_apply = time.perf_counter()
            target_scores = build_scores(target, stats)
            apply_seconds = time.perf_counter() - begin_apply
            scores[:, test_rows] = target_scores
            fold_ids[test_rows] = fold_index
            profile = {"fold": str(fold_name), "reference_rows": ref_rows,
                       "test_rows": test_rows, "stats": stats, "thresholds": {}, "v7": {}}
            for method_i, name in enumerate(ONLINE_NAMES):
                score_i = METHOD_NAMES.index(name)
                reference_score, target_score = reference_scores[score_i], target_scores[score_i]
                for budget_i, budget in enumerate(BUDGETS):
                    threshold = budget_threshold(reference_score, ref["valid"], budget)
                    ref_first = first_from_score(reference_score, threshold, ref["valid"])
                    count = int((ref_first >= 0).sum())
                    if count > int(np.floor(len(ref_rows) * budget)):
                        raise AssertionError("reference alarm budget exceeded")
                    first[budget_i, method_i, test_rows] = first_from_score(target_score, threshold, target["valid"])
                    profile["thresholds"][f"{name}|{budget}"] = {
                        "threshold": threshold, "reference_alarms": count,
                        "reference_episodes": len(ref_rows),
                        "reference_score_coverage": float(np.isfinite(reference_score).any(axis=1).mean())}
            for budget_i, budget in enumerate(BUDGETS):
                try:
                    guard, fit = v7_alarms(ref, target, budget)
                except ValueError as exc:
                    if "finite reference" not in str(exc) and "at least" not in str(exc):
                        raise
                    profile["v7"][str(budget)] = {"unavailable": str(exc)}
                else:
                    first[budget_i, -1, test_rows] = guard
                    profile["v7"][str(budget)] = {"profile": asdict(fit.profile), "audit": fit.audit}
            profiles.append(profile)
            timings.append({"setting": setting, "fold": str(fold_name),
                            "reference_episodes": len(ref_rows), "test_episodes": len(test_rows),
                            "fold_seconds": time.perf_counter() - start,
                            "all_candidate_score_apply_seconds": apply_seconds,
                            "test_queries": int(target["valid"].sum())})
            print(f"  sealed fold {fold_name}: reference={len(ref_rows)}, test={len(test_rows)}, "
                  f"{time.perf_counter()-start:.1f}s; outcomes not opened", flush=True)
        if np.any(fold_ids < 0):
            raise ValueError("incomplete test fold coverage")
        np.savez_compressed(output / f"{setting}_scores.npz", scores=scores,
                            method_names=np.asarray(METHOD_NAMES), fold_ids=fold_ids)
        all_alarms[setting] = first
        write_json(output / f"{setting}_profiles.json", profiles)
        summaries.append({"setting": setting, "folds": len(folds), "test_episodes": len(fold_ids)})
        del scores, reference_scores, target_scores
    np.savez_compressed(output / "first_alarms.npz", **all_alarms,
                        method_names=np.asarray((*ONLINE_NAMES, PRIMARY)), budgets=np.asarray(BUDGETS))
    write_json(output / "timings.json", timings)
    source_paths.update((HERE / "core.py", Path(__file__).resolve(), HERE / "PROTOCOL_ZH.md",
                         ROOT / "moe-v7-0905/method/intrinsic_guard_monitor.py",
                         ROOT / "moe-v7-0905/method/unlabeled_budget_calibration.py"))
    sources = {}
    for path in sorted(source_paths):
        print(f"Hashing source {path.name}", flush=True)
        sources[str(path.resolve())] = digest(path)
    artifacts = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "sealed_manifest.json", {
        "schema": "himoe.safe_vlaconf.trainfree.v1", "seed": SEED,
        "reference_episodes": len(reference["valid"]), "test_episodes": len(test["valid"]),
        "settings": summaries, "method_names": METHOD_NAMES, "primary": PRIMARY,
        "primary_budget": 0.03, "budgets": BUDGETS,
        "outcome_labels_opened": False, "model_training": False,
        "inherited_design_outcome_informed": True, "pristine_holdout": False,
        "sources": sources, "artifacts": artifacts,
        "elapsed_seconds": time.perf_counter() - started})
    print(f"SEALED {output}; elapsed={time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
