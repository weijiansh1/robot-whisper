#!/usr/bin/env python3
"""Check every selected detector at its own first-alarm prefix, without future queries."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from compare_collection_alarms import ALL_METHODS, load_profiles, score_all
from collection_storage import atomic_json, digest
from continuation_experiment import HERE, METHODS, load_plan


def run(plans, output):
    selected = [load_plan(path) for path in plans]
    source = Path(selected[0]["alarm_table"]).parent
    require_same = ("alarm_table", "alarm_contract", "parent_audit")
    if any(plan[key] != selected[0][key] for plan in selected for key in require_same):
        raise ValueError("Plans refer to different native corpora")
    verification = json.loads((source / "verification.json").read_text())
    path = source / "scores_and_alarms.npz"
    if digest(path) != verification["artifacts"][path.name]:
        raise ValueError("Cached source features changed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    contract = json.loads(Path(selected[0]["alarm_contract"]).read_text())
    frozen = Path(contract["frozen_source"])
    parameters, reference, clusters, _ = load_profiles(frozen / "profiles", frozen)
    cache = {key: arrays[key] for key in ("valid", "mobility", "acceleration", "periodicity")}
    checks = []
    for method in METHODS:
        index = ALL_METHODS.index(method)
        first = arrays["first"][index]
        limits = np.where(first >= 0, first, arrays["valid"].sum(1) - 1)
        keep = np.arange(arrays["valid"].shape[1])[None] <= limits[:, None]
        prefix = {key: value.copy() for key, value in cache.items()}
        for key, value in prefix.items():
            value[~keep] = False if key == "valid" else np.nan
        raw = arrays["raw_v8"].copy()
        raw[~keep] = np.nan
        vectors, _, replayed = score_all(prefix, raw, parameters, reference, clusters)
        np.testing.assert_array_equal(replayed[index], first)
        np.testing.assert_array_equal(vectors[keep], arrays["normalized_dynamics"][keep])
        checks.append(dict(method=method, native_parents=len(first), own_alarm_prefixes=int((first >= 0).sum()), passed=True))
    result = dict(status="passed", future_queries_masked=True, reference_fit=False, threshold_fit=False,
        task_or_suite_parameters=False, source_sha256=digest(Path(__file__)),
        plans={str(path): digest(path) for path in plans}, features_sha256=digest(path),
        parameters_sha256=digest(frozen / "profiles/parameters.json"), checks=checks,
        deployment_note="indices precomputed on committed native mains; same decisions verified with every future query removed")
    atomic_json(output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Verification output already exists")
    with threadpool_limits(2):
        run(args.plans, args.output)
