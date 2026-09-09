"""Score the frozen Round5 folds with exact JA, WJ and aligned Hellinger kNN."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import time

import numba
import numpy as np
import pandas as pd

from metrics import HERE, ROOT, METHODS, ALPHAS, JaccardScorer, calibrate
from core import digest, write_json
from run import make_split


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_cache(output, frame, suite):
    audits = json.loads((output / "extraction_audit.json").read_text())
    sources = set(frame.loc[frame.suite == suite, "source"])
    probabilities, ids, rows, queries = [], [], [], []
    hashes = {}
    for audit in audits:
        if audit["source"] not in sources:
            continue
        path = output / "route_cache" / audit["output"]
        assert digest(path) == audit["cache_sha256"], path
        hashes[str(path.relative_to(ROOT))] = audit["cache_sha256"]
        cache = load(path)
        probabilities.append(cache["probabilities"])
        ids.append(cache["expert_ids"])
        rows.append(cache["global_rows"])
        queries.append(cache["queries"])
    rows, queries = np.concatenate(rows), np.concatenate(queries)
    assert np.unique(rows * 52 + queries).size == len(rows)
    assert (frame.iloc[rows].suite == suite).all()
    lookup = np.full((len(frame), 52), -1, np.int32)
    lookup[rows, queries] = np.arange(len(rows))
    expected = np.maximum(frame.loc[frame.suite == suite].length.to_numpy() - 7, 0).sum()
    assert len(rows) == expected
    return np.concatenate(probabilities), np.concatenate(ids), lookup, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE.parent / "results/round5_knn")
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round6_jaccard_knn")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    old_root, output = args.baseline.resolve(), args.output.resolve()
    numba.set_num_threads(args.threads)
    old_manifest = json.loads((old_root / "sealed_manifest.json").read_text())
    frame = pd.read_csv(old_root / "index.csv")
    assert digest(old_root / "index.csv") == old_manifest["artifacts"]["index.csv"]
    for category in ("profiles", "predictions", "calibration"):
        (output / category).mkdir(exist_ok=True)
    shutil.copyfile(old_root / "index.csv", output / "index.csv")
    sources = {str(path.relative_to(ROOT)): digest(path) for path in
        (Path(__file__), HERE / "metrics.py", HERE / "extract_routes.py", HERE / "PROTOCOL_ZH.md",
         HERE.parent / "safe_protocol/core.py", HERE.parent / "safe_protocol/run.py")}
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in
        (old_root / "sealed_manifest.json", old_root / "index.csv", output / "extraction_audit.json")}
    write_json(output / "scoring_contract.json", {"sources": sources, "inputs": inputs,
        "methods": METHODS, "k": 20, "new_model_training": False,
        "calibration": "unchanged episode/task_init rules and alphas; primary task_init 5%",
        "test_outcomes_used": False, "reference_identities": "exact Round5 successful reference episode/query points",
        "python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
        "numba_threads": args.threads})
    folds = []
    for suite_i, suite in enumerate(sorted(frame.suite.unique())):
        p, ids, lookup, cache_hashes = load_cache(output, frame, suite)
        inputs.update(cache_hashes)
        part = frame.loc[frame.suite == suite].copy()
        part["global_row"] = part.index
        part = part.reset_index(drop=True)
        for old_info in [x for x in old_manifest["folds"] if x["suite"] == suite]:
            started = time.perf_counter()
            fold = old_info["fold"]
            target = output / "predictions" / f"{fold}.npz"
            if target.exists():
                raise FileExistsError(f"predictions already exist: {target}")
            stored = []
            for directory in ("profiles", "predictions"):
                name = f"{directory}/{fold}.npz"
                path = old_root / name
                assert digest(path) == old_manifest["artifacts"][name]
                inputs[str(path.relative_to(ROOT))] = old_manifest["artifacts"][name]
                stored.append(load(path))
            old_profile, old = stored
            ref, cal, test, _, unseen = make_split(part, old_info["seed"], suite_i)
            for kind, ix in (("reference", ref), ("calibration", cal), ("test", test)):
                np.testing.assert_array_equal(old[f"{kind}_rows"], part.iloc[ix].global_row)
            np.testing.assert_array_equal(old["test_unseen"], part.iloc[test].task.isin(unseen))
            rows, queries = old_profile["success_global_rows"], old_profile["success_queries"]
            locations = lookup[rows, queries]
            assert (locations >= 0).all()
            assert np.isin(rows, old["reference_rows"]).all()
            assert not np.isin(rows, np.r_[old["calibration_rows"], old["test_rows"]]).any()
            profile = {"reference_probabilities": p[locations], "reference_ids": ids[locations],
                "success_global_rows": rows, "success_queries": queries,
                "checkpoint": old["checkpoint"], "methods": np.asarray(METHODS), "alphas": np.asarray(ALPHAS)}
            scorer = JaccardScorer(profile["reference_probabilities"], profile["reference_ids"])
            needed = np.concatenate((old["calibration_rows"], old["test_rows"]))
            episode_pos, query_pos = np.where(lookup[needed] >= 0)
            flat_locations = lookup[needed[episode_pos], query_pos]
            scores = np.full((len(METHODS), len(needed), 52), np.nan, np.float32)
            for start in range(0, len(flat_locations), 512):
                stop = min(start + 512, len(flat_locations))
                batch = flat_locations[start:stop]
                scores[:, episode_pos[start:stop], query_pos[start:stop]] = scorer.score(p[batch], ids[batch])
                if start % 4096 == 0 or stop == len(flat_locations):
                    print(f"SCORE {fold}: {stop}/{len(flat_locations)} query points, {time.perf_counter()-started:.1f}s", flush=True)
            ncal = len(old["calibration_rows"])
            thresholds, first, records = calibrate(scores[:, :ncal], old["calibration_labels"],
                frame.iloc[old["calibration_rows"]].reset_index(drop=True), scores[:, ncal:])
            profile["thresholds"] = thresholds
            profile_path = output / "profiles" / f"{fold}.npz"
            np.savez_compressed(profile_path, **profile)
            prediction = {key: old[key] for key in ("reference_rows", "calibration_rows", "calibration_labels",
                          "test_rows", "test_unseen", "checkpoint", "seed")}
            np.savez_compressed(target, **prediction, scores=scores[:, ncal:], calibration_scores=scores[:, :ncal],
                methods=np.asarray(METHODS), alphas=np.asarray(ALPHAS), thresholds=thresholds, first=first)
            for row in records:
                row["fold"] = fold
            pd.DataFrame(records).to_csv(output / "calibration" / f"{fold}.csv", index=False)
            info = {key: old_info[key] for key in ("fold", "suite", "seed", "seen_tasks", "unseen_tasks",
                    "reference_episodes", "calibration_episodes", "test_episodes")}
            info.update(reference_points=len(locations), score_queries=len(flat_locations),
                        seconds=time.perf_counter()-started, reference_identities_match_round5=True)
            folds.append(info)
            write_json(output / "profiles" / f"{fold}.json", info)
            print(f"SEALED {fold}: {info['seconds']:.1f}s", flush=True)
        del p, ids, lookup
    artifacts = {str(path.relative_to(output)): digest(path) for path in (output / "index.csv", output / "scoring_contract.json")}
    for directory in ("profiles", "predictions", "calibration"):
        artifacts.update({str(path.relative_to(output)): digest(path) for path in sorted((output / directory).iterdir())
                          if path.is_file() and path.suffix in (".npz", ".json", ".csv")})
    write_json(output / "sealed_manifest.json", {"sources": sources, "inputs": inputs, "artifacts": artifacts,
        "folds": folds, "methods": METHODS, "test_outcomes_used_for_scoring": False,
        "new_model_training": False, "historically_explored_data": True, "label_free": False})
    print("ALL JA/WJ PREDICTIONS SEALED", flush=True)


if __name__ == "__main__":
    main()
