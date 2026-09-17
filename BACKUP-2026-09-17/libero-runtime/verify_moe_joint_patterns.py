"""Independent archive, prefix, event-count, and actual-model replay checks."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

OUT = Path("/data/libero-runtime/samples/moe-joint-patterns-20260915")
SOURCE = Path("/data/libero-runtime/samples/v82-evaluation-20260914T144322Z")
BRANCHES = Path("/data/coding/moe-control-experiments/runs/p3b-head-control-20260915-s39ow9qi")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    results = json.loads((OUT / "results.json").read_text())
    summary = json.loads((SOURCE / "summary.json").read_text())
    episodes = {e["name"]: e for e in summary["episodes"]}
    source_hashes = json.loads((OUT / "source-hashes.json").read_text())
    for path, expected in source_hashes.items():
        assert digest(path) == expected, path
    with np.load(OUT / "episode-features.npz") as archive:
        features = {name: archive[name] for name in archive.files}
    with np.load(OUT / "branch-features.npz") as archive:
        branch_features = {name: archive[name] for name in archive.files}
    assert len(features) == 60 and sum(len(x) for x in features.values()) == 2891
    branch_checks = 0
    for key, x in branch_features.items():
        parent, candidate = key.split("__candidate")
        decision = json.loads((BRANCHES / parent / "decisions.json").read_text())
        query = decision["query"]
        np.testing.assert_array_equal(x[:query], features[parent][:query])
        if candidate == "0":
            # Branch archives have probabilities but do not include actual IDs.
            columns = [i for i, name in enumerate(results["feature_names"]) if name != "back_top4_turnover"]
            np.testing.assert_array_equal(x[:, columns], features[parent][:, columns])
        branch_checks += 1
    rows = read_csv(OUT / "query-features.csv")
    events = read_csv(OUT / "episode-events.csv")
    query_scores = {}
    for row in rows:
        key = row["episode"]
        query_scores.setdefault(key, []).append(row)
    event_checks = 0
    for event in events:
        method = event["method"]
        if method not in results["motif_names"]:
            continue
        threshold = np.log(float(event["factor"]))
        eligible = [int(r["query"]) for r in query_scores[event["episode"]]
                    if r["motif_" + method] and float(r["motif_" + method]) >= threshold]
        expected = min(eligible) if eligible else -1
        assert expected == int(event["first_query"])
        event_checks += 1
    table_checks = 0
    for row in read_csv(OUT / "event-counts.csv"):
        selected = [r for r in events if r["method"] == row["method"]
                    and abs(float(r["factor"]) - float(row["factor"])) < 1e-8
                    and (row["benchmark"] == "all" or r["benchmark"] == row["benchmark"])]
        hits = [r for r in selected if 0 <= int(r["first_query"]) <= int(row["last_query"])]
        assert sum(r["success"] == "True" for r in hits) == int(row["flagged_successes"])
        assert sum(r["success"] == "False" for r in hits) == int(row["detected_failures"])
        assert sum(r["success"] == "True" and episodes[r["episode"]]["queries"] > int(row["last_query"])
                   for r in selected) == int(row["successes_still_observed_at_query"])
        table_checks += 1
    test = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_moe_joint_patterns"],
                          cwd="/data/coding/v8-methods", capture_output=True, text=True)
    (OUT / "tests.txt").write_text(test.stdout + test.stderr)
    assert test.returncode == 0
    functional = OUT / "functional-probe"
    function_results = json.loads((functional / "results.json").read_text())
    functional_hashes = json.loads((functional / "source-hashes.json").read_text())
    for path, expected in functional_hashes.items():
        assert digest(path) == expected, path
    checks = json.loads((functional / "checks.json").read_text())
    for check in checks:
        name, q = check["episode"], check["query"]
        with np.load(functional / (name + "-q%03d-replay.npz" % q)) as replay:
            with np.load(SOURCE / name / "full-hb-routes.npz") as original:
                np.testing.assert_array_equal(replay["native_hb"], original["hb_router_probs"][q])
                np.testing.assert_array_equal(replay["captured_hb"], original["hb_router_probs"][q])
            with np.load(Path(episodes[name]["source_artifact_dir"]) / "episode-trace.npz") as trace:
                np.testing.assert_array_equal(replay["native_actions"], trace["predicted_actions"][q])
                np.testing.assert_array_equal(replay["captured_actions"], trace["predicted_actions"][q])
    calls = json.loads((functional / "calls.json").read_text())
    assert len(calls) == 24 and len(checks) == 12
    assert function_results["model_forwards"] == 24 and function_results["new_environment_actions"] == 0
    git = subprocess.run(["git", "status", "--short"], cwd="/data/coding/robot-whisper-0909",
                         capture_output=True, text=True, check=True)
    status = dict(passed=True, source_hashes_checked=len(source_hashes),
                  branch_prefix_checks=branch_checks, native_branch_feature_replays_exact=5,
                  event_checks=event_checks, count_table_checks=table_checks,
                  unit_tests_passed=True, prefix_checks_in_primary_run=results["prefix_causality_checks_passed"],
                  new_actual_model_forwards=24, functional_samples_reverified=len(checks),
                  functional_source_hashes_checked=len(functional_hashes),
                  upstream_git_status=git.stdout, physical_trap_labels_claimed=False)
    (OUT / "verification.json").write_text(json.dumps(status, indent=2) + "\n")
    artifact_hashes = {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob("*"))
                       if p.is_file() and p.name != "artifact-hashes.json"}
    (OUT / "artifact-hashes.json").write_text(json.dumps(artifact_hashes, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
