"""Check recalibration ranks and replay the frozen hybrid from raw queries."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from fusion import HERE, ROOT, ALPHAS, trajectory_peak
from core import digest, write_json
from run_fusion import load_npz
from v7_adapter import raw_episode
sys.path.insert(0, str(HERE))
from hybrid_monitor import KnNV8Monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round7_temporal_fusion")
    parent = parser.parse_args().input.resolve()
    output = parent / "cross_task_calibration"
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    checks = 0
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
        checks += 1
    for name, expected in manifest["sources"].items():
        assert digest(ROOT / name) == expected, name
        checks += 1
    frame = pd.read_csv(parent / "outcome_alignment.csv")
    decisions = pd.read_csv(output / "matched_episode_decisions.csv")
    rank_checks = 0
    for path in sorted((output / "predictions").glob("*.npz")):
        data = load_npz(path)
        cal = frame.iloc[data["calibration_rows"]].reset_index(drop=True)
        success = data["calibration_labels"] == 0
        ids = (cal.loc[success, "task"] + "|" + cal.loc[success, "init_state_id"].astype(str)).to_numpy()
        for mi in range(2):
            peaks = trajectory_peak(data["calibration_scores"][mi, success])
            for ki, values in enumerate((peaks, np.array([peaks[ids == key].max() for key in np.unique(ids)]))):
                ordered = np.sort(values)
                for ai, alpha in enumerate(ALPHAS):
                    rank = int(np.ceil((len(ordered)+1)*(1-alpha)))
                    expected = ordered[rank-1] if rank <= len(ordered) else np.inf
                    assert data["thresholds"][ki, ai, mi] == expected
                    hit = data["scores"][mi] > expected
                    first = np.where(hit.any(1), hit.argmax(1), -1)
                    np.testing.assert_array_equal(first, data["first"][ki, ai, mi])
                    rank_checks += 1
    replays = []
    for suite in sorted(frame.suite.unique()):
        fold = f"{suite}_20260907"
        for knn_name, knn_method in (("knn10_cross_task", "knn10"), ("knn12_cross_task", "knn12_v8")):
            for version, anchor in (("v8", "v8_frozen"), ("v8.2", "v82_frozen")):
                method = f"{knn_name}_or_{anchor}"
                part = decisions.loc[decisions.fold.eq(fold) & decisions.scope.eq("unseen") & decisions.method.eq(method)]
                for failure in (True, False):
                    candidates = part.loc[part.failure.eq(failure)]
                    if not len(candidates):
                        continue
                    alarms = candidates.loc[candidates.first_alarm_query.ge(0)]
                    selected = alarms if len(alarms) else candidates
                    record = selected.iloc[len(selected)//2]
                    row = frame.iloc[int(record.global_row)]
                    monitor = KnNV8Monitor(output / "profiles" / f"{fold}.npz", row.checkpoint, knn_method, version)
                    raw = raw_episode(row)
                    trace = [monitor.update(query) for query in raw]
                    assert trace[-1]["first_alarm_query"] == record.first_alarm_query, (fold, method, int(record.global_row))
                    monitor.reset()
                    assert not monitor.update(raw[0])["alarm"]
                    replays.append(dict(fold=fold, method=method, global_row=int(record.global_row), queries=len(raw)))
        print(f"HYBRID VERIFIED {suite}", flush=True)
    write_json(output / "verification.json", dict(hash_checks=checks, calibration_rank_checks=rank_checks,
        raw_hybrid_replays=replays, all_checks_passed=True,
        hybrid_monitor_sha256=digest(HERE / "hybrid_monitor.py"), verifier_sha256=digest(Path(__file__))))
    print("ALL CROSS-TASK AND HYBRID CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
