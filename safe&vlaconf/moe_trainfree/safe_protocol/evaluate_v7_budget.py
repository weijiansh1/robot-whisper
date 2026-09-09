"""Evaluate unchanged v7 budget calibration separately from nominal CP alpha."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from core import ROOT, digest, write_json
from evaluate import attach_labels, alarm_metrics, physical_events
from v7_adapter import raw_episode, V7
from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe/v7")
    args = parser.parse_args()
    output = args.input.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        if digest(output / name) != expected:
            raise ValueError("v7 artifact changed")
    for name, expected in manifest["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("v7 implementation changed")
    frame = attach_labels(pd.read_csv(output / "index.csv"))
    events = physical_events(frame)
    metrics, timing, replay = [], [], []
    kinds = ("unlabeled_reference_budget", "success_calibration_budget")
    for info in manifest["folds"]:
        name = info["fold"]
        profile = json.loads((output / "profiles" / f"{name}.json").read_text())
        with np.load(output / "budget_predictions" / f"{name}.npz", allow_pickle=False) as stored:
            data = {k: stored[k] for k in stored.files}
        part = frame.iloc[data["test_rows"]].reset_index(drop=True)
        for kind_i, kind in enumerate(kinds):
            for alpha_i, alpha in enumerate(data["alphas"]):
                first = data["first"][kind_i, alpha_i]
                for scope, mask in (("unseen", data["test_unseen"]), ("seen", ~data["test_unseen"])):
                    metrics.append({"fold": name, "suite": info["suite"], "seed": info["seed"], "method": kind,
                                    "budget": alpha, "scope": scope, **alarm_metrics(part.loc[mask], first[mask])})
                if alpha == 0.05:
                    values = next(p for p in profile["budget_profiles"] if p["kind"] == kind and p["alpha"] == alpha)
                    fitted = GlobalIntrinsicProfile(**{k: values[k] for k in ("freeze_threshold", "acceleration_threshold", "periodicity_threshold", "periodicity_scale")})
                    for local in (0, len(part) // 2, len(part) - 1):
                        row = part.iloc[local]
                        monitor = IntrinsicGuardMonitor(fitted)
                        for query in raw_episode(row):
                            result = monitor.update(query)
                        if result["first_alarm_query"] != int(first[local]):
                            raise AssertionError("original v7 online/batch disagreement")
                        replay.append({"fold": name, "kind": kind, "global_row": int(data["test_rows"][local]), "queries": int(row.length)})
                    for local, global_row in enumerate(data["test_rows"]):
                        if global_row not in events.index:
                            continue
                        event = events.loc[global_row]
                        timing.append({"fold": name, "suite": info["suite"], "method": kind,
                            "scope": "unseen" if data["test_unseen"][local] else "seen", "budget": alpha,
                            "global_row": int(global_row), "first_alarm": int(first[local]),
                            "failed_goal_release": event.failed_goal_release, "drop_goal_release": event.drop_goal_release,
                            "reason": event.reason})
        print(f"V7 BUDGET VERIFIED {name}", flush=True)
    pd.DataFrame(metrics).to_csv(output / "budget_metrics.csv", index=False)
    pd.DataFrame(timing).to_csv(output / "budget_physical_timing.csv", index=False)
    write_json(output / "budget_verification.json", {"original_monitor_replays": replay,
        "source_hashes_checked": len(manifest["sources"]), "artifact_hashes_checked": len(manifest["artifacts"]),
        "success_filter_uses_labels": True,
        "inner_calibrator_audit_has_no_label_argument": True,
        "unlabeled_budget_is_not_success_fpr": True, "evaluator_sha256": digest(Path(__file__))})


if __name__ == "__main__":
    main()
