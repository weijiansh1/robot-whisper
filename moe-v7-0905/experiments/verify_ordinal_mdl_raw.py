#!/usr/bin/env python3
"""Verify uniformly sampled external raw routes without opening outcome labels."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import zarr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "method"))

from evaluate_ordinal_mdl_guard import DEFAULT_OUTPUT, verify_seal, write_json  # noqa: E402
from evaluate_intrinsic_guard_v7 import EXTERNAL_FEATURE, EXTERNAL_LAYER, feature, load_npz, sha256  # noqa: E402
from ordinal_mdl_guard import OrdinalMDLGuard  # noqa: E402


CACHE_ROOT = HERE.parent.parent / "VLA_MUI_HUB/cache_new/HiMoE-VLA"


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not np.array_equal(np.isfinite(left), np.isfinite(right)):
        return float("inf")
    good = np.isfinite(left)
    return float(np.max(np.abs(left[good] - right[good]))) if good.any() else 0.0


def verify(output: Path, samples: int) -> dict:
    verify_seal(output)
    layer, routes = load_npz(EXTERNAL_LAYER), load_npz(EXTERNAL_FEATURE)
    sealed = load_npz(output / "external_scores.npz")
    acceleration = feature(routes, "route_acceleration")
    periodicity = feature(routes, "lag_periodicity")
    indices = np.unique(np.linspace(0, len(layer["length"]) - 1, samples, dtype=int))
    records = []
    for index in indices:
        task = str(layer["task_names"][int(layer["task_index"][index])])
        episode, length = int(layer["episode"][index]), int(layer["length"][index])
        path = CACHE_ROOT / task / str(layer["run_id"]) / "server/routes.zarr"
        group = zarr.open_group(str(path), mode="r")
        positions = np.flatnonzero(np.asarray(group["episode_id"][:]) == episode)
        if len(positions) != length or not np.all(np.diff(positions) == 1):
            raise ValueError(f"raw episode alignment failed: {task} {episode}")
        raw = np.asarray(group["hb_router_probs"][positions[0]:positions[-1] + 1])
        monitor = OrdinalMDLGuard()
        stream = [monitor.update(query) for query in raw]
        feature_errors = {
            "mobility": max_error(np.stack([r["layer_mobility"] for r in stream]),
                                  layer["mobility"][index, :length]),
            "acceleration": max_error(np.array([r["route_acceleration"] for r in stream]),
                                      acceleration[index, :length]),
            "periodicity": max_error(np.array([r["lag_periodicity"] for r in stream]),
                                     periodicity[index, :length]),
        }
        score_errors = {name: max_error(np.array([r[name] for r in stream]), sealed[name][index, :length])
                        for name in ("gain_bits", "freeze_gain_bits", "turbulence_gain_bits")}
        decisions_equal = all(np.array_equal([r[name] for r in stream], sealed[name][index, :length])
                              for name in ("selected_model", "candidate_query"))
        first_equal = all(stream[-1][name] == sealed[name][index]
                          for name in ("first_alarm_query", "first_keypoint_query"))
        cut = max(1, length // 2)
        changed = raw.copy()
        changed[cut:] = np.roll(changed[cut:], shift=5, axis=-1)
        altered = OrdinalMDLGuard()
        altered_stream = [altered.update(query) for query in changed]
        prefix_equal = all(np.array_equal([r[name] for r in stream[:cut]],
                                          [r[name] for r in altered_stream[:cut]])
                           for name in ("gain_bits", "selected_model", "candidate_query"))
        passed = (decisions_equal and first_equal and prefix_equal
                  and max(feature_errors.values()) <= 2e-5 and max(score_errors.values()) <= 1e-10)
        records.append({
            "cache_row": int(index), "task": task, "episode": episode, "queries": length,
            "raw_path": str(path), "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "feature_max_abs_error": feature_errors, "score_max_abs_error": score_errors,
            "exact_decisions_and_keypoints": decisions_equal,
            "exact_first_alarm": first_equal, "prefix_invariant": prefix_equal, "passed": passed,
        })
        print(f"raw replay {len(records)}/{len(indices)}: row={index}, passed={passed}", flush=True)
    report = {
        "schema": "himoe.ordinal_mdl_guard.raw_replay.v1",
        "sampling": "evenly spaced cache rows, no outcome access",
        "outcome_labels_loaded": False,
        "episodes": len(records), "queries": sum(r["queries"] for r in records),
        "all_passed": all(r["passed"] for r in records),
        "verifier_sha256": sha256(Path(__file__)),
        "seal_sha256": sha256(output / "sealed_manifest.json"), "records": records,
    }
    write_json(output / "raw_replay_verification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    report = verify(args.output, args.samples)
    if not report["all_passed"]:
        raise RuntimeError("one or more raw replay checks failed; see verification artifact")


if __name__ == "__main__":
    main()
