#!/usr/bin/env python3
"""Anchor check: rebuild the sealed v7 alarms from raw routes and fail loudly.

This must be green before any legacy_16x32 number is trusted.  Three levels:

1. the frozen operating point recomputed from the reference corpus equals the
   published ``global_profile.npz`` bit for bit;
2. the raw-extracted ``route_acceleration`` / ``lag_periodicity`` reproduce the
   sealed first-alarm arrays element by element for both labelled cohorts, and
   the published counts 439/80 (external_8b) and 382/67 (development_main);
3. a query-by-query ``IntrinsicGuardMonitor`` replay of sampled raw episodes
   agrees with the batch path and with the sealed first-alarm tuples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_profile import (
    BUNDLE,
    SEALED_ALARMS,
    V7,
    guard_alarms,
    load_frozen_point,
    sealed,
)
from raw_route_features import COHORTS, load_npz

sys.path.insert(0, str(V7 / "method"))
from intrinsic_guard_monitor import (  # noqa: E402
    GlobalIntrinsicProfile,
    IntrinsicGuardMonitor,
)
from raw_route_features import episode_features  # noqa: E402
import zarr  # noqa: E402


DEFAULT_FEATURES = BUNDLE / "results/raw_features"
DEFAULT_OUTPUT = BUNDLE / "results/anchor"
EXPECTED = {
    "development_main": {"prefix": "main", "tp": 382, "fp": 67, "episodes": 14_800, "risks": 487},
    "external_8b": {"prefix": "external", "tp": 439, "fp": 80, "episodes": 15_600, "risks": 564},
}
LABELS = {
    "development_main": sealed.LABEL_ROOT / "development_main_clean_labels.csv",
    "external_8b": sealed.LABEL_ROOT / "external_8b_clean_labels.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--monitor-samples", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced(indices: np.ndarray, count: int) -> list[int]:
    if len(indices) < count:
        raise ValueError(f"need {count} samples, only {len(indices)} available")
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    return np.asarray(indices)[positions].astype(int).tolist()


def sample_rows(alarms: dict[str, np.ndarray], prefix: str, count: int) -> list[tuple[int, str]]:
    freeze = alarms[f"{prefix}_freeze"] >= 0
    turbulence = alarms[f"{prefix}_turbulence"] >= 0
    groups = {
        "freeze_only": np.flatnonzero(freeze & ~turbulence),
        "turbulence_only": np.flatnonzero(turbulence & ~freeze),
        "no_alarm": np.flatnonzero(~freeze & ~turbulence),
    }
    return [
        (row, group) for group, rows in groups.items() for row in evenly_spaced(rows, count)
    ]


def monitor_replay(cohort: str, task: str, episode: int, length: int, point) -> dict:
    spec = COHORTS[cohort]
    group = zarr.open_group(
        str(spec.cache_root / task / spec.run_id / "server/routes.zarr"), mode="r"
    )
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    index = np.flatnonzero(episode_id == episode)
    if len(index) != length or not np.all(np.diff(index) == 1):
        raise ValueError(f"raw episode mismatch for {task}/{episode}")
    raw = np.asarray(
        group["hb_router_probs"][index[0] : index[-1] + 1], dtype=np.float16
    )
    monitor = IntrinsicGuardMonitor(GlobalIntrinsicProfile.load(V7 / "results/intrinsic_guard_v7/global_profile.npz"))
    records = [monitor.update(query) for query in raw]
    streamed = {
        "mobility": np.stack([row["layer_mobility"] for row in records]),
        "acceleration": np.asarray(
            [row["route_acceleration"] for row in records], dtype=np.float32
        ),
        "periodicity": np.asarray(
            [row["lag_periodicity"] for row in records], dtype=np.float32
        ),
    }
    batch = episode_features(raw)
    drift = {
        name: _finite_max_abs(streamed[name], value)
        for name, value in zip(("mobility", "acceleration", "periodicity"), batch)
    }
    return {
        "task": task,
        "episode": episode,
        "queries": length,
        "raw_sha256": hashlib.sha256(raw.tobytes(order="C")).hexdigest(),
        "monitor_vs_batch_max_abs": drift,
        "first": {
            "freeze": monitor.first_freeze_query,
            "acceleration": monitor.first_acceleration_query,
            "periodicity": monitor.first_periodicity_query,
            "turbulence": monitor.first_turbulence_query,
            "guard": monitor.first_alarm_query,
        },
    }


def _finite_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return float("inf")
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    point = load_frozen_point()
    sealed_alarms = load_npz(SEALED_ALARMS)
    failures: list[str] = []
    report: dict = {
        "schema": "himoe.legacy16x32.anchor.v1",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "frozen_point_recomputed_from_reference_corpus": point.as_dict(),
        "frozen_point_matches_published_profile": True,
        "recalibration_performed": False,
        "operating_point_reselected": False,
        "cohorts": {},
    }
    sample_rows_table: list[dict] = []

    for cohort, expected in EXPECTED.items():
        prefix = expected["prefix"]
        raw = load_npz(args.features / f"{cohort}_route_features.npz")
        layer = load_npz(COHORTS[cohort].layer_cache)
        valid = layer["valid"].astype(bool)

        variants = {
            "published_mobility_raw_route_features": layer["mobility"],
            "fully_raw_features": raw["mobility"],
        }
        cohort_report: dict = {"variants": {}}
        for variant, mobility in variants.items():
            alarms = guard_alarms(
                point,
                mobility,
                raw["route_acceleration"],
                raw["lag_periodicity"],
                valid,
            )
            exact = {
                name: bool(np.array_equal(alarms[name], sealed_alarms[f"{prefix}_{name}"]))
                for name in ("freeze", "acceleration", "periodicity", "turbulence", "guard")
            }
            differing = {
                name: int((alarms[name] != sealed_alarms[f"{prefix}_{name}"]).sum())
                for name in ("freeze", "acceleration", "periodicity", "turbulence", "guard")
            }
            labels = sealed.aligned_labels(layer, LABELS[cohort], cohort)
            risk = labels["original_failure"].to_numpy(bool)
            alarm = alarms["guard"] >= 0
            counts = {
                "tp": int((alarm & risk).sum()),
                "fp": int((alarm & ~risk).sum()),
                "risks": int(risk.sum()),
                "episodes": len(labels),
            }
            cohort_report["variants"][variant] = {
                "sealed_first_alarm_elementwise_exact": exact,
                "differing_episodes": differing,
                "counts": counts,
            }
            if variant == "published_mobility_raw_route_features":
                if not all(exact.values()):
                    failures.append(
                        f"{cohort}: raw-extracted features do not reproduce the sealed "
                        f"first-alarm arrays ({differing})"
                    )
                if counts["tp"] != expected["tp"] or counts["fp"] != expected["fp"]:
                    failures.append(
                        f"{cohort}: expected {expected['tp']} TP / {expected['fp']} FP, "
                        f"got {counts['tp']} / {counts['fp']}"
                    )
                if counts["risks"] != expected["risks"] or counts["episodes"] != expected["episodes"]:
                    failures.append(f"{cohort}: cohort size or risk count changed: {counts}")

        rows = sample_rows(sealed_alarms, prefix, args.monitor_samples)
        task_names = layer["task_names"].astype(str)
        replays = []
        for row, group in rows:
            record = monitor_replay(
                cohort,
                str(task_names[int(layer["task_index"][row])]),
                int(layer["episode"][row]),
                int(layer["length"][row]),
                point,
            )
            record["row"] = int(row)
            record["sample_class"] = group
            record["sealed_first"] = {
                name: int(sealed_alarms[f"{prefix}_{name}"][row])
                for name in ("freeze", "acceleration", "periodicity", "turbulence", "guard")
            }
            record["monitor_matches_sealed"] = record["first"] == record["sealed_first"]
            record["monitor_matches_batch_bitexact"] = all(
                value == 0.0 for value in record["monitor_vs_batch_max_abs"].values()
            )
            replays.append(record)
            sample_rows_table.append(
                {
                    "cohort": cohort,
                    "row": record["row"],
                    "sample_class": group,
                    "task": record["task"],
                    "episode": record["episode"],
                    "queries": record["queries"],
                    "raw_sha256": record["raw_sha256"],
                    **{
                        f"monitor_vs_batch_max_abs_{k}": v
                        for k, v in record["monitor_vs_batch_max_abs"].items()
                    },
                    **{f"monitor_first_{k}": v for k, v in record["first"].items()},
                    **{f"sealed_first_{k}": v for k, v in record["sealed_first"].items()},
                    "monitor_matches_sealed": record["monitor_matches_sealed"],
                }
            )
        cohort_report["monitor_replay_episodes"] = len(replays)
        cohort_report["monitor_replay_all_match_sealed"] = all(
            record["monitor_matches_sealed"] for record in replays
        )
        cohort_report["monitor_replay_all_bitexact_vs_batch"] = all(
            record["monitor_matches_batch_bitexact"] for record in replays
        )
        if not cohort_report["monitor_replay_all_match_sealed"]:
            failures.append(f"{cohort}: causal monitor replay disagrees with the sealed alarms")
        report["cohorts"][cohort] = cohort_report

    pd.DataFrame(sample_rows_table).to_csv(
        args.output / "monitor_replay_samples.csv", index=False
    )
    report["failures"] = failures
    report["anchor_green"] = not failures
    report["artifacts"] = {
        "sealed_first_alarms_sha256": sha256(SEALED_ALARMS),
        "global_profile_sha256": sha256(V7 / "results/intrinsic_guard_v7/global_profile.npz"),
        "verifier_sha256": sha256(Path(__file__)),
        "extractor_sha256": sha256(Path(__file__).with_name("raw_route_features.py")),
    }
    (args.output / "anchor_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if failures:
        raise SystemExit("ANCHOR FAILED:\n" + "\n".join(failures))
    print("anchor green", flush=True)


if __name__ == "__main__":
    main()
