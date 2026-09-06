#!/usr/bin/env python3
"""Are v7's three branches carrying three things, or one?

The complementarity ledger showed that `v7_guard` is by construction the union
of its own two branches, and that guard and freeze agree on 94.5% of their
shared detections. That is a redundancy statement, not yet a simplification: a
union of three streams with three thresholds and three confirmation schedules is
what actually ships.

The clock-ceiling result fixes the yardstick. Recall and precision cannot
separate monitors in this corpus, because risk is defined as running to the
horizon cap and so a rule that alarms on anything still running at chunk T
catches every risk episode at a lower false-alarm rate than any routing monitor.
What remains is lead: at matched timely false-alarm rate, on what share of risk
episodes does the monitor fire before the clock would have, and by how much.

That reframes the fusion question into a fair one. v7 spends its 0.53% budget
across branches, freeze taking about 0.39% of it. **If freeze is given the whole
budget instead, does it catch up?** If it does, the turbulence machinery — two
extra streams, two thresholds, two confirmation counters, one corpus scale
constant — buys nothing and the guard collapses to a single score.

Four candidates, each re-thresholded to the same development false-alarm rate
and then applied unchanged to external, matching v7's own discipline of choosing
a quantile on development outcomes and freezing the order statistic:

    guard          the published disjunction, as the reference
    freeze_only    one stream, whole budget
    turbulence     the other side, whole budget
    fused_max      max of the three streams after dividing by their thresholds,
                   a soft OR collapsed to a single scalar with one cut
    shared_q       the same disjunction but with one quantile for all three
                   branches instead of v7's separate 0.975 / 0.70 / 0.65. This
                   is the principled one-knob fusion: a max over per-branch
                   empirical CDFs with a single cut is exactly a disjunction at
                   a common quantile, so it isolates whether the three branches
                   genuinely need separate operating points.

Run:  python experiments/fuse_v7_branches.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
V7_METHOD = WORKSPACE / "moe-v7-0905/method"
sys.path.insert(0, str(V7_METHOD))

from intrinsic_guard_monitor import (  # noqa: E402
    ACCELERATION_BASELINE_COUNT,
    ACCELERATION_BASELINE_START,
    FREEZE_BASELINE_COUNT,
    FREEZE_BASELINE_START,
    PERIODICITY_BASELINE_COUNT,
    PERIODICITY_BASELINE_START,
    first_from_score,
    first_or,
    intrinsic_score_arrays,
    quantile_higher,
    row_max,
)

LAYER_ROOT = WORKSPACE / "moe-v4-0904/results/layerwise_mobility"
FEATURE_ROOT = WORKSPACE / "double-selete/trainfree/results"
LABEL_ROOT = FEATURE_ROOT / "timeout_extension_plus10"
PROFILE = WORKSPACE / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz"
SEALED = WORKSPACE / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"
DEFAULT_OUTPUT = BUNDLE / "results/fusion"

COHORTS = {
    "development_main": ("main_reference.npz", "online_multihead_hub"),
    "development_extra": ("extra_reference.npz", "online_multihead_hub_external"),
    "external_8b": ("external_8b.npz", "online_precision_cascade_external"),
}
CLOCK_T = {
    "development_main": {
        "libero_goal": 21,
        "libero_long": 44,
        "libero_object": 20,
        "libero_spatial": 14,
    },
    "external_8b": {
        "libero_goal": 21,
        "libero_long": 44,
        "libero_object": 20,
        "libero_spatial": 15,
    },
}
QUANTILES = tuple(np.round(np.arange(0.50, 0.9991, 0.0025), 4))
STEPS_PER_CHUNK = 10
DRAWS = 2000
SEED = 20260906


def load_cohort(name: str) -> dict:
    layer_file, feature_dir = COHORTS[name]
    layer = dict(np.load(LAYER_ROOT / layer_file, allow_pickle=False))
    feature = dict(
        np.load(FEATURE_ROOT / feature_dir / "unlabeled_query_features.npz", allow_pickle=False)
    )
    if not np.array_equal(layer["episode"], feature["episode"]):
        raise ValueError(f"{name}: layer and feature caches are not row-aligned")
    names = list(feature["feature_names"].astype(str))
    streams = intrinsic_score_arrays(
        layer["mobility"],
        feature["features"][:, :, names.index("route_acceleration")],
        feature["features"][:, :, names.index("lag_periodicity")],
        float(np.load(PROFILE, allow_pickle=False)["periodicity_scale"]),
    )
    task = layer["task_names"].astype(str)[layer["task_index"].astype(int)]
    return {
        "name": name,
        "streams": streams,
        "valid": layer["valid"],
        "length": layer["length"].astype(int),
        "task": task,
        "suite": pd.Series(task).str.split("/", n=1).str[0].to_numpy(),
        "episode": layer["episode"].astype(int),
    }


def earliest_masks(cohort: dict) -> dict[str, np.ndarray]:
    """Each branch may only fire once its own baseline window has closed."""
    valid = cohort["valid"]
    query = np.arange(valid.shape[1])[None, :]
    return {
        "freeze": valid & (query >= FREEZE_BASELINE_START + FREEZE_BASELINE_COUNT),
        "acceleration": valid
        & (query >= ACCELERATION_BASELINE_START + ACCELERATION_BASELINE_COUNT - 1),
        "periodicity": valid
        & (query >= PERIODICITY_BASELINE_START + PERIODICITY_BASELINE_COUNT - 1),
    }


def guard_alarms(cohort: dict, thresholds: dict[str, float]) -> np.ndarray:
    masks = earliest_masks(cohort)
    freeze = first_from_score(cohort["streams"]["freeze"], thresholds["freeze"], masks["freeze"])
    acceleration = first_from_score(
        cohort["streams"]["acceleration_persistent"],
        thresholds["acceleration"],
        masks["acceleration"],
    )
    periodicity = first_from_score(
        cohort["streams"]["periodicity_persistent"],
        thresholds["periodicity"],
        masks["periodicity"],
    )
    turbulence = np.where(
        (acceleration >= 0) & (periodicity >= 0),
        np.maximum(acceleration, periodicity),
        -1,
    ).astype(np.int16)
    return first_or(freeze, turbulence)


def candidate_alarms(cohort: dict, name: str, cut: float, base: dict[str, float]) -> np.ndarray:
    masks = earliest_masks(cohort)
    streams = cohort["streams"]
    if name == "freeze_only":
        return first_from_score(streams["freeze"], cut, masks["freeze"])
    if name == "turbulence_only":
        acceleration = first_from_score(
            streams["acceleration_persistent"], cut * base["acceleration"], masks["acceleration"]
        )
        periodicity = first_from_score(
            streams["periodicity_persistent"], cut * base["periodicity"], masks["periodicity"]
        )
        return np.where(
            (acceleration >= 0) & (periodicity >= 0),
            np.maximum(acceleration, periodicity),
            -1,
        ).astype(np.int16)
    if name == "shared_q":
        # `cut` carries the shared quantile; the per-branch cuts come from it.
        freeze = first_from_score(streams["freeze"], base["_shared_freeze"], masks["freeze"])
        acceleration = first_from_score(
            streams["acceleration_persistent"], base["_shared_acceleration"], masks["acceleration"]
        )
        periodicity = first_from_score(
            streams["periodicity_persistent"], base["_shared_periodicity"], masks["periodicity"]
        )
        turbulence = np.where(
            (acceleration >= 0) & (periodicity >= 0),
            np.maximum(acceleration, periodicity),
            -1,
        ).astype(np.int16)
        return first_or(freeze, turbulence)
    if name == "fused_max":
        stacked = np.stack(
            [
                streams["freeze"] / base["freeze"],
                streams["acceleration_persistent"] / base["acceleration"],
                streams["periodicity_persistent"] / base["periodicity"],
            ]
        )
        combined = np.where(
            np.stack([masks["freeze"], masks["acceleration"], masks["periodicity"]]),
            stacked,
            -np.inf,
        )
        fused = np.nanmax(np.where(np.isfinite(combined), combined, np.nan), axis=0)
        return first_from_score(fused, cut, cohort["valid"] & np.isfinite(fused))
    raise ValueError(f"unknown candidate: {name}")


def calibrated_cut(reference: list[dict], name: str, quantile: float, base: dict[str, float]) -> float:
    """Label-free order statistic over pooled reference trajectory peaks."""
    peaks = []
    for cohort in reference:
        masks = earliest_masks(cohort)
        streams = cohort["streams"]
        if name == "freeze_only":
            values = np.where(masks["freeze"], streams["freeze"], np.nan)
        elif name == "turbulence_only":
            values = np.where(
                masks["acceleration"],
                streams["acceleration_persistent"] / base["acceleration"],
                np.nan,
            )
        else:
            stacked = np.stack(
                [
                    np.where(masks["freeze"], streams["freeze"] / base["freeze"], np.nan),
                    np.where(
                        masks["acceleration"],
                        streams["acceleration_persistent"] / base["acceleration"],
                        np.nan,
                    ),
                    np.where(
                        masks["periodicity"],
                        streams["periodicity_persistent"] / base["periodicity"],
                        np.nan,
                    ),
                ]
            )
            values = np.nanmax(stacked, axis=0)
        peaks.append(row_max(values))
    return quantile_higher(np.concatenate(peaks), quantile)


def score(cohort: dict, alarms: np.ndarray, labels: pd.DataFrame) -> dict:
    risk = labels["original_failure"].to_numpy(bool)
    timely = ~risk
    fired = alarms >= 0
    clock = pd.Series(cohort["suite"]).map(CLOCK_T[cohort["name"]]).to_numpy()
    earlier = fired & risk & (alarms < clock)
    lead = np.where(earlier, clock - alarms, 0)
    rng = np.random.default_rng(SEED)
    tasks = cohort["task"][risk]
    values = earlier[risk].astype(float)
    names = np.unique(tasks)
    index = {n: np.flatnonzero(tasks == n) for n in names}
    samples = np.empty(DRAWS)
    for draw in range(DRAWS):
        picked = rng.integers(0, len(names), len(names))
        rows = np.concatenate([index[names[p]] for p in picked])
        samples[draw] = values[rows].mean()
    return {
        "tp": int((fired & risk).sum()),
        "fp": int((fired & timely).sum()),
        "recall": float((fired & risk).sum() / risk.sum()),
        "timely_fpr": float((fired & timely).sum() / timely.sum()),
        "earlier_than_clock": int(earlier.sum()),
        "share_earlier": float(earlier.sum() / risk.sum()),
        "share_earlier_ci95": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "median_lead_env_steps": float(
            STEPS_PER_CHUNK * np.median(lead[earlier]) if earlier.any() else np.nan
        ),
    }


def aligned_labels(cohort: dict) -> pd.DataFrame:
    path = LABEL_ROOT / f"{cohort['name']}_clean_labels.csv"
    index = pd.DataFrame(
        {"row": np.arange(len(cohort["task"])), "task": cohort["task"], "episode": cohort["episode"]}
    )
    merged = index.merge(
        pd.read_csv(path)[["task", "episode", "original_failure"]],
        on=["task", "episode"],
        how="left",
        validate="one_to_one",
    ).sort_values("row")
    if merged["original_failure"].isna().any():
        raise ValueError(f"{cohort['name']}: labels do not align")
    return merged.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    profile = dict(np.load(PROFILE, allow_pickle=False))
    base = {
        "freeze": float(profile["freeze_threshold"]),
        "acceleration": float(profile["acceleration_threshold"]),
        "periodicity": float(profile["periodicity_threshold"]),
    }
    cohorts = {name: load_cohort(name) for name in COHORTS}
    reference = [cohorts["development_main"], cohorts["development_extra"]]

    # Anchor: the rebuilt guard must reproduce the sealed first alarms exactly.
    sealed = dict(np.load(SEALED, allow_pickle=False))
    anchors = {}
    for cohort_name, key in (("development_main", "main_guard"), ("external_8b", "external_guard")):
        rebuilt = guard_alarms(cohorts[cohort_name], base)
        anchors[cohort_name] = bool(np.array_equal(rebuilt, sealed[key]))
        if not anchors[cohort_name]:
            mismatch = int((rebuilt != sealed[key]).sum())
            raise SystemExit(
                f"guard rebuild does not match the sealed alarms for {cohort_name}: "
                f"{mismatch} episodes differ"
            )

    labels = {name: aligned_labels(cohorts[name]) for name in ("development_main", "external_8b")}
    target = score(
        cohorts["development_main"], guard_alarms(cohorts["development_main"], base),
        labels["development_main"],
    )["timely_fpr"]

    rows = []
    for name in ("guard", "freeze_only", "turbulence_only", "fused_max", "shared_q"):
        if name == "guard":
            chosen, cut = None, None
            alarms = {
                key: guard_alarms(cohorts[key], base) for key in ("development_main", "external_8b")
            }
        elif name == "shared_q":
            best = None
            for quantile in QUANTILES:
                for stream, key in (
                    ("freeze", "freeze"),
                    ("acceleration_persistent", "acceleration"),
                    ("periodicity_persistent", "periodicity"),
                ):
                    peaks = [
                        row_max(np.where(earliest_masks(c)[key], c["streams"][stream], np.nan))
                        for c in reference
                    ]
                    base[f"_shared_{key}"] = quantile_higher(
                        np.concatenate(peaks), float(quantile)
                    )
                development = score(
                    cohorts["development_main"],
                    candidate_alarms(cohorts["development_main"], name, float(quantile), base),
                    labels["development_main"],
                )
                gap = abs(development["timely_fpr"] - target)
                if best is None or gap < best[0]:
                    best = (gap, float(quantile), float(quantile))
            _, chosen, cut = best
            for stream, key in (
                ("freeze", "freeze"),
                ("acceleration_persistent", "acceleration"),
                ("periodicity_persistent", "periodicity"),
            ):
                peaks = [
                    row_max(np.where(earliest_masks(c)[key], c["streams"][stream], np.nan))
                    for c in reference
                ]
                base[f"_shared_{key}"] = quantile_higher(np.concatenate(peaks), chosen)
            alarms = {
                key: candidate_alarms(cohorts[key], name, cut, base)
                for key in ("development_main", "external_8b")
            }
        else:
            best = None
            for quantile in QUANTILES:
                cut = calibrated_cut(reference, name, float(quantile), base)
                development = score(
                    cohorts["development_main"],
                    candidate_alarms(cohorts["development_main"], name, cut, base),
                    labels["development_main"],
                )
                gap = abs(development["timely_fpr"] - target)
                if best is None or gap < best[0]:
                    best = (gap, float(quantile), cut)
            _, chosen, cut = best
            alarms = {
                key: candidate_alarms(cohorts[key], name, cut, base)
                for key in ("development_main", "external_8b")
            }
        for key in ("development_main", "external_8b"):
            rows.append(
                {
                    "candidate": name,
                    "cohort": key,
                    "quantile": chosen,
                    "cut": cut,
                    **score(cohorts[key], alarms[key], labels[key]),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "fusion.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "guard_rebuild_matches_sealed": anchors,
                "development_target_timely_fpr": target,
                "rows": rows,
            },
            indent=2,
        )
    )
    pd.set_option("display.width", 220)
    print(f"锚点：重建 guard 与封存首报逐位一致 {anchors}")
    print(f"对齐目标：development timely FPR = {target:.5f}\n")
    print(
        table[
            [
                "candidate",
                "cohort",
                "quantile",
                "tp",
                "fp",
                "recall",
                "timely_fpr",
                "earlier_than_clock",
                "share_earlier",
                "median_lead_env_steps",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
