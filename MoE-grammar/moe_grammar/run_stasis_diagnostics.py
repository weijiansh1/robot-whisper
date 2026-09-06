"""Decompose the physical-stasis channels and check the statistics that drive them.

This is the standing verification for three defects found in the first three-channel
audit:

1. ``overregularity`` mixed in a component that was the exact affine negation of
   ``hmm_innovation``, so the composite contained a channel and its own negation.
2. ``causal_recurrence_features`` filled undefined early positions with a large
   sentinel, placing part of the calibration mass at an extreme value.
3. The reported matched AUC pooled over control pairs, which weights events by their
   control count and treats correlated pairs as independent observations.

It reports per-component matched AUC with an event-level bootstrap interval, the raw
routing dynamics around onset, and how often each recurrence statistic is defined.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from moe_grammar.run_three_channel_audit import (
    METHODS,
    OVERREGULARITY_COMPONENTS,
    channel_scores,
    episode_rows,
    event_bootstrap_interval,
    raw_scores,
)

COMPONENT_KEYS = tuple(f"{name}_percentile" for name in OVERREGULARITY_COMPONENTS)
OFFSETS = tuple(range(-10, 11, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/open-world-phenotypes-v2.npz")
    )
    parser.add_argument(
        "--fold-summary", type=Path, default=Path("results-open-world-global-k12/summary.json")
    )
    parser.add_argument("--models", type=Path, default=Path("results-three-channel-v2/models"))
    parser.add_argument(
        "--output", type=Path, default=Path("results-three-channel-v2/diagnostics.json")
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def per_event_win_rates(
    score: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    state: np.ndarray,
    success: np.ndarray,
    onset: np.ndarray,
) -> list[float]:
    """Win rate of each stasis event against its own state- and query-matched controls."""
    rates: list[float] = []
    for index in range(len(starts)):
        if success[index] or onset[index] < 0:
            continue
        start, length, event_onset = int(starts[index]), int(lengths[index]), int(onset[index])
        event = score[start + min(event_onset, length - 1)]
        wins = [
            float(event > score[int(starts[control]) + event_onset])
            + 0.5 * float(event == score[int(starts[control]) + event_onset])
            for control in np.flatnonzero(success & (state == state[index]))
            if int(lengths[control]) > event_onset
        ]
        if wins:
            rates.append(float(np.mean(wins)))
    return rates


def onset_time_course(
    values: dict[str, np.ndarray],
    starts: np.ndarray,
    lengths: np.ndarray,
    state: np.ndarray,
    success: np.ndarray,
    onset: np.ndarray,
) -> list[dict[str, Any]]:
    """Compare each raw statistic against matched controls at offsets around onset."""
    events = [i for i in range(len(starts)) if not success[i] and onset[i] >= 0]
    rows: list[dict[str, Any]] = []
    for offset in OFFSETS:
        record: dict[str, Any] = {"offset": offset}
        paired: dict[str, list[float]] = {name: [] for name in values}
        count = 0
        for index in events:
            position = int(onset[index]) + offset
            if position < 1 or position >= int(lengths[index]):
                continue
            controls = [
                control
                for control in np.flatnonzero(success & (state == state[index]))
                if int(lengths[control]) > position
            ]
            if not controls:
                continue
            count += 1
            for name, series in values.items():
                event = float(series[int(starts[index]) + position])
                control_mean = float(
                    np.mean([series[int(starts[c]) + position] for c in controls])
                )
                paired[name].append(event - control_mean)
        if not count:
            continue
        record["events"] = count
        for name, differences in paired.items():
            array = np.asarray(differences)
            record[name] = {
                "paired_difference": float(array.mean()),
                "ci": event_bootstrap_interval(array.tolist(), draws=2000),
            }
        rows.append(record)
    return rows


def main() -> None:
    args = parse_args()
    arrays = np.load(args.dataset)
    folds = json.loads(args.fold_summary.read_text(encoding="utf-8"))["folds"]

    starts = arrays["anchor_starts"]
    lengths = arrays["anchor_lengths"]
    state = arrays["anchor_state"]
    success = arrays["anchor_success"]
    onset = arrays["anchor_stasis_onset"]
    size = len(arrays["anchor_base"])

    keys = tuple(METHODS) + COMPONENT_KEYS
    scored = {name: np.full(size, np.nan, dtype=np.float32) for name in keys}
    dynamics = {
        name: np.full(size, np.nan, dtype=np.float32)
        for name in ("speed", "periodic", "hmm", "belief_entropy")
    }
    defined = {name: np.zeros(size, dtype=bool) for name in ("speed_valid", "periodic_valid")}

    for fold in folds:
        index = int(fold["fold"])
        model = joblib.load(args.models / f"fold_{index}.joblib")
        raw = raw_scores(
            np.asarray(arrays["anchor_base"], dtype=np.float32),
            starts,
            lengths,
            model,
            args.device,
        )
        values = channel_scores(raw, starts, lengths, model)
        rows = episode_rows(
            starts, lengths, np.flatnonzero(np.isin(state, np.asarray(fold["test_states"])))
        )
        for name in keys:
            scored[name][rows] = np.asarray(values[name], dtype=np.float32)[rows]
        for name in dynamics:
            dynamics[name][rows] = np.asarray(raw[name], dtype=np.float32)[rows]
        for name in defined:
            defined[name][rows] = np.asarray(raw[name], dtype=bool)[rows]
        print(f"fold {index}: scored {len(rows)} anchor rows", flush=True)

    channels: dict[str, Any] = {}
    for name in keys:
        rates = per_event_win_rates(scored[name], starts, lengths, state, success, onset)
        channels[name] = {
            "event_matched_auc": float(np.mean(rates)) if rates else None,
            "event_matched_auc_ci": event_bootstrap_interval(rates),
            "matched_events": len(rates),
        }

    correlations: dict[str, float] = {}
    for name in COMPONENT_KEYS:
        both = np.isfinite(scored[name]) & np.isfinite(scored["hmm_innovation"])
        correlations[name] = float(
            np.corrcoef(scored[name][both], scored["hmm_innovation"][both])[0, 1]
        )

    summary = {
        "channels": channels,
        "component_correlation_with_hmm_innovation": correlations,
        "recurrence_defined_fraction": {
            name: float(np.mean(mask)) for name, mask in defined.items()
        },
        "onset_time_course": onset_time_course(
            dynamics, starts, lengths, state, success, onset
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n{'channel':32s} {'event AUC':>10s}  {'95% CI':>18s} {'events':>7s}")
    print("-" * 72)
    for name, row in channels.items():
        value = row["event_matched_auc"]
        interval = row["event_matched_auc_ci"]
        text = "n/a" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
        print(f"{name:32s} {value:10.4f}  {text:>18s} {row['matched_events']:7d}")
    print("\ncorrelation with hmm_innovation (a value near -1 means the composite")
    print("contains an exact negation of the innovation channel):")
    for name, value in correlations.items():
        print(f"  {name:30s} {value:+.4f}")
    print("\nfraction of anchor rows where each recurrence statistic is defined:")
    for name, value in summary["recurrence_defined_fraction"].items():
        print(f"  {name:30s} {value:.4f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
