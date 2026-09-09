#!/usr/bin/env python3
"""Evaluate true HB-MoE activation amplitudes as unsupervised stop signals.

Signals are prefix-only.  A decision after flow stage ``s`` uses activation
forwards ``0..s-1`` and never reads x[s+1:] or task outcome.  The completed
round-10 action is used only as an offline self-supervised endpoint label.

Thresholds are evaluated leave-one-complete-rollout-out (LOEO), using
``episode_id`` as the fold.  If ``--holdout-run`` is supplied, one development
rule is selected from LOEO, refit on all development rollouts, then frozen before
the second store is opened for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from himoe_activation_store import SUPPORTED_FORMATS


TOTAL_ROUNDS = 10
LIVE_DIMS = 7
STAGES = np.arange(2, TOTAL_ROUNDS, dtype=np.int64)
DEFAULT_MIN_STAGES = (4, 6, 8)
ESTIMATORS = ("constant_velocity", "constant_acceleration")
EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class ActivationFlowData:
    name: str
    store: Path
    x: np.ndarray
    routed: np.ndarray
    shared: np.ndarray
    total: np.ndarray
    rollout_id: np.ndarray
    control_step: np.ndarray
    action_std: np.ndarray
    hb_layers: np.ndarray
    topk_weight_sum_error_max: float

    @property
    def rows(self) -> int:
        return int(len(self.x))


@dataclasses.dataclass(frozen=True)
class Signal:
    name: str
    family: str
    values: np.ndarray


@dataclasses.dataclass(frozen=True)
class ControllerErrors:
    continuous_rms: np.ndarray
    translation_rms: np.ndarray
    rotation_rms: np.ndarray
    gripper_flip: np.ndarray

    def unsafe(self, tolerance: float) -> np.ndarray:
        return (self.continuous_rms > tolerance) | self.gripper_flip


def _resolve_store(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value.name.endswith(".zarr"):
        return value
    candidate = value / "activation_flow.zarr"
    if candidate.exists():
        return candidate
    server_candidate = value / "server" / "activation_flow.zarr"
    if server_candidate.exists():
        return server_candidate
    raise FileNotFoundError("cannot find activation_flow.zarr under %s" % value)


def load_activation_flow(path: str | Path, require_loeo: bool = True) -> ActivationFlowData:
    store = _resolve_store(path)
    root = zarr.open_group(str(store), mode="r")
    meta = dict(root.attrs)
    if meta.get("format") not in SUPPORTED_FORMATS:
        raise ValueError("unsupported activation-flow store: %s" % store)
    required = (
        "x_traj",
        "hb_routed_rms",
        "hb_shared_rms",
        "hb_total_mlp_rms",
        "hb_topk_weight_sum_error",
        "episode_id",
        "control_step",
    )
    missing = [name for name in required if name not in root]
    if missing:
        raise ValueError("activation store lacks arrays: %s" % missing)

    x = np.asarray(root["x_traj"][:], dtype=np.float64)
    routed = np.asarray(root["hb_routed_rms"][:], dtype=np.float64)
    shared = np.asarray(root["hb_shared_rms"][:], dtype=np.float64)
    total = np.asarray(root["hb_total_mlp_rms"][:], dtype=np.float64)
    weight_error = np.asarray(root["hb_topk_weight_sum_error"][:], dtype=np.float64)
    rollout = np.asarray(root["episode_id"][:], dtype=np.int64)
    control = np.asarray(root["control_step"][:], dtype=np.int64)
    action_std = np.asarray(meta.get("normalization_action_std"), dtype=np.float64)
    layers = np.asarray(meta.get("hb_layers"), dtype=np.int16)

    if x.ndim != 4 or x.shape[1] != TOTAL_ROUNDS + 1 or x.shape[-1] < LIVE_DIMS:
        raise ValueError("x_traj must be [N,11,A,D>=7], got %s" % (x.shape,))
    expected_activation = (len(x), len(layers), TOTAL_ROUNDS, x.shape[2])
    for name, value in (("routed", routed), ("shared", shared), ("total", total)):
        if value.shape != expected_activation:
            raise ValueError("%s activation has shape %s, expected %s" % (
                name, value.shape, expected_activation))
        if np.any(~np.isfinite(value)) or np.any(value < 0.0):
            raise ValueError("%s activation must be finite and nonnegative" % name)
    if weight_error.shape != expected_activation or np.any(~np.isfinite(weight_error)):
        raise ValueError("invalid hb_topk_weight_sum_error")
    max_weight_error = float(weight_error.max(initial=0.0))
    if max_weight_error > 0.02:
        raise ValueError(
            "actual top-k weights are not normalised: max sum error %.4g" % max_weight_error
        )
    if rollout.shape != (len(x),) or control.shape != (len(x),):
        raise ValueError("episode_id/control_step row count does not match x_traj")
    if action_std.shape != (LIVE_DIMS,) or np.any(~np.isfinite(action_std)) or np.any(action_std <= 0):
        raise ValueError("invalid normalization_action_std metadata")
    if require_loeo:
        if np.any(rollout < 0):
            raise ValueError(
                "episode_id=-1 means this capture has no complete-rollout boundary; "
                "it cannot be used for LOEO"
            )
        if len(np.unique(rollout)) < 2:
            raise ValueError("LOEO requires at least two complete rollouts")
    return ActivationFlowData(
        name=store.parent.name,
        store=store,
        x=x[..., :LIVE_DIMS],
        routed=routed,
        shared=shared,
        total=total,
        rollout_id=rollout,
        control_step=control,
        action_std=action_std,
        hb_layers=layers,
        topk_weight_sum_error_max=max_weight_error,
    )


def endpoint_predictions(x: np.ndarray) -> dict[str, np.ndarray]:
    """Constant-velocity/acceleration estimates, [query, stage, action, 7]."""
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 4 or value.shape[1] != TOTAL_ROUNDS + 1 or value.shape[-1] != LIVE_DIMS:
        raise ValueError("x must be [N,11,A,7]")
    velocity, acceleration = [], []
    for stage_value in STAGES:
        stage = int(stage_value)
        remaining = TOTAL_ROUNDS - stage
        current = value[:, stage]
        delta = current - value[:, stage - 1]
        previous = value[:, stage - 1] - value[:, stage - 2]
        velocity.append(current + remaining * delta)
        acceleration.append(
            current
            + remaining * delta
            + remaining * (remaining + 1) * (delta - previous) / 2.0
        )
    return {
        "constant_velocity": np.stack(velocity, axis=1),
        "constant_acceleration": np.stack(acceleration, axis=1),
    }


def controller_errors(
    x: np.ndarray, prediction: np.ndarray, action_std: np.ndarray
) -> ControllerErrors:
    """Physical 6D RMS plus the controller-relevant gripper sign mismatch."""
    final = np.asarray(x, dtype=np.float64)[:, None, TOTAL_ROUNDS]
    estimate = np.asarray(prediction, dtype=np.float64)
    scale = np.asarray(action_std, dtype=np.float64)
    if estimate.shape[:2] != (len(x), len(STAGES)) or estimate.shape[2:] != final.shape[2:]:
        raise ValueError("prediction must be [N,8,A,7]")
    difference = (estimate[..., :6] - final[..., :6]) * scale[None, None, None, :6]
    def rms(value: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(np.square(value), axis=(2, 3)))
    flip = np.any(
        np.sign(estimate[..., 6]) != np.sign(final[..., 6]), axis=2
    )
    return ControllerErrors(
        continuous_rms=rms(difference),
        translation_rms=rms(difference[..., :3]),
        rotation_rms=rms(difference[..., 3:6]),
        gripper_flip=flip,
    )


def _aggregate_sites(value: np.ndarray, mode: str) -> np.ndarray:
    flat = value.reshape(len(value), -1)
    if mode == "mean":
        return flat.mean(axis=1)
    if mode == "max":
        return flat.max(axis=1)
    if mode == "p90":
        return np.percentile(flat, 90, axis=1)
    raise ValueError("unknown aggregation %s" % mode)


def build_activation_signals(data: ActivationFlowData) -> list[Signal]:
    """Amplitude, normalized dynamics, rebound, and all-layer veto candidates."""
    sequences: dict[tuple[str, str], list[np.ndarray]] = {}

    def add(family: str, name: str, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (data.rows,):
            raise AssertionError("%s has shape %s" % (name, array.shape))
        sequences.setdefault((family, name), []).append(array)

    branches = {"routed": data.routed, "shared": data.shared, "total": data.total}
    for stage_value in STAGES:
        stage = int(stage_value)
        round_index = stage - 1
        for branch_name, branch in branches.items():
            current = branch[:, :, round_index]
            previous = branch[:, :, round_index - 1]
            first = branch[:, :, 0]
            query_floor = np.maximum(
                np.median(first.reshape(data.rows, -1), axis=1)[:, None, None] * 1e-3,
                EPS,
            )
            scale = np.maximum(first, query_floor)
            prefix = branch[:, :, : round_index + 1]
            prefix_median = np.median(prefix, axis=2)
            absolute_slope = np.abs(current - previous)
            relative_slope = absolute_slope / scale
            normalized_first = current / scale
            normalized_prefix = current / np.maximum(prefix_median, query_floor)
            prefix_minimum = prefix.min(axis=2)
            rebound = np.maximum(current - prefix_minimum, 0.0) / scale

            if round_index >= 2:
                older = branch[:, :, round_index - 2]
                curvature = np.abs(current - 2.0 * previous + older) / scale
            else:
                curvature = np.full_like(current, np.nan)

            for mode in ("mean", "max", "p90"):
                add("amplitude", f"{branch_name}_level_{mode}", _aggregate_sites(current, mode))
                add(
                    "query_normalized",
                    f"{branch_name}_first_ratio_{mode}",
                    _aggregate_sites(normalized_first, mode),
                )
                add(
                    "query_normalized",
                    f"{branch_name}_prefix_median_ratio_{mode}",
                    _aggregate_sites(normalized_prefix, mode),
                )
                add("slope", f"{branch_name}_slope_{mode}", _aggregate_sites(absolute_slope, mode))
                add(
                    "slope",
                    f"{branch_name}_relative_slope_{mode}",
                    _aggregate_sites(relative_slope, mode),
                )
                add("curvature", f"{branch_name}_curvature_{mode}", _aggregate_sites(curvature, mode))
                add("rebound", f"{branch_name}_rebound_{mode}", _aggregate_sites(rebound, mode))

            add("rebound", f"{branch_name}_increasing_fraction", np.mean(current > previous, axis=(1, 2)))
            layer_scale = np.maximum(first.mean(axis=2), query_floor[:, :, 0])
            add(
                "all_layer_veto",
                f"{branch_name}_all_layer_first_ratio_max",
                np.max(current.mean(axis=2) / layer_scale, axis=1),
            )
            add(
                "all_layer_veto",
                f"{branch_name}_all_layer_relative_slope_max",
                np.max(absolute_slope.mean(axis=2) / layer_scale, axis=1),
            )
            layer_curvature = (
                np.max(curvature.mean(axis=2), axis=1)
                if round_index >= 2
                else np.full(data.rows, np.nan)
            )
            add(
                "all_layer_veto",
                f"{branch_name}_all_layer_curvature_max",
                layer_curvature,
            )
            add(
                "all_layer_veto",
                f"{branch_name}_all_layer_rebound_max",
                np.max(rebound.mean(axis=2), axis=1),
            )

        routed = data.routed[:, :, round_index]
        shared = data.shared[:, :, round_index]
        total = data.total[:, :, round_index]
        for denominator_name, denominator in (("shared", shared), ("total", total)):
            ratio = routed / np.maximum(denominator, EPS)
            for mode in ("mean", "max", "p90"):
                add(
                    "branch_balance",
                    f"routed_to_{denominator_name}_{mode}",
                    _aggregate_sites(ratio, mode),
                )

    signals = [
        Signal(name=name, family=family, values=np.stack(values, axis=1))
        for (family, name), values in sorted(sequences.items())
    ]
    if len({signal.name for signal in signals}) != len(signals):
        raise AssertionError("activation signal names must be unique")
    return signals


def stops_for_threshold(values: np.ndarray, threshold: float, min_stage: int) -> np.ndarray:
    eligible = STAGES >= int(min_stage)
    trigger = np.isfinite(values) & (values <= threshold) & eligible[None]
    found = trigger.any(axis=1)
    first = np.argmax(trigger, axis=1)
    return np.where(found, STAGES[first], TOTAL_ROUNDS).astype(np.int64)


def _selected(value: np.ndarray, stops: np.ndarray, full_value: float | bool) -> np.ndarray:
    result = np.full(len(stops), full_value, dtype=value.dtype)
    early = stops < TOTAL_ROUNDS
    if np.any(early):
        index = stops[early] - int(STAGES[0])
        result[early] = value[np.flatnonzero(early), index]
    return result


def _candidate_thresholds(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.asarray([-np.inf], dtype=np.float64)
    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 257)))
    return np.concatenate(([-np.inf], candidates, [np.inf]))


def calibrate_threshold(
    values: np.ndarray,
    errors: ControllerErrors,
    tolerance: float,
    max_risk: float,
    min_stage: int,
) -> tuple[float, dict[str, float]]:
    unsafe = errors.unsafe(tolerance)
    best: tuple[tuple[float, float, float], float, dict[str, float]] | None = None
    for threshold in _candidate_thresholds(values):
        stops = stops_for_threshold(values, float(threshold), min_stage)
        selected_unsafe = _selected(unsafe, stops, False)
        risk = float(selected_unsafe.mean())
        if risk > max_risk + 1e-15:
            continue
        selected_rms = _selected(errors.continuous_rms, stops, 0.0)
        saving = float(np.mean((TOTAL_ROUNDS - stops) / TOTAL_ROUNDS))
        objective = (saving, -risk, -float(selected_rms.mean()))
        stats = {"saving": saving, "risk": risk, "mean_continuous_rms": float(selected_rms.mean())}
        if best is None or objective > best[0]:
            best = (objective, float(threshold), stats)
    if best is None:
        raise AssertionError("never-stop threshold must be feasible")
    return best[1], best[2]


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def aggregate_result(
    data: ActivationFlowData,
    signal: Signal,
    estimator: str,
    min_stage: int,
    stops: np.ndarray,
    errors: ControllerErrors,
    tolerance: float,
    thresholds: list[float],
    protocol: str,
) -> dict[str, Any]:
    continuous = _selected(errors.continuous_rms, stops, 0.0)
    translation = _selected(errors.translation_rms, stops, 0.0)
    rotation = _selected(errors.rotation_rms, stops, 0.0)
    flips = _selected(errors.gripper_flip, stops, False)
    unsafe = (continuous > tolerance) | flips
    early = stops < TOTAL_ROUNDS
    early_continuous = continuous[early]
    rollout_saving = [
        float(np.mean((TOTAL_ROUNDS - stops[data.rollout_id == rollout]) / TOTAL_ROUNDS))
        for rollout in np.unique(data.rollout_id)
    ]
    finite_thresholds = np.asarray(thresholds)[np.isfinite(thresholds)]
    return {
        "protocol": protocol,
        "family": signal.family,
        "signal": signal.name,
        "endpoint_estimator": estimator,
        "min_stage": int(min_stage),
        "saving": float(np.mean((TOTAL_ROUNDS - stops) / TOTAL_ROUNDS)),
        "unsafe_rate": float(unsafe.mean()),
        "unsafe_count": int(unsafe.sum()),
        "unsafe_rollout_count": int(len(np.unique(data.rollout_id[unsafe]))),
        "continuous_6d_rms_mean": float(continuous.mean()),
        "continuous_6d_rms_p90": float(np.percentile(continuous, 90)),
        "continuous_6d_rms_p95": float(np.percentile(continuous, 95)),
        "early_continuous_6d_rms_mean": (
            float(early_continuous.mean()) if len(early_continuous) else 0.0
        ),
        "translation_rms_mean": float(translation.mean()),
        "rotation_rms_mean": float(rotation.mean()),
        "gripper_sign_flip_rate": float(flips.mean()),
        "early_fraction": float(early.mean()),
        "mean_completed_rounds": float(stops.mean()),
        "rollout_saving_min": float(min(rollout_saving)),
        "rollout_saving_max": float(max(rollout_saving)),
        "thresholds": [_finite_or_none(value) for value in thresholds],
        "threshold_finite_mean": (
            float(finite_thresholds.mean()) if len(finite_thresholds) else None
        ),
        "stop_histogram": {
            str(stage): int(np.sum(stops == stage))
            for stage in range(int(STAGES[0]), TOTAL_ROUNDS + 1)
        },
    }


def loeo_sweep(
    data: ActivationFlowData,
    signals: list[Signal],
    errors_by_estimator: dict[str, ControllerErrors],
    tolerance: float,
    max_train_risk: float,
    min_stages: tuple[int, ...] = DEFAULT_MIN_STAGES,
) -> list[dict[str, Any]]:
    rollouts = np.unique(data.rollout_id)
    if len(rollouts) < 2 or np.any(rollouts < 0):
        raise ValueError("LOEO requires at least two valid complete-rollout ids")
    rows = []
    for signal in signals:
        for estimator in ESTIMATORS:
            error = errors_by_estimator[estimator]
            for min_stage in min_stages:
                stops = np.full(data.rows, TOTAL_ROUNDS, dtype=np.int64)
                thresholds = []
                for held_out in rollouts:
                    train = data.rollout_id != held_out
                    test = ~train
                    train_error = ControllerErrors(
                        continuous_rms=error.continuous_rms[train],
                        translation_rms=error.translation_rms[train],
                        rotation_rms=error.rotation_rms[train],
                        gripper_flip=error.gripper_flip[train],
                    )
                    threshold, _ = calibrate_threshold(
                        signal.values[train], train_error, tolerance, max_train_risk, min_stage
                    )
                    thresholds.append(threshold)
                    stops[test] = stops_for_threshold(signal.values[test], threshold, min_stage)
                rows.append(
                    aggregate_result(
                        data, signal, estimator, min_stage, stops, error,
                        tolerance, thresholds, "safety_calibrated_loeo"
                    )
                )
    return rows


def fixed_stage_rows(
    data: ActivationFlowData,
    errors_by_estimator: dict[str, ControllerErrors],
    tolerance: float,
) -> list[dict[str, Any]]:
    rows = []
    placeholder = Signal("fixed_stage", "fixed", np.empty((data.rows, len(STAGES))))
    for estimator, errors in errors_by_estimator.items():
        for stage in (*STAGES.tolist(), TOTAL_ROUNDS):
            stops = np.full(data.rows, int(stage), dtype=np.int64)
            row = aggregate_result(
                data, placeholder, estimator, int(stage), stops, errors,
                tolerance, [], "fixed_stage"
            )
            row["stage"] = int(stage)
            rows.append(row)
    return rows


def matched_fixed_row(
    rows: list[dict[str, Any]], target_saving: float, estimator: str
) -> dict[str, Any]:
    candidates = [row for row in rows if row["endpoint_estimator"] == estimator]
    if not candidates:
        raise ValueError("no fixed rows for endpoint estimator %s" % estimator)
    return min(
        candidates,
        key=lambda row: (
            abs(row["saving"] - target_saving),
            max(0.0, row["saving"] - target_saving),
            row["unsafe_rate"],
        ),
    )


def select_champion(rows: list[dict[str, Any]], accept_risk: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty LOEO sweep")
    return min(
        rows,
        key=lambda row: (
            row["unsafe_rate"] > accept_risk,
            row["unsafe_rate"],
            -row["saving"],
            row["gripper_sign_flip_rate"],
            row["continuous_6d_rms_p95"],
            row["signal"],
            row["endpoint_estimator"],
            row["min_stage"],
        ),
    )


def fit_frozen_rule(
    data: ActivationFlowData,
    signals: list[Signal],
    errors_by_estimator: dict[str, ControllerErrors],
    champion: dict[str, Any],
    tolerance: float,
    max_train_risk: float,
) -> dict[str, Any]:
    signal = next(value for value in signals if value.name == champion["signal"])
    estimator = champion["endpoint_estimator"]
    threshold, stats = calibrate_threshold(
        signal.values,
        errors_by_estimator[estimator],
        tolerance,
        max_train_risk,
        int(champion["min_stage"]),
    )
    return {
        "family": signal.family,
        "signal": signal.name,
        "endpoint_estimator": estimator,
        "min_stage": int(champion["min_stage"]),
        "threshold": float(threshold),
        "threshold_report": _finite_or_none(threshold),
        "never_stop": bool(np.isneginf(threshold)),
        "development_fit": stats,
    }


def apply_frozen_rule(
    data: ActivationFlowData,
    signals: list[Signal],
    errors_by_estimator: dict[str, ControllerErrors],
    rule: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    signal = next(value for value in signals if value.name == rule["signal"])
    stops = stops_for_threshold(signal.values, float(rule["threshold"]), int(rule["min_stage"]))
    return aggregate_result(
        data,
        signal,
        rule["endpoint_estimator"],
        int(rule["min_stage"]),
        stops,
        errors_by_estimator[rule["endpoint_estimator"]],
        tolerance,
        [float(rule["threshold"])],
        "frozen_holdout",
    )


def _errors(data: ActivationFlowData) -> tuple[dict[str, np.ndarray], dict[str, ControllerErrors]]:
    predictions = endpoint_predictions(data.x)
    return predictions, {
        name: controller_errors(data.x, prediction, data.action_std)
        for name, prediction in predictions.items()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row if key not in ("thresholds", "stop_histogram")})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _pct(value: float) -> str:
    return "%.1f%%" % (100.0 * value)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    champion = summary["champion_loeo"]
    matched = summary["matched_fixed_development"]
    lines = [
        "# HB-MoE activation early-stop report",
        "",
        "## Definition",
        "",
        "- Signal uses true weighted routed-expert RMS, shared RMS, or total-MoE RMS; router probability is not used.",
        "- Stage s sees activation forwards 0..s-1 only. Round-10 action is an offline endpoint label.",
        "- Main error is physical RMS over action dimensions 0:6; gripper dimension 6 is evaluated only by sign flip.",
        "- Folds leave one complete episode_id rollout out. The physical reference band is an equivalence baseline, not a control-safety guarantee.",
        "- The activation champion is exploratory. Admission uses the fixed-round endpoint with nearest development saving as its control.",
        "",
        "## Development LOEO",
        "",
        "| signal | endpoint | min stage | saving | unsafe | 6D RMS P95 | gripper flip |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| `%s` | %s | %d | %s | %s | %.6f | %s |"
        % (
            champion["signal"], champion["endpoint_estimator"], champion["min_stage"],
            _pct(champion["saving"]), _pct(champion["unsafe_rate"]),
            champion["continuous_6d_rms_p95"], _pct(champion["gripper_sign_flip_rate"]),
        ),
        "| fixed stage %d | %s | %d | %s | %s | %.6f | %s |"
        % (
            matched["stage"], matched["endpoint_estimator"], matched["stage"],
            _pct(matched["saving"]), _pct(matched["unsafe_rate"]),
            matched["continuous_6d_rms_p95"], _pct(matched["gripper_sign_flip_rate"]),
        ),
    ]
    if summary.get("frozen_holdout") is not None:
        held = summary["frozen_holdout"]
        held_fixed = summary["matched_fixed_holdout"]
        lines += [
            "",
            "## Frozen Holdout",
            "",
            "| method | saving | unsafe | 6D RMS P95 | gripper flip |",
            "|---|---:|---:|---:|---:|",
            "| frozen activation | %s | %s | %.6f | %s |"
            % (
                _pct(held["saving"]), _pct(held["unsafe_rate"]),
                held["continuous_6d_rms_p95"], _pct(held["gripper_sign_flip_rate"]),
            ),
            "| fixed stage %d: %s | %s | %.6f | %s |"
            % (
                held_fixed["stage"], _pct(held_fixed["saving"]),
                _pct(held_fixed["unsafe_rate"]), held_fixed["continuous_6d_rms_p95"],
                _pct(held_fixed["gripper_sign_flip_rate"]),
            ),
            "",
            "Admission versus matched fixed round: **%s**."
            % ("PASS" if summary["admission"]["passes"] else "FAIL"),
        ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="development activation_flow.zarr or parent")
    parser.add_argument("--holdout-run", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--physical-rms-tolerance", type=float, default=None)
    parser.add_argument("--max-train-risk", type=float, default=0.0)
    parser.add_argument("--accept-risk", type=float, default=0.01)
    parser.add_argument("--min-stages", default="4,6,8")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    min_stages = tuple(int(value) for value in args.min_stages.split(","))
    if not min_stages or any(stage not in STAGES for stage in min_stages):
        raise ValueError("--min-stages must be comma-separated values from 2..9")
    development = load_activation_flow(args.run, require_loeo=True)
    predictions, errors = _errors(development)
    reference_index = 8 - int(STAGES[0])
    default_tolerance = float(
        np.percentile(errors["constant_acceleration"].continuous_rms[:, reference_index], 95)
    )
    tolerance = (
        max(default_tolerance, EPS)
        if args.physical_rms_tolerance is None
        else float(args.physical_rms_tolerance)
    )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("physical RMS tolerance must be finite and positive")

    signals = build_activation_signals(development)
    loeo = loeo_sweep(
        development, signals, errors, tolerance, args.max_train_risk, min_stages
    )
    champion = select_champion(loeo, args.accept_risk)
    frozen_rule = fit_frozen_rule(
        development, signals, errors, champion, tolerance, args.max_train_risk
    )
    fixed = fixed_stage_rows(development, errors, tolerance)
    matched_development = matched_fixed_row(
        fixed, champion["saving"], champion["endpoint_estimator"]
    )

    holdout_summary = None
    holdout_fixed = []
    matched_holdout = None
    admission = {
        "status": "exploratory_only_without_holdout",
        "passes": False,
        "criterion": (
            "frozen activation unsafe and gripper-flip rates <= matched fixed; "
            "saving >= matched fixed"
        ),
    }
    if args.holdout_run:
        holdout = load_activation_flow(args.holdout_run, require_loeo=False)
        if list(holdout.hb_layers) != list(development.hb_layers):
            raise ValueError("development and holdout HB layer axes differ")
        _holdout_predictions, holdout_errors = _errors(holdout)
        holdout_signals = build_activation_signals(holdout)
        holdout_summary = apply_frozen_rule(
            holdout, holdout_signals, holdout_errors, frozen_rule, tolerance
        )
        holdout_fixed = fixed_stage_rows(holdout, holdout_errors, tolerance)
        matched_holdout = next(
            row
            for row in holdout_fixed
            if row["endpoint_estimator"] == matched_development["endpoint_estimator"]
            and row["stage"] == matched_development["stage"]
        )
        admission.update(
            status="evaluated_on_frozen_holdout",
            passes=bool(
                holdout_summary["unsafe_rate"] <= matched_holdout["unsafe_rate"] + 1e-15
                and holdout_summary["gripper_sign_flip_rate"]
                <= matched_holdout["gripper_sign_flip_rate"] + 1e-15
                and holdout_summary["saving"] + 1e-15 >= matched_holdout["saving"]
            ),
        )

    summary = {
        "definition": {
            "stages": STAGES.tolist(),
            "prefix_only": "stage s uses activation forwards 0..s-1",
            "endpoint_estimators": list(ESTIMATORS),
            "controller_metric": "physical RMS over dims 0:6; any dim-6 gripper sign flip is unsafe",
            "physical_rms_tolerance": tolerance,
            "physical_rms_tolerance_source": (
                "explicit CLI" if args.physical_rms_tolerance is not None
                else "development fixed-stage8 constant-acceleration 6D RMS P95"
            ),
            "cv": "leave one complete episode_id rollout out",
            "max_train_risk": args.max_train_risk,
            "accept_risk": args.accept_risk,
        },
        "development": {
            "store": str(development.store),
            "queries": development.rows,
            "rollouts": int(len(np.unique(development.rollout_id))),
            "signals": len(signals),
            "topk_weight_sum_error_max": development.topk_weight_sum_error_max,
        },
        "champion_loeo": champion,
        "champion_status": "exploratory; never an admission result by itself",
        "matched_fixed_development": matched_development,
        "frozen_rule": {key: value for key, value in frozen_rule.items() if key != "threshold"},
        "frozen_holdout": holdout_summary,
        "matched_fixed_holdout": matched_holdout,
        "admission": admission,
        "loeo": loeo,
        "fixed_development": fixed,
        "fixed_holdout": holdout_fixed,
    }
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    _write_csv(out / "loeo_sweep.csv", loeo)
    _write_csv(out / "fixed_development.csv", fixed)
    if holdout_fixed:
        _write_csv(out / "fixed_holdout.csv", holdout_fixed)
    write_report(out / "report.md", summary)
    print(
        "wrote %s: %d signals, %d LOEO rows; champion=%s/%s saving=%s unsafe=%s"
        % (
            out, len(signals), len(loeo), champion["signal"],
            champion["endpoint_estimator"], _pct(champion["saving"]),
            _pct(champion["unsafe_rate"]),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
