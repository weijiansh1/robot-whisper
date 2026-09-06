#!/usr/bin/env python3
"""Select the v12 progress-ratio operating point on the development cohort.

The grid is W x layer_group x K x quantile = 840 candidates, swept twice: once
with a single ``global`` threshold for the whole pooled reference corpus (the
primary, task-agnostic mode) and once with a ``per_task`` threshold (a control
that exists only to run gate K2).

Two properties are load-bearing and are enforced by the code shape rather than
by comment:

1. ``calibrate_threshold`` has no ``outcome`` argument.  Thresholds are pure
   order statistics of an unlabeled 16,000-trajectory reference corpus, so no
   development outcome can leak into a threshold value.
2. The order statistic is taken over the **per-trajectory minimum**, not over
   per-query scores.  ``moe-v7-0905/docs/CALIBRATION_VARIANTS_REPORT_ZH.md``
   §2.1 showed that the per-trajectory extremum implicitly performs the
   per-trajectory multiple-testing correction; a per-query quantile raised FPR
   by roughly two orders of magnitude.

Outcomes are read exactly once, after every threshold and alarm array exists,
and only for ``development_main``.  ``external_8b`` is never opened here.

The objective is not precision.  Because ``risk`` means "did not finish before
the horizon cap", still being alive at chunk q is itself evidence of risk, and
a detector that only alarms late inherits that base rate for free.  The
pre-registered rule therefore maximises detections in the **low survival prior**
region (matched prior < 0.25) subject to a hard timely-FPR cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
METHOD = BUNDLE / "method"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(METHOD))

import progress_ratio  # noqa: E402
from hazard_common import matched_prior, survival_prior  # noqa: E402
from progress_monitor import GlobalProgressProfile  # noqa: E402


CACHE_ROOT = BUNDLE / "results/progress_cache"
LABEL_ROOT = WORKSPACE / "double-selete/trainfree/results/timeout_extension_plus10"
DEFAULT_OUTPUT = BUNDLE / "results/operating_point"

MAIN_CACHE = CACHE_ROOT / "development_main.npz"
EXTRA_CACHE = CACHE_ROOT / "development_extra.npz"
MAIN_LABELS = LABEL_ROOT / "development_main_clean_labels.csv"

CONFIRMATION_GRID = (1, 2, 3, 4, 6)
QUANTILE_GRID = (0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
THRESHOLD_MODES = ("global", "per_task")
EPS_LENGTH_QUANTILE = 0.01  # 预注册的无标签低分位（q1）
LOW_PRIOR_MAX = 0.25
REFERENCE_EPISODES = 16_000
MIN_REFERENCE_TRAJECTORIES = 32  # progress_ratio.quantile_lower 的硬下限

FPR_CAP = 0.005
MIN_LOW_PRIOR_TP = 20

SCHEMA = "himoe.progress_ratio_v12.selection.v1"
CANDIDATE_COLUMNS = (
    "window",
    "layer_group",
    "confirmations",
    "quantile",
    "threshold_mode",
    "ratio_threshold",
    "eps_length",
    "tp",
    "fp",
    "recall",
    "precision",
    "timely_fpr",
    "low_prior_tp",
    "low_prior_fp",
    "low_prior_precision",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# 预注册的纯决策函数：先被单测钉死，再被扫描调用
# --------------------------------------------------------------------------- #


def calibrate_threshold(
    ratio: np.ndarray, valid: np.ndarray, confirmations: int, quantile: float
) -> float:
    """无标签阈值：逐轨迹最小值的下尾分位数。签名里没有 outcome，这是刻意的。"""
    return progress_ratio.quantile_lower(
        reference_minima(ratio, valid, confirmations), quantile
    )


def reference_minima(
    ratio: np.ndarray, valid: np.ndarray, confirmations: int
) -> np.ndarray:
    """``calibrate_threshold`` 的前半段，拆出来只为在扫描中复用，语义完全一致。"""
    persistent = progress_ratio.persistent_low(ratio, confirmations)
    persistent = np.where(valid, persistent, np.nan)
    return progress_ratio.row_min(persistent)


def choose(candidates: pd.DataFrame) -> pd.Series | None:
    """预注册选择规则。不满足约束返回 None，不放宽重试。"""
    feasible = candidates[
        (candidates["timely_fpr"] <= FPR_CAP)
        & (candidates["low_prior_tp"] >= MIN_LOW_PRIOR_TP)
    ]
    if feasible.empty:
        return None
    ordered = feasible.sort_values(
        ["low_prior_tp", "low_prior_precision", "window"],
        ascending=[False, False, True],
    )
    return ordered.iloc[0]


def k2_gate(global_tp: int, per_task_tp: int) -> bool:
    """global 模式的 TP 不得低于 per_task 模式的一半。"""
    if per_task_tp <= 0:
        return False
    return global_tp >= 0.5 * per_task_tp


# --------------------------------------------------------------------------- #
# I/O 与对齐
# --------------------------------------------------------------------------- #


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def task_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    return cache["task_names"].astype(str)[cache["task_index"].astype(int)]


def aligned_labels(
    cache: dict[str, np.ndarray], path: Path, cohort: str
) -> pd.DataFrame:
    """v7 evaluate_intrinsic_guard_v7.aligned_labels 的逐行等价实现。"""
    task = task_of(cache)
    index = pd.DataFrame(
        {
            "row": np.arange(len(task)),
            "task": task,
            "episode": cache["episode"].astype(int),
            "length": cache["length"].astype(int),
        }
    )
    labels = pd.read_csv(path)[
        ["task", "episode", "original_failure", "failure", "late_success_plus10_queries"]
    ]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("row")
    if merged["original_failure"].isna().any():
        raise ValueError(f"{cohort} labels do not align")
    merged["cohort"] = cohort
    merged["suite"] = merged["task"].str.split("/", n=1).str[0]
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #


def first_below_rowwise(
    values: np.ndarray, thresholds: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """``progress_ratio.first_below`` 的逐行阈值版本，用于 per_task 对照模式。

    阈值为 NaN（该任务参考量不足）时该行永不触发。
    """
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if values.shape != valid.shape or len(values) != len(thresholds):
        raise ValueError("score, validity and threshold arrays do not align")
    with np.errstate(invalid="ignore"):
        trigger = np.isfinite(values) & (values < thresholds[:, None]) & valid
    any_trigger = trigger.any(axis=1)
    first = np.full(len(values), -1, dtype=np.int16)
    first[any_trigger] = trigger[any_trigger].argmax(axis=1).astype(np.int16)
    return first


def ratio_or_nan(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def score(
    first: np.ndarray,
    risk: np.ndarray,
    prior_matrix: np.ndarray,
) -> dict[str, Any]:
    """开发集上的结果指标。这是脚本里唯一读 outcome 的地方。"""
    alarm = first >= 0
    timely = ~risk
    rows = np.flatnonzero(alarm)
    alarm_prior = prior_matrix[rows, first[rows]]
    if not np.isfinite(alarm_prior).all():
        raise ValueError("an alarm landed on a chunk with no survival prior")
    low = alarm_prior < LOW_PRIOR_MAX
    low_risk = risk[rows][low]
    low_prior_tp = int(low_risk.sum())
    low_prior_fp = int((~low_risk).sum())
    tp = int((alarm & risk).sum())
    fp = int((alarm & timely).sum())
    return {
        "tp": tp,
        "fp": fp,
        "recall": ratio_or_nan(tp, int(risk.sum())),
        "precision": ratio_or_nan(tp, tp + fp),
        "timely_fpr": ratio_or_nan(fp, int(timely.sum())),
        "low_prior_tp": low_prior_tp,
        "low_prior_fp": low_prior_fp,
        "low_prior_precision": ratio_or_nan(
            low_prior_tp, low_prior_tp + low_prior_fp
        ),
    }


def prior_lookup(labels: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
    """[E, Q] 的 P(risk | 仍在运行于 chunk q)，缺口为 NaN。"""
    priors = survival_prior(labels)
    rows, chunks = np.nonzero(valid)
    frame = pd.DataFrame(
        {"suite": labels["suite"].to_numpy(str)[rows], "alarm_chunk": chunks}
    )
    matrix = np.full(valid.shape, np.nan, dtype=np.float64)
    matrix[rows, chunks] = matched_prior(frame, priors)
    return matrix


def prior_sanity(labels: pd.DataFrame) -> dict[str, Any]:
    """先验应随 chunk 单调上升，并在 horizon cap 附近趋近 1。"""
    priors = survival_prior(labels)
    report: dict[str, Any] = {}
    for suite, table in sorted(priors.items()):
        chunks = sorted(table)
        curve = np.asarray([table[chunk] for chunk in chunks], dtype=np.float64)
        report[suite] = {
            "max_chunk": int(chunks[-1]),
            "prior_at_chunk_0": float(curve[0]),
            "prior_at_chunk_4": float(curve[4]) if len(curve) > 4 else None,
            "prior_at_max_chunk": float(curve[-1]),
            "monotone_nondecreasing": bool(np.all(np.diff(curve) >= -1e-12)),
            "decreasing_steps": int((np.diff(curve) < -1e-12).sum()),
        }
    return report


def sweep(
    pooled_ratio_source: dict[str, np.ndarray],
    main_rows: int,
    main_valid: np.ndarray,
    pooled_valid: np.ndarray,
    pooled_task: np.ndarray,
    main_task: np.ndarray,
    risk: np.ndarray,
    prior_matrix: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lag_distance = pooled_ratio_source["lag_distance"]
    adjacent = lag_distance[:, :, 0, :]
    unique_tasks = np.unique(main_task)
    rows: list[dict[str, Any]] = []
    eps_by_window: dict[int, float] = {}
    thin_reference = 0

    for window in progress_ratio.W_GRID:
        started = time.perf_counter()
        length = progress_ratio.path_length(adjacent, window)
        finite_length = length[np.isfinite(length)]
        eps_length = float(np.quantile(finite_length, EPS_LENGTH_QUANTILE))
        eps_by_window[window] = eps_length
        displacement = lag_distance[:, :, progress_ratio.LAGS.index(window), :]
        elementwise = progress_ratio.progress_ratio(displacement, length, eps_length)

        for group in progress_ratio.LAYER_GROUPS:
            grouped = progress_ratio.group_ratio(elementwise, group)
            for confirmations in CONFIRMATION_GRID:
                persistent = progress_ratio.persistent_low(grouped, confirmations)
                persistent = np.where(pooled_valid, persistent, np.nan)
                minima = progress_ratio.row_min(persistent)
                main_persistent = persistent[:main_rows]

                task_minima = {
                    task: minima[pooled_task == task] for task in unique_tasks
                }
                for quantile in QUANTILE_GRID:
                    threshold = progress_ratio.quantile_lower(minima, quantile)
                    first = progress_ratio.first_below(
                        main_persistent, threshold, main_valid
                    )
                    rows.append(
                        {
                            "window": window,
                            "layer_group": group,
                            "confirmations": confirmations,
                            "quantile": quantile,
                            "threshold_mode": "global",
                            "ratio_threshold": threshold,
                            "eps_length": eps_length,
                            **score(first, risk, prior_matrix),
                        }
                    )

                    per_task: dict[str, float] = {}
                    for task, values in task_minima.items():
                        usable = int(np.isfinite(values).sum())
                        if usable < MIN_REFERENCE_TRAJECTORIES:
                            per_task[task] = float("nan")
                            thin_reference += 1
                            continue
                        per_task[task] = progress_ratio.quantile_lower(values, quantile)
                    row_threshold = np.asarray(
                        [per_task[task] for task in main_task], dtype=np.float64
                    )
                    first = first_below_rowwise(
                        main_persistent, row_threshold, main_valid
                    )
                    finite_thresholds = np.asarray(
                        [value for value in per_task.values() if np.isfinite(value)]
                    )
                    rows.append(
                        {
                            "window": window,
                            "layer_group": group,
                            "confirmations": confirmations,
                            "quantile": quantile,
                            "threshold_mode": "per_task",
                            "ratio_threshold": float(np.median(finite_thresholds))
                            if len(finite_thresholds)
                            else float("nan"),
                            "eps_length": eps_length,
                            **score(first, risk, prior_matrix),
                        }
                    )
        print(
            f"[W={window:2d}] eps_length={eps_length:.6f} "
            f"rows={len(rows)}/{len(progress_ratio.W_GRID) * 3 * len(CONFIRMATION_GRID) * len(QUANTILE_GRID) * 2} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    return rows, {
        "eps_length_by_window": eps_by_window,
        "per_task_thin_reference_cells": thin_reference,
    }


def row_dict(row: pd.Series | None) -> dict[str, Any] | None:
    if row is None:
        return None
    output = plain(row.to_dict())
    for key in ("tp", "fp", "low_prior_tp", "low_prior_fp", "window", "confirmations"):
        if output.get(key) is not None:
            output[key] = int(output[key])
    return output


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    main_cache = load_npz(MAIN_CACHE)
    extra_cache = load_npz(EXTRA_CACHE)
    main_rows = len(main_cache["episode"])
    extra_rows = len(extra_cache["episode"])
    if main_rows + extra_rows != REFERENCE_EPISODES:
        raise ValueError(
            f"expected {REFERENCE_EPISODES:,} pooled reference trajectories, "
            f"got {main_rows + extra_rows:,}"
        )
    for cohort, cache in (("development_main", main_cache), ("development_extra", extra_cache)):
        if not np.array_equal(
            np.asarray(cache["lags"]).astype(int), np.asarray(progress_ratio.LAGS)
        ):
            raise ValueError(f"{cohort} cache lags disagree with the method module")

    main_task = task_of(main_cache)
    extra_task = task_of(extra_cache)
    overlap = sorted(set(main_task.tolist()) & set(extra_task.tolist()))
    pooled_task = np.concatenate((main_task, extra_task))
    pooled_valid = np.concatenate(
        (main_cache["valid"].astype(bool), extra_cache["valid"].astype(bool))
    )
    pooled = {
        "lag_distance": np.concatenate(
            (main_cache["lag_distance"], extra_cache["lag_distance"]), axis=0
        )
    }
    main_valid = pooled_valid[:main_rows]

    # 这是脚本里唯一一次打开 outcome 文件；external_8b 从头到尾没有被引用。
    labels = aligned_labels(main_cache, MAIN_LABELS, "development_main")
    risk = labels["original_failure"].to_numpy(bool)
    prior_matrix = prior_lookup(labels, main_valid)
    sanity = prior_sanity(labels)
    print(
        f"reference={main_rows + extra_rows:,} trajectories "
        f"({len(main_cache['task_names'])}+{len(extra_cache['task_names'])} tasks); "
        f"scored={main_rows:,} labelled trajectories, "
        f"risk={int(risk.sum()):,}, timely={int((~risk).sum()):,}",
        flush=True,
    )
    for suite, block in sanity.items():
        print(
            f"  survival prior {suite}: q0={block['prior_at_chunk_0']:.3f} "
            f"-> q{block['max_chunk']}={block['prior_at_max_chunk']:.3f} "
            f"monotone={block['monotone_nondecreasing']}",
            flush=True,
        )

    rows, sweep_meta = sweep(
        pooled,
        main_rows,
        main_valid,
        pooled_valid,
        pooled_task,
        main_task,
        risk,
        prior_matrix,
    )
    candidates = pd.DataFrame(rows)[list(CANDIDATE_COLUMNS)].sort_values(
        ["threshold_mode", "window", "layer_group", "confirmations", "quantile"],
        kind="stable",
    )
    candidate_path = args.output / "development_candidates.csv"
    candidates.to_csv(candidate_path, index=False)

    global_candidates = candidates[candidates["threshold_mode"] == "global"]
    per_task_candidates = candidates[candidates["threshold_mode"] == "per_task"]
    global_choice = choose(global_candidates)
    per_task_choice = choose(per_task_candidates)

    def feasible_count(frame: pd.DataFrame) -> int:
        return int(
            (
                (frame["timely_fpr"] <= FPR_CAP)
                & (frame["low_prior_tp"] >= MIN_LOW_PRIOR_TP)
            ).sum()
        )

    def capped_best(frame: pd.DataFrame, column: str) -> int:
        capped = frame[frame["timely_fpr"] <= FPR_CAP]
        return int(capped[column].max()) if not capped.empty else 0

    def reachable(threshold: float) -> dict[str, Any]:
        """低先验区域本身有多大：区分"检测器不行"与"区域根本够不着"。"""
        low = np.isfinite(prior_matrix) & (prior_matrix < threshold)
        any_low = low.any(axis=1)
        return {
            "low_prior_valid_chunks": int(low.sum()),
            "valid_chunks": int(main_valid.sum()),
            "low_prior_chunk_fraction": float(low.sum() / main_valid.sum()),
            "risk_episodes_with_a_low_prior_chunk": int((any_low & risk).sum()),
            "timely_episodes_with_a_low_prior_chunk": int((any_low & ~risk).sum()),
        }

    # K2：同格对照——只换阈值模式，网格坐标不动。
    matched_row = None
    if global_choice is not None:
        take = per_task_candidates[
            (per_task_candidates["window"] == global_choice["window"])
            & (per_task_candidates["layer_group"] == global_choice["layer_group"])
            & (per_task_candidates["confirmations"] == global_choice["confirmations"])
            & (per_task_candidates["quantile"] == global_choice["quantile"])
        ]
        matched_row = take.iloc[0] if not take.empty else None

    global_low = int(global_choice["low_prior_tp"]) if global_choice is not None else 0
    per_task_low = (
        int(per_task_choice["low_prior_tp"]) if per_task_choice is not None else 0
    )
    global_tp = int(global_choice["tp"]) if global_choice is not None else 0
    per_task_tp = int(per_task_choice["tp"]) if per_task_choice is not None else 0
    capped_global = capped_best(global_candidates, "low_prior_tp")
    capped_per_task = capped_best(per_task_candidates, "low_prior_tp")
    free_global = int(global_candidates["low_prior_tp"].max())
    free_per_task = int(per_task_candidates["low_prior_tp"].max())
    vacuous = global_choice is None or per_task_choice is None
    k2 = {
        "definition": (
            "global 模式选中点的低先验 TP 不得低于 per_task 模式选中点的一半"
        ),
        "primary_metric": "low_prior_tp",
        "global_low_prior_tp": global_low,
        "per_task_low_prior_tp": per_task_low,
        "verdict": bool(k2_gate(global_low, per_task_low)),
        "secondary_metric": "tp",
        "global_tp": global_tp,
        "per_task_tp": per_task_tp,
        "verdict_on_tp": bool(k2_gate(global_tp, per_task_tp)),
        # 任一模式无可行点时，上面的判定退化为 0 vs 0，只能记为 False；
        # 那是"没有选中点"的结果，不是"global 塌了"的证据。下面两个诊断
        # 才是 K2 真正要问的问题：换成 global 阈值后产量是否崩塌。
        "verdict_vacuous": bool(vacuous),
        "best_global_low_prior_tp_under_fpr_cap": capped_global,
        "best_per_task_low_prior_tp_under_fpr_cap": capped_per_task,
        "verdict_under_fpr_cap": bool(k2_gate(capped_global, capped_per_task)),
        "best_global_low_prior_tp_unconstrained": free_global,
        "best_per_task_low_prior_tp_unconstrained": free_per_task,
        "verdict_unconstrained": bool(k2_gate(free_global, free_per_task)),
        "matched_grid_cell_per_task": row_dict(matched_row),
    }

    profile_path = args.output / "global_profile.npz"
    profile_written = False
    if global_choice is not None:
        np.savez_compressed(
            profile_path,
            schema=np.asarray(progress_ratio.SCHEMA),
            window=np.asarray(int(global_choice["window"]), dtype=np.int32),
            confirmations=np.asarray(
                int(global_choice["confirmations"]), dtype=np.int32
            ),
            ratio_threshold=np.asarray(
                float(global_choice["ratio_threshold"]), dtype=np.float64
            ),
            eps_length=np.asarray(float(global_choice["eps_length"]), dtype=np.float64),
            layer_group=np.asarray(str(global_choice["layer_group"])),
        )
        reloaded = GlobalProgressProfile.load(profile_path)
        if (
            reloaded.window != int(global_choice["window"])
            or reloaded.confirmations != int(global_choice["confirmations"])
            or not math.isclose(
                reloaded.ratio_threshold, float(global_choice["ratio_threshold"])
            )
            or not math.isclose(reloaded.eps_length, float(global_choice["eps_length"]))
            or reloaded.layer_group != str(global_choice["layer_group"])
        ):
            raise ValueError("the frozen profile does not round-trip")
        with np.load(profile_path, allow_pickle=False) as archive:
            forbidden = {"task_names", "task_index"} & set(archive.files)
            if forbidden:
                raise ValueError(f"the global profile leaks task metadata: {forbidden}")
        profile_written = True

    elapsed = time.perf_counter() - started
    summary = {
        "schema": SCHEMA,
        "selection_rule": {
            "timely_fpr_at_most": FPR_CAP,
            "low_prior_tp_at_least": MIN_LOW_PRIOR_TP,
            "low_prior_definition": f"matched survival prior < {LOW_PRIOR_MAX}",
            "ranking": ["low_prior_tp desc", "low_prior_precision desc", "window asc"],
            "pre_registered": True,
            "relaxed_after_seeing_results": False,
        },
        "grid": {
            "window": list(progress_ratio.W_GRID),
            "layer_group": list(progress_ratio.LAYER_GROUPS),
            "confirmations": list(CONFIRMATION_GRID),
            "quantile": list(QUANTILE_GRID),
            "threshold_mode": list(THRESHOLD_MODES),
            "candidates_per_mode": len(global_candidates),
            "candidates_total": len(candidates),
        },
        "corpus": {
            "reference_trajectories": main_rows + extra_rows,
            "reference_tasks": len(main_cache["task_names"])
            + len(extra_cache["task_names"]),
            "reference_cohorts": ["development_main", "development_extra"],
            "reference_task_overlap": overlap,
            "development_main_episodes": main_rows,
            "development_main_tasks": int(len(main_cache["task_names"])),
            "development_extra_episodes": extra_rows,
            "development_extra_tasks": int(len(extra_cache["task_names"])),
            "scored_episodes": main_rows,
            "risk_episodes": int(risk.sum()),
            "timely_episodes": int((~risk).sum()),
            "external_data_loaded": False,
            "labels_used_for_threshold_calibration": [],
        },
        "eps_length_quantile": EPS_LENGTH_QUANTILE,
        "eps_length_by_window": plain(sweep_meta["eps_length_by_window"]),
        "per_task_thin_reference_cells": sweep_meta["per_task_thin_reference_cells"],
        "feasible": global_choice is not None,
        "feasible_global_candidates": feasible_count(global_candidates),
        "feasible_per_task_candidates": feasible_count(per_task_candidates),
        "fpr_cap_only_global_candidates": int(
            (global_candidates["timely_fpr"] <= FPR_CAP).sum()
        ),
        "fpr_cap_only_per_task_candidates": int(
            (per_task_candidates["timely_fpr"] <= FPR_CAP).sum()
        ),
        "low_prior_reachability": reachable(LOW_PRIOR_MAX),
        "selected": row_dict(global_choice),
        "per_task_selected": row_dict(per_task_choice),
        "k2_gate": k2,
        "profile_written": profile_written,
        "survival_prior_sanity": plain(sanity),
        "seconds": round(elapsed, 1),
        "artifacts": {
            "candidate_csv_sha256": sha256(candidate_path),
            "script_sha256": sha256(Path(__file__)),
            "main_cache_sha256": sha256(MAIN_CACHE),
            "extra_cache_sha256": sha256(EXTRA_CACHE),
            **(
                {"global_profile_sha256": sha256(profile_path)}
                if profile_written
                else {}
            ),
        },
    }
    selection_path = args.output / "selection.json"
    selection_path.write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"\nfeasible global candidates: {summary['feasible_global_candidates']}"
        f"/{len(global_candidates)} "
        f"(under the FPR cap alone: {summary['fpr_cap_only_global_candidates']})"
    )
    if global_choice is None:
        print("NO global operating point satisfies the pre-registered constraints.")
        print("feasible: false -- the full candidate table is kept, nothing relaxed.")
        print(
            f"  best low_prior_tp under timely_fpr<={FPR_CAP}: "
            f"global={capped_global}, per_task={capped_per_task} "
            f"(rule needs >= {MIN_LOW_PRIOR_TP})"
        )
    else:
        print(
            "selected (global): "
            f"W={int(global_choice['window'])} "
            f"group={global_choice['layer_group']} "
            f"K={int(global_choice['confirmations'])} "
            f"quantile={global_choice['quantile']} "
            f"r*={global_choice['ratio_threshold']:.6f} "
            f"eps_length={global_choice['eps_length']:.6f}"
        )
        print(
            f"  tp={int(global_choice['tp'])} fp={int(global_choice['fp'])} "
            f"recall={global_choice['recall']:.4f} "
            f"precision={global_choice['precision']:.4f} "
            f"timely_fpr={global_choice['timely_fpr']:.5f}"
        )
        print(
            f"  low_prior_tp={int(global_choice['low_prior_tp'])} "
            f"low_prior_fp={int(global_choice['low_prior_fp'])} "
            f"low_prior_precision={global_choice['low_prior_precision']:.4f}"
        )
    verdict = "PASS" if k2["verdict"] else "FAIL"
    print(
        f"K2 (selected points): global low_prior_tp={k2['global_low_prior_tp']} vs "
        f"per_task low_prior_tp={k2['per_task_low_prior_tp']} -> {verdict}"
        + (" [VACUOUS: at least one mode has no feasible point]" if vacuous else "")
    )
    print(
        f"K2 diagnostic under the FPR cap: global={capped_global} vs "
        f"per_task={capped_per_task} -> "
        f"{'PASS' if k2['verdict_under_fpr_cap'] else 'FAIL'}"
    )
    print(
        f"K2 diagnostic unconstrained: global={free_global} vs "
        f"per_task={free_per_task} -> "
        f"{'PASS' if k2['verdict_unconstrained'] else 'FAIL'}"
    )
    print(f"elapsed {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
