#!/usr/bin/env python3
"""K1 / K3: does the (high mobility, low R) "circling" phenotype exist at all?

Everything novel in the progress-ratio method lives in one cell of a 2x2 that
splits window mobility ``L`` (how far the routing travelled) against the
progress ratio ``R = D / L`` (how much of that travel was net displacement):

                low mobility          high mobility
    high R      slow but directed     healthy fast progress
    low R       frozen (v7 already    circling / repeated regrasp
                catches this)         <- the only new detection claimed

K1 asks whether that cell holds at least ``K1_MIN_RISK_EPISODES`` real failures.
K3 asks whether the R alarms on risk episodes are anything more than a relabelled
subset of the v7 ``main_freeze`` alarms.

**This script reports diagnostics, not a selection.**  Task 7 swept all 840
global candidates and found that *zero* satisfy the pre-registered rule
(``timely_fpr <= 0.005`` and ``low_prior_tp >= 20``); ``selection.json`` therefore
records ``feasible: false`` and no ``global_profile.npz`` exists.  K1 and K3 are
descriptive questions about the corpus, so they can still be answered -- but only
by reporting the whole picture rather than by anointing one point.  Two families
of five candidates are evaluated (see ``POINT_RULE``), every number is labelled a
diagnostic, and nothing here writes a profile or contradicts ``selection.json``.
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
import select_operating_point_v12 as sel  # noqa: E402


DEFAULT_OUTPUT = BUNDLE / "results/phenotype"
CANDIDATE_CSV = BUNDLE / "results/operating_point/development_candidates.csv"
SELECTION_JSON = BUNDLE / "results/operating_point/selection.json"
V7_ALARMS = WORKSPACE / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"
V7_KEY = "main_freeze"

SCHEMA = "himoe.progress_ratio_v12.phenotype.v1"
CELLS = ("low_mob_low_R", "low_mob_high_R", "high_mob_low_R", "high_mob_high_R")

K1_MIN_RISK_EPISODES = 20
K3_MIN_NOVEL_SHARE = 0.10

POINTS_PER_FAMILY = 5
POINT_SORT_KEYS = (
    "low_prior_tp",
    "low_prior_precision",
    "window",
    "confirmations",
    "quantile",
    "layer_group",
)
POINT_SORT_ASCENDING = (False, False, True, True, True, True)
POINT_RULE = (
    "Diagnostic points only -- NOT a selection.  Both families are drawn from "
    "results/operating_point/development_candidates.csv restricted to "
    "threshold_mode == 'global', sorted by low_prior_tp desc, then "
    "low_prior_precision desc, then window asc, confirmations asc, quantile asc, "
    "layer_group asc, taking the first 5.  Family 'fpr_capped' first filters "
    f"timely_fpr <= {sel.FPR_CAP}; family 'unconstrained' applies no filter."
)


# --------------------------------------------------------------------------- #
# 预注册的纯闸门函数：先被单测钉死，再被诊断脚本调用
# --------------------------------------------------------------------------- #


def assign_cells(
    mobility: np.ndarray, ratio: np.ndarray, mobility_median: float, ratio_threshold: float
) -> np.ndarray:
    """报警时刻的 2x2 表型标签。"""
    high_mobility = np.asarray(mobility, dtype=np.float32) >= mobility_median
    low_ratio = np.asarray(ratio, dtype=np.float32) < ratio_threshold
    labels = np.empty(len(high_mobility), dtype=object)
    labels[~high_mobility & low_ratio] = "low_mob_low_R"
    labels[~high_mobility & ~low_ratio] = "low_mob_high_R"
    labels[high_mobility & low_ratio] = "high_mob_low_R"
    labels[high_mobility & ~low_ratio] = "high_mob_high_R"
    return labels


def k1_gate(cells: np.ndarray, risk: np.ndarray) -> tuple[int, bool]:
    count = int(((cells == "high_mob_low_R") & np.asarray(risk, dtype=bool)).sum())
    return count, count >= K1_MIN_RISK_EPISODES


def k3_gate(
    ratio_first: np.ndarray, freeze_first: np.ndarray, risk: np.ndarray
) -> tuple[float, bool]:
    risk = np.asarray(risk, dtype=bool)
    detected = risk & (np.asarray(ratio_first) >= 0)
    if not detected.any():
        return 0.0, False
    novel = detected & (np.asarray(freeze_first) < 0)
    share = float(novel.sum() / detected.sum())
    return share, share >= K3_MIN_NOVEL_SHARE


# --------------------------------------------------------------------------- #
# 诊断点的选取（陈述式规则，见 POINT_RULE）
# --------------------------------------------------------------------------- #


def rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        list(POINT_SORT_KEYS), ascending=list(POINT_SORT_ASCENDING), kind="stable"
    )


def diagnostic_points(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    """两族各 5 个点。规则写在 POINT_RULE 里，并原样写进 gates.json。"""
    globals_only = candidates[candidates["threshold_mode"] == "global"]
    families = {
        "fpr_capped": globals_only[globals_only["timely_fpr"] <= sel.FPR_CAP],
        "unconstrained": globals_only,
    }
    points: list[dict[str, Any]] = []
    for family, frame in families.items():
        for _, row in rank(frame).head(POINTS_PER_FAMILY).iterrows():
            record = sel.plain(row.to_dict())
            record["family"] = family
            for key in ("window", "confirmations", "tp", "fp", "low_prior_tp", "low_prior_fp"):
                record[key] = int(record[key])
            points.append(record)
    return points


# --------------------------------------------------------------------------- #
# 每点的几何量：L 与 R 都按同一层组取中位数（progress_ratio.group_ratio）
# --------------------------------------------------------------------------- #


class Geometry:
    """按 (W) -> (W, group) -> (W, group, K) 逐级缓存，避免重复重算。"""

    def __init__(self, lag_distance: np.ndarray, valid: np.ndarray) -> None:
        self.adjacent = lag_distance[:, :, 0, :]
        self.lag_distance = lag_distance
        self.valid = valid
        self._window: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        self._group: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        self._persistent: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}

    def window(self, window: int) -> tuple[np.ndarray, np.ndarray, float]:
        """[E, Q, 8] 路径长、逐层 R，以及该 W 的 eps_length。"""
        if window not in self._window:
            length = progress_ratio.path_length(self.adjacent, window)
            eps_length = float(
                np.quantile(length[np.isfinite(length)], sel.EPS_LENGTH_QUANTILE)
            )
            displacement = self.lag_distance[:, :, progress_ratio.LAGS.index(window), :]
            ratio = progress_ratio.progress_ratio(displacement, length, eps_length)
            self._window[window] = (length, ratio, eps_length)
        return self._window[window]

    def group(self, window: int, group: str) -> tuple[np.ndarray, np.ndarray]:
        """[E, Q] 组中位数路径长与组中位数 R。两者用同一个 group_ratio 归约。"""
        key = (window, group)
        if key not in self._group:
            length, ratio, _ = self.window(window)
            self._group[key] = (
                progress_ratio.group_ratio(length, group),
                progress_ratio.group_ratio(ratio, group),
            )
        return self._group[key]

    def persistent(self, window: int, group: str, confirmations: int) -> tuple[np.ndarray, np.ndarray]:
        """滚动最大值分数与逐轨迹最小值，语义与选点扫描逐行一致。"""
        key = (window, group, confirmations)
        if key not in self._persistent:
            _, grouped_ratio = self.group(window, group)
            persistent = progress_ratio.persistent_low(grouped_ratio, confirmations)
            persistent = np.where(self.valid, persistent, np.nan)
            self._persistent[key] = (persistent, progress_ratio.row_min(persistent))
        return self._persistent[key]


def occupancy(
    mobility: np.ndarray,
    ratio: np.ndarray,
    usable: np.ndarray,
    risk_rows: np.ndarray,
    mobility_median: float,
    ratio_threshold: float,
) -> dict[str, Any]:
    """全部可用 chunk 的四格占比，按 episode 是否为 risk 拆分。"""
    high = mobility >= mobility_median
    low = ratio < ratio_threshold
    masks = {
        "low_mob_low_R": ~high & low,
        "low_mob_high_R": ~high & ~low,
        "high_mob_low_R": high & low,
        "high_mob_high_R": high & ~low,
    }
    report: dict[str, Any] = {}
    for name, rows in (("risk", risk_rows), ("timely", ~risk_rows)):
        pool = usable & rows[:, None]
        total = int(pool.sum())
        block: dict[str, Any] = {"valid_chunks": total, "counts": {}, "fraction": {}}
        for cell, mask in masks.items():
            count = int((pool & mask).sum())
            block["counts"][cell] = count
            block["fraction"][cell] = float(count / total) if total else float("nan")
        # episode 级：这条轨迹在全程里有没有出现过该表型
        block["episodes"] = int(rows.sum())
        block["episodes_with_cell"] = {
            cell: int((((pool & mask).any(axis=1)) & rows).sum())
            for cell, mask in masks.items()
        }
        report[name] = block
    return report


def cross_tab(cells: np.ndarray, risk: np.ndarray) -> dict[str, dict[str, int]]:
    risk = np.asarray(risk, dtype=bool)
    return {
        "risk": {cell: int(((cells == cell) & risk).sum()) for cell in CELLS},
        "timely": {cell: int(((cells == cell) & ~risk).sum()) for cell in CELLS},
    }


def overlap(
    ratio_first: np.ndarray, freeze_first: np.ndarray, risk: np.ndarray
) -> dict[str, Any]:
    """与 v7 main_freeze 的逐 episode 重叠。"""
    risk = np.asarray(risk, dtype=bool)
    ratio_hit = ratio_first >= 0
    freeze_hit = freeze_first >= 0
    both = ratio_hit & freeze_hit
    report: dict[str, Any] = {
        "v7_key": V7_KEY,
        "v7_risk_alarms": int((freeze_hit & risk).sum()),
        "v7_timely_alarms": int((freeze_hit & ~risk).sum()),
    }
    for name, rows in (("risk", risk), ("timely", ~risk)):
        report[name] = {
            "ratio_alarms": int((ratio_hit & rows).sum()),
            "ratio_and_freeze": int((both & rows).sum()),
            "ratio_only_novel": int((ratio_hit & ~freeze_hit & rows).sum()),
            "freeze_only": int((~ratio_hit & freeze_hit & rows).sum()),
            "neither": int((~ratio_hit & ~freeze_hit & rows).sum()),
            "ratio_strictly_earlier_when_both": int(
                (both & rows & (ratio_first < freeze_first)).sum()
            ),
            "ratio_strictly_later_when_both": int(
                (both & rows & (ratio_first > freeze_first)).sum()
            ),
        }
    return report


def format_table(counts: dict[str, dict[str, int]], mobility_median: float) -> str:
    lines = [
        f"    mobility_median(L) = {mobility_median:.6f}   "
        "(alarms are low-R by construction: the alarm rule *is* R < r*)",
        "                     |  low mobility   |  high mobility  |",
        "    -----------------+-----------------+-----------------+",
    ]
    for label, low_cell, high_cell in (
        ("high R", "low_mob_high_R", "high_mob_high_R"),
        ("low R ", "low_mob_low_R", "high_mob_low_R"),
    ):
        risk_low = counts["risk"][low_cell]
        risk_high = counts["risk"][high_cell]
        timely_low = counts["timely"][low_cell]
        timely_high = counts["timely"][high_cell]
        lines.append(
            f"    {label}  risk/timely| {risk_low:6d} /{timely_low:7d} |"
            f" {risk_high:6d} /{timely_high:7d} |"
        )
    lines.append("    -----------------+-----------------+-----------------+")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    main_cache = sel.load_npz(sel.MAIN_CACHE)
    extra_cache = sel.load_npz(sel.EXTRA_CACHE)
    main_rows = len(main_cache["episode"])
    pooled_valid = np.concatenate(
        (main_cache["valid"].astype(bool), extra_cache["valid"].astype(bool))
    )
    pooled_lag = np.concatenate(
        (main_cache["lag_distance"], extra_cache["lag_distance"]), axis=0
    )
    main_valid = pooled_valid[:main_rows]

    labels = sel.aligned_labels(main_cache, sel.MAIN_LABELS, "development_main")
    risk = labels["original_failure"].to_numpy(bool)
    task = labels["task"].to_numpy(str)
    suite = labels["suite"].to_numpy(str)
    episode = labels["episode"].to_numpy(int)
    prior_matrix = sel.prior_lookup(labels, main_valid)

    with np.load(V7_ALARMS, allow_pickle=False) as archive:
        freeze_first = np.asarray(archive[V7_KEY]).astype(np.int16)
    if freeze_first.shape != (main_rows,):
        raise ValueError(
            f"v7 {V7_KEY} has shape {freeze_first.shape}, expected ({main_rows},)"
        )

    selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    if selection.get("feasible") is not False:
        raise ValueError(
            "selection.json no longer records feasible: false; this script is "
            "written for the infeasible-selection regime and must be revisited"
        )

    candidates = pd.read_csv(CANDIDATE_CSV)
    points = diagnostic_points(candidates)

    print(
        f"development_main: {main_rows:,} episodes, risk={int(risk.sum()):,}, "
        f"timely={int((~risk).sum()):,}; pooled reference={len(pooled_valid):,}",
        flush=True,
    )
    print(
        f"v7 {V7_KEY}: {int((freeze_first >= 0).sum()):,} alarmed episodes "
        f"({int(((freeze_first >= 0) & risk).sum()):,} risk / "
        f"{int(((freeze_first >= 0) & ~risk).sum()):,} timely)",
        flush=True,
    )
    print(
        "\nselection.json says feasible=false (Task 7: 0/840 global candidates met "
        "the pre-registered rule).  Every number below is a DIAGNOSTIC over 10 "
        "stated points, not an operating point.\n",
        flush=True,
    )

    geometry = Geometry(pooled_lag, pooled_valid)
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    median_split: dict[str, Any] = {}

    for point in points:
        window = point["window"]
        group = point["layer_group"]
        confirmations = point["confirmations"]
        threshold = float(point["ratio_threshold"])

        grouped_length, grouped_ratio = geometry.group(window, group)
        _, _, eps_length = geometry.window(window)
        if not math.isclose(eps_length, float(point["eps_length"]), rel_tol=1e-9):
            raise ValueError(
                f"recomputed eps_length {eps_length} disagrees with the candidate table"
            )
        persistent, minima = geometry.persistent(window, group, confirmations)
        recomputed = progress_ratio.quantile_lower(minima, float(point["quantile"]))
        if not math.isclose(recomputed, threshold, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"recomputed threshold {recomputed} disagrees with the candidate "
                f"table value {threshold}"
            )

        first = progress_ratio.first_below(persistent[:main_rows], threshold, main_valid)
        alarm = first >= 0
        if int((alarm & risk).sum()) != point["tp"] or int((alarm & ~risk).sum()) != point["fp"]:
            raise ValueError("recomputed alarms disagree with the candidate table")

        main_length = grouped_length[:main_rows]
        main_ratio = grouped_ratio[:main_rows]
        usable = main_valid & np.isfinite(main_length) & np.isfinite(main_ratio)
        mobility_median = float(np.median(main_length[usable]))
        pooled_usable = pooled_valid & np.isfinite(grouped_length) & np.isfinite(grouped_ratio)
        pooled_median = float(np.median(grouped_length[pooled_usable]))

        alarm_rows = np.flatnonzero(alarm)
        alarm_chunk = first[alarm_rows].astype(int)
        alarm_length = main_length[alarm_rows, alarm_chunk]
        alarm_ratio = main_ratio[alarm_rows, alarm_chunk]
        cells = assign_cells(alarm_length, alarm_ratio, mobility_median, threshold)
        alarm_risk = risk[alarm_rows]

        k1_count, k1_pass = k1_gate(cells, alarm_risk)
        k3_share, k3_pass = k3_gate(first, freeze_first, risk)
        counts = cross_tab(cells, alarm_risk)
        overlaps = overlap(first, freeze_first, risk)

        label = (
            f"W={window} group={group} K={confirmations} q={point['quantile']} "
            f"r*={threshold:.6f}"
        )
        for position, row in enumerate(alarm_rows):
            rows.append(
                {
                    "family": point["family"],
                    "window": window,
                    "layer_group": group,
                    "confirmations": confirmations,
                    "quantile": point["quantile"],
                    "ratio_threshold": threshold,
                    "eps_length": eps_length,
                    "mobility_median": mobility_median,
                    "row": int(row),
                    "task": task[row],
                    "suite": suite[row],
                    "episode": int(episode[row]),
                    "alarm_chunk": int(alarm_chunk[position]),
                    "path_length_L": float(alarm_length[position]),
                    "progress_ratio_R": float(alarm_ratio[position]),
                    "cell": str(cells[position]),
                    "risk": bool(alarm_risk[position]),
                    "survival_prior": float(prior_matrix[row, alarm_chunk[position]]),
                }
            )

        occupancy_at_threshold = occupancy(
            main_length, main_ratio, usable, risk, mobility_median, threshold
        )
        reports.append(
            {
                "family": point["family"],
                "label": label,
                "window": window,
                "layer_group": group,
                "confirmations": confirmations,
                "quantile": float(point["quantile"]),
                "ratio_threshold": threshold,
                "eps_length": eps_length,
                "mobility_definition": (
                    "window path length L at the alarm chunk, reduced over the same "
                    "layer group with the group median (progress_ratio.group_ratio)"
                ),
                "mobility_median": mobility_median,
                "mobility_median_source": (
                    "median of the group-median L over all valid development_main "
                    f"chunks at W={window}, group={group} "
                    f"({int(usable.sum()):,} chunks)"
                ),
                "mobility_median_pooled_reference": pooled_median,
                "candidate_table": {
                    key: point[key]
                    for key in (
                        "tp",
                        "fp",
                        "recall",
                        "precision",
                        "timely_fpr",
                        "low_prior_tp",
                        "low_prior_fp",
                        "low_prior_precision",
                    )
                },
                "alarmed_episodes": int(alarm.sum()),
                "cell_counts": counts,
                "k1": {
                    "metric": "risk episodes whose alarm lands in high_mob_low_R",
                    "count": k1_count,
                    "minimum": K1_MIN_RISK_EPISODES,
                    "verdict": bool(k1_pass),
                    "share_of_risk_alarms": (
                        float(k1_count / int((alarm & risk).sum()))
                        if int((alarm & risk).sum())
                        else float("nan")
                    ),
                },
                "k3": {
                    "metric": "share of risk detections that v7 main_freeze misses",
                    "share": k3_share,
                    "minimum": K3_MIN_NOVEL_SHARE,
                    "verdict": bool(k3_pass),
                },
                "overlap_with_v7": overlaps,
                "corpus_occupancy_at_r_star": occupancy_at_threshold,
                "diagnostic_only": True,
            }
        )

        print(f"[{point['family']}] {label}", flush=True)
        print(
            f"    candidate table: tp={point['tp']} fp={point['fp']} "
            f"timely_fpr={point['timely_fpr']:.5f} low_prior_tp={point['low_prior_tp']}",
            flush=True,
        )
        print(format_table(counts, mobility_median), flush=True)
        risk_alarms = int((alarm & risk).sum())
        print(
            f"    K1: high_mob_low_R risk episodes = {k1_count} "
            f"(of {risk_alarms} risk alarms) vs minimum {K1_MIN_RISK_EPISODES} -> "
            f"{'PASS' if k1_pass else 'FAIL'}",
            flush=True,
        )
        print(
            f"    K3: novel share = {k3_share:.4f} "
            f"({overlaps['risk']['ratio_only_novel']}/{risk_alarms} risk detections "
            f"missed by v7 {V7_KEY}) vs minimum {K3_MIN_NOVEL_SHARE} -> "
            f"{'PASS' if k3_pass else 'FAIL'}",
            flush=True,
        )
        risk_occ = occupancy_at_threshold["risk"]["fraction"]
        timely_occ = occupancy_at_threshold["timely"]["fraction"]
        print(
            "    corpus chunks at this r*: high_mob_low_R "
            f"risk={risk_occ['high_mob_low_R']:.5f} timely={timely_occ['high_mob_low_R']:.5f}",
            flush=True,
        )
        print(flush=True)

        # 与阈值无关的对照：两轴都在开发集中位数处切分。
        key = f"W{window}_{group}"
        if key not in median_split:
            ratio_median = float(np.median(main_ratio[usable]))
            median_split[key] = {
                "window": window,
                "layer_group": group,
                "split": "both axes at their development_main medians (threshold-free)",
                "mobility_median": mobility_median,
                "ratio_median": ratio_median,
                "occupancy": occupancy(
                    main_length, main_ratio, usable, risk, mobility_median, ratio_median
                ),
            }

    cells_frame = pd.DataFrame(rows)
    cells_path = args.output / "cells.csv"
    cells_frame.to_csv(cells_path, index=False)

    k1_counts = [report["k1"]["count"] for report in reports]
    k3_shares = [report["k3"]["share"] for report in reports]
    summary = {
        "points_evaluated": len(reports),
        "k1_pass_count": int(sum(report["k1"]["verdict"] for report in reports)),
        "k1_count_min": int(min(k1_counts)),
        "k1_count_max": int(max(k1_counts)),
        "k1_verdict_any_point": bool(any(report["k1"]["verdict"] for report in reports)),
        "k3_pass_count": int(sum(report["k3"]["verdict"] for report in reports)),
        "k3_share_min": float(min(k3_shares)),
        "k3_share_max": float(max(k3_shares)),
        "k3_verdict_any_point": bool(any(report["k3"]["verdict"] for report in reports)),
    }

    gates = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "why_diagnostic_only": (
            "Task 7 found 0/840 global candidates satisfying the pre-registered rule "
            f"(timely_fpr <= {sel.FPR_CAP} and low_prior_tp >= {sel.MIN_LOW_PRIOR_TP}); "
            "results/operating_point/selection.json records feasible: false and no "
            "global_profile.npz exists.  Nothing here selects an operating point."
        ),
        "selection_state": {
            "source": str(SELECTION_JSON.relative_to(BUNDLE)),
            "feasible": selection["feasible"],
            "selected": selection["selected"],
            "profile_written": selection["profile_written"],
            "best_global_low_prior_tp_under_fpr_cap": selection["k2_gate"][
                "best_global_low_prior_tp_under_fpr_cap"
            ],
        },
        "point_selection_rule": POINT_RULE,
        "point_sort_keys": list(POINT_SORT_KEYS),
        "point_sort_ascending": list(POINT_SORT_ASCENDING),
        "points_per_family": POINTS_PER_FAMILY,
        "gate_definitions": {
            "K1_MIN_RISK_EPISODES": K1_MIN_RISK_EPISODES,
            "K3_MIN_NOVEL_SHARE": K3_MIN_NOVEL_SHARE,
            "alarms_are_low_R_by_construction": (
                "the alarm rule is persistent_low(R, K) < r* and persistent_low is a "
                "rolling max, so R at the alarm chunk is always < r*; the two high-R "
                "cells are therefore empty at alarm time by definition, and K1 is "
                "really asking how the low-R alarms split across mobility"
            ),
        },
        "corpus": {
            "development_main_episodes": main_rows,
            "risk_episodes": int(risk.sum()),
            "timely_episodes": int((~risk).sum()),
            "valid_chunks": int(main_valid.sum()),
            "pooled_reference_episodes": int(len(pooled_valid)),
            "v7_alarms": str(V7_ALARMS),
            "v7_key": V7_KEY,
            "v7_alarmed_episodes": int((freeze_first >= 0).sum()),
            "v7_risk_alarms": int(((freeze_first >= 0) & risk).sum()),
            "v7_timely_alarms": int(((freeze_first >= 0) & ~risk).sum()),
        },
        "points": reports,
        "corpus_occupancy_median_split": median_split,
        "summary": summary,
        "seconds": round(time.perf_counter() - started, 1),
        "artifacts": {
            "cells_csv_sha256": sel.sha256(cells_path),
            "script_sha256": sel.sha256(Path(__file__)),
            "candidate_csv_sha256": sel.sha256(CANDIDATE_CSV),
            "v7_alarms_sha256": sel.sha256(V7_ALARMS),
        },
    }
    gates_path = args.output / "gates.json"
    gates_path.write_text(
        json.dumps(sel.plain(gates), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=" * 100, flush=True)
    print("threshold-free control: both axes split at their development medians", flush=True)
    for key, block in median_split.items():
        risk_block = block["occupancy"]["risk"]
        timely_block = block["occupancy"]["timely"]
        print(
            f"  {key}: L_median={block['mobility_median']:.6f} "
            f"R_median={block['ratio_median']:.6f}",
            flush=True,
        )
        for cell in CELLS:
            print(
                f"      {cell:>16s}  risk {risk_block['fraction'][cell]:.4f} "
                f"({risk_block['counts'][cell]:>7,} chunks, "
                f"{risk_block['episodes_with_cell'][cell]:>4,}/{risk_block['episodes']:,} episodes)"
                f"   timely {timely_block['fraction'][cell]:.4f} "
                f"({timely_block['counts'][cell]:>7,} chunks, "
                f"{timely_block['episodes_with_cell'][cell]:>5,}/{timely_block['episodes']:,} episodes)",
                flush=True,
            )
    print("=" * 100, flush=True)
    print(
        f"K1 across {len(reports)} diagnostic points: "
        f"count in [{summary['k1_count_min']}, {summary['k1_count_max']}], "
        f"{summary['k1_pass_count']}/{len(reports)} reach {K1_MIN_RISK_EPISODES}",
        flush=True,
    )
    print(
        f"K3 across {len(reports)} diagnostic points: "
        f"share in [{summary['k3_share_min']:.4f}, {summary['k3_share_max']:.4f}], "
        f"{summary['k3_pass_count']}/{len(reports)} reach {K3_MIN_NOVEL_SHARE}",
        flush=True,
    )
    print(f"wrote {cells_path} ({len(cells_frame):,} rows) and {gates_path}", flush=True)
    print(f"elapsed {gates['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
