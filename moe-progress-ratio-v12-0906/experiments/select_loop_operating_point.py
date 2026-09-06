#!/usr/bin/env python3
"""修订一：合取判据的开发集选点，目标是 loop（兜圈），不是泛化的 risk。

第一轮的规则是单一阈值 ``R(q, W) < r*``，0/840 可行，FPR 上限内低先验产量封顶
在 17（`results/operating_point/selection.json`，`feasible: false`，不得改写）。
诊断（`experiments/diagnose_phenotype_timing.py`）显示量没错、规则错了：按存活先
验分区后，freeze 格（低 mobility ∧ 低 R）是晚期信号 1.55x/5.35x，兜圈格（高
mobility ∧ 低 R）是早期信号 5.60x/1.93x。单一 `R` 阈值把两格相加，晚期格以数量
压倒早期格。mobility 轴携带的是时间信息，不是冗余信息。

修订后的判据：

    stuck(q)  <=>  L(q, W) >= ell*  AND  R(q, W) < r*，连续确认 K 次

两个常量都是无标签 order statistic，但地位不同：

* ``ell*`` 是 pooled reference `L` 在该 (W, 层组) 上的**中位数**，**预注册、不扫描**。
  它是机制声明（"只覆盖高 mobility 那一格"），不是可调参数；让它可调会把这次修订
  变成二次搜索。freeze 格不再由本判据覆盖，它归 v7。
* ``r*`` 的校准与第一轮逐字相同：``quantile_lower(row_min(persistent_low(R, K)), q)``，
  取自 16,000 条 pooled reference 轨迹，签名里没有 outcome。

目标同时改为 loop。`analysis_trap_taxonomy/REPORT.zh.md` 确认三类失败，低先验区间
内兜圈格对三者的富集倍数为 loop 5.12–10.86x、static 0.27–1.01x、纯 phantom
0.07–0.34x，因此本方法只能命名为 loop 早期检测器。**覆盖上限必须与任何正面结果
同时报告：loop ∪ static 只占开发集失败的 73.9%。**

本脚本只打开 development cohort；``external_8b`` 从头到尾没有被引用。

Run:  python experiments/select_loop_operating_point.py
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio  # noqa: E402
import select_operating_point_v12 as base  # noqa: E402
from progress_monitor import GlobalProgressProfile  # noqa: E402


DEFAULT_OUTPUT = BUNDLE / "results/loop_operating_point"
TAXONOMY = WORKSPACE / "analysis_trap_taxonomy/results"
BELIEF_EPISODES = TAXONOMY / "belief_mismatch_episodes.csv.gz"
EVENTS = TAXONOMY / "events.csv.gz"
V4_MOBILITY = WORKSPACE / "moe-v4-0904/results/layerwise_mobility/main_reference.npz"

DEVELOPMENT_RUN = "seed1000_1007"  # external_8b 对应 seed1008_1015，本脚本不读
JOIN_KEY = ["task", "init_state_id", "flow_noise_seed"]

CONFIRMATION_GRID = base.CONFIRMATION_GRID
QUANTILE_GRID = base.QUANTILE_GRID
EPS_LENGTH_QUANTILE = base.EPS_LENGTH_QUANTILE
LOW_PRIOR_MAX = base.LOW_PRIOR_MAX
REFERENCE_EPISODES = base.REFERENCE_EPISODES

# 预注册常量，第一轮之后未改动，也不因结果放宽。
FPR_CAP = 0.005
MIN_LOW_PRIOR_LOOP_TP = 20
# §R4：loop ∪ static 覆盖开发集失败的 73.9%，其余在本 2x2 上结构性不可达。
COVERAGE_CEILING_DEVELOPMENT = 0.739

SCHEMA = "himoe.progress_ratio_v12.loop_selection.v1"
TARGETS = ("loop", "static", "phantom", "pure_phantom")
CANDIDATE_COLUMNS = (
    "window",
    "layer_group",
    "confirmations",
    "quantile",
    "ell_star",
    "ratio_threshold",
    "eps_length",
    "alarm_episodes",
    "tp",
    "fp",
    "recall",
    "precision",
    "timely_fpr",
    "low_prior_alarms",
    "low_prior_tp",
    "low_prior_fp",
    "low_prior_precision",
    "loop_tp",
    "loop_recall",
    "low_prior_loop_tp",
    "low_prior_loop_precision",
    "low_prior_loop_lift",
    "low_prior_static_tp",
    "low_prior_static_precision",
    "low_prior_static_lift",
    "low_prior_phantom_tp",
    "low_prior_phantom_lift",
    "low_prior_pure_phantom_tp",
    "low_prior_pure_phantom_lift",
    # 第一轮规则（只有 R < r*）在同一格上的对照，用来分离"合取带来的增量"
    # 与"换目标带来的增量"。同格 r* 与第一轮逐位相同。
    "ratio_only_timely_fpr",
    "ratio_only_low_prior_tp",
    "ratio_only_low_prior_loop_tp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# 预注册的纯决策函数：先被单测钉死，再被扫描调用
# --------------------------------------------------------------------------- #


def choose(candidates: pd.DataFrame) -> pd.Series | None:
    """预注册选择规则（A1）。不满足约束返回 None，不放宽重试。"""
    feasible = candidates[
        (candidates["timely_fpr"] <= FPR_CAP)
        & (candidates["low_prior_loop_tp"] >= MIN_LOW_PRIOR_LOOP_TP)
    ]
    if feasible.empty:
        return None
    ordered = feasible.sort_values(
        ["low_prior_loop_tp", "low_prior_loop_precision", "window"],
        ascending=[False, False, True],
    )
    return ordered.iloc[0]


def a2_gate(loop_lift: float, static_lift: float, phantom_lift: float) -> bool:
    """A2：必须是 loop 特异，否则只是又一个通用晚期量。"""
    return loop_lift > static_lift and loop_lift > phantom_lift


def persistent_high(values: np.ndarray, confirmations: int) -> np.ndarray:
    """滚动最小值，``progress_ratio.persistent_low`` 的对偶。

    高于阈值即代表窗口内每一个 query 都高于阈值，于是合取判据的"连续 K 次"
    可以由两个滚动统计量各自承担，报警函数本身不需要再看历史。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"persistent_high expects [episode, query], got {values.shape}")
    if confirmations < 1:
        raise ValueError("confirmations must be positive")
    if confirmations == 1:
        return values.copy()
    output = np.full_like(values, np.nan)
    for query in range(confirmations - 1, values.shape[1]):
        window = values[:, query - confirmations + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = window[good].min(axis=1)
    return output


def length_threshold(grouped_length: np.ndarray, valid: np.ndarray) -> float:
    """``ell*``：pooled reference `L` 的中位数。预注册，不扫描，不读 outcome。"""
    values = np.asarray(grouped_length, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("path length and validity masks do not align")
    usable = values[valid & np.isfinite(values)]
    if usable.size == 0:
        raise ValueError("the pooled reference contains no finite path length")
    return float(np.median(usable))


def first_conjunctive_alarm(
    persistent_ratio: np.ndarray,
    persistent_length: np.ndarray,
    ratio_threshold: float,
    length_floor: float,
    valid: np.ndarray,
) -> np.ndarray:
    """``L >= ell* AND R < r*`` 的首次成立位置；从未成立返回 -1。

    两个输入都应当已经是 K 次确认的滚动统计量（``persistent_low`` 与
    ``persistent_high``），因此这里只做一次逐点合取。
    """
    persistent_length = np.asarray(persistent_length, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if persistent_length.shape != valid.shape:
        raise ValueError("path length and validity masks do not align")
    with np.errstate(invalid="ignore"):
        eligible = (
            valid
            & np.isfinite(persistent_length)
            & (persistent_length >= length_floor)
        )
    return progress_ratio.first_below(persistent_ratio, ratio_threshold, eligible)


def lift(observed: float, expected: float) -> float:
    """提升倍数 = 实际检出数 / 按匹配存活先验的期望检出数。"""
    return base.ratio_or_nan(observed, expected)


# --------------------------------------------------------------------------- #
# 标签：risk 之外还要 loop / static / phantom
# --------------------------------------------------------------------------- #


def flow_noise_seed(cache: dict[str, np.ndarray]) -> np.ndarray:
    """progress cache 不带 flow_noise_seed，从行对齐的 v4 缓存里取，并先核对。"""
    mobility = base.load_npz(V4_MOBILITY)
    if len(mobility["episode"]) != len(cache["episode"]):
        raise ValueError("the v4 mobility cache has a different row count")
    if not np.array_equal(
        base.task_of(mobility), base.task_of(cache)
    ) or not np.array_equal(mobility["episode"], cache["episode"]):
        raise ValueError("the v4 mobility cache is not row-aligned with the progress cache")
    if not np.array_equal(mobility["init_state_id"], cache["init_state_id"]):
        raise ValueError("init_state_id disagrees between the two caches")
    return mobility["flow_noise_seed"].astype(int)


def failure_modes(labels: pd.DataFrame, cache: dict[str, np.ndarray]) -> pd.DataFrame:
    """按 (task, init_state_id, flow_noise_seed) 接上 trap taxonomy 的三类失败。"""
    keyed = labels.assign(
        init_state_id=cache["init_state_id"].astype(int),
        flow_noise_seed=flow_noise_seed(cache),
    )
    if keyed.duplicated(JOIN_KEY).any():
        raise ValueError("the cohort key is not unique")

    episodes = pd.read_csv(BELIEF_EPISODES)
    events = pd.read_csv(EVENTS)
    events["task"] = events["task_key"]  # events 里 suite 与 task 是分开存的
    taxonomy = episodes.merge(
        events[["run", "task", "episode", "loop_onset", "static_onset"]],
        on=["run", "task", "episode"],
        validate="one_to_one",
    )
    taxonomy = taxonomy[taxonomy["run"] == DEVELOPMENT_RUN]
    columns = JOIN_KEY + ["success", "belief_mismatch", "loop_onset", "static_onset"]
    merged = keyed.merge(
        taxonomy[columns], on=JOIN_KEY, how="left", validate="one_to_one"
    )
    if merged["success"].isna().any():
        raise ValueError("the trap taxonomy does not cover the development cohort")

    fail = ~merged["success"].to_numpy(bool)
    risk = merged["original_failure"].to_numpy(bool)
    if not np.array_equal(fail, risk):
        raise ValueError("taxonomy failure and original_failure disagree row-for-row")
    merged["loop"] = (merged["loop_onset"].to_numpy(int) >= 0) & fail
    merged["static"] = (merged["static_onset"].to_numpy(int) >= 0) & fail
    merged["phantom"] = merged["belief_mismatch"].to_numpy(bool) & fail
    merged["pure_phantom"] = (
        merged["phantom"] & ~merged["loop"] & ~merged["static"]
    )
    return merged


def target_priors(labels: pd.DataFrame, valid: np.ndarray) -> dict[str, np.ndarray]:
    """每一类目标的 [E, Q] 匹配存活先验 P(target | 仍在运行于 chunk q)。

    ``hazard_common.survival_prior`` 读的是 ``original_failure`` 列，所以把该列
    替换为目标指示量即可复用，语义仍是"按 suite、按 chunk 的存活基线"。
    """
    return {
        target: base.prior_lookup(
            labels.assign(original_failure=labels[target].to_numpy(bool)), valid
        )
        for target in TARGETS
    }


# --------------------------------------------------------------------------- #
# 打分
# --------------------------------------------------------------------------- #


def score(
    first: np.ndarray,
    labels: pd.DataFrame,
    prior_matrix: np.ndarray,
    priors: dict[str, np.ndarray],
) -> dict[str, Any]:
    """开发集上的结果指标。这是脚本里唯一读 outcome 的地方。"""
    risk = labels["original_failure"].to_numpy(bool)
    alarm = first >= 0
    rows = np.flatnonzero(alarm)
    chunks = first[rows]
    alarm_prior = prior_matrix[rows, chunks]
    if not np.isfinite(alarm_prior).all():
        raise ValueError("an alarm landed on a chunk with no survival prior")

    low = alarm_prior < LOW_PRIOR_MAX
    low_rows, low_chunks = rows[low], chunks[low]
    low_risk = risk[low_rows]
    tp = int((alarm & risk).sum())
    fp = int((alarm & ~risk).sum())
    record: dict[str, Any] = {
        "alarm_episodes": int(alarm.sum()),
        "tp": tp,
        "fp": fp,
        "recall": base.ratio_or_nan(tp, int(risk.sum())),
        "precision": base.ratio_or_nan(tp, tp + fp),
        "timely_fpr": base.ratio_or_nan(fp, int((~risk).sum())),
        "low_prior_alarms": int(len(low_rows)),
        "low_prior_tp": int(low_risk.sum()),
        "low_prior_fp": int((~low_risk).sum()),
        "low_prior_precision": base.ratio_or_nan(int(low_risk.sum()), len(low_rows)),
    }
    for target in TARGETS:
        flag = labels[target].to_numpy(bool)
        observed = int(flag[low_rows].sum())
        expected = float(priors[target][low_rows, low_chunks].sum())
        record[f"{target}_tp"] = int((alarm & flag).sum())
        record[f"{target}_recall"] = base.ratio_or_nan(
            int((alarm & flag).sum()), int(flag.sum())
        )
        record[f"low_prior_{target}_tp"] = observed
        record[f"low_prior_{target}_precision"] = base.ratio_or_nan(
            observed, len(low_rows)
        )
        record[f"low_prior_{target}_lift"] = lift(observed, expected)
        record[f"low_prior_{target}_expected"] = expected
    return record


def control_score(
    first: np.ndarray, labels: pd.DataFrame, prior_matrix: np.ndarray
) -> dict[str, Any]:
    """第一轮规则在同一格上的三个数，用作 mobility 轴增量的对照。"""
    risk = labels["original_failure"].to_numpy(bool)
    loop = labels["loop"].to_numpy(bool)
    rows = np.flatnonzero(first >= 0)
    low_rows = rows[prior_matrix[rows, first[rows]] < LOW_PRIOR_MAX]
    return {
        "ratio_only_timely_fpr": base.ratio_or_nan(
            int(((first >= 0) & ~risk).sum()), int((~risk).sum())
        ),
        "ratio_only_low_prior_tp": int(risk[low_rows].sum()),
        "ratio_only_low_prior_loop_tp": int(loop[low_rows].sum()),
    }


def sweep(
    pooled_lag_distance: np.ndarray,
    pooled_valid: np.ndarray,
    main_rows: int,
    labels: pd.DataFrame,
    prior_matrix: np.ndarray,
    priors: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    adjacent = pooled_lag_distance[:, :, 0, :]
    main_valid = pooled_valid[:main_rows]
    rows: list[dict[str, Any]] = []
    eps_by_window: dict[int, float] = {}

    for window in progress_ratio.W_GRID:
        started = time.perf_counter()
        length = progress_ratio.path_length(adjacent, window)
        eps_length = float(
            np.quantile(length[np.isfinite(length)], EPS_LENGTH_QUANTILE)
        )
        eps_by_window[window] = eps_length
        displacement = pooled_lag_distance[:, :, progress_ratio.LAGS.index(window), :]
        elementwise = progress_ratio.progress_ratio(displacement, length, eps_length)

        for group in progress_ratio.LAYER_GROUPS:
            grouped_ratio = progress_ratio.group_ratio(elementwise, group)
            grouped_length = progress_ratio.group_ratio(length, group)
            # ell* 只依赖 (W, 层组)，是 pooled reference 的中位数，不进入网格。
            ell_star = length_threshold(grouped_length, pooled_valid)

            for confirmations in CONFIRMATION_GRID:
                persistent_ratio = progress_ratio.persistent_low(
                    grouped_ratio, confirmations
                )
                persistent_ratio = np.where(pooled_valid, persistent_ratio, np.nan)
                # r* 的校准与第一轮逐字相同：只看 R，不看 L，不看 outcome。
                minima = progress_ratio.row_min(persistent_ratio)
                persistent_length = persistent_high(grouped_length, confirmations)
                main_ratio = persistent_ratio[:main_rows]
                main_length = persistent_length[:main_rows]

                for quantile in QUANTILE_GRID:
                    ratio_threshold = progress_ratio.quantile_lower(minima, quantile)
                    first = first_conjunctive_alarm(
                        main_ratio, main_length, ratio_threshold, ell_star, main_valid
                    )
                    control = progress_ratio.first_below(
                        main_ratio, ratio_threshold, main_valid
                    )
                    rows.append(
                        {
                            "window": window,
                            "layer_group": group,
                            "confirmations": confirmations,
                            "quantile": quantile,
                            "ell_star": ell_star,
                            "ratio_threshold": ratio_threshold,
                            "eps_length": eps_length,
                            **score(first, labels, prior_matrix, priors),
                            **control_score(control, labels, prior_matrix),
                        }
                    )
        print(
            f"[W={window:2d}] eps_length={eps_length:.6f} rows={len(rows)}/840 "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    return rows, eps_by_window


# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    main_cache = base.load_npz(base.MAIN_CACHE)
    extra_cache = base.load_npz(base.EXTRA_CACHE)
    main_rows = len(main_cache["episode"])
    extra_rows = len(extra_cache["episode"])
    if main_rows + extra_rows != REFERENCE_EPISODES:
        raise ValueError(
            f"expected {REFERENCE_EPISODES:,} pooled reference trajectories, "
            f"got {main_rows + extra_rows:,}"
        )
    for cohort, cache in (
        ("development_main", main_cache),
        ("development_extra", extra_cache),
    ):
        if not np.array_equal(
            np.asarray(cache["lags"]).astype(int), np.asarray(progress_ratio.LAGS)
        ):
            raise ValueError(f"{cohort} cache lags disagree with the method module")

    pooled_valid = np.concatenate(
        (main_cache["valid"].astype(bool), extra_cache["valid"].astype(bool))
    )
    pooled_lag_distance = np.concatenate(
        (main_cache["lag_distance"], extra_cache["lag_distance"]), axis=0
    )
    main_valid = pooled_valid[:main_rows]

    # 唯一一次打开 outcome：risk 标签 + trap taxonomy 的三类失败模式。
    labels = failure_modes(
        base.aligned_labels(main_cache, base.MAIN_LABELS, "development_main"), main_cache
    )
    risk = labels["original_failure"].to_numpy(bool)
    prior_matrix = base.prior_lookup(labels, main_valid)
    priors = target_priors(labels, main_valid)

    counts = {
        target: int(labels[target].to_numpy(bool).sum()) for target in TARGETS
    }
    covered = int((labels["loop"] | labels["static"]).to_numpy(bool).sum())
    coverage = covered / int(risk.sum())
    print(
        f"reference={main_rows + extra_rows:,} trajectories; scored={main_rows:,}; "
        f"risk={int(risk.sum())}, loop={counts['loop']}, static={counts['static']}, "
        f"phantom={counts['phantom']} (pure {counts['pure_phantom']})",
        flush=True,
    )
    print(
        f"coverage ceiling (loop or static) = {covered}/{int(risk.sum())} = "
        f"{coverage:.3%}; the remaining {1 - coverage:.1%} is structurally "
        "unreachable on this 2x2",
        flush=True,
    )

    rows, eps_by_window = sweep(
        pooled_lag_distance, pooled_valid, main_rows, labels, prior_matrix, priors
    )
    candidates = pd.DataFrame(rows)
    extra_columns = [
        column for column in candidates.columns if column not in CANDIDATE_COLUMNS
    ]
    candidates = candidates[list(CANDIDATE_COLUMNS) + extra_columns].sort_values(
        ["window", "layer_group", "confirmations", "quantile"], kind="stable"
    )
    if len(candidates) != 840:
        raise ValueError(f"expected 840 candidates, got {len(candidates)}")
    candidate_path = args.output / "development_candidates.csv"
    candidates.to_csv(candidate_path, index=False)

    choice = choose(candidates)
    feasible_mask = (candidates["timely_fpr"] <= FPR_CAP) & (
        candidates["low_prior_loop_tp"] >= MIN_LOW_PRIOR_LOOP_TP
    )
    capped = candidates[candidates["timely_fpr"] <= FPR_CAP]

    def lift_block(row: pd.Series | None) -> dict[str, Any]:
        if row is None:
            return {
                "loop_lift": None,
                "static_lift": None,
                "phantom_lift": None,
                "phantom_all_lift": None,
                "verdict": False,
            }
        return {
            "loop_lift": float(row["low_prior_loop_lift"]),
            "static_lift": float(row["low_prior_static_lift"]),
            "phantom_lift": float(row["low_prior_pure_phantom_lift"]),
            "phantom_all_lift": float(row["low_prior_phantom_lift"]),
            "verdict": bool(
                a2_gate(
                    float(row["low_prior_loop_lift"]),
                    float(row["low_prior_static_lift"]),
                    float(row["low_prior_pure_phantom_lift"]),
                )
            ),
        }

    # A1 不可行时 A2 无处可判。按第一轮 K2 的处理方式标注 vacuous，并在
    # "FPR 上限内低先验 loop TP 最高"的那一格上另记一次诊断读数——它不是
    # 选中点，也不改变 A1 的否定结论。
    near_miss = None
    if choice is None and not capped.empty:
        near_miss = capped.sort_values(
            ["low_prior_loop_tp", "low_prior_loop_precision", "window"],
            ascending=[False, False, True],
        ).iloc[0]
    a2: dict[str, Any] = {
        "definition": (
            "低先验区间内对 loop 的提升倍数 > 对 static 与纯 phantom 的提升倍数"
        ),
        "phantom_variant": (
            "pure_phantom (phantom & ~loop & ~static)，与 §R4 富集表同口径；"
            "全部 phantom 的提升倍数一并记录，因为它与 loop 大量重叠"
        ),
        **lift_block(choice),
        "verdict_vacuous": choice is None,
        "diagnostic_on_best_candidate_under_fpr_cap": (
            {**lift_block(near_miss), "candidate": base.row_dict(near_miss)}
            if near_miss is not None
            else None
        ),
    }

    profile_path = args.output / "global_profile.npz"
    profile_written = False
    if choice is not None:
        np.savez_compressed(
            profile_path,
            schema=np.asarray(progress_ratio.SCHEMA),
            window=np.asarray(int(choice["window"]), dtype=np.int32),
            confirmations=np.asarray(int(choice["confirmations"]), dtype=np.int32),
            ratio_threshold=np.asarray(
                float(choice["ratio_threshold"]), dtype=np.float64
            ),
            eps_length=np.asarray(float(choice["eps_length"]), dtype=np.float64),
            layer_group=np.asarray(str(choice["layer_group"])),
            length_threshold=np.asarray(float(choice["ell_star"]), dtype=np.float64),
        )
        reloaded = GlobalProgressProfile.load(profile_path)
        if (
            reloaded.window != int(choice["window"])
            or reloaded.confirmations != int(choice["confirmations"])
            or not math.isclose(
                reloaded.ratio_threshold, float(choice["ratio_threshold"])
            )
            or not math.isclose(reloaded.eps_length, float(choice["eps_length"]))
            or reloaded.layer_group != str(choice["layer_group"])
        ):
            raise ValueError("the frozen profile does not round-trip")
        with np.load(profile_path, allow_pickle=False) as archive:
            forbidden = {"task_names", "task_index"} & set(archive.files)
            if forbidden:
                raise ValueError(f"the loop profile leaks task metadata: {forbidden}")
            if not math.isclose(
                float(archive["length_threshold"]), float(choice["ell_star"])
            ):
                raise ValueError("the frozen profile lost ell*")
        profile_written = True

    elapsed = time.perf_counter() - started
    summary = {
        "schema": SCHEMA,
        "rule": "stuck(q) <=> L(q,W) >= ell* AND R(q,W) < r*, K consecutive queries",
        "amendment": {
            "post_hoc": True,
            "supersedes": "results/operating_point/selection.json (round one, feasible: false)",
            "round_one_low_prior_tp_ceiling_under_fpr_cap": 17,
            "ell_star_definition": "median of pooled reference L at that (W, layer group)",
            "ell_star_swept": False,
            "r_star_definition": (
                "quantile_lower(row_min(persistent_low(R, K)), quantile) over the "
                "pooled 16,000-trajectory reference corpus; no outcome in the signature"
            ),
            "freeze_cell_covered": False,
        },
        "selection_rule": {
            "timely_fpr_at_most": FPR_CAP,
            "low_prior_loop_tp_at_least": MIN_LOW_PRIOR_LOOP_TP,
            "low_prior_definition": f"matched survival prior for risk < {LOW_PRIOR_MAX}",
            "ranking": [
                "low_prior_loop_tp desc",
                "low_prior_loop_precision desc",
                "window asc",
            ],
            "pre_registered": True,
            "relaxed_after_seeing_results": False,
        },
        "grid": {
            "window": list(progress_ratio.W_GRID),
            "layer_group": list(progress_ratio.LAYER_GROUPS),
            "confirmations": list(CONFIRMATION_GRID),
            "quantile": list(QUANTILE_GRID),
            "candidates": len(candidates),
            "ell_star_is_derived_not_swept": True,
        },
        "corpus": {
            "reference_trajectories": main_rows + extra_rows,
            "reference_cohorts": ["development_main", "development_extra"],
            "development_main_episodes": main_rows,
            "development_extra_episodes": extra_rows,
            "scored_episodes": main_rows,
            "risk_episodes": int(risk.sum()),
            "timely_episodes": int((~risk).sum()),
            "loop_episodes": counts["loop"],
            "static_episodes": counts["static"],
            "phantom_episodes": counts["phantom"],
            "pure_phantom_episodes": counts["pure_phantom"],
            "external_data_loaded": False,
            "labels_used_for_threshold_calibration": [],
        },
        "coverage_ceiling": {
            "definition": "loop 或 static 的失败占全部开发集失败的比例",
            "development_declared": COVERAGE_CEILING_DEVELOPMENT,
            "development_measured": coverage,
            "development_covered_failures": covered,
            "development_failures": int(risk.sum()),
            "development_unreachable_fraction": 1.0 - coverage,
            "external_declared": 0.764,
            "phantom_blind_spot": (
                "phantom grasp（平稳、有方向地朝错误目标前进）落在原设计标注为"
                "健康的高 mobility ∧ 高 R 格，本判据结构性不可达"
            ),
            "note": "任何正面结果都必须与该上限同时报告，不得单独给 recall",
        },
        "eps_length_quantile": EPS_LENGTH_QUANTILE,
        "eps_length_by_window": base.plain(eps_by_window),
        "ell_star_by_window_and_group": base.plain(
            {
                f"W{int(row.window)}/{row.layer_group}": float(row.ell_star)
                for row in candidates.drop_duplicates(
                    ["window", "layer_group"]
                ).itertuples()
            }
        ),
        "feasible": choice is not None,
        "feasible_candidates": int(feasible_mask.sum()),
        "fpr_cap_only_candidates": int((candidates["timely_fpr"] <= FPR_CAP).sum()),
        "best_low_prior_loop_tp_under_fpr_cap": int(
            capped["low_prior_loop_tp"].max()
        )
        if not capped.empty
        else 0,
        "best_low_prior_loop_tp_unconstrained": int(
            candidates["low_prior_loop_tp"].max()
        ),
        "best_low_prior_tp_under_fpr_cap": int(capped["low_prior_tp"].max())
        if not capped.empty
        else 0,
        "ratio_only_control": {
            "definition": (
                "同一格、同一个 r*（与第一轮逐位相同）下去掉 L >= ell* 之后的结果，"
                "用来分离合取的增量与换目标的增量"
            ),
            "best_low_prior_loop_tp_under_fpr_cap": int(
                candidates[candidates["ratio_only_timely_fpr"] <= FPR_CAP][
                    "ratio_only_low_prior_loop_tp"
                ].max()
            ),
            "best_low_prior_tp_under_fpr_cap": int(
                candidates[candidates["ratio_only_timely_fpr"] <= FPR_CAP][
                    "ratio_only_low_prior_tp"
                ].max()
            ),
        },
        "a1_gate": {
            "definition": (
                f"low_prior_loop_tp >= {MIN_LOW_PRIOR_LOOP_TP} 且 "
                f"timely_fpr <= {FPR_CAP}"
            ),
            "verdict": choice is not None,
        },
        "a2_gate": a2,
        "selected": base.row_dict(choice),
        "profile_written": profile_written,
        "monitor_supports_conjunction": False,
        "monitor_note": (
            "method/progress_monitor.py 仍是单一 R 阈值的在线实现，尚未读 "
            "length_threshold；profile 已带该常量，在线端的合取实现属于后续任务"
        ),
        "seconds": round(elapsed, 1),
        "artifacts": {
            "candidate_csv_sha256": base.sha256(candidate_path),
            "script_sha256": base.sha256(Path(__file__)),
            "main_cache_sha256": base.sha256(base.MAIN_CACHE),
            "extra_cache_sha256": base.sha256(base.EXTRA_CACHE),
            **(
                {"global_profile_sha256": base.sha256(profile_path)}
                if profile_written
                else {}
            ),
        },
    }
    selection_path = args.output / "selection.json"
    selection_path.write_text(
        json.dumps(base.plain(summary), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    print(
        f"\nfeasible candidates: {summary['feasible_candidates']}/{len(candidates)} "
        f"(under the FPR cap alone: {summary['fpr_cap_only_candidates']})"
    )
    if choice is None:
        print("NO conjunctive operating point satisfies the pre-registered constraints.")
        print("feasible: false -- the full candidate table is kept, nothing relaxed.")
        print(
            f"  best low_prior_loop_tp under timely_fpr<={FPR_CAP}: "
            f"{summary['best_low_prior_loop_tp_under_fpr_cap']} "
            f"(rule needs >= {MIN_LOW_PRIOR_LOOP_TP}); ratio-only control at the "
            f"same r*: {summary['ratio_only_control']['best_low_prior_loop_tp_under_fpr_cap']}"
        )
        if near_miss is not None:
            print(
                "  diagnostic at that candidate (NOT a selection): "
                f"W={int(near_miss['window'])} group={near_miss['layer_group']} "
                f"K={int(near_miss['confirmations'])} q={near_miss['quantile']} "
                f"low_prior_tp={int(near_miss['low_prior_tp'])} "
                f"lifts loop={near_miss['low_prior_loop_lift']:.2f}x "
                f"static={near_miss['low_prior_static_lift']:.2f}x "
                f"pure_phantom={near_miss['low_prior_pure_phantom_lift']:.2f}x"
            )
    else:
        print(
            "selected: "
            f"W={int(choice['window'])} group={choice['layer_group']} "
            f"K={int(choice['confirmations'])} quantile={choice['quantile']} "
            f"ell*={choice['ell_star']:.6f} r*={choice['ratio_threshold']:.6f} "
            f"eps_length={choice['eps_length']:.6f}"
        )
        print(
            f"  low_prior_loop_tp={int(choice['low_prior_loop_tp'])} "
            f"low_prior_loop_precision={choice['low_prior_loop_precision']:.4f} "
            f"low_prior_alarms={int(choice['low_prior_alarms'])} "
            f"(round one ceiling on low_prior_tp was 17; here low_prior_tp="
            f"{int(choice['low_prior_tp'])})"
        )
        print(
            f"  loop_tp={int(choice['loop_tp'])}/{counts['loop']} "
            f"tp={int(choice['tp'])} fp={int(choice['fp'])} "
            f"recall={choice['recall']:.4f} precision={choice['precision']:.4f} "
            f"timely_fpr={choice['timely_fpr']:.5f}"
        )
        print(
            f"  low-prior lifts: loop={a2['loop_lift']:.2f}x "
            f"static={a2['static_lift']:.2f}x "
            f"pure_phantom={a2['phantom_lift']:.2f}x "
            f"(all phantom {a2['phantom_all_lift']:.2f}x)"
        )
    print(f"A1 {'PASS' if summary['a1_gate']['verdict'] else 'FAIL'}   ", end="")
    print(f"A2 {'PASS' if a2['verdict'] else 'FAIL'}")
    print(
        f"coverage ceiling: loop or static = {coverage:.1%} of development failures; "
        f"recall must never be reported without it"
    )
    print(f"elapsed {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
