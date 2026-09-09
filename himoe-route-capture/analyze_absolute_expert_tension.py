"""Audit gate-preserving, unnormalised selected-expert tension.

The input archive is produced by ``analyze_unweighted_expert_norm.py``.  It
contains the captured top-k gate weights and an offline reconstruction of each
selected expert output from the stored fp16 hidden state.  This analysis keeps
the gate weights and removes only the scale normalisation used by D and C.

For one action token, let ``v_e`` be a selected expert output, ``alpha_e`` its
renormalised top-k gate weight, and ``r = sum_e alpha_e v_e``.  With RMS over
the hidden width,

    s1 = sum_e alpha_e RMS(v_e)
    s2 = sum_e alpha_e RMS(v_e)^2
    rho = RMS(r)
    D_abs = sqrt(max(s2 - rho^2, 0))
    C_abs = s1 - rho

The corresponding scale-free quantities are ``D = 1-rho^2/s2`` and
``C = 1-rho/s1``.  Thus ``D_abs = sqrt(s2 * D)`` and ``C_abs = s1 * C``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


N_DENOISE = 10


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=here / "analysis" / "unweighted-expert-norm" / "unweighted_expert_norms.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "analysis" / "absolute-expert-tension",
    )
    return parser.parse_args()


def compute_token_metrics(
    weights: np.ndarray,
    raw_rms: np.ndarray,
    routed_rms: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return gate-preserving metrics before averaging action tokens."""
    weights = np.asarray(weights, dtype=np.float64)
    raw_rms = np.asarray(raw_rms, dtype=np.float64)
    routed_rms = np.asarray(routed_rms, dtype=np.float64)
    if weights.shape != raw_rms.shape:
        raise ValueError(f"weight/raw shape mismatch: {weights.shape} != {raw_rms.shape}")
    if weights.shape[:-1] != routed_rms.shape:
        raise ValueError(
            f"routed shape mismatch: {routed_rms.shape} != {weights.shape[:-1]}"
        )
    if np.any(weights < 0) or np.any(raw_rms < 0) or np.any(routed_rms < 0):
        raise ValueError("weights and RMS values must be nonnegative")
    weight_sum = weights.sum(axis=-1)
    if not np.allclose(weight_sum, 1.0, atol=5e-4, rtol=0.0):
        raise ValueError("selected gate weights are not top-k normalised")

    s1 = np.sum(weights * raw_rms, axis=-1)
    s2 = np.sum(weights * np.square(raw_rms), axis=-1)
    d_energy = s2 - np.square(routed_rms)
    tolerance = 1e-7 * np.maximum(1.0, s2)
    if np.any(d_energy < -tolerance):
        raise ValueError("routed energy exceeds the weighted expert second moment")
    d_energy = np.maximum(d_energy, 0.0)
    tiny = np.finfo(np.float64).tiny
    return {
        "s1": s1,
        "s2": s2,
        "routed_rms": routed_rms,
        "d_abs": np.sqrt(d_energy),
        "c_abs": s1 - routed_rms,
        "d_normalized": 1.0 - np.square(routed_rms) / np.maximum(s2, tiny),
        "c_normalized": 1.0 - routed_rms / np.maximum(s1, tiny),
    }


def _linear_slope(value: np.ndarray) -> np.ndarray:
    time = np.arange(value.shape[-1], dtype=np.float64)
    centered = time - time.mean()
    return np.sum(value * centered, axis=-1) / np.sum(np.square(centered))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = rankdata(np.asarray(x, dtype=np.float64))
    y_rank = rankdata(np.asarray(y, dtype=np.float64))
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = np.sqrt(np.sum(np.square(x_rank)) * np.sum(np.square(y_rank)))
    if denominator == 0:
        return float("nan")
    return float(np.sum(x_rank * y_rank) / denominator)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def _trajectory(value: np.ndarray, task_names: list[str]) -> dict[str, Any]:
    result = {}
    for task_axis, task in enumerate(task_names):
        task_value = value[task_axis]
        change = task_value[..., -1] - task_value[..., 0]
        curve = np.median(task_value, axis=(0, 1))
        result[task] = {
            "median_by_round": curve.tolist(),
            "d0_median": float(curve[0]),
            "d9_median": float(curve[-1]),
            "median_d9_minus_d0": float(np.median(change)),
            "fraction_candidates_d9_below_d0": float(np.mean(change < 0)),
        }
    return result


def _candidate_dispersion(
    value: np.ndarray, task_names: list[str]
) -> dict[str, Any]:
    """Median K=32 population SD across the 16 fixed-state pools."""
    result = {}
    for task_axis, task in enumerate(task_names):
        pool_std = value[task_axis].std(axis=1, ddof=0)
        d0 = float(np.median(pool_std[:, 0]))
        d9 = float(np.median(pool_std[:, -1]))
        result[task] = {
            "median_pool_candidate_std_by_round": np.median(pool_std, axis=0).tolist(),
            "d0_median_pool_candidate_std": d0,
            "d9_median_pool_candidate_std": d9,
            "d9_over_d0": d9 / d0,
        }
    return result


def _within_pool_scale_correlation(
    value: np.ndarray,
    scale: np.ndarray,
    task_names: list[str],
) -> dict[str, Any]:
    result = {}
    for task_axis, task in enumerate(task_names):
        per_round = []
        for denoise in range(value.shape[-1]):
            correlations = [
                _spearman(
                    value[task_axis, pool, :, denoise],
                    scale[task_axis, pool, :, denoise],
                )
                for pool in range(value.shape[1])
            ]
            per_round.append(float(np.nanmean(correlations)))
        result[task] = {
            "mean_pool_spearman_by_round": per_round,
            "d0": per_round[0],
            "d9": per_round[-1],
        }
    return result


def _success_exploration(
    value: np.ndarray,
    success: np.ndarray,
    task_names: list[str],
) -> dict[str, Any]:
    features = {
        "d0": value[..., 0],
        "d9": value[..., -1],
        "linear_slope_d0_d9": _linear_slope(value),
    }
    output: dict[str, Any] = {
        "status": "post_hoc_effect_sizes_only_no_permutation_or_multiplicity_inference",
        "higher_score_predicts_success": True,
        "features": {},
    }
    for name, score in features.items():
        per_task = []
        informative = []
        for task_axis in range(len(task_names)):
            pool_auc = np.asarray(
                [
                    _auc(score[task_axis, pool], success[task_axis, pool])
                    for pool in range(score.shape[1])
                ]
            )
            valid = pool_auc[np.isfinite(pool_auc)]
            per_task.append(float(valid.mean()) if len(valid) else None)
            informative.append(int(len(valid)))
        macro_values = [value for value in per_task if value is not None]
        output["features"][name] = {
            "per_task_auc": dict(zip(task_names, per_task)),
            "informative_pools": dict(zip(task_names, informative)),
            "equal_task_macro_auc": float(np.mean(macro_values)),
        }
    return output


def analyze(source: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(source) as archive:
        required = {
            "task_names",
            "selected_expert_weight",
            "selected_expert_raw_rms",
            "weighted_expert_mass_by_token",
            "routed_rms_by_token",
            "success",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                "source archive lacks token-level arrays; rerun "
                f"analyze_unweighted_expert_norm.py (missing {missing})"
            )
        task_names = archive["task_names"].tolist()
        weights = np.asarray(archive["selected_expert_weight"], dtype=np.float64)
        raw_rms = np.asarray(archive["selected_expert_raw_rms"], dtype=np.float64)
        stored_s1 = np.asarray(
            archive["weighted_expert_mass_by_token"], dtype=np.float64
        )
        routed_rms = np.asarray(archive["routed_rms_by_token"], dtype=np.float64)
        success = np.asarray(archive["success"], dtype=np.int8)

    token = compute_token_metrics(weights, raw_rms, routed_rms)
    candidate = {name: value.mean(axis=-1) for name, value in token.items()}
    if candidate["d_abs"].shape[-1] != N_DENOISE:
        raise ValueError(f"expected {N_DENOISE} denoise rounds")

    metrics = ("d_abs", "c_abs", "d_normalized", "c_normalized")
    trajectories = {
        name: _trajectory(candidate[name], task_names) for name in metrics
    }
    dispersions = {
        name: _candidate_dispersion(candidate[name], task_names) for name in metrics
    }
    scale_correlations = {
        name: _within_pool_scale_correlation(
            candidate[name], candidate["s1"], task_names
        )
        for name in metrics
    }

    d_identity_error = np.max(
        np.abs(
            np.square(token["d_abs"])
            - token["s2"] * token["d_normalized"]
        )
    )
    c_identity_error = np.max(
        np.abs(
            token["c_abs"] - token["s1"] * token["c_normalized"]
        )
    )
    summary = {
        "experiment": "absolute_selected_expert_tension_cross_task_v1",
        "status": "descriptive_post_hoc_offline_reconstruction",
        "definition": {
            "gate": "captured top4 weights renormalized to sum to one",
            "s1": "sum_e alpha_e * RMS(E_e(h))",
            "s2": "sum_e alpha_e * RMS(E_e(h))^2",
            "rho": "RMS(sum_e alpha_e * E_e(h))",
            "D_abs": "sqrt(max(s2 - rho^2, 0))",
            "C_abs": "s1 - rho",
            "D_normalized": "1 - rho^2 / s2",
            "C_normalized": "1 - rho / s1",
            "identity_D": "D_abs = sqrt(s2 * D_normalized)",
            "identity_C": "C_abs = s1 * C_normalized",
        },
        "source": str(source.resolve()),
        "fixed_hb_layer": 5,
        "tasks": task_names,
        "pools_per_task": int(weights.shape[1]),
        "candidates_per_pool": int(weights.shape[2]),
        "denoise_rounds": int(weights.shape[3]),
        "action_tokens": int(weights.shape[4]),
        "selected_experts": int(weights.shape[5]),
        "aggregation": {
            "candidate_metric": "mean over 10 action tokens",
            "trajectory_level": "median over 16 pools x 32 candidates",
            "K32_dispersion": (
                "population SD over 32 candidates inside each fixed-state pool, "
                "then median over 16 pools"
            ),
            "scale_correlation": (
                "Spearman over 32 candidates inside each pool, then equal-pool mean"
            ),
        },
        "validation": {
            "max_abs_selected_weight_sum_minus_one": float(
                np.max(np.abs(weights.sum(axis=-1) - 1.0))
            ),
            "max_abs_recomputed_s1_minus_stored_s1": float(
                np.max(np.abs(token["s1"] - stored_s1))
            ),
            "max_abs_D_identity_error": float(d_identity_error),
            "max_abs_C_identity_error": float(c_identity_error),
        },
        "trajectory": trajectories,
        "K32_dispersion": dispersions,
        "within_pool_spearman_with_s1": scale_correlations,
        "success_exploratory": {
            "d_abs": _success_exploration(
                candidate["d_abs"], success, task_names
            ),
            "c_abs": _success_exploration(
                candidate["c_abs"], success, task_names
            ),
        },
        "limitations": [
            "Selected expert ids and gate weights are captured, but expert outputs are reconstructed offline from stored fp16 hidden states.",
            "D_abs and C_abs describe selected-expert output geometry, not persistent expert internal state or model confidence.",
            "Both absolute metrics retain output scale; they are not expert-identity-specific effects.",
            "Success AUCs are post-hoc descriptive effect sizes with no permutation p-value, multiplicity correction, or confirmation split.",
        ],
    }
    arrays = {
        "task_names": np.asarray(task_names),
        "s1": candidate["s1"].astype(np.float32),
        "s2": candidate["s2"].astype(np.float32),
        "routed_rms": candidate["routed_rms"].astype(np.float32),
        "D_abs": candidate["d_abs"].astype(np.float32),
        "C_abs": candidate["c_abs"].astype(np.float32),
        "D_normalized": candidate["d_normalized"].astype(np.float32),
        "C_normalized": candidate["c_normalized"].astype(np.float32),
        "success": success,
    }
    return summary, arrays


def _render_report(summary: dict[str, Any]) -> str:
    tasks = summary["tasks"]
    trajectory = summary["trajectory"]
    dispersion = summary["K32_dispersion"]
    correlation = summary["within_pool_spearman_with_s1"]
    lines = [
        "# 保留 gate 的未归一化专家张力",
        "",
        "本报告只回答一个问题：去掉 `D/C` 的尺度归一化、但保留真实 top-4 gate 权重后，",
        "selected experts 的输出几何如何变化。它不把输出范数称为专家内部状态或置信度。",
        "",
        "## 定义",
        "",
        "对同一个 action token，令 `v_e=E_e(h)`、`alpha_e` 为归一化后的 top-4 gate 权重、",
        "`r=sum_e alpha_e v_e`、`rho=RMS(r)`：",
        "",
        "- `s1=sum_e alpha_e RMS(v_e)`",
        "- `s2=sum_e alpha_e RMS(v_e)^2`",
        "- `D_abs=sqrt(max(s2-rho^2, 0))`",
        "- `C_abs=s1-rho`",
        "",
        "这才是“去掉归一化但保留 gate”的定义。与比例量的精确关系是",
        "`D_abs=sqrt(s2*D)`、`C_abs=s1*C`。下面先对 10 个 action token 取均值。",
        "",
        "## 绝对量轨迹",
        "",
        "| task | D_abs d0 | D_abs d9 | delta | down | C_abs d0 | C_abs d9 | delta | down |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        d = trajectory["d_abs"][task]
        c = trajectory["c_abs"][task]
        lines.append(
            "| %s | %.5f | %.5f | %+.5f | %.1f%% | %.5f | %.5f | %+.5f | %.1f%% |"
            % (
                task,
                d["d0_median"],
                d["d9_median"],
                d["median_d9_minus_d0"],
                100 * d["fraction_candidates_d9_below_d0"],
                c["d0_median"],
                c["d9_median"],
                c["median_d9_minus_d0"],
                100 * c["fraction_candidates_d9_below_d0"],
            )
        )
    lines += [
        "",
        "Goal 的绝对张力下降，Spatial 的绝对张力上升，Long 接近持平；因此 absolute",
        "量没有五任务同向的“越来越肯定”结论。normalized `D/C` 的中位数则五任务均上升，",
        "它表达的是张力占总 expert scale 的比例增加，并不与 absolute 量同义。",
        "",
        "## 同状态 K=32 离散度",
        "",
        "数值为每个状态内 32 个候选的 population SD，再取 16 个状态的中位数。ratio<1 表示候选间收缩。",
        "",
        "| task | D_abs SD d0 | d9 | ratio | C_abs SD d0 | d9 | ratio | D ratio | C ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        da = dispersion["d_abs"][task]
        ca = dispersion["c_abs"][task]
        dn = dispersion["d_normalized"][task]
        cn = dispersion["c_normalized"][task]
        lines.append(
            "| %s | %.6f | %.6f | %.3f | %.6f | %.6f | %.3f | %.3f | %.3f |"
            % (
                task,
                da["d0_median_pool_candidate_std"],
                da["d9_median_pool_candidate_std"],
                da["d9_over_d0"],
                ca["d0_median_pool_candidate_std"],
                ca["d9_median_pool_candidate_std"],
                ca["d9_over_d0"],
                dn["d9_over_d0"],
                cn["d9_over_d0"],
            )
        )
    lines += [
        "",
        "normalized `D/C` 的 K32 离散度五任务都收缩；absolute 量在 spatial-stove 反而扩张。",
        "这更像 hidden/denoise 驱动的候选几何收缩，不能单独归因于某个专家。",
        "",
        "## 尺度混杂",
        "",
        "下表是在每个同状态 K32 池内算 Spearman，再对 16 个池等权平均。",
        "",
        "| task | rho(D_abs,s1) d0 | d9 | rho(C_abs,s1) d0 | d9 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in tasks:
        d = correlation["d_abs"][task]
        c = correlation["c_abs"][task]
        lines.append(
            "| %s | %+.3f | %+.3f | %+.3f | %+.3f |"
            % (task, d["d0"], d["d9"], c["d0"], c["d9"])
        )
    lines += [
        "",
        "`D_abs` 与总 expert scale `s1` 的同池相关约 0.97，`C_abs` 也很高。",
        "这不是计算错误，而是 absolute 定义必然保留尺度；所以它适合诊断实际张力幅度，",
        "不适合作为独立的 expert-specific 或 confidence 证据。",
        "",
        "## Success 探索",
        "",
        "下表 AUC>0.5 表示分数越高越容易成功。goal-middle 没有成败混合池，因此不进入 macro。",
        "这些是事后 effect size，没有 permutation、multiple-testing 校正或确认集。",
        "",
        "| score | feature | macro AUC | goal-top | long | ramekin | stove |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for metric in ("d_abs", "c_abs"):
        result = summary["success_exploratory"][metric]
        for feature in ("d0", "d9", "linear_slope_d0_d9"):
            row = result["features"][feature]
            per = row["per_task_auc"]
            lines.append(
                "| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f |"
                % (
                    metric,
                    feature,
                    row["equal_task_macro_auc"],
                    per["goal-top"],
                    per["long-t08"],
                    per["spatial-ramekin"],
                    per["spatial-stove"],
                )
            )
    lines += [
        "",
        "d0 的约 0.59 macro AUC 与 raw amplitude 的既有结果几乎相同，且随后轮次方向改变。",
        "结合上面的尺度相关，它不是新的成功路由结论。",
        "",
        "## 边界",
        "",
        "- expert IDs 和 gate 权重来自捕获；expert outputs 来自 fp16 hidden 的离线重建。",
        "- 这里只分析 HB5、selected top-4 和第一个控制状态；没有 all-32 expert 对照。",
        "- `D_abs/C_abs` 是 selected-expert 输出几何，不是持久的专家内部状态。",
        "- 本报告是探索性审计，不提供显著性声明。",
        "",
        "候选级数组见 `arrays.npz`，完整数值见 `summary.json`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary, arrays = analyze(args.source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (args.out_dir / "REPORT.md").write_text(_render_report(summary))
    np.savez_compressed(args.out_dir / "arrays.npz", **arrays)
    print(args.out_dir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
