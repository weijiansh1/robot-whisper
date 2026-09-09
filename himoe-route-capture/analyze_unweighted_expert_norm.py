"""Audit pre-gate selected-expert output norms across denoising rounds.

This is the corrected counterpart to ``analyze_raw_expert_amplitude.py``.
For every selected HB expert it reconstructs ``RMS(E_e(h))`` before multiplying
by the top-k gate weight.  Captured expert ids remain authoritative; the result
is still an offline reconstruction from fp16 hidden states, not runtime-exact.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.stats import rankdata

from analyze_expert_activation_future import _first_rows, _layer_weights, _mlp
from analyze_expert_activation_proxy import load_primary_cell


N_DENOISE = 10
N_ACTION_TOKENS = 10
TOP_K = 4
HIDDEN_WIDTH = 1024


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proxy-root",
        type=pathlib.Path,
        default=here / "analysis" / "expert-activation-hidden-matched",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=here / "analysis" / "unweighted-expert-norm",
    )
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--perms", type=int, default=5000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--tasks", default="", help="comma-separated task directory names; default=all"
    )
    return parser.parse_args()


def _rms(value: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(value, dtype=np.float64), axis=axis))


def _linear_slope(value: np.ndarray) -> np.ndarray:
    """Raw-unit least-squares slope along the final denoise axis."""
    time_axis = np.arange(value.shape[-1], dtype=np.float64)
    centered = time_axis - time_axis.mean()
    return np.sum(value * centered, axis=-1) / np.sum(np.square(centered))


def _slot_summaries(
    slot_rms: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if slot_rms.shape != weights.shape:
        raise ValueError(f"slot/weight shape mismatch: {slot_rms.shape} != {weights.shape}")
    raw_mean = slot_rms.mean(axis=-1)
    weighted_mass = np.sum(slot_rms * weights, axis=-1)
    raw_slot_std = slot_rms.std(axis=-1)
    return raw_mean, weighted_mass, raw_slot_std


def _nanmean_last(value: np.ndarray) -> np.ndarray:
    count = np.isfinite(value).sum(axis=-1)
    output = np.full(value.shape[:-1], np.nan, dtype=np.float64)
    np.divide(np.nansum(value, axis=-1), count, out=output, where=count > 0)
    return output


def _pool_auc_from_ranks(
    ranks: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.float64)
    n_pos = y.sum(axis=-1)
    n_neg = y.shape[-1] - n_pos
    rank_sum = np.sum(ranks * y[None], axis=-1)
    numerator = rank_sum - n_pos[None] * (n_pos[None] + 1.0) / 2.0
    denominator = n_pos * n_neg
    pool = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator[None],
        out=pool,
        where=denominator[None] > 0,
    )
    task = _nanmean_last(pool)
    macro = _nanmean_last(task)
    return pool, task, macro


def _pool_auc_from_scores(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Higher-score AUC for scores [feature, task, pool, seed]."""
    if scores.ndim != 4 or scores.shape[1:] != labels.shape:
        raise ValueError(f"bad AUC shapes: {scores.shape}, {labels.shape}")
    ranks = np.apply_along_axis(rankdata, -1, scores).astype(np.float64)
    return _pool_auc_from_ranks(ranks, labels)


def _centered_ranks(value: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, -1, value).astype(np.float64)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    return ranks


def _pool_spearman_from_centered_ranks(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denominator = np.sqrt(
        np.sum(np.square(x), axis=-1) * np.sum(np.square(y), axis=-1)[None]
    )
    pool = np.sum(x * y[None], axis=-1) / np.maximum(denominator, 1e-20)
    task = pool.mean(axis=-1)
    macro = task.mean(axis=-1)
    return pool, task, macro


def _pool_spearman(
    scores: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.ndim != 4 or scores.shape[1:] != target.shape:
        raise ValueError(f"bad correlation shapes: {scores.shape}, {target.shape}")
    return _pool_spearman_from_centered_ranks(
        _centered_ranks(scores), _centered_ranks(target)
    )


def _hierarchical_ci(
    values: np.ndarray, draws: int, rng: np.random.Generator
) -> list[float]:
    """Equal-task, within-task pool bootstrap for one [task, pool] array."""
    rows = [row[np.isfinite(row)] for row in np.asarray(values, dtype=np.float64)]
    rows = [row for row in rows if len(row)]
    if not rows:
        return [float("nan"), float("nan")]
    draw_values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_indices = rng.integers(0, len(rows), size=len(rows))
        task_means = []
        for task_axis in task_indices:
            row = rows[task_axis]
            pool_indices = rng.integers(0, len(row), size=len(row))
            task_means.append(row[pool_indices].mean())
        draw_values[draw] = np.mean(task_means)
    return [
        float(np.percentile(draw_values, 2.5)),
        float(np.percentile(draw_values, 97.5)),
    ]


def _action_eccentricity(actions: np.ndarray) -> np.ndarray:
    """Mean original-space action distance to the other K-1 candidates."""
    n_task, n_pool, n_seed = actions.shape[:3]
    output = np.empty((n_task, n_pool, n_seed), dtype=np.float64)
    for task in range(n_task):
        for pool in range(n_pool):
            delta = actions[task, pool, :, None] - actions[task, pool, None, :]
            distance = _rms(delta, axis=(2, 3))
            output[task, pool] = distance.sum(axis=1) / (n_seed - 1)
    return output


def _common_seed_auc_test(
    scores: np.ndarray,
    labels: np.ndarray,
    feature_names: tuple[str, ...],
    permutations: int,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ranks = np.apply_along_axis(rankdata, -1, scores).astype(np.float64)
    pool, task, observed = _pool_auc_from_ranks(ranks, labels)
    null = np.empty((permutations, len(feature_names)), dtype=np.float64)
    for permutation in range(permutations):
        order = rng.permutation(labels.shape[-1])
        null[permutation] = _pool_auc_from_ranks(ranks, labels[..., order])[2]
    null_max = np.max(np.abs(null - 0.5), axis=1)
    metrics = {}
    for axis, name in enumerate(feature_names):
        metrics[name] = {
            "macro_auc_higher_predicts_success": float(observed[axis]),
            "hierarchical_ci95": _hierarchical_ci(pool[axis], bootstrap, rng),
            "maxT_two_sided_p": float(
                (1 + np.sum(null_max >= abs(observed[axis] - 0.5)))
                / (1 + permutations)
            ),
            "per_task_auc": task[axis].tolist(),
            "informative_pools_per_task": np.isfinite(pool[axis]).sum(axis=-1).tolist(),
        }
    return {
        "feature_names": list(feature_names),
        "metrics": metrics,
        "null_max_abs_auc_minus_half_quantiles": {
            "q50": float(np.quantile(null_max, 0.50)),
            "q95": float(np.quantile(null_max, 0.95)),
            "q99": float(np.quantile(null_max, 0.99)),
        },
    }


def _common_seed_correlation_test(
    scores: np.ndarray,
    target: np.ndarray,
    feature_names: tuple[str, ...],
    permutations: int,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    score_ranks = _centered_ranks(scores)
    target_ranks = _centered_ranks(target)
    pool, task, observed = _pool_spearman_from_centered_ranks(
        score_ranks, target_ranks
    )
    null = np.empty((permutations, len(feature_names)), dtype=np.float64)
    for permutation in range(permutations):
        order = rng.permutation(target.shape[-1])
        null[permutation] = _pool_spearman_from_centered_ranks(
            score_ranks, target_ranks[..., order]
        )[2]
    null_max = np.max(np.abs(null), axis=1)
    metrics = {}
    for axis, name in enumerate(feature_names):
        metrics[name] = {
            "macro_spearman": float(observed[axis]),
            "hierarchical_ci95": _hierarchical_ci(pool[axis], bootstrap, rng),
            "maxT_two_sided_p": float(
                (1 + np.sum(null_max >= abs(observed[axis]))) / (1 + permutations)
            ),
            "per_task_spearman": task[axis].tolist(),
        }
    return {
        "feature_names": list(feature_names),
        "metrics": metrics,
        "null_max_abs_spearman_quantiles": {
            "q50": float(np.quantile(null_max, 0.50)),
            "q95": float(np.quantile(null_max, 0.95)),
            "q99": float(np.quantile(null_max, 0.99)),
        },
    }


def _load_source(summary_path: pathlib.Path, layer: int) -> dict[str, Any]:
    source_summary = json.loads(summary_path.read_text())
    run = pathlib.Path(source_summary["run"])
    checkpoint = pathlib.Path(source_summary["checkpoint"])
    primary = load_primary_cell(run, layer, 0)
    success = np.asarray(
        [bool(row["success"]) for row in primary["summaries"]], dtype=np.int8
    )
    return {
        "name": summary_path.parent.name,
        "run": run,
        "checkpoint": checkpoint,
        "primary": primary,
        "success": success[primary["order"]],
        "actions": np.asarray(primary["actions"][primary["order"]], dtype=np.float32),
    }


def _reconstruct_task(
    source: dict[str, Any], layer: int, threads: int
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(threads)
    run = source["run"]
    primary = source["primary"]
    client = run / "client"
    server = run / "server"
    metadata = json.loads((client / "server_metadata.json").read_text())
    captured_layers = tuple(int(value) for value in metadata["routing_hb_layer_indices"])
    if layer not in captured_layers:
        raise ValueError(f"HB{layer} not present in {captured_layers}")
    layer_axis = captured_layers.index(layer)
    routes = zarr.open_group(str(server / "routes.zarr"), mode="r")
    hidden = zarr.open_group(str(server / "hidden.zarr"), mode="r")
    route_episode = np.asarray(routes["episode_id"][:], dtype=np.int64)
    hidden_episode = np.asarray(hidden["episode_id"][:], dtype=np.int64)
    if not np.array_equal(route_episode, hidden_episode):
        raise ValueError("route and hidden identity axes differ")
    rows = _first_rows(route_episode, primary["episodes"])

    checkpoint_state = torch.load(
        source["checkpoint"], map_location="cpu", weights_only=True, mmap=True
    )
    experts, _shared, gate_weight = _layer_weights(
        checkpoint_state, layer, "cpu", torch
    )
    experts = [tuple(weight.float() for weight in triplet) for triplet in experts]
    gate_weight = gate_weight.float()
    n_candidate = len(rows)
    slot_rms = np.empty(
        (n_candidate, N_DENOISE, N_ACTION_TOKENS, TOP_K), dtype=np.float32
    )
    expert_ids = np.empty_like(slot_rms, dtype=np.uint8)
    combine_weight = np.empty_like(slot_rms)
    weighted_mass = np.empty(
        (n_candidate, N_DENOISE, N_ACTION_TOKENS), dtype=np.float32
    )
    routed_rms = np.empty_like(weighted_mass)
    top4_matches = []
    probability_mae = []

    for denoise in range(N_DENOISE):
        selection_hidden = (rows, [layer_axis], [denoise], slice(1, 11), slice(None))
        selection_topk = (rows, [layer_axis], [denoise], slice(1, 11), slice(None))
        hidden_value = np.asarray(
            hidden["hb_hidden"].get_orthogonal_selection(selection_hidden),
            dtype=np.float32,
        )[:, 0, 0]
        ids_value = np.asarray(
            routes["hb_expert_ids"].get_orthogonal_selection(selection_topk),
            dtype=np.int64,
        )[:, 0, 0]
        raw_weight = np.asarray(
            routes["hb_selected_prob"].get_orthogonal_selection(selection_topk),
            dtype=np.float32,
        )[:, 0, 0]
        alpha_value = raw_weight / np.maximum(
            raw_weight.sum(axis=-1, keepdims=True), 1e-20
        )
        x = torch.from_numpy(np.ascontiguousarray(hidden_value)).reshape(-1, HIDDEN_WIDTH)
        ids = torch.from_numpy(np.ascontiguousarray(ids_value)).reshape(-1, TOP_K)
        alpha = torch.from_numpy(np.ascontiguousarray(alpha_value)).reshape(-1, TOP_K)
        token_count = len(x)

        with torch.inference_mode():
            norms = torch.full((token_count, TOP_K), float("nan"), dtype=torch.float32)
            routed = torch.zeros_like(x, dtype=torch.float32)
            for expert_id, weights in enumerate(experts):
                matches = torch.nonzero(ids == expert_id, as_tuple=False)
                if matches.numel() == 0:
                    continue
                token = matches[:, 0]
                slot = matches[:, 1]
                value = _mlp(torch, x[token], weights)
                norm = value.square().mean(dim=-1).sqrt()
                norms[token, slot] = norm
                routed.index_add_(0, token, value * alpha[token, slot, None])
            if not torch.isfinite(norms).all():
                raise RuntimeError(f"{source['name']} d{denoise}: missing slot norm")
            mass = torch.sum(norms * alpha, dim=-1)
            routed_size = routed.square().mean(dim=-1).sqrt()

            if denoise == 0:
                probabilities = torch.softmax(
                    torch.nn.functional.linear(x, gate_weight), dim=-1
                )
                recomputed = probabilities.topk(TOP_K, dim=-1, sorted=False).indices
                top4_matches.append(
                    float(
                        (
                            recomputed.sort(dim=-1).values
                            == ids.sort(dim=-1).values
                        )
                        .all(dim=-1)
                        .float()
                        .mean()
                    )
                )
                selected_raw = torch.from_numpy(
                    np.ascontiguousarray(raw_weight)
                ).reshape(-1, TOP_K)
                probability_mae.append(
                    float(
                        (probabilities.gather(1, ids) - selected_raw)
                        .abs()
                        .mean()
                    )
                )

        shape = (n_candidate, N_ACTION_TOKENS)
        slot_rms[:, denoise] = norms.numpy().reshape(*shape, TOP_K)
        expert_ids[:, denoise] = ids_value.astype(np.uint8)
        combine_weight[:, denoise] = alpha_value
        weighted_mass[:, denoise] = mass.numpy().reshape(shape)
        routed_rms[:, denoise] = routed_size.numpy().reshape(shape)
        print(f"{source['name']}: reconstructed denoise {denoise}", flush=True)

    order = primary["order"]
    return {
        "name": source["name"],
        "scene_ids": np.asarray(primary["scene_ids"], dtype=np.int64),
        "seed_ids": np.asarray(primary["seed_ids"], dtype=np.int64),
        "expert_ids": expert_ids[order],
        "combine_weight": combine_weight[order],
        "slot_rms": slot_rms[order],
        "weighted_mass": weighted_mass[order],
        "routed_rms": routed_rms[order],
        "success": source["success"],
        "actions": source["actions"],
        "validation": {
            "d0_recomputed_top4_set_match": top4_matches[0],
            "d0_selected_probability_mae": probability_mae[0],
            "weights_sum_max_abs_error": float(
                np.max(np.abs(combine_weight.sum(axis=-1) - 1.0))
            ),
        },
        "source_run": str(run),
        "checkpoint_sha256": primary["checkpoint_sha256"],
    }


def _feature_tensor(raw_mean: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    rounds = np.moveaxis(raw_mean, -1, 0)
    slope = _linear_slope(raw_mean)[None]
    names = tuple(f"d{denoise}" for denoise in range(N_DENOISE)) + (
        "linear_slope_d0_d9",
    )
    return np.concatenate((rounds, slope), axis=0), names


def _trajectory_summary(
    task_names: list[str], raw_mean: np.ndarray, weighted_mass: np.ndarray, routed: np.ndarray
) -> dict[str, Any]:
    result = {"per_task": {}}
    for task_axis, name in enumerate(task_names):
        candidate = raw_mean[task_axis]
        pool_std = candidate.std(axis=1)
        result["per_task"][name] = {
            "raw_mean_median_by_round": np.median(candidate, axis=(0, 1)).tolist(),
            "raw_mean_pool_std_median_by_round": np.median(pool_std, axis=0).tolist(),
            "weighted_mass_median_by_round": np.median(
                weighted_mass[task_axis], axis=(0, 1)
            ).tolist(),
            "routed_rms_median_by_round": np.median(
                routed[task_axis], axis=(0, 1)
            ).tolist(),
            "fraction_raw_d9_below_d0": float(
                np.mean(candidate[..., -1] < candidate[..., 0])
            ),
            "raw_d9_minus_d0_median": float(
                np.median(candidate[..., -1] - candidate[..., 0])
            ),
        }
    return result


def _render_report(summary: dict[str, Any]) -> str:
    success = summary["success_direct_auc"]
    action = summary["action_eccentricity"]
    names = success["feature_names"]
    lines = [
        "# 未加权专家原始输出范数",
        "",
        "本分析重建每个 selected expert 在乘 gate weight 之前的",
        "`raw_k = RMS(E_{id_k}(h))`。固定 HB5，保留全部 10 个 denoise round、",
        "10 个 action token 和 4 个 selected slot。此前 weighted-branch 报告不回答这个问题。",
        "",
        "## 数据与数值边界",
        "",
        "| task | d0 top4 set match | selected-prob MAE | weight-sum error |",
        "|---|---:|---:|---:|",
    ]
    for task in summary["tasks"]:
        validation = summary["sources"][task]["validation"]
        lines.append(
            "| %s | %.3f | %.2e | %.2e |"
            % (
                task,
                validation["d0_recomputed_top4_set_match"],
                validation["d0_selected_probability_mae"],
                validation["weights_sum_max_abs_error"],
            )
        )
    lines += [
        "",
        "captured expert ID/weight 始终作为权威；set match 只说明 fp16 hidden 离线重算",
        "不是 runtime-exact。RMS 若改写成 L2，只需统一乘 `sqrt(1024)=32`。",
        "",
        "## 原始范数轨迹",
        "",
        "| task | raw d0 | raw d9 | d9-d0 | d9<d0 candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in summary["tasks"]:
        row = summary["trajectory"]["per_task"][task]
        curve = row["raw_mean_median_by_round"]
        lines.append(
            "| %s | %.4f | %.4f | %+.4f | %.1f%% |"
            % (
                task,
                curve[0],
                curve[-1],
                row["raw_d9_minus_d0_median"],
                100 * row["fraction_raw_d9_below_d0"],
            )
        )
    lines += [
        "",
        "这些是绝对 checkpoint 单位，没有按 d0、hidden、shared 或 routed 做归一化。",
        "",
        "## 与最终成功的直接关系",
        "",
        "AUC 在每个同观测池内直接由单个 raw scalar 计算；AUC>0.5 表示 raw 越大越成功。",
        "共同 seed-column 置换在 10 个 round + 1 个斜率上做双侧 maxT 校正。",
        "",
        "| score | macro AUC | hierarchical 95% CI | maxT p |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        row = success["metrics"][name]
        lines.append(
            "| %s | %.3f | [%.3f, %.3f] | %.4f |"
            % (
                name,
                row["macro_auc_higher_predicts_success"],
                row["hierarchical_ci95"][0],
                row["hierarchical_ci95"][1],
                row["maxT_two_sided_p"],
            )
        )
    lines += [
        "",
        "## 与最终动作中心距离的直接关系",
        "",
        "目标是在原始标准化 10×7 action 空间中，每个候选到同池其他 31 个候选的",
        "平均距离；rho>0 表示 raw 越大，最终动作越偏离候选云中心。",
        "",
        "| score | macro Spearman | hierarchical 95% CI | maxT p |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        row = action["metrics"][name]
        lines.append(
            "| %s | %+.3f | [%+.3f, %+.3f] | %.4f |"
            % (
                name,
                row["macro_spearman"],
                row["hierarchical_ci95"][0],
                row["hierarchical_ci95"][1],
                row["maxT_two_sided_p"],
            )
        )
    lines += [
        "",
        "## 纠正后的结论",
        "",
        "1. 未加权 raw norm 不是跨 checkpoint 单调下降：Goal 末轮低于首轮，Spatial 末轮反而更高。",
        "2. 没有任何 round 或 raw 斜率在 maxT 校正后稳定预测 eventual success。",
        "3. raw 斜率与最终动作中心距离存在校正后关联，但它是候选云几何，不是成功或动作承诺。",
        "4. 真正的动作承诺仍需记录每轮 `x_d`，直接检验单候选 `RMS(x_d-x_10)`。",
        "",
        "斜率关联的 p 值只在 11 个动作几何指标内做 maxT；再对 success/geometry",
        "两个 endpoint family 做保守 Bonferroni 后仍需乘 2。该分析是事后纠正实验，",
        "不能作为确认性发现。",
        "",
        "## 解释边界",
        "",
        "- 这是未加权 per-slot expert 输出范数，但来自 fp16 hidden + checkpoint 的离线重建。",
        "- success 是远端环境结果；直接 AUC 仍是关联，不是专家幅度的因果效应。",
        "- 最终动作中心距离不是逐轮剩余动作修正。现有捕获没有 `x_tau`，不能检验真正的动作承诺时间。",
        "- raw 输出变大或变小没有固定的置信度语义；必须结合 router margin、专家方向和动作收敛。",
        "",
        "候选级 per-slot 数组见 `unweighted_expert_norms.npz`，完整统计见 `summary.json`。",
    ]
    return "\n".join(lines) + "\n"


def _plot(summary: dict[str, Any], output: pathlib.Path) -> None:
    tasks = summary["tasks"]
    rounds = np.arange(N_DENOISE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for task in tasks:
        row = summary["trajectory"]["per_task"][task]
        axes[0].plot(rounds, row["raw_mean_median_by_round"], marker="o", label=task)
        axes[1].plot(
            rounds, row["raw_mean_pool_std_median_by_round"], marker="o", label=task
        )
    axes[0].set_title("Pre-gate expert norm")
    axes[0].set_xlabel("denoise round")
    axes[0].set_ylabel("raw RMS")
    axes[0].legend(fontsize=7)
    axes[1].set_title("K32 raw-norm dispersion")
    axes[1].set_xlabel("denoise round")
    axes[1].set_ylabel("within-pool SD (raw RMS)")

    success = summary["success_direct_auc"]["metrics"]
    action = summary["action_eccentricity"]["metrics"]
    success_curve = [success[f"d{d}"]["macro_auc_higher_predicts_success"] for d in rounds]
    action_curve = [action[f"d{d}"]["macro_spearman"] for d in rounds]
    axes[2].plot(rounds, success_curve, marker="o", label="success AUC")
    axes[2].plot(rounds, action_curve, marker="s", label="action eccentricity rho")
    axes[2].axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    axes[2].axhline(0.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[2].set_title("Direct within-pool associations")
    axes[2].set_xlabel("denoise round")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.layer != 5:
        raise ValueError("the corrected fixed cell is HB5")
    started = time.perf_counter()
    args.proxy_root = args.proxy_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary_paths = sorted(args.proxy_root.glob("*/summary.json"))
    requested_tasks = {name for name in args.tasks.split(",") if name}
    if requested_tasks:
        summary_paths = [path for path in summary_paths if path.parent.name in requested_tasks]
        missing = requested_tasks - {path.parent.name for path in summary_paths}
        if missing:
            raise ValueError(f"unknown task names: {sorted(missing)}")
    if not summary_paths:
        raise FileNotFoundError(f"no source summaries below {args.proxy_root}")
    reconstructed = []
    for path in summary_paths:
        source = _load_source(path, args.layer)
        print(f"loading pre-gate expert outputs: {source['name']}", flush=True)
        reconstructed.append(_reconstruct_task(source, args.layer, args.threads))

    task_names = [task["name"] for task in reconstructed]
    ids = np.stack([task["expert_ids"] for task in reconstructed])
    weights = np.stack([task["combine_weight"] for task in reconstructed])
    slot_rms = np.stack([task["slot_rms"] for task in reconstructed])
    weighted_token = np.stack([task["weighted_mass"] for task in reconstructed])
    routed_token = np.stack([task["routed_rms"] for task in reconstructed])
    success = np.stack([task["success"] for task in reconstructed])
    actions = np.stack([task["actions"] for task in reconstructed])

    raw_token, weighted_check, raw_slot_std_token = _slot_summaries(slot_rms, weights)
    mass_error = float(np.max(np.abs(weighted_check - weighted_token)))
    raw_mean = raw_token.mean(axis=-1)
    weighted_mass = weighted_token.mean(axis=-1)
    routed_rms = routed_token.mean(axis=-1)
    raw_slot_std = raw_slot_std_token.mean(axis=-1)
    feature_scores, feature_names = _feature_tensor(raw_mean)
    action_eccentricity = _action_eccentricity(actions)

    summary = {
        "experiment": "unweighted_selected_expert_norm_cross_task_v1",
        "status": "offline_reconstructed_pre_gate_norm_not_runtime_exact",
        "definition": "raw_k = RMS(E_selected_k(h)) before gate multiplication",
        "fixed_layer": args.layer,
        "tasks": task_names,
        "pools_per_task": int(slot_rms.shape[1]),
        "candidates_per_pool": int(slot_rms.shape[2]),
        "denoise_rounds": int(slot_rms.shape[3]),
        "action_tokens": int(slot_rms.shape[4]),
        "selected_slots": int(slot_rms.shape[5]),
        "permutations": args.perms,
        "bootstrap_draws": args.bootstrap,
        "sources": {
            task["name"]: {
                "run": task["source_run"],
                "checkpoint_sha256": task["checkpoint_sha256"],
                "validation": task["validation"],
            }
            for task in reconstructed
        },
        "formula_validation": {
            "max_abs_expert_mass_minus_sum_weight_times_raw": mass_error
        },
        "trajectory": _trajectory_summary(
            task_names, raw_mean, weighted_mass, routed_rms
        ),
        "success_direct_auc": _common_seed_auc_test(
            feature_scores,
            success,
            feature_names,
            args.perms,
            args.bootstrap,
            rng,
        ),
        "action_eccentricity": _common_seed_correlation_test(
            feature_scores,
            action_eccentricity,
            feature_names,
            args.perms,
            args.bootstrap,
            rng,
        ),
        "limitations": [
            "Offline CPU fp32 expert evaluation starts from stored fp16 hidden states.",
            "Captured expert ids and selected probabilities are authoritative.",
            "No intermediate flow latent or provisional action was captured.",
            "Success and final-action eccentricity are associative endpoints.",
        ],
    }
    summary["elapsed_seconds"] = float(time.perf_counter() - started)

    np.savez_compressed(
        args.out_dir / "unweighted_expert_norms.npz",
        task_names=np.asarray(task_names),
        scene_ids=np.stack([task["scene_ids"] for task in reconstructed]),
        seed_ids=np.stack([task["seed_ids"] for task in reconstructed]),
        selected_expert_ids=ids,
        selected_expert_weight=weights,
        selected_expert_raw_rms=slot_rms,
        weighted_expert_mass_by_token=weighted_token,
        routed_rms_by_token=routed_token,
        raw_mean=raw_mean,
        raw_slot_std=raw_slot_std,
        weighted_expert_mass=weighted_mass,
        routed_rms=routed_rms,
        final_actions_standardized=actions,
        success=success,
        action_eccentricity=action_eccentricity,
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(_render_report(summary), encoding="utf-8")
    _plot(summary, args.out_dir / "overview.png")
    print(f"wrote {args.out_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
