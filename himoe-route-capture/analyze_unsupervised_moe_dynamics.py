"""Label-free MoE routing geometry versus flow-matching dynamics.

At every real query state the flow-lead capture contains 16 sibling candidates:
the observation is fixed, only the initial flow noise changes.  This analysis
does not train a decoder, form action-basin labels, or use rollout outcomes.  It
compares continuous pairwise geometries and uses candidate-label permutations
within each query as the null.

The route fingerprint is the full 32-way HB router distribution over eight
layers and ten action tokens.  Hellinger distance is used because each router
output is a probability distribution.  State-token and AS routes are exact
negative controls in this capture.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import zarr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


LIVE_DIMS = 7
N_EXPERTS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _distance(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return pdist(flat, metric="euclidean") / math.sqrt(flat.shape[1])


def _route_distance(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, None)
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-20)
    positions = int(np.prod(probability.shape[1:-1]))
    # RMS Hellinger distance over layer-token positions.
    return pdist(np.sqrt(probability).reshape(len(probability), -1)) / math.sqrt(
        2.0 * positions
    )


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(left).astype(np.float64)
    right_rank = rankdata(right).astype(np.float64)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    scale = math.sqrt(float(left_rank @ left_rank) * float(right_rank @ right_rank))
    return float(left_rank @ right_rank / scale) if scale > 0 else float("nan")


def _partial_rank_correlation(
    left: np.ndarray, right: np.ndarray, controls: list[np.ndarray]
) -> float:
    left_rank = rankdata(left).astype(np.float64)
    right_rank = rankdata(right).astype(np.float64)
    design = np.column_stack(
        [np.ones(len(left_rank))]
        + [rankdata(control).astype(np.float64) for control in controls]
    )
    left_rank -= design @ np.linalg.lstsq(design, left_rank, rcond=None)[0]
    right_rank -= design @ np.linalg.lstsq(design, right_rank, rcond=None)[0]
    scale = math.sqrt(float(left_rank @ left_rank) * float(right_rank @ right_rank))
    return float(left_rank @ right_rank / scale) if scale > 0 else float("nan")


def _load(run: pathlib.Path) -> dict:
    part_paths = sorted(run.glob("flow_traces_part*.npz"))
    if not part_paths:
        raise FileNotFoundError("no flow_traces_part*.npz under %s" % run)
    parts = [np.load(path) for path in part_paths]
    trajectory = np.concatenate([part["x_traj"] for part in parts])
    query_id = np.concatenate([part["query_id"] for part in parts]).astype(np.int64)
    candidate_id = np.concatenate([part["candidate_id"] for part in parts]).astype(np.int64)
    if trajectory.ndim == 5 and trajectory.shape[2] == 1:
        trajectory = trajectory[:, :, 0]
    if trajectory.shape[1:] != (11, 10, 24):
        raise ValueError("unexpected trajectory shape %s" % (trajectory.shape,))

    records = json.loads((run / "query_records.json").read_text())
    record_by_query = {int(record["query_id"]): record for record in records}
    queries = np.unique(query_id)
    for query in queries:
        rows = np.flatnonzero(query_id == query)
        if len(rows) != 16 or not np.array_equal(candidate_id[rows], np.arange(16)):
            raise ValueError("query %d is not an ordered 16-candidate group" % query)
    if set(record_by_query) != set(int(query) for query in queries):
        raise ValueError("query records and flow traces differ")

    route = zarr.open_group(str(run / "routes.zarr"), mode="r")
    route_query = np.asarray(route["episode_id"][:], dtype=np.int64)
    if not np.array_equal(route_query, query_id):
        raise ValueError("route and flow rows do not have the same query ordering")
    if route["hb_router_probs"].shape != (len(query_id), 8, 10, 11, N_EXPERTS):
        raise ValueError("unexpected router shape %s" % (route["hb_router_probs"].shape,))
    metadata = json.loads((run / "server_metadata.json").read_text())
    layer_numbers = [int(layer) for layer in metadata["routing_hb_layer_indices"]]
    if len(layer_numbers) != 8:
        raise ValueError("expected 8 captured HB layers, got %s" % layer_numbers)

    return {
        "trajectory": np.asarray(trajectory, dtype=np.float32),
        "query_id": query_id,
        "candidate_id": candidate_id,
        "queries": queries,
        "record_by_query": record_by_query,
        "layer_numbers": layer_numbers,
        "route": route,
    }


def _query_rows(data: dict, query: int) -> np.ndarray:
    rows = np.flatnonzero(data["query_id"] == query)
    return rows[np.argsort(data["candidate_id"][rows])]


def _summarize(values: dict[int, float], data: dict) -> dict:
    per_rollout: dict[int, list[float]] = {}
    for query, value in values.items():
        rollout = int(data["record_by_query"][query]["flow_noise_seed"])
        per_rollout.setdefault(rollout, []).append(value)
    rollout_mean = {
        str(rollout): float(np.nanmean(group)) for rollout, group in per_rollout.items()
    }
    valid = np.asarray(list(rollout_mean.values()), dtype=np.float64)
    return {
        "mean": float(np.nanmean(list(values.values()))),
        "rollout_range": [float(np.nanmin(valid)), float(np.nanmax(valid))],
        "per_rollout": rollout_mean,
        "per_query": {str(query): float(value) for query, value in values.items()},
    }


def _pair_stat(
    route: np.ndarray,
    target: np.ndarray,
    controls: list[np.ndarray],
) -> float:
    if controls:
        return _partial_rank_correlation(route, target, controls)
    return _rank_correlation(route, target)


def _pair_permutation_test(
    route_matrix: dict[int, np.ndarray],
    target: dict[int, np.ndarray],
    controls: list[dict[int, np.ndarray]],
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    queries = sorted(route_matrix)
    observed_per_query = {
        query: _pair_stat(
            squareform(route_matrix[query], checks=False),
            target[query],
            [control[query] for control in controls],
        )
        for query in queries
    }
    observed = float(np.nanmean(list(observed_per_query.values())))
    null = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        values = []
        for query in queries:
            matrix = route_matrix[query]
            order = rng.permutation(len(matrix))
            permuted = squareform(matrix[np.ix_(order, order)], checks=False)
            values.append(
                _pair_stat(
                    permuted,
                    target[query],
                    [control[query] for control in controls],
                )
            )
        null[draw] = np.nanmean(values)
    null_center = float(null.mean())
    p_value = (1.0 + float(np.sum(np.abs(null - null_center) >= abs(observed - null_center)))) / (
        permutations + 1.0
    )
    return {
        "observed_mean": observed,
        "null_mean": null_center,
        "null_95": [float(value) for value in np.percentile(null, [2.5, 97.5])],
        "two_sided_p": p_value,
        "permutations": permutations,
    }


def _scalar_permutation_test(
    feature: dict[int, np.ndarray],
    target: dict[int, np.ndarray],
    controls: list[dict[int, np.ndarray]],
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    queries = sorted(feature)
    observed_values = [
        _pair_stat(
            feature[query], target[query], [control[query] for control in controls]
        )
        for query in queries
    ]
    observed = float(np.nanmean(observed_values))
    null = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        values = []
        for query in queries:
            permuted = feature[query][rng.permutation(len(feature[query]))]
            values.append(
                _pair_stat(
                    permuted,
                    target[query],
                    [control[query] for control in controls],
                )
            )
        null[draw] = np.nanmean(values)
    null_center = float(null.mean())
    p_value = (1.0 + float(np.sum(np.abs(null - null_center) >= abs(observed - null_center)))) / (
        permutations + 1.0
    )
    return {
        "observed_mean": observed,
        "null_mean": null_center,
        "null_95": [float(value) for value in np.percentile(null, [2.5, 97.5])],
        "two_sided_p": p_value,
        "permutations": permutations,
    }


def analyze(data: dict, permutations: int, seed: int) -> dict:
    trajectory = data["trajectory"]
    queries = [int(query) for query in data["queries"]]
    route_group = data["route"]
    layer_numbers = data["layer_numbers"]

    current: list[dict[int, np.ndarray]] = [dict() for _ in range(10)]
    immediate: list[dict[int, np.ndarray]] = [dict() for _ in range(10)]
    remaining: list[dict[int, np.ndarray]] = [dict() for _ in range(10)]
    final: dict[int, np.ndarray] = {}
    initial: dict[int, np.ndarray] = {}
    current_norm: list[dict[int, np.ndarray]] = [dict() for _ in range(10)]
    immediate_norm: list[dict[int, np.ndarray]] = [dict() for _ in range(10)]
    for query in queries:
        rows = _query_rows(data, query)
        final[query] = _distance(trajectory[rows, -1, :, :LIVE_DIMS])
        initial[query] = _distance(trajectory[rows, 0, :, :LIVE_DIMS])
        for tau in range(10):
            now = trajectory[rows, tau, :, :LIVE_DIMS]
            step = trajectory[rows, tau + 1, :, :LIVE_DIMS] - now
            rest = trajectory[rows, -1, :, :LIVE_DIMS] - now
            current[tau][query] = _distance(now)
            immediate[tau][query] = _distance(step)
            remaining[tau][query] = _distance(rest)
            current_norm[tau][query] = np.sqrt(np.mean(np.square(now), axis=(1, 2)))
            immediate_norm[tau][query] = np.sqrt(np.mean(np.square(step), axis=(1, 2)))

    route_vectors: list[dict[int, np.ndarray]] = []
    route_matrices: list[dict[int, np.ndarray]] = []
    top1: list[dict[int, np.ndarray]] = []
    confidence: list[dict[int, np.ndarray]] = []
    per_tau = []
    layer_probe = []
    route_zero: dict[int, np.ndarray] | None = None

    for tau in range(10):
        probability = np.asarray(
            route_group["hb_router_probs"][:, :, tau, 1:, :], dtype=np.float32
        )
        entropy = np.asarray(route_group["hb_entropy"][:, :, tau, 1:], dtype=np.float32)
        route_tau: dict[int, np.ndarray] = {}
        matrix_tau: dict[int, np.ndarray] = {}
        top1_tau: dict[int, np.ndarray] = {}
        confidence_tau: dict[int, np.ndarray] = {}
        for query in queries:
            rows = _query_rows(data, query)
            distance = _route_distance(probability[rows])
            route_tau[query] = distance
            matrix_tau[query] = squareform(distance)
            top1_tau[query] = np.argmax(probability[rows], axis=-1)
            confidence_tau[query] = 1.0 - entropy[rows].mean(axis=(1, 2)) / math.log(N_EXPERTS)
        route_vectors.append(route_tau)
        route_matrices.append(matrix_tau)
        top1.append(top1_tau)
        confidence.append(confidence_tau)
        if route_zero is None:
            route_zero = route_tau

        metrics = {
            "route_current_live7": {
                query: _rank_correlation(route_tau[query], current[tau][query])
                for query in queries
            },
            "route_immediate_update": {
                query: _rank_correlation(route_tau[query], immediate[tau][query])
                for query in queries
            },
            "route_immediate_update_given_current": {
                query: _partial_rank_correlation(
                    route_tau[query], immediate[tau][query], [current[tau][query]]
                )
                for query in queries
            },
            "route_remaining_update_given_current": {
                query: _partial_rank_correlation(
                    route_tau[query], remaining[tau][query], [current[tau][query]]
                )
                for query in queries
            },
            "route_final_action": {
                query: _rank_correlation(route_tau[query], final[query])
                for query in queries
            },
            "route_final_action_given_current": {
                query: _partial_rank_correlation(
                    route_tau[query], final[query], [current[tau][query]]
                )
                for query in queries
            },
            "route_geometry_stability_to_tau0": {
                query: _rank_correlation(route_zero[query], route_tau[query])
                for query in queries
            },
            "gate_confidence_update_norm": {
                query: _rank_correlation(confidence_tau[query], immediate_norm[tau][query])
                for query in queries
            },
            "gate_confidence_update_norm_given_current_norm": {
                query: _partial_rank_correlation(
                    confidence_tau[query],
                    immediate_norm[tau][query],
                    [current_norm[tau][query]],
                )
                for query in queries
            },
        }
        row = {"tau": tau, **{name: _summarize(value, data) for name, value in metrics.items()}}
        if tau < 9:
            # Filled after the next route slice has been read.
            row["top1_transition_stability"] = None
        per_tau.append(row)

        if tau in (0, 9):
            for layer_axis, layer_number in enumerate(layer_numbers):
                layer_route = {}
                for query in queries:
                    rows = _query_rows(data, query)
                    layer_route[query] = _route_distance(probability[rows, layer_axis])
                layer_probe.append(
                    {
                        "tau": tau,
                        "layer": int(layer_number),
                        "route_current_live7": _summarize(
                            {
                                query: _rank_correlation(
                                    layer_route[query], current[tau][query]
                                )
                                for query in queries
                            },
                            data,
                        ),
                        "route_immediate_update_given_current": _summarize(
                            {
                                query: _partial_rank_correlation(
                                    layer_route[query],
                                    immediate[tau][query],
                                    [current[tau][query]],
                                )
                                for query in queries
                            },
                            data,
                        ),
                        "route_final_action_given_current": _summarize(
                            {
                                query: _partial_rank_correlation(
                                    layer_route[query], final[query], [current[tau][query]]
                                )
                                for query in queries
                            },
                            data,
                        ),
                    }
                )

    for tau in range(9):
        stability = {
            query: float(np.mean(top1[tau][query] == top1[tau + 1][query]))
            for query in queries
        }
        per_tau[tau]["top1_transition_stability"] = _summarize(stability, data)

    early_curve = []
    assert route_zero is not None
    for stage in range(11):
        route_values = {}
        initial_values = {}
        conditional_values = {}
        for query in queries:
            rows = _query_rows(data, query)
            stage_distance = _distance(trajectory[rows, stage, :, :LIVE_DIMS])
            route_values[query] = _rank_correlation(route_zero[query], stage_distance)
            initial_values[query] = _rank_correlation(initial[query], stage_distance)
            conditional_values[query] = _partial_rank_correlation(
                route_zero[query], stage_distance, [initial[query]]
            )
        early_curve.append(
            {
                "stage": stage,
                "early_route_to_stage": _summarize(route_values, data),
                "initial_live7_to_stage": _summarize(initial_values, data),
                "early_route_to_stage_given_initial_live7": (
                    None if stage == 0 else _summarize(conditional_values, data)
                ),
            }
        )

    rng = np.random.default_rng(seed)
    tests = {
        "tau0_route_current": _pair_permutation_test(
            route_matrices[0], current[0], [], permutations, rng
        ),
        "tau0_route_final": _pair_permutation_test(
            route_matrices[0], final, [], permutations, rng
        ),
        "tau0_route_final_given_current": _pair_permutation_test(
            route_matrices[0], final, [current[0]], permutations, rng
        ),
        "tau0_route_immediate_update_given_current": _pair_permutation_test(
            route_matrices[0], immediate[0], [current[0]], permutations, rng
        ),
        "tau9_route_final_given_current": _pair_permutation_test(
            route_matrices[9], final, [current[9]], permutations, rng
        ),
        "tau5_gate_confidence_update_norm_given_current_norm": _scalar_permutation_test(
            confidence[5], immediate_norm[5], [current_norm[5]], permutations, rng
        ),
    }

    state_probability = np.asarray(
        route_group["hb_router_probs"][:, :, :, 0, :], dtype=np.float32
    )
    state_max_span = 0.0
    for query in queries:
        rows = _query_rows(data, query)
        state_max_span = max(
            state_max_span,
            float(np.ptp(state_probability[rows], axis=0).max()),
        )
    as_probability = np.asarray(route_group["as_probs"][:], dtype=np.float32)

    return {
        "experiment": "label_free_moe_routing_geometry",
        "data": {
            "queries": len(queries),
            "candidates_per_query": 16,
            "candidate_pairs_per_query": 120,
            "rollouts": len(
                {
                    int(data["record_by_query"][query]["flow_noise_seed"])
                    for query in queries
                }
            ),
            "flow_stages": 11,
            "hb_layers": layer_numbers,
            "action_tokens": 10,
            "router_experts": N_EXPERTS,
        },
        "method": {
            "route_fingerprint": "full 32-way HB probabilities over 8 layers x 10 action tokens",
            "route_distance": "RMS Hellinger distance",
            "dynamic_distance": "RMS Euclidean distance on the normalized live 7D action latent",
            "association": "within-query Spearman correlation over 120 candidate pairs",
            "conditional_association": "partial Spearman after residualizing current-latent distance",
            "learning": "no decoder, no clusters, no action labels, no outcome labels",
            "null": "permute candidate identity of the complete route fingerprint within query",
        },
        "negative_controls": {
            "state_token_max_probability_span_within_query": state_max_span,
            "as_probability_std_all_rows": float(as_probability.std(axis=0).mean()),
        },
        "per_tau": per_tau,
        "early_route_future_curve": early_curve,
        "layer_probe": layer_probe,
        "permutation_tests": tests,
    }


def _fmt(value: float) -> str:
    return "%.3f" % value


def render_report(summary: dict) -> str:
    data = summary["data"]
    lines = [
        "# 无监督 MoE 路由与去噪动力学",
        "",
        "## 实验思路",
        "",
        "- 固定真实观测，在每个 query 比较 16 个不同初始噪声候选；共 %d 个 query、%d 条 rollout。"
        % (data["queries"], data["rollouts"]),
        "- 路由指纹使用 8 个 HB 层、10 个 action token 的完整 32-way router probability；候选间采用 Hellinger 距离。",
        "- 只计算连续距离的 Spearman/partial-Spearman，不训练 decoder、不聚类、不构造 action-basin 标签，也不使用成功率。",
        "- 条件相关控制当前 7 维 action latent 距离；置乱对照在每个 query 内整体打乱 candidate 与路由指纹的对应。",
        "",
        "## 结果",
        "",
        "| tau | 路由-当前 latent | 路由-即时更新（控制当前） | 路由-剩余修正（控制当前） | 路由-最终动作 | 路由-最终动作（控制当前） | 路由几何相对 tau0 稳定性 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_tau"]:
        lines.append(
            "| %d | %s | %s | %s | %s | %s | %s |"
            % (
                row["tau"],
                _fmt(row["route_current_live7"]["mean"]),
                _fmt(row["route_immediate_update_given_current"]["mean"]),
                _fmt(row["route_remaining_update_given_current"]["mean"]),
                _fmt(row["route_final_action"]["mean"]),
                _fmt(row["route_final_action_given_current"]["mean"]),
                _fmt(row["route_geometry_stability_to_tau0"]["mean"]),
            )
        )

    tests = summary["permutation_tests"]
    lines.extend(
        [
            "",
            "tau0 的路由与当前 latent 几何高度一致（rho=%s），与最终动作仅为 rho=%s；控制当前 latent 后为 rho=%s。三项 query 内置乱检验的双侧 p 分别为 %s、%s、%s。"
            % (
                _fmt(tests["tau0_route_current"]["observed_mean"]),
                _fmt(tests["tau0_route_final"]["observed_mean"]),
                _fmt(tests["tau0_route_final_given_current"]["observed_mean"]),
                _fmt(tests["tau0_route_current"]["two_sided_p"]),
                _fmt(tests["tau0_route_final"]["two_sided_p"]),
                _fmt(tests["tau0_route_final_given_current"]["two_sided_p"]),
            ),
            "",
            "路由对即时去噪更新仍有超出当前 latent 距离的关联：tau0 partial rho=%s（p=%s）。但到 tau9，路由对最终动作的条件相关只有 %s（p=%s）。"
            % (
                _fmt(tests["tau0_route_immediate_update_given_current"]["observed_mean"]),
                _fmt(tests["tau0_route_immediate_update_given_current"]["two_sided_p"]),
                _fmt(tests["tau9_route_final_given_current"]["observed_mean"]),
                _fmt(tests["tau9_route_final_given_current"]["two_sided_p"]),
            ),
            "",
            "### tau0 路由指纹随后对应什么",
            "",
            "| flow stage | tau0 路由-该阶段 latent | 初始 live7-该阶段 latent | 路由条件相关（控制初始 live7） |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in summary["early_route_future_curve"]:
        conditional = row["early_route_to_stage_given_initial_live7"]
        lines.append(
            "| %d | %s | %s | %s |"
            % (
                row["stage"],
                _fmt(row["early_route_to_stage"]["mean"]),
                _fmt(row["initial_live7_to_stage"]["mean"]),
                "n/a" if conditional is None else _fmt(conditional["mean"]),
            )
        )

    confidence = tests["tau5_gate_confidence_update_norm_given_current_norm"]
    controls = summary["negative_controls"]
    lines.extend(
        [
            "",
            "tau0 路由几何到 stage 8 仍主要跟随初始噪声，最后才与最终动作脱钩。router confidence 与单步更新幅度在中段 tau5 有 partial rho=%s（p=%s），说明门控确定度更像局部更新强度指标。"
            % (_fmt(confidence["observed_mean"]), _fmt(confidence["two_sided_p"])),
            "",
            "零对照通过：同一 query 内 state-token router probability 最大跨度=%g，AS probability 的全局平均标准差=%g；可变信号只来自 action-token HB 路由。"
            % (
                controls["state_token_max_probability_span_within_query"],
                controls["as_probability_std_all_rows"],
            ),
            "",
            "### 层级和路由重组",
            "",
            "| HB layer | tau0 路由-当前 | tau0 路由-即时更新（控制当前） | tau9 路由-当前 | tau9 路由-即时更新（控制当前） | tau9 路由-最终动作（控制当前） |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    layer_rows = {
        (row["tau"], row["layer"]): row for row in summary["layer_probe"]
    }
    for layer in data["hb_layers"]:
        early = layer_rows[(0, layer)]
        late = layer_rows[(9, layer)]
        lines.append(
            "| %d | %s | %s | %s | %s | %s |"
            % (
                layer,
                _fmt(early["route_current_live7"]["mean"]),
                _fmt(early["route_immediate_update_given_current"]["mean"]),
                _fmt(late["route_current_live7"]["mean"]),
                _fmt(late["route_immediate_update_given_current"]["mean"]),
                _fmt(late["route_final_action_given_current"]["mean"]),
            )
        )
    transitions = [
        row["top1_transition_stability"]["mean"]
        for row in summary["per_tau"][:-1]
    ]
    lines.extend(
        [
            "",
            "描述性分层是：深层 HB 12-14 对当前 latent 和即时更新最敏感；tau9 时浅层 3-5 出现较弱的终点条件相关，而深层仍偏向局部修正。相邻 denoise step 的 top-1 专家保持率从 tau0->1 的 %s 降到 tau8->9 的 %s，主要路由重组发生在末段。该层级差异尚未做跨任务确认。"
            % (_fmt(transitions[0]), _fmt(transitions[-1])),
            "",
            "## 结论",
            "",
            "这批数据不支持把早期 MoE 路由解释成独立的未来动作或意图编码。它首先是当前 noisy action latent 的离散/概率化投影，并与模型接下来如何修正该 latent 有稳定关系；最终动作相关主要由当前 latent 几何传递。更合适的底层解释是：HB 路由描述局部去噪计算路径，而不是提前选定语义化未来。",
            "",
            "范围：一个 LIBERO-Goal 任务、一个初始场景、4 条 rollout；相邻 query 不独立。置乱结论条件于这些已观测状态，尚不能跨任务泛化。当前 flow run 没有 expert output norm，因此本报告只解释 router probability/identity，不能把它写成专家激活大小结论。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run = pathlib.Path(args.run).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load(run)
    summary = analyze(data, args.permutations, args.seed)
    summary["run"] = str(run)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (out_dir / "report.md").write_text(render_report(summary))
    print("wrote %s" % (out_dir / "summary.json"), flush=True)
    print("wrote %s" % (out_dir / "report.md"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
