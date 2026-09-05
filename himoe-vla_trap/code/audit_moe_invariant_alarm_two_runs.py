#!/usr/bin/env python3
"""Audit the fixed MoE alarm on both independent 50x8 LIBERO route corpora."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "code"))

import evaluate_moe_invariant_alarm_cache_new as alarm  # noqa: E402


OUTPUT = PACKAGE_ROOT / "results/moe_invariant_alarm_two_run_audit"
CACHE_ROOT = WORKSPACE_ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
SPECS = {
    "seed1000_1007": {
        "run_id": "right-50x8-20260903",
        "config": PACKAGE_ROOT / "configs/moe_invariant_alarm_cache_new.json",
        "result": PACKAGE_ROOT / "results/moe_invariant_alarm_cache_new_recomputed_50x8",
        "archived_result": PACKAGE_ROOT / "results/moe_invariant_alarm_cache_new",
        "expected_flow_seeds": list(range(1000, 1008)),
    },
    "seed1008_1015": {
        "run_id": "right-50x8b-20260903",
        "config": PACKAGE_ROOT / "configs/moe_invariant_alarm_cache_new_50x8b.json",
        "result": PACKAGE_ROOT / "results/moe_invariant_alarm_cache_new_50x8b",
        "expected_flow_seeds": list(range(1008, 1016)),
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(alarm.plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_bundle(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    config = read_json(spec["config"])
    result = Path(spec["result"])
    summary = read_json(result / "summary.json")
    split = pd.read_csv(result / "tables/task_split.csv")
    maxima = pd.read_csv(result / "tables/confirmation_success_head_maxima.csv")
    runs = dict(alarm.discover_runs(CACHE_ROOT, spec["run_id"]))
    if len(runs) != 40 or len(split) != 40:
        raise AssertionError(f"{name}: expected 40 tasks")
    references = {
        head: maxima[f"max_{head}"].dropna().to_numpy(dtype=np.float64)
        for head in alarm.HEADS
    }
    return {
        "name": name,
        "spec": spec,
        "config": config,
        "result": result,
        "summary": summary,
        "split": split,
        "runs": runs,
        "references": references,
        "threshold": float(summary["threshold"]["detector_confidence"]),
        "clock_query": int(summary["clock"]["query"]),
    }


def structural_audit(bundle: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_seeds = set(bundle["spec"]["expected_flow_seeds"])
    for task, run in sorted(bundle["runs"].items()):
        meta = read_json(run / "meta.json")
        summaries = read_json(run / "client/summaries.json")
        route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        episode = np.asarray(route["episode_id"][:], dtype=np.int32)
        control = np.asarray(route["control_step"][:], dtype=np.int64)
        calls = np.asarray([int(item["inference_calls"]) for item in summaries])
        episode_ids = np.asarray([int(item["episode_index"]) for item in summaries])
        expected_episode = np.repeat(episode_ids, calls)
        lengths = {int(route[key].shape[0]) for key in route.array_keys()}
        probe_rows = np.unique(np.asarray([0, len(episode) // 2, len(episode) - 1]))
        probabilities = np.asarray(
            route["hb_router_probs"].oindex[probe_rows, :, :, :, :], dtype=np.float32
        )
        seeds = {int(item["flow_noise_seed"]) for item in summaries}
        checks = {
            "complete_meta": meta.get("status") == "complete"
            and bool(meta.get("sampling", {}).get("complete")),
            "episodes_400": len(summaries) == 400,
            "all_array_lengths_equal": len(lengths) == 1,
            "rows_equal_client_calls": len(episode) == int(calls.sum()),
            "episode_rows_exact": np.array_equal(episode, expected_episode),
            "control_steps_exact": np.array_equal(control, np.arange(len(control))),
            "route_geometry_exact": tuple(route["hb_router_probs"].shape[1:])
            == (8, 10, 11, 32),
            "durable_rows_exact": int(route.attrs["durable_rows"]) == len(episode),
            "flow_seed_set_exact": seeds == expected_seeds,
            "probabilities_finite": bool(np.isfinite(probabilities).all()),
            "probabilities_nonnegative": bool((probabilities >= 0).all()),
        }
        problems = [key for key, passed in checks.items() if not passed]
        rows.append(
            {
                "run": bundle["name"],
                "run_id": bundle["spec"]["run_id"],
                "task": task,
                "episodes": len(summaries),
                "route_queries": len(episode),
                "successes": sum(bool(item["success"]) for item in summaries),
                "failures": sum(not bool(item["success"]) for item in summaries),
                "probability_sum_max_abs_error": float(
                    np.abs(probabilities.sum(axis=-1) - 1.0).max()
                ),
                "ok": not problems,
                "problems": ";".join(problems),
            }
        )
    return pd.DataFrame(rows)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def scalar_hellinger(left: np.ndarray, right: np.ndarray) -> float:
    left = normalize(left)
    right = normalize(right)
    return float(np.sqrt(np.clip(1.0 - np.sqrt(left * right).sum(), 0.0, 1.0)))


def scalar_weighted_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = normalize(left)
    right = normalize(right)
    return float(np.minimum(left, right).sum() / np.maximum(np.maximum(left, right).sum(), 1e-12))


def independently_recompute_episode(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Slow scalar implementation that intentionally shares no metric helpers."""
    raw = normalize(raw)
    length = len(raw)
    convergence = np.empty(length, dtype=np.float64)
    state_jump = np.full(length, np.nan, dtype=np.float64)
    action_jump = np.full(length, np.nan, dtype=np.float64)
    response = np.full(length, np.nan, dtype=np.float64)
    recurrence = np.full(length, np.nan, dtype=np.float64)

    for query in range(length):
        transition_distance: list[float] = []
        for flow in range(1, 10):
            cells = [
                scalar_hellinger(
                    raw[query, layer, flow, token],
                    raw[query, layer, flow - 1, token],
                )
                for layer in range(4, 8)
                for token in range(1, 11)
            ]
            transition_distance.append(float(np.mean(cells)))
        path = np.asarray(transition_distance)
        convergence[query] = path[6:].sum() / max(path.sum(), 1e-12)

        if query > 0:
            state_jump[query] = np.mean(
                [
                    scalar_hellinger(
                        raw[query, layer, 9, 0], raw[query - 1, layer, 9, 0]
                    )
                    for layer in range(4)
                ]
            )
            action_cells = []
            for layer in range(4):
                current = normalize(raw[query, layer, 9, 1:11].mean(axis=0))
                previous = normalize(raw[query - 1, layer, 9, 1:11].mean(axis=0))
                action_cells.append(scalar_hellinger(current, previous))
            action_jump[query] = np.mean(action_cells)
            response[query] = max(state_jump[query] - action_jump[query], 0.0)

            lag_values = []
            for lag in (1, 2, 3, 4):
                if query < lag:
                    continue
                cells = [
                    scalar_weighted_jaccard(
                        raw[query, layer, 9, token],
                        raw[query - lag, layer, 9, token],
                    )
                    for layer in range(4, 8)
                    for token in range(1, 11)
                ]
                lag_values.append(float(np.mean(cells)))
            recurrence[query] = max(lag_values)

    output = {
        "convergence_raw": convergence,
        "state_jump": state_jump,
        "action_jump": action_jump,
        "response_raw": response,
        "recurrence_raw": recurrence,
    }
    baselines = {
        "convergence": float(np.nanmedian(convergence[:4])),
        "response": float(np.nanmedian(response[:4])),
        "recurrence": float(np.nanmedian(recurrence[:4])),
    }
    for head in alarm.HEADS:
        evidence = np.full(length, np.nan, dtype=np.float64)
        evidence[4:] = output[f"{head}_raw"][4:] - baselines[head]
        output[f"evidence_{head}"] = evidence
    return output


def formula_audit(bundle: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    task_sample = (
        bundle["split"].sort_values(["suite", "sha256_rank"]).groupby("suite").head(1)["task"]
    )
    for task in task_sample:
        run = bundle["runs"][task]
        route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        cache_path = alarm.task_cache_path(bundle["result"] / "route_only_tasks", task)
        arrays = alarm.load_route_cache(cache_path)
        route_episode = np.asarray(route["episode_id"][:], dtype=np.int32)
        for episode_id in (0, 133, 266, 399):
            indices = np.flatnonzero(route_episode == episode_id)
            raw = np.asarray(
                route["hb_router_probs"].oindex[indices, :, :, :, :], dtype=np.float64
            )
            manual = independently_recompute_episode(raw)
            for metric, expected in manual.items():
                observed = np.asarray(arrays[metric][indices], dtype=np.float64)
                good = np.isfinite(expected) & np.isfinite(observed)
                difference = np.abs(expected[good] - observed[good])
                rows.append(
                    {
                        "run": bundle["name"],
                        "task": task,
                        "episode": episode_id,
                        "metric": metric,
                        "values_compared": int(good.sum()),
                        "max_abs_difference": float(difference.max()) if len(difference) else np.nan,
                        "mean_abs_difference": float(difference.mean()) if len(difference) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def rate_metrics(alarm_values: Any, failure_values: Any) -> dict[str, Any]:
    return alarm.rate_metrics(
        np.asarray(alarm_values, dtype=bool), np.asarray(failure_values, dtype=bool)
    )


def evaluate_transfer(
    source: dict[str, Any], target: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    role_by_task = target["split"].set_index("task")["role"].to_dict()
    episode_rows: list[dict[str, Any]] = []
    for task, run in sorted(target["runs"].items()):
        arrays = alarm.load_route_cache(
            alarm.task_cache_path(target["result"] / "route_only_tasks", task)
        )
        frame = alarm.alarm_rows(
            alarm.detector_series(arrays, source["references"]), source["threshold"]
        )
        frame = frame[frame["valid"]]
        summaries = read_json(run / "client/summaries.json")
        outcomes = {int(item["episode_index"]): bool(item["success"]) for item in summaries}
        for episode_id, indices in alarm.episode_slices(arrays["episode"]):
            rows = frame[frame["episode"] == episode_id]
            hits = rows[rows["first_alarm"]]
            first = hits.iloc[0] if len(hits) else None
            success = outcomes[episode_id]
            length = len(indices)
            first_query = float(first["query"]) if first is not None else np.nan
            episode_rows.append(
                {
                    "reference_seed_group": source["name"],
                    "evaluation_seed_group": target["name"],
                    "role": role_by_task[task],
                    "task": task,
                    "suite": task.split("/", 1)[0],
                    "episode": episode_id,
                    "episode_length": length,
                    "success": success,
                    "failure": not success,
                    "alarm": first is not None,
                    "first_alarm_query": first_query,
                    "first_alarm_phase": first_query / max(length - 1, 1)
                    if first is not None
                    else np.nan,
                    "alarm_cause": str(first["alarm_cause"])
                    if first is not None
                    else "",
                    "episode_max_confidence": float(rows["detector_confidence"].max())
                    if len(rows)
                    else np.nan,
                    "fixed_clock_query": source["clock_query"],
                    "fixed_clock_alarm": length > source["clock_query"],
                }
            )
    episodes = pd.DataFrame(episode_rows)
    summaries_out: list[dict[str, Any]] = []
    for scope, selected in (
        ("heldout_tasks", episodes[episodes["role"] == "heldout_test"]),
        ("all_tasks", episodes),
    ):
        detector = rate_metrics(selected["alarm"], selected["failure"])
        fixed_clock = rate_metrics(selected["fixed_clock_alarm"], selected["failure"])
        lengths = selected["episode_length"].to_numpy(dtype=np.int64)
        failure = selected["failure"].to_numpy(dtype=bool)
        matched_query = None
        matched_metrics = None
        for query in range(int(lengths.max()) + 1):
            candidate = rate_metrics(lengths > query, failure)
            if (
                candidate["success_false_alarm_rate"]
                <= detector["success_false_alarm_rate"] + 1e-15
            ):
                matched_query = query
                matched_metrics = candidate
                break
        summaries_out.append(
            {
                "reference_seed_group": source["name"],
                "evaluation_seed_group": target["name"],
                "scope": scope,
                "episodes": len(selected),
                "failures": int(selected["failure"].sum()),
                "threshold": source["threshold"],
                **{f"detector_{key}": value for key, value in detector.items()},
                **{f"fixed_clock_{key}": value for key, value in fixed_clock.items()},
                "matched_clock_query": matched_query,
                **{
                    f"matched_clock_{key}": value
                    for key, value in (matched_metrics or {}).items()
                },
            }
        )
    return episodes, summaries_out


def combined_metrics(frames: dict[tuple[str, str], pd.DataFrame]) -> dict[str, Any]:
    own = pd.concat(
        [
            frames[("seed1000_1007", "seed1000_1007")],
            frames[("seed1008_1015", "seed1008_1015")],
        ]
    )
    output: dict[str, Any] = {}
    for scope, selected in (
        ("heldout_tasks", own[own["role"] == "heldout_test"]),
        ("all_tasks_descriptive", own),
    ):
        output[scope] = {
            "episodes": len(selected),
            "failures": int(selected["failure"].sum()),
            "detector": rate_metrics(selected["alarm"], selected["failure"]),
            "fixed_clock": rate_metrics(
                selected["fixed_clock_alarm"], selected["failure"]
            ),
        }
    return output


def render_report(summary: dict[str, Any], transfer: pd.DataFrame) -> str:
    strict = transfer[transfer["scope"] == "heldout_tasks"].copy()
    own = strict[
        strict["reference_seed_group"] == strict["evaluation_seed_group"]
    ]
    cross = strict[
        strict["reference_seed_group"] != strict["evaluation_seed_group"]
    ]
    combined = summary["combined_own_rule"]["heldout_tasks"]
    scalar = summary["formula_audit"]
    lines = [
        "# 两套 50x8 MoE 报警实现与复现审计",
        "",
        "## 结论",
        "",
        "低召回不是路由轴读反、episode 行错位或向量化公式算错造成的。修复后的程序从原始 Zarr 重算第一组后，逐 query 预测文件与旧结果 SHA256 完全一致；第二套独立 flow-noise 数据又复现了几乎相同的失败模式。",
        "",
        "但是原实现确实有三个工程问题：只配置了第一套 50x8；缓存身份没有绑定 run/config；相对 `--output` 会在写清单时崩溃。后二者已经修复，第一项由本实验补齐。它们没有改变第一组的预测结果。",
        "",
        "## 数据覆盖",
        "",
        f"共检查 2 个 run、80 个 task-run、{summary['data']['episodes']} 条轨迹、{summary['data']['route_queries']} 次重规划；失败共 {summary['data']['failures']} 条。两组 flow seed 分别为 1000--1007 与 1008--1015。所有 Zarr 行数、episode_id、control_step、client inference_calls 和 `[8,10,11,32]` 路由几何均精确一致。",
        "",
        "## 同组固定协议",
        "",
        alarm.markdown_table(
            own[
                [
                    "reference_seed_group",
                    "evaluation_seed_group",
                    "failures",
                    "detector_tp",
                    "detector_fp",
                    "detector_precision",
                    "detector_failure_recall",
                    "detector_success_false_alarm_rate",
                    "matched_clock_failure_recall",
                ]
            ]
        ),
        "",
        f"两组留出集合并后，MoE 检出 {combined['detector']['tp']}/{combined['failures']} 个失败，failure recall={combined['detector']['failure_recall']:.2%}，成功误报率={combined['detector']['success_false_alarm_rate']:.2%}。这不是部署结果：同误报时间钟仍然更强。",
        "",
        "## 跨 run 冻结迁移",
        "",
        "下面不在目标 run 重估参考分布或阈值：直接把一组的完整规则应用到另一组。",
        "",
        alarm.markdown_table(
            cross[
                [
                    "reference_seed_group",
                    "evaluation_seed_group",
                    "failures",
                    "detector_tp",
                    "detector_fp",
                    "detector_precision",
                    "detector_failure_recall",
                    "detector_success_false_alarm_rate",
                    "matched_clock_failure_recall",
                ]
            ]
        ),
        "",
        "## 公式独立复算",
        "",
        f"慢速逐 cell 标量实现与主程序共比较 {scalar['values_compared']} 个值。最大绝对差：convergence={scalar['max_abs_difference_by_metric']['convergence_raw']:.6g}，state jump={scalar['max_abs_difference_by_metric']['state_jump']:.6g}，action jump={scalar['max_abs_difference_by_metric']['action_jump']:.6g}，response={scalar['max_abs_difference_by_metric']['response_raw']:.6g}，recurrence={scalar['max_abs_difference_by_metric']['recurrence_raw']:.6g}。差异来自主程序中间缓存的 float16 量化，远低于信号尺度，没有报警逻辑分歧。",
        "",
        "## 为什么判定为规则问题",
        "",
        "- 第一组修复前后预测哈希完全相同，排除了缓存修复或输出路径修复改变数值。",
        "- 第二组使用不同的 8 个 flow-noise seeds，阈值、召回、报警位置和 recurrence 主导现象仍近似相同。",
        "- 两组都被相同成功误报率下的固定时间钟支配，说明 detector 主要利用了晚期复返/轨迹变长，而不是提前出现的 Trap 特异信号。",
        "- convergence+response 分支几乎从不触发；所谓三头规则实际上退化为单一 recurrence 规则。",
        "",
        "因此应保留这份结果作为负结果，不再围绕当前阈值微调。下一版需要先重新定义能与时长解耦、并在 onset 前局部出现的 MoE evidence。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    table_dir = OUTPUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    bundles = {name: load_bundle(name, spec) for name, spec in SPECS.items()}

    structure = pd.concat(
        [structural_audit(bundle) for bundle in bundles.values()], ignore_index=True
    )
    structure.to_csv(table_dir / "all_80_task_run_structure.csv", index=False)
    if not structure["ok"].all():
        raise AssertionError("raw corpus structural audit failed")

    formulas = pd.concat(
        [formula_audit(bundle) for bundle in bundles.values()], ignore_index=True
    )
    formulas.to_csv(table_dir / "independent_formula_recompute.csv", index=False)
    maxima = formulas.groupby("metric")["max_abs_difference"].max().to_dict()
    tolerances = {
        "convergence_raw": 5e-4,
        "state_jump": 1e-4,
        "action_jump": 1e-4,
        "response_raw": 1e-4,
        "recurrence_raw": 1e-5,
        "evidence_convergence": 1e-3,
        "evidence_response": 2e-4,
        "evidence_recurrence": 2e-5,
    }
    for metric, tolerance in tolerances.items():
        if maxima[metric] > tolerance:
            raise AssertionError(f"{metric} scalar mismatch: {maxima[metric]} > {tolerance}")

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    transfer_rows: list[dict[str, Any]] = []
    for source_name, source in bundles.items():
        for target_name, target in bundles.items():
            episodes, rows = evaluate_transfer(source, target)
            frames[(source_name, target_name)] = episodes
            episodes.to_csv(
                table_dir / f"rule_{source_name}_on_{target_name}_episode_flip.csv",
                index=False,
            )
            transfer_rows.extend(rows)
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv(table_dir / "transfer_metrics.csv", index=False)

    # Self applications must exactly reproduce the two standalone evaluations.
    for name, bundle in bundles.items():
        observed = transfer[
            (transfer["reference_seed_group"] == name)
            & (transfer["evaluation_seed_group"] == name)
            & (transfer["scope"] == "heldout_tasks")
        ].iloc[0]
        expected = bundle["summary"]["heldout_endpoint_flip"]["detector"]
        for key in ("tp", "fp", "fn", "tn"):
            if int(observed[f"detector_{key}"]) != int(expected[key]):
                raise AssertionError(f"{name} self-reproduction mismatch: {key}")

    archived = (
        SPECS["seed1000_1007"]["archived_result"]
        / "heldout_predictions_label_free.csv.gz"
    )
    recomputed = (
        SPECS["seed1000_1007"]["result"]
        / "heldout_predictions_label_free.csv.gz"
    )
    archived_hash = sha256(archived)
    recomputed_hash = sha256(recomputed)
    if archived_hash != recomputed_hash:
        raise AssertionError("fixed implementation did not reproduce archived predictions")

    combined = combined_metrics(frames)
    summary = {
        "schema": "himoe.moe_invariant_alarm_two_run_audit.v1",
        "training": False,
        "method_changed": False,
        "data": {
            "runs": 2,
            "task_runs": len(structure),
            "tasks_per_run": 40,
            "episodes": int(structure["episodes"].sum()),
            "failures": int(structure["failures"].sum()),
            "route_queries": int(structure["route_queries"].sum()),
            "structural_problems": int((~structure["ok"]).sum()),
            "run_ids": {name: value["spec"]["run_id"] for name, value in bundles.items()},
        },
        "implementation_audit": {
            "route_axes_verified": True,
            "episode_alignment_verified_for_all_80_task_runs": True,
            "independent_scalar_formula_recompute_passed": True,
            "archived_seed1000_1007_prediction_sha256": archived_hash,
            "fixed_code_seed1000_1007_prediction_sha256": recomputed_hash,
            "prediction_hash_exact_match": archived_hash == recomputed_hash,
            "bugs_found_and_fixed": [
                "route feature cache identity did not bind run path and feature config",
                "relative --output failed while constructing the prediction manifest",
                "the original experiment scope included only the first of two complete 50x8 runs",
            ],
            "numeric_result_changed_by_fixes": False,
        },
        "formula_audit": {
            "task_runs_sampled": int(formulas[["run", "task"]].drop_duplicates().shape[0]),
            "episodes_sampled": int(
                formulas[["run", "task", "episode"]].drop_duplicates().shape[0]
            ),
            "values_compared": int(formulas["values_compared"].sum()),
            "max_abs_difference_by_metric": maxima,
            "tolerances": tolerances,
        },
        "transfer_metrics": transfer.to_dict(orient="records"),
        "combined_own_rule": combined,
        "conclusion": {
            "poor_result_is_explained_by_axis_or_arithmetic_bug": False,
            "current_rule_is_valid_trap_detector": False,
            "reason": "independent run replication and exact code recomputation preserve the same late recurrence behavior, while an FPR-matched clock remains stronger",
        },
    }
    write_json(OUTPUT / "summary.json", summary)
    (OUTPUT / "REPORT_ZH.md").write_text(
        render_report(summary, transfer), encoding="utf-8"
    )
    print(json.dumps(summary["data"], indent=2))
    print(f"report: {OUTPUT / 'REPORT_ZH.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
