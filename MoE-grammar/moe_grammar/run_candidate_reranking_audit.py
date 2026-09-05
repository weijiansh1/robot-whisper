from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from moe_grammar.candidate_reranking import (
    RouteArchive,
    clock_prefix_belief,
    cluster_bootstrap_mean,
    discover_run_traces,
    extract_query_words,
    filter_prefix,
    materialize_snapshots,
    pairwise_rms,
    selector_indexes,
    stratified_pair_auc,
)
from moe_grammar.features import step_feature_names
from moe_grammar.open_world import PHENOTYPE_NAMES


WORKSPACE = Path("/home/jovyan/work/himoe-vla")
RECOVERY_ROOT = WORKSPACE / "trap-recovery-depth-20260904"
DEFAULT_SOURCES = (
    ("formal80", RECOVERY_ROOT / "runs"),
    ("cap52", RECOVERY_ROOT / "runs_cap52"),
    ("k8_routed", RECOVERY_ROOT / "runs_routed"),
)

SELECTORS = (
    "action_medoid",
    "grammar",
    "grammar_action_diverse",
    "grammar_route_diverse",
    "anti_grammar",
    "grammar_clock",
    "grammar_local1",
    "grammar_no_prefix",
)


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be COHORT=PATH")
    cohort, raw_path = value.split("=", 1)
    if not cohort or not raw_path:
        raise argparse.ArgumentTypeError("source must be COHORT=PATH")
    return cohort, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        help="Candidate cohort as COHORT=PATH; repeatable. Defaults to the recovery datasets.",
    )
    parser.add_argument("--route-root", type=Path, default=RECOVERY_ROOT / "runs")
    parser.add_argument(
        "--models-dir", type=Path, default=Path("results-open-world-global-k12/models")
    )
    parser.add_argument(
        "--split-summary", type=Path, default=Path("results-open-world-global-k12/summary.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results-candidate-reranking"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diversity-beta", type=float, default=0.25)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def heldout_fold_map(path: Path) -> dict[int, int]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    output: dict[int, int] = {}
    for fold in summary["folds"]:
        fold_index = int(fold["fold"])
        for state in fold["test_states"]:
            state = int(state)
            if state in output:
                raise ValueError(f"init state {state} occurs in multiple test folds")
            output[state] = fold_index
    if set(output) != set(range(50)):
        raise ValueError("test folds do not partition all 50 init states")
    return output


def model_for_fold(models: dict[int, dict[str, Any]], directory: Path, fold: int) -> dict[str, Any]:
    if fold not in models:
        model = joblib.load(directory / f"fold_{fold}.joblib")
        if model.get("conditioning") != "global" or model.get("phase_states") != 12:
            raise ValueError("candidate audit requires the selected global K=12 grammar")
        models[fold] = model
    return models[fold]


def _predictive_mean(grammar: Any, previous_belief: np.ndarray | None) -> np.ndarray:
    predictive = (
        grammar.initial_
        if previous_belief is None
        else np.asarray(previous_belief) @ grammar.transition_
    )
    return predictive @ grammar.centers_


def score_snapshot(
    snapshot: Any,
    model: dict[str, Any],
    fold: int,
    device: str,
    diversity_beta: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = step_feature_names()
    prefix_base = extract_query_words(
        snapshot.prefix_probability,
        snapshot.prefix_expert_ids,
        names,
        device,
    )
    candidate_base = extract_query_words(
        snapshot.candidate_probability,
        snapshot.candidate_expert_ids,
        names,
        device,
    )
    scaler = model["scaler"]
    prefix_scaled = scaler.transform(prefix_base, np.zeros(len(prefix_base), dtype=np.int16))
    candidate_scaled = scaler.transform(
        candidate_base, np.zeros(len(candidate_base), dtype=np.int16)
    )
    grammar = model["prefix_grammar"]
    full_belief = filter_prefix(grammar, prefix_scaled)
    clock_belief = clock_prefix_belief(grammar, len(prefix_scaled))
    local_belief = filter_prefix(grammar, prefix_scaled[-1:])
    full_surprise, _, candidate_phase, candidate_entropy = grammar.score_candidates(
        candidate_scaled, full_belief
    )
    clock_surprise, _, _, _ = grammar.score_candidates(candidate_scaled, clock_belief)
    local_surprise, _, _, _ = grammar.score_candidates(candidate_scaled, local_belief)
    no_prefix_surprise, _, _, _ = grammar.score_candidates(candidate_scaled, None)
    indexes = selector_indexes(
        full_surprise,
        snapshot.actions,
        candidate_scaled,
        diversity_beta=diversity_beta,
    )
    indexes.update(
        grammar_clock=int(np.argmin(clock_surprise)),
        grammar_local1=int(np.argmin(local_surprise)),
        grammar_no_prefix=int(np.argmin(no_prefix_surprise)),
    )
    action_distance = pairwise_rms(snapshot.actions)
    route_distance = pairwise_rms(candidate_scaled)
    predictive_mean = _predictive_mean(grammar, full_belief)
    next_word_rms = np.sqrt(np.square(candidate_scaled - predictive_mean).mean(axis=1))
    snapshot_key = f"{snapshot.cluster}/{snapshot.arm}"

    candidate_rows = []
    for row, candidate_id in enumerate(snapshot.candidate_ids):
        record = {
            "snapshot": snapshot_key,
            "cluster": snapshot.cluster,
            "cohort": snapshot.cohort,
            "run_id": snapshot.run_id,
            "arm": snapshot.arm,
            "init_state_id": snapshot.init_state_id,
            "fold": fold,
            "fork_query": snapshot.fork_query,
            "budget": snapshot.budget,
            "candidate": int(candidate_id),
            "success": int(snapshot.success[row]),
            "grammar_surprise": float(full_surprise[row]),
            "clock_surprise": float(clock_surprise[row]),
            "local1_surprise": float(local_surprise[row]),
            "no_prefix_surprise": float(no_prefix_surprise[row]),
            "candidate_phase": float(candidate_phase[row]),
            "candidate_belief_entropy": float(candidate_entropy[row]),
            "next_word_rms": float(next_word_rms[row]),
            "action_novelty": float(action_distance[row].mean()),
            "route_novelty": float(route_distance[row].mean()),
        }
        record.update(
            {
                f"phenotype_{name}": float(candidate_base[row, column])
                for column, name in enumerate(PHENOTYPE_NAMES)
            }
        )
        candidate_rows.append(record)

    random_success = float(snapshot.success.mean())
    oracle_success = int(snapshot.success.any())
    snapshot_row: dict[str, Any] = {
        "snapshot": snapshot_key,
        "cluster": snapshot.cluster,
        "cohort": snapshot.cohort,
        "run_id": snapshot.run_id,
        "arm": snapshot.arm,
        "init_state_id": snapshot.init_state_id,
        "fold": fold,
        "fork_query": snapshot.fork_query,
        "budget": snapshot.budget,
        "candidates": len(snapshot.success),
        "candidate_successes": int(snapshot.success.sum()),
        "random_success": random_success,
        "oracle_success": oracle_success,
        "opportunity": int(0 < snapshot.success.sum() < len(snapshot.success)),
        "grammar_range": float(np.ptp(full_surprise)),
        "grammar_std": float(np.std(full_surprise)),
        "action_pair_rms": float(action_distance[np.triu_indices(len(action_distance), 1)].mean()),
        "route_pair_rms": float(route_distance[np.triu_indices(len(route_distance), 1)].mean()),
        "full_clock_top1_agree": int(np.argmin(full_surprise) == np.argmin(clock_surprise)),
        "snapshot_state_sha256": snapshot.snapshot_state_sha256,
        "policy_state_sha256": snapshot.policy_state_sha256,
        "duplicate_route_matches": snapshot.duplicate_route_matches,
    }
    for selector, index in indexes.items():
        snapshot_row[f"{selector}_candidate"] = int(snapshot.candidate_ids[index])
        snapshot_row[f"{selector}_success"] = int(snapshot.success[index])
    return candidate_rows, snapshot_row


def aggregate_selector(
    snapshots: pd.DataFrame,
    selector: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if selector == "random":
        selected = snapshots["random_success"].to_numpy(dtype=np.float64)
    elif selector == "oracle":
        selected = snapshots["oracle_success"].to_numpy(dtype=np.float64)
    else:
        selected = snapshots[f"{selector}_success"].to_numpy(dtype=np.float64)
    random = snapshots["random_success"].to_numpy(dtype=np.float64)
    oracle = snapshots["oracle_success"].to_numpy(dtype=np.float64)
    cluster = snapshots["cluster"].to_numpy()
    rate_low, rate_high = cluster_bootstrap_mean(
        selected, cluster, draws=draws, seed=seed + 17
    )
    uplift_low, uplift_high = cluster_bootstrap_mean(
        selected - random, cluster, draws=draws, seed=seed + 29
    )
    denominator = float(np.sum(oracle - random))
    opportunity = snapshots["opportunity"].to_numpy(dtype=bool)
    return {
        "selector": selector,
        "selected_successes": float(selected.sum()),
        "success_rate": float(selected.mean()),
        "success_rate_ci": [rate_low, rate_high],
        "uplift_vs_random": float(np.mean(selected - random)),
        "uplift_vs_random_ci": [uplift_low, uplift_high],
        "oracle_gap_closed": (
            float(np.sum(selected - random) / denominator) if denominator > 0 else None
        ),
        "opportunity_success_rate": (
            float(selected[opportunity].mean()) if opportunity.any() else None
        ),
    }


def bootstrap_auc(
    candidates: pd.DataFrame,
    score_column: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    score = candidates[score_column].to_numpy(dtype=np.float64)
    success = candidates["success"].to_numpy(dtype=bool)
    snapshot = candidates["snapshot"].to_numpy()
    cluster = candidates["cluster"].to_numpy()
    estimate = stratified_pair_auc(score, success, snapshot)
    clusters = np.unique(cluster)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        pieces = []
        for draw_index, sampled in enumerate(rng.choice(clusters, len(clusters), replace=True)):
            part = candidates[cluster == sampled].copy()
            part["snapshot"] = part["snapshot"].astype(str) + f"/boot{draw_index}"
            pieces.append(part)
        sample = pd.concat(pieces, ignore_index=True)
        value = stratified_pair_auc(
            sample[score_column].to_numpy(),
            sample["success"].to_numpy(dtype=bool),
            sample["snapshot"].to_numpy(),
        )
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high)


def cluster_signflip_pvalue(candidates: pd.DataFrame, score_column: str) -> tuple[int, float]:
    effects = []
    for cluster in candidates["cluster"].unique():
        selected = candidates[candidates["cluster"] == cluster]
        auc = stratified_pair_auc(
            selected[score_column].to_numpy(dtype=np.float64),
            selected["success"].to_numpy(dtype=bool),
            selected["snapshot"].to_numpy(),
        )
        if np.isfinite(auc):
            effects.append(auc - 0.5)
    if not effects:
        return 0, float("nan")
    effects_array = np.asarray(effects, dtype=np.float64)
    observed = float(effects_array.mean())
    if len(effects) > 20:
        return len(effects), float("nan")
    null = np.asarray(
        [
            np.mean(effects_array * np.asarray(signs, dtype=np.float64))
            for signs in itertools.product((-1.0, 1.0), repeat=len(effects))
        ]
    )
    return len(effects), float(np.mean(null >= observed - 1e-12))


def cohort_summary(
    snapshots: pd.DataFrame,
    candidates: pd.DataFrame,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    selector_results = [
        aggregate_selector(snapshots, selector, draws, seed + index * 101)
        for index, selector in enumerate(("random", *SELECTORS, "oracle"))
    ]
    score_columns = {
        "full_prefix": "grammar_surprise",
        "clock_only": "clock_surprise",
        "last_query_only": "local1_surprise",
        "no_prefix": "no_prefix_surprise",
        "predictive_mean_rms": "next_word_rms",
    }
    auc_results = {}
    for index, (name, column) in enumerate(score_columns.items()):
        auc, auc_low, auc_high = bootstrap_auc(
            candidates, column, draws, seed + 701 + index * 31
        )
        auc_results[name] = {"auc": auc, "ci": [auc_low, auc_high]}
    informative_trunks, signflip_p = cluster_signflip_pvalue(
        candidates, "grammar_surprise"
    )
    full_auc = auc_results["full_prefix"]
    return {
        "trunks": int(snapshots["cluster"].nunique()),
        "snapshots": int(len(snapshots)),
        "candidates": int(len(candidates)),
        "opportunity_snapshots": int(snapshots["opportunity"].sum()),
        "candidate_successes": int(candidates["success"].sum()),
        "informative_trunks": informative_trunks,
        "within_snapshot_auc_by_score": auc_results,
        "full_prefix_one_sided_cluster_signflip_p": signflip_p,
        "within_snapshot_grammar_auc": full_auc["auc"],
        "within_snapshot_grammar_auc_ci": full_auc["ci"],
        "full_clock_top1_agreement": float(snapshots["full_clock_top1_agree"].mean()),
        "median_grammar_range": float(snapshots["grammar_range"].median()),
        "median_action_pair_rms": float(snapshots["action_pair_rms"].median()),
        "median_route_pair_rms": float(snapshots["route_pair_rms"].median()),
        "selectors": selector_results,
    }


def render_report(summary: dict[str, Any]) -> str:
    primary = summary["analyses"]["primary_k4_budget16"]
    selectors = {row["selector"]: row for row in primary["selectors"]}
    grammar = selectors["grammar"]
    random = selectors["random"]
    medoid = selectors["action_medoid"]
    oracle = selectors["oracle"]
    auc_scores = primary["within_snapshot_auc_by_score"]

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.1f}%"

    lines = [
        "# 第三阶段：健康语法候选重排审计",
        "",
        "## 结论",
        "",
        (
            f"主分析包含 {primary['trunks']} 条独立失败 trunk、"
            f"{primary['snapshots']} 个同状态 fork snapshot、"
            f"{primary['candidates']} 个候选；其中只有 "
            f"{primary['opportunity_snapshots']} 个 snapshot 同时含成功和失败候选。"
        ),
        "",
        (
            f"full-prefix grammar 选择成功率为 {percent(grammar['success_rate'])}，"
            f"随机选择期望为 {percent(random['success_rate'])}，"
            f"action medoid 为 {percent(medoid['success_rate'])}，"
            f"best-of-K oracle 为 {percent(oracle['success_rate'])}。"
        ),
        "",
        (
            f"grammar 相对随机的绝对变化为 {percent(grammar['uplift_vs_random'])}，"
            f"trunk-cluster bootstrap 95% CI "
            f"[{percent(grammar['uplift_vs_random_ci'][0])}, "
            f"{percent(grammar['uplift_vs_random_ci'][1])}]；"
            f"在真正有选择机会的 snapshot 上命中成功候选的比例为 "
            f"{percent(grammar['opportunity_success_rate'])}。"
        ),
        "",
        (
            f"候选内（只比较同一 snapshot 的成功/失败对）grammar AUC 为 "
            f"{primary['within_snapshot_grammar_auc']:.3f}，95% CI "
            f"[{primary['within_snapshot_grammar_auc_ci'][0]:.3f}, "
            f"{primary['within_snapshot_grammar_auc_ci'][1]:.3f}]。"
        ),
        "",
        (
            "但 full-prefix / clock-only / last-query-only / no-prefix AUC 分别为 "
            f"{auc_scores['full_prefix']['auc']:.3f} / "
            f"{auc_scores['clock_only']['auc']:.3f} / "
            f"{auc_scores['last_query_only']['auc']:.3f} / "
            f"{auc_scores['no_prefix']['auc']:.3f}。"
        ),
        "",
        (
            f"真正提供成功/失败配对的独立 trunk 只有 {primary['informative_trunks']} 条；"
            "以 trunk 为单位的单边精确 sign-flip p="
            f"{primary['full_prefix_one_sided_cluster_signflip_p']:.3f}。"
        ),
        "",
        "因此当前正向信号属于候选当前 query 的健康 emission compatibility；"
        "没有证据表明完整前缀比无前缀或内部时钟提供了额外的候选级排序信息。",
        "",
        "## 实现与防泄漏",
        "",
        "- 每个候选只使用共享 trunk 的全部 MoE 前缀和候选当前 query 的 22 维多轨道表型。",
        "- HMM forward belief 压缩截止当前的完整前缀；候选由同一个冻结 belief 并行打分，互不更新。",
        "- 每个 init state 使用将该状态置于测试集的健康语法 fold；训练只含成功 episode。",
        "- success 只在选择完成后用于评估；oracle 是唯一读取 outcome 的上界。",
        "- 所有候选的 `sim_state[0]` 与 `policy_state[0]` 均逐位一致；route IDs 逐 query 回查服务端 Zarr，"
        "再读取完整 32-way gate 概率。",
        "- 置信区间按 trunk 重抽，而非把同一 trunk 的多个 fork 点伪装成独立样本。",
        "",
        "## 选择器",
        "",
        "| selector | success | uplift vs random | opportunity hit | oracle gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary["selectors"]:
        lines.append(
            f"| {row['selector']} | {percent(row['success_rate'])} | "
            f"{percent(row['uplift_vs_random'])} | "
            f"{percent(row['opportunity_success_rate'])} | "
            f"{percent(row['oracle_gap_closed'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是候选级、同状态、严格因果输入的离线反事实审计，但样本量仍小。"
            "绝对成功率会受 fork 时机与 16-query 预算影响，因此核心证据是同 snapshot 的配对排序，"
            "不是跨 snapshot 的原始分数相关。",
            "",
            "`k8_routed` 只有一条 trunk，只作为复现实例，不并入主置信区间。"
            "没有成功候选的 snapshot 可检验分数稳定性，但不能检验选择能力。",
            "",
            "完整逐候选分数见 `candidate_scores.csv`，逐 snapshot 选择见 "
            "`snapshot_selections.csv`，机器可读汇总见 `summary.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws <= 0 or args.diversity_beta < 0:
        raise ValueError("bootstrap draws must be positive and diversity beta non-negative")
    started = time.time()
    sources = args.source or list(DEFAULT_SOURCES)
    route_paths = list(args.route_root.glob("_server*/routes.zarr"))
    direct = args.route_root / "_server" / "routes.zarr"
    if direct.exists():
        route_paths.append(direct)
    archive = RouteArchive(route_paths)
    fold_map = heldout_fold_map(args.split_summary)
    models: dict[int, dict[str, Any]] = {}
    all_candidate_rows: list[dict[str, Any]] = []
    all_snapshot_rows: list[dict[str, Any]] = []
    located_runs = 0
    for cohort, source in sources:
        runs = discover_run_traces(cohort, source)
        print(f"{cohort}: {len(runs)} routed completed failure trunks", flush=True)
        for run in runs:
            located = archive.locate(run)
            snapshots = materialize_snapshots(located)
            fold = fold_map[run.init_state_id]
            model = model_for_fold(models, args.models_dir, fold)
            for snapshot in snapshots:
                candidate_rows, snapshot_row = score_snapshot(
                    snapshot,
                    model,
                    fold,
                    args.device,
                    args.diversity_beta,
                )
                all_candidate_rows.extend(candidate_rows)
                all_snapshot_rows.append(snapshot_row)
            located_runs += 1
            print(
                f"  {run.run_id}: fold={fold} snapshots={len(snapshots)} "
                f"route_matches={located.duplicate_matches}",
                flush=True,
            )
    if not all_snapshot_rows:
        raise ValueError("no routed candidate snapshots were found")
    candidates = pd.DataFrame(all_candidate_rows)
    snapshots = pd.DataFrame(all_snapshot_rows)
    primary_mask = (snapshots["candidates"] == 4) & (snapshots["budget"] == 16)
    primary_keys = set(snapshots.loc[primary_mask, "snapshot"])
    analyses: dict[str, Any] = {
        "primary_k4_budget16": cohort_summary(
            snapshots[primary_mask].reset_index(drop=True),
            candidates[candidates["snapshot"].isin(primary_keys)].reset_index(drop=True),
            args.bootstrap_draws,
            args.seed,
        )
    }
    for cohort in sorted(snapshots["cohort"].unique()):
        selected_snapshots = snapshots[snapshots["cohort"] == cohort].reset_index(drop=True)
        keys = set(selected_snapshots["snapshot"])
        analyses[cohort] = cohort_summary(
            selected_snapshots,
            candidates[candidates["snapshot"].isin(keys)].reset_index(drop=True),
            args.bootstrap_draws,
            args.seed + sum(map(ord, cohort)),
        )
    summary = {
            "schema_version": 2,
        "protocol": {
            "definition": "causal-full-prefix-candidate-reranking-v2",
            "grammar": "global healthy-only 12-phase Gaussian HMM",
            "fold_assignment": "candidate init state is held out by its test-state fold",
            "candidate_input": "shared routing prefix plus candidate current-query routing only",
            "outcome_input": "evaluation only",
            "diversity_beta": args.diversity_beta,
            "bootstrap_unit": "trunk",
            "bootstrap_draws": args.bootstrap_draws,
            "sources": {cohort: str(path) for cohort, path in sources},
            "route_stores": [str(store.path) for store in archive.stores],
        },
        "counts": {
            "located_runs": located_runs,
            "snapshots": int(len(snapshots)),
            "candidates": int(len(candidates)),
        },
        "analyses": analyses,
        "elapsed_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "candidate_scores.csv", index=False)
    snapshots.to_csv(args.output_dir / "snapshot_selections.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.zh.md").write_text(render_report(summary), encoding="utf-8")
    print(
        f"wrote {args.output_dir}: {len(snapshots)} snapshots, "
        f"{len(candidates)} candidates in {time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
