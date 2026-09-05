#!/usr/bin/env python3
"""Family-wise permutation audit for all early structured-MoE readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_early_structure as core


HERE = Path(__file__).resolve().parent

EARLY_FAMILIES = {
    "route_identity": ("route_identity",),
    "route_geometry": ("route_geometry",),
    "route_temporal": ("route_temporal",),
    "route_current": core.FAMILIES["route_current"],
    "route_history": core.FAMILIES["route_history"],
    "hidden_identity": ("hidden_identity",),
    "hidden_structure": ("hidden_structure",),
    "hidden_temporal": ("hidden_temporal",),
    "hidden_current": core.FAMILIES["hidden_current"],
    "hidden_history": core.FAMILIES["hidden_history"],
    "moe_current": core.FAMILIES["moe_current"],
    "moe_history": core.FAMILIES["moe_history"],
    "route_state": core.FAMILIES["route_state"],
    "route_action_1_3": core.FAMILIES["route_action_1_3"],
    "route_action_4_7": core.FAMILIES["route_action_4_7"],
    "route_action_8_10": core.FAMILIES["route_action_8_10"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=core.RUN)
    parser.add_argument(
        "--cache", type=Path, default=HERE / "results" / "early_structure_features.npz"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--permutations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--target-mode",
        choices=("clean", "stasis-vs-rest"),
        default="clean",
        help="clean excludes non-stasis failures; stasis-vs-rest keeps them as negatives",
    )
    parser.add_argument(
        "--permutation-mode",
        choices=("within-state", "common-seed-column"),
        default="within-state",
    )
    parser.add_argument("--name", default="early_familywise")
    return parser.parse_args()


def permute_common_seed_column(
    labels: np.ndarray,
    state: np.ndarray,
    noise_seed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Move the same complete seed column in every initial state."""
    states = np.unique(state)
    seeds = np.unique(noise_seed)
    source = rng.permutation(seeds)
    output = np.empty_like(labels)
    for destination_seed, source_seed in zip(seeds, source):
        for state_value in states:
            destination = np.flatnonzero(
                (state == state_value) & (noise_seed == destination_seed)
            )
            origin = np.flatnonzero((state == state_value) & (noise_seed == source_seed))
            if len(destination) != 1 or len(origin) != 1:
                raise ValueError("expected a complete state x seed grid")
            output[destination[0]] = labels[origin[0]]
    return output


def main() -> None:
    args = parse_args()
    episodes = core.load_episodes(args.run)
    sim = core.load_sim(args.run, episodes)
    labels, included, _, target = core.build_targets(episodes, sim)
    if args.target_mode == "stasis-vs-rest":
        included = np.ones(len(episodes), dtype=bool)
    if args.permutation_mode == "common-seed-column" and not np.all(included):
        raise ValueError("common-seed-column permutation requires the complete 16 x 32 grid")
    blocks = core.load_or_extract(args.run, episodes, args.cache, False, args.seed)
    state = np.asarray([episode.state for episode in episodes], dtype=np.int16)
    noise_seed = np.asarray([episode.noise_seed for episode in episodes], dtype=np.int16)
    folds = core.double_holdout_folds(state, noise_seed, included)

    models = {}
    observed = {}
    for family_index, (family, names) in enumerate(EARLY_FAMILIES.items()):
        observed[family] = {}
        for horizon in core.EARLY_HORIZONS:
            horizon_index = core.HORIZONS.index(horizon)
            key = (family, horizon)
            models[key] = core.build_fold_models(
                blocks,
                names,
                horizon_index,
                folds,
                args.seed + 1_000_003 * family_index + 10_007 * horizon_index,
            )
            score = core.predict(models[key], labels)
            auc = core.state_auc(labels, score, state, included)[0]
            observed[family][str(horizon)] = {"auc": auc}
            print(f"{family:20s} t{horizon}: {auc:.3f}", flush=True)

    rng = np.random.default_rng(args.seed + 991)
    test_keys = list(models)
    null = np.empty((args.permutations, len(test_keys)), dtype=np.float32)
    for permutation in range(args.permutations):
        if args.permutation_mode == "within-state":
            shuffled = core.permute_within_state(labels, state, included, rng)
        else:
            shuffled = permute_common_seed_column(labels, state, noise_seed, rng)
        for test_index, key in enumerate(test_keys):
            score = core.predict(models[key], shuffled)
            null[permutation, test_index] = core.state_auc(
                shuffled, score, state, included
            )[0]
        if (permutation + 1) % 50 == 0:
            print(f"permutation {permutation + 1:04d}/{args.permutations}", flush=True)

    max_null = null.max(axis=1)
    for test_index, (family, horizon) in enumerate(test_keys):
        auc = observed[family][str(horizon)]["auc"]
        observed[family][str(horizon)].update(
            {
                "uncorrected_p": float(
                    (1 + np.sum(null[:, test_index] >= auc)) / (args.permutations + 1)
                ),
                "familywise_maxT_p": float(
                    (1 + np.sum(max_null >= auc)) / (args.permutations + 1)
                ),
            }
        )

    best_key = max(
        test_keys, key=lambda item: observed[item[0]][str(item[1])]["auc"]
    )
    summary = {
        "scope": {
            "families": list(EARLY_FAMILIES),
            "horizons": list(core.EARLY_HORIZONS),
            "tests": len(test_keys),
            "permutations": args.permutations,
            "target_mode": args.target_mode,
            "permutation": args.permutation_mode,
            "retraining": "model retrained in every double-holdout fold",
        },
        "target": target,
        "results": observed,
        "best_observed": {
            "family": best_key[0],
            "horizon": best_key[1],
            **observed[best_key[0]][str(best_key[1])],
        },
        "max_null_quantiles": {
            str(q): float(np.quantile(max_null, q)) for q in (0.5, 0.9, 0.95, 0.99)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.name}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    lines = [
        "# 早期 MoE 组织方式 family-wise 审计",
        "",
        f"范围：{len(EARLY_FAMILIES)} 种组织方式 × t7/t12，共 {len(test_keys)} 个检验；target={args.target_mode}，permutation={args.permutation_mode}，并在 16 个 init-state + seed 双留出折中重新训练。",
        "",
        "| 特征组织 | t7 AUC | t7 raw p | t7 maxT p | t12 AUC | t12 raw p | t12 maxT p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in EARLY_FAMILIES:
        r7 = observed[family]["7"]
        r12 = observed[family]["12"]
        lines.append(
            f"| {family} | {r7['auc']:.3f} | {r7['uncorrected_p']:.4f} | {r7['familywise_maxT_p']:.4f} "
            f"| {r12['auc']:.3f} | {r12['uncorrected_p']:.4f} | {r12['familywise_maxT_p']:.4f} |"
        )
    best = summary["best_observed"]
    lines.extend(
        [
            "",
            f"全族最强的是 `{best['family']}` at t{best['horizon']}，AUC={best['auc']:.3f}，raw p={best['uncorrected_p']:.4f}，32-test maxT p={best['familywise_maxT_p']:.4f}。",
            f"max-null 的 95% 分位数为 {summary['max_null_quantiles']['0.95']:.3f}。",
            "",
            "该审计回答的是看过全部 route/hidden/位置拆分后，最佳早期峰是否仍超过搜索零分布；它不改变主实验预先指定的 `moe_history` t7/t12 检验。",
            "",
        ]
    )
    (args.output_dir / f"{args.name}_report.md").write_text("\n".join(lines))
    np.savez_compressed(
        args.output_dir / f"{args.name}_null.npz",
        null=null,
        max_null=max_null,
        test=np.asarray([f"{family}:t{horizon}" for family, horizon in test_keys]),
    )
    print(f"wrote {args.output_dir / f'{args.name}_report.md'}", flush=True)


if __name__ == "__main__":
    main()
