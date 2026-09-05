#!/usr/bin/env python3
"""Compare the early hidden-state signal with emitted action-chunk history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_early_structure as core


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=core.RUN)
    parser.add_argument(
        "--cache", type=Path, default=HERE / "results" / "early_structure_features.npz"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def extract_action(run: Path, episodes: list[core.Episode]) -> np.ndarray:
    feature_dim = 10 * 7 * 5
    output = np.empty(
        (len(episodes), len(core.HORIZONS), feature_dim), dtype=np.float32
    )
    for episode in episodes:
        with np.load(core.episode_path(run, episode), allow_pickle=False) as payload:
            action = np.asarray(
                payload["actions"][: max(core.HORIZONS) + 1], dtype=np.float32
            ).reshape(-1, 10 * 7)
        for horizon_index, horizon in enumerate(core.HORIZONS):
            lo = horizon - core.HISTORY + 1
            output[episode.index, horizon_index] = core.physical_history(
                action[lo : horizon + 1]
            )
    return output


def main() -> None:
    args = parse_args()
    episodes = core.load_episodes(args.run)
    sim = core.load_sim(args.run, episodes)
    labels, _, _, target = core.build_targets(episodes, sim)
    included = np.ones(len(episodes), dtype=bool)
    blocks = core.load_or_extract(args.run, episodes, args.cache, False, args.seed)
    blocks["action"] = extract_action(args.run, episodes)
    state = np.asarray([episode.state for episode in episodes], dtype=np.int16)
    noise_seed = np.asarray([episode.noise_seed for episode in episodes], dtype=np.int16)
    folds = core.double_holdout_folds(state, noise_seed, included)
    families = {
        "action_history": ("action",),
        "hidden_identity": ("hidden_identity",),
        "hidden_history": core.FAMILIES["hidden_history"],
        "action_plus_hidden_identity": ("action", "hidden_identity"),
        "action_plus_hidden_history": (
            "action",
            "hidden_identity",
            "hidden_structure",
            "hidden_temporal",
        ),
        "sim_history": ("sim",),
    }
    result = {}
    for family_index, (family, names) in enumerate(families.items()):
        result[family] = {}
        for horizon_index, horizon in enumerate(core.HORIZONS):
            models = core.build_fold_models(
                blocks,
                names,
                horizon_index,
                folds,
                args.seed + 100_003 * family_index + 1_009 * horizon_index,
            )
            score = core.predict(models, labels)
            pair_auc, macro_auc, by_state = core.state_auc(
                labels, score, state, included
            )
            result[family][str(horizon)] = {
                "pair_weighted_within_state_auc": pair_auc,
                "macro_state_auc": macro_auc,
                "by_state_auc": {str(key): value for key, value in by_state.items()},
            }
            print(f"{family:28s} t{horizon}: {pair_auc:.3f}", flush=True)

    summary = {
        "target": "stasis trap versus success and non-stasis failure",
        "target_metadata": target,
        "cross_validation": "init-state + flow-noise-seed double holdout",
        "history": core.HISTORY,
        "results": result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "early_action_control_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    lines = [
        "# Early action-output control",
        "",
        "All values are pair-weighted within-initial-state AUC under init-state + seed double holdout. The target keeps non-stasis failures as negatives.",
        "",
        "| feature | t7 | t12 | t20 | t27 | t34 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in families:
        values = [
            result[family][str(horizon)]["pair_weighted_within_state_auc"]
            for horizon in core.HORIZONS
        ]
        lines.append(
            f"| {family} | " + " | ".join(f"{value:.3f}" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "The emitted action chunk is not a strong t12 baseline. Concatenating action and hidden blocks under the fixed linear readout does not improve on hidden alone, so this audit does not establish conditional hidden-state value beyond actions.",
            "",
        ]
    )
    (args.output_dir / "early_action_control_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
