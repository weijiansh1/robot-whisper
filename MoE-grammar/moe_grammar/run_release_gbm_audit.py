"""Can a gradient-boosted model localize the release moment that a threshold could not?

A single calibrated threshold on state-token mobility reached only 14.3% hit rate at an
18.5% success false-alarm rate, against 7.7% for firing at a random query. The signal is
real but not per-query decidable. This audit asks whether a nonlinear combination of every
routing view extracted in this directory does better.

The task is deliberately posed inside the failure set: for episodes with an annotated
release, each query is positive when it lies within +/-2 of that release and negative
otherwise. Every episode here runs to its run horizon, so episode length -- the confound
that dominates every uncontrolled number on this corpus -- is constant and cannot be
learned. No outcome label enters the features or the target.

Splits are blocked by initial state, so the sibling noise branches of a state never
straddle the train/test boundary. Two nulls are reported: labels permuted within episode,
which keeps the feature distribution and destroys only the timing, and the random-query
rate from the threshold audit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

LABELS = Path(
    "/home/jovyan/work/himoe-vla/VLA_MUI_HUB/physical-failure-labels/results/failures.jsonl"
)
WINDOW = 2
LAGS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-dir", type=Path, default=Path("artifacts/token-matching"))
    parser.add_argument("--mobility-dir", type=Path, default=Path("artifacts/state-action-mobility"))
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-release-gbm"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--rounds", type=int, default=400)
    return parser.parse_args()


def load_release(path: Path) -> dict[tuple[str, int, int], int]:
    output: dict[tuple[str, int, int], int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["data_root"] != "cache_new":
            continue
        for physics in (record.get("goal_subject_physics") or {}).values():
            snapshot = physics.get("first_release_snapshot")
            if snapshot is None:
                continue
            key = (record["task_name"], record["init_state_id"], record["flow_noise_seed"])
            output[key] = int(snapshot)
            break
    return output


def causal_lags(values: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Append lagged copies and first differences, padded at episode starts."""
    blocks = [values]
    for lag in LAGS:
        shifted = np.zeros_like(values)
        for start, length in zip(starts, lengths):
            if length > lag:
                shifted[start + lag : start + length] = values[start : start + length - lag]
                shifted[start : start + lag] = values[start]
            else:
                shifted[start : start + length] = values[start]
        blocks.append(shifted)
        blocks.append(values - shifted)
    return np.concatenate(blocks, axis=1)


def build(args: argparse.Namespace) -> dict[str, Any]:
    release = load_release(LABELS)
    rows_x, rows_y, rows_state, rows_episode, rows_query = [], [], [], [], []
    names: list[str] | None = None
    for path in sorted(glob.glob(str(args.matching_dir / "*.npz"))):
        stem = os.path.basename(path)
        matching = np.load(path, allow_pickle=False)
        mobility = np.load(args.mobility_dir / stem, allow_pickle=False)
        outcome = np.load(args.features_dir / stem, allow_pickle=False)
        task = json.loads(str(matching["metadata_json"].item()))["run_key"].split("/")[-1]
        episode_id = matching["episode_id"]
        step = matching["episode_step"]
        state_move = mobility["state_move"].reshape(len(episode_id), -1)
        action_move = mobility["action_move"].reshape(len(episode_id), -1)
        block = np.hstack(
            [
                matching["features"],
                np.nan_to_num(state_move, nan=0.0),
                np.nan_to_num(action_move, nan=0.0),
                outcome["behavior_features"],
            ]
        ).astype(np.float32)
        if names is None:
            names = (
                [str(x) for x in matching["feature_names"]]
                + [f"state_move_{i}" for i in range(state_move.shape[1])]
                + [f"action_move_{i}" for i in range(action_move.shape[1])]
                + [str(x) for x in outcome["behavior_feature_names"]]
            )
        for episode in np.unique(episode_id):
            selected = np.flatnonzero(episode_id == episode)
            if outcome["success"][selected[0]]:
                continue
            key = (
                task,
                int(outcome["init_state_id"][selected[0]]),
                int(outcome["flow_noise_seed"][selected[0]]),
            )
            moment = release.get(key)
            if moment is None:
                continue
            queries = step[selected]
            rows_x.append(block[selected])
            rows_y.append((np.abs(queries - moment) <= WINDOW).astype(np.int8))
            rows_state.append(np.full(len(selected), key[1], dtype=np.int16))
            rows_episode.append(np.full(len(selected), len(rows_episode), dtype=np.int32))
            rows_query.append(queries.astype(np.int16))
    starts = np.cumsum([0] + [len(x) for x in rows_x[:-1]])
    lengths = np.asarray([len(x) for x in rows_x])
    design = causal_lags(np.vstack(rows_x), starts, lengths)
    full_names = list(names or [])
    full_names = (
        full_names
        + [f"{n}|lag{lag}" for lag in LAGS for n in (names or [])]
        + [f"{n}|d{lag}" for lag in LAGS for n in (names or [])]
    )
    order = [0] + [i for i in range(1, len(full_names))]
    del order
    return {
        "x": design,
        "y": np.concatenate(rows_y),
        "state": np.concatenate(rows_state),
        "episode": np.concatenate(rows_episode),
        "query": np.concatenate(rows_query),
        "names": full_names,
        "episodes": len(rows_x),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("building design matrix...", flush=True)
    data = build(args)
    x, y = data["x"], data["y"]
    print(
        f"{data['episodes']} 个带脱手标注的失败 episode, {len(y):,} 个 query 行, "
        f"正类 {y.mean():.2%}, 特征 {x.shape[1]}",
        flush=True,
    )

    states = np.unique(data["state"])
    order = np.random.default_rng(args.seed).permutation(states)
    folds = [order[index :: args.folds] for index in range(args.folds)]

    rng = np.random.default_rng(args.seed)
    permuted = y.copy()
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        permuted[rows] = y[rng.permutation(rows)]

    # Ablations. The behaviour columns are proprioception and the policy's own action,
    # not MoE, so a routing claim needs them removed. And the release moment has a
    # characteristic time, so the query index alone is the baseline the routing must beat.
    names = data["names"]
    behaviour = {"action_arm_rms","action_arm_token_std","action_mean_delta","proprio_delta","gripper_flip_fraction"}
    is_beh = np.asarray([n.split("|")[0] in behaviour for n in names])
    moe_only = np.flatnonzero(~is_beh)
    clock = data["query"].astype(np.float32)[:, None]
    variants = {
        "all": x,
        "moe_only": x[:, moe_only],
        "behaviour_only": x[:, np.flatnonzero(is_beh)],
        "clock_only": clock,
        "clock_plus_moe": np.hstack([clock, x[:, moe_only]]),
    }
    print("消融特征数: " + ", ".join(f"{k}={v.shape[1]}" for k,v in variants.items()), flush=True)
    results: dict[str, list[float]] = {"real_auc": [], "real_ap": [], "null_auc": [], "null_ap": []}
    ablation: dict[str, list[float]] = {k: [] for k in variants}
    ablation_top1: dict[str, np.ndarray] = {k: np.zeros(len(y)) for k in variants}
    predictions = np.zeros(len(y))
    for fold, test_states in enumerate(folds):
        test = np.isin(data["state"], test_states)
        train = ~test
        for tag, target, store_auc, store_ap in (
            ("real", y, "real_auc", "real_ap"),
            ("null", permuted, "null_auc", "null_ap"),
        ):
            model = lgb.train(
                {
                    "objective": "binary",
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "min_data_in_leaf": 100,
                    "feature_fraction": 0.5,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "verbose": -1,
                    "seed": args.seed + fold,
                    # This table is ~31k rows; handing it 90 threads makes the
                    # synchronisation cost dominate the split search.
                    "num_threads": 16,
                    "force_row_wise": True,
                },
                lgb.Dataset(x[train], label=target[train]),
                num_boost_round=args.rounds,
            )
            score = model.predict(x[test])
            results[store_auc].append(roc_auc_score(target[test], score))
            results[store_ap].append(average_precision_score(target[test], score))
            if tag == "real":
                predictions[test] = score
        for vname, vx in variants.items():
            vm = lgb.train(
                {"objective":"binary","learning_rate":0.05,"num_leaves":63,"min_data_in_leaf":100,
                 "feature_fraction":0.5 if vx.shape[1]>4 else 1.0,"bagging_fraction":0.8,
                 "bagging_freq":1,"verbose":-1,"seed":args.seed+fold,"num_threads":16,
                 "force_row_wise":True},
                lgb.Dataset(vx[train], label=y[train]), num_boost_round=args.rounds)
            sc = vm.predict(vx[test])
            ablation[vname].append(roc_auc_score(y[test], sc))
            ablation_top1[vname][test] = sc
        print(
            f"fold {fold}: real AUC {results['real_auc'][-1]:.4f} AP {results['real_ap'][-1]:.4f} | "
            f"null AUC {results['null_auc'][-1]:.4f} AP {results['null_ap'][-1]:.4f}",
            flush=True,
        )

    hits = 0
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        hits += int(y[rows][np.argmax(predictions[rows])] == 1)
    top1 = {}
    for vname, sc in ablation_top1.items():
        h = 0
        for episode in np.unique(data["episode"]):
            rows = np.flatnonzero(data["episode"] == episode)
            h += int(y[rows][np.argmax(sc[rows])] == 1)
        top1[vname] = h / data["episodes"]
    summary = {
        "ablation_auc": {k: float(np.mean(v)) for k, v in ablation.items()},
        "ablation_top1": top1,
        "episodes": data["episodes"],
        "queries": int(len(y)),
        "positive_rate": float(y.mean()),
        "features": int(x.shape[1]),
        "real_auc": float(np.mean(results["real_auc"])),
        "real_ap": float(np.mean(results["real_ap"])),
        "null_auc": float(np.mean(results["null_auc"])),
        "null_ap": float(np.mean(results["null_ap"])),
        "top1_localisation": hits / data["episodes"],
        "top1_chance": float(y.mean()),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"\n真实 AUC {summary['real_auc']:.4f}  AP {summary['real_ap']:.4f}  "
        f"(正类基率 {summary['positive_rate']:.4f})\n"
        f"标签置换 AUC {summary['null_auc']:.4f}  AP {summary['null_ap']:.4f}\n"
        f"逐 episode 取最高分那一步落在 ±{WINDOW} 内: {summary['top1_localisation']:.1%} "
        f"(随机 {summary['top1_chance']:.1%})"
    )
    print(f"\n{'消融':18s} {'AUC':>8s} {'top1 定位':>10s}")
    for k in variants:
        print(f"{k:18s} {summary['ablation_auc'][k]:8.4f} {top1[k]:10.1%}")


if __name__ == "__main__":
    main()
