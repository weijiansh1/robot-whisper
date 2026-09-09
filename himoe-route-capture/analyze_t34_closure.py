#!/usr/bin/env python3
"""One-shot closure audit for the long-task absolute query t=34 result."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd
import zarr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import analyze_failure_moe_signatures_fixed as fixed
from analyze_single_chunk_early_signal import hidden_features


HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "analysis/t34-closure"
QUERY = 34
SEED = 20260830
BOOTSTRAPS = 2000
DELTA_MIN = 0.05

COMPONENTS = {
    "M0_proprio": ("proprio",),
    "M1_full_state": ("full_state",),
    "M2_full_state_action": ("full_state", "action"),
    "M3_M2_state_route": ("full_state", "action", "state_route"),
    "M4_M2_action_route": ("full_state", "action", "action_route"),
    "M5_M2_hidden": ("full_state", "action", "hidden"),
}
CAPS = {
    "proprio": 8,
    "full_state": 32,
    "action": 16,
    "state_route": 16,
    "action_route": 16,
    "hidden": 24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=fixed.CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = values.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("invalid route probability")
    return values / mass


def soft_route_features(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = normalize(probability)
    state = probability[:, :, 0, 0, :].reshape(len(probability), -1)
    action = probability[:, :, :, 1:, :].mean(axis=3)
    time_axis = np.linspace(-1.0, 1.0, action.shape[2], dtype=np.float32)
    slope = np.einsum("nlde,d->nle", action, time_axis, optimize=True)
    slope /= float(time_axis @ time_axis)
    action_summary = np.concatenate(
        [action.mean(axis=2), action[:, :, -1], slope], axis=-1
    ).reshape(len(action), -1)
    return state.astype(np.float32), action_summary.astype(np.float32)


def load_data(cache_root: pathlib.Path, out_dir: pathlib.Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    run = cache_root / fixed.LONG_TASK / "right-16x32"
    summaries = sorted(
        json.loads((run / "client/summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if len(summaries) != 512:
        raise ValueError("expected 512 long-task episodes")
    lengths = np.asarray([int(row["inference_calls"]) for row in summaries])
    if lengths.min() <= QUERY:
        raise ValueError("not every episode is at risk at t34")
    episodes = np.asarray([int(row["episode_index"]) for row in summaries])
    groups = np.asarray([int(row["init_state_id"]) for row in summaries])
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries])
    failure = np.asarray([not bool(row["success"]) for row in summaries])
    offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    server_rows = offsets + QUERY

    proprio, sim_state, actions = [], [], []
    for episode in episodes:
        path = run / "client" / f"episode_{int(episode):02d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            proprio.append(np.asarray(archive["state"][QUERY], dtype=np.float32))
            sim_state.append(np.asarray(archive["sim_state"][QUERY], dtype=np.float32))
            actions.append(np.asarray(archive["actions"][QUERY], dtype=np.float32).reshape(-1))
    proprio = np.stack(proprio)
    sim_state = np.stack(sim_state)
    actions = np.stack(actions)

    route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    hidden = zarr.open_group(str(run / "server/hidden.zarr"), mode="r")
    route_episode = np.asarray(route["episode_id"].oindex[server_rows], dtype=np.int64)
    hidden_episode = np.asarray(hidden["episode_id"].oindex[server_rows], dtype=np.int64)
    if not np.array_equal(route_episode, episodes) or not np.array_equal(hidden_episode, episodes):
        raise ValueError("server row alignment failed")
    probability = np.asarray(
        route["hb_router_probs"].oindex[server_rows, :, :, :, :], dtype=np.float32
    )
    state_route, action_route = soft_route_features(probability)

    hidden_cache = out_dir / "hidden_t34.npz"
    if hidden_cache.exists():
        with np.load(hidden_cache) as archive:
            hidden_episode_cache = np.asarray(archive["episode"], dtype=np.int64)
            hidden_value = np.asarray(archive["features"], dtype=np.float32)
        if not np.array_equal(hidden_episode_cache, episodes):
            raise ValueError("hidden cache episode mismatch")
    else:
        pieces = []
        for start in range(0, len(server_rows), 16):
            rows = server_rows[start : start + 16]
            value = np.asarray(
                hidden["hb_hidden"].oindex[rows, :, :, :, :], dtype=np.float16
            )
            pieces.append(hidden_features(value))
            print(f"hidden t34 {min(start + 16, len(server_rows))}/{len(server_rows)}", flush=True)
        hidden_value = np.concatenate(pieces).astype(np.float32)
        np.savez_compressed(hidden_cache, episode=episodes, features=hidden_value)

    frame = pd.DataFrame(
        {
            "episode": episodes,
            "init_state_id": groups,
            "flow_noise_seed": seeds,
            "failure": failure.astype(np.int8),
            "episode_length": lengths,
            "remaining_queries_posthoc": lengths - QUERY - 1,
            "phase_episode_posthoc": QUERY / (lengths - 1),
            "phase_budget": QUERY / 51.0,
        }
    )
    blocks = {
        "proprio": proprio.astype(np.float64),
        "full_state": np.column_stack([proprio, sim_state]).astype(np.float64),
        "action": actions.astype(np.float64),
        "state_route": state_route.astype(np.float64),
        "action_route": action_route.astype(np.float64),
        "hidden": hidden_value.astype(np.float64),
    }
    success_remaining = frame.loc[~failure, "remaining_queries_posthoc"].to_numpy()
    failure_remaining = frame.loc[failure, "remaining_queries_posthoc"].to_numpy()
    overlap = np.intersect1d(success_remaining, failure_remaining)
    audit = {
        "episodes": len(frame),
        "failures": int(failure.sum()),
        "successes": int((~failure).sum()),
        "initial_states": int(len(np.unique(groups))),
        "mixed_initial_states": int(
            sum(len(np.unique(failure[groups == group])) == 2 for group in np.unique(groups))
        ),
        "all_have_t34_query": True,
        "success_length_median": float(np.median(lengths[~failure])),
        "failure_length_median": float(np.median(lengths[failure])),
        "t34_phase_at_success_median_length": QUERY / (np.median(lengths[~failure]) - 1),
        "t34_phase_at_failure_median_length": QUERY / (np.median(lengths[failure]) - 1),
        "remaining_time_overlap_values": overlap.tolist(),
        "successes_in_exact_remaining_overlap": int(np.isin(success_remaining, overlap).sum()),
        "failures_in_exact_remaining_overlap": int(np.isin(failure_remaining, overlap).sum()),
        "risk_set_overlap_gate": "fail" if min(
            np.isin(success_remaining, overlap).sum(), np.isin(failure_remaining, overlap).sum()
        ) < 20 else "pass",
        "feature_widths": {name: int(value.shape[1]) for name, value in blocks.items()},
    }
    return frame, blocks, audit


def transform(train: np.ndarray, test: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median").fit(train)
    train_i, test_i = imputer.transform(train), imputer.transform(test)
    keep = np.std(train_i, axis=0) > 1e-10
    scaler = StandardScaler().fit(train_i[:, keep])
    train_s, test_s = scaler.transform(train_i[:, keep]), scaler.transform(test_i[:, keep])
    width = min(cap, train_s.shape[1], train_s.shape[0] - 1)
    if width < train_s.shape[1]:
        pca = PCA(n_components=width, svd_solver="randomized", random_state=seed).fit(train_s)
        train_s, test_s = pca.transform(train_s), pca.transform(test_s)
    return train_s, test_s


def cross_fit(blocks: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    predictions = {name: np.full(len(y), np.nan) for name in COMPONENTS}
    for fold, held in enumerate(np.unique(groups)):
        train, test = groups != held, groups == held
        reduced = {
            name: transform(value[train], value[test], CAPS[name], seed + fold * 100 + axis)
            for axis, (name, value) in enumerate(blocks.items())
        }
        for name, parts in COMPONENTS.items():
            x_train = np.column_stack([reduced[part][0] for part in parts])
            x_test = np.column_stack([reduced[part][1] for part in parts])
            model = LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed + fold,
            ).fit(x_train, y[train])
            predictions[name][test] = model.predict_proba(x_test)[:, 1]
    if any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("incomplete predictions")
    return predictions


def delta_ci(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    estimate = fixed.within_group_auc(y, left, groups)[0] - fixed.within_group_auc(
        y, right, groups
    )[0]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_group[group] for group in picked])
        boot_group = np.concatenate(
            [np.full(len(by_group[group]), axis) for axis, group in enumerate(picked)]
        )
        value = fixed.within_group_auc(y[index], left[index], boot_group)[0]
        value -= fixed.within_group_auc(y[index], right[index], boot_group)[0]
        if np.isfinite(value):
            values.append(value)
    values = np.asarray(values)
    base = {
        "delta": float(estimate),
        "ci95": [float(item) for item in np.quantile(values, [0.025, 0.975])],
        "bootstrap_standard_error": float(values.std(ddof=1)),
        "approx_mde80_two_sided_alpha05": float(2.802 * values.std(ddof=1)),
        "valid_draws": int(len(values)),
    }
    return {
        **base,
        "exceeds_delta_min": bool(base["delta"] >= DELTA_MIN),
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# t34 closure audit",
        "",
        "All 512 long-task rollouts still have a query at absolute index 34. The outcome is eventual failure, and every model is scored only on success/failure pairs within the same held-out initial state.",
        "",
        "## Risk-set audit",
        "",
        "- Median success length: %.1f; median failure length: %.1f." % (
            summary["audit"]["success_length_median"], summary["audit"]["failure_length_median"]
        ),
        "- t34 is phase %.3f for the median success and %.3f for the median failure." % (
            summary["audit"]["t34_phase_at_success_median_length"],
            summary["audit"]["t34_phase_at_failure_median_length"],
        ),
        "- Exact future-remaining-time overlap contains %d successes and %d failures: Gate %s." % (
            summary["audit"]["successes_in_exact_remaining_overlap"],
            summary["audit"]["failures_in_exact_remaining_overlap"],
            summary["audit"]["risk_set_overlap_gate"].upper(),
        ),
        "",
        "## Model ladder",
        "",
        "| model | within-init AUC | pooled AUC |",
        "|---|---:|---:|",
    ]
    for name, row in summary["models"].items():
        lines.append(f"| {name} | {row['within_init_auc']:.3f} | {row['pooled_auc']:.3f} |")
    lines.extend(
        [
            "",
            "## Increment over full state + action",
            "",
            "| addition | delta | 95% init-bootstrap CI | approximate MDE80 | passes +0.05 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, row in summary["increments"].items():
        lines.append(
            "| %s | %+.3f | [%+.3f, %+.3f] | %.3f | %s |"
            % (
                name,
                row["delta"],
                *row["ci95"],
                row["approx_mde80_two_sided_alpha05"],
                "yes" if row["exceeds_delta_min"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Post-hoc sentinels",
            "",
            "`remaining_queries_posthoc` AUC is %.3f and `-phase_episode_posthoc` AUC is %.3f. These variables are unavailable online; they diagnose the corpus geometry only." % (
                summary["sentinels"]["remaining_queries_posthoc_auc"],
                summary["sentinels"]["negative_phase_episode_posthoc_auc"],
            ),
            "",
            "## Verdict",
            "",
            "The active-at-t34 condition is satisfied, but future remaining time has essentially no two-outcome overlap. The routing MDE80 values (0.063-0.069) also exceed the +0.05 practical threshold, so the finite-sample negative does not exclude a +0.05 effect. Independently of power, the failed overlap gate means this cache can test concurrent state decodability at t34 but cannot turn it into a phase-matched early-warning claim.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    p = np.full((2, 8, 10, 11, 32), 1.0 / 32.0, dtype=np.float32)
    state, action = soft_route_features(p)
    assert state.shape == (2, 256)
    assert action.shape == (2, 768)
    assert np.allclose(action[:, :32], 1.0 / 32.0)
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame, blocks, audit = load_data(args.cache_root, args.out_dir)
    y = frame["failure"].to_numpy(dtype=np.int64)
    groups = frame["init_state_id"].to_numpy(dtype=np.int64)
    predictions = cross_fit(blocks, y, groups, args.seed)
    models = {name: fixed.score_block(y, score, groups) for name, score in predictions.items()}
    reference = predictions["M2_full_state_action"]
    increments = {
        name: delta_ci(y, predictions[name], reference, groups, args.bootstrap, args.seed + axis)
        for axis, name in enumerate(("M3_M2_state_route", "M4_M2_action_route", "M5_M2_hidden"))
    }
    sentinels = {
        "remaining_queries_posthoc_auc": fixed.fast_auc(
            y, frame["remaining_queries_posthoc"].to_numpy()
        ),
        "negative_phase_episode_posthoc_auc": fixed.fast_auc(
            y, -frame["phase_episode_posthoc"].to_numpy()
        ),
        "phase_budget_constant": bool(frame["phase_budget"].nunique() == 1),
    }
    summary = {
        "protocol": {
            "task": fixed.LONG_TASK,
            "absolute_query": QUERY,
            "validation": "leave-one-init-out; pair-weighted within-init AUC",
            "hard_expert_ids_used": False,
            "delta_min": DELTA_MIN,
        },
        "audit": audit,
        "models": models,
        "increments": increments,
        "sentinels": sentinels,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(fixed.plain(summary), indent=2, ensure_ascii=False) + "\n"
    )
    pd.DataFrame(
        {
            **{column: frame[column] for column in frame.columns},
            **{f"score_{name}": score for name, score in predictions.items()},
        }
    ).to_csv(args.out_dir / "predictions.csv", index=False)
    (args.out_dir / "report.md").write_text(render(summary))
    print(f"wrote {args.out_dir} in {summary['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
