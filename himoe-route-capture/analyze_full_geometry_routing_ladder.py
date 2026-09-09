#!/usr/bin/env python3
"""Close the provisional q0+8 routing result with a full-geometry ladder.

The cohort and landmark are copied from ``analyze_failure_moe_signatures_fixed``:
the first query where pot 2 reaches its goal proxy plus eight queries, restricted
to success, stagnation-core and active-return rollouts whose annotated physical
event is still in the future.  Evaluation is always within initial state and all
preprocessing is fit without the held-out initial state.

This audit adds the cached controls omitted from the original physical block: EEF
orientation, both gripper coordinates, object quaternions/tilt, EEF-object and
object-goal relative vectors, object motion, EEF-object motion coupling, and the
complete recorded simulator state (time, qpos and qvel).  The landmark's goal
reference remains transductive to preserve the published cohort.  Hard expert
identities are deliberately excluded.
"""

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
DEFAULT_OUT = HERE / "analysis/full-geometry-routing-ladder"
DEFAULT_ROUTE_CACHE = (
    HERE / "analysis/failure-moe-signatures-fixed/route_reduced.npz"
)
SEED = 20260829
BOOTSTRAPS = 2000
DELTA_MIN = 0.05

SOFT_ROUTE_SIGNALS = (
    "route_speed",
    "route_recurrence_advantage",
    "route_recurrence_lag_fraction",
    "route_anchor_advantage",
    "route_entropy",
    "route_top1_mass",
    "route_layer_synchrony",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=fixed.CACHE_ROOT)
    parser.add_argument("--route-cache", type=pathlib.Path, default=DEFAULT_ROUTE_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-hidden", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _step(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float64)
    if len(values) > 1:
        out[1:] = np.linalg.norm(np.diff(values, axis=0), axis=1)
    return out


def _quat_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _quat_step(values: np.ndarray) -> np.ndarray:
    q = _quat_normalize(values)
    out = np.zeros(len(q), dtype=np.float64)
    if len(q) > 1:
        dot = np.abs(np.sum(q[1:] * q[:-1], axis=1))
        out[1:] = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
    return out


def _quat_tilt(values: np.ndarray) -> np.ndarray:
    """World-z versus local-z angle for MuJoCo free-joint wxyz quaternions."""
    q = _quat_normalize(values)
    rzz = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return np.arccos(np.clip(rzz, -1.0, 1.0))


def _series_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}__{name}": value
        for name, value in fixed.summarize_window(values).items()
    }


def _layout_slices(run: pathlib.Path) -> dict[str, slice]:
    layout = json.loads((run / "client/sim_layout.json").read_text())
    result: dict[str, slice] = {}
    for row in layout["joints"]:
        name = str(row["joint"])
        if name in ("moka_pot_1_joint0", "moka_pot_2_joint0"):
            lo = int(row["state_lo"])
            result["pot1" if "_1_" in name else "pot2"] = slice(lo, lo + 7)
    if set(result) != {"pot1", "pot2"}:
        raise ValueError("long-task object free joints were not found")
    return result


def _geometry_row(
    state: np.ndarray,
    sim: np.ndarray,
    q0: int,
    cut: int,
    slices: dict[str, slice],
    goals: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    window = slice(q0, cut + 1)
    state = np.asarray(state, dtype=np.float64)
    eef = state[:, :3]
    eef_rot = state[:, 3:6]
    fingers = state[:, 6:8]
    gripper_abs = np.mean(np.abs(fingers), axis=1)
    pots = {
        key: np.asarray(sim[:, sl], dtype=np.float64) for key, sl in slices.items()
    }

    basic_series: dict[str, np.ndarray] = {
        "eef_x": eef[:, 0],
        "eef_y": eef[:, 1],
        "eef_z": eef[:, 2],
        "eef_rot_x": eef_rot[:, 0],
        "eef_rot_y": eef_rot[:, 1],
        "eef_rot_z": eef_rot[:, 2],
        "finger_left": fingers[:, 0],
        "finger_right": fingers[:, 1],
        "gripper_aperture_abs": gripper_abs,
        "eef_step": _step(eef),
        "eef_rotation_step": _step(eef_rot),
        "gripper_step": np.r_[0.0, np.abs(np.diff(gripper_abs))],
    }
    full_series = dict(basic_series)
    for axis in range(sim.shape[1]):
        full_series[f"sim_state_{axis:02d}"] = np.asarray(sim[:, axis], dtype=np.float64)
    for key, pose in pots.items():
        pos, quat = pose[:, :3], pose[:, 3:7]
        rel = pos - eef
        goal_rel = pos - goals[key][None, :]
        full_series.update(
            {
                f"{key}_x": pos[:, 0],
                f"{key}_y": pos[:, 1],
                f"{key}_z": pos[:, 2],
                f"{key}_qw": quat[:, 0],
                f"{key}_qx": quat[:, 1],
                f"{key}_qy": quat[:, 2],
                f"{key}_qz": quat[:, 3],
                f"eef_{key}_dx": rel[:, 0],
                f"eef_{key}_dy": rel[:, 1],
                f"eef_{key}_dz": rel[:, 2],
                f"eef_{key}_distance": np.linalg.norm(rel, axis=1),
                f"{key}_goal_dx": goal_rel[:, 0],
                f"{key}_goal_dy": goal_rel[:, 1],
                f"{key}_goal_dz": goal_rel[:, 2],
                f"{key}_goal_distance": np.linalg.norm(goal_rel, axis=1),
                f"{key}_step": _step(pos),
                f"{key}_rotation_step": _quat_step(quat),
                f"{key}_tilt": _quat_tilt(quat),
                f"eef_{key}_motion_mismatch": _step(pos - eef),
            }
        )
    pot_delta = pots["pot2"][:, :3] - pots["pot1"][:, :3]
    full_series.update(
        {
            "pot1_pot2_dx": pot_delta[:, 0],
            "pot1_pot2_dy": pot_delta[:, 1],
            "pot1_pot2_dz": pot_delta[:, 2],
            "pot1_pot2_distance": np.linalg.norm(pot_delta, axis=1),
            "nearest_object_distance": np.minimum(
                full_series["eef_pot1_distance"], full_series["eef_pot2_distance"]
            ),
            "nearest_goal_distance": np.minimum(
                full_series["pot1_goal_distance"], full_series["pot2_goal_distance"]
            ),
            "mean_goal_distance": 0.5
            * (full_series["pot1_goal_distance"] + full_series["pot2_goal_distance"]),
        }
    )
    basic: dict[str, float] = {}
    full: dict[str, float] = {}
    for name, values in basic_series.items():
        basic.update(_series_summary(name, values[window]))
    for name, values in full_series.items():
        full.update(_series_summary(name, values[window]))
    return basic, full


def _load_episode_arrays(run: pathlib.Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    path = run / "client" / f"episode_{episode:02d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["state"], dtype=np.float32),
            np.asarray(archive["sim_state"], dtype=np.float32),
        )


def _route_columns(frame: pd.DataFrame) -> list[str]:
    return fixed.block_columns(frame, SOFT_ROUTE_SIGNALS)


def build_dataset(
    cache_root: pathlib.Path,
    route_cache: pathlib.Path,
    out_dir: pathlib.Path,
    include_hidden: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    physical, summaries = fixed.load_client(cache_root)
    with np.load(route_cache) as archive:
        reduced = {key: archive[key] for key in archive.files}
    successes = [episode for episode, row in physical.items() if row["success"]]
    goals = fixed.goal_reference(physical, successes)
    labels = fixed.derive_labels(physical, goals)

    action_cache = fixed.build_caches(physical, reduced, "route_action", "occ_action")
    state_cache = fixed.build_caches(physical, reduced, "route_state", "occ_state")
    signals = fixed.PHYSICAL_SIGNALS + fixed.ACTION_SIGNALS + SOFT_ROUTE_SIGNALS
    action_frame = fixed.build_features(labels, action_cache, goals, signals)
    state_frame = fixed.build_features(labels, state_cache, goals, SOFT_ROUTE_SIGNALS)

    valid = action_frame["cut_available"].to_numpy(dtype=bool).copy()
    known = action_frame["physical_onset_query"].to_numpy() >= 0
    valid &= (~known) | (
        action_frame["prospective_cut_query"].to_numpy()
        < action_frame["physical_onset_query"].to_numpy()
    )
    valid &= action_frame["physical_type"].isin(
        ["success", "stagnation_core", "active_return"]
    ).to_numpy()
    cohort = action_frame[valid].sort_values("episode").reset_index(drop=True)
    state_frame = state_frame.set_index("episode").loc[cohort["episode"]].reset_index()
    if len(cohort) != 444:
        raise ValueError(f"expected the published clean-444 cohort, got {len(cohort)}")

    run = cache_root / fixed.LONG_TASK / "right-16x32"
    slices = _layout_slices(run)
    basic_rows: list[dict[str, float]] = []
    full_rows: list[dict[str, float]] = []
    for row in cohort.itertuples(index=False):
        state, sim = _load_episode_arrays(run, int(row.episode))
        basic, full = _geometry_row(
            state,
            sim,
            int(row.pot2_anchor_query),
            int(row.prospective_cut_query),
            slices,
            goals,
        )
        basic_rows.append(basic)
        full_rows.append(full)

    blocks: dict[str, np.ndarray] = {
        "basic": pd.DataFrame(basic_rows).to_numpy(dtype=np.float64),
        "full_geometry": pd.DataFrame(full_rows).to_numpy(dtype=np.float64),
        "legacy_physical": cohort[
            fixed.block_columns(cohort, fixed.PHYSICAL_SIGNALS)
        ].to_numpy(dtype=np.float64),
        "action": cohort[
            fixed.block_columns(cohort, fixed.ACTION_SIGNALS)
        ].to_numpy(dtype=np.float64),
        "state_route": state_frame[_route_columns(state_frame)].to_numpy(dtype=np.float64),
        "action_route": cohort[_route_columns(cohort)].to_numpy(dtype=np.float64),
    }

    hidden_cache = out_dir / "hidden_features.npz"
    if include_hidden:
        if hidden_cache.exists():
            with np.load(hidden_cache) as archive:
                hidden_episode = np.asarray(archive["episode"], dtype=np.int64)
                hidden_matrix = np.asarray(archive["features"], dtype=np.float32)
            if not np.array_equal(hidden_episode, cohort["episode"].to_numpy()):
                raise ValueError("hidden feature cache cohort changed")
        else:
            group = zarr.open_group(str(run / "server/hidden.zarr"), mode="r")
            offsets = {
                int(e): int(o)
                for e, o in zip(reduced["episode"], reduced["episode_offset"])
            }
            hidden_rows = []
            for axis, row in enumerate(cohort.itertuples(index=False), start=1):
                start = offsets[int(row.episode)] + int(row.pot2_anchor_query)
                stop = offsets[int(row.episode)] + int(row.prospective_cut_query) + 1
                values = np.asarray(group["hb_hidden"][start:stop], dtype=np.float16)
                query_features = hidden_features(values).astype(np.float64)
                time_axis = np.linspace(0.0, 1.0, len(query_features))
                slope = np.polyfit(time_axis, query_features, 1)[0]
                hidden_rows.append(
                    np.concatenate(
                        [query_features[-1], query_features.mean(0), query_features.std(0), slope]
                    )
                )
                if axis % 32 == 0 or axis == len(cohort):
                    print(f"hidden summaries {axis}/{len(cohort)}", flush=True)
            hidden_matrix = np.asarray(hidden_rows, dtype=np.float32)
            np.savez_compressed(
                hidden_cache,
                episode=cohort["episode"].to_numpy(dtype=np.int64),
                features=hidden_matrix,
            )
        blocks["hidden"] = hidden_matrix.astype(np.float64)

    metadata = {
        "n": len(cohort),
        "failures": int((cohort["physical_type"] != "success").sum()),
        "successes": int((cohort["physical_type"] == "success").sum()),
        "initial_states": int(cohort["init_state_id"].nunique()),
        "class_counts": {
            str(key): int(value)
            for key, value in cohort["physical_type"].value_counts().items()
        },
        "feature_widths": {key: int(value.shape[1]) for key, value in blocks.items()},
        "goal_reference": "global successful terminal mean, matching the published clean-444 result",
    }
    return cohort, blocks, metadata


MODEL_COMPONENTS = {
    "legacy_M2": ("legacy_physical", "action"),
    "legacy_route_only": ("action_route",),
    "legacy_joint": ("legacy_physical", "action", "action_route"),
    "M0_basic_proprio": ("basic",),
    "M1_full_geometry": ("full_geometry",),
    "M2_full_geometry_action": ("full_geometry", "action"),
    "M3_M2_state_route": ("full_geometry", "action", "state_route"),
    "M4_M2_action_route": ("full_geometry", "action", "action_route"),
    "M5_M2_hidden": ("full_geometry", "action", "hidden"),
}

PCA_CAP = {
    "basic": 24,
    "full_geometry": 64,
    "legacy_physical": 24,
    "action": 16,
    "state_route": 16,
    "action_route": 16,
    "hidden": 24,
}


def _fit_component(
    train: np.ndarray, test: np.ndarray, cap: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median").fit(train)
    train_i, test_i = imputer.transform(train), imputer.transform(test)
    keep = np.std(train_i, axis=0) > 1e-10
    if not np.any(keep):
        raise ValueError("constant feature component")
    scaler = StandardScaler().fit(train_i[:, keep])
    train_s, test_s = scaler.transform(train_i[:, keep]), scaler.transform(test_i[:, keep])
    width = min(cap, train_s.shape[1], train_s.shape[0] - 1)
    if width < train_s.shape[1]:
        pca = PCA(n_components=width, svd_solver="randomized", random_state=seed).fit(train_s)
        train_s, test_s = pca.transform(train_s), pca.transform(test_s)
    return train_s, test_s


def cross_fit(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    logistic_c: float = 0.1,
    pca_caps: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    pca_caps = pca_caps or PCA_CAP
    predictions = {
        name: np.full(len(y), np.nan, dtype=np.float64)
        for name, parts in MODEL_COMPONENTS.items()
        if all(part in blocks for part in parts)
    }
    for fold, held in enumerate(np.unique(groups)):
        train, test = groups != held, groups == held
        transformed = {
            name: _fit_component(
                values[train], values[test], pca_caps[name], seed + fold * 101 + axis
            )
            for axis, (name, values) in enumerate(blocks.items())
        }
        for name, parts in MODEL_COMPONENTS.items():
            if name not in predictions:
                continue
            x_train = np.column_stack([transformed[part][0] for part in parts])
            x_test = np.column_stack([transformed[part][1] for part in parts])
            model = LogisticRegression(
                C=logistic_c,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed + fold,
            ).fit(x_train, y[train])
            predictions[name][test] = model.predict_proba(x_test)[:, 1]
    if any(np.any(~np.isfinite(score)) for score in predictions.values()):
        raise RuntimeError("incomplete leave-one-init-out prediction")
    return predictions


def sensitivity_grid(
    blocks: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, seed: int
) -> list[dict[str, Any]]:
    geometry_caps = (32, 64, 128, 256)
    specifications = (
        ("C_0.03", 0.03, PCA_CAP),
        ("primary", 0.1, PCA_CAP),
        ("C_0.3", 0.3, PCA_CAP),
        *(
            (
                f"geometry_PCA_{width}",
                0.1,
                {**PCA_CAP, "full_geometry": min(blocks["full_geometry"].shape[1], width)},
            )
            for width in geometry_caps
        ),
    )
    rows = []
    for axis, (name, logistic_c, caps) in enumerate(specifications):
        predictions = cross_fit(
            blocks, y, groups, seed, logistic_c=logistic_c, pca_caps=caps
        )
        reference = fixed.within_group_auc(
            y, predictions["M2_full_geometry_action"], groups
        )[0]
        row: dict[str, Any] = {
            "specification": name,
            "logistic_c": logistic_c,
            "pca_caps": caps,
            "M2_auc": reference,
        }
        for model in ("M3_M2_state_route", "M4_M2_action_route", "M5_M2_hidden"):
            if model not in predictions:
                continue
            auc = fixed.within_group_auc(y, predictions[model], groups)[0]
            row[f"{model}_auc"] = auc
            row[f"{model}_delta"] = auc - reference
        rows.append(row)
        print(f"sensitivity {name} complete", flush=True)
    return rows


def _delta_bootstrap(
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
        delta = fixed.within_group_auc(y[index], left[index], boot_group)[0] - fixed.within_group_auc(
            y[index], right[index], boot_group
        )[0]
        if np.isfinite(delta):
            values.append(delta)
    values = np.asarray(values)
    standard_error = float(values.std(ddof=1))
    return {
        "delta": float(estimate),
        "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "bootstrap_standard_error": standard_error,
        "approx_mde80_two_sided_alpha05": 2.802 * standard_error,
        "exceeds_preregistered_delta_min": bool(estimate >= DELTA_MIN),
        "valid_draws": int(len(values)),
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full cached-state closure audit of the q0+8 result",
        "",
        "## Design",
        "",
        "- Task scope: long/SCENE8 only.",
        "- Exact clean-444 landmark cohort from the fixed-prefix audit.",
        "- Positive label: later stagnation-core or active-return failure; every labelled physical event is after the cut.",
        "- Leave one initial state out; AUC counts only success/failure pairs inside the same held-out initial state.",
        "- Every model imputer, scaler and PCA is fit on training initial states only. The landmark/goal reference is transductive and frozen before these folds.",
        "- Hard expert IDs are excluded. State/action routing means fp16 soft gate probabilities.",
        "- Practical effect threshold was fixed at +%.2f within-init AUC." % DELTA_MIN,
        "",
        "## Cohort",
        "",
        "`n=%d`, failures `%d`, successes `%d`, initial states `%d`." % (
            summary["cohort"]["n"],
            summary["cohort"]["failures"],
            summary["cohort"]["successes"],
            summary["cohort"]["initial_states"],
        ),
        "",
        "## Ladder",
        "",
        "| model | within-init AUC | pooled AUC | feature components |",
        "|---|---:|---:|---|",
    ]
    for name, result in summary["models"].items():
        components = ", ".join(MODEL_COMPONENTS[name])
        lines.append(
            f"| {name} | {result['within_init_auc']:.3f} | {result['pooled_auc']:.3f} | {components} |"
        )
    lines.extend(
        [
            "",
            "## Increment over M2",
            "",
            "| added representation | delta AUC | 95% init-bootstrap CI | approximate MDE80 | point estimate >= +0.05 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name in ("M3_M2_state_route", "M4_M2_action_route", "M5_M2_hidden"):
        if name not in summary["increments"]:
            continue
        row = summary["increments"][name]
        decision = "yes" if row["exceeds_preregistered_delta_min"] else "no"
        lines.append(
            "| %s | %+.3f | [%+.3f, %+.3f] | %.3f | %s |"
            % (name, row["delta"], *row["ci95"], row["approx_mde80_two_sided_alpha05"], decision)
        )
    route = summary["increments"]["M4_M2_action_route"]
    state = summary["increments"]["M3_M2_state_route"]
    sensitivity = summary["sensitivity"]
    route_range = [row["M4_M2_action_route_delta"] for row in sensitivity]
    state_range = [row["M3_M2_state_route_delta"] for row in sensitivity]
    hidden_range = [
        row["M5_M2_hidden_delta"] for row in sensitivity if "M5_M2_hidden_delta" in row
    ]
    lines.extend(
        [
            "",
            "## Specification sensitivity",
            "",
            "Across C=0.03/0.1/0.3 and geometry PCA widths 32/64/128/256, state-route deltas span "
            "[%+.3f, %+.3f] and action-route deltas span [%+.3f, %+.3f]. Hidden deltas span [%+.3f, %+.3f]."
            % (
                min(state_range), max(state_range), min(route_range), max(route_range),
                min(hidden_range), max(hidden_range),
            ),
            "",
            "## Verdict",
            "",
            (
            "The action-soft-routing increment over the full cached physical state plus action is "
                f"{route['delta']:+.3f}; the state-soft-routing increment is {state['delta']:+.3f}. "
                "Interpret these against their initial-state bootstrap intervals and the reported MDE, not against pooled AUC."
            ),
            "",
            "The legacy-like `route_only - physical+action` contrast is shown only as a diagnostic. It does not reproduce the published +0.237 protocol because hard-occupancy features, scaling and dimensionality differ. It is not an additive routing effect; the valid comparisons here are M3/M4 minus M2.",
            "",
            "The hidden point estimate exceeds +0.05, but its 95% interval includes effects well below +0.05. This is a tentative nonzero signal, not evidence that the true increment reaches the practical threshold.",
            "",
            "`hidden` is the router input, not an expert output contribution. The full physical block contains every recorded sim-state coordinate, but the cache still lacks RGB/contact/force and is not complete environment geometry. The global successful-terminal goal reference is retained to match the published cohort, so this is a transductive cohort audit rather than strict unseen-init evaluation.",
            "",
            "Bootstrap intervals resample fixed OOF test-cluster scores; they do not refit the full training pipeline inside each bootstrap draw. MDE is therefore conditional on the fitted OOF models.",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> None:
    q = np.asarray([[1.0, 0.0, 0.0, 0.0], [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]])
    assert np.allclose(_quat_tilt(q), [0.0, np.pi / 2.0])
    qx = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert np.allclose(_quat_tilt(qx), [0.0, np.pi])
    assert np.allclose(_quat_step(q), [0.0, np.pi / 2.0])
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cohort, blocks, metadata = build_dataset(
        args.cache_root, args.route_cache, args.out_dir, not args.skip_hidden
    )
    y = (cohort["physical_type"] != "success").to_numpy(dtype=np.int64)
    groups = cohort["init_state_id"].to_numpy(dtype=np.int64)
    predictions = cross_fit(blocks, y, groups, args.seed)
    models = {
        name: fixed.score_block(y, score, groups)
        for name, score in predictions.items()
    }
    increments = {}
    reference = predictions["M2_full_geometry_action"]
    for name in ("M3_M2_state_route", "M4_M2_action_route", "M5_M2_hidden"):
        if name in predictions:
            increments[name] = _delta_bootstrap(
                y, predictions[name], reference, groups, args.bootstrap, args.seed + len(increments)
            )
    increments["legacy_route_only_minus_legacy_M2"] = _delta_bootstrap(
        y,
        predictions["legacy_route_only"],
        predictions["legacy_M2"],
        groups,
        args.bootstrap,
        args.seed + 90,
    )
    increments["legacy_joint_minus_legacy_M2"] = _delta_bootstrap(
        y,
        predictions["legacy_joint"],
        predictions["legacy_M2"],
        groups,
        args.bootstrap,
        args.seed + 91,
    )
    summary = {
        "protocol": {
            "landmark": "first pot2 goal-proxy query + 8",
            "validation": "leave-one-init-out, pair-weighted within-init AUC",
            "delta_min": DELTA_MIN,
            "pca_caps": PCA_CAP,
            "hard_expert_id_used": False,
        },
        "cohort": metadata,
        "models": models,
        "increments": increments,
        "sensitivity": sensitivity_grid(blocks, y, groups, args.seed + 10000),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(fixed.plain(summary), indent=2, ensure_ascii=False) + "\n"
    )
    cohort[[
        "episode", "init_state_id", "flow_noise_seed", "physical_type",
        "pot2_anchor_query", "prospective_cut_query", "physical_onset_query",
    ]].to_csv(args.out_dir / "cohort.csv", index=False)
    pd.DataFrame(
        {
            "episode": cohort["episode"],
            "label": y,
            "init_state_id": groups,
            **{f"score_{name}": score for name, score in predictions.items()},
        }
    ).to_csv(args.out_dir / "predictions.csv", index=False)
    (args.out_dir / "report.md").write_text(render_report(summary))
    print(f"wrote {args.out_dir} in {summary['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
