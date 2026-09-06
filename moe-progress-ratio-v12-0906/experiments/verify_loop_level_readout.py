#!/usr/bin/env python3
"""Is there anything in the nine denoising steps we have been throwing away?

The router tensor is `[8 layers, 10 denoising steps, 11 tokens, 32 experts]`, and
every detector in this line reads `[:, 9, 1:11, :]` — the final step only. The
other nine steps are discarded, then a scalar per query is strung into a time
series and thresholded across queries.

That cross-query machinery is where every failure in this bundle came from. The
pooled trajectory-peak quantile drifts 44.5 points with corpus horizon
composition; the relative baseline assumes the rollout started normally;
segments flicker with a median run of one chunk; and above all, everything
indexed by chunk competes with a clock rule that gets recall 1.000 at a lower
false-alarm rate than any of it.

A loop-level readout has none of that. One query in, one verdict out, no
history, no baseline, no persistence, no cross-query threshold — and nothing
that is a function of how long the rollout has been running. It is also the
most local possible form, which matches the one capability this bundle did
establish: routing works as an event-anchored local readout and fails as a
temporal predictor.

The test is the matched-pose contrast that already worked. At a gripper closure,
same task, matched end-effector displacement and query index, the object either
came along or stayed behind. Four feature sets are scored on the identical
matched pairs:

    A  step 9, with a cross-query reference   the current practice
    B  step 9, no reference                   what one query gives without history
    C  full loop, no reference                the pure loop-level claim
    D  full loop, with a reference            the combination

If C does not beat B, the nine discarded steps carry nothing. If C reaches A,
a single query without any history matches what the current practice needs a
reference query to get.

Run:  python experiments/verify_loop_level_readout.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402

ROUTE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
TAXONOMY = WORKSPACE / "analysis_trap_taxonomy/results"
LAYER_ROOT = WORKSPACE / "moe-v4-0904/results/layerwise_mobility"
DEFAULT_OUTPUT = BUNDLE / "results/loop_level"

RUNS = {"seed1000_1007": "right-50x8-20260903", "seed1008_1015": "right-50x8b-20260903"}
KEY = ["task", "init_state_id", "flow_noise_seed"]
EEF_CALIPER = 0.025
QUERY_CALIPER = 3
MAX_CONTROLS = 5
RELATIVE = (0, 1, 2)
DRAWS = 2000
SEED = 20260906
EXPECTED = {"seed1000_1007": 116, "seed1008_1015": 116}


def events() -> pd.DataFrame:
    frame = pd.read_csv(TAXONOMY / "belief_mismatch_events.csv.gz")
    frame["case"] = frame["missed_grasp_then_departure"].astype(bool)
    frame["control"] = frame["coupled_target_motion"].astype(bool)
    return frame[frame["case"] | frame["control"]].copy()


def match(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Same run and task, matched displacement and query index, other episode."""
    pairs = []
    for run, block in frame.groupby("run"):
        cases = block[block["case"]]
        controls = block[block["control"]]
        for _, case in cases.iterrows():
            pool = controls[
                (controls["task"] == case["task"])
                & (controls["episode"] != case["episode"])
                & (
                    (
                        controls["eef_max_displacement_after_m"]
                        - case["eef_max_displacement_after_m"]
                    ).abs()
                    <= EEF_CALIPER
                )
                & (
                    (
                        controls["post_closure_state_query"]
                        - case["post_closure_state_query"]
                    ).abs()
                    <= QUERY_CALIPER
                )
            ]
            if pool.empty:
                pairs.append({"run": run, "task": case["task"], "matched": False})
                continue
            picked = pool.head(MAX_CONTROLS)
            for _, control in picked.iterrows():
                for role, row in (("case", case), ("control", control)):
                    pairs.append(
                        {
                            "run": run,
                            "task": case["task"],
                            "stratum": f"{run}|{case['task']}|{case['episode']}",
                            "role": role,
                            "episode": int(row["episode"]),
                            "closure": int(row["closure_action_query"]),
                            "post_closure": int(row["post_closure_state_query"]),
                            "matched": True,
                        }
                    )
    return pd.DataFrame(pairs)


def episode_routes(run: str, task: str, episode: int, cache: dict) -> np.ndarray:
    key = (run, task, episode)
    if key not in cache:
        group = zarr.open_group(
            str(ROUTE_ROOT / task / RUNS[run] / "server/routes.zarr"), mode="r"
        )
        identifiers = np.asarray(group["episode_id"][:], dtype=int)
        positions = np.flatnonzero(identifiers == episode)
        if len(positions) and not np.all(np.diff(positions) == 1):
            raise ValueError(f"non-contiguous episode {episode} in {task}")
        cache[key] = np.asarray(
            group["hb_router_probs"][positions[0] : positions[-1] + 1], dtype=np.float32
        )
    return cache[key]


def normalise(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    affinity = np.sqrt(normalise(left) * normalise(right)).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def entropy(values: np.ndarray) -> np.ndarray:
    probability = np.clip(normalise(values), 1e-12, 1.0)
    return -(probability * np.log(probability)).sum(axis=-1) / np.log(pr.N_EXPERTS)


def features(query: np.ndarray, reference: np.ndarray | None) -> dict[str, float]:
    """`query` is one [8, 10, 11, 32] tensor; `reference` is the pre-closure one."""
    action = normalise(query[:, :, pr.ACTION, :])          # [8, 10, 10, 32]
    final = action[:, pr.FINAL_FLOW]                       # [8, 10, 32]
    out: dict[str, float] = {}

    # --- step 9 only, no reference -------------------------------------------
    final_entropy = entropy(final).mean(axis=1)            # [8]
    for index, name in enumerate(pr.LAYER_NAMES):
        out[f"B_entropy_{name}"] = float(final_entropy[index])

    # --- full loop, no reference ---------------------------------------------
    speed = hellinger(action[:, 1:], action[:, :-1]).mean(axis=2)   # [8, 9]
    path = speed.sum(axis=1)
    endpoint = hellinger(action[:, -1], action[:, 0]).mean(axis=1)  # [8]
    cumulative = np.cumsum(speed, axis=1) / np.maximum(path[:, None], 1e-12)
    settle_step = (cumulative < 0.9).sum(axis=1).astype(np.float32)
    state_speed = hellinger(
        query[:, 1:, 0, :], query[:, :-1, 0, :]
    ).sum(axis=1)                                                    # [8]
    for index, name in enumerate(pr.LAYER_NAMES):
        out[f"C_path_{name}"] = float(path[index])
        out[f"C_straight_{name}"] = float(endpoint[index] / max(path[index], 1e-12))
        out[f"C_settle_{name}"] = float(settle_step[index])
        out[f"C_late_over_early_{name}"] = float(
            speed[index, -3:].mean() / max(speed[index, :3].mean(), 1e-12)
        )
        out[f"C_state_path_{name}"] = float(state_speed[index])
    out["C_settle_spread"] = float(settle_step.std())
    out["C_entropy_drop"] = float(
        (entropy(action[:, 0]).mean(axis=1) - final_entropy).mean()
    )

    # --- reference-dependent --------------------------------------------------
    if reference is not None:
        reference_final = normalise(reference[:, pr.FINAL_FLOW, pr.ACTION, :])
        drift = hellinger(final, reference_final).mean(axis=1)
        reference_action = normalise(reference[:, :, pr.ACTION, :])
        loop_drift = hellinger(action, reference_action).mean(axis=(1, 2))
        for index, name in enumerate(pr.LAYER_NAMES):
            out[f"A_drift_{name}"] = float(drift[index])
            out[f"D_loopdrift_{name}"] = float(loop_drift[index])
    return out


def stratum_scores(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-stratum matched AUC and weight, computed once so bootstraps are cheap."""
    records = []
    for stratum, block in frame.groupby("stratum", sort=False):
        case = block[block["role"] == "case"][column].to_numpy(np.float64)
        control = block[block["role"] == "control"][column].to_numpy(np.float64)
        if not len(case) or not len(control):
            continue
        usable = np.isfinite(case[:, None]) & np.isfinite(control[None, :])
        if not usable.any():
            continue
        wins = (case[:, None] > control[None, :])[usable].mean()
        ties = (case[:, None] == control[None, :])[usable].mean()
        records.append(
            {
                "stratum": stratum,
                "task": block["task"].iloc[0],
                "auc": float(wins + 0.5 * ties),
                "weight": float(len(control)),
            }
        )
    return pd.DataFrame(records)


def matched_auc(frame: pd.DataFrame, column: str) -> float:
    scores = stratum_scores(frame, column)
    if scores.empty:
        return float("nan")
    return float((scores["auc"] * scores["weight"]).sum() / scores["weight"].sum())


def set_auc(frame: pd.DataFrame, columns: list[str]) -> tuple[float, float, float]:
    """Task-blocked held-out matched AUC for a whole feature set."""
    matrix = frame[columns].to_numpy(np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    target = (frame["role"] == "case").to_numpy(int)
    groups = frame["task"].to_numpy()
    predicted = np.full(len(frame), np.nan)
    splits = min(5, len(np.unique(groups)))
    for train, test in GroupKFold(n_splits=splits).split(matrix, target, groups):
        centre, scale = matrix[train].mean(axis=0), matrix[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        model = LogisticRegression(max_iter=4000, C=1.0).fit(
            (matrix[train] - centre) / scale, target[train]
        )
        predicted[test] = model.predict_proba((matrix[test] - centre) / scale)[:, 1]
    per_stratum = stratum_scores(frame.assign(_score=predicted), "_score")
    if per_stratum.empty:
        return float("nan"), float("nan"), float("nan")
    auc = per_stratum["auc"].to_numpy()
    weight = per_stratum["weight"].to_numpy()
    point = float((auc * weight).sum() / weight.sum())
    tasks = per_stratum["task"].to_numpy()
    names = np.unique(tasks)
    index = {name: np.flatnonzero(tasks == name) for name in names}
    rng = np.random.default_rng(SEED)
    samples = np.empty(DRAWS)
    for draw in range(DRAWS):
        picked = rng.integers(0, len(names), len(names))
        rows = np.concatenate([index[names[position]] for position in picked])
        samples[draw] = (auc[rows] * weight[rows]).sum() / weight[rows].sum()
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = events()
    counts = {
        run: int(frame[(frame["run"] == run) & frame["case"]].shape[0])
        for run in RUNS
    }
    if counts != EXPECTED:
        raise SystemExit(f"case counts changed: {counts} against {EXPECTED}")

    rng = np.random.default_rng(SEED)
    design = match(frame, rng)
    unmatched = int((~design["matched"]).sum())
    design = design[design["matched"]].reset_index(drop=True)

    cache: dict = {}
    rows = []
    for relative in RELATIVE:
        for _, row in design.iterrows():
            routes = episode_routes(row["run"], row["task"], int(row["episode"]), cache)
            query = int(row["post_closure"]) + relative
            reference_index = int(row["closure"]) - 1
            if not (0 <= query < len(routes)) or not (0 <= reference_index < len(routes)):
                continue
            record = features(routes[query], routes[reference_index])
            record.update(
                {
                    "relative": relative,
                    "stratum": row["stratum"],
                    "task": row["task"],
                    "role": row["role"],
                }
            )
            rows.append(record)
        print(f"  rel {relative} done, {len(rows)} rows", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "loop_features.csv.gz", index=False, compression="gzip")

    sets = {
        "A_step9_reference": [c for c in table.columns if c.startswith("A_")],
        "B_step9_no_reference": [c for c in table.columns if c.startswith("B_")],
        "C_loop_no_reference": [c for c in table.columns if c.startswith("C_")],
        "D_loop_reference": [
            c for c in table.columns if c.startswith(("C_", "D_", "A_"))
        ],
    }
    results = []
    for relative in RELATIVE:
        block = table[table["relative"] == relative]
        for name, columns in sets.items():
            point, low, high = set_auc(block, columns)
            best = max(
                ((matched_auc(block, c), c) for c in columns),
                key=lambda item: abs(item[0] - 0.5) if np.isfinite(item[0]) else -1,
            )
            results.append(
                {
                    "relative": relative,
                    "feature_set": name,
                    "n_features": len(columns),
                    "set_auc": point,
                    "set_auc_ci95": [low, high],
                    "best_single_auc": best[0],
                    "best_single_feature": best[1],
                }
            )
        print(f"  scored rel {relative}", flush=True)

    scored = pd.DataFrame(results)
    scored.to_csv(args.output / "set_comparison.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "cases_per_run": counts,
                "unmatched_cases": unmatched,
                "strata": int(design["stratum"].nunique()),
                "tasks": int(design["task"].nunique()),
                "rows": int(len(table)),
                "results": results,
            },
            indent=2,
        )
    )
    pd.set_option("display.width", 200)
    print(
        f"\n匹配 {design['stratum'].nunique()} 个 stratum / {design['task'].nunique()} 任务，"
        f"未匹配 case {unmatched}\n"
    )
    print(
        scored[
            [
                "relative",
                "feature_set",
                "n_features",
                "set_auc",
                "set_auc_ci95",
                "best_single_auc",
                "best_single_feature",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    main()
