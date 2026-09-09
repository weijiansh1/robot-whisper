#!/usr/bin/env python3
"""Test physical-return -> routing-return -> action-replay traps.

The existing failure periodicity audit starts from fixed temporal lags.  This
analysis starts from physical recurrence: for each query it finds the most
similar eligible history state, then measures aligned HB route and action
return at that pair.  See analysis/replanning-reset-trap/PREREG.md.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import zarr
from scipy.stats import rankdata, wilcoxon


HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/replanning-reset-trap"
MOVED_M = 0.03
LIFT_M = 0.01
GOAL_M = 0.05
PROGRESS_M = 0.01
BLOCK = 512
DEFAULT_PERMUTATIONS = 5000
DEFAULT_BOOTSTRAPS = 5000


@dataclass(frozen=True)
class SearchConfig:
    name: str
    min_lag: int
    max_lag: int | None


SEARCHES = (
    SearchConfig("lag2_8", 2, 8),
    SearchConfig("lag3_8", 3, 8),
    SearchConfig("lag2_16", 2, 16),
    SearchConfig("lag2_all", 2, None),
)

CORE_METRICS = (
    "physical_return_p90",
    "state_route_return_p90",
    "action_route_return_p90",
    "action_return_p90",
    "route_reset_p90",
    "action_replay_p90",
    "joint_return_p90",
)

TOPOLOGY_METRICS = (
    "physical_excess_over_lag1_p90",
    "action_route_excess_over_lag1_p90",
    "action_excess_over_lag1_p90",
    "backward_joint_min_excess_p90",
    "backward_joint_candidate_rate",
)

METRIC_LABELS = {
    "physical_return_p90": "physical return",
    "state_route_return_p90": "state-token route return",
    "action_route_return_p90": "action-token route return",
    "action_return_p90": "action-chunk return",
    "route_reset_p90": "physical + route + no-progress",
    "action_replay_p90": "physical + action + no-progress",
    "joint_return_p90": "joint replanning-reset score",
    "physical_excess_over_lag1_p90": "physical return excess over lag 1",
    "action_route_excess_over_lag1_p90": "route return excess over lag 1",
    "action_excess_over_lag1_p90": "action return excess over lag 1",
    "backward_joint_min_excess_p90": "joint backward-return minimum excess",
    "backward_joint_candidate_rate": "strict backward-loop candidate rate",
}


@dataclass
class TaskData:
    key: str
    run: pathlib.Path
    summaries: list[dict]
    lengths: np.ndarray
    offsets: np.ndarray
    failure: np.ndarray
    scenes: np.ndarray
    seeds: np.ndarray
    prefix: int
    layout: dict
    sim: list[np.ndarray]
    state: list[np.ndarray]
    qpos_z: list[np.ndarray]
    action_z: list[np.ndarray]
    goal_distance: list[np.ndarray]
    hard_ids: np.ndarray
    soft_state: np.ndarray
    targets: list[dict]
    modes: list[str]
    stasis: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def episode_path(client: pathlib.Path, episode_index: int) -> pathlib.Path:
    return client / ("episode_%02d.npz" % episode_index)


def standardize_parts(parts: list[np.ndarray]) -> list[np.ndarray]:
    lengths = [len(part) for part in parts]
    values = np.concatenate(parts).astype(np.float32, copy=False)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    keep = scale > 1e-6
    if not np.any(keep):
        raise ValueError("all physical/action channels are constant")
    values = (values[:, keep] - center[keep]) / scale[keep]
    cuts = np.cumsum(lengths)[:-1]
    return [part.astype(np.float32, copy=False) for part in np.split(values, cuts)]


def canonical_qpos(sim: np.ndarray, layout: dict) -> np.ndarray:
    nq = int(layout["nq"])
    qpos = np.asarray(sim[:, 1 : 1 + nq], np.float32).copy()
    for joint in layout["joints"]:
        lo = int(joint["state_lo"]) - 1
        hi = int(joint["state_hi"]) - 1
        if hi - lo != 7:
            continue
        quat = qpos[:, lo + 3 : hi]
        flip = quat[:, 0] < 0
        quat[flip] *= -1
    return qpos


def discover_runs(cache_root: pathlib.Path) -> list[pathlib.Path]:
    runs = []
    for summary in sorted(cache_root.glob("libero_*/*/right-16x32/client/summaries.json")):
        run = summary.parents[1]
        if (run / "server/routes.zarr").exists() and (
            run / "client/sim_layout.json"
        ).exists():
            runs.append(run)
    if not runs:
        raise RuntimeError("no right-16x32 LIBERO runs found")
    return runs


def identify_targets(
    summaries: list[dict], layout: dict, sims: list[np.ndarray]
) -> list[dict]:
    success = np.asarray([bool(row["success"]) for row in summaries])
    targets = []
    for joint in layout["joints"]:
        lo, hi = int(joint["state_lo"]), int(joint["state_hi"])
        if joint["is_robot"] or hi - lo != 7:
            continue
        sl = slice(lo, lo + 3)
        moved = float(
            np.mean(
                [
                    np.linalg.norm(sims[i][-1, sl] - sims[i][0, sl])
                    for i in np.flatnonzero(success)
                ]
            )
        )
        if moved <= MOVED_M:
            continue
        finals = np.stack([sims[i][-1, sl] for i in np.flatnonzero(success)])
        targets.append(
            {
                "name": str(joint["joint"]),
                "lo": lo,
                "hi": lo + 3,
                "goal": finals.mean(axis=0),
                "moved_success_mean_m": moved,
            }
        )
    return targets


def target_pose(sim: np.ndarray, targets: list[dict]) -> np.ndarray:
    if not targets:
        return np.zeros((len(sim), 1), np.float32)
    return np.concatenate(
        [sim[:, int(target["lo"]) : int(target["hi"])] for target in targets], axis=1
    ).astype(np.float32, copy=False)


def build_goal_distance(
    summaries: list[dict], sims: list[np.ndarray], targets: list[dict]
) -> list[np.ndarray]:
    if not targets:
        return [np.zeros(len(sim), np.float32) for sim in sims]
    success = np.asarray([bool(row["success"]) for row in summaries])
    terminal = np.stack(
        [target_pose(sims[i], targets)[-1] for i in np.flatnonzero(success)]
    )
    success_index = np.flatnonzero(success)
    out = []
    for i, (row, sim) in enumerate(zip(summaries, sims)):
        keep = np.asarray(
            [
                int(summaries[j]["init_state_id"]) != int(row["init_state_id"])
                and int(summaries[j]["flow_noise_seed"])
                != int(row["flow_noise_seed"])
                for j in success_index
            ],
            bool,
        )
        refs = terminal[keep]
        if not len(refs):
            refs = terminal
        pose = target_pose(sim, targets)
        out.append(
            np.linalg.norm(pose[:, None, :] - refs[None, :, :], axis=-1)
            .min(axis=1)
            .astype(np.float32)
        )
    return out


def classify_failures(
    summaries: list[dict], sims: list[np.ndarray], targets: list[dict],
    goal_distance: list[np.ndarray]
) -> tuple[list[str], np.ndarray]:
    modes = ["success" if bool(row["success"]) else "unclassified" for row in summaries]
    stasis = np.zeros(len(summaries), bool)
    for i, row in enumerate(summaries):
        if bool(row["success"]):
            continue
        sim = sims[i]
        lifted = placed = reached_lost = 0
        travel = 0.0
        for target in targets:
            lo, hi = int(target["lo"]), int(target["hi"])
            position = sim[:, lo:hi]
            height = float((position[:, 2] - position[0, 2]).max())
            distance = np.linalg.norm(position - np.asarray(target["goal"]), axis=1)
            travel += float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
            lifted += int(height > LIFT_M)
            if distance[-1] <= GOAL_M:
                placed += 1
            elif distance.min() <= GOAL_M:
                reached_lost += 1
        n_targets = len(targets)
        if not n_targets:
            mode = "unclassified"
        elif lifted == 0:
            mode = "never_grasped"
        elif reached_lost:
            mode = "reached_then_lost"
        elif placed and placed < n_targets:
            mode = "partial"
        elif travel > 0.40:
            mode = "misplaced"
        else:
            mode = "lifted_not_placed"
        modes[i] = mode
        distance = goal_distance[i]
        best = np.minimum.accumulate(distance)
        future = np.minimum.accumulate(distance[::-1])[::-1]
        stasis[i] = bool(np.any((best - future <= PROGRESS_M) & (best > GOAL_M)))
    return modes, stasis


def load_task(run: pathlib.Path, cache_root: pathlib.Path) -> TaskData:
    key = str(run.relative_to(cache_root).parent)
    summaries = sorted(
        json.loads((run / "client/summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    expected_indices = list(range(len(summaries)))
    actual_indices = [int(row["episode_index"]) for row in summaries]
    if actual_indices != expected_indices:
        raise ValueError(f"non-contiguous episode indices in {key}")
    lengths = np.asarray([int(row["inference_calls"]) for row in summaries], np.int32)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    total = int(lengths.sum())
    scenes = np.asarray([int(row["init_state_id"]) for row in summaries], np.int32)
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries], np.int32)
    failure = np.asarray([not bool(row["success"]) for row in summaries])
    layout = json.loads((run / "client/sim_layout.json").read_text())

    sims, states, actions, qpos = [], [], [], []
    for row, length in zip(summaries, lengths):
        path = episode_path(run / "client", int(row["episode_index"]))
        with np.load(path, allow_pickle=False) as episode:
            sim = np.asarray(episode["sim_state"], np.float32)
            state = np.asarray(episode["state"], np.float32)
            action = np.asarray(episode["actions"], np.float32).reshape(int(length), -1)
        if len(sim) != length or len(state) != length or len(action) != length:
            raise ValueError(f"client length mismatch in {path}")
        sims.append(sim)
        states.append(state)
        actions.append(action)
        qpos.append(canonical_qpos(sim, layout))

    route = zarr.open(str(run / "server/routes.zarr"), mode="r")
    episode_id = np.asarray(route["episode_id"][:], np.int32)
    expected_id = np.repeat(np.arange(len(summaries), dtype=np.int32), lengths)
    if not np.array_equal(episode_id, expected_id):
        raise ValueError(f"server/client episode alignment failed in {key}")
    hard_ids = np.asarray(route["hb_expert_ids"][:], np.uint8)
    if hard_ids.shape != (total, 8, 10, 11, 4):
        raise ValueError(f"unexpected route shape {hard_ids.shape} in {key}")
    soft_state = np.empty((total, 8, 32), np.float32)
    probs = route["hb_router_probs"]
    for first in range(0, total, BLOCK):
        stop = min(first + BLOCK, total)
        soft_state[first:stop] = np.asarray(
            probs[first:stop, :, 0, 0, :], np.float32
        )

    targets = identify_targets(summaries, layout, sims)
    goal_distance = build_goal_distance(summaries, sims, targets)
    modes, stasis = classify_failures(summaries, sims, targets, goal_distance)
    return TaskData(
        key=key,
        run=run,
        summaries=summaries,
        lengths=lengths,
        offsets=offsets,
        failure=failure,
        scenes=scenes,
        seeds=seeds,
        prefix=int(lengths.min()),
        layout=layout,
        sim=sims,
        state=states,
        qpos_z=standardize_parts(qpos),
        action_z=standardize_parts(actions),
        goal_distance=goal_distance,
        hard_ids=hard_ids,
        soft_state=soft_state,
        targets=targets,
        modes=modes,
        stasis=stasis,
    )


def top4_overlap(a: np.ndarray, b: np.ndarray) -> float:
    matches = (a[..., :, None] == b[..., None, :]).sum(axis=(-1, -2))
    return float(matches.mean() / 4.0)


def geometric_mean(*values: float) -> float:
    return float(math.exp(np.mean(np.log(np.maximum(values, 1e-12)))))


def pair_metrics(
    data: TaskData, episode_k: int, k: int, episode_j: int, j: int
) -> dict[str, float | int]:
    row_k = int(data.offsets[episode_k] + k)
    row_j = int(data.offsets[episode_j] + j)
    physical_distance = float(
        np.sqrt(np.mean(np.square(data.qpos_z[episode_k][k] - data.qpos_z[episode_j][j])))
    )
    action_distance = float(
        np.sqrt(
            np.mean(
                np.square(data.action_z[episode_k][k] - data.action_z[episode_j][j])
            )
        )
    )
    state_route = top4_overlap(
        data.hard_ids[row_k, :, 0, 0, :], data.hard_ids[row_j, :, 0, 0, :]
    )
    action_route = top4_overlap(
        data.hard_ids[row_k, :, :, 1:, :], data.hard_ids[row_j, :, :, 1:, :]
    )
    soft_state = float(
        np.sqrt(
            np.maximum(data.soft_state[row_k], 0)
            * np.maximum(data.soft_state[row_j], 0)
        )
        .sum(axis=-1)
        .mean()
    )
    progress_delta = float(
        data.goal_distance[episode_j][j] - data.goal_distance[episode_k][k]
    )
    physical_similarity = float(math.exp(-0.5 * physical_distance**2))
    action_similarity = float(math.exp(-0.5 * action_distance**2))
    progress_similarity = float(
        math.exp(-0.5 * (abs(progress_delta) / PROGRESS_M) ** 2)
    )
    route_reset = geometric_mean(
        physical_similarity, action_route, progress_similarity
    )
    action_replay = geometric_mean(
        physical_similarity, action_similarity, progress_similarity
    )
    joint_return = geometric_mean(
        physical_similarity, action_route, action_similarity, progress_similarity
    )
    return {
        "past_episode": int(episode_j),
        "past_query": int(j),
        "current_episode": int(episode_k),
        "current_query": int(k),
        "lag": int(k - j) if episode_k == episode_j else -1,
        "physical_distance": physical_distance,
        "physical_similarity": physical_similarity,
        "state_route_similarity": state_route,
        "soft_state_similarity": soft_state,
        "action_route_similarity": action_route,
        "action_distance": action_distance,
        "action_similarity": action_similarity,
        "progress_delta_m": progress_delta,
        "progress_similarity": progress_similarity,
        "route_reset_score": route_reset,
        "action_replay_score": action_replay,
        "joint_return_score": joint_return,
    }


def history_match(data: TaskData, episode: int, k: int, config: SearchConfig) -> dict:
    last = k - config.min_lag
    if last < 0:
        raise ValueError("query has no eligible history")
    first = 0 if config.max_lag is None else max(0, k - config.max_lag)
    candidates = np.arange(first, last + 1, dtype=np.int32)
    distance = np.sqrt(
        np.mean(
            np.square(data.qpos_z[episode][candidates] - data.qpos_z[episode][k]),
            axis=1,
        )
    )
    j = int(candidates[int(np.argmin(distance))])
    matched = pair_metrics(data, episode, k, episode, j)
    adjacent = pair_metrics(data, episode, k, episode, k - 1)
    physical_excess = float(
        matched["physical_similarity"] - adjacent["physical_similarity"]
    )
    state_route_excess = float(
        matched["state_route_similarity"] - adjacent["state_route_similarity"]
    )
    action_route_excess = float(
        matched["action_route_similarity"] - adjacent["action_route_similarity"]
    )
    action_excess = float(
        matched["action_similarity"] - adjacent["action_similarity"]
    )
    minimum_excess = min(physical_excess, action_route_excess, action_excess)
    matched.update(
        {
            "adjacent_physical_similarity": adjacent["physical_similarity"],
            "adjacent_state_route_similarity": adjacent["state_route_similarity"],
            "adjacent_action_route_similarity": adjacent["action_route_similarity"],
            "adjacent_action_similarity": adjacent["action_similarity"],
            "physical_excess_over_lag1": physical_excess,
            "state_route_excess_over_lag1": state_route_excess,
            "action_route_excess_over_lag1": action_route_excess,
            "action_excess_over_lag1": action_excess,
            "backward_joint_min_excess": minimum_excess,
            "backward_joint_candidate": bool(
                minimum_excess > 0
                and abs(float(matched["progress_delta_m"])) <= PROGRESS_M
            ),
        }
    )
    return matched


def compute_query_matches(data: TaskData, config: SearchConfig) -> list[list[dict]]:
    out: list[list[dict]] = []
    for episode, length in enumerate(data.lengths):
        rows = []
        for k in range(config.min_lag, int(length)):
            row = history_match(data, episode, k, config)
            row.update(
                {
                    "task": data.key,
                    "episode": int(episode),
                    "scene": int(data.scenes[episode]),
                    "flow_noise_seed": int(data.seeds[episode]),
                    "failure": bool(data.failure[episode]),
                    "failure_mode": data.modes[episode],
                    "stasis": bool(data.stasis[episode]),
                    "search": config.name,
                }
            )
            rows.append(row)
        out.append(rows)
    return out


def calibrate_thresholds(
    data: TaskData, events: list[list[dict]], config: SearchConfig
) -> dict[str, float]:
    control = [
        row
        for episode, rows in enumerate(events)
        if not data.failure[episode]
        for row in rows
        if int(row["current_query"]) < data.prefix
    ]
    if not control:
        raise ValueError(f"no success controls for {data.key}/{config.name}")
    physical = np.asarray([row["physical_distance"] for row in control])
    physical_threshold = float(np.quantile(physical, 0.25))
    recurrent = [
        row for row in control if row["physical_distance"] <= physical_threshold
    ]
    if not recurrent:
        recurrent = control
    return {
        "physical_distance_q25": physical_threshold,
        "action_route_similarity_q75_given_physical": float(
            np.quantile([row["action_route_similarity"] for row in recurrent], 0.75)
        ),
        "state_route_similarity_q75_given_physical": float(
            np.quantile([row["state_route_similarity"] for row in recurrent], 0.75)
        ),
        "action_distance_q25_given_physical": float(
            np.quantile([row["action_distance"] for row in recurrent], 0.25)
        ),
        "progress_abs_m": PROGRESS_M,
        "success_prefix_query_pairs": len(control),
        "success_prefix_physical_pairs": len(recurrent),
    }


def aggregate_episode(
    rows: list[dict], start: int, stop: int, thresholds: dict[str, float]
) -> dict[str, float | int]:
    selected = [
        row for row in rows if start <= int(row["current_query"]) < stop
    ]
    if not selected:
        raise ValueError(f"empty episode window [{start}, {stop})")

    def q90(key: str) -> float:
        return float(np.quantile([row[key] for row in selected], 0.90))

    out: dict[str, float | int] = {
        "query_pairs": len(selected),
        "physical_return_p90": q90("physical_similarity"),
        "state_route_return_p90": q90("state_route_similarity"),
        "soft_state_return_p90": q90("soft_state_similarity"),
        "action_route_return_p90": q90("action_route_similarity"),
        "action_return_p90": q90("action_similarity"),
        "route_reset_p90": q90("route_reset_score"),
        "action_replay_p90": q90("action_replay_score"),
        "joint_return_p90": q90("joint_return_score"),
        "physical_excess_over_lag1_p90": q90("physical_excess_over_lag1"),
        "action_route_excess_over_lag1_p90": q90(
            "action_route_excess_over_lag1"
        ),
        "action_excess_over_lag1_p90": q90("action_excess_over_lag1"),
        "backward_joint_min_excess_p90": q90("backward_joint_min_excess"),
        "backward_joint_candidate_rate": float(
            np.mean([bool(row["backward_joint_candidate"]) for row in selected])
        ),
        "nonminimum_lag_rate": float(
            np.mean(
                [
                    int(row["lag"])
                    > min(int(item["lag"]) for item in selected)
                    for row in selected
                ]
            )
        ),
    }
    closest = min(selected, key=lambda row: float(row["physical_distance"]))
    out.update(
        {
            "closest_physical_distance": float(closest["physical_distance"]),
            "closest_lag": int(closest["lag"]),
            "state_route_at_closest_x": float(closest["state_route_similarity"]),
            "action_route_at_closest_x": float(closest["action_route_similarity"]),
            "action_similarity_at_closest_x": float(closest["action_similarity"]),
            "progress_abs_at_closest_x_m": abs(float(closest["progress_delta_m"])),
        }
    )

    phys = np.asarray(
        [
            float(row["physical_distance"])
            <= thresholds["physical_distance_q25"]
            for row in selected
        ],
        bool,
    )
    route = np.asarray(
        [
            float(row["action_route_similarity"])
            >= thresholds["action_route_similarity_q75_given_physical"]
            for row in selected
        ],
        bool,
    )
    action = np.asarray(
        [
            float(row["action_distance"])
            <= thresholds["action_distance_q25_given_physical"]
            for row in selected
        ],
        bool,
    )
    no_progress = np.asarray(
        [abs(float(row["progress_delta_m"])) <= PROGRESS_M for row in selected], bool
    )
    denom = max(int(phys.sum()), 1)
    out.update(
        {
            "physical_repeat_count": int(phys.sum()),
            "route_repeat_action_repeat_rate_given_x": float(
                (phys & route & action).sum() / denom
            ),
            "route_repeat_action_different_rate_given_x": float(
                (phys & route & ~action).sum() / denom
            ),
            "route_different_action_repeat_rate_given_x": float(
                (phys & ~route & action).sum() / denom
            ),
            "route_different_action_different_rate_given_x": float(
                (phys & ~route & ~action).sum() / denom
            ),
            "joint_candidate_rate": float(
                (phys & route & action & no_progress).mean()
            ),
        }
    )
    return out


def build_episode_records(
    data: TaskData, events: list[list[dict]], thresholds: dict[str, float],
    search: str
) -> dict[str, list[dict]]:
    result = {"prefix": [], "suffix": [], "full": []}
    for episode, rows in enumerate(events):
        length = int(data.lengths[episode])
        base = {
            "task": data.key,
            "episode": int(episode),
            "scene": int(data.scenes[episode]),
            "flow_noise_seed": int(data.seeds[episode]),
            "failure": bool(data.failure[episode]),
            "failure_mode": data.modes[episode],
            "stasis": bool(data.stasis[episode]),
            "episode_length": length,
            "search": search,
        }
        windows = {
            "prefix": (0, data.prefix),
            "suffix": (length - data.prefix, length),
            "full": (0, length),
        }
        for name, (start, stop) in windows.items():
            record = dict(base)
            record["window"] = name
            record.update(aggregate_episode(rows, start, stop, thresholds))
            result[name].append(record)
    return result


def group_key(record: dict) -> str:
    return f"{record['task']}|{record['scene']}"


def stratified_test(
    records: list[dict], label_key: str, metric_names: tuple[str, ...],
    permutations: int, bootstraps: int, rng: np.random.Generator
) -> dict:
    labels = np.asarray([bool(record[label_key]) for record in records])
    values = np.column_stack(
        [[float(record[name]) for record in records] for name in metric_names]
    )
    groups = np.asarray([group_key(record) for record in records])
    valid_groups = []
    valid = np.zeros(len(records), bool)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        if labels[index].any() and (~labels[index]).any():
            valid_groups.append(index)
            valid[index] = True
    use = np.flatnonzero(valid)
    if not len(valid_groups):
        return {
            "status": "no mixed strata",
            "records": len(records),
            "positive": int(labels.sum()),
        }
    local = {original: i for i, original in enumerate(use)}
    group_local = [np.asarray([local[i] for i in index]) for index in valid_groups]
    y = labels[use]
    x = values[use]
    ranks = np.empty_like(x, dtype=np.float64)
    base = 0.0
    denominator = 0.0
    group_components = []
    for index in group_local:
        n1, n0 = int(y[index].sum()), int((~y[index]).sum())
        group_rank = np.column_stack(
            [rankdata(x[index, column]) for column in range(x.shape[1])]
        )
        ranks[index] = group_rank
        group_base = n1 * (n1 + 1) / 2.0
        wins = y[index].astype(np.float64) @ group_rank - group_base
        pairs = float(n1 * n0)
        group_components.append((wins, pairs))
        base += group_base
        denominator += pairs
    observed = (y.astype(np.float64) @ ranks - base) / denominator

    null_chunks = []
    batch_size = 100
    for first in range(0, permutations, batch_size):
        batch = min(batch_size, permutations - first)
        permuted = np.empty((batch, len(use)), np.float64)
        for index in group_local:
            order = np.argsort(rng.random((batch, len(index))), axis=1)
            permuted[:, index] = y[index][order]
        null_chunks.append((permuted @ ranks - base) / denominator)
    null = np.concatenate(null_chunks, axis=0)
    max_null = np.max(null - 0.5, axis=1)
    max_abs_null = np.max(np.abs(null - 0.5), axis=1)

    components = group_components
    bootstrap_auc = np.empty((bootstraps, len(metric_names)), np.float64)
    for draw in range(bootstraps):
        chosen = rng.integers(0, len(components), len(components))
        wins = sum((components[i][0] for i in chosen), np.zeros(len(metric_names)))
        pairs = sum(components[i][1] for i in chosen)
        bootstrap_auc[draw] = wins / pairs

    result = {}
    for column, name in enumerate(metric_names):
        effect = observed[column] - 0.5
        result[name] = {
            "auc_positive": float(observed[column]),
            "cluster_bootstrap_ci": [
                float(np.quantile(bootstrap_auc[:, column], 0.025)),
                float(np.quantile(bootstrap_auc[:, column], 0.975)),
            ],
            "permutation_p_one_sided": float(
                (1 + np.sum(null[:, column] - 0.5 >= effect - 1e-15))
                / (permutations + 1)
            ),
            "permutation_p_two_sided": float(
                (1 + np.sum(np.abs(null[:, column] - 0.5) >= abs(effect) - 1e-15))
                / (permutations + 1)
            ),
            "familywise_p_one_sided": float(
                (1 + np.sum(max_null >= effect - 1e-15)) / (permutations + 1)
            ),
            "familywise_p_two_sided": float(
                (1 + np.sum(max_abs_null >= abs(effect) - 1e-15))
                / (permutations + 1)
            ),
        }
    return {
        "status": "ok",
        "records_in_mixed_strata": int(len(use)),
        "positive_in_mixed_strata": int(y.sum()),
        "negative_in_mixed_strata": int((~y).sum()),
        "mixed_strata": len(valid_groups),
        "permutations": permutations,
        "bootstraps": bootstraps,
        "metrics": result,
    }


def stratified_auc_contrast(
    records: list[dict], label_key: str, left: str, right: str,
    permutations: int, bootstraps: int, rng: np.random.Generator
) -> dict:
    labels = np.asarray([bool(record[label_key]) for record in records])
    values = np.column_stack(
        [
            [float(record[left]) for record in records],
            [float(record[right]) for record in records],
        ]
    )
    groups = np.asarray([group_key(record) for record in records])
    components = []
    null_parts = []
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        y = labels[index]
        n1, n0 = int(y.sum()), int((~y).sum())
        if not n1 or not n0:
            continue
        ranks = np.column_stack(
            [rankdata(values[index, column]) for column in range(2)]
        )
        base = n1 * (n1 + 1) / 2.0
        wins = y.astype(np.float64) @ ranks - base
        components.append({"index": index, "y": y, "ranks": ranks, "wins": wins,
                           "pairs": float(n1 * n0), "base": base})
    if not components:
        return {"status": "no mixed strata"}
    wins = sum((item["wins"] for item in components), np.zeros(2))
    pairs = sum(item["pairs"] for item in components)
    auc = wins / pairs
    observed = float(auc[0] - auc[1])

    batch_size = 100
    for first in range(0, permutations, batch_size):
        batch = min(batch_size, permutations - first)
        batch_wins = np.zeros((batch, 2), np.float64)
        for item in components:
            y = item["y"]
            order = np.argsort(rng.random((batch, len(y))), axis=1)
            permuted = y[order]
            batch_wins += permuted @ item["ranks"] - item["base"]
        null_parts.append(batch_wins[:, 0] / pairs - batch_wins[:, 1] / pairs)
    null = np.concatenate(null_parts)

    bootstrap = np.empty(bootstraps, np.float64)
    for draw in range(bootstraps):
        chosen = rng.integers(0, len(components), len(components))
        draw_wins = sum((components[i]["wins"] for i in chosen), np.zeros(2))
        draw_pairs = sum(components[i]["pairs"] for i in chosen)
        draw_auc = draw_wins / draw_pairs
        bootstrap[draw] = draw_auc[0] - draw_auc[1]
    return {
        "status": "ok",
        "left": left,
        "right": right,
        "left_auc": float(auc[0]),
        "right_auc": float(auc[1]),
        "auc_delta": observed,
        "cluster_bootstrap_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "permutation_p_left_greater": float(
            (1 + np.sum(null >= observed - 1e-15)) / (permutations + 1)
        ),
        "permutation_p_two_sided": float(
            (1 + np.sum(np.abs(null) >= abs(observed) - 1e-15))
            / (permutations + 1)
        ),
    }


def lag_summary(rows: list[dict]) -> dict:
    out = {}
    for label, subset in (
        ("all", rows),
        ("success", [row for row in rows if not bool(row["failure"])]),
        ("failure", [row for row in rows if bool(row["failure"])]),
    ):
        counts: dict[str, int] = {}
        for row in subset:
            key = str(int(row["lag"]))
            counts[key] = counts.get(key, 0) + 1
        out[label] = {
            "n": len(subset),
            "counts": counts,
            "minimum_lag_fraction": float(
                sum(int(row["lag"]) == 2 for row in subset) / max(len(subset), 1)
            ),
        }
    return out


def nearest_external_query(
    data: TaskData, reference_episode: int, reference_query: int
) -> tuple[int, int] | None:
    candidates = np.flatnonzero(
        (data.scenes == data.scenes[reference_episode])
        & (np.arange(len(data.summaries)) != reference_episode)
    )
    if not len(candidates):
        return None
    ref = data.qpos_z[reference_episode][reference_query]
    best: tuple[float, int, int] | None = None
    for episode in candidates:
        distance = np.sqrt(np.mean(np.square(data.qpos_z[episode] - ref), axis=1))
        query = int(np.argmin(distance))
        item = (float(distance[query]), int(episode), query)
        if best is None or item < best:
            best = item
    assert best is not None
    return best[1], best[2]


def detect_slips_and_attempts(
    data: TaskData, all_history: SearchConfig
) -> tuple[list[dict], list[dict]]:
    slips: list[dict] = []
    attempts: list[dict] = []
    first_attempts: dict[tuple[int, int], tuple[int, int]] = {}
    onset_catalog: list[dict] = []

    for episode, sim in enumerate(data.sim):
        for target_index, target in enumerate(data.targets):
            lo, hi = int(target["lo"]), int(target["hi"])
            position = sim[:, lo:hi]
            lifted = position[:, 2] - position[0, 2] > LIFT_M
            previous = np.r_[False, lifted[:-1]]
            onsets = np.flatnonzero(lifted & ~previous)
            for attempt_index, onset in enumerate(onsets, start=1):
                pre_query = max(0, int(onset) - 1)
                onset_catalog.append(
                    {
                        "episode": episode,
                        "target_index": target_index,
                        "target": target["name"],
                        "attempt_index": attempt_index,
                        "onset_query": int(onset),
                        "pre_query": pre_query,
                    }
                )
                key = (episode, target_index)
                if attempt_index == 1:
                    first_attempts[key] = (episode, pre_query)
                    continue
                first_episode, first_query = first_attempts[key]
                if pre_query - first_query < 1:
                    continue
                own = pair_metrics(
                    data, episode, pre_query, first_episode, first_query
                )
                record = {
                    "task": data.key,
                    "episode": int(episode),
                    "scene": int(data.scenes[episode]),
                    "flow_noise_seed": int(data.seeds[episode]),
                    "failure": bool(data.failure[episode]),
                    "failure_mode": data.modes[episode],
                    "target": target["name"],
                    "attempt_index": attempt_index,
                    "first_pre_query": first_query,
                    "repeat_pre_query": pre_query,
                    **own,
                }
                attempts.append(record)

            falling = np.flatnonzero(~lifted & np.r_[False, lifted[:-1]])
            goal = np.asarray(target["goal"])
            for query in falling:
                query = int(query)
                if query < all_history.min_lag:
                    continue
                off_goal = float(np.linalg.norm(position[query] - goal)) > GOAL_M
                if not off_goal:
                    continue
                own = history_match(data, episode, query, all_history)
                baseline_index = nearest_external_query(
                    data, int(own["past_episode"]), int(own["past_query"])
                )
                record = {
                    "task": data.key,
                    "episode": int(episode),
                    "scene": int(data.scenes[episode]),
                    "flow_noise_seed": int(data.seeds[episode]),
                    "failure": bool(data.failure[episode]),
                    "failure_mode": data.modes[episode],
                    "target": target["name"],
                    "post_slip_query": query,
                    **own,
                }
                if baseline_index is not None:
                    control_episode, control_query = baseline_index
                    control = pair_metrics(
                        data,
                        int(own["past_episode"]),
                        int(own["past_query"]),
                        control_episode,
                        control_query,
                    )
                    record.update(
                        {
                            "seed_control_episode": control_episode,
                            "seed_control_query": control_query,
                            "seed_control_physical_distance": control[
                                "physical_distance"
                            ],
                            "seed_control_action_route_similarity": control[
                                "action_route_similarity"
                            ],
                            "seed_control_action_similarity": control[
                                "action_similarity"
                            ],
                            "route_return_minus_seed_control": float(
                                own["action_route_similarity"]
                                - control["action_route_similarity"]
                            ),
                            "action_return_minus_seed_control": float(
                                own["action_similarity"]
                                - control["action_similarity"]
                            ),
                        }
                    )
                slips.append(record)

    # Cross-seed baselines for repeated attempts use another episode's first
    # attempt in the same task, scene, and target whenever one exists.
    for record in attempts:
        episode = int(record["episode"])
        target_name = record["target"]
        first_query = int(record["first_pre_query"])
        candidates = [
            item
            for item in onset_catalog
            if item["attempt_index"] == 1
            and item["target"] == target_name
            and item["episode"] != episode
            and data.scenes[item["episode"]] == data.scenes[episode]
        ]
        if not candidates:
            continue
        ref = data.qpos_z[episode][first_query]
        best = min(
            candidates,
            key=lambda item: float(
                np.sqrt(
                    np.mean(
                        np.square(
                            data.qpos_z[item["episode"]][item["pre_query"]] - ref
                        )
                    )
                )
            ),
        )
        control = pair_metrics(
            data, episode, first_query, int(best["episode"]), int(best["pre_query"])
        )
        record.update(
            {
                "seed_control_episode": int(best["episode"]),
                "seed_control_query": int(best["pre_query"]),
                "seed_control_physical_distance": control["physical_distance"],
                "seed_control_action_route_similarity": control[
                    "action_route_similarity"
                ],
                "seed_control_action_similarity": control["action_similarity"],
                "route_return_minus_seed_control": float(
                    record["action_route_similarity"]
                    - control["action_route_similarity"]
                ),
                "action_return_minus_seed_control": float(
                    record["action_similarity"] - control["action_similarity"]
                ),
            }
        )
    return slips, attempts


def summarize_events(events: list[dict]) -> dict:
    if not events:
        return {"n": 0}
    numeric = (
        "physical_distance",
        "state_route_similarity",
        "action_route_similarity",
        "action_similarity",
        "progress_delta_m",
        "joint_return_score",
        "seed_control_physical_distance",
        "route_return_minus_seed_control",
        "action_return_minus_seed_control",
    )
    summary = {
        "n": len(events),
        "failures": int(sum(bool(row["failure"]) for row in events)),
        "successes": int(sum(not bool(row["failure"]) for row in events)),
        "by_mode": {},
        "medians": {},
    }
    for mode in sorted({str(row["failure_mode"]) for row in events}):
        summary["by_mode"][mode] = sum(
            str(row["failure_mode"]) == mode for row in events
        )
    for name in numeric:
        values = [float(row[name]) for row in events if name in row]
        if values:
            summary["medians"][name] = float(np.median(values))
    controlled = [row for row in events if "route_return_minus_seed_control" in row]
    if controlled:
        route_delta = np.asarray(
            [float(row["route_return_minus_seed_control"]) for row in controlled]
        )
        action_delta = np.asarray(
            [float(row["action_return_minus_seed_control"]) for row in controlled]
        )

        def signed_test(values: np.ndarray) -> float:
            if np.allclose(values, 0):
                return 1.0
            return float(
                wilcoxon(values, alternative="greater", method="approx").pvalue
            )

        summary["same_seed_vs_cross_seed_control"] = {
            "n": len(controlled),
            "route_delta_median": float(np.median(route_delta)),
            "route_positive_fraction": float(np.mean(route_delta > 0)),
            "route_wilcoxon_p_one_sided": signed_test(route_delta),
            "action_delta_median": float(np.median(action_delta)),
            "action_positive_fraction": float(np.mean(action_delta > 0)),
            "action_wilcoxon_p_one_sided": signed_test(action_delta),
        }
    return summary


def mode_summary(records: list[dict]) -> dict:
    out = {}
    for mode in sorted({str(row["failure_mode"]) for row in records}):
        subset = [row for row in records if str(row["failure_mode"]) == mode]
        out[mode] = {
            "n": len(subset),
            **{
                name: float(np.mean([float(row[name]) for row in subset]))
                for name in CORE_METRICS
            },
        }
    return out


def quadrant_summary(records: list[dict]) -> dict:
    fields = (
        "route_repeat_action_repeat_rate_given_x",
        "route_repeat_action_different_rate_given_x",
        "route_different_action_repeat_rate_given_x",
        "route_different_action_different_rate_given_x",
        "joint_candidate_rate",
    )
    out = {}
    for label, flag in (("success", False), ("failure", True)):
        subset = [row for row in records if bool(row["failure"]) == flag]
        out[label] = {
            "n": len(subset),
            **{
                field: float(np.mean([float(row[field]) for row in subset]))
                for field in fields
            },
        }
    return out


def strict_candidate_summary(records: list[dict]) -> dict:
    """Count rare strict-loop pairs without hiding them behind episode AUC ties."""
    out = {}
    for label, flag in (("success", False), ("failure", True)):
        subset = [row for row in records if bool(row["failure"]) == flag]
        candidate_pairs = sum(
            float(row["backward_joint_candidate_rate"]) * int(row["query_pairs"])
            for row in subset
        )
        out[label] = {
            "episodes": len(subset),
            "episodes_with_candidate": sum(
                float(row["backward_joint_candidate_rate"]) > 0 for row in subset
            ),
            "query_pairs": sum(int(row["query_pairs"]) for row in subset),
            "candidate_pairs": int(round(candidate_pairs)),
        }
    return out


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render_report(result: dict) -> str:
    lines = [
        "# Replanning-reset trap audit",
        "",
        "This audit first matches each query to its physically nearest eligible",
        "history query, then asks whether aligned HB routing and the 10x7 action",
        "chunk also return without progress.",
        "",
        "## Coverage",
        "",
        "| task | episodes | failures | shared prefix | targets |",
        "|---|---:|---:|---:|---|",
    ]
    for row in result["coverage"]:
        lines.append(
            "| %s | %d | %d | %d | %s |"
            % (
                row["task"],
                row["episodes"],
                row["failures"],
                row["shared_prefix"],
                ", ".join(row["targets"]) or "none",
            )
        )

    primary = result["primary_prefix"]
    lines.extend(
        [
            "",
            "## Primary equal-prefix result",
            "",
            "AUC above 0.5 means the score is larger in failures after comparing",
            "only within task x initial-state strata.",
            "",
            "| endpoint | failure AUC | cluster 95% CI | p one-sided | family-wise p |",
            "|---|---:|---|---:|---:|",
        ]
    )
    if primary.get("status") == "ok":
        for name in CORE_METRICS:
            row = primary["metrics"][name]
            lines.append(
                "| %s | %s | [%s, %s] | %.4f | %.4f |"
                % (
                    METRIC_LABELS[name],
                    fmt(row["auc_positive"]),
                    fmt(row["cluster_bootstrap_ci"][0]),
                    fmt(row["cluster_bootstrap_ci"][1]),
                    row["permutation_p_one_sided"],
                    row["familywise_p_one_sided"],
                )
            )
        joint = primary["metrics"]["joint_return_p90"]
        if joint["permutation_p_one_sided"] < 0.05 and joint["auc_positive"] > 0.5:
            bottom = (
                "The prespecified joint-return endpoint is enriched in failures "
                "within the equal prefix."
            )
        else:
            bottom = (
                "The prespecified joint-return endpoint is not confirmed in the "
                "equal-prefix comparison."
            )
        lines.extend(["", "**Primary reading:** " + bottom])

    lags = result["selected_lags"]
    lines.extend(
        [
            "",
            "## Backward-return topology control",
            "",
            "A matched history point is a true backward return only if it is more",
            "similar to the current query than the immediately preceding query.",
            "This control was added after the continuous primary score was seen.",
            "",
            "The physical nearest-neighbor search selected the minimum allowed lag",
            "(lag 2) for %.1f%% of all pairs, %.1f%% of successes, and %.1f%% of failures."
            % (
                100 * lags["all"]["minimum_lag_fraction"],
                100 * lags["success"]["minimum_lag_fraction"],
                100 * lags["failure"]["minimum_lag_fraction"],
            ),
            "",
            "| topology endpoint | failure AUC | 95% CI | p one-sided | family-wise p |",
            "|---|---:|---|---:|---:|",
        ]
    )
    topology = result["topology_control"]
    if topology.get("status") == "ok":
        for name in TOPOLOGY_METRICS:
            row = topology["metrics"][name]
            lines.append(
                "| %s | %s | [%s, %s] | %.4f | %.4f |"
                % (
                    METRIC_LABELS[name],
                    fmt(row["auc_positive"]),
                    fmt(row["cluster_bootstrap_ci"][0]),
                    fmt(row["cluster_bootstrap_ci"][1]),
                    row["permutation_p_one_sided"],
                    row["familywise_p_one_sided"],
                )
            )
        strict = topology["metrics"]["backward_joint_candidate_rate"]
        if strict["auc_positive"] > 0.5 and strict["permutation_p_one_sided"] < 0.05:
            lines.extend(
                [
                    "",
                    "Strict backward-loop candidates are enriched in failures; this",
                    "supports nonlocal return beyond ordinary adjacent persistence.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Strict backward-loop candidates are not enriched. The positive",
                    "continuous return score should therefore be read as persistence/",
                    "low-motion geometry, not as a confirmed nonlocal routing loop.",
                ]
            )

        candidate_counts = result["strict_candidate_counts"]
        success_counts = candidate_counts["success"]
        failure_counts = candidate_counts["failure"]
        lines.extend(
            [
                "",
                "In the equal-prefix windows, the strict rule selected %d/%d failure"
                % (
                    failure_counts["candidate_pairs"],
                    failure_counts["query_pairs"],
                ),
                "pairs across %d/%d episodes and %d/%d success pairs across %d/%d episodes."
                % (
                    failure_counts["episodes_with_candidate"],
                    failure_counts["episodes"],
                    success_counts["candidate_pairs"],
                    success_counts["query_pairs"],
                    success_counts["episodes_with_candidate"],
                    success_counts["episodes"],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Increment and per-task checks",
            "",
            "| AUC contrast | delta | 95% CI | p(left > right) |",
            "|---|---:|---|---:|",
        ]
    )
    for name, row in result["auc_contrasts"].items():
        if row.get("status") != "ok":
            continue
        lines.append(
            "| %s | %+.3f | [%+.3f, %+.3f] | %.4f |"
            % (
                name,
                row["auc_delta"],
                row["cluster_bootstrap_ci"][0],
                row["cluster_bootstrap_ci"][1],
                row["permutation_p_left_greater"],
            )
        )
    lines.extend(
        [
            "",
            "| task | joint AUC | strict backward-loop AUC |",
            "|---|---:|---:|",
        ]
    )
    for task, test in result["per_task_primary"].items():
        if test.get("status") != "ok":
            lines.append(f"| {task} | {test.get('status', 'unavailable')} | - |")
            continue
        lines.append(
            "| %s | %s | %s |"
            % (
                task,
                fmt(test["metrics"]["joint_return_p90"]["auc_positive"]),
                fmt(
                    test["metrics"]["backward_joint_candidate_rate"][
                        "auc_positive"
                    ]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Search-range sensitivity",
            "",
            "| history search | joint AUC | 95% CI | p one-sided |",
            "|---|---:|---|---:|",
        ]
    )
    for name, test in result["sensitivity"].items():
        if test.get("status") != "ok":
            continue
        row = test["metrics"]["joint_return_p90"]
        lines.append(
            "| %s | %s | [%s, %s] | %.4f |"
            % (
                name,
                fmt(row["auc_positive"]),
                fmt(row["cluster_bootstrap_ci"][0]),
                fmt(row["cluster_bootstrap_ci"][1]),
                row["permutation_p_one_sided"],
            )
        )

    suffix = result["terminal_suffix"]
    lines.extend(
        [
            "",
            "## Same-length terminal window",
            "",
            "This window can see late slips but is phase-confounded and is not a",
            "prediction test.",
            "",
        ]
    )
    if suffix.get("status") == "ok":
        row = suffix["metrics"]["joint_return_p90"]
        lines.append(
            "Joint-return failure AUC %s [%s, %s], p=%.4f."
            % (
                fmt(row["auc_positive"]),
                fmt(row["cluster_bootstrap_ci"][0]),
                fmt(row["cluster_bootstrap_ci"][1]),
                row["permutation_p_one_sided"],
            )
        )

    quadrants = result["quadrants"]
    lines.extend(
        [
            "",
            "## Physically recurrent pair quadrants",
            "",
            "Thresholds are calibrated from successful equal-prefix pairs. Values",
            "are mean within-episode rates conditional on physical recurrence.",
            "",
            "| outcome | route repeat / action repeat | route repeat / action different | route different / action repeat | route different / action different | joint no-progress rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("success", "failure"):
        row = quadrants[label]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                label,
                fmt(row["route_repeat_action_repeat_rate_given_x"]),
                fmt(row["route_repeat_action_different_rate_given_x"]),
                fmt(row["route_different_action_repeat_rate_given_x"]),
                fmt(row["route_different_action_different_rate_given_x"]),
                fmt(row["joint_candidate_rate"]),
            )
        )

    lines.extend(["", "## Failure modes", ""])
    lines.append(
        "The failure-only comparisons are descriptive because event-like modes are rare."
    )
    lines.extend(
        [
            "",
            "| mode | n | joint-return mean | physical-return mean | action-route mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode, row in result["failure_modes"].items():
        lines.append(
            "| %s | %d | %s | %s | %s |"
            % (
                mode,
                row["n"],
                fmt(row["joint_return_p90"]),
                fmt(row["physical_return_p90"]),
                fmt(row["action_route_return_p90"]),
            )
        )

    lines.extend(["", "## Slip and repeated-attempt events", ""])
    for label, key in (
        ("Off-goal lift-loss proxy events", "slips"),
        ("Repeated lift attempts", "attempts"),
    ):
        row = result[key]
        lines.append(
            "- %s: n=%d (%d failure, %d success)."
            % (label, row["n"], row.get("failures", 0), row.get("successes", 0))
        )
        control = row.get("same_seed_vs_cross_seed_control")
        if control:
            lines.append(
                "  Same-seed return minus cross-seed physical-match median: route %+.3f "
                "(p=%.4f), action %+.3f (p=%.4f)."
                % (
                    control["route_delta_median"],
                    control["route_wilcoxon_p_one_sided"],
                    control["action_delta_median"],
                    control["action_wilcoxon_p_one_sided"],
                )
            )
            lines.append(
                "  Median physical distance: within-episode return %.3f versus "
                "cross-seed control %.3f."
                % (
                    row["medians"].get("physical_distance", float("nan")),
                    row["medians"].get(
                        "seed_control_physical_distance", float("nan")
                    ),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- The physical proxy is MuJoCo qpos, not a stored RGB or vision embedding.",
            "- This model has recorded state-token and action-token HB routing, not an independent visual-token MoE stream.",
            "- Every within-episode comparison holds the rollout noise seed fixed, but exact snapshot re-inference across multiple seeds requires images or simulator replay that are absent here.",
            "- Lift-loss is a chunk-level kinematic proxy, not a semantic slip annotation.",
            "- Routing/action recurrence is associative. A causal memory intervention would require online replay.",
            "- Terminal-window and failure-mode results must not be called early warning.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    a = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], np.uint8)
    b = np.asarray([[4, 3, 2, 1], [5, 6, 9, 10]], np.uint8)
    assert abs(top4_overlap(a, b) - 0.75) < 1e-12
    assert abs(geometric_mean(1.0, 1.0, 1.0, 1.0) - 1.0) < 1e-12
    records = []
    for scene in range(4):
        for episode in range(8):
            positive = episode >= 4
            records.append(
                {
                    "task": "synthetic",
                    "scene": scene,
                    "positive": positive,
                    "score": float(episode) + (1.0 if positive else 0.0),
                }
            )
    result = stratified_test(
        records,
        "positive",
        ("score",),
        199,
        199,
        np.random.default_rng(7),
    )
    assert result["metrics"]["score"]["auc_positive"] == 1.0
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_records = {
        config.name: {"prefix": [], "suffix": [], "full": []}
        for config in SEARCHES
    }
    threshold_output: dict[str, dict] = {}
    coverage = []
    query_rows: list[dict] = []
    slip_rows: list[dict] = []
    attempt_rows: list[dict] = []

    for run in discover_runs(args.cache_root):
        data = load_task(run, args.cache_root)
        print(
            f"loaded {data.key}: {len(data.summaries)} episodes, "
            f"{int(data.failure.sum())} failures, prefix {data.prefix}",
            flush=True,
        )
        coverage.append(
            {
                "task": data.key,
                "episodes": len(data.summaries),
                "failures": int(data.failure.sum()),
                "shared_prefix": data.prefix,
                "targets": [target["name"] for target in data.targets],
                "control_steps": int(data.lengths.sum()),
            }
        )
        task_events: dict[str, list[list[dict]]] = {}
        for config in SEARCHES:
            events = compute_query_matches(data, config)
            task_events[config.name] = events
            thresholds = calibrate_thresholds(data, events, config)
            threshold_output[f"{data.key}|{config.name}"] = thresholds
            records = build_episode_records(data, events, thresholds, config.name)
            for window in ("prefix", "suffix", "full"):
                all_records[config.name][window].extend(records[window])
            if config.name == "lag2_8":
                for rows in events:
                    query_rows.extend(rows)
            print(f"  matched {config.name}", flush=True)
        slips, attempts = detect_slips_and_attempts(data, SEARCHES[-1])
        slip_rows.extend(slips)
        attempt_rows.extend(attempts)
        print(
            f"  detected {len(slips)} off-goal lift losses and "
            f"{len(attempts)} repeated lift attempts",
            flush=True,
        )
        del task_events, data
        gc.collect()

    primary_records = all_records["lag2_8"]["prefix"]
    primary = stratified_test(
        primary_records,
        "failure",
        CORE_METRICS,
        args.permutations,
        args.bootstraps,
        rng,
    )
    topology = stratified_test(
        primary_records,
        "failure",
        TOPOLOGY_METRICS,
        args.permutations,
        args.bootstraps,
        rng,
    )
    task_primary = {}
    for task in sorted({str(row["task"]) for row in primary_records}):
        subset = [row for row in primary_records if str(row["task"]) == task]
        if not any(bool(row["failure"]) for row in subset):
            task_primary[task] = {"status": "all success", "records": len(subset)}
            continue
        task_primary[task] = stratified_test(
            subset,
            "failure",
            ("joint_return_p90", "backward_joint_candidate_rate"),
            args.permutations,
            args.bootstraps,
            rng,
        )
    contrasts = {
        "action_route_minus_physical": stratified_auc_contrast(
            primary_records,
            "failure",
            "action_route_return_p90",
            "physical_return_p90",
            args.permutations,
            args.bootstraps,
            rng,
        ),
        "joint_minus_physical": stratified_auc_contrast(
            primary_records,
            "failure",
            "joint_return_p90",
            "physical_return_p90",
            args.permutations,
            args.bootstraps,
            rng,
        ),
        "joint_minus_action": stratified_auc_contrast(
            primary_records,
            "failure",
            "joint_return_p90",
            "action_return_p90",
            args.permutations,
            args.bootstraps,
            rng,
        ),
        "action_route_minus_action": stratified_auc_contrast(
            primary_records,
            "failure",
            "action_route_return_p90",
            "action_return_p90",
            args.permutations,
            args.bootstraps,
            rng,
        ),
    }
    sensitivity = {}
    for config in SEARCHES:
        sensitivity[config.name] = stratified_test(
            all_records[config.name]["prefix"],
            "failure",
            ("joint_return_p90",),
            args.permutations,
            args.bootstraps,
            rng,
        )
    suffix = stratified_test(
        all_records["lag2_8"]["suffix"],
        "failure",
        CORE_METRICS,
        args.permutations,
        args.bootstraps,
        rng,
    )

    failure_full = [
        row for row in all_records["lag2_8"]["full"] if bool(row["failure"])
    ]
    reached_records = [
        {**row, "reached_then_lost": row["failure_mode"] == "reached_then_lost"}
        for row in failure_full
    ]
    event_records = [
        {
            **row,
            "event_mode": row["failure_mode"]
            in {"reached_then_lost", "misplaced"},
        }
        for row in failure_full
    ]
    failure_type_tests = {
        "reached_then_lost_vs_other_failure": stratified_test(
            reached_records,
            "reached_then_lost",
            ("joint_return_p90",),
            args.permutations,
            args.bootstraps,
            rng,
        ),
        "event_mode_vs_other_failure": stratified_test(
            event_records,
            "event_mode",
            ("joint_return_p90",),
            args.permutations,
            args.bootstraps,
            rng,
        ),
    }

    result = {
        "schema": "replanning-reset-trap/1",
        "source": str(args.cache_root),
        "run_id": "right-16x32",
        "preregistered_spec": str(args.out_dir / "PREREG.md"),
        "definitions": {
            "physical_proxy": "within-task standardized MuJoCo qpos; simulation time and qvel excluded",
            "history_match": "nearest physical history query within frozen lag range",
            "state_route": "actual top-4 overlap, aligned HB layers, denoise=0, state token",
            "action_route": "actual top-4 overlap, aligned HB layer x denoise x action token",
            "same_seed": "flow_noise_seed is fixed within each rollout",
            "primary_window": "longest equal prefix shared by all episodes in each task",
            "primary_endpoint": "90th percentile query-level joint return score",
        },
        "coverage": coverage,
        "thresholds": threshold_output,
        "primary_prefix": primary,
        "topology_control": topology,
        "per_task_primary": task_primary,
        "auc_contrasts": contrasts,
        "selected_lags": lag_summary(query_rows),
        "strict_candidate_counts": strict_candidate_summary(primary_records),
        "sensitivity": sensitivity,
        "terminal_suffix": suffix,
        "quadrants": quadrant_summary(primary_records),
        "failure_modes": mode_summary(failure_full),
        "failure_type_tests": failure_type_tests,
        "slips": summarize_events(slip_rows),
        "attempts": summarize_events(attempt_rows),
        "limitations": [
            "RGB frames and vision embeddings were not stored",
            "exact snapshot re-inference across multiple seeds requires replay",
            "state-token routing is not an independent visual-token route",
            "terminal and failure-only analyses are descriptive",
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(result))
    write_csv(args.out_dir / "query_matches.csv", query_rows)
    write_csv(args.out_dir / "episode_metrics.csv", all_records["lag2_8"]["prefix"])
    write_csv(args.out_dir / "slip_events.csv", slip_rows)
    write_csv(args.out_dir / "attempt_events.csv", attempt_rows)
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
