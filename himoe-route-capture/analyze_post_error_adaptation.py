#!/usr/bin/env python3
"""Second-order (post-error adaptation) landmark feasibility + state/action route decomposition.

Design note in ``analysis/post-error-adaptation-20260828/PREREG.md``.

Stage A (``gate``)   enumerates absolute-threshold first-order physical-event
                     landmarks over all 2560 rollouts and reports, for every
                     landmark, the dual-outcome risk set that a second-order
                     "did the policy adapt" test would need.
Stage B (``ladder``) fits the prespecified nested model ladder
                     physical -> +route -> +action -> +joint(+hidden) on the
                     landmark that scores best in Stage A, together with a
                     one-dimensional onset-index sentinel.
Stage C (``tokens``) is descriptive only: it decomposes HB routing into the
                     state token (suffix token 0) and the action tokens
                     (suffix tokens 1..10) and reports their denoise-axis
                     structure and their dynamics around confirmed long-task
                     physical events.

No causal claim is made anywhere.  Stage C is explicitly descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/post-error-adaptation-20260828"

MOVED_M = 0.03          # object counts as a task target if success mean travel exceeds this
LIFT_M = 0.01           # object counts as lifted at this height above its own start
GOAL_M = 0.05           # success-goal neighbourhood radius
BLOCK = 512             # zarr read block
N_DENOISE = 10
N_TOKENS = 11
N_LAYERS = 8
N_EXPERTS = 32
TOPK = 4

SHORT = {
    "libero_goal/open_the_middle_drawer_of_the_cabinet": "goal/mid_drawer",
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside": "goal/top_drawer",
    "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "long/SCENE8",
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": "spatial/ramekin",
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": "spatial/stove",
}

# Stage A gate thresholds, fixed before any routing was touched (see PREREG.md S4).
GATE = {
    "min_events": 60,
    "min_per_class": 20,
    "min_mixed_clusters": 6,
    "min_events_in_mixed_clusters": 60,
    "min_per_class_in_mixed_clusters": 20,
    "max_event_rate": 0.60,          # above this the "landmark" is not an event
    "min_setback_fraction": 0.50,    # event must actually lose ground on the goal
}


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    p.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    p.add_argument("--stage", choices=("all", "gate", "ladder", "tokens"), default="all")
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--route-task", default="libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    stops = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [len(mask)]
    return list(zip(starts, stops))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC for P(score | label==1) > P(score | label==0)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(int)
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cluster_bootstrap_ci(values: np.ndarray, labels: np.ndarray, clusters: np.ndarray,
                         rng: np.random.Generator, draws: int) -> tuple[float, float]:
    uniq = np.unique(clusters)
    out = []
    for _ in range(draws):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in pick])
        a = auc(values[idx], labels[idx])
        if not math.isnan(a):
            out.append(a)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def hellinger(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.maximum(p, 0.0) * np.maximum(q, 0.0)).sum(axis=axis)


def top4_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean fraction of shared top-4 experts over all leading axes."""
    return (a[..., :, None] == b[..., None, :]).sum(axis=(-1, -2)) / float(TOPK)


# --------------------------------------------------------------------------- #
# client-side loading
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    index: int
    init_state_id: int
    flow_noise_seed: int
    success: bool
    length: int
    eef: np.ndarray
    gripper: np.ndarray
    gripper_cmd: np.ndarray
    sim: np.ndarray


@dataclass
class Task:
    key: str
    run: pathlib.Path
    episodes: list[Episode]
    targets: list[dict] = field(default_factory=list)
    distractors: list[dict] = field(default_factory=list)


def discover_runs(cache_root: pathlib.Path) -> list[pathlib.Path]:
    runs = [s.parents[1] for s in sorted(cache_root.glob("libero_*/*/right-16x32/client/summaries.json"))
            if (s.parents[1] / "client/sim_layout.json").exists()]
    if not runs:
        raise RuntimeError(f"no right-16x32 LIBERO runs under {cache_root}")
    return runs


def load_task(run: pathlib.Path, cache_root: pathlib.Path) -> Task:
    key = str(run.relative_to(cache_root).parent)
    summaries = sorted(json.loads((run / "client/summaries.json").read_text()),
                       key=lambda r: int(r["episode_index"]))
    layout = json.loads((run / "client/sim_layout.json").read_text())
    episodes = []
    for row in summaries:
        path = run / "client" / ("episode_%02d.npz" % int(row["episode_index"]))
        with np.load(path, allow_pickle=False) as z:
            state = np.asarray(z["state"], np.float64)
            sim = np.asarray(z["sim_state"], np.float64)
            act = np.asarray(z["actions"], np.float64)
        if not (len(state) == len(sim) == len(act) == int(row["inference_calls"])):
            raise ValueError(f"length mismatch in {path}")
        episodes.append(Episode(
            index=int(row["episode_index"]), init_state_id=int(row["init_state_id"]),
            flow_noise_seed=int(row["flow_noise_seed"]), success=bool(row["success"]),
            length=len(state), eef=state[:, :3], gripper=state[:, 6:8].mean(axis=1),
            gripper_cmd=act[:, :, 6].mean(axis=1), sim=sim))
    task = Task(key=key, run=run, episodes=episodes)
    succ = [i for i, e in enumerate(episodes) if e.success]
    for joint in layout["joints"]:
        lo, hi = int(joint["state_lo"]), int(joint["state_hi"])
        if joint["is_robot"] or hi - lo != 7:
            continue
        sl = slice(lo, lo + 3)
        moved = float(np.mean([np.linalg.norm(episodes[i].sim[-1, sl] - episodes[i].sim[0, sl])
                               for i in succ])) if succ else 0.0
        rec = {"name": str(joint["joint"]), "lo": lo, "hi": lo + 3,
               "moved_success_mean_m": moved}
        if moved > MOVED_M and succ:
            rec["terminal"] = np.stack([episodes[i].sim[-1, sl] for i in succ])
            rec["success_index"] = np.asarray(succ, int)
            task.targets.append(rec)
        else:
            task.distractors.append(rec)
    return task


def goal_distance(task: Task, episode: Episode, target: dict) -> np.ndarray:
    """Distance to the nearest success terminal pose, leaving out same-init and same-seed successes."""
    keep = np.asarray([task.episodes[j].init_state_id != episode.init_state_id
                       and task.episodes[j].flow_noise_seed != episode.flow_noise_seed
                       for j in target["success_index"]], bool)
    refs = target["terminal"][keep] if keep.any() else target["terminal"]
    pos = episode.sim[:, target["lo"]:target["hi"]]
    return np.linalg.norm(pos[:, None, :] - refs[None, :, :], axis=-1).min(axis=1)


# --------------------------------------------------------------------------- #
# Stage A: first-order landmark inventory
# --------------------------------------------------------------------------- #
def landmark_events(task: Task, episode: Episode, family: str, thr: float) -> list[dict]:
    """Return every onset of a first-order physical-event landmark in this rollout.

    ``onset`` is the query index at which the event is first *observable* in the
    cache.  Because ``sim_state[k]`` is recorded before chunk k executes, the
    chunk that caused the event is ``onset - 1`` with +-1 chunk uncertainty.
    """
    out = []
    for target in task.targets:
        pos = episode.sim[:, target["lo"]:target["hi"]]
        gd = goal_distance(task, episode, target)
        best = np.minimum.accumulate(gd)
        zrel = pos[:, 2] - pos[0, 2]
        de = np.linalg.norm(episode.eef - pos, axis=1)
        step = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
        T = episode.length
        onsets: list[int] = []
        if family == "liftloss":            # object stops being lifted, away from the goal
            lifted = zrel > LIFT_M
            loss = np.flatnonzero(~lifted & np.r_[False, lifted[:-1]])
            onsets = [int(k) for k in loss if gd[k] > GOAL_M and (zrel[k - 1] - zrel[k]) >= thr]
        elif family == "heightloss":        # held object descends, away from the goal
            for k in range(1, T):
                if (zrel[k - 1] - zrel[k] >= thr and zrel[k - 1] > LIFT_M
                        and gd[k] > GOAL_M and episode.gripper_cmd[k - 1] > 0):
                    onsets.append(k)
        elif family == "drop_separation":   # descends AND the gripper separates from it
            for k in range(1, T):
                if (zrel[k - 1] - zrel[k] >= thr and zrel[k - 1] > LIFT_M and gd[k] > GOAL_M
                        and episode.gripper_cmd[k - 1] > 0 and de[k] > de[k - 1] + 0.005):
                    onsets.append(k)
        elif family == "goal_regression":   # object loses ground it had already gained
            fire = (gd - best) >= thr
            onsets = [int(k) for k in np.flatnonzero(fire & ~np.r_[False, fire[:-1]])]
        elif family == "goal_nbr_loss":     # object reached the goal neighbourhood and left it
            inside = gd <= GOAL_M
            if inside.any():
                first = int(np.flatnonzero(inside)[0])
                left = np.flatnonzero(gd[first:] > GOAL_M + thr)
                if len(left):
                    onsets = [int(first + left[0])]
        elif family == "release_offgoal":   # gripper opens while the object is up and off-goal
            opening = np.r_[False, (episode.gripper_cmd[1:] <= 0) & (episode.gripper_cmd[:-1] > 0)]
            onsets = [int(k) for k in np.flatnonzero(opening & (zrel > LIFT_M) & (gd > GOAL_M))]
        elif family == "grasp_fail":        # closed near the object, released, never lifted it
            run = (episode.gripper_cmd > 0) & (de <= thr)
            for s, e in runs_of_true(run):
                if e >= T:
                    continue
                if (zrel[s:e + 1].max() - zrel[s]) < 0.005:
                    onsets.append(int(e))
        elif family == "eef_approach_leave":
            near = de <= GOAL_M
            if near.any():
                first = int(np.flatnonzero(near)[0])
                left = np.flatnonzero(de[first:] > thr)
                if len(left):
                    onsets = [int(first + left[0])]
        elif family == "push_ungrasped":    # object moves while the gripper is commanded open
            hit = np.flatnonzero((step >= thr) & (episode.gripper_cmd <= 0))
            onsets = [int(hit[0])] if len(hit) else []
        else:
            raise ValueError(f"unknown landmark family {family}")
        for k in onsets:
            if k < 1 or k >= T:
                continue
            out.append({"target": target["name"], "onset": int(k),
                        "goal_dist": float(gd[k]), "pre_best": float(best[k - 1]),
                        "regression_m": float(gd[k] - best[k - 1]),
                        "height_m": float(zrel[k]), "eef_obj_m": float(de[k]),
                        "severity_m": float(max(gd[k] - best[k - 1], 0.0))})
    out.sort(key=lambda r: r["onset"])
    return out


LANDMARK_GRID = [
    ("liftloss", 0.0), ("liftloss", 0.01),
    ("heightloss", 0.02), ("heightloss", 0.03), ("heightloss", 0.05),
    ("drop_separation", 0.01), ("drop_separation", 0.02), ("drop_separation", 0.03),
    ("goal_regression", 0.01), ("goal_regression", 0.02), ("goal_regression", 0.03),
    ("goal_regression", 0.05),
    ("goal_nbr_loss", 0.025), ("goal_nbr_loss", 0.05),
    ("release_offgoal", 0.0),
    ("grasp_fail", 0.05), ("grasp_fail", 0.08), ("grasp_fail", 0.10),
    ("eef_approach_leave", 0.12), ("eef_approach_leave", 0.15),
    ("push_ungrasped", 0.01), ("push_ungrasped", 0.02),
]
HORIZONS = (4, 8)


def local_outcome(task: Task, episode: Episode, event: dict, horizon: int) -> dict | None:
    """Fixed-horizon second-order outcome: does the goal distance come back?"""
    target = next(t for t in task.targets if t["name"] == event["target"])
    gd = goal_distance(task, episode, target)
    k = event["onset"]
    if k + horizon >= episode.length:
        return None
    pre = float(np.minimum.accumulate(gd)[k - 1])
    level = pre + 0.5 * max(gd[k] - pre, 0.0)
    window = gd[k + 1:k + 1 + horizon]
    return {"Y_local": int(not bool((window <= level).any())),
            "recovered_m": float(gd[k] - window.min())}


def stage_gate(tasks: list[Task], out_dir: pathlib.Path) -> dict:
    rows = []
    for task in tasks:
        if not task.targets:
            continue
        for family, thr in LANDMARK_GRID:
            for episode in task.episodes:
                events = landmark_events(task, episode, family, thr)
                if not events:
                    continue
                first = events[0]
                rec = {"task": task.key, "task_short": SHORT.get(task.key, task.key),
                       "family": family, "threshold": thr,
                       "episode": episode.index, "init_state_id": episode.init_state_id,
                       "flow_noise_seed": episode.flow_noise_seed,
                       "terminal_success": int(episode.success),
                       "Y_terminal": int(not episode.success),
                       "episode_length": episode.length, "n_events": len(events),
                       **{k: v for k, v in first.items()}}
                for h in HORIZONS:
                    loc = local_outcome(task, episode, first, h)
                    rec[f"Y_local_h{h}"] = -1 if loc is None else loc["Y_local"]
                    rec[f"recovered_m_h{h}"] = np.nan if loc is None else loc["recovered_m"]
                rows.append(rec)
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "landmark_events.csv", index=False)

    inventory = []
    for (task_key, family, thr), grp in frame.groupby(["task_short", "family", "threshold"]):
        for label in ["Y_terminal"] + [f"Y_local_h{h}" for h in HORIZONS]:
            sub = grp[grp[label] >= 0]
            if not len(sub):
                continue
            y = sub[label].to_numpy()
            counts = sub.groupby("init_state_id")[label].agg(["size", "sum"])
            mixed = (counts["sum"] > 0) & (counts["sum"] < counts["size"])
            in_mixed = sub["init_state_id"].isin(counts.index[mixed]).to_numpy()
            setback = float((sub["regression_m"].to_numpy() > 0.005).mean())
            on0, on1 = sub.onset[y == 0], sub.onset[y == 1]
            inventory.append({
                "task": task_key, "family": family, "threshold": thr, "label": label,
                "n_events": int(len(sub)), "event_rate": round(len(sub) / 512.0, 4),
                "n_Y0": int((y == 0).sum()), "n_Y1": int((y == 1).sum()),
                "n_clusters": int(len(counts)), "n_mixed_clusters": int(mixed.sum()),
                "n_in_mixed": int(in_mixed.sum()),
                "n_Y0_in_mixed": int((y[in_mixed] == 0).sum()),
                "n_Y1_in_mixed": int((y[in_mixed] == 1).sum()),
                "setback_fraction": round(setback, 3),
                "onset_med_Y0": float(on0.median()) if len(on0) else np.nan,
                "onset_med_Y1": float(on1.median()) if len(on1) else np.nan,
                "lead_med_Y0": float((sub.episode_length - 1 - sub.onset)[y == 0].median()) if len(on0) else np.nan,
                "lead_med_Y1": float((sub.episode_length - 1 - sub.onset)[y == 1].median()) if len(on1) else np.nan,
                "sentinel_auc_onset": round(auc(sub.onset.to_numpy(float), y), 3),
            })
    inv = pd.DataFrame(inventory)
    inv["passes_gate"] = (
        (inv.n_events >= GATE["min_events"])
        & (inv[["n_Y0", "n_Y1"]].min(axis=1) >= GATE["min_per_class"])
        & (inv.n_mixed_clusters >= GATE["min_mixed_clusters"])
        & (inv.n_in_mixed >= GATE["min_events_in_mixed_clusters"])
        & (inv[["n_Y0_in_mixed", "n_Y1_in_mixed"]].min(axis=1) >= GATE["min_per_class_in_mixed_clusters"])
        & (inv.event_rate <= GATE["max_event_rate"])
        & (inv.setback_fraction >= GATE["min_setback_fraction"])
    )
    inv = inv.sort_values(["n_in_mixed", "n_events"], ascending=False)
    inv.to_csv(out_dir / "landmark_inventory.csv", index=False)

    passing = inv[inv.passes_gate]
    decision = {
        "gate_thresholds": GATE,
        "n_landmark_definitions": int(len(inv)),
        "n_passing": int(len(passing)),
        "gate": "PASS" if len(passing) else "FAIL",
        "passing": passing.to_dict("records"),
        "best_by_mixed_support": inv.head(12).to_dict("records"),
    }
    (out_dir / "gate_decision.json").write_text(json.dumps(decision, indent=2, default=float))
    return decision


# --------------------------------------------------------------------------- #
# route descriptors: state token vs action tokens
# --------------------------------------------------------------------------- #
@dataclass
class RouteDescriptors:
    """Per-query reduced HB routing, split by suffix token role."""
    episode_id: np.ndarray
    control_step: np.ndarray
    ids_state: np.ndarray        # (N, 8, 4)      denoise-invariant (verified)
    probs_state: np.ndarray      # (N, 8, 32)
    entropy_state: np.ndarray    # (N, 8)
    ids_action: np.ndarray       # (N, 8, 10, 10, 4)  layer x denoise x token
    probs_action: np.ndarray     # (N, 8, 10, 32)     token-averaged
    entropy_action: np.ndarray   # (N, 8, 10)         token-averaged
    state_denoise_max_dev: float


def load_routes(run: pathlib.Path) -> RouteDescriptors:
    store = zarr.open(store=str(run / "server/routes.zarr"), mode="r")
    total = int(store["episode_id"].shape[0])
    ids_state = np.empty((total, N_LAYERS, TOPK), np.uint8)
    probs_state = np.empty((total, N_LAYERS, N_EXPERTS), np.float32)
    entropy_state = np.empty((total, N_LAYERS), np.float32)
    ids_action = np.empty((total, N_LAYERS, N_DENOISE, N_TOKENS - 1, TOPK), np.uint8)
    probs_action = np.empty((total, N_LAYERS, N_DENOISE, N_EXPERTS), np.float32)
    entropy_action = np.empty((total, N_LAYERS, N_DENOISE), np.float32)
    dev = 0.0
    ids = store["hb_expert_ids"]
    probs = store["hb_router_probs"]
    ent = store["hb_entropy"]
    for a in range(0, total, BLOCK):
        b = min(a + BLOCK, total)
        bid = np.asarray(ids[a:b], np.uint8)
        bpr = np.asarray(probs[a:b], np.float32)
        ben = np.asarray(ent[a:b], np.float32)
        ids_state[a:b] = bid[:, :, 0, 0, :]
        probs_state[a:b] = bpr[:, :, 0, 0, :]
        entropy_state[a:b] = ben[:, :, 0, 0]
        dev = max(dev, float(np.abs(bpr[:, :, :, 0, :] - bpr[:, :, :1, 0, :]).max()))
        ids_action[a:b] = bid[:, :, :, 1:, :]
        probs_action[a:b] = bpr[:, :, :, 1:, :].mean(axis=3)
        entropy_action[a:b] = ben[:, :, :, 1:].mean(axis=3)
        del bid, bpr, ben
    return RouteDescriptors(
        episode_id=np.asarray(store["episode_id"][:], np.int32),
        control_step=np.asarray(store["control_step"][:], np.int32),
        ids_state=ids_state, probs_state=probs_state, entropy_state=entropy_state,
        ids_action=ids_action, probs_action=probs_action, entropy_action=entropy_action,
        state_denoise_max_dev=dev)


def query_features(rd: RouteDescriptors, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Scalar per-query descriptors for a contiguous set of rows of one episode."""
    n = len(rows)
    out: dict[str, np.ndarray] = {}
    ids_s = rd.ids_state[rows]
    pr_s = rd.probs_state[rows]
    ids_a = rd.ids_action[rows]
    pr_a = rd.probs_action[rows]
    out["state_entropy"] = rd.entropy_state[rows].mean(axis=1)
    out["action_entropy"] = rd.entropy_action[rows].mean(axis=(1, 2))
    out["state_top1_mass"] = pr_s.max(axis=-1).mean(axis=1)
    out["action_top1_mass"] = pr_a.max(axis=-1).mean(axis=(1, 2))
    speed_s = np.full(n, np.nan)
    speed_a = np.full(n, np.nan)
    soft_s = np.full(n, np.nan)
    soft_a = np.full(n, np.nan)
    if n > 1:
        speed_s[1:] = 1.0 - top4_overlap(ids_s[1:], ids_s[:-1]).mean(axis=1)
        speed_a[1:] = 1.0 - top4_overlap(ids_a[1:], ids_a[:-1]).mean(axis=(1, 2, 3))
        soft_s[1:] = 1.0 - hellinger(pr_s[1:], pr_s[:-1]).mean(axis=1)
        soft_a[1:] = 1.0 - hellinger(pr_a[1:], pr_a[:-1]).mean(axis=(1, 2))
    out["state_route_speed"] = speed_s
    out["action_route_speed"] = speed_a
    out["state_route_soft_speed"] = soft_s
    out["action_route_soft_speed"] = soft_a
    # within-query denoise drift of the action tokens (state token has none by construction)
    drift = 1.0 - top4_overlap(ids_a[:, :, 1:, :, :], ids_a[:, :, :-1, :, :]).mean(axis=(1, 2, 3))
    out["action_denoise_drift"] = drift
    out["action_denoise_span"] = 1.0 - top4_overlap(ids_a[:, :, -1, :, :],
                                                    ids_a[:, :, 0, :, :]).mean(axis=(1, 2))
    # state/action agreement inside the same query
    out["state_action_overlap"] = top4_overlap(
        ids_s[:, :, None, None, :], ids_a).mean(axis=(1, 2, 3))
    # recurrence: best soft similarity to queries at lag 2..8
    rec_s = np.full(n, np.nan)
    rec_a = np.full(n, np.nan)
    for k in range(n):
        lo, hi = max(0, k - 8), k - 1
        if hi < lo:
            continue
        js = np.arange(lo, hi + 1)
        rec_s[k] = float(hellinger(pr_s[k][None], pr_s[js]).mean(axis=1).max())
        rec_a[k] = float(hellinger(pr_a[k][None], pr_a[js]).mean(axis=(1, 2)).max())
    out["state_route_recurrence"] = rec_s
    out["action_route_recurrence"] = rec_a
    return out


# --------------------------------------------------------------------------- #
# Stage B: nested model ladder with sentinel
# --------------------------------------------------------------------------- #
def physical_block(task: Task, episode: Episode, event: dict, cap: int) -> np.ndarray:
    """Physical control block: task phase, event location, current state, severity, recoverability.

    ``onset`` and ``onset / cap`` are included on purpose: they are the
    task-phase and remaining-budget controls named in PREREG S5.  The
    one-dimensional sentinel below is exactly this pair, so any increment is
    reported net of timing.
    """
    target = next(t for t in task.targets if t["name"] == event["target"])
    gd = goal_distance(task, episode, target)
    k = event["onset"]
    pos = episode.sim[:, target["lo"]:target["hi"]]
    lo = max(0, k - 3)
    return np.asarray([
        float(k), float(k) / float(cap),                              # task phase
        gd[k], float(np.minimum.accumulate(gd)[k - 1]),                # progress state
        event["regression_m"], event["severity_m"],                    # event severity
        event["height_m"], event["eef_obj_m"], event["goal_dist"],     # recoverability
        float(np.linalg.norm(episode.eef[k] - episode.eef[lo])),
        float(np.linalg.norm(pos[k] - pos[lo])),
        float(episode.gripper[k]), float(episode.gripper_cmd[k]),
        float(episode.eef[k, 0]), float(episode.eef[k, 1]), float(episode.eef[k, 2]),
        float(pos[k, 0]), float(pos[k, 1]), float(pos[k, 2]),
        float(gd[max(0, k - 2)] - gd[k]),
    ], float)


def fit_ladder(blocks: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
               rng: np.random.Generator, draws: int, pca_dims: int = 8) -> dict:
    """Leave-one-init-out ladder.  Every transform is fit inside the training fold."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    results = {}
    oof = {}
    for name, x in blocks.items():
        pred = np.full(len(y), np.nan)
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            if len(np.unique(y[tr])) < 2:
                continue
            steps = [("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler())]
            if x.shape[1] > 4 * max(int(tr.sum()), 1) ** 0.5:
                steps.append(("pca", PCA(n_components=min(pca_dims, int(tr.sum()) - 1,
                                                          x.shape[1]), random_state=0)))
            steps.append(("lr", LogisticRegression(max_iter=5000, C=0.5)))
            pipe = Pipeline(steps)
            pipe.fit(x[tr], y[tr])
            pred[te] = pipe.predict_proba(x[te])[:, 1]
        ok = ~np.isnan(pred)
        a = auc(pred[ok], y[ok])
        lo, hi = cluster_bootstrap_ci(pred[ok], y[ok], groups[ok], rng, draws)
        results[name] = {"auc": a, "ci95": [lo, hi], "dims": int(x.shape[1]),
                         "n_scored": int(ok.sum())}
        oof[name] = pred
    return {"blocks": results, "oof": oof}


def select_candidates(inv) -> list[dict]:
    """Two prespecified fallbacks when the gate fails (PREREG S7).

    ``setback`` = the largest dual-outcome risk set whose landmark really loses
    ground toward the goal.  ``support`` = the largest dual-outcome risk set of
    any kind, which in this cache is a landmark that is not an error at all.
    """
    inv = inv.copy()
    inv["balance"] = inv[["n_Y0_in_mixed", "n_Y1_in_mixed"]].min(axis=1)
    out = []
    real = inv[(inv.setback_fraction >= GATE["min_setback_fraction"])
               & (inv.event_rate <= GATE["max_event_rate"])]
    if len(real):
        row = real.sort_values(["balance", "n_in_mixed"], ascending=False).iloc[0].to_dict()
        row["role"] = "setback_landmark"
        out.append(row)
    row = inv.sort_values(["balance", "n_in_mixed"], ascending=False).iloc[0].to_dict()
    row["role"] = "max_support_landmark"
    if not out or (row["family"], row["threshold"], row["label"], row["task"]) != \
            (out[0]["family"], out[0]["threshold"], out[0]["label"], out[0]["task"]):
        out.append(row)
    return out


def hidden_block(run: pathlib.Path, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Router-input contextual hidden state at the landmark query (state / action tokens)."""
    store = zarr.open(store=str(run / "server/hidden.zarr"), mode="r")
    arr = store["hb_hidden"]
    state, action = [], []
    for r in rows:
        block = np.asarray(arr[int(r), :, 0, :, :], np.float32)   # (8, 11, 1024)
        state.append(block[:, 0, :].ravel())
        action.append(block[:, 1:, :].mean(axis=1).ravel())
    return np.asarray(state), np.asarray(action)


def run_one_ladder(task: Task, chosen: dict, rd: RouteDescriptors, args) -> dict:
    import pandas as pd
    family, thr, label = chosen["family"], float(chosen["threshold"]), chosen["label"]
    cap = int(max(e.length for e in task.episodes))
    rows, phys, sentinel = [], [], []
    for episode in task.episodes:
        events = landmark_events(task, episode, family, thr)
        if not events:
            continue
        ev = events[0]
        if label == "Y_terminal":
            y = int(not episode.success)
        else:
            h = int(label.split("h")[-1])
            loc = local_outcome(task, episode, ev, h)
            if loc is None:
                continue
            y = loc["Y_local"]
        rows.append({"episode": episode.index, "init_state_id": episode.init_state_id,
                     "onset": ev["onset"], "y": y, "target": ev["target"],
                     "terminal_success": int(episode.success)})
        phys.append(physical_block(task, episode, ev, cap))
        sentinel.append([float(ev["onset"]), float(ev["onset"]) / cap])
    meta = pd.DataFrame(rows)
    counts = meta.groupby("init_state_id").y.agg(["size", "sum"])
    mixed = counts.index[(counts["sum"] > 0) & (counts["sum"] < counts["size"])]
    keep = meta.init_state_id.isin(mixed).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    phys = np.asarray(phys)[keep]
    sentinel = np.asarray(sentinel)[keep]
    y = meta.y.to_numpy()
    groups = meta.init_state_id.to_numpy()
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
        return {"status": "NOT_EVALUABLE", "landmark": chosen,
                "n_used": int(len(y)), "n_Y1": int(y.sum()),
                "n_clusters": int(len(np.unique(groups)))}

    offsets = np.r_[0, np.cumsum([e.length for e in task.episodes])[:-1]]
    route_state, route_action = [], []
    zrows = []
    for _, r in meta.iterrows():
        ep = task.episodes[int(r.episode)]
        k = int(r.onset)
        base = int(offsets[int(r.episode)])
        zrows.append(base + k)
        feats = query_features(rd, base + np.arange(ep.length))
        window = [max(0, k - 2), max(0, k - 1), k]
        route_state.append(np.concatenate([
            [np.nanmean([feats[key][i] for i in window])
             for key in ("state_route_speed", "state_route_soft_speed", "state_entropy",
                         "state_top1_mass", "state_route_recurrence")],
            rd.probs_state[base + k].mean(axis=0)]))
        route_action.append(np.concatenate([
            [np.nanmean([feats[key][i] for i in window])
             for key in ("action_route_speed", "action_route_soft_speed", "action_entropy",
                         "action_top1_mass", "action_route_recurrence",
                         "action_denoise_drift", "action_denoise_span",
                         "state_action_overlap")],
            rd.probs_action[base + k].mean(axis=(0, 1))]))
    route_state = np.asarray(route_state)
    route_action = np.asarray(route_action)
    zrows = np.asarray(zrows)
    chunk = np.asarray([task.episodes[int(r.episode)].sim[int(r.onset), 1:10]
                        for _, r in meta.iterrows()])
    hid_s, hid_a = hidden_block(task.run, zrows)

    blocks = {
        "sentinel_onset_only": sentinel,
        "M_phys": phys,
        "M_phys+route_state": np.hstack([phys, route_state]),
        "M_phys+route_action": np.hstack([phys, route_action]),
        "M_phys+route": np.hstack([phys, route_state, route_action]),
        "M_phys+action": np.hstack([phys, chunk]),
        "M_joint": np.hstack([phys, route_state, route_action, chunk]),
        "M_joint+hidden": np.hstack([phys, route_state, route_action, chunk, hid_s, hid_a]),
    }
    rng = np.random.default_rng(args.seed)
    fit = fit_ladder(blocks, y, groups, rng, args.bootstraps)
    deltas = {}
    for name, base_name in (("M_phys+route_state", "M_phys+action"),
                            ("M_phys+route_action", "M_phys+action"),
                            ("M_phys+route", "M_phys+action"),
                            ("M_joint", "M_phys+action"),
                            ("M_joint+hidden", "M_joint"),
                            ("M_phys", "sentinel_onset_only")):
        pred, base = fit["oof"][name], fit["oof"][base_name]
        m = ~np.isnan(pred) & ~np.isnan(base)
        uniq = np.unique(groups[m])
        draws = []
        for _ in range(args.bootstraps):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero((groups == c) & m) for c in pick])
            d = auc(pred[idx], y[idx]) - auc(base[idx], y[idx])
            if not math.isnan(d):
                draws.append(d)
        deltas[f"{name}_minus_{base_name}"] = {
            "delta": auc(pred[m], y[m]) - auc(base[m], y[m]),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
            if draws else [float("nan"), float("nan")]}
    return {"status": "DEMONSTRATION_ONLY_GATE_FAILED", "role": chosen.get("role"),
            "landmark": {k: chosen[k] for k in ("task", "family", "threshold", "label",
                                                "n_events", "event_rate", "n_Y0", "n_Y1",
                                                "n_mixed_clusters", "setback_fraction")},
            "n_used": int(len(y)), "n_Y1": int(y.sum()),
            "n_clusters": int(len(np.unique(groups))),
            "blocks": fit["blocks"], "increments": deltas,
            "cohort": meta.to_dict("records")}


def stage_ladder(tasks: list[Task], decision: dict, args, out_dir: pathlib.Path) -> dict:
    import pandas as pd
    inv = pd.read_csv(out_dir / "landmark_inventory.csv")
    passing = inv[inv.passes_gate]
    candidates = (passing.assign(role="gate_passed").to_dict("records")
                  if len(passing) else select_candidates(inv))
    results = []
    by_task: dict[str, RouteDescriptors] = {}
    for chosen in candidates:
        task = next(t for t in tasks if SHORT.get(t.key, t.key) == chosen["task"])
        if task.key not in by_task:
            by_task.clear()
            by_task[task.key] = load_routes(task.run)
        res = run_one_ladder(task, chosen, by_task[task.key], args)
        if len(passing):
            res["status"] = "GATE_PASSED"
        results.append(res)
    by_task.clear()
    out = {"gate": "PASS" if len(passing) else "FAIL", "runs": results}
    (out_dir / "ladder.json").write_text(json.dumps(out, indent=2, default=float))
    frames = []
    for res in results:
        if "cohort" in res:
            f = pd.DataFrame(res.pop("cohort"))
            f["role"] = res.get("role")
            frames.append(f)
    if frames:
        pd.concat(frames).to_csv(out_dir / "ladder_cohort.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# Stage C: state-token vs action-token routing decomposition (descriptive)
# --------------------------------------------------------------------------- #
HB_LAYER_IDS = (2, 3, 4, 5, 12, 13, 14, 15)


def near_tie_diagnostics(run: pathlib.Path, n_rows: int = 2048) -> dict:
    """How much of a top-4 change is a real routing change vs a boundary tie?

    Cross-checks ``analysis/near-tie/summary.json`` (boundary_tie_rate 0.391,
    gap_exactly_zero_rate 0.238, gap_median 4.88e-4 in bf16).
    """
    store = zarr.open(store=str(run / "server/routes.zarr"), mode="r")
    pr = np.asarray(store["hb_router_probs"][:n_rows], np.float32)
    ids = np.asarray(store["hb_expert_ids"][:n_rows], np.uint8)
    eid = np.asarray(store["episode_id"][:n_rows], np.int32)
    srt = np.sort(pr, axis=-1)[..., ::-1]
    same = eid[1:] == eid[:-1]
    out = {
        "state_p1": float(srt[:, :, 0, 0, 0].mean()),
        "action_p1": float(srt[:, :, :, 1:, 0].mean()),
        "state_top4_boundary_gap": float((srt[:, :, 0, 0, 3] - srt[:, :, 0, 0, 4]).mean()),
        "action_top4_boundary_gap": float((srt[:, :, :, 1:, 3] - srt[:, :, :, 1:, 4]).mean()),
        "state_p1_by_layer": {str(L): float(srt[:, i, 0, 0, 0].mean())
                              for i, L in enumerate(HB_LAYER_IDS)},
        "action_p1_by_layer": {str(L): float(srt[:, i, :, 1:, 0].mean())
                               for i, L in enumerate(HB_LAYER_IDS)},
    }
    if same.any():
        out["state_adjacent_query_top4_turnover"] = float(
            (1.0 - top4_overlap(ids[1:, :, 0, 0, :], ids[:-1, :, 0, 0, :]).mean(axis=1))[same].mean())
        out["action_adjacent_query_top4_turnover"] = float(
            (1.0 - top4_overlap(ids[1:, :, :, 1:, :], ids[:-1, :, :, 1:, :]).mean(axis=(1, 2, 3)))[same].mean())
        out["state_adjacent_query_1_minus_hellinger"] = float(
            (1.0 - hellinger(pr[1:, :, 0, 0, :], pr[:-1, :, 0, 0, :]).mean(axis=1))[same].mean())
        out["action_adjacent_query_1_minus_hellinger"] = float(
            (1.0 - hellinger(pr[1:, :, :, 1:, :].mean(3), pr[:-1, :, :, 1:, :].mean(3)).mean(axis=(1, 2)))[same].mean())
    out["action_adjacent_denoise_top4_turnover"] = float(
        (1.0 - top4_overlap(ids[:, :, 1:, 1:, :], ids[:, :, :-1, 1:, :]).mean(axis=(1, 2, 3))).mean())
    out["action_adjacent_denoise_1_minus_hellinger"] = float(
        (1.0 - hellinger(pr[:, :, 1:, 1:, :].mean(3), pr[:, :, :-1, 1:, :].mean(3)).mean(axis=(1, 2))).mean())
    return out


EVENT_CLASSES = ("active_return", "stagnation_core", "success")
REL_WINDOW = np.arange(-10, 7)


def stage_tokens(tasks: list[Task], args, out_dir: pathlib.Path) -> dict:
    import pandas as pd
    task = next(t for t in tasks if t.key == args.route_task)
    label_path = HERE / "analysis/failure-moe-signatures/episode_results.csv"
    labels = pd.read_csv(label_path)
    labels = labels[labels.task == task.key]
    rd = load_routes(task.run)
    offsets = np.r_[0, np.cumsum([e.length for e in task.episodes])[:-1]]

    # -- C1 structural: does the state token move along the denoise axis at all?
    store = zarr.open(store=str(task.run / "server/routes.zarr"), mode="r")
    probe = np.asarray(store["hb_router_probs"][:BLOCK], np.float32)
    ids_probe = np.asarray(store["hb_expert_ids"][:BLOCK], np.uint8)
    structural = {
        "state_prob_max_abs_dev_across_denoise": rd.state_denoise_max_dev,
        "state_top4_identical_across_denoise": float(
            (ids_probe[:, :, :, 0, :] == ids_probe[:, :, :1, 0, :]).all(axis=(2, 3)).mean()),
        "action_top4_overlap_vs_denoise0": [
            float(top4_overlap(ids_probe[:, :, d, 1:, :], ids_probe[:, :, 0, 1:, :]).mean())
            for d in range(N_DENOISE)],
        "action_prob_l1_vs_denoise0": [
            float(np.abs(probe[:, :, d, 1:, :] - probe[:, :, 0, 1:, :]).sum(-1).mean())
            for d in range(N_DENOISE)],
        "action_entropy_by_denoise": [
            float(np.asarray(store["hb_entropy"][:BLOCK], np.float32)[:, :, d, 1:].mean())
            for d in range(N_DENOISE)],
        "state_action_top4_overlap": float(
            top4_overlap(ids_probe[:, :, :, :1, :], ids_probe[:, :, :, 1:, :]).mean()),
        "max_entropy_ln32": float(np.log(N_EXPERTS)),
        "state_entropy_mean": float(rd.entropy_state.mean()),
        "action_entropy_mean": float(rd.entropy_action.mean()),
        "state_entropy_normalised": float(rd.entropy_state.mean() / np.log(N_EXPERTS)),
        "action_entropy_normalised": float(rd.entropy_action.mean() / np.log(N_EXPERTS)),
        "state_top1_mass_mean": float(rd.probs_state.max(-1).mean()),
        "action_top1_mass_mean": float(rd.probs_action.max(-1).mean()),
        "uniform_prob": 1.0 / N_EXPERTS,
        **near_tie_diagnostics(task.run),
    }
    del probe, ids_probe

    # -- C2 event-aligned dynamics of the two token roles
    keys = ("state_route_speed", "action_route_speed", "state_route_soft_speed",
            "action_route_soft_speed", "state_entropy", "action_entropy",
            "state_top1_mass", "action_top1_mass", "state_route_recurrence",
            "action_route_recurrence", "action_denoise_drift", "action_denoise_span",
            "state_action_overlap")
    curves = {c: {k: [] for k in keys} for c in EVENT_CLASSES}
    inits = {c: [] for c in EVENT_CLASSES}
    for _, row in labels.iterrows():
        cls = row.physical_type
        if cls not in EVENT_CLASSES:
            continue
        onset = int(row.physical_onset_query)
        ep = task.episodes[int(row.episode)]
        rows_ep = offsets[int(row.episode)] + np.arange(ep.length)
        feats = query_features(rd, rows_ep)
        idx = onset + REL_WINDOW
        valid = (idx >= 0) & (idx < ep.length)
        if not valid.all():
            continue
        ref = np.isin(REL_WINDOW, np.arange(-10, -6))
        for k in keys:
            v = feats[k][idx].astype(float)
            base = np.nanmean(v[ref])
            curves[cls][k].append(v - base)
        inits[cls].append(int(row.init_state_id))

    pooled_sd = {}
    for k in keys:
        vals = np.concatenate([np.asarray(curves[c][k]).ravel() for c in EVENT_CLASSES
                               if len(curves[c][k])])
        pooled_sd[k] = float(np.nanstd(vals)) or 1.0

    windows = {"lead_-6_-2": (-6, -2), "sync_-1_+1": (-1, 1), "after_+2_+6": (2, 6)}
    rng = np.random.default_rng(args.seed)
    effects = []
    for k in keys:
        for cls in ("active_return", "stagnation_core"):
            a = np.asarray(curves[cls][k], float)
            b = np.asarray(curves["success"][k], float)
            if not len(a) or not len(b):
                continue
            ga, gb = np.asarray(inits[cls]), np.asarray(inits["success"])
            shared = np.intersect1d(np.unique(ga), np.unique(gb))
            for wname, (lo, hi) in windows.items():
                sel = (REL_WINDOW >= lo) & (REL_WINDOW <= hi)
                per = []
                for g in shared:
                    va = np.nanmean(a[ga == g][:, sel])
                    vb = np.nanmean(b[gb == g][:, sel])
                    if np.isfinite(va) and np.isfinite(vb):
                        per.append(va - vb)
                if not per:
                    continue
                per = np.asarray(per)
                boots = [np.nanmean(rng.choice(per, len(per), replace=True))
                         for _ in range(min(args.bootstraps, 2000))]
                effects.append({
                    "signal": k, "class": cls, "window": wname,
                    "n_matched_inits": int(len(per)),
                    "effect_sd": float(per.mean() / pooled_sd[k]),
                    "ci_lo": float(np.percentile(boots, 2.5) / pooled_sd[k]),
                    "ci_hi": float(np.percentile(boots, 97.5) / pooled_sd[k]),
                })
    eff = pd.DataFrame(effects)
    eff["separated"] = (eff.effect_sd.abs() >= 0.2) & (np.sign(eff.ci_lo) == np.sign(eff.ci_hi))
    eff.to_csv(out_dir / "token_event_effects.csv", index=False)

    # -- C2b phase-free: state and action are measured on the SAME queries of the
    #    SAME episodes, so comparing their deviation onsets carries no phase confound.
    paired = []
    pairs = [("state_route_speed", "action_route_speed"),
             ("state_route_soft_speed", "action_route_soft_speed"),
             ("state_entropy", "action_entropy"),
             ("state_top1_mass", "action_top1_mass"),
             ("state_route_recurrence", "action_route_recurrence")]
    for cls in EVENT_CLASSES:
        g = np.asarray(inits[cls])
        if not len(g):
            continue
        for sk, ak in pairs:
            a = np.asarray(curves[cls][sk], float) / pooled_sd[sk]
            b = np.asarray(curves[cls][ak], float) / pooled_sd[ak]
            for j, rel in enumerate(REL_WINDOW):
                for role, arr in (("state", a), ("action", b), ("action_minus_state", b - a)):
                    per = np.asarray([np.nanmean(arr[g == u, j]) for u in np.unique(g)])
                    per = per[np.isfinite(per)]
                    if len(per) < 3:
                        continue
                    boots = [np.nanmean(rng.choice(per, len(per), replace=True))
                             for _ in range(min(args.bootstraps, 2000))]
                    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
                    paired.append({"class": cls, "signal": sk.replace("state_", ""),
                                   "role": role, "rel_query": int(rel),
                                   "n_inits": int(len(per)), "effect_sd": float(per.mean()),
                                   "ci_lo": lo, "ci_hi": hi,
                                   "deviates": bool(abs(per.mean()) >= 0.2 and lo * hi > 0)})
    pr = pd.DataFrame(paired)
    pr.to_csv(out_dir / "token_paired_onsets.csv", index=False)
    first_dev = []
    for (cls, sig, role), grp in pr.groupby(["class", "signal", "role"]):
        d = grp[grp.deviates & (grp.rel_query >= -9)].sort_values("rel_query")
        first_dev.append({"class": cls, "signal": sig, "role": role,
                          "first_deviating_rel_query": int(d.rel_query.iloc[0]) if len(d) else None,
                          "n_deviating": int(len(d))})
    pd.DataFrame(first_dev).to_csv(out_dir / "token_first_deviation.csv", index=False)

    # -- C3 denoise-axis discriminability of the action tokens
    denoise_auc = []
    cls_rows, cls_y, cls_g = [], [], []
    for _, row in labels.iterrows():
        if row.physical_type not in ("active_return", "stagnation_core"):
            continue
        onset = int(row.physical_onset_query)
        cls_rows.append(offsets[int(row.episode)] + onset)
        cls_y.append(int(row.physical_type == "active_return"))
        cls_g.append(int(row.init_state_id))
    cls_rows = np.asarray(cls_rows)
    cls_y = np.asarray(cls_y)
    cls_g = np.asarray(cls_g)
    cls_onset = np.asarray([int(r.physical_onset_query) for _, r in labels.iterrows()
                            if r.physical_type in ("active_return", "stagnation_core")], float)
    if len(cls_rows):
        denoise_auc.append({"token": "SENTINEL_onset_query", "denoise": -2,
                            "auc_entropy": auc(cls_onset, cls_y),
                            "auc_top1": auc(cls_onset, cls_y)})
        state_scores = rd.entropy_state[cls_rows].mean(axis=1)
        denoise_auc.append({"token": "state", "denoise": -1,
                            "auc_entropy": auc(state_scores, cls_y),
                            "auc_top1": auc(rd.probs_state[cls_rows].max(-1).mean(1), cls_y)})
        for d in range(N_DENOISE):
            denoise_auc.append({
                "token": "action", "denoise": d,
                "auc_entropy": auc(rd.entropy_action[cls_rows, :, d].mean(axis=1), cls_y),
                "auc_top1": auc(rd.probs_action[cls_rows, :, d, :].max(-1).mean(1), cls_y)})
    dn = pd.DataFrame(denoise_auc)
    dn.to_csv(out_dir / "denoise_axis_auc.csv", index=False)

    mean_curves = {c: {k: np.nanmean(np.asarray(curves[c][k], float), axis=0).tolist()
                       for k in keys if len(curves[c][k])} for c in EVENT_CLASSES}
    summary = {
        "structural": structural,
        "n_curves": {c: len(inits[c]) for c in EVENT_CLASSES},
        "rel_window": REL_WINDOW.tolist(),
        "mean_curves": mean_curves,
        "pooled_sd": pooled_sd,
    }
    (out_dir / "token_axis_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    make_figure(summary, eff, dn, out_dir)
    del rd
    return {"structural": structural, "effects": eff, "denoise_auc": dn,
            "n_curves": summary["n_curves"]}


def make_figure(summary: dict, eff, dn, out_dir: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = np.asarray(summary["rel_window"])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0))
    colors = {"active_return": "#C0392B", "stagnation_core": "#2C6FBB", "success": "#7F8C8D"}
    panels = [("state_route_speed", "action_route_speed", "hard top-4 route speed"),
              ("state_route_soft_speed", "action_route_soft_speed", "soft (Hellinger) route speed"),
              ("state_entropy", "action_entropy", "router entropy"),
              ("state_route_recurrence", "action_route_recurrence", "route recurrence (lag 2-8)")]
    for ax, (sk, ak, title) in zip(axes.ravel()[:4], panels):
        for cls, c in colors.items():
            if sk in summary["mean_curves"].get(cls, {}):
                ax.plot(rel, summary["mean_curves"][cls][sk], color=c, ls="--", lw=1.6,
                        label=f"{cls} state")
            if ak in summary["mean_curves"].get(cls, {}):
                ax.plot(rel, summary["mean_curves"][cls][ak], color=c, ls="-", lw=1.9,
                        label=f"{cls} action")
        ax.axvline(0, color="k", lw=0.8, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("query relative to physical onset")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=6, ncol=2)

    ax = axes.ravel()[4]
    st = summary["structural"]
    ax.plot(range(N_DENOISE), st["action_top4_overlap_vs_denoise0"], "o-", color="#C0392B",
            label="action token")
    ax.axhline(1.0, color="#2C6FBB", ls="--", label="state token (exactly constant)")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("flow denoise iteration")
    ax.set_ylabel("top-4 overlap with iteration 0")
    ax.set_title("denoise axis: state token is degenerate", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.text(0.03, 0.06,
            "action top-4 turns over 70%% while its prob distribution moves\n"
            "only 1-H = %.1e (top-4 boundary gap %.1e, near-tie regime)"
            % (st.get("action_adjacent_denoise_1_minus_hellinger", float("nan")),
               st.get("action_top4_boundary_gap", float("nan"))),
            transform=ax.transAxes, fontsize=6.5, va="bottom")

    ax = axes.ravel()[5]
    if len(dn):
        a = dn[dn.token == "action"]
        ax.plot(a.denoise, a.auc_entropy, "o-", color="#C0392B", label="action entropy")
        ax.plot(a.denoise, a.auc_top1, "s-", color="#E28E2C", label="action top-1 mass")
        s = dn[dn.token == "state"]
        if len(s):
            ax.axhline(float(s.auc_entropy.iloc[0]), color="#2C6FBB", ls="--",
                       label="state entropy (denoise-invariant)")
            ax.axhline(float(s.auc_top1.iloc[0]), color="#2C6FBB", ls=":",
                       label="state top-1 mass")
        sen = dn[dn.token == "SENTINEL_onset_query"]
        if len(sen):
            v = float(sen.auc_entropy.iloc[0])
            ax.axhline(v, color="#111111", ls="-.", lw=2.0,
                       label=f"SENTINEL onset query alone ({v:.3f})")
            ax.axhline(1.0 - v, color="#111111", ls="-.", lw=2.0, alpha=0.5)
    ax.axhline(0.5, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("flow denoise iteration")
    ax.set_ylabel("AUC (EEF-return vs stasis, at onset)")
    ax.set_title("denoise-axis discriminability, descriptive", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.suptitle("state-token vs action-token HB routing, long/SCENE8 (descriptive)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "token_decomposition.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #
def self_test() -> None:
    rng = np.random.default_rng(0)
    s = np.r_[np.zeros(50), np.ones(50)]
    lab = np.r_[np.zeros(50), np.ones(50)].astype(int)
    assert abs(auc(s, lab) - 1.0) < 1e-9
    assert abs(auc(-s, lab) - 0.0) < 1e-9
    assert abs(auc(rng.normal(size=200), rng.integers(0, 2, 200)) - 0.5) < 0.15
    assert abs(auc(np.zeros(100), lab) - 0.5) < 1e-9

    assert runs_of_true([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    assert runs_of_true([1, 1]) == [(0, 2)]
    assert runs_of_true([0, 0]) == []

    a = np.array([[0, 1, 2, 3]], np.uint8)
    b = np.array([[2, 3, 4, 5]], np.uint8)
    assert abs(float(top4_overlap(a, b).mean()) - 0.5) < 1e-9
    assert abs(float(top4_overlap(a, a).mean()) - 1.0) < 1e-9

    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert abs(float(hellinger(p, p).mean()) - 1.0) < 1e-9
    assert abs(float(hellinger(np.array([1.0, 0.0]), np.array([0.0, 1.0])).mean())) < 1e-9

    # landmark detector on a synthetic rollout: object rises then is dropped off-goal
    T = 12
    sim = np.zeros((T, 47))
    sim[:, 10:13] = np.array([0.0, 0.0, 0.0])
    sim[4:8, 12] = np.array([0.03, 0.06, 0.06, 0.06])
    sim[8:, 12] = 0.0
    ep = Episode(index=0, init_state_id=0, flow_noise_seed=0, success=False, length=T,
                 eef=np.zeros((T, 3)), gripper=np.zeros(T),
                 gripper_cmd=np.r_[np.zeros(4), np.ones(4), -np.ones(4)], sim=sim)
    task = Task(key="synthetic", run=pathlib.Path("."), episodes=[ep])
    task.targets = [{"name": "obj", "lo": 10, "hi": 13,
                     "terminal": np.array([[1.0, 0.0, 0.0]]), "success_index": np.array([0])}]
    ev = landmark_events(task, ep, "liftloss", 0.0)
    assert len(ev) == 1 and ev[0]["onset"] == 8, ev
    ev = landmark_events(task, ep, "heightloss", 0.03)
    assert [e["onset"] for e in ev] == [8], ev
    ev = landmark_events(task, ep, "goal_regression", 0.01)
    assert ev == [], ev

    x = np.linspace(0, 1, 60)[:, None]
    y = (x[:, 0] > 0.5).astype(int)
    g = np.repeat(np.arange(6), 10)
    fit = fit_ladder({"x": x}, y, g, np.random.default_rng(1), 50)
    assert fit["blocks"]["x"]["auc"] > 0.9, fit

    lo, hi = cluster_bootstrap_ci(np.r_[np.zeros(30), np.ones(30)],
                                  np.r_[np.zeros(30), np.ones(30)].astype(int),
                                  np.repeat(np.arange(6), 10), np.random.default_rng(2), 100)
    assert lo > 0.8 and hi <= 1.0, (lo, hi)
    print("self-test OK")


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [load_task(run, args.cache_root) for run in discover_runs(args.cache_root)]
    summary: dict = {"seed": args.seed, "stage": args.stage,
                     "tasks": [t.key for t in tasks]}
    if args.stage in ("all", "gate"):
        summary["gate"] = stage_gate(tasks, args.out_dir)
        print("gate:", summary["gate"]["gate"],
              f"({summary['gate']['n_passing']}/{summary['gate']['n_landmark_definitions']} "
              "landmark definitions pass)")
    if args.stage in ("all", "ladder"):
        summary["ladder"] = stage_ladder(tasks, summary.get("gate", {}), args, args.out_dir)
        for res in summary["ladder"]["runs"]:
            print("ladder:", res["status"], res.get("role"), res["landmark"],
                  f"n={res['n_used']}")
            for k, v in res.get("blocks", {}).items():
                print(f"  {k:24s} AUC={v['auc']:.3f} "
                      f"CI=[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] dims={v['dims']}")
    if args.stage in ("all", "tokens"):
        res = stage_tokens(tasks, args, args.out_dir)
        summary["tokens"] = {"structural": res["structural"], "n_curves": res["n_curves"]}
        print("tokens: state denoise max |dp| =",
              res["structural"]["state_prob_max_abs_dev_across_denoise"])
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
