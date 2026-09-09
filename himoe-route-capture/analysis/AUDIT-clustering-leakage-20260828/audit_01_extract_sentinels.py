#!/usr/bin/env python3
"""Extract NON-ROUTING sentinel tapes (S2 physical EEF, S3 action chunk).

Uses the SAME relative-phase anchoring as the audited routing pipeline:
PhaseConfig(start=0.5, end=1.0, bins=10)  [analyze_failure_routing_clusters.py:48-50]
so that the only thing that changes vs the routing pipeline is the information
source, not the time normalization.

Output: sentinel_tapes.npz  (physical (2560,10,D1), action (2560,10,D2), meta)
"""
from __future__ import annotations

import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CACHE = pathlib.Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA")
OUT = pathlib.Path(
    "/home/jovyan/work/himoe-vla/himoe-route-capture/analysis/AUDIT-clustering-leakage-20260828"
)
SRC = pathlib.Path(
    "/home/jovyan/work/himoe-vla/himoe-route-capture/analysis/all-outcome-routing-clusters"
)

PHASE_START, PHASE_END, PHASE_BINS = 0.5, 1.0, 10


def phase_interpolate(values: np.ndarray) -> np.ndarray:
    """Verbatim copy of analyze_failure_routing_clusters.phase_interpolate."""
    values = np.asarray(values)
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(PHASE_START, PHASE_END, PHASE_BINS, dtype=np.float64)
    upper = np.clip(np.searchsorted(source, target, side="left"), 1, len(source) - 1)
    lower = upper - 1
    width = source[upper] - source[lower]
    alpha = ((target - source[lower]) / width).astype(np.float32)
    shape = (len(target),) + (1,) * (values.ndim - 1)
    return values[lower] * (1.0 - alpha.reshape(shape)) + values[upper] * alpha.reshape(shape)


def _win(x: np.ndarray, half: int) -> np.ndarray:
    """Padded sliding window index array."""
    n = len(x)
    idx = np.clip(np.arange(n)[:, None] + np.arange(-half, half + 1)[None, :], 0, n - 1)
    return idx


def physical_tape(state: np.ndarray) -> np.ndarray:
    """state (T,8) = eef xyz(3) + eef axis-angle(3) + 2 gripper joints -> (T,36)."""
    s = np.asarray(state, dtype=np.float64)
    T = len(s)
    pos, rot, grip = s[:, :3], s[:, 3:6], s[:, 6:8]
    v = np.vstack([np.zeros((1, 3)), np.diff(pos, axis=0)])
    a = np.vstack([np.zeros((1, 3)), np.diff(v, axis=0)])
    j = np.vstack([np.zeros((1, 3)), np.diff(a, axis=0)])
    w = np.vstack([np.zeros((1, 3)), np.diff(rot, axis=0)])
    speed = np.linalg.norm(v, axis=1)
    open_amt = grip[:, 0] - grip[:, 1]
    gv = np.concatenate([[0.0], np.diff(open_amt)])
    idx5 = _win(pos, 2)
    win_pos = pos[idx5]                                     # (T,5,3)
    seg = np.linalg.norm(np.diff(win_pos, axis=1), axis=2)  # (T,4)
    path5 = seg.sum(axis=1)
    net5 = np.linalg.norm(win_pos[:, -1] - win_pos[:, 0], axis=1)
    straight = net5 / np.maximum(path5, 1e-9)
    sp5 = speed[idx5]
    prev_v = np.vstack([np.zeros((1, 3)), v[:-1]])
    denom = np.maximum(np.linalg.norm(v, axis=1) * np.linalg.norm(prev_v, axis=1), 1e-9)
    bend = np.sum(v * prev_v, axis=1) / denom
    disp0 = np.linalg.norm(pos - pos[0], axis=1)
    idx3 = _win(pos, 1)
    net3 = np.linalg.norm(pos[idx3][:, -1] - pos[idx3][:, 0], axis=1)
    med = np.cumsum(pos, axis=0) / np.arange(1, T + 1)[:, None]
    cols = [
        pos, rot, grip,
        open_amt[:, None], v, speed[:, None], a,
        np.linalg.norm(a, axis=1)[:, None], w,
        np.linalg.norm(w, axis=1)[:, None],
        np.linalg.norm(j, axis=1)[:, None],
        gv[:, None], disp0[:, None], net3[:, None], net5[:, None],
        path5[:, None], straight[:, None],
        sp5.mean(axis=1)[:, None], sp5.std(axis=1)[:, None],
        bend[:, None], (pos - med),
    ]
    out = np.concatenate([c if c.ndim == 2 else c[:, None] for c in cols], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def action_tape(actions: np.ndarray) -> np.ndarray:
    """actions (T,10,7) -> (T,44) chunk statistics only (no routing, no sim)."""
    A = np.asarray(actions, dtype=np.float64)
    T = len(A)
    mean = A.mean(axis=1)
    std = A.std(axis=1)
    first, last = A[:, 0, :], A[:, -1, :]
    drift = last - first
    trans = np.linalg.norm(mean[:, :3], axis=1)
    rotn = np.linalg.norm(mean[:, 3:6], axis=1)
    grip = mean[:, 6]
    internal = np.abs(np.diff(A, axis=1)).mean(axis=(1, 2))
    sign_changes = (np.diff(np.sign(A[:, :, 6]), axis=1) != 0).sum(axis=1).astype(float)
    prev = np.vstack([mean[:1], mean[:-1]])
    step = np.linalg.norm(mean - prev, axis=1)
    denom = np.maximum(np.linalg.norm(mean, axis=1) * np.linalg.norm(prev, axis=1), 1e-9)
    cos_prev = np.sum(mean * prev, axis=1) / denom
    intra_path = np.linalg.norm(np.diff(A[:, :, :3], axis=1), axis=2).sum(axis=1)
    intra_net = np.linalg.norm(A[:, -1, :3] - A[:, 0, :3], axis=1)
    cols = [
        mean, std, first, last, drift,
        trans[:, None], rotn[:, None], grip[:, None],
        internal[:, None], sign_changes[:, None], step[:, None],
        cos_prev[:, None], intra_path[:, None], intra_net[:, None],
        (intra_net / np.maximum(intra_path, 1e-9))[:, None],
        np.abs(A[:, :, :3]).mean(axis=(1, 2))[:, None],
        np.abs(A[:, :, 3:6]).mean(axis=(1, 2))[:, None],
        A[:, :, :3].std(axis=1).mean(axis=1)[:, None],
        np.arange(T, dtype=float)[:, None] * 0.0,  # placeholder, dropped by std filter
    ]
    out = np.concatenate([c if c.ndim == 2 else c[:, None] for c in cols], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def one_episode(arg):
    task, episode = arg
    path = CACHE / task / "right-16x32" / "client" / f"episode_{episode:02d}.npz"
    with np.load(path, allow_pickle=True) as d:
        state = np.asarray(d["state"])
        actions = np.asarray(d["actions"])
    ph = phase_interpolate(physical_tape(state))
    ac = phase_interpolate(action_tape(actions))
    return episode, ph.astype(np.float32), ac.astype(np.float32), len(state)


def main() -> None:
    manifest = json.loads((SRC / "feature_cache_manifest.json").read_text())
    meta = {
        n: np.load(SRC / "feature_cache" / manifest["arrays"][n]["file"], allow_pickle=False)
        for n in ("meta_task", "meta_episode", "meta_episode_length", "meta_failure",
                  "meta_init_state_id", "meta_flow_noise_seed")
    }
    tasks = meta["meta_task"]
    episodes = meta["meta_episode"]
    order = {(t, int(e)): i for i, (t, e) in enumerate(zip(tasks, episodes))}

    phys = None
    act = None
    lens = np.zeros(len(tasks), dtype=int)
    with ProcessPoolExecutor(max_workers=24) as pool:
        for task in np.unique(tasks):
            args = [(task, int(e)) for e in episodes[tasks == task]]
            for episode, ph, ac, T in pool.map(one_episode, args, chunksize=8):
                i = order[(task, episode)]
                if phys is None:
                    phys = np.zeros((len(tasks),) + ph.shape, dtype=np.float32)
                    act = np.zeros((len(tasks),) + ac.shape, dtype=np.float32)
                phys[i], act[i], lens[i] = ph, ac, T
            print("done", task, flush=True)

    assert (lens == meta["meta_episode_length"]).all(), "episode length mismatch vs cache"
    np.savez_compressed(
        OUT / "sentinel_tapes.npz",
        physical=phys, action=act,
        task=tasks, episode=episodes,
        episode_length=meta["meta_episode_length"], failure=meta["meta_failure"],
        init_state_id=meta["meta_init_state_id"],
        flow_noise_seed=meta["meta_flow_noise_seed"],
    )
    print("physical", phys.shape, "action", act.shape)


if __name__ == "__main__":
    main()
