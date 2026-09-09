#!/usr/bin/env python3
"""The three detector arms and the operating-point table they are scored on.

One engine, three arms, differing only in what they ask of the same per-chunk
routing channel:

  threshold    (control)   is the value large *now*?
  change_point             did the value *shift*, with the episode's own early
                           chunks as the baseline?
  persistence              has the value been anomalous *repeatedly*, against a
                           frozen population baseline?

Every operating point is one (channel, layer, direction, statistic, threshold).
For each one the engine stores a *per-task* count table, so that the pooled
result and every leave-one-task-out result are the same numbers summed over
different task sets - a LOTO number can never come from a different pipeline
than the pooled number it is compared with.

Detector input is chunk index + routing only.  No cap, no phase, no task
identity, no suite, no per-task threshold.
"""

from __future__ import annotations

import numpy as np

import modes_common as M

# Statistic families.  `self_baseline` sits in the change-point arm: it is the
# degenerate two-regime contrast (first W chunks vs now) and is the cleanest
# test of the "self baseline fails when the baseline window is already
# abnormal" half of the hypothesis.
ARMS = {
    "threshold": ("threshold",),
    "change_point": ("change_point", "self_baseline"),
    "persistence": ("persist_frac", "persist_run", "persist_cycles"),
}
STAT_ARM = {s: a for a, ss in ARMS.items() for s in ss}
GATES = (0.90, 0.975)   # frozen development quantiles of the standardised value

# Threshold grid, parameterised by how many *episodes in the cohort* the
# threshold lets alarm at all.  The grid is label-free: it counts alarms, not
# false alarms, so it can be re-derived on a cohort whose outcomes are unknown.
# Two ways of carrying it to external are reported side by side:
#   frozen value  - external is scored at development's numeric thresholds
#                   (strictest transfer; the alarm *rate* is then free to move)
#   rate matched  - external re-derives the threshold at the same k, using its
#                   own unlabelled routing distribution (the alarm rate is held
#                   fixed, which is what an operator actually controls)
# Duplicates are kept, never uniqued, so the two tables align row for row.
GRID_K = (4, 6, 9, 13, 19, 27, 38, 54, 76, 108, 152, 215, 304, 430, 608, 860, 1216)

# A channel that barely moves inside an episode cannot support a change-point
# or a persistence statistic: its "run length" is just the episode length, so
# the detector collapses into "still running at chunk q0".  Channels above this
# fraction are kept in the table but excluded from the headline pool.
MAX_WITHIN_EPISODE_CONSTANT = 0.25

GROUP_NONRISK, GROUP_DROP, GROUP_GRASP, GROUP_OTHER = 0, 1, 2, 3
N_GROUP = 4
N_LEAD = len(M.LEADS)


def pool_masks(ops) -> dict[str, np.ndarray]:
    """Channel pools, fixed in advance and shared by every script.

    headline      routing channels that actually move inside an episode
    flagged       + channels that mostly do not (set_dwell), where a run-length
                  statistic degenerates into "still running at chunk q0"
    null_noise    white noise - the information floor
    null_elapsed  the chunk counter itself.  A persistence statistic on it *is*
                  the cap-free fixed-chunk baseline, so it measures how much of
                  an arm's score is horizon length rather than routing.
    null_epconst  channels drawn once per episode and held constant - the
                  bit-constant controls
    """
    q = ops["quantity"].to_numpy()
    ctrl = ops["is_control"].to_numpy()
    movable = (ops["within_episode_constant"].to_numpy()
               <= MAX_WITHIN_EPISODE_CONSTANT)
    return {
        "headline": ~ctrl & movable,
        "flagged": ~ctrl,
        "null_noise": q == "ctrl_white_noise",
        "null_elapsed": q == "ctrl_const_elapsed",
        "null_epconst": ctrl & ~movable,
    }


def episode_groups(frame: dict) -> np.ndarray:
    g = np.full(len(frame["risk"]), GROUP_NONRISK, np.int64)
    g[frame["risk"]] = GROUP_OTHER
    g[frame["risk"] & (frame["mode"] == M.MODE_DROP)] = GROUP_DROP
    g[frame["risk"] & (frame["mode"] == M.MODE_GRASP)] = GROUP_GRASP
    return g


def task_codes(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    names, code = np.unique(frame["task"], return_inverse=True)
    return names, code.astype(np.int64)


def statistics(u: np.ndarray, valid: np.ndarray, gates: dict[float, float]
               ) -> dict[str, np.ndarray]:
    """All six statistics for one standardised channel, all causal."""
    out = {
        "threshold": M.stat_threshold(u, valid),
        "change_point": M.stat_change_point(u, valid),
        "self_baseline": M.stat_self_baseline(u, valid),
    }
    for level, gate in gates.items():
        tag = f"g{int(round(level * 1000)):03d}"
        out[f"persist_frac|{tag}"] = M.stat_persist_fraction(u, valid, gate)
        out[f"persist_run|{tag}"] = M.stat_persist_run(u, valid, gate)
        out[f"persist_cycles|{tag}"] = M.stat_persist_cycles(u, valid, gate)
    return out


def stat_family(name: str) -> str:
    return name.split("|")[0]


def grid_from_episodes(runmax: np.ndarray) -> np.ndarray:
    """Thresholds that let exactly GRID_K episodes alarm - a label-free grid.

    Taken from the order statistics of the per-episode running maximum over
    *all* episodes, so threshold k lets ~k episodes alarm at lead 0.  `>=` is
    used at scoring time, so a tie group at the threshold inflates the count
    above k; that is intended - dropping the tie group is the documented
    failure mode, not the tie itself.  Duplicates are never removed, so the
    development and external tables have identical row order.
    """
    final = runmax[:, -1]
    fin = final[np.isfinite(final)]
    if fin.size <= max(GRID_K):
        return np.full(len(GRID_K), np.inf)
    order = np.sort(fin)[::-1]
    return order[np.asarray(GRID_K) - 1].astype(np.float64)


def _episode_running_max(stat: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.where(np.isnan(stat), -np.inf, stat), axis=1)


def count_table(runmax: np.ndarray, thresholds: np.ndarray, length: np.ndarray,
                group: np.ndarray, task_code: np.ndarray, n_task: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """(n_threshold, n_task, N_GROUP, N_LEAD) alarm counts, plus median alarm chunk.

    An episode contributes to lead budget b when its alarm fires and
    length - alarm_chunk >= b.  Counting every budget at once is one bincount
    over the key (task, group, #budgets-met).
    """
    leads = np.asarray(M.LEADS)
    n_th = len(thresholds)
    out = np.zeros((n_th, n_task, N_GROUP, N_LEAD), np.int64)
    med_chunk = np.full(n_th, np.nan)
    n_bin = n_task * N_GROUP * (N_LEAD + 1)
    base_key = (task_code * N_GROUP + group) * (N_LEAD + 1)
    for j, h in enumerate(thresholds):
        mask = runmax >= h
        hit = mask.any(axis=1)
        first = np.where(hit, np.argmax(mask, axis=1), -1)
        lead = np.where(hit, length - first, -1)
        met = (lead[:, None] >= leads[None, :]).sum(axis=1)   # 0..N_LEAD
        met = np.where(hit, met, 0)
        counts = np.bincount(base_key + met, minlength=n_bin)
        counts = counts.reshape(n_task, N_GROUP, N_LEAD + 1)
        # counts[..., j] is "meets exactly j budgets"; the suffix sum from the
        # top gives "meets budget b" for every b.
        out[j] = np.cumsum(counts[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:]
        if hit.any():
            med_chunk[j] = float(np.median(first[hit]))
    return out, med_chunk


def standardised(x: np.ndarray, valid: np.ndarray, ref: dict, sign: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Standardised channel plus the mask of chunks it is actually defined on."""
    u = M.standardise(x, valid, ref, sign)
    ok = valid & np.isfinite(u)
    return np.where(ok, u, 0.0), ok


def within_episode_constant(x: np.ndarray, valid: np.ndarray,
                            length: np.ndarray) -> float:
    xv = np.where(valid, x, np.nan)
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanmin(xv, axis=1), np.nanmax(xv, axis=1)
    take = length >= 4
    return float((~(hi[take] > lo[take])).mean())


def calibrate(x: np.ndarray, frame: dict) -> dict:
    """Development-only calibration for one channel: per-chunk reference + gates."""
    valid = frame["valid"]
    ref = M.chunk_reference(x, valid)
    gates = {}
    for sign in (+1, -1):
        u, ok = standardised(x, valid, ref, sign)
        pool = u[ok]
        gates[sign] = {level: float(np.quantile(pool, level)) for level in GATES}
    return {"ref": ref, "gates": gates,
            "within_episode_constant": within_episode_constant(
                x, valid, frame["length"])}


def channel_table(x: np.ndarray, frame: dict, calib: dict, sign: int,
                  task_code: np.ndarray, n_task: int, group: np.ndarray,
                  grid_source: dict | None) -> dict:
    """Every operating point for one (channel, direction).

    `grid_source` is None when the grid is being *defined* (development) and a
    dict of frozen thresholds when it is being *applied* (external).  The
    per-chunk reference and the persistence gates are always the development
    ones - external is never used to calibrate anything.
    """
    u, ok = standardised(x, frame["valid"], calib["ref"], sign)
    stats = statistics(u, ok, calib["gates"][sign])
    result = {}
    for name, stat in stats.items():
        rm = _episode_running_max(stat)
        th = grid_from_episodes(rm) if grid_source is None \
            else grid_source[name]
        counts, med = count_table(rm, th, frame["length"], group, task_code, n_task)
        result[name] = {"thresholds": th, "counts": counts, "median_chunk": med}
    return result
