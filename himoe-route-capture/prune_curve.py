"""Retrospective partial-rollout pruning: does the routing pick the branches worth keeping?

Every earlier routing result here is a *decode*: fit a probe on the routing at control
step t and see how well it separates the 64 outcomes.  That is not the same claim as the
one a RAD-style method needs.  RAD prunes with no labels at all -- it keeps the candidates
that sit in the densest region of the internal-state cloud and drops the rest -- so the
question is whether successful rollouts *form* that dense region, not whether a supervised
probe can find them.  A 0.85 decode and a useless density rule are entirely compatible.

So each control step t is treated as a pruning decision: 64 partial rollouts are alive,
each is scored using only information available at t, the top B are kept, and the metric is
how enriched the kept set is in eventual successes.

  arms          density and medoid scores over six representations, plus supervised
                LOO-ridge references and the random / oracle bounds
  budgets       B in 1, 2, 4, 8, 16, 32
  primary       Enrichment@B = precision@B - base rate
  null          label permutation with the kept sets held fixed (exact for the
                unsupervised arms, since their scores never see a label), and an
                exact hypergeometric p per cell

Two traps this script is built to avoid.

Survivorship.  Successes finish in 17-21 control steps and failures run to the 300-action
cap at 30, so past the first episode's exit "still running" is itself the label.  The sweep
therefore stops at the last step where all 64 are alive and prints that cap.

Metric saturation.  The proposal names KeepSuccess@B -- the chance of retaining at least
one success -- as the primary endpoint.  At 27/64 a *random* prune to B=8 keeps a success
99.1% of the time and B=16 keeps one 100.0% of the time, so above B=4 it cannot separate any
method from chance.  It is reported, but Enrichment@B is what the verdict rests on.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import hypergeom

from within64_lib import action_token_probs, load_run
from within64_analyze import _hat, _loo_scores, reduce_dims
from pc_meaning6 import knn_residual
from object_control import object_features, robot_qpos_features

BUDGETS = (1, 2, 4, 8, 16, 32)
WINDOW = 3          # control steps of routing history in the temporal arm
N_PCA = 30          # unsupervised, label-free, and shared by observed and permuted runs
N_PERM = 2000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- representations

def build_blocks(run, client, sim_dir):
    """Per-control-step feature blocks, one row per episode, episode order fixed.

    Returns ``(blocks, y, cap, meta)`` where ``blocks[name][t]`` is a [64, d] array.
    """
    eps = sorted(run.episodes, key=lambda e: e.index)
    y = np.array([int(e.success) for e in eps])
    n_control = np.array([e.n_control for e in eps])
    cap = int(n_control.min()) - 1           # last step where every branch is still alive

    layout_path = sim_dir / "sim_layout.json"
    layout = json.loads(layout_path.read_text()) if layout_path.exists() else None

    route, proprio, actions, world = [], [], [], []
    for e in eps:
        route.append(action_token_probs(e).reshape(e.n_control, -1))
        with np.load(client / ("episode_%02d.npz" % e.index), allow_pickle=True) as z:
            proprio.append(np.asarray(z["state"], np.float64))
            actions.append(np.asarray(z["actions"], np.float64).reshape(e.n_control, -1))
            sim = np.asarray(z["sim_state"], np.float64) if "sim_state" in z.files else None
        if layout is not None and sim is not None:
            world.append(np.column_stack([robot_qpos_features(sim, layout),
                                          object_features(sim, layout)]))
        else:
            world.append(proprio[-1])

    # The residual control is computed on the pooled matrix, exactly as object_control.py
    # does it: for a row at (episode e, step t) the 15 proprio neighbours are drawn from
    # every step of every *other* episode.  Restricting the neighbour pool to the same
    # control step would leave only 63 candidates and turn the control into a much
    # coarser one, so the two analyses would stop being comparable.
    ep_id = np.concatenate([[e.index] * e.n_control for e in eps])
    step_id = np.concatenate([np.arange(e.n_control) for e in eps])
    route_pooled = np.concatenate(route)
    world_pooled = np.concatenate(world)
    resid_pooled = knn_residual(route_pooled, world_pooled, ep_id)

    order = {e.index: i for i, e in enumerate(eps)}
    rows = np.empty((cap + 1, len(eps)), dtype=int)
    for t in range(cap + 1):
        at_t = np.flatnonzero(step_id == t)
        rows[t] = at_t[np.argsort([order[ep_id[r]] for r in at_t])]

    blocks: dict[str, dict[int, np.ndarray]] = {
        "route": {t: route_pooled[rows[t]] for t in range(cap + 1)},
        "route_resid": {t: resid_pooled[rows[t]] for t in range(cap + 1)},
        "state": {t: world_pooled[rows[t]] for t in range(cap + 1)},
        "action": {t: np.stack([actions[i][t] for i in range(len(eps))])
                   for t in range(cap + 1)},
    }
    # R_{t-2..t} plus the one-step routing increment: the proposal's claim is that the
    # information is in the routing *dynamics*, not in one static allocation
    blocks["route_win"] = {
        t: np.hstack([blocks["route"][t - k] for k in range(WINDOW - 1, -1, -1)]
                     + [blocks["route"][t] - blocks["route"][t - 1]])
        for t in range(WINDOW - 1, cap + 1)}
    # where the chunk actually put the world -- the A^3 analogue, and the strongest
    # non-routing signal available at decision time
    if layout is not None:
        blocks["induced"] = {t: world_pooled[rows[t + 1]] for t in range(cap)}

    # A block with no between-branch variance carries no information, but it does not
    # produce a null result -- it produces a spurious *negative* one.  With a constant
    # design the ridge collapses to the intercept and the leave-one-out fit becomes
    # (mean of the others), which is mechanically lowest exactly where the held-out label
    # is highest, so the probe anti-ranks the branches perfectly.  The world state at
    # control step 0 is exactly this case: all 64 branches share the init state.  Drop
    # such cells rather than report -6.78.
    degenerate = []
    for name, steps in blocks.items():
        for t in [t for t in sorted(steps) if float(steps[t].std(0).max()) <= 1e-12]:
            degenerate.append("%s@%d" % (name, t))
            del steps[t]

    meta = {"n_episodes": len(eps), "n_success": int(y.sum()),
            "degenerate_blocks_dropped": degenerate,
            "all_alive_cap": cap, "control_steps_mean": float(n_control.mean()),
            "control_steps_min": int(n_control.min()), "control_steps_max": int(n_control.max()),
            "world_state_dims": int(world_pooled.shape[1]),
            "world_state_source": "sim_state" if layout is not None else "observation/state (8-dim eef)"}
    return blocks, y, cap, meta, float(n_control.mean())


def embed(z: np.ndarray, standardise: bool) -> np.ndarray:
    """Label-free PCA to N_PCA dims; distances are then Euclidean in that subspace.

    ``standardise`` is on only for the physical-state blocks, whose columns genuinely mix
    units (metres for the slide joints, radians for the hinges, dimensionless quaternion
    components), so an unstandardised distance would be whichever unit happens to be
    largest.  The routing and action blocks are left raw: their columns are already in one
    unit, and z-scoring 2560 expert probabilities divides the near-constant experts by a
    tiny standard deviation and lets pure noise dominate the metric.
    """
    if standardise:
        z = (z - z.mean(0)) / (z.std(0) + 1e-12)
    return reduce_dims(z, N_PCA)


# ---------------------------------------------------------------- selection rules

def _pairwise(z: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def density_scores(d: np.ndarray, k: int) -> np.ndarray:
    """RAD's rule: minus the mean distance to the k nearest neighbours.

    k is tied to the retained budget B, so "keep the B densest" is literally "keep the B
    candidates whose B nearest neighbours are closest" -- no clustering hyperparameter and
    no threshold to tune.  B=1 would make k=1, i.e. a rule that rewards a tight pair rather
    than a dense region, so k is floored at 2.
    """
    k = int(np.clip(k, 2, d.shape[0] - 1))
    return -np.sort(d, axis=1)[:, :k].mean(1)


def medoid_scores(d: np.ndarray) -> np.ndarray:
    """Minus the mean distance to every other candidate -- the A^3 consensus rule.

    Unlike the density score this does not depend on B, so one ranking serves every budget.
    """
    return -np.where(np.isinf(d), 0.0, d).sum(1) / (d.shape[0] - 1)


def topk(score: np.ndarray, b: int) -> np.ndarray:
    """Indices of the b highest scores; ties broken by index so the run is deterministic."""
    return np.lexsort((np.arange(len(score)), -score))[:b]


# ---------------------------------------------------------------- evaluation

def cell(kept: np.ndarray, y: np.ndarray) -> dict:
    n, s, b = len(y), int(y.sum()), len(kept)
    hits = int(y[kept].sum())
    base = s / n
    # the selection never looked at a label, so "which b of the n" is exactly a
    # hypergeometric draw under the null and no permutation is needed for the per-cell p
    rv = hypergeom(n, s, b)
    p_obs = rv.pmf(hits)
    two_sided = float(sum(rv.pmf(k) for k in range(max(0, b - (n - s)), min(b, s) + 1)
                          if rv.pmf(k) <= p_obs + 1e-12))
    return {"precision": round(hits / b, 4),
            "enrichment": round(hits / b - base, 4),
            "keep_success": bool(hits > 0),
            "hypergeom_p": round(min(1.0, two_sided), 4)}


def null_sd(n: int, s: int, b: int) -> float:
    """Standard deviation of precision@B when the kept set is drawn at random.

    Raw enrichment is not comparable across budgets -- at B=1 it is +0.578 whenever the
    single kept branch happens to be a success, which happens 42% of the time by chance,
    so a search for the largest raw enrichment always lands on B=1 and reports nothing.
    Dividing by this puts every (step, budget) cell on one scale.
    """
    p = s / n
    return float(np.sqrt(b * p * (1 - p) * (n - b) / (n - 1)) / b)


def evaluate_bundle(blocks, y, cap, mean_len, n_perm=N_PERM) -> dict:
    n = len(y)
    base = float(y.mean())
    perms = np.stack([RNG.permutation(y) for _ in range(n_perm)], axis=1)   # [n, n_perm]

    arms: dict[str, dict] = {}
    sd = {b: null_sd(n, int(y.sum()), b) for b in BUDGETS if b < n}

    def record(name, t, b, kept, null_enrich=None):
        r = cell(kept, y)
        r.update(step=int(t), budget=int(b))
        # z has to come off the unrounded enrichment.  Taking it off the rounded one
        # (-0.4219 rather than -0.421875) inflates |z| by ~1e-3, which is normally
        # invisible but is fatal where the observed cell and its own null coincide
        # exactly: the null then never reaches the observed value and a p of 1.0 is
        # reported as 1/(1+n_perm).  ``z_exact`` is what every comparison below uses.
        r["z_exact"] = (float(y[kept].mean()) - base) / sd[b]
        r["z"] = round(r["z_exact"], 3)
        # fraction of the full 64-branch rollout budget spent if the prune happens here
        r["compute_fraction"] = round((n * t + b * (mean_len - t)) / (n * mean_len), 4)
        if null_enrich is not None:
            r["null_enrichment_p95"] = round(float(np.quantile(np.abs(null_enrich), 0.95)), 4)
        arms.setdefault(name, {"cells": [], "null_max": None, "null_sum": None})[
            "cells"].append(r)

    # ---- unsupervised arms.  The kept set is a fixed function of the features, so a
    # label permutation with the kept sets held fixed is an exact null, and the same
    # permutations give the max-statistic FWER across the whole (t, B) grid.
    for block, steps in blocks.items():
        standardise = block in ("state", "induced")
        for rule in ("density", "medoid"):
            name = "%s/%s" % (block, rule)
            grid_null, grid_sum = np.zeros(n_perm), np.zeros(n_perm)
            for t in sorted(steps):
                d = _pairwise(embed(steps[t], standardise))
                med = medoid_scores(d)
                for b in BUDGETS:
                    if b >= n:
                        continue
                    kept = topk(med if rule == "medoid" else density_scores(d, b), b)
                    null = perms[kept].mean(0) - base          # [n_perm]
                    grid_null = np.maximum(grid_null, np.abs(null) / sd[b])
                    grid_sum += null / sd[b]
                    record(name, t, b, kept, null)
            arms[name]["null_max"], arms[name]["null_sum"] = grid_null, grid_sum

    # ---- supervised references.  Here the kept set *does* depend on the labels, so the
    # permutation has to refit; the ridge hat matrix does not depend on y, so all
    # permutations come out of one matrix product per step.
    for block in ("route", "route_win", "route_resid", "state", "action", "induced"):
        if block not in blocks:
            continue
        name = "%s/supervised" % block
        grid_null, grid_sum = np.zeros(n_perm), np.zeros(n_perm)
        for t in sorted(blocks[block]):
            z = embed(blocks[block][t], block in ("state", "induced"))
            hat, h_diag = _hat(z)
            y_pm = np.where(y == 1, 1.0, -1.0)
            obs = _loo_scores(hat, h_diag, y_pm[:, None])[:, 0]
            null_scores = _loo_scores(hat, h_diag, np.where(perms == 1, 1.0, -1.0))
            null_rank = np.argsort(-null_scores, axis=0)       # [n, n_perm]
            for b in BUDGETS:
                if b >= n:
                    continue
                kept = topk(obs, b)
                null = np.take_along_axis(perms, null_rank[:b], axis=0).mean(0) - base
                grid_null = np.maximum(grid_null, np.abs(null) / sd[b])
                grid_sum += null / sd[b]
                record(name, t, b, kept, null)
        arms[name]["null_max"], arms[name]["null_sum"] = grid_null, grid_sum

    # ---- bounds.  "random" is averaged over 200 independent draws per cell rather than
    # taken as one draw: a single random prune is itself a noisy number, and what this arm
    # is for is confirming that the machinery sits on zero.
    for t in range(cap + 1):
        for b in BUDGETS:
            if b >= n:
                continue
            record("oracle", t, b, topk(y.astype(float), b))
            draws = np.array([y[RNG.permutation(n)[:b]].mean() for _ in range(200)])
            arms.setdefault("random", {"cells": [], "null_max": None})["cells"].append(
                {"step": int(t), "budget": int(b),
                 "precision": round(float(draws.mean()), 4),
                 "enrichment": round(float(draws.mean() - base), 4),
                 "z_exact": float(draws.mean() - base) / sd[b],
                 "z": round(float(draws.mean() - base) / sd[b], 3),
                 "keep_success": True,
                 "hypergeom_p": 1.0,
                 "compute_fraction": round((n * t + b * (mean_len - t)) / (n * mean_len), 4)})

    out = {}
    for name, a in arms.items():
        cells = a["cells"]
        best = max(cells, key=lambda c: abs(c["z_exact"]))
        entry = {"cells": cells,
                 "best": {k: best[k] for k in ("step", "budget", "enrichment", "z",
                                               "precision", "hypergeom_p")}}
        if a["null_max"] is not None:
            # the tolerance matters for the same reason: a cell whose null reproduces
            # the observed statistic exactly must count as "reached", not as a miss
            fwer = float((1 + (a["null_max"] >= abs(best["z_exact"]) - 1e-9).sum())
                         / (1 + n_perm))
            entry["best"]["fwer_p"] = round(fwer, 4)
            entry["grid_null_max_z_p95"] = round(float(np.quantile(a["null_max"], 0.95)), 4)
            # Whether an arm is *systematically* tilted is a different question from
            # whether its single best cell is extreme, and it is the one that carries
            # across bundles.  The cells are heavily correlated -- adjacent steps and
            # nested budgets over the same 64 branches -- so the mean z cannot be tested
            # against 1/sqrt(n_cells).  The permutation null of the same mean carries
            # that correlation with it and needs no independence assumption.
            mean_z = float(np.mean([c["z_exact"] for c in cells]))
            null_mean = a["null_sum"] / len(cells)
            entry["mean_z"] = round(mean_z, 3)
            entry["mean_z_p"] = round(float(
                (1 + (np.abs(null_mean) >= abs(mean_z) - 1e-9).sum()) / (1 + n_perm)), 4)
        out[name] = entry
    return out


# ---------------------------------------------------------------- entry point

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--sim-dir", help="defaults to --client-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--permutations", type=int, default=N_PERM)
    args = ap.parse_args()

    client = pathlib.Path(args.client_dir)
    sim_dir = pathlib.Path(args.sim_dir) if args.sim_dir else client
    run = load_run(args.server_dir, args.client_dir)
    blocks, y, cap, meta, mean_len = build_blocks(run, client, sim_dir)

    if cap < 1:
        sys.exit("the first episode exits at control step %d; nothing to prune" % (cap + 1))
    print("%d episodes, %d successes (base %.3f)" % (meta["n_episodes"], meta["n_success"],
                                                     y.mean()))
    print("all-alive cap: control step %d  (episode lengths %d..%d, mean %.1f)"
          % (cap, meta["control_steps_min"], meta["control_steps_max"], mean_len))
    print("world state: %s, %d dims" % (meta["world_state_source"], meta["world_state_dims"]))
    print("blocks: %s\n" % ", ".join("%s(%d steps)" % (k, len(v)) for k, v in blocks.items()))

    arms = evaluate_bundle(blocks, y, cap, mean_len, args.permutations)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "budgets": list(BUDGETS),
                               "window": WINDOW, "permutations": args.permutations,
                               "arms": arms}, indent=2))

    print("%-24s %-6s %-4s %-8s %-7s %-9s %-9s %s"
          % ("arm", "step", "B", "enrich", "z", "FWER", "mean_z", "mean_z_p"))
    for name in sorted(arms):
        a, b = arms[name], arms[name]["best"]
        print("%-24s %-6d %-4d %+-8.3f %+-7.2f %-9s %-9s %s"
              % (name, b["step"], b["budget"], b["enrichment"], b["z"],
                 ("%.4f" % b["fwer_p"]) if "fwer_p" in b else "-",
                 ("%+.3f" % a["mean_z"]) if "mean_z" in a else "-",
                 ("%.4f" % a["mean_z_p"]) if "mean_z_p" in a else "-"))
    print("\n最强 cell（按 |z| 在整个 step x budget 网格上取），FWER 一列已经为这次搜索付过账。")

    # the per-step profile at one budget is where the window question actually lives:
    # the grid maximum says whether anything is there at all, this says when
    show = [a for a in ("route/density", "route_win/density", "route_resid/density",
                        "state/density", "action/density", "induced/density",
                        "route/supervised", "state/supervised", "random")
            if a in arms]
    for budget in (8, 16):
        print("\nEnrichment@%d 逐控制步（基础成功率 %.3f）" % (budget, y.mean()))
        print("%-5s %s" % ("step", " ".join("%-14s" % a.replace("/density", "/dens")
                                            .replace("/supervised", "/sup") for a in show)))
        for t in range(cap + 1):
            row = []
            for a in show:
                c = [c for c in arms[a]["cells"] if c["step"] == t and c["budget"] == budget]
                row.append("%+.3f (%+.1f)" % (c[0]["enrichment"], c[0]["z"]) if c else "-")
            print("%-5d %s" % (t, " ".join("%-14s" % v for v in row)))
    print("\n括号内是按随机减枝的零分布标准差归一后的 z；单个 cell 的 |z| > 2 才值得看，"
          "而且要对照上表的 FWER。")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
