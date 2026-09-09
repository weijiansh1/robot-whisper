"""Is a same-state fork trustworthy at the phases where selection would actually matter?

The candidate-diversity screen was retracted once because its noise floor used the
observation's end-effector position, and ``env.step()`` returns images rendered one
substep behind the state it reports, which inflated the floor to ~3 mm.  The same runs
also stored the full simulator state, and re-reading those columns says something the
retraction did not: the floor is *exactly zero* at 26 of 45 probe points, so identical
inputs reproduce bit-for-bit -- but at the contact phases it is 2e-4 to 0.52, and at three
of them it is larger than the spread between different candidates.  Contact is where
grasping and pulling happen, so it is exactly where a selection experiment has to work.

branch_snapshot.restore_full_state already handles the three pieces of state that
get_sim_state() omits (OSC controller, episode clock, qacc_warmstart).  What it cannot
reach is the constraint solver's own scratch -- efc_* and the contact-pair cache -- which
mj_resetData clears but set_state_from_flattened does not.  Hence three measurements:

  A  inference determinism   same restored observation, same flow-noise tensor, R times.
                             Every noise floor below is defined relative to this, so if it
                             is not zero nothing else here means what it says.
  B  drift vs candidate index  N candidates from one snapshot, re-running candidate 0
                             every few branches.  Order-dependent error is the dangerous
                             kind: it correlates with candidate index and so biases a
                             ranking rather than just blurring it.
  C  the same with a hard reset  sim.reset() before the restore, to see whether clearing
                             mjData's scratch removes it.

Verdict rule, fixed before running: the fork is usable at a phase if the drift after N
branches stays under 5% of the between-candidate spread at that phase.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")
from branch_snapshot import restore_full_state, save_full_state  # noqa: E402

from himoe_libero_bridge.client import PolicyClient  # noqa: E402
from himoe_libero_bridge.libero_runtime import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
)
from himoe_libero_bridge.preprocess import build_policy_observation  # noqa: E402
from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)

from rollout_with_routes import sim_joint_layout  # noqa: E402


def hard_restore(env, snap):
    """restore_full_state, but wipe mjData's solver scratch first.

    set_state_from_flattened writes time/qpos/qvel/act and nothing else, so efc_* and the
    contact-pair cache survive from whatever branch ran before.  mj_resetData clears them;
    qacc_warmstart is cleared too but restore_full_state writes it back from the snapshot,
    so the result is the snapshot's mjData rather than a zeroed one.
    """
    base = getattr(env, "env", env)
    base.sim.reset()
    return restore_full_state(env, snap)


def replay_restore(env, snap, initial, prefix_actions, settle):
    """Rebuild the branch point from env.reset() instead of restoring into a live env.

    This is the fallback for phases where neither restore mode reproduces: reset the
    episode outright and re-drive it with the same actions that reached the snapshot.
    Nothing carries over because nothing is being written into a used mjData -- the branch
    point is re-derived rather than re-installed.  It costs the prefix again per branch.
    """
    env.reset()
    obs = env.set_init_state(initial)
    for _ in range(settle):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
    for a in prefix_actions:
        obs, _, _, _ = env.step(np.asarray(a, np.float64).tolist())
    return obs


def split_state(state: np.ndarray, nq: int) -> tuple[np.ndarray, np.ndarray]:
    """[time, qpos, qvel] -> (qpos, qvel).

    Reported separately because a single norm over all 79 columns mixes metres with
    metres per second and the resulting number is not interpretable as either.
    """
    return state[1:1 + nq], state[1 + nq:]


def spread(rows: np.ndarray) -> dict:
    """Mean and max pairwise distance.

    Both, because the mean is what a unimodal cloud shows and the max is what a split
    shows: 14 candidates down one route and 2 down another is precisely the case worth
    selecting in, and averaging over all pairs buries it under the majority.
    """
    if len(rows) < 2:
        return {"mean": 0.0, "max": 0.0}
    d = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=-1)
    iu = np.triu_indices(len(rows), k=1)
    return {"mean": float(d[iu].mean()), "max": float(d[iu].max())}


def probe(env, client, prompt, cfg, layout, n_candidates, seed_base, repeat_every,
          initial=None, prefix_actions=()):
    nq = int(layout["nq"])
    snap = save_full_state(env)
    modes = [("soft", lambda e, s: restore_full_state(e, s)),
             ("hard", lambda e, s: hard_restore(e, s))]
    if initial is not None:
        modes.append(("replay", lambda e, s: replay_restore(
            e, s, initial, prefix_actions, cfg.settle_steps)))

    # ---- A: same input, many times.  Any nonzero here rewrites every floor below.
    obs = restore_full_state(env, snap)
    req = dict(build_policy_observation(obs, prompt))
    req[FLOW_NOISE_KEY] = np.random.default_rng(seed_base).standard_normal(
        FLOW_NOISE_SHAPE).astype(np.float32)
    repeats = [np.asarray(validate_action_response(client.infer(dict(req)))[ACTION_KEY],
                          np.float64) for _ in range(8)]
    infer_err = float(max(np.abs(r - repeats[0]).max() for r in repeats[1:]))

    # one query per candidate, all from the same restored observation
    chunks = []
    for i in range(n_candidates):
        obs = restore_full_state(env, snap)
        r = dict(build_policy_observation(obs, prompt))
        r[FLOW_NOISE_KEY] = np.random.default_rng(seed_base + 1 + i).standard_normal(
            FLOW_NOISE_SHAPE).astype(np.float32)
        chunks.append(np.asarray(
            validate_action_response(client.infer(r))[ACTION_KEY], np.float64))

    # ---- B and C: execute every candidate, re-running candidate 0 periodically
    out = {"inference_max_abs_action_diff": infer_err, "n_candidates": n_candidates}
    for tag, restore in modes:
        landings, drift, reference = [], [], None
        for i in range(n_candidates):
            restore(env, snap)
            for a in chunks[i][: cfg.replan_steps]:
                env.step(a.tolist())
            landings.append(np.asarray(env.get_sim_state(), np.float64))
            if i % repeat_every == 0:
                restore(env, snap)
                for a in chunks[0][: cfg.replan_steps]:
                    env.step(a.tolist())
                here = np.asarray(env.get_sim_state(), np.float64)
                if reference is None:
                    reference = here
                drift.append({"after_branches": i + 1,
                              "max_abs": float(np.abs(here - reference).max())})
        landings = np.array(landings)
        qpos = np.array([split_state(s, nq)[0] for s in landings])
        qvel = np.array([split_state(s, nq)[1] for s in landings])
        final = drift[-1]["max_abs"]
        gap = spread(qpos)["mean"]
        out[tag] = {"qpos_spread": spread(qpos), "qvel_spread": spread(qvel),
                    "drift": drift, "final_drift_max_abs": final,
                    "drift_over_spread": (final / gap) if gap else float("inf"),
                    "usable": bool(gap > 0 and final < 0.05 * gap)}
    # leave the environment back at the branch point so the prefix stream that walks to
    # the next phase continues from the snapshot rather than from the last candidate
    return out, restore_full_state(env, snap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--phases", default="6,9,11,12")
    ap.add_argument("--n-candidates", type=int, default=32)
    ap.add_argument("--repeat-every", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--prefix-seed", type=int, default=1000)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    phases = sorted(int(p) for p in args.phases.split(","))
    cfg = EpisodeConfig(
        task_suite="libero_goal", task_id=args.task_id, init_state_id=args.init_state_id,
        seed=7, host="127.0.0.1", port=args.port, libero_root=args.libero_root,
        output_root="/tmp", settle_steps=10, max_steps=1000,
        replan_steps=args.replan_steps, inference_timeout=300.0)

    env, obs, _task, prompt = _load_task(cfg)
    layout = sim_joint_layout(env)
    rows = []
    try:
        with PolicyClient(host="127.0.0.1", port=args.port, connect_timeout=600.0,
                          inference_timeout=300.0) as client:
            # the state the episode starts from, kept so a branch can be re-derived
            # from scratch rather than restored into an already-used simulator
            initial = np.asarray(env.get_sim_state(), np.float64).copy()
            prefix_actions: list = []
            for _ in range(cfg.settle_steps):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
            prefix_rng = np.random.default_rng(args.prefix_seed)
            print("task %d init %d | %s" % (args.task_id, args.init_state_id, prompt),
                  flush=True)
            print("%-6s %-11s %-11s %-11s %-11s %-11s %s"
                  % ("阶段", "候选离散", "软恢复", "硬恢复", "重放恢复", "推理重复差",
                     "软/硬/重放可用"), flush=True)
            for k in range(max(phases) + 1):
                if k in phases:
                    r, obs = probe(env, client, prompt, cfg, layout,
                                   args.n_candidates, args.seed_base + 100 * k,
                                   args.repeat_every, initial, list(prefix_actions))
                    r["phase"] = k
                    rows.append(r)
                    rep = r.get("replay")
                    print("%-6d %-11.5f %-11.3e %-11.3e %-11.3e %-11.3e %s"
                          % (k, r["soft"]["qpos_spread"]["mean"],
                             r["soft"]["final_drift_max_abs"],
                             r["hard"]["final_drift_max_abs"],
                             rep["final_drift_max_abs"] if rep else float("nan"),
                             r["inference_max_abs_action_diff"],
                             "/".join(str(r[m]["usable"]) for m in ("soft", "hard", "replay")
                                      if m in r)),
                          flush=True)
                    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(args.out).write_text(json.dumps(rows, indent=2))
                # advance one control step along a fixed prefix stream
                req = dict(build_policy_observation(obs, prompt))
                req[FLOW_NOISE_KEY] = prefix_rng.standard_normal(
                    FLOW_NOISE_SHAPE).astype(np.float32)
                actions = np.asarray(
                    validate_action_response(client.infer(req))[ACTION_KEY], np.float64)
                for a in actions[: cfg.replan_steps]:
                    obs, _, _, _ = env.step(a.tolist())
                    prefix_actions.append(a.copy())
    finally:
        try:
            env.close()
        except BaseException:
            pass

    print("\n门槛：N=%d 次分叉后的漂移 < 候选离散度的 5%%" % args.n_candidates)
    for m in ("soft", "hard", "replay"):
        have = [r[m] for r in rows if m in r]
        if have:
            print("%-7s 全相位通过: %-6s  (%d/%d 相位)"
                  % (m, all(h["usable"] for h in have),
                     sum(h["usable"] for h in have), len(have)))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
