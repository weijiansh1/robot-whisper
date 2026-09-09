"""Same-state candidate forking: does the routing tell good candidates from bad ones?

Everything measured so far compares episodes that are in *different* world states by
the time the signal appears, so "routing predicts the outcome" cannot be separated
from "routing reads the scene".  This closes that: at one snapshot, N candidates are
drawn by varying the flow noise only, then each is executed from the identical
restored state, so the only thing that differs between them is the candidate itself.

  snapshot s_q  ->  N queries, same observation, different flow noise
                ->  N (action chunk, routing) pairs
                ->  restore s_q N times, execute each chunk, measure what it did

The label is drawer travel over the candidate chunk plus a short policy continuation.
The chunk on its own turned out to be degenerate: measured over 15 snapshots, at 11 of
them all eight candidates moved the drawer by exactly zero, because before the pull
begins nothing the arm does shows up in that joint within ten actions.  Eventual
episode success is the opposite problem -- a dozen replans away, so it would mostly
measure those.  The continuation is the middle ground, and it runs under noise shared
across candidates so the candidate chunk stays the only difference.

Snapshot restore was verified by probe_snapshot_fidelity.py (rerun error 5e-15 against
a between-candidate gap of 0.28) and probe_warmstart.py (5e-16 after seven other
candidates run in between -- but only if the solver warm start is left alone and
robosuite's Python-side step counter is reset).  Candidate 0 is re-executed at every
snapshot here, and its drift is recorded next to the between-candidate spread so the
signal is always reported against its own noise floor.

Routing is recorded server-side (serve_with_recorder.py --store-full-probs), which
appends one row per inference call; this script records the call order so the rows
can be joined back to candidates.  Run in the LIBERO env against a live server.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import LIBERO_DUMMY_ACTION, EpisodeConfig, _load_task
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import ACTION_KEY, FLOW_NOISE_KEY, FLOW_NOISE_SHAPE, \
    validate_action_response

from branch_snapshot import restore_full_state, save_full_state
from rollout_with_routes import sim_joint_layout

PROGRESS_JOINT = "wooden_cabinet_1_middle_level"


def restore(env, snap, mode="hard"):
    """Put the world back exactly, including everything MuJoCo's state vector omits.

    ``regenerate_obs_from_state`` restores only time/qpos/qvel/act.  Three other
    pieces of state survive it and make consecutive branches differ:

      OSC controller   goal_pos / goal_ori / ori_ref / the two interpolators.  The
                       controller is *not* re-derived from the simulator, so each
                       candidate inherits the goal its predecessor was servoing to.
                       Measured on this scene: 3.1e-03 divergence between two branches
                       that differ only in the action executed just before the fork.
      episode clock    robosuite's timestep/done live in Python; without them the
                       horizon keeps ticking across branches and a long forking
                       session dies mid-candidate with "executing action in
                       terminated episode".
      qacc_warmstart   the constraint solver warm start, in mjData rather than
                       mjSimState.  Scene-dependent: irrelevant while nothing is in
                       contact (task 0 phase 8: 8.6e-14 either way) but decisive once
                       the gripper touches something (task 2 phase 8: 4.23 without,
                       5.1e-15 with).

    branch_snapshot.restore_full_state handles all three.  It replaces an earlier
    version here that restored only the clock and deliberately left the warm start
    alone on the strength of a 5e-16 agreement measured on this scene -- that number
    holds where nothing is touching, which is why the gap went unnoticed.

    One thing it still cannot reach is the constraint solver's own scratch (efc_* and
    the contact-pair cache), which lives in mjData and is not part of the flattened
    state either.  ``hard`` clears it with sim.reset() before restoring; probe_fork_
    fidelity.py measures 0.0 drift after 32 branches that way against 5e-15 without,
    both far under the 5.8e-3..1.3e-2 spread between candidates, so this is insurance
    rather than a correction.
    """
    if mode == "hard":
        getattr(env, "env", env).sim.reset()
    return restore_full_state(env, snap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--client-dir", required=True,
                    help="capture whose recorded actions are replayed to reach each snapshot")
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--episodes", default="0,1,2,3,4")
    ap.add_argument("--fork-steps", default="6,10,11,12",
                    help="10-12 is the window where the routing decode beats the world "
                         "state; 6 is a pre-window control")
    ap.add_argument("--n-candidates", type=int, default=32,
                    help="the previous pilot ran 8, which puts the per-snapshot Spearman "
                         "SE at 1/sqrt(7)=0.38 and leaves only rho>~0.15 resolvable -- "
                         "not enough to tell 'no signal' from 'no power'")
    ap.add_argument("--restore-mode", choices=("soft", "hard"), default="hard",
                    help="hard adds sim.reset() before the restore, clearing the solver "
                         "scratch that set_state_from_flattened leaves behind; measured "
                         "at exactly 0.0 drift after 32 branches vs 5e-15 for soft")
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--continuation-steps", type=int, default=5,
                    help="control steps of policy continuation after each candidate "
                         "chunk, under noise shared across candidates")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = pathlib.Path(args.client_dir)
    summaries = {s["episode_index"]: s for s in json.loads((src / "summaries.json").read_text())}
    episodes = [int(v) for v in args.episodes.split(",")]
    fork_steps = [int(v) for v in args.fork_steps.split(",")]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, call_index = [], 0
    with PolicyClient(host=args.host, port=args.port, connect_timeout=600.0,
                      inference_timeout=300.0) as client:
        meta = client.metadata
        (out / "server_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

        for ep in episodes:
            summary = summaries[ep]
            with np.load(src / ("episode_%02d.npz" % ep)) as z:
                recorded = z["actions"]
            for fork_step in fork_steps:
                if fork_step >= summary["inference_calls"]:
                    print("ep %d: no control step %d, skipping" % (ep, fork_step))
                    continue
                config = EpisodeConfig(
                    task_suite=args.benchmark, task_id=int(summary["task_id"]),
                    init_state_id=int(summary["init_state_id"]), seed=int(summary["seed"]),
                    host=args.host, port=args.port, libero_root=args.libero_root,
                    output_root=str(out), settle_steps=10, max_steps=300,
                    replan_steps=args.replan_steps)
                env, observation, _task, prompt = _load_task(config)
                layout = sim_joint_layout(env)
                d_lo = [j for j in layout["joints"] if j["joint"] == PROGRESS_JOINT][0]["state_lo"]
                try:
                    for _ in range(config.settle_steps):
                        observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
                    for t in range(fork_step):
                        for a in recorded[t][: args.replan_steps]:
                            observation, _, _, _ = env.step(np.asarray(a, np.float64).tolist())

                    snap = save_full_state(env)
                    d0 = float(snap["sim"][d_lo])
                    # The query observation has to come from the restore, not from the
                    # live step that produced the snapshot.  env.step() returns images
                    # rendered one substep behind the state it reports, so the live and
                    # the restored observation of the same instant differ by 190 grey
                    # levels over 52.5% of the wrist image.  Every candidate is executed
                    # from the restore, so querying on the live observation would hand
                    # the snapshot a different input from all of its own branches.
                    policy_observation = build_policy_observation(
                        restore(env, snap, args.restore_mode), prompt)

                    # N candidates from the one observation; flow noise is the only input
                    # that differs, and the seed is derived from (ep, step, i) so the whole
                    # pilot is re-runnable
                    chunks, rows = [], []
                    for i in range(args.n_candidates):
                        rng = np.random.default_rng((ep * 1000 + fork_step) * 100 + i)
                        request = dict(policy_observation)
                        request[FLOW_NOISE_KEY] = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
                        response = validate_action_response(client.infer(request))
                        chunks.append(np.asarray(response[ACTION_KEY], np.float32))
                        rows.append(call_index)
                        call_index += 1

                    # Execute each candidate from the identical restored state, then let
                    # the policy carry on for a few more control steps.  The chunk alone
                    # is not a usable label: before the pull starts, all N candidates move
                    # the drawer by exactly zero.  The continuation noise is keyed on
                    # (ep, step, k) and not on the candidate, so every candidate faces the
                    # same future draws -- common random numbers, leaving the candidate
                    # chunk as the only difference.
                    # Execution order is shuffled, and the shuffle is seeded on the
                    # snapshot so it stays reproducible.  Whatever residual error a
                    # branch inherits from its predecessors grows with position in the
                    # sequence; running candidates in index order would make that error
                    # a monotone function of candidate identity, which is a bias in the
                    # ranking rather than noise around it.  Shuffling converts it.
                    order = np.random.default_rng(
                        (ep * 1000 + fork_step) * 100 + 77).permutation(len(chunks))
                    drift_probe = []
                    for pos, i in enumerate(order):
                        chunk = chunks[i]
                        restore(env, snap, args.restore_mode)
                        obs_i = None
                        for a in chunk[: args.replan_steps]:
                            obs_i, _, _, _ = env.step(np.asarray(a, np.float64).tolist())
                        state_after_chunk = np.asarray(env.get_sim_state(), np.float64)
                        after_chunk = float(state_after_chunk[d_lo])
                        eef_after_chunk = np.asarray(obs_i["robot0_eef_pos"], np.float64)
                        gripper_after_chunk = np.asarray(
                            obs_i.get("robot0_gripper_qpos", np.zeros(2)), np.float64)
                        success = bool(env.check_success())
                        n_extra = 0
                        for k in range(args.continuation_steps):
                            if success:
                                break
                            po = build_policy_observation(obs_i, prompt)
                            crn = np.random.default_rng(
                                (ep * 1000 + fork_step) * 100 + 90 + k)
                            req = dict(po)
                            req[FLOW_NOISE_KEY] = crn.standard_normal(
                                FLOW_NOISE_SHAPE).astype(np.float32)
                            resp = validate_action_response(client.infer(req))
                            call_index += 1          # the server records these rows too
                            for a in np.asarray(resp[ACTION_KEY])[: args.replan_steps]:
                                obs_i, _, _, _ = env.step(np.asarray(a, np.float64).tolist())
                                n_extra += 1
                                success = bool(env.check_success())
                                if success:
                                    break
                        end = np.asarray(env.get_sim_state(), np.float64)
                        nq = int(layout["nq"])
                        records.append({
                            "episode": ep, "fork_step": fork_step, "candidate": int(i),
                            "exec_position": int(pos),
                            "trace_row": rows[i],
                            "drawer_before": d0,
                            "drawer_after_chunk": after_chunk,
                            "drawer_delta_chunk": after_chunk - d0,
                            "drawer_after": float(end[d_lo]),
                            "drawer_delta": float(end[d_lo]) - d0,
                            "success_in_window": success,
                            "extra_action_steps": n_extra,
                            "episode_success": bool(summary["success"]),
                            # The screen so far compared candidates on a single 79-dim
                            # distance that mixes metres with metres per second, which is
                            # not interpretable as either.  Keep the pieces apart, and
                            # keep the readings that a position distance cannot show:
                            # whether the gripper closed, and whether anything was moved.
                            "qpos_after_chunk": state_after_chunk[1:1 + nq].tolist(),
                            "qvel_after_chunk": state_after_chunk[1 + nq:].tolist(),
                            "eef_after_chunk": eef_after_chunk.tolist(),
                            "gripper_after_chunk": gripper_after_chunk.tolist(),
                            "object_qpos_moved": float(np.abs(
                                state_after_chunk[1:1 + nq] - snap["sim"][1:1 + nq]).max()),
                        })

                    # Standing noise-floor check: candidate ``order[0]`` again, after
                    # every other branch has run.  Recorded rather than asserted -- what
                    # matters is not that the drift is zero but that it is far below the
                    # spread between candidates, since that spread is the signal.
                    first = int(order[0])
                    restore(env, snap, args.restore_mode)
                    for a in chunks[first][: args.replan_steps]:
                        env.step(np.asarray(a, np.float64).tolist())
                    recheck = float(np.asarray(env.get_sim_state(), np.float64)[d_lo])
                    block = records[-args.n_candidates:]
                    drift = abs(recheck - [r for r in block if r["candidate"] == first][0]
                                ["drawer_after_chunk"])
                    deltas = [r["drawer_delta"] for r in block]
                    chunk_deltas = [r["drawer_delta_chunk"] for r in block]
                    spread = max(deltas) - min(deltas)
                    chunk_spread = max(chunk_deltas) - min(chunk_deltas)
                    qp = np.array([r["qpos_after_chunk"] for r in block])
                    pair = np.linalg.norm(qp[:, None, :] - qp[None, :, :], axis=-1)
                    iu = np.triu_indices(len(qp), k=1)
                    for r in block:
                        r["rerun_drift"] = drift
                        r["candidate_spread"] = spread
                        # mean and max together: a unimodal cloud shows up in the mean, a
                        # split into two behaviours only in the max, and the split is the
                        # case where there is actually something to select
                        r["qpos_spread_mean"] = float(pair[iu].mean())
                        r["qpos_spread_max"] = float(pair[iu].max())
                    if chunk_spread > 0 and drift > 0.2 * chunk_spread:
                        raise RuntimeError(
                            "ep %d step %d: rerun drift %.3e is not small against the "
                            "between-candidate spread %.3e -- candidates cannot be "
                            "ranked here" % (ep, fork_step, drift, chunk_spread))
                    print("ep %-2d step %-3d  drawer %+.4f  cand %+.4f..%+.4f  spread %.2e "
                          "(chunk %.2e)  qpos mean/max %.2e/%.2e  noise %.2e  succ %d/%d"
                          % (ep, fork_step, d0, min(deltas), max(deltas), spread,
                             chunk_spread, block[0]["qpos_spread_mean"],
                             block[0]["qpos_spread_max"], drift,
                             sum(r["success_in_window"] for r in block), args.n_candidates))
                    np.save(out / ("chunks_ep%02d_t%02d.npy" % (ep, fork_step)), np.stack(chunks))
                    # only the numeric part is worth persisting; the controller and
                    # clock state is not portable, and the fork is re-derivable anyway
                    # by replaying the recorded prefix
                    np.save(out / ("snap_ep%02d_t%02d.npy" % (ep, fork_step)), snap["sim"])
                finally:
                    try:
                        env.close()
                    except BaseException:
                        pass

    (out / "fork_records.json").write_text(json.dumps(records, indent=2))
    print("\n%d candidates over %d snapshots; trace rows 0..%d"
          % (len(records), len(records) // args.n_candidates, call_index - 1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
