#!/usr/bin/env python3
"""Detect with routing, then *choose* with routing: one rollout, one rescue attempt.

The difference from triggered_fork_collect.py
---------------------------------------------
That script forked K full continuations at the alarm and counted how many happened to
succeed.  It answers "is this state rescuable", but it is not something you could deploy:
at run time you get to continue exactly one trajectory, and nothing tells you which of the
K to keep.

This script closes that loop.  At the alarm step it samples K candidate action chunks --
K policy calls against the *same* frozen state, each on its own flow-noise stream, no
simulation -- scores each candidate by how much its routing moved away from the previous
chunk, executes the single best one, and lets the rollout run to termination.

    trunk runs, routing scored online  ->  alarm at t*
    K candidate chunks at t*, routing captured for each   (K policy calls, ~1 s each)
    pick argmax of routing change      ->  execute that chunk, continue to the horizon

Why the largest routing change
------------------------------
The failure signature this detector fires on is a *collapse* in chunk-to-chunk routing
change: the arm stops, the observation stops moving, and the router keeps returning the
same experts.  So among candidates drawn at that moment, the one whose routing moves
furthest from the stuck state is the one proposing to do something different.  Picking it
is a directed intervention rather than a reroll.

Control arms
------------
Selection is only meaningful against alternatives, so each run replays the same saved state
three times:

    max     take the candidate with the largest routing change   (the hypothesis)
    random  take a uniformly drawn candidate                     (does *selection* matter,
                                                                  or is resampling enough?)
    min     take the smallest routing change                     (is the direction right?)

If max and random rescue equally often, routing is not choosing anything -- resampling
alone is doing the work.  If min is worse than max, the direction of the score is real.

Cost is about three continuations from the alarm, roughly 3 x 20 queries, a few minutes.

Usage
-----
    python3 routing_rescue_collect.py --host H --port P --benchmark libero_10 \
        --task-id 8 --init-state-id 7 --seed 20260830 --k 8 \
        --libero-root ... --out .../rescue_i7
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    ROUTING_CAPTURE_KEY,
    ROUTING_EXPERT_IDS_KEY,
    validate_action_response,
)

from branch_snapshot import restore_full_state, save_full_state
from rolling_star_collect import (
    EPISODE_ID_KEY,
    _array_sha256,
    _atomic_json,
    _atomic_npz,
    _frame,
    _write_video,
    branch_noise,
    trunk_noise,
)
from triggered_fork_collect import DENOISE, DEEP_LAYERS, TOPK, W, W0, K_CONSEC, THETA
from triggered_fork_collect import Detector, jaccard_distance

SCHEMA = "himoe.routing_rescue.v1"
ARMS = ("max", "random", "min")


def request(client, observation, noise, episode_id, routes=True):
    request_body = dict(observation)
    request_body[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, dtype=np.float32)
    request_body[EPISODE_ID_KEY] = int(episode_id)
    if routes:
        request_body[ROUTING_CAPTURE_KEY] = True
    response = validate_action_response(client.infer(request_body))
    if response.get(FLOW_NOISE_SHA256_KEY) != _array_sha256(noise):
        raise RuntimeError("server did not acknowledge the exact flow-noise tensor")
    actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
    ids = response.get(ROUTING_EXPERT_IDS_KEY)
    if routes and ids is None:
        raise RuntimeError("policy response carried no routing; start the server with "
                           "route capture enabled")
    return actions, (None if ids is None else np.asarray(ids))


def episode_id_for(worker: int, role: int, index: int) -> int:
    value = (worker + 1) * 100_000_000 + 70_000_000 + role * 1000 + index
    if not -(2 ** 31) <= value < 2 ** 31:
        raise ValueError("episode id does not fit int32")
    return value


def continue_rollout(environment, observation, prompt, client, args, build_obs,
                     first_chunk, start_action_steps, start_query, arm_index,
                     capture_frames=True):
    """Execute a chosen chunk, then run on to success or the horizon."""
    action_steps = int(start_action_steps)
    success = bool(environment.check_success())
    frames: List[np.ndarray] = []
    queries = 0
    chunk = first_chunk

    while action_steps < args.max_steps and not success:
        take = min(args.replan_steps, len(chunk), args.max_steps - action_steps)
        for index in range(take):
            observation, _r, _d, _i = environment.step(chunk[index].tolist())
            action_steps += 1
            if capture_frames:
                frames.append(_frame(observation))
            success = bool(environment.check_success())
            if success:
                break
        queries += 1
        if success or action_steps >= args.max_steps:
            break
        policy_observation = build_obs(observation, prompt)
        chunk, _ = request(
            client, policy_observation,
            branch_noise(args.seed, args.task_id, args.init_state_id,
                         start_query, arm_index, start_query + queries),
            # role 2 + arm_index: each arm needs its own episode id, or all three
            # continuations land on the same id in the server's route store and the
            # per-arm routing cannot be recovered afterwards.
            episode_id_for(args.worker_id, 2 + arm_index, 0),
            routes=False,
        )
    return dict(success=success, queries=queries, action_steps=action_steps), frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--init-state-id", type=int, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--k", type=int, default=8,
                        help="candidate chunks sampled at the alarm")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--max-trunk-queries", type=int, default=52)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    args = parser.parse_args()
    args.arms = ARMS

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    environment, observation, task, prompt = _load_task(EpisodeConfig(
        task_suite=args.benchmark, task_id=args.task_id,
        init_state_id=args.init_state_id, seed=args.environment_seed,
        host=args.host, port=args.port, libero_root=args.libero_root,
        output_root=str(out), settle_steps=args.settle_steps,
        max_steps=args.max_steps, replan_steps=args.replan_steps,
        inference_timeout=args.inference_timeout,
    ))
    for _ in range(args.settle_steps):
        observation, _r, _d, _i = environment.step(LIBERO_DUMMY_ACTION.tolist())

    from himoe_libero_bridge.preprocess import build_policy_observation

    with PolicyClient(host=args.host, port=args.port, connect_timeout=600.0,
                      inference_timeout=args.inference_timeout) as client:
        validate_policy_suite(client.metadata, args.benchmark)

        detector = Detector()
        action_steps = 0
        trunk_success = False
        alarm_state = None
        trunk_frames: List[np.ndarray] = []

        print("trunk", flush=True)
        for query in range(args.max_trunk_queries):
            if action_steps >= args.max_steps or trunk_success:
                break
            policy_observation = build_policy_observation(observation, prompt)
            snapshot = save_full_state(environment)
            # keep the routing of the chunk *before* this one: the detector's distance is
            # chunk-to-chunk, so candidates must be scored against the same reference.
            previous_ids_here = detector.previous

            actions, ids = request(
                client, policy_observation,
                trunk_noise(args.seed, args.task_id, args.init_state_id, query),
                episode_id_for(args.worker_id, 0, 0),
            )
            fired = detector.push(query, ids)
            ratio = detector.ratios[-1]
            print("  q%02d  r=%s%s" % (query, "  n/a" if ratio is None else "%.3f" % ratio,
                                       "   ALARM" if fired else ""), flush=True)
            if fired and alarm_state is None:
                alarm_state = dict(snapshot=snapshot, observation=dict(policy_observation),
                                   action_steps=action_steps, query=query,
                                   previous_ids=previous_ids_here)

            take = min(args.replan_steps, len(actions), args.max_steps - action_steps)
            for index in range(take):
                observation, _r, _d, _i = environment.step(actions[index].tolist())
                action_steps += 1
                trunk_frames.append(_frame(observation))
                trunk_success = bool(environment.check_success())
                if trunk_success:
                    break

        trunk_record = dict(success=trunk_success, queries=len(detector.ratios),
                            action_steps=action_steps, fired_at=detector.fired_at,
                            ratios=[None if r is None else round(r, 6)
                                    for r in detector.ratios])
        _atomic_json(out / "trunk.json", trunk_record)
        if trunk_frames:
            _write_video(out / ("trunk_%s.mp4" % ("success" if trunk_success else "failure")),
                         trunk_frames)
        print("trunk: success=%s queries=%d alarm=%s"
              % (trunk_success, len(detector.ratios), detector.fired_at), flush=True)

        if alarm_state is None or trunk_success:
            _atomic_json(out / "manifest.json", dict(
                schema=SCHEMA, status="no_alarm" if alarm_state is None else "trunk_succeeded",
                trunk=trunk_record, wall_s=round(time.time() - started, 1)))
            print("nothing to rescue", flush=True)
            return

        # ---- sample K candidate chunks at the alarm, score each by routing change ----
        print("\nsampling %d candidate chunks at q%d" % (args.k, alarm_state["query"]),
              flush=True)
        previous_ids = alarm_state["previous_ids"]
        candidates = []
        for k in range(args.k):
            restore_full_state(environment, alarm_state["snapshot"])
            noise = branch_noise(args.seed, args.task_id, args.init_state_id,
                                 alarm_state["query"], k, alarm_state["query"])
            chunk, ids = request(client, alarm_state["observation"], noise,
                                 episode_id_for(args.worker_id, 1, k))
            score = jaccard_distance(previous_ids, ids)
            candidates.append(dict(k=k, score=score, chunk=chunk))
            print("  candidate %d  routing change %.4f" % (k, score), flush=True)

        scores = np.array([c["score"] for c in candidates])
        trunk_change = float(detector.distances[-1]) if detector.distances else float("nan")
        # The random arm must not collide with max or min, or the run silently tests two
        # points instead of three -- which is exactly what happened on the first draw.
        hi, lo = int(np.argmax(scores)), int(np.argmin(scores))
        middle = [i for i in range(args.k) if i not in (hi, lo)]
        rng = np.random.default_rng(args.seed)
        picks = {
            "max": hi,
            "min": lo,
            "random": int(rng.choice(middle)) if middle else hi,
        }
        print("  trunk's own change at the alarm: %.4f" % trunk_change, flush=True)
        print("  picks: " + "  ".join("%s=c%d(%.4f)" % (a, picks[a], scores[picks[a]])
                                      for a in ARMS), flush=True)

        # ---- execute each pick from the same state, run to termination ----
        results = {}
        for arm_index, arm in enumerate(ARMS):
            restore_full_state(environment, alarm_state["snapshot"])
            record, frames = continue_rollout(
                environment, None, prompt, client, args, build_policy_observation,
                candidates[picks[arm]]["chunk"], alarm_state["action_steps"],
                alarm_state["query"], arm_index)
            record["candidate"] = picks[arm]
            record["routing_change"] = float(scores[picks[arm]])
            results[arm] = record
            if frames:
                _write_video(out / ("%s_c%02d_%s.mp4" % (
                    arm, picks[arm], "success" if record["success"] else "failure")), frames)
            print("  arm %-6s candidate c%d -> %s in %d queries"
                  % (arm, picks[arm], "SUCCESS" if record["success"] else "failure",
                     record["queries"]), flush=True)

        _atomic_json(out / "manifest.json", dict(
            schema=SCHEMA, status="complete",
            detector=dict(layers="HB 12-15", denoise=DENOISE, topk=TOPK,
                          W=W, W0=W0, K=K_CONSEC, theta=THETA),
            task=dict(benchmark=args.benchmark, task_id=args.task_id,
                      init_state_id=args.init_state_id, prompt=prompt, seed=args.seed),
            trunk=trunk_record, alarm_query=alarm_state["query"], k=args.k,
            candidate_scores=[round(float(s), 6) for s in scores],
            trunk_change_at_alarm=round(trunk_change, 6),
            picks=picks, arms=results,
            wall_s=round(time.time() - started, 1)))

        print("\n" + "=" * 60)
        print("trunk on its original chunk : FAILURE")
        for arm in ARMS:
            r = results[arm]
            print("pick %-6s (c%d, change %.4f) : %s"
                  % (arm, r["candidate"], r["routing_change"],
                     "SUCCESS in %d queries" % r["queries"] if r["success"] else "failure"))
        print("=" * 60)


if __name__ == "__main__":
    main()
