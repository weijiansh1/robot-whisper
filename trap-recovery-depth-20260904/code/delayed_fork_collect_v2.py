#!/usr/bin/env python3
"""Run one rollout, wait for the routing detector to fire, then resample the noise there.

The question
------------
Rolling-star already shows that two branches restored from a bit-identical simulator state
can end differently when only the flow-noise stream changes.  But every one of its 22 fork
points sits at trunk query 0-5 — about 2.5 s into a 26 s episode — while the detector does
not fire until q18 at the earliest, q28 median.  Nothing in that corpus forks at the moment
of trouble, so "the detector fired, we resampled, it recovered" has never been tested.

This script tests it.  One rollout, one fork, at the step the detector picks:

    trunk runs, routing scored online   ->  alarm at t*  ->  save_full_state
    trunk continues on its ORIGINAL noise to termination     (the control, free)
    restore t*, fan out K branches on FRESH noise            (the intervention)

The trunk's own continuation is the control that matters here: same state, same moment,
the only difference is whether the noise was resampled.  A rescue is K branches containing
a success while the trunk fails.

Why it is cheap
---------------
Rolling-star spent ~44 min per fork point because forking at q0 leaves each of 16 branches
49 queries to run.  Forking at q28 leaves 24.  Measured wall cost on that run is 3.39 s per
query (0.86 s inference plus sim, state I/O and route capture), so:

    trunk to the alarm            28 queries   ~1.6 min
    trunk continues (control)     24 queries   ~1.4 min
    K=8 branches from the alarm  8x24 queries  ~11 min
                                              ~14 min total

The detector is frozen, not tuned here
--------------------------------------
The policy response carries `routing/expert_ids` shaped (10 denoise, 8 HB layers, 10 action
steps, top-4) when `routing/capture` is set, so the trigger is computable in the client with
no server change.  That fixes the configuration to **top-4 Jaccard over the action tokens**,
which is exactly the variant that transfers across corpora at one threshold: 77.3% correct
on rolling-star and 80.9% on the HUB SCENE8 corpus, with false-alarm rates matching within
2.6 points.

    d(t)  = 1 - |top4(t-1) & top4(t)| / |top4(t-1) | top4(t)|,  averaged over
            HB layers 12-15 x 10 action steps at denoise 9
    r(t)  = mean(last W of d) / mean(first W0 of d)
    alarm = K consecutive steps with r(t) < THETA

Dividing by the rollout's own opening is what makes THETA portable: r is dimensionless, so
no per-scene calibration is needed.  Identity is used only to match a site against itself
one step earlier — renumbering all 32 experts leaves every d(t) unchanged.

Usage
-----
    python3 triggered_fork_collect.py \
        --host HOST --port PORT --benchmark libero_10 --task-id 8 \
        --init-state-id 0 --worker-id 0 --k 8 \
        --libero-root /path/to/LIBERO --out runs/triggered-fork-0

The server must be running with route capture enabled, the same way rolling-star ran it;
otherwise the response carries no expert ids and this script stops immediately rather than
silently falling back to a different trigger.

`--control-arm` additionally forks at a random earlier query.  That is what separates "the
detector picked a good moment" from "resampling works anywhere", and it roughly doubles the
cost.  It is off by default because the trunk continuation already answers the narrow
question, and one group is one group.
"""

from __future__ import annotations

import argparse
import json
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
from rollout_with_routes import sim_joint_layout
from rolling_star_collect import (
    EPISODE_ID_KEY,
    _array_sha256,
    _atomic_json,
    _atomic_npz,
    _run_branch,
    _write_video,
    trunk_noise,
)

SCHEMA = "himoe.triggered_fork.v1"

# Frozen before this run.  Do not tune against the rollout you are about to fork.
DEEP_LAYERS = slice(4, 8)      # HB 12-15
DENOISE = 9                    # last flow step
TOPK = 4                       # what the protocol exposes
W, W0, K_CONSEC, THETA = 4, 8, 3, 0.95


def jaccard_distance(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean top-4 set-overlap distance over HB 12-15 x 10 action steps.

    Both arrays are (denoise, layer, action_step, top-4); only the last denoise round is
    read.  Unchanged under any fixed renumbering of the 32 experts, so this measures
    whether the same experts are still carrying the token, never which ones they are.
    """
    a, b = previous[DENOISE, DEEP_LAYERS], current[DENOISE, DEEP_LAYERS]
    scores = []
    for layer in range(a.shape[0]):
        for step in range(a.shape[1]):
            sa, sb = set(a[layer, step].tolist()), set(b[layer, step].tolist())
            scores.append(1.0 - len(sa & sb) / max(len(sa | sb), 1))
    return float(np.mean(scores))


class Detector:
    """Online alarm; fed one expert-id trace per trunk query."""

    def __init__(self) -> None:
        self.distances: List[float] = []
        self.ratios: List[Optional[float]] = []
        self.previous: Optional[np.ndarray] = None
        self.run = 0
        self.fired_at: Optional[int] = None

    def push(self, query: int, ids: np.ndarray) -> bool:
        if self.previous is not None:
            self.distances.append(jaccard_distance(self.previous, ids))
        self.previous = ids

        ratio = None
        if len(self.distances) >= max(W, W0):
            base = float(np.mean(self.distances[:W0]))
            if base > 0:
                ratio = float(np.mean(self.distances[-W:]) / base)
        self.ratios.append(ratio)

        # Require 2*W0 history so the ratio is never read while its own baseline is still
        # inside the trailing window.
        if ratio is not None and ratio < THETA and len(self.distances) >= 2 * W0:
            self.run += 1
        else:
            self.run = 0
        if self.run >= K_CONSEC and self.fired_at is None:
            self.fired_at = query
            return True
        return False


def request_with_routes(
    client: PolicyClient,
    policy_observation: Mapping[str, Any],
    noise: np.ndarray,
    episode_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """One policy call that keeps the routing trace.

    `rolling_star_collect._request` validates the same way but discards everything except
    the actions, so the trunk needs its own call here.  The flow-noise acknowledgement is
    checked identically — a server that does not echo the exact tensor invalidates the
    whole premise that branches differ only through noise.
    """
    request = dict(policy_observation)
    request[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, dtype=np.float32)
    request[EPISODE_ID_KEY] = int(episode_id)
    request[ROUTING_CAPTURE_KEY] = True
    response = validate_action_response(client.infer(request))
    if response.get(FLOW_NOISE_SHA256_KEY) != _array_sha256(noise):
        raise RuntimeError("server did not acknowledge the exact flow-noise tensor")
    if ROUTING_EXPERT_IDS_KEY not in response:
        raise RuntimeError(
            "policy response carried no %s; start the server with route capture enabled "
            "— without it the trigger cannot be computed online" % ROUTING_EXPERT_IDS_KEY
        )
    actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
    return actions, np.asarray(response[ROUTING_EXPERT_IDS_KEY])


_BRANCH_ROUTES: List[np.ndarray] = []


def _request_capturing_routes(client, policy_observation, noise, episode_id):
    """Drop-in for ``rolling_star_collect._request`` that also keeps the routing ids.

    The stock one validates identically but discards everything except the actions, so a
    branch could not be scored by the same detector that fired on its trunk.  Ids land in
    the module-level buffer; ``fan_out`` drains it per candidate.
    """
    request = dict(policy_observation)
    request[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, dtype=np.float32)
    request[EPISODE_ID_KEY] = int(episode_id)
    request[ROUTING_CAPTURE_KEY] = True
    response = validate_action_response(client.infer(request))
    if response.get(FLOW_NOISE_SHA256_KEY) != _array_sha256(noise):
        raise RuntimeError("server did not acknowledge the exact flow-noise tensor")
    actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError("unexpected action shape %s" % (actions.shape,))
    ids = response.get(ROUTING_EXPERT_IDS_KEY)
    _BRANCH_ROUTES.append(None if ids is None else np.asarray(ids, dtype=np.uint8))
    return actions, float(response.get("server/inference_ms", np.nan))


import rolling_star_collect as _rsc  # noqa: E402
_rsc._request = _request_capturing_routes


def episode_id_for(worker: int, role: int, index: int) -> int:
    """role 0 = trunk, 1 = triggered fork, 2 = control fork.  Disjoint from rolling-star."""
    value = (worker + 1) * 100_000_000 + 80_000_000 + role * 1000 + index
    if not -(2 ** 31) <= value < 2 ** 31:
        raise ValueError("episode id does not fit int32")
    return value


def fan_out(
    environment: Any,
    snapshot: Mapping[str, Any],
    policy_observation: Mapping[str, Any],
    prompt: str,
    client: PolicyClient,
    args: argparse.Namespace,
    fork_query: int,
    action_steps_at_fork: int,
    role: int,
    directory: pathlib.Path,
) -> List[Dict[str, Any]]:
    """Restore the saved state K times, each on its own noise stream, run to termination."""
    directory.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    for candidate in range(args.k):
        restore_full_state(environment, snapshot)
        _BRANCH_ROUTES.clear()
        summary, arrays, frames = _run_branch(
            environment, policy_observation, prompt, client, args.seed, args.task_id,
            args.init_state_id, fork_query, candidate,
            episode_id_for(args.worker_id, role, candidate),
            action_steps_at_fork,
            (action_steps_at_fork + args.fork_budget_queries * args.replan_steps
             if args.fork_budget_queries else args.max_steps),
            args.replan_steps, True,
        )
        if _BRANCH_ROUTES and all(x is not None for x in _BRANCH_ROUTES):
            arrays = dict(arrays)
            arrays["routing_expert_ids"] = np.stack(_BRANCH_ROUTES)
        _atomic_npz(directory / ("candidate_%02d.npz" % candidate), arrays)
        summary["branch_routes_captured"] = "routing_expert_ids" in arrays
        summary["fork_query"] = fork_query
        summary["role"] = role
        summary["fork_budget_queries"] = (args.fork_budget_queries or
                                          (args.max_steps - action_steps_at_fork)
                                          // args.replan_steps)
        summary["within_original_cap"] = bool(
            summary["success"]
            and fork_query + summary["inference_calls"] <= args.max_trunk_queries)
        _atomic_json(directory / ("candidate_%02d.json" % candidate), summary)
        if frames:
            outcome = "success" if summary["success"] else "failure"
            _write_video(directory / ("c%02d_%s.mp4" % (candidate, outcome)), frames)
        summaries.append(summary)
        print("  fork q%d candidate=%d success=%s queries=%d"
              % (fork_query, candidate, summary["success"], summary["inference_calls"]),
              flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--init-state-id", type=int, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--max-trunk-queries", type=int, default=52)
    parser.add_argument("--control-arm", action="store_true")
    parser.add_argument(
        "--delay-offsets", default="4,8,16",
        help="fork these many queries AFTER the alarm as well, comma separated; "
             "empty string disables. The alarm itself is offset 0.")
    parser.add_argument(
        "--fork-budget-queries", type=int, default=0,
        help="give every branch this many queries from its own fork point, so the "
             "arms have equal opportunity. 0 = run to the original cap (unequal).")
    parser.add_argument(
        "--failures-only", action="store_true",
        help="skip forking when the trunk succeeded on its own noise")
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _atomic_json(out / "experiment_config.json", dict(
        schema=SCHEMA, benchmark=args.benchmark, task_id=args.task_id,
        init_state_id=args.init_state_id, worker_id=args.worker_id, k=args.k,
        seed=args.seed, environment_seed=args.environment_seed,
        settle_steps=args.settle_steps, max_steps=args.max_steps,
        replan_steps=args.replan_steps, control_arm=bool(args.control_arm),
        detector=dict(layers="HB 12-15", denoise=DENOISE, topk=TOPK,
                      W=W, W0=W0, K=K_CONSEC, theta=THETA),
    ))

    environment, observation, task, prompt = _load_task(EpisodeConfig(
        task_suite=args.benchmark, task_id=args.task_id,
        init_state_id=args.init_state_id, seed=args.environment_seed,
        host=args.host, port=args.port, libero_root=args.libero_root,
        output_root=str(out), settle_steps=args.settle_steps,
        max_steps=args.max_steps, replan_steps=args.replan_steps,
        inference_timeout=args.inference_timeout,
    ))
    for _ in range(args.settle_steps):
        observation, _reward, _done, _info = environment.step(LIBERO_DUMMY_ACTION.tolist())
    _atomic_json(out / "sim_layout.json", sim_joint_layout(environment))

    with PolicyClient(host=args.host, port=args.port, connect_timeout=600.0,
                      inference_timeout=args.inference_timeout) as client:
        validate_policy_suite(client.metadata, args.benchmark)
        _atomic_json(out / "server_metadata.json", dict(client.metadata))

        from himoe_libero_bridge.preprocess import build_policy_observation

        detector = Detector()
        action_steps = 0
        trunk_success = False
        fork: Optional[Tuple[Any, Dict[str, Any], int, int]] = None
        # One saved state per query, so the control arm can rewind without replaying.
        rewind: Dict[int, Tuple[Any, Dict[str, Any], int]] = {}

        print("trunk", flush=True)
        for query in range(args.max_trunk_queries):
            if action_steps >= args.max_steps or trunk_success:
                break
            policy_observation = build_policy_observation(observation, prompt)
            rewind[query] = (save_full_state(environment), dict(policy_observation),
                             action_steps)

            actions, ids = request_with_routes(
                client, policy_observation,
                trunk_noise(args.seed, args.task_id, args.init_state_id, query),
                episode_id_for(args.worker_id, 0, 0),
            )
            fired = detector.push(query, ids)
            ratio = detector.ratios[-1]
            print("  q%02d  r=%s%s" % (
                query, "  n/a" if ratio is None else "%.3f" % ratio,
                "   ALARM" if fired else ""), flush=True)
            if fired:
                snapshot, obs, steps = rewind[query]
                fork = (snapshot, obs, steps, query)

            take = min(args.replan_steps, len(actions), args.max_steps - action_steps)
            for index in range(take):
                observation, _r, _d, _i = environment.step(actions[index].tolist())
                action_steps += 1
                trunk_success = bool(environment.check_success())
                if trunk_success:
                    break

        trunk_record = dict(
            success=trunk_success, queries=len(detector.ratios),
            action_steps=action_steps, fired_at=detector.fired_at,
            distances=[round(x, 6) for x in detector.distances],
            ratios=[None if x is None else round(x, 6) for x in detector.ratios],
        )
        _atomic_json(out / "trunk.json", trunk_record)
        print("trunk: success=%s queries=%d alarm=%s"
              % (trunk_success, len(detector.ratios), detector.fired_at), flush=True)

        if fork is None:
            _atomic_json(out / "manifest.json", dict(
                schema=SCHEMA, status="no_alarm", trunk=trunk_record,
                wall_s=round(time.time() - started, 1),
                note="detector never fired on this rollout; nothing to intervene on"))
            print("no alarm — nothing to fork", flush=True)
            return

        if args.failures_only and trunk_success:
            _atomic_json(out / "manifest.json", dict(
                schema=SCHEMA, status="trunk_succeeded", trunk=trunk_record,
                wall_s=round(time.time() - started, 1),
                note="trunk succeeded on its own noise; nothing to rescue"))
            print("trunk succeeded — skipping forks (--failures-only)", flush=True)
            return

        snapshot, fork_observation, fork_action_steps, fork_query = fork
        print("triggered fork at q%d (trunk %s)"
              % (fork_query, "succeeded" if trunk_success else "failed"), flush=True)
        triggered = fan_out(environment, snapshot, fork_observation, prompt, client,
                            args, fork_query, fork_action_steps, 1, out / "triggered")

        delays = [int(x) for x in args.delay_offsets.split(",") if x.strip()]
        delayed: Dict[str, Any] = {}
        for slot, delta in enumerate(sorted(set(d for d in delays if d > 0))):
            q = fork_query + delta
            if q not in rewind:
                print("delayed +%d: q%d not available" % (delta, q), flush=True)
                continue
            d_snapshot, d_obs, d_steps = rewind[q]
            print("delayed fork +%d at q%d (budget %d)"
                  % (delta, q, args.max_trunk_queries - q), flush=True)
            runs = fan_out(environment, d_snapshot, d_obs, prompt, client, args,
                           q, d_steps, 3 + slot, out / ("delayed_+%d" % delta))
            delayed["+%d" % delta] = dict(
                delta=delta, query=q,
                residual_budget=args.max_trunk_queries - q,
                n=len(runs), successes=sum(1 for r in runs if r["success"]))

        control: List[Dict[str, Any]] = []
        control_query = None
        if args.control_arm:
            eligible = [q for q in sorted(rewind) if 2 * W0 <= q < fork_query]
            if eligible:
                control_query = int(np.random.default_rng(args.seed).choice(eligible))
                c_snapshot, c_obs, c_steps = rewind[control_query]
                print("control fork at q%d" % control_query, flush=True)
                control = fan_out(environment, c_snapshot, c_obs, prompt, client, args,
                                  control_query, c_steps, 2, out / "control")
            else:
                print("no eligible control query before the alarm", flush=True)

        rescued = sum(1 for s in triggered if s["success"])
        _atomic_json(out / "manifest.json", dict(
            schema=SCHEMA, status="complete",
            detector=dict(layers="HB 12-15", denoise=DENOISE, topk=TOPK,
                          W=W, W0=W0, K=K_CONSEC, theta=THETA,
                          note="frozen before this run; see module docstring"),
            task=dict(benchmark=args.benchmark, task_id=args.task_id,
                      init_state_id=args.init_state_id, prompt=prompt),
            trunk=trunk_record, fork_query=fork_query, control_query=control_query,
            k=args.k,
            triggered=dict(n=len(triggered), successes=rescued,
                           residual_budget=args.max_trunk_queries - fork_query),
            delayed=delayed,
            control=dict(n=len(control), successes=sum(1 for s in control if s["success"])),
            wall_s=round(time.time() - started, 1),
        ))

        print("\n" + "=" * 62)
        print("trunk on its original noise  : %s"
              % ("SUCCESS" if trunk_success else "FAILURE"))
        print("resampled at the alarm (q%d) : %d/%d succeeded"
              % (fork_query, rescued, args.k))
        for key in sorted(delayed, key=lambda k: int(k)):
            d = delayed[key]
            print("resampled %s after alarm (q%-2d): %d/%d succeeded  budget=%d"
                  % (key, d["query"], d["successes"], d["n"], d["residual_budget"]))
        if control:
            print("resampled at q%-2d (control)   : %d/%d succeeded"
                  % (control_query, sum(1 for s in control if s["success"]), len(control)))
        print("=" * 62)
        if trunk_success:
            print("trunk succeeded — this rollout cannot answer the question.")
        elif rescued:
            print("RESCUE: same state, same moment, only the noise differs.")
        else:
            print("no rescue: resampling at the alarm did not recover this state.")


if __name__ == "__main__":
    main()
