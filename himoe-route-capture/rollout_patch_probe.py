"""Transplant one control step's HB routing between rollouts and see what moves.

Every behavioural ablation so far perturbs all ~20 control steps at once, which
closed-loop replanning absorbs, and does it by re-selecting among 32 experts whose
router is within 0.2% of maximum entropy on the action tokens.  Neither tests the
one place the routing state is known to carry outcome information beyond the full
physical world state: control steps 11-12.

Protocol, for a branch step K:

  phase A   run each recipient seed untouched, recording its control-step-K HB
            routing into a named slot, and note the outcome
  phase B   re-run the same seed with the same flow noise, overriding control step
            K only, under four arms:

              self            the recipient's own slot -- an identity check, and
                              the run is invalid if it does not reproduce phase A
              donor_success   the slot of a seed that succeeded
              donor_fail      the slot of a different seed that failed
              random          uniformly random top-4, router weights kept

Three outcomes are each informative.  Outcomes flip under donor_success and not
under donor_fail: routing is causal at the window.  Actions move but outcomes do
not: routing carries information without control authority.  Actions barely move:
the routed branch's 5.6% share of the block output norm is the binding constraint,
and "routing is a mirror" is established with a magnitude rather than asserted.

Needs serve_patched_router.py.  Routing slots live in that server's memory, so the
donor comes from a live rollout in the same process -- re-running a stored capture
with identical seeds does not reproduce it (27/64 against an original 23/64).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)

RECORD_KEY = "patch/record"
APPLY_KEY = "patch/apply"
RANDOM_KEY = "patch/random"
RECORDED_SITES_KEY = "patch/recorded_sites"
APPLIED_SITES_KEY = "patch/applied_sites"

ARMS = ("self", "donor_success", "donor_fail", "random")


def run_one(config, client, flow_noise_seed, branch_control, record_slot=None,
            apply_slot=None, randomise=False):
    """One episode; the patch keys are attached to control step ``branch_control`` only."""
    environment, observation, task, prompt = _load_task(config)
    rng = np.random.default_rng(flow_noise_seed)
    action_seq, state_seq = [], []
    patch_sites = {"recorded": 0, "applied": 0}
    control = 0
    steps = 0
    success = False
    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())

        while steps < config.max_steps and not success:
            policy_observation = build_policy_observation(observation, prompt)
            request = dict(policy_observation)
            request[FLOW_NOISE_KEY] = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
            if control == branch_control:
                if record_slot is not None:
                    request[RECORD_KEY] = record_slot
                if apply_slot is not None:
                    request[APPLY_KEY] = apply_slot
                if randomise:
                    request[RANDOM_KEY] = True

            response = validate_action_response(client.infer(request))
            actions = response[ACTION_KEY]
            if control == branch_control:
                patch_sites["recorded"] = int(response.get(RECORDED_SITES_KEY, 0))
                patch_sites["applied"] = int(response.get(APPLIED_SITES_KEY, 0))

            action_seq.append(np.asarray(actions[: config.replan_steps], np.float32))
            state_seq.append(np.asarray(policy_observation["observation/state"], np.float32))

            for action in actions[: config.replan_steps]:
                if steps >= config.max_steps:
                    break
                observation, _reward, _done, _info = environment.step(action.tolist())
                steps += 1
                success = bool(environment.check_success())
                if success:
                    break
            control += 1
    finally:
        try:
            environment.close()
        except BaseException:
            pass

    reached = control > branch_control
    summary = {
        "flow_noise_seed": flow_noise_seed,
        "branch_control": branch_control,
        "success": success,
        "action_steps": steps,
        "inference_calls": control,
        "reached_branch": reached,
        "recorded_sites": patch_sites["recorded"],
        "applied_sites": patch_sites["applied"],
    }
    actions = np.stack(action_seq)
    return summary, actions, np.stack(state_seq)


def _branch_action(actions: np.ndarray, branch_control: int) -> np.ndarray | None:
    if branch_control >= len(actions):
        return None
    return actions[branch_control]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--branch-control", type=int, default=11)
    ap.add_argument("--n-recipients", type=int, default=16)
    ap.add_argument("--noise-seed-base", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--label", default="patch-probe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inference-timeout", type=float, default=300.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [args.noise_seed_base + i for i in range(args.n_recipients)]

    def episode_config(seed: int) -> EpisodeConfig:
        return EpisodeConfig(
            task_suite=args.benchmark,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
            seed=args.seed,
            host=args.host,
            port=args.port,
            libero_root=args.libero_root,
            output_root=str(out),
            settle_steps=args.settle_steps,
            max_steps=args.max_steps,
            replan_steps=args.replan_steps,
            inference_timeout=args.inference_timeout,
        )

    started = time.time()
    with PolicyClient(
        host=args.host, port=args.port, connect_timeout=600.0,
        inference_timeout=args.inference_timeout,
    ) as client:
        metadata = client.metadata
        if metadata.get("route_patcher") != "serve_patched_router":
            raise RuntimeError("this driver needs serve_patched_router.py")
        validate_policy_suite(metadata, args.benchmark)
        (out / "server_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        # -- phase A: untouched, recording the branch step -------------------
        # Every episode's arrays go to disk.  The first run of this probe kept only
        # the console log, so its action deltas could not be re-derived afterwards
        # and the "routing moves the action" claim had nothing behind it.
        def save(tag, summary, actions, states):
            np.savez_compressed(
                out / ("episode_%s.npz" % tag),
                actions=actions, states=states,
                summary=np.frombuffer(json.dumps(summary).encode(), dtype=np.uint8),
            )

        base: dict[int, dict] = {}
        for seed in seeds:
            summary, actions, states = run_one(
                episode_config(seed), client, seed, args.branch_control,
                record_slot="s%d" % seed,
            )
            save("none_%d" % seed, summary, actions, states)
            summary.update(arm="none", donor=None)
            base[seed] = {
                "summary": summary,
                "branch_action": _branch_action(actions, args.branch_control),
            }
            print("[A] seed %d  %s  steps=%d  recorded=%d"
                  % (seed, "OK " if summary["success"] else "FAIL",
                     summary["action_steps"], summary["recorded_sites"]), flush=True)

        usable = [s for s in seeds if base[s]["summary"]["reached_branch"]]
        winners = [s for s in usable if base[s]["summary"]["success"]]
        losers = [s for s in usable if not base[s]["summary"]["success"]]
        print("[A] %d/%d reached control step %d; %d succeeded, %d failed"
              % (len(usable), len(seeds), args.branch_control, len(winners), len(losers)),
              flush=True)
        if not winners or len(losers) < 2:
            raise RuntimeError(
                "need at least one successful and two failed donors at this branch step; "
                "got %d/%d -- pick an init state with more dynamic range"
                % (len(winners), len(losers))
            )

        # -- phase B: the four arms ------------------------------------------
        records = [dict(base[s]["summary"]) for s in seeds]
        for arm in args.arms:
            for position, seed in enumerate(usable):
                if arm == "self":
                    donor, randomise = seed, False
                elif arm == "donor_success":
                    # Must exclude the recipient: a donor_success cell whose donor is
                    # itself is a self-patch, contributes a guaranteed zero delta and
                    # zero flip, and silently dilutes the arm.
                    pool = [s for s in winners if s != seed]
                    donor, randomise = pool[position % len(pool)], False
                elif arm == "donor_fail":
                    pool = [s for s in losers if s != seed]
                    donor, randomise = pool[position % len(pool)], False
                else:
                    donor, randomise = None, True
                if donor is not None and arm != "self" and donor == seed:
                    raise AssertionError("donor must differ from the recipient in arm %s" % arm)

                summary, actions, states = run_one(
                    episode_config(seed), client, seed, args.branch_control,
                    apply_slot=None if donor is None else "s%d" % donor,
                    randomise=randomise,
                )
                summary.update(arm=arm, donor=donor)
                save("%s_%d" % (arm, seed), summary, actions, states)

                reference = base[seed]["branch_action"]
                patched = _branch_action(actions, args.branch_control)
                if reference is not None and patched is not None:
                    summary["branch_action_max_abs_delta"] = float(
                        np.abs(patched - reference).max()
                    )
                    summary["branch_action_rms_delta"] = float(
                        np.sqrt(((patched - reference) ** 2).mean())
                    )
                summary["outcome_flipped"] = bool(
                    summary["success"] != base[seed]["summary"]["success"]
                )
                records.append(summary)
                print("[B] %-13s seed %d donor %-5s  %s  applied=%d  dA=%.2e  %s"
                      % (arm, seed, donor, "OK " if summary["success"] else "FAIL",
                         summary["applied_sites"],
                         summary.get("branch_action_max_abs_delta", float("nan")),
                         "FLIP" if summary["outcome_flipped"] else ""), flush=True)

    (out / "summaries.json").write_text(json.dumps(records, indent=2))
    (out / "experiment_config.json").write_text(json.dumps({
        "protocol": "route-patch-probe-v1",
        "label": args.label,
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "branch_control": args.branch_control,
        "n_recipients": args.n_recipients,
        "noise_seed_base": args.noise_seed_base,
        "environment_seed": args.seed,
        "settle_steps": args.settle_steps,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "arms": list(args.arms),
    }, indent=2, sort_keys=True))

    identity = [r for r in records if r.get("arm") == "self"]
    exact = sum(1 for r in identity if r.get("branch_action_max_abs_delta") == 0.0)
    print("\nself-patch identity: %d/%d branch actions bit-exact" % (exact, len(identity)))
    if identity and exact != len(identity):
        print("WARNING: the self arm did not reproduce phase A; every other arm's "
              "delta is contaminated by that much run-to-run drift")
    print("%s: %d episodes in %.1f min" % (args.label, len(records), (time.time() - started) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
