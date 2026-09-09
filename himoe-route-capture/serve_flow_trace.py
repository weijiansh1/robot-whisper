"""Serve HiMoE and capture the flow trajectory alongside the routing field.

The Routing Lead Test needs, for every candidate, both halves of the comparison at
every Euler step: the provisional action chunk x^(tau) and the HB routing R^(tau).
The routing half already exists (HiMoERouteRecorder carries a denoise axis); the
action half does not, because `sample_actions` keeps x_t as a local and mutates it
in place.

`MoEVLA.denoise_step` receives x_t as an argument, so wrapping it captures the
provisional chunk at entry, before the Euler update that follows the call:

    x^(0) = the injected noise      (entry of call 0)
    x^(tau) for tau = 1..T-1        (entry of call tau)
    x^(T) = the returned chunk

giving T+1 states from T calls plus the return value.  The clone is mandatory:
`x_t += dt * v_t` in `sample_actions` mutates the same tensor, so a stored
reference would end up holding the final chunk T times over.

x^(tau) is returned in the model's own 24-dim normalised action space, NOT the
7-dim unnormalised LIBERO action.  Basin clustering has to happen in one
consistent space, and the model's space is the one the routing was produced in.

Candidates are stamped so the flat zarr can be sliced back apart: the request's
`flow/query_id` becomes the record's episode_id and `flow/candidate_id` becomes its
control_step.  Both are advertised in the metadata, because every other server
hands the observation straight to the policy and an unexpected key would reach it.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")

FLUSH_EVERY = 64

QUERY_KEY = "flow/query_id"
CANDIDATE_KEY = "flow/candidate_id"
TRAJ_KEY = "flow/x_traj"


class FlowTracer:
    """Records x_t and v_t for every denoise_step call, per policy query.

    Capturing only the entry x_t loses the LAST Euler update: T calls give
    x^(0)..x^(T-1), and x^(T) -- the chunk the policy actually returns -- happens
    after the final call returns.  Using x^(T-1) as "the final chunk" makes the
    tau=T-1 comparison a self-comparison, which shows up as a rho of exactly
    +1.000.  So v_t is captured too and the last state is reconstructed:

        x^(tau+1) = x^(tau) + dt * v^(tau),     dt = -1 / num_steps

    Wrapping `sample_actions` instead would be the obvious fix and does NOT work:
    `policy.py:41` binds `self._sample_actions = self.model.sample_actions` at
    construction, so a later replacement is never called.  `denoise_step` is
    resolved dynamically inside `sample_actions`, which is why this hook lands.
    """

    def __init__(self, model):
        self.model = model
        self.states: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.enabled = False
        self._inner = model.denoise_step
        self.dt = -1.0 / float(model.config.num_steps)

        def traced(*args, **kwargs):
            if not self.enabled:
                return self._inner(*args, **kwargs)
            # x_t is the 6th positional parameter after self; accept either form
            x_t = kwargs["x_t"] if "x_t" in kwargs else args[5]
            # copy=True is mandatory: sample_actions does `x_t += dt * v_t`, so
            # every call receives the same tensor object and a stored reference
            # would end up holding the final chunk T times over
            self.states.append(x_t.detach().to("cpu", copy=True).float().numpy())
            v_t = self._inner(*args, **kwargs)
            self.velocities.append(v_t.detach().to("cpu", copy=True).float().numpy())
            return v_t

        model.denoise_step = traced

    def begin(self):
        self.states = []
        self.velocities = []
        self.enabled = True

    def cancel(self):
        """Disable capture and discard any incomplete trajectory."""

        self.enabled = False
        self.states = []
        self.velocities = []

    def trajectory(self) -> np.ndarray:
        """[T+1, n_action_steps, max_action_dim], x^(0) through x^(T)."""
        if not self.enabled or not self.states or not self.velocities:
            raise RuntimeError("no active flow trajectory")
        states = np.stack(self.states)
        final = states[-1] + self.dt * np.stack(self.velocities)[-1]
        return np.concatenate([states, final[None]], axis=0)

    def euler_residual(self) -> float:
        """Max |x^(tau+1) - (x^(tau) + dt v^(tau))| over the captured steps.

        Zero confirms the hook sees the same x_t and v_t that sample_actions
        combines, which is what licenses extrapolating the final state the same
        way.  Non-zero means the hook is on the wrong tensor.
        """
        if not self.enabled or not self.states or not self.velocities:
            raise RuntimeError("no active flow trajectory")
        states, velocities = np.stack(self.states), np.stack(self.velocities)
        if len(states) == 1:
            return 0.0
        predicted = states[:-1] + self.dt * velocities[:-1]
        return float(np.abs(predicted - states[1:]).max())

    def finish(self) -> tuple[np.ndarray, float]:
        """Return one batch-first D+1 trajectory and disable capture."""

        try:
            residual = self.euler_residual()
            trajectory = np.moveaxis(self.trajectory(), 1, 0)
        finally:
            self.cancel()
        return trajectory, residual

    def close(self):
        self.cancel()
        self.model.denoise_step = self._inner


def _flush(out, trajectories, keys, parts):
    """Write one part file and clear the buffers, so a kill costs at most FLUSH_EVERY."""
    if not trajectories:
        return
    np.savez_compressed(
        out / ("flow_traces_part%04d.npz" % parts[0]),
        x_traj=np.stack(trajectories).astype(np.float32),
        query_id=np.asarray([k[0] for k in keys], np.int32),
        candidate_id=np.asarray([k[1] for k in keys], np.int32),
    )
    parts[0] += 1
    trajectories.clear()
    keys.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--suite", default="goal")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--libero-wrist-layout", default="checkpoint-right")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch threads; only used by --gpu cpu")
    args = ap.parse_args()

    from himoe_libero_bridge.server import PolicyServer, create_policy

    if args.gpu == "cpu":
        # create_policy sets CUDA_VISIBLE_DEVICES and requires a device, so the CPU
        # path builds the policy directly.  policy.py:71 opens
        # torch.autocast(device_type='cuda'), a no-op without CUDA, and the AS gate
        # then hits fp32 weights with a bf16 data_mask -- hence the shim.
        # NOTE this is not the deployed numerical path: bf16 tie-breaking in the HB
        # top-4 is kernel-specific, so per-candidate routing will not match a GPU run.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        from probe_flow_lead_cpu import install_cpu_autocast

        install_cpu_autocast(torch)
        if args.threads:
            torch.set_num_threads(args.threads)
        from himoe_libero_bridge.policies import HiMoEPolicy

        policy = HiMoEPolicy(
            checkpoint_dir=args.checkpoint_dir, suite=args.suite,
            upstream_root=args.upstream_root, require_cuda=False,
            libero_wrist_layout=args.libero_wrist_layout,
        )
        print("running on CPU with %d threads" % torch.get_num_threads(), flush=True)
    else:
        policy = create_policy(
            "himoe", args.checkpoint_dir, args.upstream_root, args.gpu,
            args.suite, args.libero_wrist_layout,
        )

    from himoe_route_store import ZarrRouteWriter
    from himoe_router_recorder import HiMoERouteRecorder, discover_gates

    core = policy._policy.model
    gates = discover_gates(core)
    cfg = core.config
    n_suffix = int(cfg.n_action_steps) + 1
    n_denoise = int(cfg.num_steps)
    print("gates=%d  n_suffix=%d  n_denoise=%d" % (len(gates), n_suffix, n_denoise), flush=True)

    # full 32-dim softmax is mandatory here: the audit quantities M_4 and D_45 are
    # not recoverable from the gate's renormalised top-4 weights
    recorder = HiMoERouteRecorder(core, store_full_probs=True).attach()
    tracer = FlowTracer(core)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = ZarrRouteWriter(str(out / "routes.zarr"), n_denoise=n_denoise,
                             n_suffix=n_suffix, store_full_probs=True, chunk_steps=32)

    inner = policy.infer
    stats = {"calls": 0, "infer_s": 0.0}
    trajectories: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []
    parts = [0]

    def infer(observation):
        query = int(observation.pop(QUERY_KEY, 0))
        candidate = int(observation.pop(CANDIDATE_KEY, stats["calls"]))
        recorder.begin_control_step(episode_id=query, control_step=stats["calls"])
        tracer.begin()
        start = time.perf_counter()
        response = inner(observation)
        stats["infer_s"] += time.perf_counter() - start
        record = recorder.end_control_step()
        writer.append(record)
        # the policy returns post-processed 7-dim actions; the trajectory is kept in
        # the model's own 24-dim normalised space, which is where the routing lives
        # The residual is never exactly zero: denoise_step returns v_t in bfloat16,
        # so it lands on multiples of |dt| * ulp_bf16(v) -- about 1e-3 for the
        # observed |v| <= 6.4.  That is three orders below the inter-candidate
        # distances this capture exists to measure.  A hook on the wrong tensor
        # would instead give a residual of order |x| itself, hence the loose bound:
        # this guards against mis-wiring, not against rounding.
        residual = tracer.euler_residual()
        if residual > 0.05:
            raise RuntimeError(
                "Euler residual %.3e: too large to be bf16 velocity rounding, so the "
                "tracer is not on the tensors sample_actions integrates" % residual)
        trajectories.append(tracer.trajectory())
        keys.append((query, candidate))
        # flush in parts: holding everything until shutdown means a SIGKILL --
        # e.g. the container's 64 GiB cap being hit by a concurrent job -- loses
        # the whole run, and this capture is ~85 min of CPU
        if len(trajectories) >= FLUSH_EVERY:
            _flush(out, trajectories, keys, parts)
        stats["calls"] += 1
        if stats["calls"] % 64 == 0:
            print("captured %d queries (%.0f ms each)"
                  % (stats["calls"], 1000 * stats["infer_s"] / stats["calls"]), flush=True)
        return response

    policy.infer = infer
    policy.metadata["flow_tracer"] = "serve_flow_trace"
    policy.metadata["flow_query_key"] = QUERY_KEY
    policy.metadata["flow_candidate_key"] = CANDIDATE_KEY
    policy.metadata["route_axis_n_denoise"] = n_denoise
    policy.metadata["route_axis_n_suffix"] = n_suffix
    policy.metadata["n_action_steps"] = int(cfg.n_action_steps)
    policy.metadata["num_steps"] = n_denoise

    done = {"shutdown": False}

    def shutdown():
        if done["shutdown"]:
            return
        done["shutdown"] = True
        writer.close()
        recorder.close()
        tracer.close()
        _flush(out, trajectories, keys, parts)
        (out / "capture_summary.json").write_text(json.dumps({
            "queries": stats["calls"],
            "mean_infer_ms": 1000 * stats["infer_s"] / max(1, stats["calls"]),
            "n_suffix": n_suffix, "n_denoise": n_denoise,
            "flow_states_per_query": n_denoise,
        }, indent=2))
        print("\ncaptured %d queries" % stats["calls"], flush=True)

    import signal

    def _bye(*_a):
        try:
            shutdown()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    print("serving on ws://%s:%d" % (args.host, args.port), flush=True)
    try:
        PolicyServer(policy, args.host, args.port,
                     getattr(policy, "backend_name", "himoe")).serve_forever()
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
