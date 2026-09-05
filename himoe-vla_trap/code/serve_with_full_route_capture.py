"""Serve the real HiMoE checkpoint with HiMoERouteRecorder attached.

Wraps, rather than edits, the audited bridge: create_policy() builds the policy
exactly as the normal server does, then the recorder is attached to the live
MoEVLA module and policy.infer is bracketed with begin/end_control_step.

Run in the model env (python 3.11).  The legacy mode captures every request as
before; ``--request-gated-capture`` enables the candidate-only wire contract.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import uuid

import numpy as np

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = pathlib.Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "himoe-libero-wrist-fix/src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "himoe-route-capture"))

from himoe_candidate_capture_protocol import (  # noqa: E402
    ACK_KEY,
    CANDIDATE_ID_KEY,
    CAPTURE_KEY,
    FLUSH_AFTER_KEY,
    QUERY_ID_KEY,
    SCHEMA as CANDIDATE_CAPTURE_SCHEMA,
    SNAPSHOT_INDEX_KEY,
    VLM_FEATURE_KEY,
    CandidateCaptureCoordinator,
    CaptureIdentity,
    array_digest,
    capture_ack,
    pop_capture_directive,
    request_digests,
)


FULL_ROUTER_PROBS_RESPONSE_KEY = "recorder/hb_router_probs"


def capture_run_id_for_resume(out: pathlib.Path) -> str:
    """Read and cross-check the run identity before opening any store for append."""

    import zarr

    ids = []
    for name in ("routes.zarr", "hidden.zarr", "flow_trajectory.zarr"):
        path = out / name
        if not path.exists():
            raise FileNotFoundError("resume requires %s" % path)
        value = zarr.open_group(str(path), mode="r").attrs.get("capture_run_id")
        if not isinstance(value, str) or not value:
            raise ValueError("resume store %s has no capture_run_id" % path)
        ids.append(value)
    if len(set(ids)) != 1:
        raise ValueError("resume stores do not share one capture_run_id")
    return ids[0]


def candidate_capture_metadata(
    capture_run_id: str,
    *,
    row_count: int = 0,
    durable_through_row: int = -1,
    vlm_feature_dim: int | None = None,
) -> dict:
    """Metadata discovery contract advertised by request-gated servers."""

    row_count = int(row_count)
    durable_through_row = int(durable_through_row)
    if row_count < 0 or not -1 <= durable_through_row < row_count:
        raise ValueError("candidate capture metadata has an invalid initial boundary")
    request_keys = {
        "capture": CAPTURE_KEY,
        "query_id": QUERY_ID_KEY,
        "snapshot_index": SNAPSHOT_INDEX_KEY,
        "candidate_id": CANDIDATE_ID_KEY,
        "flush_after": FLUSH_AFTER_KEY,
    }
    metadata = {
        "candidate_capture_supported": True,
        "candidate_capture_schema": CANDIDATE_CAPTURE_SCHEMA,
        "candidate_capture_run_id": capture_run_id,
        "candidate_capture_initial_row_count": row_count,
        "candidate_capture_initial_durable_through_row": durable_through_row,
        "candidate_capture_request_keys": request_keys,
        "candidate_capture_ack_key": ACK_KEY,
        "candidate_capture_protocol": {
            "schema": CANDIDATE_CAPTURE_SCHEMA,
            "request_gated": True,
            "initial_state": {
                "row_count": row_count,
                "durable_through_row": durable_through_row,
            },
            "request_keys": request_keys,
            "response": {
                "ack_key": ACK_KEY,
                "fields": [
                    "schema",
                    "capture_run_id",
                    "query_id",
                    "captured",
                    "row",
                    "row_count",
                    "durable_through_row",
                ],
            },
            "stores": [
                "routes.zarr",
                "hidden.zarr",
                "flow_trajectory.zarr",
            ],
        },
    }
    if vlm_feature_dim is not None:
        if int(vlm_feature_dim) <= 0:
            raise ValueError("VLM feature dimension must be positive")
        metadata.update(
            {
                "candidate_vlm_feature_key": VLM_FEATURE_KEY,
                "candidate_vlm_feature_dim": int(vlm_feature_dim),
                "candidate_vlm_feature_pooling": (
                    "mask-weighted mean of frozen PaliGemma final-layer prefix tokens"
                ),
            }
        )
    return metadata


class RequestGatedCandidateRecorder:
    """Testable per-request transaction around one policy inference."""

    def __init__(
        self,
        inner,
        recorder,
        tracer,
        stores: CandidateCaptureCoordinator,
        *,
        vlm_tracer=None,
        start_call_ordinal: int = 0,
        euler_residual_limit: float = 0.05,
    ) -> None:
        self.inner = inner
        self.recorder = recorder
        self.tracer = tracer
        self.stores = stores
        self.vlm_tracer = vlm_tracer
        self.call_ordinal = int(start_call_ordinal)
        self.euler_residual_limit = float(euler_residual_limit)
        self.calls = 0
        self.captured = 0
        self.as_collapsed = 0
        self.infer_s = 0.0
        self.overhead_s = 0.0
        self.max_euler_residual = 0.0

    def _cancel_capture(self) -> None:
        self.recorder.enabled = False
        self.recorder._buf.clear()
        self.tracer.cancel()
        if self.vlm_tracer is not None:
            self.vlm_tracer.cancel()

    def infer(self, observation):
        from himoe_libero_bridge.protocol import (
            ACTION_KEY,
            FLOW_NOISE_SHA256_KEY,
            validate_action_response,
        )

        directive = pop_capture_directive(observation)
        if "episode_id" in observation:
            legacy_query_id = observation.pop("episode_id")
            if isinstance(legacy_query_id, (bool, np.bool_)):
                raise ValueError("episode_id must be an integer when supplied")
            if int(legacy_query_id) != directive.query_id:
                raise ValueError("episode_id and recorder/query_id disagree")

        ordinal = self.call_ordinal
        if ordinal > np.iinfo(np.int32).max:
            raise RuntimeError("local call ordinal no longer fits route store int32 identity")
        self.call_ordinal += 1
        self.calls += 1
        started = time.perf_counter()

        if not directive.capture:
            response = validate_action_response(self.inner(observation))
            self.infer_s += time.perf_counter() - started
            response[ACK_KEY] = capture_ack(
                capture_run_id=self.stores.capture_run_id,
                query_id=directive.query_id,
                captured=False,
                row=-1,
                row_count=self.stores.rows,
                durable_through_row=self.stores.durable_through_row,
            )
            return response

        observation_digest, noise_digest = request_digests(observation)
        self.recorder.begin_control_step(
            episode_id=directive.query_id,
            control_step=ordinal,
        )
        self.tracer.begin()
        if self.vlm_tracer is not None:
            self.vlm_tracer.begin()
        try:
            response = validate_action_response(self.inner(observation))
            inferred = time.perf_counter()
            record = self.recorder.end_control_step()
            trajectory, residual = self.tracer.finish()
            vlm_feature = None if self.vlm_tracer is None else self.vlm_tracer.finish()
            if residual > self.euler_residual_limit:
                raise RuntimeError(
                    "Euler residual %.3e exceeds %.3e; flow tracer is not on the "
                    "tensors integrated by sample_actions"
                    % (residual, self.euler_residual_limit)
                )
            response_noise_digest = response.get(FLOW_NOISE_SHA256_KEY)
            if response_noise_digest != noise_digest.hex():
                raise RuntimeError("policy did not acknowledge the captured flow/noise")

            row = self.stores.rows
            identity = CaptureIdentity(
                capture_row=row,
                query_id=directive.query_id,
                snapshot_index=directive.snapshot_index,
                candidate_id=directive.candidate_id,
                local_call_ordinal=ordinal,
                observation_sha256=observation_digest,
                flow_noise_sha256=noise_digest,
                actions_sha256=array_digest(response[ACTION_KEY]),
            )
            assigned = self.stores.append(record, identity, trajectory)
            if assigned != row:
                raise RuntimeError("capture coordinator changed the assigned row")
            if directive.flush_after:
                self.stores.flush()
        except Exception:
            self._cancel_capture()
            raise

        finished = time.perf_counter()
        self.infer_s += inferred - started
        self.overhead_s += finished - inferred
        self.captured += 1
        self.as_collapsed += int(record.as_collapsed)
        self.max_euler_residual = max(self.max_euler_residual, residual)
        if vlm_feature is not None:
            if vlm_feature.shape[0] != 1:
                raise RuntimeError("request-gated capture expects one VLM feature row")
            response[VLM_FEATURE_KEY] = vlm_feature[0]
        response[ACK_KEY] = capture_ack(
            capture_run_id=self.stores.capture_run_id,
            query_id=directive.query_id,
            captured=True,
            row=row,
            row_count=self.stores.rows,
            durable_through_row=self.stores.durable_through_row,
        )
        return response


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
    ap.add_argument("--store-full-probs", action="store_true")
    ap.add_argument(
        "--return-full-probs",
        action="store_true",
        help="legacy capture mode only: return the recorder's per-request HB "
             "probabilities as recorder/hb_router_probs with shape [8,10,S,32]; "
             "intended for online train-free diagnostics",
    )
    ap.add_argument("--store-hidden", action="store_true",
                    help="also write hidden.zarr: the 1024-dim tensors the routers read "
                         "(~65x the routing bytes; the only part not recoverable later)")
    ap.add_argument(
        "--request-gated-capture",
        action="store_true",
        help="capture only requests carrying recorder/capture=true; writes aligned "
             "full routes, hidden states and flow trajectories",
    )
    ap.add_argument(
        "--resume-capture",
        action="store_true",
        help="resume request-gated stores at their last shared durable row",
    )
    ap.add_argument(
        "--resume-capture-rows",
        type=int,
        default=None,
        help="with --resume-capture, truncate all stores to this artifact-confirmed "
             "captured-row prefix before serving",
    )
    ap.add_argument("--chunk-steps", type=int, default=32)
    ap.add_argument("--pin", default=None,
                    help="pin table from build_switch_pin.py; freezes the state "
                         "token's HB routing so the gripper gate cannot fire")
    ap.add_argument("--pin-regime", default=None, choices=["on", "off"],
                    help="which regime to freeze to; omit for the zero control, "
                         "which attaches the hooks and changes nothing")
    ap.add_argument("--trunc-rounds", type=int, default=None,
                    help="early-stop the flow loop: run r denoise rounds, jump "
                         "to t=0 with the r-th velocity (x_hat = x - t*v), skip "
                         "the remaining forwards; r == num_steps is exactly the "
                         "unpatched sampler (scale t/dt == 1 at the last round)")
    ap.add_argument("--no-route-capture", action="store_true",
                    help="serve without recorder/writer; with --trunc-rounds the "
                         "skipped rounds would leave partial denoise rows, so "
                         "routing capture must be off for truncated arms")
    ap.add_argument("--threads", type=int, default=32,
                    help="torch threads; only used by --gpu cpu")
    ap.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="exit through the normal SIGTERM flush path if this runner PID disappears",
    )
    ap.add_argument(
        "--low-memory-load",
        action="store_true",
        help="CPU only: mmap checkpoint into a meta-constructed model; default off",
    )
    args = ap.parse_args()

    if args.low_memory_load and args.gpu != "cpu":
        raise SystemExit("--low-memory-load is only supported with --gpu cpu")
    if args.resume_capture and not args.request_gated_capture:
        raise SystemExit("--resume-capture requires --request-gated-capture")
    if args.resume_capture_rows is not None and not args.resume_capture:
        raise SystemExit("--resume-capture-rows requires --resume-capture")
    if args.resume_capture_rows is not None and args.resume_capture_rows < 0:
        raise SystemExit("--resume-capture-rows must be non-negative")
    if args.parent_pid is not None and args.parent_pid <= 1:
        raise SystemExit("--parent-pid must identify a non-init process")
    if args.request_gated_capture and args.no_route_capture:
        raise SystemExit("--request-gated-capture cannot be combined with --no-route-capture")
    if args.return_full_probs and args.no_route_capture:
        raise SystemExit("--return-full-probs requires route capture")
    if args.return_full_probs and args.request_gated_capture:
        raise SystemExit("--return-full-probs currently supports legacy capture mode only")
    if args.return_full_probs:
        args.store_full_probs = True
    if args.request_gated_capture:
        # These are protocol guarantees, rather than optional storage knobs, in
        # candidate-only mode.  Keep the legacy defaults unchanged otherwise.
        args.store_full_probs = True
        args.store_hidden = True

    from himoe_libero_bridge.server import PolicyServer, create_policy

    if args.gpu == "cpu":
        # mirror serve_flow_trace.py's CPU path: create_policy requires a CUDA
        # device, so build the policy directly; policy.py opens
        # torch.autocast(device_type='cuda'), a no-op without CUDA, and the AS
        # gate then hits fp32 weights with a bf16 data_mask -- hence the shim.
        # NOTE bf16 tie-breaking is kernel-specific: CPU routing/actions will not
        # match a GPU run; CPU arms must carry their own CPU control.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch as _t

        from probe_flow_lead_cpu import install_cpu_autocast

        install_cpu_autocast(_t)
        _t.set_num_threads(args.threads)
        from himoe_libero_bridge.policies import HiMoEPolicy

        if args.low_memory_load:
            from himoe_libero_bridge.suites import get_suite
            from himoe_low_memory_load import low_memory_himoe_load

            with low_memory_himoe_load(
                args.checkpoint_dir,
                args.upstream_root,
                get_suite(args.suite).train_config,
            ) as load_audit:
                policy = HiMoEPolicy(
                    checkpoint_dir=args.checkpoint_dir, suite=args.suite,
                    upstream_root=args.upstream_root, require_cuda=False,
                    libero_wrist_layout=args.libero_wrist_layout,
                )
            expected_weights = str(
                (pathlib.Path(policy.metadata["checkpoint"]) / "pytorch_model.pth")
                .expanduser()
                .resolve()
            )
            if load_audit.checkpoint_path != expected_weights:
                raise RuntimeError("low-memory loader checkpoint identity mismatch")
            load_metadata = load_audit.as_metadata()
            load_metadata.update(
                {
                    "checkpoint_sha256": policy.metadata["checkpoint_sha256"],
                    "himoe_upstream_commit": policy.metadata["himoe_upstream_commit"],
                    "himoe_patch_sha256": policy.metadata["himoe_patch_sha256"],
                    "himoe_patches_verified_applied": policy.metadata[
                        "himoe_patches_verified_applied"
                    ],
                }
            )
            policy.metadata["checkpoint_load_audit"] = load_metadata
            print(
                "low-memory checkpoint load: %d tensors, %.2f GiB logical"
                % (
                    load_audit.state_key_count,
                    load_audit.logical_tensor_bytes / (1024 ** 3),
                ),
                flush=True,
            )
        else:
            policy = HiMoEPolicy(
                checkpoint_dir=args.checkpoint_dir, suite=args.suite,
                upstream_root=args.upstream_root, require_cuda=False,
                libero_wrist_layout=args.libero_wrist_layout,
            )
        print("running on CPU with %d threads" % _t.get_num_threads(), flush=True)
    else:
        policy = create_policy(
            "himoe", args.checkpoint_dir, args.upstream_root, args.gpu,
            args.suite, args.libero_wrist_layout,
        )

    # imported after create_policy so CUDA_VISIBLE_DEVICES is already set
    from himoe_flow_trajectory_store import ZarrFlowTrajectoryWriter
    from himoe_hidden_store import ZarrHiddenWriter
    from himoe_route_store import ZarrRouteWriter
    from himoe_router_recorder import HiMoERouteRecorder, discover_gates
    from serve_flow_trace import FlowTracer

    core = policy._policy.model
    gates = discover_gates(core)
    print("discovered %d gates" % len(gates), flush=True)

    #: attached BEFORE the recorder, so what lands in routes.zarr is the routing
    #: the model actually executed rather than the routing it would have chosen.
    #: The recorder's own top-k check compares against the gate's return value,
    #: which a pin deliberately contradicts, so it is disabled under a pin.
    pin_probe = None
    if args.pin:
        import torch as _torch
        from himoe_switch_intervene import StateTokenPin, load_pin

        def _resolve(root, dotted):
            node = root
            for part in dotted.split("."):
                node = getattr(node, part) if not part.isdigit() else node[int(part)]
            return node

        table = load_pin(args.pin)
        pin_probe = StateTokenPin(gates, _torch, table, args.pin_regime)
        pin_probe.attach(lambda n: _resolve(core, n))
        print("PIN: %s regime=%s layers=%s (from %s / %s)"
              % (pathlib.Path(args.pin).name, args.pin_regime,
                 sorted(table["layers"]), table["suite"], table["task"][:40]),
              flush=True)
    for g in gates:
        print("  %-58s %-3s experts=%-3d top_k=%d"
              % (g.name, g.kind, g.n_experts, g.top_k), flush=True)

    cfg = policy._policy.model.config
    n_suffix = int(cfg.n_action_steps) + 1  # state token + action tokens
    n_denoise = int(cfg.num_steps)
    print("n_suffix=%d  n_denoise=%d" % (n_suffix, n_denoise), flush=True)

    trunc_stats = {"jump": 0, "skip": 0, "full": 0}
    if args.trunc_rounds is not None:
        r = int(args.trunc_rounds)
        if not (1 <= r <= n_denoise):
            raise SystemExit("--trunc-rounds must be in 1..%d" % n_denoise)
        if not args.no_route_capture and r < n_denoise:
            raise SystemExit("truncated arms must run with --no-route-capture")
        import torch as _torch2
        _orig_denoise = core.denoise_step
        _dt = 1.0 / n_denoise

        def _trunc_denoise(state, ppm, pam, kv, dm, x_t, timestep):
            # round index from the timestep itself: t = 1 - m*dt at call m
            t = float(timestep.reshape(-1)[0])
            m = int(round((1.0 - t) / _dt))
            if m < r - 1:
                trunc_stats["full"] += 1
                return _orig_denoise(state, ppm, pam, kv, dm, x_t, timestep)
            if m == r - 1:
                trunc_stats["jump"] += 1
                v = _orig_denoise(state, ppm, pam, kv, dm, x_t, timestep)
                # one Euler step of -dt lands on x - t*v  (exact identity at r=n)
                return v * (t / _dt)
            trunc_stats["skip"] += 1
            return _torch2.zeros_like(x_t)

        core.denoise_step = _trunc_denoise
        print("trunc-rounds=%d attached (jump scale at t=%.1f)"
              % (r, 1.0 - _dt * (r - 1)), flush=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capture_run_id = None
    if args.request_gated_capture:
        capture_run_id = (
            capture_run_id_for_resume(out) if args.resume_capture else uuid.uuid4().hex
        )
    recorder = writer = None
    if not args.no_route_capture:
        recorder = HiMoERouteRecorder(
            core, store_full_probs=args.store_full_probs,
            store_hidden=args.store_hidden,
            # under a pin the gate's returned top-k is deliberately not the top-k
            # of its own scores, so the recorder's correctness check would fire on
            # every call; the check already ran clean on the un-pinned arms
            verify_steps=0 if args.pin else 4,
        ).attach()
        writer = ZarrRouteWriter(
            str(out / "routes.zarr"),
            n_denoise=n_denoise,
            n_suffix=n_suffix,
            store_full_probs=args.store_full_probs,
            chunk_steps=args.chunk_steps,
            overwrite=not args.resume_capture,
            resume=args.resume_capture,
            capture_run_id=capture_run_id,
            auto_flush=not args.request_gated_capture,
        )
    hidden_writer = None
    if args.store_hidden:
        # From the router weight itself, not from a config path: MoEGate does
        # F.linear(h, W) with W of shape [n_experts, hidden_dim], so this is by
        # construction the width of the tensor the hook captures.  (An earlier
        # version read core.config.gemma_expert_config.hidden_size, which does
        # not exist on MoEVLAConfig.)
        # The two families do NOT share a width: HB reads the 1024-dim hidden
        # state, AS reads the 24-dim data_mask.  Take each from its own routers
        # and assert the family is internally consistent.
        def _width(kind: str) -> int:
            w = {int(g.module.weight.shape[1]) for g in gates if g.kind == kind}
            if len(w) != 1:
                raise RuntimeError("%s routers disagree on input width: %s"
                                   % (kind, sorted(w)))
            return w.pop()
        hb_hidden_dim, as_hidden_dim = _width("HB"), _width("AS")
        print("store_hidden on: hb=%d as=%d" % (hb_hidden_dim, as_hidden_dim),
              flush=True)
        hidden_writer = ZarrHiddenWriter(
            str(out / "hidden.zarr"),
            n_denoise=n_denoise, n_suffix=n_suffix,
            hb_hidden_dim=hb_hidden_dim, as_hidden_dim=as_hidden_dim,
            chunk_steps=args.chunk_steps if args.request_gated_capture else 8,
            overwrite=not args.resume_capture,
            resume=args.resume_capture,
            capture_run_id=capture_run_id,
            auto_flush=not args.request_gated_capture,
        )

    stats = {"calls": 0, "as_collapsed": 0, "overhead_s": 0.0, "infer_s": 0.0}
    inner = policy.infer
    episode_id = {"value": 0}
    tracer = vlm_tracer = flow_writer = candidate_stores = capture_session = None
    if args.request_gated_capture:
        tracer = FlowTracer(core)
        from himoe_vlm_feature_tracer import VLMFeatureTracer

        vlm_tracer = VLMFeatureTracer(core)
        flow_writer = ZarrFlowTrajectoryWriter(
            str(out / "flow_trajectory.zarr"),
            n_denoise=n_denoise,
            n_action_steps=int(cfg.n_action_steps),
            max_action_dim=int(cfg.max_action_dim),
            chunk_queries=args.chunk_steps,
            overwrite=not args.resume_capture,
            resume=args.resume_capture,
            capture_run_id=capture_run_id,
            auto_flush=False,
        )
        candidate_stores = CandidateCaptureCoordinator(
            writer,
            hidden_writer,
            flow_writer,
            resume_rows=args.resume_capture_rows,
        )
        capture_session = RequestGatedCandidateRecorder(
            inner,
            recorder,
            tracer,
            candidate_stores,
            vlm_tracer=vlm_tracer,
            start_call_ordinal=candidate_stores.next_local_call_ordinal,
        )

        def infer(observation):
            response = capture_session.infer(observation)
            if capture_session.calls % 50 == 0:
                print(
                    "served %d queries, captured %d rows "
                    "(infer %.0f ms, capture %.0f ms)"
                    % (
                        capture_session.calls,
                        capture_session.captured,
                        1000 * capture_session.infer_s / max(1, capture_session.calls),
                        1000 * capture_session.overhead_s
                        / max(1, capture_session.captured),
                    ),
                    flush=True,
                )
            return response

    else:

        def infer(observation):
            # the LIBERO client restarts flow noise per episode; use the caller's
            # episode marker if present, else keep a running control-step counter
            ep = int(observation.pop("episode_id", episode_id["value"]))
            if recorder is not None:
                recorder.begin_control_step(episode_id=ep, control_step=stats["calls"])
            t0 = time.perf_counter()
            response = inner(observation)
            t1 = time.perf_counter()
            if recorder is not None:
                rec = recorder.end_control_step()
                writer.append(rec)
                if hidden_writer is not None:
                    hidden_writer.append(rec)
                if args.return_full_probs:
                    probabilities = np.asarray(rec.hb_router_probs)
                    if probabilities.shape[0] != 1:
                        raise RuntimeError(
                            "--return-full-probs requires batch size one, got %d"
                            % probabilities.shape[0]
                        )
                    response[FULL_ROUTER_PROBS_RESPONSE_KEY] = np.ascontiguousarray(
                        probabilities[0], dtype=np.float16
                    )
                stats["as_collapsed"] += int(rec.as_collapsed)
            t2 = time.perf_counter()
            stats["calls"] += 1
            stats["infer_s"] += t1 - t0
            stats["overhead_s"] += t2 - t1
            if stats["calls"] % 50 == 0:
                print("captured %d control steps  (infer %.0f ms, capture %.0f ms)"
                      % (stats["calls"],
                         1000 * stats["infer_s"] / stats["calls"],
                         1000 * stats["overhead_s"] / stats["calls"]), flush=True)
            return response

    policy.infer = infer
    policy.metadata["route_recorder"] = "himoe_router_recorder"
    policy.metadata["route_recorder_gates"] = len(gates)
    # Advertised so a client can tell whether stamping episode_id on the request
    # will actually be consumed here.  Clients must not send it blind: every
    # other server hands the observation straight to the policy.
    policy.metadata["episode_id_key"] = "episode_id"
    # The denoise axis and the suffix-token axis are two different config fields
    # that happen to be adjacent in the stored array.  On this checkpoint they are
    # 10 and 11; on the class default (n_action_steps=50) they would be 10 and 51.
    # Record both so the analysis side can assert instead of infer from shape.
    policy.metadata["n_action_steps"] = int(cfg.n_action_steps)
    policy.metadata["num_steps"] = int(cfg.num_steps)
    policy.metadata["route_axis_n_denoise"] = n_denoise
    policy.metadata["route_axis_n_suffix"] = n_suffix
    if args.return_full_probs:
        policy.metadata.update(
            {
                "full_router_probs_response_key": FULL_ROUTER_PROBS_RESPONSE_KEY,
                "full_router_probs_response_shape": [8, n_denoise, n_suffix, 32],
                "full_router_probs_response_dtype": "float16",
            }
        )
    if args.request_gated_capture:
        policy.metadata.update(
            candidate_capture_metadata(
                candidate_stores.capture_run_id,
                row_count=candidate_stores.rows,
                durable_through_row=candidate_stores.durable_through_row,
                vlm_feature_dim=vlm_tracer.feature_dim,
            )
        )

    done = {"shutdown": False}

    def shutdown():
        # idempotent: the signal handler and the serve_forever finally-block can
        # both reach here, and flushing a closed writer would raise
        if done["shutdown"]:
            return
        done["shutdown"] = True
        if candidate_stores is not None:
            candidate_stores.close()
        else:
            if writer is not None:
                writer.close()
            # must close before recorder.close(): a dropped final chunk here is up
            # to chunk_steps control steps of hidden state, and unlike routes.zarr
            # there is no way to notice from the data itself -- the row count would
            # simply be short of capture_summary.control_steps.
            if hidden_writer is not None:
                hidden_writer.close()
        if tracer is not None:
            tracer.close()
        if vlm_tracer is not None:
            vlm_tracer.close()
        if recorder is not None:
            recorder.close()
        session_calls = stats["calls"] if capture_session is None else capture_session.calls
        captured_rows = (
            stats["calls"] if capture_session is None else candidate_stores.rows
        )
        as_collapsed = (
            stats["as_collapsed"]
            if capture_session is None
            else capture_session.as_collapsed
        )
        infer_s = stats["infer_s"] if capture_session is None else capture_session.infer_s
        overhead_s = (
            stats["overhead_s"]
            if capture_session is None
            else capture_session.overhead_s
        )
        session_captured = (
            stats["calls"]
            if capture_session is None
            else capture_session.captured
        )
        summary = {
            "control_steps": captured_rows,
            "as_collapsed": as_collapsed,
            "mean_infer_ms": 1000 * infer_s / max(1, session_calls),
            "mean_capture_ms": 1000 * overhead_s / max(1, session_captured),
            "gates": [{"name": g.name, "layer": g.layer_idx, "kind": g.kind,
                       "experts": g.n_experts, "top_k": g.top_k} for g in gates],
            "n_suffix": n_suffix,
            "n_denoise": n_denoise,
            "store_full_probs": bool(args.store_full_probs),
            "return_full_probs": bool(args.return_full_probs),
            "store_hidden": bool(args.store_hidden),
            "hook_verified_calls": (0 if recorder is None else recorder.verified_calls),
            "hook_verify_failures": ([] if recorder is None else recorder.verify_failures),
            "trunc_rounds": args.trunc_rounds,
            "trunc_calls": (dict(trunc_stats) if args.trunc_rounds is not None else None),
            "route_capture": not args.no_route_capture,
            "pin": (None if pin_probe is None else pin_probe.report()),
            "checkpoint_load_audit": policy.metadata.get("checkpoint_load_audit"),
        }
        if capture_session is not None:
            summary.update(
                {
                    "request_gated_capture": True,
                    "capture_run_id": capture_run_id,
                    "session_queries": session_calls,
                    "session_captured": session_captured,
                    "captured_rows": captured_rows,
                    "durable_through_row": candidate_stores.durable_through_row,
                    "max_euler_residual": capture_session.max_euler_residual,
                    "resume_capture": bool(args.resume_capture),
                    "resume_capture_rows": args.resume_capture_rows,
                }
            )
        (out / "capture_summary.json").write_text(json.dumps(summary, indent=2))
        print("\n=== capture summary ===")
        print(json.dumps({k: v for k, v in summary.items() if k != "gates"}, indent=2))

    # NOT atexit: zarr's async executor is already torn down by interpreter exit
    # ("cannot schedule new futures after shutdown"), so the final partial chunk
    # would be silently lost.  Flush from the signal handler, while the loop is
    # still alive, then hard-exit.
    import signal

    def _bye(*_a):
        try:
            shutdown()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    if args.parent_pid is not None:
        import threading

        def _watch_parent():
            while True:
                time.sleep(1.0)
                try:
                    os.kill(args.parent_pid, 0)
                except ProcessLookupError:
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                except PermissionError:
                    # A live process outside our permission boundary still counts
                    # as present; only ESRCH licenses shutting down the model.
                    continue

        threading.Thread(
            target=_watch_parent,
            name="himoe-parent-watch",
            daemon=True,
        ).start()

    print("serving on ws://%s:%d" % (args.host, args.port), flush=True)
    # SIGTERM is the normal stop and hard-exits inside the handler, so this
    # finally-block is for the abnormal paths: an exception out of serve_forever
    # would otherwise drop the last unflushed chunk and never write
    # capture_summary.json, leaving a run that looks complete but is short.
    try:
        PolicyServer(policy, args.host, args.port,
                     getattr(policy, "backend_name", "himoe")).serve_forever()
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
