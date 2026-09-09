"""Serve HiMoE with a route-only adaptive flow-denoising stopper.

Each policy query is independent.  After every suffix pass, the sampler reduces
the eight HB-MoE action-token routes to eight normalized 32-expert probability
vectors.  It can stop after the configured number of consecutive small
RMS-Hellinger changes and returns the current action block immediately.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-route-capture")

from adaptive_denoise_stopper import ConsecutiveRouteStopper, StopConfig


ENABLED_KEY = "adaptive/enabled"
ARM_KEY = "adaptive/arm"
EPISODE_KEY = "adaptive/episode_id"
STEPS_KEY = "adaptive/steps"
STOPPED_KEY = "adaptive/stopped_early"
CHANGES_KEY = "adaptive/route_changes"
FIRST_ELIGIBLE_KEY = "adaptive/first_eligible_step"
SAMPLE_MS_KEY = "adaptive/sample_ms"
ROUTE_METRICS = ("probability_hellinger", "dedup_jaccard", "dedup_count")


def storage_compatible_action_route(probability: Any, n_action: int) -> Any:
    """Match the route representation used to calibrate the threshold.

    ``ZarrRouteWriter`` stores full router probabilities as float16.  The
    offline metric then restores float32, renormalizes every token, averages
    action tokens, and renormalizes once more.  Repeating those operations here
    is necessary because changes near 0.005 are sensitive to float16 rounding.
    """

    probability = probability.half().float()
    probability = probability / probability.sum(dim=-1, keepdim=True)
    probability = probability[:, -n_action:, :].mean(dim=1)
    return probability / probability.sum(dim=-1, keepdim=True)


def deduplicated_jaccard_change(previous: Any, current: Any) -> float:
    """Mean per-layer Jaccard distance between active-expert sets."""

    if previous.shape != current.shape or previous.dtype != current.dtype:
        raise ValueError("expert-set tensors must have aligned shapes and dtypes")
    if previous.dtype != previous.new_empty((), dtype=previous.dtype).bool().dtype:
        raise ValueError("expert-set tensors must be boolean")
    intersection = (previous & current).sum(dim=-1).float()
    union = (previous | current).sum(dim=-1).float()
    if bool((union <= 0).any()):
        raise ValueError("expert sets must be non-empty")
    return float((1.0 - intersection / union).mean().detach().cpu())


def deduplicated_count_change(previous: Any, current: Any) -> float:
    """Mean absolute change in the number of distinct experts per layer."""

    if previous.shape != current.shape or previous.dtype != current.dtype:
        raise ValueError("expert-set tensors must have aligned shapes and dtypes")
    if previous.dtype != previous.new_empty((), dtype=previous.dtype).bool().dtype:
        raise ValueError("expert-set tensors must be boolean")
    difference = (previous.sum(dim=-1) - current.sum(dim=-1)).abs().float()
    return float(difference.mean().detach().cpu())


class AdaptiveRouteSampler:
    """A source-equivalent sampler with an optional route-only early return."""

    def __init__(
        self,
        model: Any,
        hb_gates: list[Any],
        config: StopConfig,
        route_metric: str = "probability_hellinger",
    ) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.hb_gates = list(hb_gates)
        self.config = config
        if route_metric not in ROUTE_METRICS:
            raise ValueError("unknown route metric: %s" % route_metric)
        self.route_metric = route_metric
        self.enabled = False
        self.collecting = False
        self._round_routes: list[Any] = []
        self._handles = [
            gate.module.register_forward_hook(self._make_hook()) for gate in self.hb_gates
        ]
        self.last_record: dict[str, Any] | None = None

    def _make_hook(self):
        from himoe_router_recorder import _router_logits

        def hook(module, args, output):
            if not self.collecting:
                return None
            gate_input = args[0]
            batch, sequence, _hidden = gate_input.shape
            n_action = int(self.model.config.n_action_steps)
            if sequence < n_action:
                raise RuntimeError(
                    "HB route has %d tokens, fewer than %d action tokens"
                    % (sequence, n_action)
                )
            with self.torch.no_grad():
                if self.route_metric == "probability_hellinger":
                    route = _router_logits(module, gate_input, "HB").softmax(dim=-1)
                    route = route.reshape(batch, sequence, -1)
                    route = storage_compatible_action_route(route, n_action)
                else:
                    topk_index = output[0].reshape(batch, sequence, -1)
                    topk_index = topk_index[:, -n_action:, :].reshape(batch, -1)
                    route = self.torch.zeros(
                        (batch, int(module.n_routed_experts)),
                        dtype=self.torch.bool,
                        device=topk_index.device,
                    )
                    route.scatter_(1, topk_index, True)
            self._round_routes.append(route)
            return None

        return hook

    def begin(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.collecting = False
        self._round_routes = []
        self.last_record = None

    def suspend(self) -> None:
        self.enabled = False
        self.collecting = False
        self._round_routes = []
        self.last_record = None

    def _finish_round(self) -> Any:
        if len(self._round_routes) != len(self.hb_gates):
            raise RuntimeError(
                "expected %d HB routes in one denoise pass, captured %d"
                % (len(self.hb_gates), len(self._round_routes))
            )
        # [batch, layers, experts]
        route = self.torch.stack(self._round_routes, dim=1)
        self._round_routes = []
        return route

    def _change(self, previous: Any, current: Any) -> float:
        if self.route_metric == "dedup_jaccard":
            return deduplicated_jaccard_change(previous, current)
        if self.route_metric == "dedup_count":
            return deduplicated_count_change(previous, current)
        squared = 0.5 * (
            (previous.clamp_min(0).sqrt() - current.clamp_min(0).sqrt()) ** 2
        ).sum(dim=-1)
        # Batch size is one in the deployed LIBERO policy.  Averaging the batch
        # as well keeps the scalar definition explicit if that ever changes.
        return float(squared.mean().sqrt().detach().cpu())

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        data_mask,
        noise=None,
    ):
        from moevla.models.moevla import make_att_2d_masks

        model = self.model
        batch = state.shape[0]
        device = state.device
        state = state.to(dtype=self.torch.float32)
        if noise is None:
            shape = (batch, model.config.n_action_steps, model.config.max_action_dim)
            noise = model.sample_noise(shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        pre_att_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pre_position_ids = self.torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.paligemma_with_expert.forward(
            pre_attention_mask=pre_att_masks,
            pre_position_ids=pre_position_ids,
            suf_attention_mask=None,
            suf_position_ids=None,
            past_key_values=None,
            prefix_embs=prefix_embs,
            suffix_embs=None,
            data_mask=None,
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )

        dt = self.torch.tensor(
            -1.0 / model.config.num_steps, dtype=self.torch.float32, device=device
        )
        x_t = noise
        flow_time = self.torch.tensor(1.0, dtype=self.torch.float32, device=device)
        rule = ConsecutiveRouteStopper(self.config)
        previous_route = None
        changes: list[float] = []
        completed = 0
        first_eligible = None

        while flow_time >= -dt / 2:
            self._round_routes = []
            self.collecting = True
            try:
                velocity = model.denoise_step(
                    state,
                    prefix_pad_masks,
                    prefix_att_masks,
                    past_key_values,
                    data_mask,
                    x_t,
                    flow_time.expand(batch),
                )
            finally:
                self.collecting = False
            current_route = self._finish_round()

            x_t += dt * velocity
            flow_time += dt
            completed += 1

            if previous_route is not None:
                change = self._change(previous_route, current_route)
                changes.append(change)
                eligible = rule.observe(completed, change)
                if eligible and first_eligible is None:
                    first_eligible = completed
                if self.enabled and eligible:
                    break
            previous_route = current_route

        expected = int(model.config.num_steps)
        if not 1 <= completed <= expected:
            raise RuntimeError("invalid denoise step count: %d" % completed)
        self.last_record = {
            "enabled": self.enabled,
            "completed_steps": completed,
            "stopped_early": completed < expected,
            "route_changes": changes,
            "first_eligible_step": first_eligible,
            "route_metric": self.route_metric,
        }
        return x_t

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", default="cpu")
    parser.add_argument("--suite", default="goal")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--libero-wrist-layout", default="checkpoint-right")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.0047)
    parser.add_argument("--min-steps", type=int, default=8)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--route-metric", choices=ROUTE_METRICS, default=ROUTE_METRICS[0])
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--verify-baseline", action="store_true")
    args = parser.parse_args()

    config = StopConfig(
        threshold=args.threshold,
        min_steps=args.min_steps,
        consecutive=args.consecutive,
        max_steps=10,
    )
    from himoe_libero_bridge.server import PolicyServer, create_policy

    if args.gpu == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        from probe_flow_lead_cpu import install_cpu_autocast

        install_cpu_autocast(torch)
        torch.set_num_threads(args.threads)
        from himoe_libero_bridge.policies import HiMoEPolicy

        policy = HiMoEPolicy(
            checkpoint_dir=args.checkpoint_dir,
            suite=args.suite,
            upstream_root=args.upstream_root,
            require_cuda=False,
            libero_wrist_layout=args.libero_wrist_layout,
        )
        print("running on CPU with %d threads" % torch.get_num_threads(), flush=True)
    else:
        policy = create_policy(
            "himoe",
            args.checkpoint_dir,
            args.upstream_root,
            args.gpu,
            args.suite,
            args.libero_wrist_layout,
        )

    from himoe_router_recorder import discover_gates

    upstream_policy = policy._policy
    core = upstream_policy.model
    hb_gates = [gate for gate in discover_gates(core) if gate.kind == "HB"]
    if len(hb_gates) != 8:
        raise RuntimeError("expected 8 HB gates, found %d" % len(hb_gates))
    original_sample = upstream_policy._sample_actions
    sampler = AdaptiveRouteSampler(core, hb_gates, config, route_metric=args.route_metric)
    upstream_policy._sample_actions = sampler.sample_actions

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    stats = {"calls": 0, "baseline_self_check_max_abs": None}
    inner = policy.infer
    patched_sample = sampler.sample_actions

    def flush() -> None:
        (out / "adaptive_records.json").write_text(json.dumps(records, indent=2))
        mean_steps = float(np.mean([row["completed_steps"] for row in records])) if records else 0.0
        (out / "capture_summary.json").write_text(
            json.dumps(
                {
                    "calls": stats["calls"],
                    "mean_steps": mean_steps,
                    "baseline_self_check_max_abs": stats["baseline_self_check_max_abs"],
                    "route_metric": args.route_metric,
                    "stop_config": config.__dict__,
                },
                indent=2,
            )
        )

    def infer(observation):
        enabled_value = observation.pop(ENABLED_KEY, False)
        if not isinstance(enabled_value, (bool, np.bool_)):
            raise ValueError("%s must be bool" % ENABLED_KEY)
        enabled = bool(enabled_value)
        arm = str(observation.pop(ARM_KEY, "adaptive" if enabled else "baseline"))
        episode = int(observation.pop(EPISODE_KEY, -1))

        reference = None
        if args.verify_baseline and not enabled and stats["baseline_self_check_max_abs"] is None:
            sampler.suspend()
            upstream_policy._sample_actions = original_sample
            reference = inner(dict(observation))
            upstream_policy._sample_actions = patched_sample

        sampler.begin(enabled)
        started = time.perf_counter()
        response = inner(observation)
        sample_ms = (time.perf_counter() - started) * 1000.0
        record = dict(sampler.last_record or {})
        if not record:
            raise RuntimeError("adaptive sampler produced no query record")

        if reference is not None:
            difference = float(
                np.max(
                    np.abs(
                        np.asarray(reference["actions"], dtype=np.float64)
                        - np.asarray(response["actions"], dtype=np.float64)
                    )
                )
            )
            stats["baseline_self_check_max_abs"] = difference
            if difference > 1e-6:
                raise RuntimeError(
                    "patched 10-step sampler differs from upstream by %.3e" % difference
                )
            print("baseline sampler self-check max|delta action|=%.3e" % difference, flush=True)

        response[STEPS_KEY] = np.int16(record["completed_steps"])
        response[STOPPED_KEY] = bool(record["stopped_early"])
        response[CHANGES_KEY] = np.asarray(record["route_changes"], dtype=np.float32)
        response[FIRST_ELIGIBLE_KEY] = np.int16(record["first_eligible_step"] or -1)
        response[SAMPLE_MS_KEY] = np.float32(sample_ms)
        record.update(
            {
                "call": stats["calls"],
                "arm": arm,
                "episode_id": episode,
                "sample_ms": sample_ms,
            }
        )
        records.append(record)
        stats["calls"] += 1
        flush()
        print(
            "query %-3d arm=%-8s rounds=%2d stop=%s change=%s"
            % (
                stats["calls"] - 1,
                arm,
                record["completed_steps"],
                record["stopped_early"],
                " ".join("%.4f" % value for value in record["route_changes"]),
            ),
            flush=True,
        )
        return response

    policy.infer = infer
    policy.metadata.update(
        {
            "adaptive_denoise_supported": True,
            "adaptive_enabled_key": ENABLED_KEY,
            "adaptive_stop_config": config.__dict__,
            "adaptive_route_metric": args.route_metric,
        }
    )

    done = {"value": False}

    def shutdown(*_unused) -> None:
        if done["value"]:
            return
        done["value"] = True
        flush()
        sampler.close()
        print("saved %d adaptive query records" % len(records), flush=True)

    import signal

    def terminate(*_unused) -> None:
        try:
            shutdown()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    print(
        "serving on ws://%s:%d metric=%s threshold=%.4f min=%d consecutive=%d"
        % (
            args.host,
            args.port,
            args.route_metric,
            config.threshold,
            config.min_steps,
            config.consecutive,
        ),
        flush=True,
    )
    try:
        PolicyServer(policy, args.host, args.port, policy.backend_name).serve_forever()
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
