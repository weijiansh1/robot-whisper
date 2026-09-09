#!/usr/bin/env python3
"""Action sensitivity of the policy to HB gate biases at alarm states: state token versus action tokens, learned direction versus random."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from collection_routes import CAPTURE_KEY, PROBS_KEY
from collection_storage import atomic_json, load_snapshot
from scope_bias_control import BIAS_KEY, PROTOCOL


def scaled(delta, rms):
    """Centre per position, scale to a per-position RMS of `rms`, cap |bias| at 8."""
    delta = delta - delta.mean(-1, keepdims=True)
    norm = np.sqrt(np.square(delta).mean(-1, keepdims=True))
    delta = delta * (rms / np.maximum(norm, 1e-12)) * (norm > 1e-9)
    return np.clip(delta, -8.0, 8.0).astype(np.float32)


def bias_set(effect, rng):
    """effect: [2, 8, 32] standardised success-minus-failure routing direction (state token, action tokens)."""
    state, action = effect[0], effect[1]
    zero = np.zeros((8, 10, 11, 32), np.float32)
    out = {"native": zero}
    for rms in (0.35, 1.0, 3.0):
        b = zero.copy()
        b[:, :, 0, :] = state[:, None, :]
        out["state_dir_%.2f" % rms] = scaled(b, rms)
    b = zero.copy(); b[:, :, 0, :] = state[:, None, :]; b = scaled(b, 1.0)
    out["state_random_1.00"] = np.take_along_axis(b, np.argsort(rng.random(b.shape), axis=-1), axis=-1)
    b = zero.copy(); b[:, :, 0, :] = -state[:, None, :]
    out["state_negdir_1.00"] = scaled(b, 1.0)
    b = zero.copy(); b[4:, :, 1:, :] = action[4:, None, None, :]
    out["action_back_dir_0.35"] = scaled(b, 0.35)
    b = zero.copy(); b[:, :, 1:, :] = action[:, None, None, :]
    out["action_all_dir_1.00"] = scaled(b, 1.0)
    b = zero.copy(); b[:, :, 0, :] = state[:, None, :]; b[:, :, 1:, :] = action[:, None, None, :]
    out["both_dir_1.00"] = scaled(b, 1.0)
    return out


def run(args):
    from himoe_libero_bridge.client import PolicyClient
    from himoe_libero_bridge.preprocess import build_policy_observation
    client = PolicyClient(port=args.port, connect_timeout=60, inference_timeout=300)
    if client.metadata.get("scope_bias_protocol") != PROTOCOL:
        raise ValueError("Not a scope-bias server")
    plan = json.loads((args.run / "plan.json").read_text())
    effect = np.load(args.direction)["effect"]
    rng = np.random.default_rng(20260909)
    biases = bias_set(effect, rng)
    rows, forwards = [], 0
    for task in plan["tasks"]:
        replay = json.loads((args.run / "tasks" / task["main_id"] / "replay/result.json").read_text())
        original = json.loads((Path(task["parent_directory"]) / "result.json").read_text())
        for event in replay["events"]:
            if event["timing"] not in args.timings:
                continue
            location = args.run / "tasks" / task["main_id"] / "replay/events" / event["event_id"]
            saved = load_snapshot(location / "snapshot")
            physics = json.loads((location / "physics.json").read_text())
            obs = saved["observation"]
            policy_rng = np.random.default_rng()
            policy_rng.bit_generator.state = saved["policy_rng"]
            noise = policy_rng.standard_normal((10, 24)).astype(np.float32)
            request = build_policy_observation(obs, original["prompt"])
            request.update({"flow/noise": noise, "routing/capture": True, CAPTURE_KEY: True})
            results = {}
            for name, bias in biases.items():
                req = dict(request)
                req[BIAS_KEY] = bias
                tick = time.monotonic()
                response = client.infer(req)
                forwards += 1
                if response["flow/noise_sha256"] != hashlib.sha256(noise.tobytes()).hexdigest():
                    raise ValueError("Noise identity")
                results[name] = dict(actions=np.asarray(response["actions"], np.float32),
                                     probs=np.asarray(response[PROBS_KEY], np.float32), seconds=time.monotonic() - tick)
            native = results["native"]
            eef = np.asarray(physics["eef"], float)
            target = None if physics.get("target_position") is None else np.asarray(physics["target_position"], float)
            for name, res in results.items():
                d = res["actions"] - native["actions"]
                p_nat, p_eff = native["probs"][:, -1], res["probs"][:, -1]
                top_nat, top_eff = p_nat.argmax(-1), p_eff.argmax(-1)
                toward = None
                if target is not None:
                    goal = target[:2] - eef[:2]
                    step = d[:3, :2].mean(0)
                    toward = float(np.dot(step, goal) / (np.linalg.norm(step) * np.linalg.norm(goal) + 1e-9))
                rows.append(dict(main_id=task["main_id"], failed=task["failed"], timing=event["timing"], base_task=task["base_task"],
                    bias=name, xyz_rms=float(np.sqrt(np.square(d[:, :3]).mean())), rot_rms=float(np.sqrt(np.square(d[:, 3:6]).mean())),
                    gripper_mean_change=float(d[:, 6].mean()), gripper_sign_flips=int(np.sum(np.sign(res["actions"][:, 6]) != np.sign(native["actions"][:, 6]))),
                    native_xyz_rms=float(np.sqrt(np.square(native["actions"][:, :3]).mean())),
                    state_top1_changed=float((top_nat[:, 0] != top_eff[:, 0]).mean()),
                    action_top1_changed=float((top_nat[:, 1:] != top_eff[:, 1:]).mean()),
                    toward_target_cosine=toward, seconds=res["seconds"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, dict(protocol=PROTOCOL, forwards=forwards, biases=list(biases), rows=rows))
    summary = {}
    for name in biases:
        sub = [r for r in rows if r["bias"] == name]
        summary[name] = dict(n=len(sub), xyz_rms_median=float(np.median([r["xyz_rms"] for r in sub])),
                             xyz_rms_p90=float(np.percentile([r["xyz_rms"] for r in sub], 90)),
                             gripper_flips_mean=float(np.mean([r["gripper_sign_flips"] for r in sub])),
                             state_top1_changed=float(np.mean([r["state_top1_changed"] for r in sub])),
                             action_top1_changed=float(np.mean([r["action_top1_changed"] for r in sub])),
                             toward_cosine_median=float(np.nanmedian([r["toward_target_cosine"] if r["toward_target_cosine"] is not None else np.nan for r in sub])))
    print("native xyz action RMS median:", round(float(np.median([r["native_xyz_rms"] for r in rows if r["bias"] == "native"])), 3))
    print("%-22s %4s %9s %9s %8s %9s %9s %8s" % ("bias", "n", "xyzRMS50", "xyzRMS90", "gripFlp", "stateTop1", "actTop1", "toward"))
    for name, v in summary.items():
        print("%-22s %4d %9.4f %9.4f %8.2f %9.2f %9.2f %8.2f" % (name, v["n"], v["xyz_rms_median"], v["xyz_rms_p90"], v["gripper_flips_mean"],
                                                                 v["state_top1_changed"], v["action_top1_changed"], v["toward_cosine_median"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("design/repair_main_20260908"))
    parser.add_argument("--direction", type=Path, default=Path("design/post_alarm_routing_analysis_20260909/contrast_direction.npz"))
    parser.add_argument("--timings", nargs="+", default=["mid", "late"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("design/scope_bias_probe_20260909.json"))
    run(parser.parse_args())
