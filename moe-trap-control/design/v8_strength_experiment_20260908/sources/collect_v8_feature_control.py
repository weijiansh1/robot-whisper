#!/usr/bin/env python3
"""Complete C0 and paired feature-controlled suffixes from frozen native v8 alarms."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from collect_adaptive_control import RecoverySession
from collect_preflight_worker import input_digest, restore
from collection_routes import ALL_FIELDS, CAPTURE_KEY, PROBS_KEY, EFFECTIVE_IDS_KEY
from collection_storage import ShardWriter, atomic_json, digest, load_snapshot, records
from v8_feature_control import (PROTOCOL, ARMS, V8Monitor, BIAS_KEY, EFFECTIVE_PROBS, NATIVE_PROBS,
    make_bias, noise_for, seed_for, flow_features)


class FeatureSession(RecoverySession):
    def __init__(self, args, cache):
        super().__init__(args, cache)
        if self.client.metadata.get("v8_feature_protocol") != PROTOCOL:
            raise ValueError("Wrong feature-control server")
        self.report.update(protocol=PROTOCOL, contract=args.control["contract"], first_v8_alarm=self.task["first_v8_alarm"],
            active_heads=self.task["active_heads"], preexisting_parent_success=self.original["success"])

    def query(self, obs, noise, bias=None):
        if bias is None:
            return super().query(obs, noise)
        request = self.build_observation(obs, self.prompt)
        request.update({"flow/noise": noise, "routing/capture": True, CAPTURE_KEY: True, BIAS_KEY: bias})
        tick = time.monotonic()
        response = self.client.infer(request)
        elapsed = time.monotonic() - tick
        self.report["actual_model_queries"] += 1
        if response["collection/routing_mode"] != "v8_feature_bias":
            raise ValueError("Missing real gate intervention")
        for field in ALL_FIELDS:
            expected = ((8, 10, 11, 32 if field == PROBS_KEY else 4) if "/hb_" in field else
                        (4, 10, 11, 3 if field.endswith("as_probs") else 1))
            if np.asarray(response[field]).shape != expected or not np.isfinite(response[field]).all():
                raise ValueError("Incomplete controlled route capture")
        if any("hidden" in key.lower() for key in response):
            raise ValueError("Hidden capture is forbidden")
        if response["flow/noise_sha256"] != hashlib.sha256(noise.tobytes()).hexdigest():
            raise ValueError("Controlled noise differs from the shadow")
        return request, response, elapsed

    def advance(self, obs, actions, limit, steps, replicate=None):
        if replicate is None:
            return super().advance(obs, actions, limit, steps)
        tick, count, success = time.monotonic(), 0, False
        for action in actions[:limit]:
            seed = seed_for(self.args.main_id, replicate, steps + count, "environment")
            random.seed(seed)
            np.random.seed(seed)
            obs, _, done, _ = self.env.step(action.tolist())
            count += 1
            success = bool(self.env.check_success())
            if success:
                break
            if done:
                raise ValueError("Premature feature branch termination")
        return obs, count, success, time.monotonic() - tick

    def feature_branches(self):
        if self.report["c0"]["status"] != "passed":
            raise ValueError("Complete C0 required before intervention")
        event = self.report["events"][0]
        q0 = event["start_query"]
        location = self.args.output / "events" / event["event_id"] / "snapshot"
        saved = load_snapshot(location)
        prefix, previous = V8Monitor(), None
        for row in records(self.parent / "main"):
            if int(row["query"]) >= q0:
                break
            prefix.update(row[PROBS_KEY])
            previous = row[PROBS_KEY]
        if prefix.first_alarm != q0 - 1 or prefix.first_alarm != self.task["first_v8_alarm"]:
            raise ValueError("v8 causal prefix differs from the planned alarm")
        self.report["v8_prefix_replay_passed"] = True
        for replicate in range(self.args.control["replicates"]):
            for arm in self.args.control["arms"]:
                spec = ARMS[arm]
                directory = self.args.output / "branches" / ("r%d_%s" % (replicate, arm))
                self.new_env()
                obs = restore(self.env, saved, self.rng)
                monitor, effective_monitor = copy.deepcopy(prefix.v7), copy.deepcopy(prefix)
                previous_shadow = previous.copy()
                branch = dict(status="running", arm=arm, control=spec, replicate=replicate,
                    event_id=event["event_id"], start_query=q0, parent_main_id=self.args.main_id,
                    parent_commit_sha256=self.task["parent_commit_sha256"],
                    snapshot_manifest_sha256=digest(location / "manifest.json"),
                    starts_after_main_complete=True, starts_after_c0=True,
                    remaining_action_budget=self.horizon - saved["action_steps"],
                    queries=0, action_steps=0, success=False, controlled_queries=0, deployment_model_queries=0)
                self.report["branches"].append(branch)
                writer = ShardWriter(directory / "suffix")
                control_writer = ShardWriter(directory / "control", block_size=5)
                try:
                    while not branch["success"] and branch["action_steps"] < branch["remaining_action_budget"]:
                        index = branch["queries"]
                        q, steps = q0 + index, saved["action_steps"] + branch["action_steps"]
                        active = index < spec["duration"]
                        noise = noise_for(self.args.main_id, replicate, index)
                        request, shadow, inference = self.query(obs, noise)
                        response = shadow
                        bias = np.zeros((8, 10, 11, 32), np.float32)
                        if active:
                            bias = make_bias(shadow[PROBS_KEY], previous_shadow, spec["operator"], spec["strength"],
                                seed_for(self.args.main_id, replicate, index, "direction"))
                            controlled_request, response, duration = self.query(obs, noise, bias)
                            inference += duration
                            if input_digest(request) != input_digest(controlled_request):
                                raise ValueError("Shadow and control observation mismatch")
                            control = {"shadow/" + field: shadow[field] for field in ALL_FIELDS}
                            control.update(query=np.int32(q), relative_query=np.int32(index),
                                shadow_actions=shadow["actions"], noise=noise, logit_bias=bias,
                                native_probs_fp32=response[NATIVE_PROBS], effective_probs_fp32=response[EFFECTIVE_PROBS],
                                input_sha256=np.asarray(input_digest(request), dtype="S64"), previous_shadow=previous_shadow)
                            control_writer.append(control)
                            branch["controlled_queries"] += 1
                        branch["deployment_model_queries"] += 2 if active else 1
                        effective = response[EFFECTIVE_PROBS] if active else response[PROBS_KEY]
                        alarm = monitor.update(response[PROBS_KEY])
                        effective_alarm = effective_monitor.update(effective)
                        before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                        limit = min(10, self.horizon - steps)
                        obs, count, success, simulation = self.advance(obs, response["actions"], limit, steps, replicate)
                        after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                        if not np.isfinite(after).all():
                            raise ValueError("Non-finite branch physics")
                        row = self.record(request, response, q, steps, count, before, after, success, alarm, inference, simulation)
                        sorted_actual = np.sort(response[EFFECTIVE_IDS_KEY], -1)
                        sorted_shadow = np.sort(shadow[EFFECTIVE_IDS_KEY], -1)
                        row.update(relative_query=np.int32(index), control_active=np.bool_(active),
                            policy_seed=np.uint32(seed_for(self.args.main_id, replicate, index, "policy")),
                            environment_first_seed=np.uint32(seed_for(self.args.main_id, replicate, steps, "environment")),
                            environment_last_seed=np.uint32(seed_for(self.args.main_id, replicate, steps + count - 1, "environment")),
                            v8_effective_raw=effective_alarm["v8_raw"], v8_effective_scores=effective_alarm["v8_scores"],
                            v8_effective_freeze=np.float32(effective_alarm["freeze_score"]),
                            v8_shadow_raw=flow_features(shadow[PROBS_KEY])[0],
                            bias_rms=np.float32(np.sqrt(np.square(bias[4:, :, 1:]).mean())),
                            changed_top4_fraction=np.float32(np.any(sorted_actual[4:, :, 1:] != sorted_shadow[4:, :, 1:], -1).mean()),
                            action_rms_vs_shadow=np.float32(np.sqrt(np.square(response["actions"][:, :7].astype(np.float64) -
                                                                                 shadow["actions"][:, :7]).mean())))
                        writer.append(row)
                        previous_shadow = shadow[PROBS_KEY].copy()
                        branch.update(queries=index + 1, action_steps=branch["action_steps"] + count, success=success)
                        if branch["queries"] % 8 == 0:
                            self.save()
                    branch["shards"], branch["control_shards"] = writer.close(), control_writer.close()
                finally:
                    writer.close()
                    control_writer.close()
                branch["status"] = "completed"
                atomic_json(directory / "branch.json", branch)
                self.close_env()
                self.save()


def run(args, cache):
    started = time.monotonic()
    session = FeatureSession(args, cache)
    try:
        session.replay()
        session.feature_branches()
        session.report["status"] = "completed"
    except BaseException:
        session.report.update(status="failed", invalid_pair=True, error=traceback.format_exc())
        raise
    finally:
        session.close_env()
        session.report["elapsed_seconds"] = time.monotonic() - started
        session.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("pro", "plus"), required=True)
    parser.add_argument("--model", choices=("long",), default="long")
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--render-gpu", type=int, choices=(0, 3), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--exact-noise-fastpath", action="store_true")
    args, cache = parser.parse_args(), {}
    try:
        for line in sys.stdin:
            message = json.loads(line)
            task = copy.copy(args)
            task.variant, task.seed, task.output = message["variant"], message["seed"], Path(message["output"])
            task.main_id, task.init_index, task.control = message["main_id"], message["init_index"], message["control"]
            cache["jobs"] = cache.get("jobs", 0) + 1
            result = dict(variant=task.variant, pid=__import__("os").getpid(), worker_job_index=cache["jobs"])
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    run(task, cache)
                result["exit_code"] = 0
            except Exception:
                traceback.print_exc(file=sys.stderr)
                result["exit_code"] = 1
            print("COLLECTION_RESULT " + json.dumps(result), flush=True)
    finally:
        if cache.get("client"):
            cache["client"].close()


if __name__ == "__main__":
    main()
