#!/usr/bin/env python3
"""Run a paired strength sweep using committed, previously audited C0 states."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from collect_adaptive_control import RecoverySession
from collect_v8_feature_control import FeatureSession
from collect_preflight_worker import input_digest, restore
from collection_routes import ALL_FIELDS, CAPTURE_KEY, PROBS_KEY, EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY
from collection_storage import ShardWriter, atomic_json, digest, load_snapshot, records
from v8_feature_control import EFFECTIVE_PROBS, NATIVE_PROBS
from v8_strength_control import PROTOCOL, BIAS_KEY, LOGIT_FIELDS, ARMS, strength_bias, noise_for, seed_for, V8Monitor, flow_features


class StrengthSession(RecoverySession):
    advance = FeatureSession.advance

    def __init__(self, args, cache):
        super().__init__(args, cache)
        if self.client.metadata.get("v8_strength_protocol") != PROTOCOL:
            raise ValueError("Wrong strength-control endpoint")
        replay = Path(self.task["replay_directory"])
        for file, key in (("result.json", "replay_result_sha256"), ("c0/branch.json", "replay_c0_sha256")):
            if digest(replay / file) != self.task[key]:
                raise ValueError("Previously audited C0 changed")
        old = json.loads((replay / "result.json").read_text())
        if old["status"] != "completed" or old["c0"]["status"] != "passed" or old["events"] != [self.task["event"]]:
            raise ValueError("Missing exact previous C0 event")
        self.report.update(protocol=PROTOCOL, contract=args.control["contract"], replicate=args.control["replicate"],
            first_v8_alarm=self.task["first_v8_alarm"], active_heads=self.task["active_heads"],
            preexisting_parent_success=self.original["success"], events=[self.task["event"]],
            c0=dict(status="passed", reused=True, compared_queries=0, replay_directory=str(replay),
                replay_result_sha256=self.task["replay_result_sha256"], replay_c0_sha256=self.task["replay_c0_sha256"]),
            weak_full_suffix_exact=False, baseline_first_query_exact=False)
        self.replay_directory = replay

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
            raise ValueError("Missing actual gate intervention")
        if response["flow/noise_sha256"] != hashlib.sha256(noise.tobytes()).hexdigest():
            raise ValueError("Mismatched flow noise")
        for field in (*ALL_FIELDS, *LOGIT_FIELDS, EFFECTIVE_PROBS, NATIVE_PROBS):
            if not np.isfinite(response[field]).all():
                raise ValueError("Non-finite strength response")
        if any("hidden" in key.lower() for key in response):
            raise ValueError("Hidden capture forbidden")
        return request, response, elapsed

    def collect(self):
        spec = self.args.control
        replicate = spec["replicate"]
        event = self.task["event"]
        q0 = event["start_query"]
        location = self.replay_directory / "events" / event["event_id"] / "snapshot"
        if digest(location / "manifest.json") != event["snapshot_manifest_sha256"]:
            raise ValueError("Original snapshot changed")
        saved = load_snapshot(location)
        prefix, previous = V8Monitor(), None
        for row in records(self.parent / "main"):
            if int(row["query"]) >= q0:
                break
            prefix.update(row[PROBS_KEY])
            previous = row[PROBS_KEY]
        if prefix.first_alarm != q0 - 1 or prefix.first_alarm != self.task["first_v8_alarm"]:
            raise ValueError("Changed v8 first-alarm prefix")
        old_native = list(records(self.replay_directory / "branches" / ("r%d_native" % replicate) / "suffix"))
        old_weak = list(records(self.replay_directory / "branches" / ("r%d_combined" % replicate) / "suffix"))
        for arm in spec["arms"]:
            control = ARMS[arm]
            directory = self.args.output / "branches" / arm
            self.new_env()
            obs = restore(self.env, saved, self.rng)
            monitor, effective_monitor = copy.deepcopy(prefix.v7), copy.deepcopy(prefix)
            previous_shadow = previous.copy()
            branch = dict(status="running", arm=arm, control=control, replicate=replicate,
                parent_main_id=self.args.main_id, event_id=event["event_id"], start_query=q0,
                starts_after_main_complete=True, starts_after_c0=True,
                snapshot_manifest_sha256=event["snapshot_manifest_sha256"],
                remaining_action_budget=520 - saved["action_steps"], queries=0, action_steps=0,
                success=False, controlled_queries=0, deployment_model_queries=0)
            self.report["branches"].append(branch)
            writer = ShardWriter(directory / "suffix")
            evidence_writer = ShardWriter(directory / "control", block_size=4)
            try:
                while not branch["success"] and branch["action_steps"] < branch["remaining_action_budget"]:
                    index = branch["queries"]
                    q, steps = q0 + index, saved["action_steps"] + branch["action_steps"]
                    active = index < control["duration"]
                    noise = noise_for(self.args.main_id, replicate, index)
                    request, shadow, inference = self.query(obs, noise)
                    if index == 0:
                        for key in (*ALL_FIELDS, "actions"):
                            np.testing.assert_array_equal(shadow[key], old_native[0][key], err_msg="Native state fidelity " + key)
                        np.testing.assert_array_equal(request["flow/noise"], old_native[0]["noise"])
                        if input_digest(request) != old_native[0]["input_sha256"].item().decode():
                            raise ValueError("Restored native observation differs")
                        self.report["baseline_first_query_exact"] = True
                    response = shadow
                    bias = np.zeros((8, 10, 11, 32), np.float32)
                    if active:
                        bias = strength_bias(shadow[PROBS_KEY], previous_shadow, control,
                            seed_for(self.args.main_id, replicate, index, "direction"))
                        controlled_request, response, duration = self.query(obs, noise, bias)
                        inference += duration
                        if input_digest(request) != input_digest(controlled_request):
                            raise ValueError("Shadow and control observations differ")
                        evidence = {"shadow/" + key: shadow[key] for key in ALL_FIELDS}
                        evidence.update({key: response[key] for key in LOGIT_FIELDS})
                        evidence.update(query=np.int32(q), relative_query=np.int32(index), shadow_actions=shadow["actions"],
                            noise=noise, logit_bias=bias, native_probs_fp32=response[NATIVE_PROBS],
                            effective_probs_fp32=response[EFFECTIVE_PROBS], previous_shadow=previous_shadow,
                            arithmetic_dtypes=np.asarray(response["v8_strength/dtypes"], dtype="S24"),
                            input_sha256=np.asarray(input_digest(request), dtype="S64"))
                        evidence_writer.append(evidence)
                        branch["controlled_queries"] += 1
                    effective = response[EFFECTIVE_PROBS] if active else response[PROBS_KEY]
                    alarm = monitor.update(response[PROBS_KEY])
                    status = effective_monitor.update(effective)
                    before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    obs, count, success, simulation = self.advance(obs, response["actions"], min(10, 520 - steps), steps, replicate)
                    after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    if not np.isfinite(after).all():
                        raise ValueError("Non-finite strength physics")
                    row = self.record(request, response, q, steps, count, before, after, success, alarm, inference, simulation)
                    difference = response["actions"].astype(np.float64) - shadow["actions"]
                    row.update(relative_query=np.int32(index), control_active=np.bool_(active),
                        policy_seed=np.uint32(seed_for(self.args.main_id, replicate, index, "policy")),
                        environment_first_seed=np.uint32(seed_for(self.args.main_id, replicate, steps, "environment")),
                        environment_last_seed=np.uint32(seed_for(self.args.main_id, replicate, steps + count - 1, "environment")),
                        v8_effective_raw=status["v8_raw"], v8_effective_scores=status["v8_scores"],
                        v8_effective_freeze=np.float32(status["freeze_score"]), v8_shadow_raw=flow_features(shadow[PROBS_KEY])[0],
                        bias_rms=np.float32(np.sqrt(np.square(bias[4:, :, 1:]).mean())),
                        changed_top4_fraction=np.float32(np.any(np.sort(response[EFFECTIVE_IDS_KEY][4:, :, 1:], -1) !=
                                                               np.sort(shadow[EFFECTIVE_IDS_KEY][4:, :, 1:], -1), -1).mean()),
                        action_rms_vs_shadow=np.float32(np.sqrt(np.square(difference).mean())),
                        action_max_abs_vs_shadow=np.float32(np.abs(difference).max()),
                        top1_combine_mean=np.float32(response[EFFECTIVE_WEIGHTS_KEY][4:, :, 1:].max(-1).mean()))
                    if arm == "combined1_5":
                        if index >= len(old_weak):
                            raise ValueError("Weak reference suffix length changed")
                        for key in old_weak[index]:
                            if key not in ("inference_seconds", "environment_seconds"):
                                np.testing.assert_array_equal(row[key], old_weak[index][key], err_msg="Weak full suffix fidelity " + key)
                    writer.append(row)
                    previous_shadow = shadow[PROBS_KEY].copy()
                    branch.update(queries=index + 1, action_steps=branch["action_steps"] + count,
                                  deployment_model_queries=branch["deployment_model_queries"] + (2 if active else 1), success=success)
                    if branch["queries"] % 8 == 0:
                        self.save()
                branch["shards"], branch["control_shards"] = writer.close(), evidence_writer.close()
            finally:
                writer.close()
                evidence_writer.close()
            if arm == "combined1_5":
                if branch["queries"] != len(old_weak):
                    raise ValueError("Weak reference terminated at a different query")
                self.report["weak_full_suffix_exact"] = True
            branch["status"] = "completed"
            atomic_json(directory / "branch.json", branch)
            self.close_env()
            self.save()


def run(args, cache):
    started = time.monotonic()
    session = StrengthSession(args, cache)
    try:
        session.collect()
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
    parser.add_argument("--model", default="long", choices=("long",))
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
            result = dict(variant=task.variant, pid=os.getpid(), worker_job_index=cache["jobs"])
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
