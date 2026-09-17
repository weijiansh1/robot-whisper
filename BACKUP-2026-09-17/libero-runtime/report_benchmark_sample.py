"""Report completed benchmark trials separately from infrastructure failures."""

import argparse
import csv
import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "srv" / "src"))

from himoe_libero_bridge.batch import wilson_interval


def failure_estimate(failures, completed):
    interval = wilson_interval(failures, completed)
    interval.pop("successes")
    return {
        "failures": failures,
        "completed": completed,
        "failure_rate": failures / completed if completed else None,
        "failure_rate_wilson_95": interval,
    }


def read_batch(batch_dir):
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    plan = json.loads((batch_dir / "scenario-plan.json").read_text())
    lookup = {(j["base_task_id"], j["init_state_id"]): j for j in plan["jobs"]}
    rows = []
    for job in manifest["jobs"]:
        scenario = lookup[(job["task_id"], job["init_state_id"])]
        result = job.get("result") or {}
        if job["state"] == "completed":
            if not isinstance(result.get("success"), bool):
                raise ValueError("Completed episode lacks a boolean success flag")
            outcome = "succeeded" if result["success"] else "task_failed"
        else:
            outcome = "infra_failed" if job["state"] == "failed" else job["state"]
        perturbation = scenario["perturbation"]
        rows.append({
            "benchmark": plan["benchmark"],
            "base_task_id": job["task_id"],
            "base_task_name": scenario["base_task_name"],
            "task_name": scenario["task_name"],
            "init_state_id": job["init_state_id"],
            "flow_noise_seed": job["flow_noise_seed"],
            "difficulty": perturbation.get("difficulty_level") if isinstance(perturbation, dict) else None,
            "outcome": outcome,
            "action_steps": result.get("action_steps"),
            "inference_calls": result.get("inference_calls"),
            "duration_seconds": result.get("duration_seconds"),
            "artifact_dir": result.get("artifact_dir"),
            "video": result.get("video_path"),
        })
    counts = {key: sum(row["outcome"] == key for row in rows)
              for key in ("succeeded", "task_failed", "infra_failed", "pending", "running")}
    completed = counts["succeeded"] + counts["task_failed"]
    per_task = []
    for task_id in sorted({row["base_task_id"] for row in rows}):
        selected = [row for row in rows if row["base_task_id"] == task_id]
        succeeded = sum(row["outcome"] == "succeeded" for row in selected)
        failed = sum(row["outcome"] == "task_failed" for row in selected)
        per_task.append({
            "base_task_id": task_id, "base_task_name": selected[0]["base_task_name"],
            "succeeded": succeeded, **failure_estimate(failed, succeeded + failed),
        })
    result = {
        "benchmark": plan["benchmark"], "batch_dir": str(batch_dir),
        "planned": len(rows), "counts": counts,
        "finished": counts["pending"] + counts["running"] == 0,
        "success_rate": counts["succeeded"] / completed if completed else None,
        **failure_estimate(counts["task_failed"], completed),
        "per_task": per_task,
        "checkpoint_sha256": plan["server_metadata"]["checkpoint_sha256"],
        "sample_seed": plan["sample_seed"], "initial_state_ids": plan["init_state_ids"],
    }
    return result, rows


def verify_artifacts(rows):
    import imageio.v2 as imageio
    import numpy as np
    from himoe_libero_bridge.episode_trace import load_episode_trace

    verified = []
    for row in rows:
        if row["outcome"] not in ("succeeded", "task_failed"):
            continue
        artifact = Path(row["artifact_dir"])
        summary = json.loads((artifact / "summary.json").read_text())
        manifest, trace = load_episode_trace(artifact / "episode-trace.json")
        assert summary["status"] == "completed"
        assert manifest["result"]["trace_complete"]
        assert summary["success"] == (row["outcome"] == "succeeded")
        assert len(trace["successes"]) == row["action_steps"]
        assert trace["predicted_actions"].shape == (row["inference_calls"], 10, 7)
        assert int(trace["executed_lengths"].sum()) == row["action_steps"]
        assert summary["init_state_id"] == row["init_state_id"]
        assert summary["flow_noise_seed"] == row["flow_noise_seed"]
        assert summary["task_name"] == row["task_name"]
        with imageio.get_reader(row["video"]) as reader:
            frames = reader.count_frames()
            first = reader.get_data(0)
            last = reader.get_data(frames - 1)
        assert frames == summary["video_frames"] == 11 + row["action_steps"]
        assert first.std() > 1 and last.std() > 1
        verified.append({
            "benchmark": row["benchmark"], "base_task_id": row["base_task_id"],
            "init_state_id": row["init_state_id"], "video_frames": frames,
            "first_last_pixel_difference": float(np.abs(first.astype(float) - last).mean()),
            "trace_hashes_and_shapes": "passed",
        })
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    results, rows = [], []
    for batch in args.batches:
        result, batch_rows = read_batch(batch.resolve())
        results.append(result)
        rows.extend(batch_rows)
    report = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "HiMoE LIBERO-10 weights without further training. Plus camera viewpoints and Pro position swap only. Earlier exploratory episodes excluded.",
        "uncertainty": "Approximate 95% Wilson intervals for the sampled episode failure proportions. Tasks and initial states can be correlated; these are not benchmark-wide guarantees.",
        "benchmarks": results,
    }
    if args.verify:
        report["artifact_verification"] = verify_artifacts(rows)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        with (args.output / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"benchmarks": [
        {key: result[key] for key in ("benchmark", "planned", "completed", "counts", "failure_rate", "failure_rate_wilson_95", "finished")}
        for result in results
    ]}, indent=2))


if __name__ == "__main__":
    main()
