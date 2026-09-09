#!/usr/bin/env python3
"""Post-hoc diagnostics of convergence without task recovery; no policy fitting."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, digest, records
from v8_closed_loop import ARMS


def run(args):
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "passed":
        raise ValueError("Complete audit required")
    root = Path(audit["run"])
    plan = json.loads((root / "plan.json").read_text())
    tasks = audit["tasks"]
    groups = {}
    for arm in ARMS[1:]:
        rows = [(t, next(b for b in t["branches"] if b["arm"] == arm)) for t in tasks]
        confirmed = [(t, b) for t, b in rows if b["exit_reason"] == "population_confirmed"]
        groups[arm] = dict(confirmed=len(confirmed),
            originally_failed=sum(not t["native_success"] for t, b in confirmed),
            final_failed=sum(not b["success"] for t, b in confirmed),
            failed_without_post_exit_exceedance=sum(not b["success"] and not b["post_exit_recurrence"] for t, b in confirmed),
            confirmed_parent_ids=[t["main_id"] for t, b in confirmed])
    harmed = []
    for task in tasks:
        if not task["native_success"]:
            continue
        for branch in task["branches"]:
            if branch["success"]:
                continue
            spec = next(t for t in plan["tasks"] if t["main_id"] == task["main_id"])
            original = json.loads((Path(spec["parent_directory"]) / "result.json").read_text())
            event = spec["events"][0]["event_id"]
            directory = root / "events" / event / branch["arm"]
            suffix = list(records(directory / "suffix"))
            first = next((row for row in suffix if int(row["candidate_id"]) != 0), None)
            change = None
            if first is not None:
                pool = list(records(directory / "pools" / str(int(first["relative_query"])) / "candidates"))
                a, b = np.asarray(first["actions"], float), np.asarray(pool[0]["actions"], float)
                change = dict(query=int(first["query"]), candidate=int(first["candidate_id"]),
                    continuous_action_rms=float(np.sqrt(((a[:, :6]-b[:, :6])**2).mean())),
                    gripper_sign_mismatch_steps=int(np.count_nonzero(np.sign(a[:, 6]) != np.sign(b[:, 6]))),
                    selected_risk=float(first["selected_risk"]), default_risk=float(first["default_risk"]))
            harmed.append(dict(main_id=task["main_id"], benchmark=task["benchmark"], category=task["category"],
                base_task=task["base_task"], arm=branch["arm"], native_terminal_step=original["action_steps"],
                actual_terminal_step=branch["final_action_steps"], first_changed_chunk=change,
                interpretation="Same-observation command comparison, not proof of the physical cause of failure"))
    candidates = [t for t in tasks if t["trigger_heads"]["curvature"] and not t["native_success"] and
        any(b["arm"] == "iid_v8" and b["exit_reason"] == "population_confirmed" and not b["post_exit_recurrence"] for b in t["branches"])]
    example = min(candidates, key=lambda t: (t["start_query"], t["main_id"])) if candidates else None
    proof = None
    args.output.mkdir(parents=True, exist_ok=False)
    if example is not None:
        spec = next(t for t in plan["tasks"] if t["main_id"] == example["main_id"])
        event = spec["events"][0]["event_id"]
        original = json.loads((Path(spec["parent_directory"]) / "result.json").read_text())
        branch = next(b for b in example["branches"] if b["arm"] == "iid_v8")
        last = branch["exit_query"]-example["start_query"]
        proof = dict(main_id=example["main_id"], benchmark=example["benchmark"], base_task=example["base_task"],
            prompt=original["prompt"], alarm_query=example["start_query"]-1, exit_query=branch["exit_query"],
            exit_action_step=(branch["exit_query"]+1)*10, final_action_step=branch["final_action_steps"],
            success=branch["success"], post_exit_recurrence=branch["post_exit_recurrence"],
            confirming_pools=[p for p in branch["pool_diagnostics"] if last-2 <= p["index"] <= last])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
        colors = dict(native="#697780", iid_v8="#007e91", guided_v8="#b74368")
        for arm in colors:
            rows = list(records(root / "events" / event / arm / "suffix"))
            axes[0].plot([int(r["query"]) for r in rows], [float(r["component_scores"][4]) for r in rows],
                label=arm, color=colors[arm], marker=".")
        axes[0].axhline(plan["thresholds"][4], color="#a33e35", linestyle="--", label="frozen curvature threshold")
        axes[0].axhline(plan["thresholds"][4]-plan["margins"][4], color="#344e41", linestyle=":", label="curvature recovery target")
        pools = branch["pool_diagnostics"]
        axes[1].plot([example["start_query"]+p["index"] for p in pools], [p["maximum"] for p in pools],
            color=colors["iid_v8"], marker="o", label="maximum R among all 16 iid candidates")
        axes[1].axhline(-1, color="#344e41", linestyle="--", label="whole-population recovery target")
        for axis in axes:
            axis.axvline(branch["exit_query"]+.5, color="#222222", linestyle=":", label="recovery exit")
            axis.grid(axis="y", alpha=.2)
            axis.legend(fontsize=8)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set(title="Curvature below target; task still failed at step 520", ylabel="Current curvature score")
        axes[1].set(xlabel="Original query index", ylabel="Worst candidate margin score R")
        figure.savefig(args.output / "converged_failure.png", dpi=180)
        figure.savefig(args.output / "converged_failure.pdf")
        plt.close(figure)
    result = dict(audit_sha256=digest(args.audit), analyzer_sha256=digest(__file__), post_hoc=True,
        threshold_tuning=False, new_gpu_forwards=0,
        trigger_counts={name: sum(t["trigger_heads"][name] for t in tasks) for name in ("freeze", "turbulence", "inversion", "curvature")},
        convergence=groups, harmed_cases=harmed, converged_failure_example=proof)
    atomic_json(args.output / "summary.json", result)
    print(json.dumps(dict(trigger_counts=result["trigger_counts"], convergence=groups,
                         example_main_id=None if proof is None else proof["main_id"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
