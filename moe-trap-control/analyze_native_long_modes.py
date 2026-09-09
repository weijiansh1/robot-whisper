#!/usr/bin/env python3
"""Summarize audited mode responses and plot an independently verified rescue."""

import argparse
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, digest, records
from mode_control import FAMILIES


def analyze(args):
    audit, physical = json.loads(args.audit.read_text()), json.loads(args.physical.read_text())
    if audit["status"] != "passed" or physical["status"] != "passed" or audit["plan_sha256"] != physical["plan_sha256"]:
        raise ValueError("Full independent audit and physical verification required")
    run = Path(audit["run"])
    plan, summary = json.loads((run / "plan.json").read_text()), json.loads((run / "summary.json").read_text())
    rows = [row for task in audit["tasks"] for row in task["branches"]]
    families, responses = [], []
    for family in FAMILIES:
        arms = [arm for arm in audit["arms"] if arm["arm"].startswith(family+"_r")]
        selected = [row for row in rows if row["family"] == family]
        families.append(dict(family=family, rescued=[a["rescued"] for a in arms], harmed=[a["harmed"] for a in arms],
            full_cohort_success=[a["full_cohort_success"] for a in arms],
            mean_full_cohort_success=float(np.mean([a["full_cohort_success"] for a in arms])),
            total_branch_model_queries=sum(a["actual_branch_model_queries"] for a in arms),
            selected_action_changed_chunks=sum(a["changed_chunks"] for a in arms)))
        for mode in sorted(audit["entry_mode_counts"]):
            group = [row for row in selected if row["entry_mode"] == mode]
            responses.append(dict(family=family, entry_mode=mode, parents=len({r["main_id"] for r in group}),
                repeats=len(group), original_failure_repeats=sum(not r["original_success"] for r in group),
                original_success_repeats=sum(r["original_success"] for r in group),
                rescued=sum(r["rescued"] for r in group), harmed=sum(r["harmed"] for r in group)))
    contrasts = []
    for task in plan["tasks"]:
        event = task["events"][0]
        for repeat in range(2):
            root = run / "events" / event["event_id"]
            mode = root / ("mode_r"+str(repeat)) / "pools/0"
            scalar = root / ("scalar_r"+str(repeat)) / "pools/0"
            pm = json.loads((mode / "selection.json").read_text())["candidate"]
            ps = json.loads((scalar / "selection.json").read_text())["candidate"]
            pool = list(records(mode / "candidates"))
            delta = pool[pm]["actions"][:, :6].astype(float)-pool[ps]["actions"][:, :6].astype(float)
            contrasts.append(dict(main_id=task["main_id"], repeat=repeat, mode_pick=pm, scalar_pick=ps,
                different=pm != ps, command_rms_6d=float(np.sqrt(np.mean(delta**2))),
                mode_scalar_risk=float(pool[pm]["scalar_cost"]), scalar_scalar_risk=float(pool[ps]["scalar_cost"]),
                mode_vector_cost=float(pool[pm]["vector_cost"]), scalar_vector_cost=float(pool[ps]["vector_cost"])))
    samples, telemetry = json.loads((run / "gpu_samples.json").read_text()), []
    for gpu in (0, 1, 2, 3):
        selected = [(sample, next(row for row in sample["gpus"] if row["gpu"] == gpu)) for sample in samples]
        full = [row for sample, row in selected if sample["active_tasks_by_gpu"][str(gpu)] == 8]
        telemetry.append(dict(gpu=gpu, samples=len(selected), eight_worker_samples=len(full),
            mean_utilization=float(np.mean([row["utilization_percent"] for _, row in selected])),
            eight_worker_mean_utilization=float(np.mean([row["utilization_percent"] for row in full])),
            eight_worker_mean_power_w=float(np.mean([row["power_w"] for row in full])),
            peak_power_w=max(row["power_w"] for _, row in selected)))
    rescues = [r for r in rows if r["rescued"]]
    case_details = []
    for main_id in sorted({r["main_id"] for r in rescues}):
        task = next(t for t in plan["tasks"] if t["main_id"] == main_id)
        root = run / "events" / task["events"][0]["event_id"]
        case = dict(main_id=main_id, base_task=task["base_task"], alarm_query=task["events"][0]["alarm_query"], arms=[])
        for arm in ("native", "resample_r0", "mode_r0", "withdraw_r0", "withdraw_r1"):
            suffix = list(records(root / arm / "suffix"))
            branch = json.loads((root / arm / "branch.json").read_text())
            for row in suffix:
                f, a, p, i, c = row["normalized_scores"]
                np.testing.assert_allclose(row["selected_risk"], max(f, min(a, p), i, c), rtol=0, atol=1e-12)
            case["arms"].append(dict(arm=arm, success=branch["success"], final_action_steps=branch["final_action_steps"],
                final_risk=float(suffix[-1]["selected_risk"]), final_mode=suffix[-1]["mode"].item().decode(),
                low_risk_queries=sum(float(r["selected_risk"]) < 0 for r in suffix)))
        case_details.append(case)
    result = dict(status="passed", run=str(run), audit_sha256=digest(args.audit), physical_sha256=digest(args.physical),
        analyzer_sha256=digest(__file__), families=families, mode_responses=responses,
        first_pool_contrasts=contrasts, different_first_selections=sum(r["different"] for r in contrasts),
        first_selection_pairs=len(contrasts), gpu=telemetry, actual_model_queries=summary["actual_model_queries"],
        collection_seconds=summary["collection_elapsed_seconds"], queries_per_second=summary["queries_per_second"],
        unique_rescued_parents=len({r["main_id"] for r in rescues}), rescued_branches=len(rescues), rescue_cases=case_details,
        mode_counts=audit["entry_mode_counts"], physical_env_steps=physical["env_step_calls"])
    atomic_json(args.output, result)
    if case_details:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        case = case_details[0]
        task = next(t for t in plan["tasks"] if t["main_id"] == case["main_id"])
        root = run / "events" / task["events"][0]["event_id"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        styles = [("native", "#626262", "--"), ("mode_r0", "#bd4154", "-"),
            ("withdraw_r0", "#197b56", "-"), ("withdraw_r1", "#2675b8", ":")]
        for arm, color, style in styles:
            suffix = list(records(root / arm / "suffix"))
            branch = json.loads((root / arm / "branch.json").read_text())
            x = [int(r["action_steps_before"]) for r in suffix]
            label = arm+(" (success)" if branch["success"] else " (failure)")
            axes[0].plot(x, [float(r["selected_risk"]) for r in suffix], color=color, linestyle=style, label=label)
            path = root / arm / "physical"
            physical_rows = list(records(path)) if path.exists() else []
            px = [int(r["action_step"]) for r in physical_rows]+x
            pz = [float(r["eef_before"][2]) for r in physical_rows]+[float(r["proprio"][2]) for r in suffix]
            axes[1].plot(px, pz, color=color, linestyle=style)
        axes[0].axhline(0., color="black", linewidth=.8)
        axes[0].axhline(-1., color="black", linewidth=.8, linestyle=":")
        axes[0].set_ylabel("Scalar MoE risk (margin units)")
        axes[0].legend(fontsize=9, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01))
        axes[1].set_ylabel("End-effector height (m)")
        axes[1].set_xlabel("Absolute executed environment step (limit 520)")
        for axis in axes:
            axis.grid(alpha=.2)
            axis.set_xlim(350, 520)
        fig.suptitle("Native Long: two moka pots, entry mode F\nOne rescued parent; paired withdrawal repeat still fails", fontsize=12)
        fig.savefig(args.output.with_suffix(".png"), dpi=180)
        plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if key not in ("first_pool_contrasts", "mode_responses")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())
