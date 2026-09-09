"""Export motion-model diagnostics, recovery tracking and paired task outcomes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(args):
    report = json.loads(args.analysis.read_text())
    cohort = report["primary"]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    models = report["dynamics_model"]["diagnostic_heldout"]["models"]
    values = [models[k]["equal_parent_vector_rmse_mm"] for k in ("arx", "persistence", "kinematic")]
    axes[0].bar([0,1,2], values, color=["#168879", "#6c7780", "#bd7742"])
    axes[0].set_xticks([0,1,2], ["Identified ARX", "Previous velocity", "Requested delta"])
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("One-step vector RMSE (mm)")
    axes[0].set_title("Motion-model diagnostic holdout")
    for i, value in enumerate(values):
        axes[0].text(i, value+.07, "%.3f"%value, ha="center", fontsize=9)
    axes[0].set_ylim(0, max(values)*1.18)
    controllers = [("withdraw", "P", "#647680"), ("mpc", "MPC", "#168879"),
                   ("mpc_disturbance", "MPC + observer", "#ba7841")]
    for i, (arm, label, color) in enumerate(controllers):
        value = cohort["mechanistic"][arm]["mean_parent_final_target_error_mm"]
        axes[1].bar(i, value, color=color)
        axes[1].text(i, value+.25, "%.2f"%value, ha="center", fontsize=9)
    axes[1].set_xticks([0,1,2], [v[1] for v in controllers])
    axes[1].set_ylim(bottom=0)
    axes[1].margins(y=.2)
    axes[1].set_ylabel("Final recovery-target error (mm)")
    axes[1].set_title("Equal weight per original main")
    methods = list(cohort["methods"])
    x = np.arange(len(methods))
    outcome_controllers = [("hold", "Hold", "#986a88")]+controllers
    for offset, (arm, label, color) in zip([-.30, -.10, .10, .30], outcome_controllers):
        rows = [cohort["methods"][m][arm] for m in methods]
        axes[2].bar(x+offset, [r["rescues"] for r in rows], .19, color=color, label=label)
        axes[2].bar(x+offset, [-r["harms"] for r in rows], .19, color=color, alpha=.4, hatch="//")
    axes[2].axhline(0, color="#444444", lw=.8)
    axes[2].set_xticks(x, ["v7", "v8", "E-kNN", "E/C-kNN", "Stall", "Random"], fontsize=8)
    axes[2].set_ylabel("Rescued (+) / harmed (-) original mains")
    axes[2].set_title("Complete task outcomes")
    axes[2].set_ylim(-1.4, 1.55)
    axes[2].set_yticks([-1, 0, 1])
    axes[2].legend(fontsize=8, ncol=2, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Identified recovery control: %d perturbation mains, %d native failures and %d successes\nSame targets, 16 recovery steps, 1 cm input bound and original total action budget"%
        (cohort["parents"], cohort["original_failures"], cohort["original_successes"]), fontsize=12)
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
