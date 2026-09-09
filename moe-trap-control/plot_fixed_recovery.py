#!/usr/bin/env python3
"""Export trigger coverage, paired task utility and measured physical displacement."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(args):
    analysis=json.loads(args.analysis.read_text())
    cohort=analysis["primary"]
    methods=list(cohort["methods"])
    labels=["v7","v8","E-kNN","E/C-kNN","Stall","Random"]
    x=np.arange(len(methods))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold"})
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout="constrained")
    rows=[cohort["methods"][m] for m in methods]
    fail=[r["withdraw"]["triggered_original_failures"] for r in rows]
    healthy=[r["withdraw"]["triggered_original_successes"] for r in rows]
    axes[0].bar(x,fail,color="#557da0",label="Originally failed")
    axes[0].bar(x,healthy,bottom=fail,color="#d38b3d",label="Originally successful")
    axes[0].set_title("Effective trigger coverage")
    axes[0].set_ylabel("Unique original mains")
    axes[0].set_ylim(0,cohort["parents"]+1)
    axes[0].legend(fontsize=8,loc="upper right")
    for arm,offset,color in (("hold",-.18,"#7b838b"),("withdraw",.18,"#188677")):
        rescue=np.asarray([r[arm]["rescues"] for r in rows])
        harm=np.asarray([r[arm]["harms"] for r in rows])
        axes[1].bar(x+offset,rescue,.34,color=color,label=arm.capitalize()+" rescues")
        axes[1].bar(x+offset,-harm,.34,color=color,alpha=.4,hatch="//",label=arm.capitalize()+" harms")
        z=[r[arm]["median_observed_recovery_displacement_m"][2]*1000 if r[arm]["effective_triggers"] else np.nan for r in rows]
        axes[2].bar(x+offset,z,.34,color=color,label=arm.capitalize())
    axes[1].axhline(0,color="#555555",lw=.8)
    axes[1].set_title("Full-task outcome changes")
    axes[1].set_ylabel("Rescued (+) / harmed (-) mains")
    axes[1].set_ylim(-1.6,3.4)
    axes[1].set_yticks([-1,0,1,2,3])
    axes[1].legend(fontsize=8,loc="upper left",ncol=2)
    axes[2].axhline(40,color="#555555",ls="--",lw=.8,label="Lift target")
    axes[2].set_title("Actual displacement after recovery")
    axes[2].set_ylabel("Median end-effector height change (mm)")
    axes[2].set_ylim(0,55)
    axes[2].legend(fontsize=8,loc="upper left")
    for ax in axes:
        ax.set_xticks(x,labels)
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Fixed recovery module, different trigger policies\n%d perturbation mains: %d failed, %d successful; unchanged total action budget"%
        (cohort["parents"],cohort["original_failures"],cohort["original_successes"]),fontsize=13)
    fig.savefig(args.output,dpi=170)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
