"""Export comparison tables, operating-point plots and the same illustrated queries."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import HERE, METHODS
from core import digest, write_json
from run_jaccard import load
from plot_knn import save

LABELS = {
    "dyn_success_knn_k20": "10D dynamic Euclidean kNN",
    "route_ja_k20": "Aligned Top-4 JA kNN",
    "route_wj_site_k20": "WJ kNN: mean over sites",
    "route_wj_aligned_k20": "WJ kNN: aligned concatenation",
    "route_hellinger_aligned_k20": "Aligned Hellinger kNN",
    "route_success_knn_k20": "Earlier pooled routing kNN",
    "eef_motion_low": "Low EEF motion",
}
COLORS = dict(zip(LABELS, ("#bf4142", "#c38318", "#267aa7", "#20978b", "#8663a0", "#7a838b", "#41733a")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round6_jaccard_knn")
    parser.add_argument("--baseline", type=Path, default=HERE.parent / "results/round5_knn")
    args = parser.parse_args()
    output, baseline = args.input.resolve(), args.baseline.resolve()
    evaluation = json.loads((output / "evaluation_summary.json").read_text())
    for name,expected in evaluation["artifacts"].items():
        assert digest(output/name)==expected,name
    ranks=pd.read_csv(output / "ranking_metrics.csv")
    alarms=pd.read_csv(output / "alarm_metrics.csv")
    same=ranks.loc[(ranks.scope=="unseen") & (ranks.view=="common_horizon")]
    horizon=same.groupby("method")[["task_macro_auc","within_init_macro_auc","scorable_fraction"]].mean()
    horizon.to_csv(output / "common_horizon_summary.csv")
    grouped=alarms.loc[(alarms.scope=="unseen") & (alarms.calibration=="task_init")]
    operations=grouped.groupby(["method","alpha"])[["recall","fpr","t_det"]].mean().reset_index()
    operations.to_csv(output / "unseen_operating_points.csv",index=False)
    main_point=grouped.loc[np.isclose(grouped.alpha,.05)]
    main_point.groupby(["suite","method"])[["recall","fpr","t_det"]].mean().to_csv(output / "suite_group5_summary.csv")
    same.groupby(["suite","method"])[["task_macro_auc","within_init_macro_auc","scorable_fraction"]].mean().to_csv(output / "suite_common_horizon_summary.csv")
    timing=pd.read_csv(output / "physical_timing.csv")
    timing=timing.loc[(timing.scope=="unseen") & (timing.calibration=="task_init") & timing.drop_goal_release.notna()]
    physical=[]
    for method,group in timing.groupby("method"):
        detected=group.first_alarm>=0
        lag=group.loc[detected].first_alarm-group.loc[detected].drop_goal_release
        physical.append({"method":method,"appearances":len(group),"unique_trajectories":group.global_row.nunique(),
            "detected":int(detected.sum()),"strictly_before_release":int((lag<0).sum()),
            "median_detected_lag_queries":float(lag.median())})
    pd.DataFrame(physical).to_csv(output / "physical_summary.csv",index=False)
    figure_dir=output / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(14.5,6.5),layout="constrained",gridspec_kw={"width_ratios":[1,1.25]})
    names=list(LABELS)
    axes[0].barh(np.arange(len(names)),horizon.loc[names].task_macro_auc,
                  color=[COLORS[name] for name in names],height=.62)
    axes[0].set_yticks(np.arange(len(names)),[LABELS[name] for name in names],fontsize=9)
    axes[0].invert_yaxis()
    axes[0].axvline(.5,color="#777777",lw=.8,ls="--")
    axes[0].set(xlim=(0,1),xlabel="Task-macro AUROC",title="Same-task common observation horizon")
    for i,name in enumerate(names):
        axes[0].text(horizon.loc[name].task_macro_auc+.014,i,f"{horizon.loc[name].task_macro_auc:.3f}",va="center",fontsize=9)
        group=operations.loc[operations.method==name].sort_values("alpha")
        axes[1].plot(group.fpr*100,group.recall*100,".-",lw=1,color=COLORS[name],label=LABELS[name])
        chosen=group.loc[np.isclose(group.alpha,.05)]
        axes[1].scatter(chosen.fpr*100,chosen.recall*100,s=75,color=COLORS[name],edgecolor="white",zorder=4)
    axes[1].set(xlim=(0,100*operations.loc[operations.method.isin(names)].fpr.max()+2),ylim=(0,103),
                 xlabel="Actual false-positive rate (%)",ylabel="Failure recall (%)",
                 title="Trajectory alarms; task/init group calibration")
    axes[1].legend(loc="lower right",fontsize=8,frameon=False)
    fig.suptitle("Train-free JA / WJ kNN: identical successful reference points and k=20\n"
                 "Unseen tasks, equal-weight mean over 12 suite/split settings; large circles: nominal 5% group calibration",fontsize=12)
    save(fig,figure_dir / "metric_comparison")
    source=baseline / "projection10/projected_points.csv"
    old_manifest=json.loads((baseline / "projection10/manifest.json").read_text())
    assert digest(source)==old_manifest["artifacts"]["projected_points.csv"]
    points=pd.read_csv(source)
    for fold,locations in points.groupby("fold").groups.items():
        data=load(output / "predictions" / f"{fold}.npz")
        rows=pd.Index(data["test_rows"]).get_indexer(points.loc[locations,"global_row"])
        assert (rows>=0).all() and data["test_unseen"][rows].all()
        queries=points.loc[locations,"query"].to_numpy(int)
        ai=list(data["alphas"]).index(.05)
        for mi,method in enumerate(METHODS):
            score=data["scores"][mi,rows,queries]
            threshold=float(data["thresholds"][1,ai,mi])
            points.loc[locations,f"{method}_score"]=score
            points.loc[locations,f"{method}_threshold"]=threshold
            points.loc[locations,f"{method}_ratio"]=score.astype(float)/threshold
    for method in METHODS:
        points[f"{method}_exceeds"]=points[f"{method}_score"]>points[f"{method}_threshold"]
    points.to_csv(output / "illustration_points.csv",index=False)
    tables=("common_horizon_summary.csv","unseen_operating_points.csv","suite_group5_summary.csv",
            "suite_common_horizon_summary.csv","physical_summary.csv","illustration_points.csv")
    write_json(output / "summary_manifest.json",{"summarizer_sha256":digest(Path(__file__)),
        "illustration_source_sha256":digest(source),"illustration_note":"Same 1162 selected queries; PC1/PC2 are the OLD dynamic-feature PCA coordinates, not a route-distance embedding",
        "artifacts":{str(path.relative_to(output)):digest(path) for path in
            [*(output/name for name in tables),*sorted(figure_dir.glob("*.png")),*sorted(figure_dir.glob("*.pdf"))]}})
    print(horizon.loc[names].to_string(),flush=True)
    print(pd.DataFrame(physical).set_index("method").loc[names].to_string(),flush=True)


if __name__ == "__main__":
    main()
