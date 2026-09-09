#!/usr/bin/env python3
"""Recall-versus-FPR curves: unified detector, null arms, survival baseline."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import protocol as P

COHORT_LABEL = {
    "development_main": "development_main (in-sample)",
    "external_8b": "external_8b (sealed)",
    "legacy_main16x32": "legacy_16x32 (sealed)",
    "corpus": "whole corpus, 1358 risks",
}
STYLE = {
    "unified": ("#1f77b4", "-", 2.2, "unified detector"),
    "unified_plus_prior": ("#17becf", "-", 1.8, "unified + survival-prior offset"),
    "ablate_raw_only": ("#2ca02c", "--", 1.5, "ablation: raw only"),
    "ablate_no_self": ("#9467bd", "--", 1.3, "ablation: drop self"),
    "ablate_no_gating": ("#8c564b", "--", 1.3, "ablation: drop chunk gating"),
    "null_white": ("#d62728", ":", 1.8, "null: white noise"),
    "null_episode_const": ("#ff7f0e", ":", 1.8, "null: episode-constant noise"),
}


def monotone(front: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    out = front.sort_values(x).reset_index(drop=True)
    out[y] = np.maximum.accumulate(out[y].to_numpy())
    return out


def curve_for(table: pd.DataFrame, name: str, cohort: str) -> pd.DataFrame | None:
    sub = table[table["detector"] == name]
    if sub.empty or f"{cohort}.iw_fpr" not in sub.columns:
        return None
    sub = sub.dropna(subset=[f"{cohort}.iw_fpr", f"{cohort}.iw_recall"])
    if sub.empty:
        return None
    front = P.pareto_front(sub, "development_main.iw_fpr", "development_main.iw_recall")
    return monotone(front, f"{cohort}.iw_fpr", f"{cohort}.iw_recall")


def main() -> None:
    table = pd.read_csv(P.RESULTS / "operating_points_pooled.csv.gz")
    cohorts = ["external_8b", "legacy_main16x32", "corpus", "development_main"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    for ax, cohort in zip(axes.ravel(), cohorts):
        for name, (colour, ls, lw, label) in STYLE.items():
            c = curve_for(table, name, cohort)
            if c is None:
                continue
            ax.plot(c[f"{cohort}.iw_fpr"], c[f"{cohort}.iw_recall"],
                    color=colour, ls=ls, lw=lw, label=label)
        surv = table[table["detector"] == "survival_prior"]
        if not surv.empty and f"{cohort}.iw_fpr" in surv.columns:
            s = surv.dropna(subset=[f"{cohort}.iw_fpr"])
            corner = s.loc[s[f"{cohort}.iw_recall"] >= 0.999]
            if len(corner):
                best = corner.loc[corner[f"{cohort}.iw_fpr"].idxmin()]
                ax.scatter([best[f"{cohort}.iw_fpr"]], [best[f"{cohort}.iw_recall"]],
                           marker="*", s=260, color="k", zorder=5,
                           label="survival baseline (no information)")
                ax.axhline(best[f"{cohort}.iw_recall"], color="k", lw=0.6, alpha=0.4)
                ax.axvline(best[f"{cohort}.iw_fpr"], color="k", lw=0.6, alpha=0.4)
            ss = s.sort_values(f"{cohort}.iw_fpr").copy()
            ss[f"{cohort}.iw_recall"] = np.maximum.accumulate(
                ss[f"{cohort}.iw_recall"].to_numpy()
            )
            ax.step(ss[f"{cohort}.iw_fpr"], ss[f"{cohort}.iw_recall"], where="post",
                    color="k", lw=1.2, alpha=0.55,
                    label="survival baseline envelope" if cohort == "external_8b" else None)
        ax.axhline(0.80, color="crimson", lw=1.0, ls="-.", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("in-window false-alarm rate (FP / non-risk episodes)")
        ax.set_ylabel("in-window risk recall (phase $\\leq$ 0.65)")
        ax.set_title(COHORT_LABEL[cohort], fontsize=11)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Unified task-agnostic early-warning detector: operating curve\n"
        "every point is chosen on development_main and applied unchanged to the "
        "sealed cohorts",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(P.RESULTS / "recall_vs_fpr.png", dpi=160)
    print("wrote", P.RESULTS / "recall_vs_fpr.png")


if __name__ == "__main__":
    main()
