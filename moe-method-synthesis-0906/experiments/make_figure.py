"""One summary figure: saturation, headroom, and the vote trade-off."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import ledger_core as lc  # noqa: E402
import synth_core as sc  # noqa: E402

REF = lc.REFERENCE_BUDGET


def main() -> None:
    sat = pd.read_csv(sc.RESULTS / "saturation_curve.csv")
    ceil = pd.read_csv(sc.RESULTS / "headroom_ceiling.csv")
    hon = pd.read_csv(sc.RESULTS / "honest_operating_points.csv")
    dec = pd.read_csv(sc.RESULTS / "vote_decomposition.csv")
    prof = pd.read_csv(sc.RESULTS / "family_cut_profile.csv")

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))

    a = ax[0]
    for arm, style in (("family_representatives", "o-"),
                       ("all_admissible_detectors", "s--")):
        s = sat[(sat.arm == arm) & (sat.budget == REF)]
        a.plot(s.step, s.ext_recall, style, label=arm.replace("_", " "), ms=4)
    a.axhline(lc.HONEST_ARM["in_window_recall"], color="0.4", ls=":",
              label="standing honest arm")
    a.set_xlabel("methods added (order fixed on development)")
    a.set_ylabel("external in-window recall")
    a.set_title(f"saturation, per-detector budget {REF}")
    a.legend(fontsize=7)
    a.grid(alpha=.3)

    a = ax[1]
    for arm, style in (("ORACLE_in_sample_419", "^-"),
                       ("ORACLE_in_sample_622", "v-"),
                       ("NULL_rate_matched_419", "x:")):
        c = ceil[ceil.arm == arm]
        a.plot(c.ext_fp_target, c.best_tp / 564, style, label=arm, ms=4)
    for arm, m in (("family_union", "o"), ("dev_greedy8", "s")):
        h = hon[hon.arm.str.startswith(arm)].sort_values("ext_fp")
        a.plot(h.ext_fp, h.ext_recall, m + "-", label=f"honest {arm}", ms=4)
    a.scatter([lc.HONEST_ARM["fp"]], [lc.HONEST_ARM["in_window_recall"]],
              marker="*", s=180, color="k", zorder=5, label="standing honest arm")
    a.set_xscale("log")
    a.set_xlabel("external false alarms")
    a.set_ylabel("external in-window recall")
    a.set_title("headroom at matched external FP")
    a.legend(fontsize=6)
    a.grid(alpha=.3)

    a = ax[2]
    d = dec[(dec.arm == f"families@{REF}") & (dec.cohort == "external_8b")
            & (dec.k <= 5) & (dec.suite != "_base")]
    for s, g in d.groupby("suite"):
        a.plot(g.fpr, g.recall, "o-", label=s, ms=4)
    a.set_xscale("log")
    a.set_xlabel("external FPR")
    a.set_ylabel("external recall")
    a.set_title("vote threshold k = 1..5, per suite")
    a.legend(fontsize=7)
    a.grid(alpha=.3)

    a = ax[3]
    p = prof[(prof.arm == "development_shared419") & (prof.suite == "pooled")
             & (prof.similarity == "excess_overlap")]
    a.plot(p.cut, p.n_families_observed, "o-", label="observed", ms=4)
    a.plot(p.cut, p.n_families_null, "x:", label="rate-matched null", ms=5)
    a.set_yscale("log")
    a.set_xlabel("excess-overlap merge threshold")
    a.set_ylabel("families among 419 detectors")
    a.set_title("families vs null")
    a.legend(fontsize=7)
    a.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(sc.RESULTS / "synthesis_overview.png", dpi=140)
    print("wrote synthesis_overview.png")


if __name__ == "__main__":
    main()
