#!/usr/bin/env python3
"""Method-specific audits for the other two votes of the "consensus core".

A. Is every event / lag / grammar feature a shadow of episode length?
   -> per-feature within-task Spearman vs (T) and vs (failure), side by side.
B. `late_stasis_indicator` as a step-cap detector.
C. Re-run the published event clustering on
     (i)  the real 57-d matrix (reproduction check)
     (ii) a LENGTH-ONLY sentinel matrix of the same shape
     (iii) the real matrix with the terminal / late features removed
D. Jaccard of the 1-bit nuisance `length >= step cap` against each of the three
   "independently discovered" blocks, next to their mutual Jaccards.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from audit_pipeline import (
    BASE, OUT, SEED, CANDIDATE_K, load_cache, prepare_embedding, ward_labels,
    canonicalize, block_table, adjusted_rand_score, adjusted_mutual_info_score,
)

STEP_CAP = {
    "libero_goal/open_the_middle_drawer_of_the_cabinet": 30,
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside": 30,
    "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": 52,
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": 22,
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": 22,
}
# features whose value is fixed by, or explicitly reads, the terminal anchors
TERMINAL_FEATURES = [
    "soft_terminal_speed_ratio", "terminal_return_advantage",
    "terminal_return_indicator", "terminal_low_speed_indicator",
    "soft_terminal_drift_normalized", "layer_low_terminal_fraction",
    "late_low_speed_fraction", "late_stasis_indicator",
    "soft_late_early_ratio", "hard_late_early_ratio",
    "late_internal_early_ratio", "return_late_fraction", "peak_late_fraction",
]


def jac(a, b):
    return round(float((a & b).sum() / max((a | b).sum(), 1)), 4)


def cluster_event_matrix(matrix, score, k=4, seed=20260827):
    emb = prepare_embedding(matrix, seed)
    return canonicalize(ward_labels(emb, k), score), emb


def main() -> None:
    cache = load_cache()
    failure = np.asarray(cache["meta_failure"])
    task = np.asarray(cache["meta_task"]).astype(str)
    length = np.asarray(cache["meta_episode_length"])
    at_cap = length >= np.array([STEP_CAP[t] for t in task])
    core = np.load(OUT / "core_membership.npz", allow_pickle=False)
    result = {}

    ev = np.load(BASE / "route-change-events/event_features.npz", allow_pickle=True)
    Xe = np.asarray(ev["core"], dtype=np.float64)
    names = [str(x) for x in ev["core_names"]]
    score_e = np.asarray(ev["absolute_scale"])[
        :, [str(x) for x in ev["absolute_scale_names"]].index("soft_speed_mean_abs")]
    pub_labels = np.asarray(ev["labels"])
    assert (np.asarray(ev["labels"]) == pd.read_csv(
        BASE / "route-change-events/assignments.csv").event_cluster.values).all()

    # ------------------------------------------------------------------- D
    result["one_bit_nuisance_vs_blocks"] = dict(
        at_cap_n=int(at_cap.sum()), at_cap_failures=int((at_cap & failure).sum()),
        jaccard={
            "at_cap ~ aligned raw C1": jac(at_cap, core["raw_c1"]),
            "at_cap ~ event C0": jac(at_cap, core["event_c0"]),
            "at_cap ~ lag C7": jac(at_cap, core["lag_c7"]),
            "at_cap ~ consensus": jac(at_cap, core["consensus"]),
            "at_cap ~ union": jac(at_cap, core["union"]),
            "rawC1 ~ eventC0 (published)": jac(core["raw_c1"], core["event_c0"]),
            "rawC1 ~ lagC7 (published)": jac(core["raw_c1"], core["lag_c7"]),
            "eventC0 ~ lagC7 (published)": jac(core["event_c0"], core["lag_c7"]),
        },
    )

    # ------------------------------------------------------------------- A
    # per-feature within-task Spearman vs T and vs failure (moka pot, the only
    # task with enough failures for a stable within-task correlation)
    t8 = task == "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    rows = []
    for j, nm in enumerate(names):
        rt = spearmanr(Xe[t8, j], length[t8]).statistic
        rf = spearmanr(Xe[t8, j], failure[t8].astype(float)).statistic
        rows.append(dict(feature=nm, rho_length=round(float(rt), 3),
                         rho_failure=round(float(rf), 3),
                         gap=round(float(abs(rf) - abs(rt)), 3),
                         terminal_window=nm in TERMINAL_FEATURES))
    rows.sort(key=lambda r: -abs(r["rho_failure"]))
    result["event_feature_within_task_correlations_scene8"] = rows

    # ------------------------------------------------------------------- B
    lsi = Xe[:, names.index("late_stasis_indicator")] > 0.5
    result["late_stasis_indicator"] = dict(
        prevalence=round(float(lsi.mean()), 4),
        p_at_cap_given_flag=round(float(at_cap[lsi].mean()), 4),
        p_failure_given_flag=round(float(failure[lsi].mean()), 4),
        p_flag_given_failure=round(float(lsi[failure].mean()), 4),
        as_a_rule=dict(n=int(lsi.sum()), failure=int((lsi & failure).sum()),
                       precision=round(float(failure[lsi].mean()), 4),
                       failure_recall=round(float(lsi[failure].mean()), 4)),
        jaccard_vs_event_C0=jac(lsi, core["event_c0"]),
    )

    # ------------------------------------------------------------------- C
    variants = {}
    # (i) reproduction
    lab, _ = cluster_event_matrix(Xe, score_e, 4)
    variants["real_57d"] = dict(
        ari_vs_published=round(float(adjusted_rand_score(pub_labels, lab)), 4),
        blocks=block_table(lab, failure))
    # (ii) length-only sentinel with matched shape and matched column moments
    rng = np.random.default_rng(SEED)
    Ln = (length - length.mean()) / length.std()
    Xl = np.empty_like(Xe)
    for j in range(Xe.shape[1]):
        r = spearmanr(Xe[:, j], length).statistic
        r = 0.0 if not np.isfinite(r) else r
        z = r * Ln + np.sqrt(max(1 - r * r, 1e-6)) * rng.standard_normal(len(Ln))
        Xl[:, j] = z * Xe[:, j].std() + Xe[:, j].mean()
    lab_l, _ = cluster_event_matrix(Xl, score_e, 4)
    bt = block_table(lab_l, failure)
    variants["length_only_sentinel_57d"] = dict(
        note="each column = episode length + Gaussian noise, matched to the real "
             "column's Spearman rho with length and to its mean/sd. Zero routing.",
        ari_vs_published=round(float(adjusted_rand_score(pub_labels, lab_l)), 4),
        blocks=bt, best_f1=max(bt, key=lambda r: r["f1"]),
        best_precision=max([r for r in bt if r["n"] >= 100] or bt,
                           key=lambda r: (r["precision"], r["n"])),
        best_jaccard_vs_event_C0=max(jac(lab_l == r["block"], core["event_c0"]) for r in bt),
    )
    # (iii) drop terminal/late features
    keep = [j for j, nm in enumerate(names) if nm not in TERMINAL_FEATURES]
    lab_d, _ = cluster_event_matrix(Xe[:, keep], score_e, 4)
    bt_d = block_table(lab_d, failure)
    variants["drop_terminal_features"] = dict(
        dropped=len(names) - len(keep), kept=len(keep),
        ari_vs_published=round(float(adjusted_rand_score(pub_labels, lab_d)), 4),
        blocks=bt_d, best_f1=max(bt_d, key=lambda r: r["f1"]),
        best_jaccard_vs_event_C0=max(jac(lab_d == r["block"], core["event_c0"]) for r in bt_d))
    # (iv) drop only late_stasis_indicator
    keep1 = [j for j in range(len(names)) if names[j] != "late_stasis_indicator"]
    lab_1, _ = cluster_event_matrix(Xe[:, keep1], score_e, 4)
    bt_1 = block_table(lab_1, failure)
    variants["drop_late_stasis_indicator_only"] = dict(
        ari_vs_published=round(float(adjusted_rand_score(pub_labels, lab_1)), 4),
        best_f1=max(bt_1, key=lambda r: r["f1"]),
        best_jaccard_vs_event_C0=max(jac(lab_1 == r["block"], core["event_c0"]) for r in bt_1))
    # (v) residualize every column on (task one-hot x natural cubic spline of T)
    from sklearn.linear_model import LinearRegression
    tasks_u = sorted(set(task))
    tid = np.array([tasks_u.index(t) for t in task])
    oh = np.eye(len(tasks_u))[tid]
    Z = np.c_[oh, oh * length[:, None], oh * (length ** 2)[:, None]]
    resid = Xe - LinearRegression().fit(Z, Xe).predict(Z)
    lab_r, _ = cluster_event_matrix(resid, score_e, 4)
    bt_r = block_table(lab_r, failure)
    variants["residualized_on_task_x_length"] = dict(
        ari_vs_published=round(float(adjusted_rand_score(pub_labels, lab_r)), 4),
        blocks=bt_r, best_f1=max(bt_r, key=lambda r: r["f1"]),
        best_precision=max([r for r in bt_r if r["n"] >= 100] or bt_r,
                           key=lambda r: (r["precision"], r["n"])),
        best_jaccard_vs_event_C0=max(jac(lab_r == r["block"], core["event_c0"]) for r in bt_r),
        ami_outcome=round(float(adjusted_mutual_info_score(failure.astype(int), lab_r)), 4),
        ami_length=round(float(adjusted_mutual_info_score(length, lab_r)), 4))
    result["event_clustering_variants"] = variants

    # ------------------------------------------------------------------ lag
    rep = np.load(BASE / "alternative-routing-organizations/representation_features.npz",
                  allow_pickle=True)
    Xg = np.asarray(rep["lag_spectrum"], dtype=np.float64)
    resid_g = Xg - LinearRegression().fit(Z, Xg).predict(Z)
    score_g = np.asarray(cache["diagnostic_mean_soft_speed"])
    lag_out = {}
    for nm, M in (("real", Xg), ("residualized_on_task_x_length", resid_g)):
        emb = prepare_embedding(M, SEED)
        lab = canonicalize(ward_labels(emb, 6), score_g)
        bt = block_table(lab, failure)
        lag_out[nm] = dict(best_f1=max(bt, key=lambda r: r["f1"]),
                           best_jaccard_vs_lag_C7=max(jac(lab == r["block"], core["lag_c7"])
                                                      for r in bt),
                           ami_outcome=round(float(adjusted_mutual_info_score(
                               failure.astype(int), lab)), 4),
                           ami_length=round(float(adjusted_mutual_info_score(length, lab)), 4))
    result["lag_spectrum_ward_k6"] = lag_out

    (OUT / "event_and_lag.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "event_feature_within_task_correlations_scene8"},
                     indent=2, ensure_ascii=False)[:6000])
    print("\ntop 15 event features by |rho(failure)| within scene8:")
    for r in result["event_feature_within_task_correlations_scene8"][:15]:
        print(f"  {r['feature']:34s} rho_T={r['rho_length']:+.3f} rho_fail={r['rho_failure']:+.3f}"
              f" gap={r['gap']:+.3f} terminal={r['terminal_window']}")


if __name__ == "__main__":
    main()
