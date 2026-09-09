"""Assemble every number the report cites into one JSON, read back from the
result tables, so nothing in the write-up is typed by hand."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ledger_core as lc
import synth_core as sc

R = sc.RESULTS
REF = lc.REFERENCE_BUDGET


def j(path):
    return json.loads((R / path).read_text())


def main() -> None:
    census = j("census.json")
    fams = j("families.json")
    cov = j("coverage.json")
    blind = j("blindspot.json")

    cl = pd.read_csv(R / "coverage_ledger.csv")
    prof = pd.read_csv(R / "uncaught_profile.csv")
    sat = pd.read_csv(R / "saturation_curve.csv")
    sole = pd.read_csv(R / "sole_catcher.csv")
    ov = pd.read_csv(R / "alarm_overlap.csv")
    dec = pd.read_csv(R / "vote_decomposition.csv")
    fpp = pd.read_csv(R / "fp_profile.csv")
    irr = pd.read_csv(R / "irreducible_fp_tests.csv")
    rc = pd.read_csv(R / "uncaught_routing_contrast.csv")
    ceil = pd.read_csv(R / "headroom_ceiling.csv")
    hon = pd.read_csv(R / "honest_operating_points.csv")
    gap = pd.read_csv(R / "honesty_gap.csv")
    bm = pd.read_csv(R / "blindspot_matrix.csv")
    fam_tbl = pd.read_csv(R / "families.csv")
    cutp = pd.read_csv(R / "family_cut_profile.csv")

    tag = f"families@{REF}"
    out: dict = {"schema": "himoe.method_synthesis.key_numbers.v1",
                 "reference_per_detector_budget": REF}

    out["census"] = {
        "external_detectors": census["external_8b"]["n_detectors"],
        "development_detectors": census["development_main"]["n_detectors"],
        "shared_detectors": census["n_shared_detectors"],
        "shared_excluding_v7": census["n_shared_excluding_v7"],
        "external_distinct_alarm_vectors": census["external_8b"]["n_distinct_alarm_vectors"],
        "development_distinct_alarm_vectors": census["development_main"]["n_distinct_alarm_vectors"],
        "external_detectors_that_duplicate_another":
            census["external_8b"]["n_detectors_in_duplicate_groups"],
        "development_detectors_that_duplicate_another":
            census["development_main"]["n_detectors_in_duplicate_groups"],
        "negative_control_length_recall": census["external_8b"]["negative_control_recall"],
        "negative_control_length_fp": census["external_8b"]["negative_control_fp"],
        "external_risks": census["external_8b"]["n_risk"],
        "development_risks": census["development_main"]["n_risk"],
    }

    out["families"] = {
        "n_families_development_419": fams["development_shared419"]["n_families_headline"],
        "n_families_external_419": fams["external_shared419"]["n_families_headline"],
        "n_families_null_419": fams["development_shared419"]["n_families_null_headline"],
        "n_named_quantities_419": fams["development_shared419"]["n_named_quantities"],
        "n_families_external_622": fams["external_all622"]["n_families_headline"],
        "n_named_quantities_622": fams["external_all622"]["n_named_quantities"],
        "n_families_null_622": fams["external_all622"]["n_families_null_headline"],
        "n_singletons_622": fams["external_all622"]["n_singleton_families"],
        "replication_ari": fams["family_replication_dev_to_ext"]["adjusted_rand_index"],
        "replication_ari_null": fams["family_replication_dev_to_ext"]["ari_null_permuted"],
        "effective_rank_419_dev": fams["development_shared419"]["effective_rank_observed"],
        "effective_rank_419_null": fams["development_shared419"]["effective_rank_null"],
        "cut_sensitivity_419_dev": cutp[(cutp.arm == "development_shared419")
                                        & (cutp.similarity == "excess_overlap")
                                        & (cutp.suite == "pooled")][
            ["cut", "n_families_observed", "n_families_null"]].to_dict("records"),
        "per_suite_419_dev": cutp[(cutp.arm == "development_shared419")
                                  & (cutp.suite != "pooled")][
            ["suite", "n_families_observed", "n_families_null"]].to_dict("records"),
        "development_family_table": fam_tbl[fam_tbl.arm == "development_shared419"][
            ["family", "size", "medoid", "n_roots", "roots", "best_tp",
             "fp_at_best_tp"]].to_dict("records"),
    }

    out["coverage"] = {
        "unconstrained_union": {
            "external": cov["external_8b_unconstrained_union"],
            "development": cov["development_main_unconstrained_union"],
        },
        "ladder_external": cl[cl.cohort == "external_8b"][
            ["budget", "union_recall", "n_uncaught", "union_fp", "union_fpr",
             "union_precision", "n_caught_by_exactly_one", "n_admissible_detectors",
             "ceiling_recall_all_admissible", "ceiling_fp_all_admissible",
             "ceiling_n_uncaught"]].to_dict("records"),
        "ladder_development": cl[cl.cohort == "development_main"][
            ["budget", "union_recall", "n_uncaught", "union_fp", "union_fpr",
             "ceiling_recall_all_admissible", "ceiling_n_uncaught"]].to_dict("records"),
        "sole_catcher_external_at_ref": sole[(sole.cohort == "external_8b")
                                             & (sole.budget == REF)][
            ["family", "n_sole_catches", "share_of_sole", "detector"]].to_dict("records"),
    }

    for coh in ("external_8b", "development_main"):
        p = prof[(prof.cohort == coh) & (prof.arm == tag)]
        out["coverage"][f"uncaught_{coh}"] = {
            "n_uncaught": int(p[p.field == "suite"].n_uncaught.sum()),
            "by_suite": p[p.field == "suite"][
                ["value", "n_risk", "n_uncaught", "uncaught_share",
                 "share_of_all_uncaught"]].to_dict("records"),
            "by_mode": p[p.field == "physical_mode"][
                ["value", "n_risk", "n_uncaught", "uncaught_share",
                 "share_of_all_uncaught"]].to_dict("records"),
            "mean_length_within_suite_uncaught_vs_caught": p[
                p.field == "mean_length_within_suite"][
                ["value", "uncaught_share", "share_of_all_uncaught"]].to_dict("records"),
            "top_tasks": p[p.field == "task"].nlargest(6, "n_uncaught")[
                ["value", "n_risk", "n_uncaught", "uncaught_share"]].to_dict("records"),
        }

    out["coverage"]["routing_contrast_external"] = rc[rc.cohort == "external_8b"][
        ["layer", "contrast", "n_a", "n_b", "mean_a", "mean_b", "difference",
         "null_mean", "z_vs_stratified_null", "p_two_sided"]].to_dict("records")

    for arm in ("family_representatives", "all_admissible_detectors"):
        s = sat[(sat.arm == arm) & (sat.budget == REF)]
        out.setdefault("saturation", {})[arm] = s[
            ["step", "added", "dev_recall", "dev_fp", "ext_recall", "ext_fp",
             "ext_marginal_tp", "ext_marginal_fp"]].to_dict("records")
    s = sat[(sat.arm == "all_admissible_detectors") & (sat.budget == REF)]
    first5 = s[s.step <= 5]
    rest = s[s.step > 5]
    out["saturation"]["exchange_rate"] = {
        "first5_ext_tp": int(first5.ext_marginal_tp.sum()),
        "first5_ext_fp": int(first5.ext_marginal_fp.sum()),
        "first5_fp_per_tp": float(first5.ext_marginal_fp.sum() / max(first5.ext_marginal_tp.sum(), 1)),
        "beyond5_ext_tp": int(rest.ext_marginal_tp.sum()),
        "beyond5_ext_fp": int(rest.ext_marginal_fp.sum()),
        "beyond5_fp_per_tp": float(rest.ext_marginal_fp.sum() / max(rest.ext_marginal_tp.sum(), 1)),
    }

    out["false_alarms"] = {
        "overlap_external": ov[(ov.arm == tag) & (ov.cohort == "external_8b")][
            ["suite", "partition", "n_flagged_union", "concentration",
             "concentration_null", "concentration_null_within",
             "share_flagged_by_exactly_one"]].to_dict("records"),
        "overlap_development": ov[(ov.arm == tag) & (ov.cohort == "development_main")][
            ["suite", "partition", "n_flagged_union", "concentration",
             "concentration_null_within", "share_flagged_by_exactly_one"]].to_dict("records"),
        "overlap_greedy_top5_external": ov[(ov.arm == f"greedy_top5@{REF}")
                                           & (ov.cohort == "external_8b")][
            ["suite", "partition", "concentration", "concentration_null_within",
             "share_flagged_by_exactly_one"]].to_dict("records"),
        "vote_decomposition_external": dec[(dec.arm == tag)
                                           & (dec.cohort == "external_8b")
                                           & (dec.k <= 3)][
            ["suite", "k", "tp", "fp", "recall", "fpr", "precision",
             "fp_removed_vs_k1", "tp_lost_vs_k1", "fp_removed_per_tp_lost"]].to_dict("records"),
        "profile_external": fpp[(fpp.arm == tag) & (fpp.cohort == "external_8b")][
            ["group", "n", "share_of_safe", "mean_length_fraction_of_cap",
             "mean_length_fraction_all_safe", "top_suite", "top_suite_share",
             "top_task", "top_task_share", "n_distinct_tasks"]].to_dict("records"),
        "stratified_tests": irr.to_dict("records"),
    }

    out["blind_spots"] = {
        "modes_external": blind["modes"]["external_8b"],
        "modes_development": blind["modes"]["development_main"],
        "family_x_mode_external_pooled": bm[(bm.cohort == "external_8b")
                                            & (bm.suite == "pooled")][
            ["method", "mode", "n_risk_in_cell", "recall_in_cell"]].to_dict("records"),
        "systematic_misses_external": bm[(bm.cohort == "external_8b")
                                         & (bm.suite != "pooled")
                                         & (bm.p_lower <= 0.05)
                                         & (bm.n_risk_in_cell >= 10)][
            ["method", "suite", "mode", "n_risk_in_cell", "recall_in_cell",
             "recall_on_suite", "p_lower"]].to_dict("records"),
    }

    out["headroom"] = {
        "standing_honest_arm": lc.HONEST_ARM,
        "ceiling": ceil[["arm", "ext_fp_target", "single_tp", "or_pair_tp",
                         "best_k", "best_tp", "best_fp"]].to_dict("records"),
        "honest_points": hon[["arm", "n_methods", "dev_tp", "dev_fp", "ext_tp",
                              "ext_fp", "ext_recall", "ext_fpr",
                              "ext_precision"]].to_dict("records"),
        "gap": gap.to_dict("records"),
    }

    pairs = pd.read_csv(R / "family_pair_complementarity.csv")
    pe = pairs[(pairs.cohort == "external_8b")]
    out["complementarity"] = {
        "median_tp_jaccard_by_suite": pe.groupby("suite").tp_jaccard.median().to_dict(),
        "median_fp_jaccard_by_suite": pe.groupby("suite").fp_jaccard.median().to_dict(),
        "best_pairs_pooled": pe[pe.suite == "pooled"].nlargest(
            5, "exchange_rate_tp_per_fp")[
            ["family_a", "family_b", "tp_jaccard", "fp_jaccard", "union_tp",
             "union_fp", "marginal_tp_over_better", "marginal_fp_over_better",
             "exchange_rate_tp_per_fp"]].to_dict("records"),
    }
    nullc = ceil[ceil.arm == "NULL_rate_matched_419"]
    out["complementarity"]["null_ceiling"] = nullc[
        ["ext_fp_target", "single_tp", "or_pair_tp", "best_tp"]].to_dict("records")

    (R / "key_numbers.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {R / 'key_numbers.json'}")
    print(json.dumps(out["headroom"]["gap"][4:8], indent=2, default=float))


if __name__ == "__main__":
    main()
