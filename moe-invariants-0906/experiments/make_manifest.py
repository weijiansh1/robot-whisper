"""Collect every number the report quotes into one file, and write the
manifest.  The report must not contain a figure that is not in here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
OUT = BUNDLE / "results"
sys.path.insert(0, str(HERE))
from phase1_definitional import family  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    rel = json.loads((OUT / "phase1_relations.json").read_text())
    fam = json.loads((OUT / "phase1_families.json").read_text())
    nlin = json.loads((OUT / "phase1_nonlinear.json").read_text())
    frozen = json.loads((OUT / "phase1_frozen_candidates.json").read_text())
    freeze = json.loads((OUT / "phase1_freeze.json").read_text())
    ph2 = json.loads((OUT / "phase2_breach_summary.json").read_text())
    desc = json.loads((OUT / "top_invariant_description.json").read_text())
    ctl = {c: json.loads((OUT / f"phase1_controls_{c}.json").read_text())
           for c in ("development_main", "external_8b")}
    bank = json.loads((OUT / "bank_summary.json").read_text())

    cands = frozen["candidates"]
    pc_ranks = [i + 1 for i, c in enumerate(cands)
                if c["category"] == "definitional:entropy_identity"]
    emp = [c for c in cands if c["category"].startswith("empirical")]

    # tightest distinct empirical invariant per definitional family pair
    seen, top_emp = set(), []
    for c in emp:
        k = tuple(sorted([family(c["target"])] + [family(p) for p in c["preds"]]))
        if k in seen:
            continue
        seen.add(k)
        top_emp.append({
            "rank_overall": cands.index(c) + 1,
            "target": c["target"], "preds": c["preds"],
            "category": c["category"], "source": c["source"],
            "rel_resid_dev_main": c["rel_resid_dev_main"],
            "coefs_raw_dev_main": c["coefs_raw_dev_main"],
            "intercept_dev_main": c["intercept_dev_main"],
            "rel_resid_transported_external_8b":
                c["stability"]["rel_resid_transported_external_8b"],
            "rel_resid_transported_development_extra":
                c["stability"]["rel_resid_transported_development_extra"],
            "worst_rel_resid_any_task": c["stability"]["worst_rel_resid_any_task"],
            "n_task_strata": c["stability"]["n_task_strata"],
            "coef_cv_across_tasks": c["stability"]["coef_cv_across_tasks"],
            "coef_cv_across_suites": c["stability"]["coef_cv_across_suites"],
            "coef_cv_across_cohorts": c["stability"]["coef_cv_across_cohorts"],
            "max_angle_deg_across_tasks": c["stability"]["max_angle_deg_across_tasks"],
            "null_cell": c.get("null_cell") or c.get("null_cell_nl"),
            "null_epi": c.get("null_epi") or c.get("null_epi_nl"),
        })
        if len(top_emp) >= 10:
            break

    info = json.loads((OUT / "bank" /
                       "development_main_within_episode_info.json").read_text())
    eta = {d["var"]: d for d in info}
    used_vars = sorted({v for c in top_emp for v in [c["target"]] + c["preds"]})

    drift = {"definition": "std(x[step9] - x[step0]) / std(x); no coefficient "
                           "is fitted, so this is the conserved-quantity form "
                           "of the top frozen relation"}
    for coh in ("development_main", "development_extra", "external_8b"):
        drift[coh] = {m: {L: desc[coh][m][L]["unfitted_rel_drift_std"]
                          for L in desc[coh][m]} for m in desc[coh]}

    summary = {
        "bundle": "moe-invariants-0906",
        "bank": {c: {k: bank[c][k] for k in ("rows", "n_vars", "n_episodes", "n_tasks")}
                 for c in bank},
        "n_canonical_vars": rel["n_canonical_vars"],
        "n_family_vars": rel["n_family_vars"],
        "null_floors": {
            "linear_pair": rel["pair_null_floor"],
            "linear_pair_nondefinitional": rel["pair_null_floor_nondefinitional"],
            "nonlinear_pair": nlin["floor_nondefinitional_pairs"],
            "nonlinear_resolution_floor": nlin["resolution_floor"],
        },
        "surrogate_check": rel["shuffle_check"],
        "negative_control_random_sets": rel["negative_control"],
        "positive_control": {
            **{k: rel["positive_control"][k] for k in
               ("worst_rel_resid", "worst_oos_external", "raw_coef_min",
                "raw_coef_max", "best_rank_in_free_triple_search",
                "n_in_top100_of_free_triple_search")},
            "ranks_in_frozen_list": pc_ranks,
            "n_empirical_candidates_above_it": sum(1 for r in pc_ranks if r > 16),
            "raw_max_abs_residual_development_main":
                ctl["development_main"]["summary"]["C1_max_abs_residual_any_layer"],
            "raw_max_abs_residual_external_8b":
                ctl["external_8b"]["summary"]["C1_max_abs_residual_any_layer"],
        },
        "definitional_audit": {c: ctl[c]["summary"] for c in ctl},
        "taylor_slope_per_layer": {
            c: {L: v["C7_taylor_slope"]["fitted_slope"]
                for L, v in ctl[c]["per_layer"].items()} for c in ctl},
        "state_mobility_step_invariance": {
            c: {L: v["C8_state_mobility_step_invariant"]["spread_over_level"]
                for L, v in ctl[c]["per_layer"].items()} for c in ctl},
        "top_empirical_invariants": top_emp,
        "within_episode_information": {v: {
            "eta2_between_episode": eta[v]["eta2_between_episode"],
            "n_unique": eta[v]["n_unique"], "std": eta[v]["std"]}
            for v in used_vars},
        "set_dwell": {v: {"eta2_between_episode": eta[v]["eta2_between_episode"],
                          "n_unique": eta[v]["n_unique"]}
                      for v in eta if "set_dwell" in v},
        "F1_cross_layer_top": fam["F1_cross_layer"][:5],
        "F2_front_back_token_entropy_top": fam["F2_front_back_token_entropy"][:5],
        "F2_front_back_overall_top": fam["F2_front_back"][:3],
        "F3_per_layer_no_entropy_trio_top": [
            r for r in fam["F3_per_layer"]
            if r["axis"] == "functional_no_entropy_trio"][:5],
        "F4_step_axis_top20": fam["F4_step_axis"]["ranked"][:20],
        "F4_mobility_L2": [r for r in fam["F4_step_axis"]["ranked"]
                           if r["metric"] == "mobility" and r["layer"] == "L2"],
        "F5_spectrum": fam["F5_spectrum"],
        "nonlinear_top_cross_metric": [
            r for r in nlin["top_nondefinitional"]
            if r["category"] == "empirical:cross_metric"][:6],
        "unfitted_denoising_drift": drift,
        "phase2": ph2,
        "freeze": freeze,
    }
    (OUT / "summary_for_report.json").write_text(json.dumps(summary, indent=1))

    files = sorted(p for p in OUT.rglob("*")
                   if p.is_file() and not p.name.startswith("_")
                   and p.name != "manifest.json")
    manifest = {
        "schema": "himoe.invariants.manifest.v1",
        "bundle": "moe-invariants-0906",
        "question": "Are there near-invariant relations among MoE routing "
                    "functionals?  Ranked by tightness, not by discrimination.",
        "protocol": {
            "phase1": "label-blind search, frozen and hashed before phase 2",
            "phase2": "breach of the frozen invariants, cap-free scoring",
            "freeze_sha256": freeze["sha256"],
        },
        "inputs": {
            "moe-flow-semantics-0906/results/step_profiles": "{cohort}_metrics.npy "
                "[n,52,8,10,8], {cohort}_mobility.npy, {cohort}_state_mobility.npy",
            "moe-unused-channels-0906/results/channels": "{cohort}_quantities.npy "
                "[n,52,8,8]",
            "moe-hb-front-back-0905/results/layer_graphs": "{cohort}.npz "
                "metrics [n,52,8,11]",
            "moe-capfree-0906/experiments/capfree_protocol.py": "imported unchanged "
                "for phase-2 scoring",
        },
        "experiments": sorted(p.name for p in (BUNDLE / "experiments").glob("*.py")),
        "tests": sorted(p.name for p in (BUNDLE / "tests").glob("*.py")),
        "results": [{"path": str(p.relative_to(BUNDLE)),
                     "bytes": p.stat().st_size, "sha256": sha(p)} for p in files],
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "device": "cpu",
        },
        "headline": {
            "positive_control_rank": pc_ranks[0],
            "positive_control_rel_resid": rel["positive_control"]["worst_rel_resid"],
            "tightest_non_definitional_relation": {
                "statement": top_emp[0]["target"] + " ~ " + "+".join(top_emp[0]["preds"]),
                "rel_resid_dev_main": top_emp[0]["rel_resid_dev_main"],
                "rel_resid_transported_external_8b":
                    top_emp[0]["rel_resid_transported_external_8b"],
                "surrogate_floor_for_that_pair": top_emp[0]["null_cell"],
            },
            "phase2_verdict": ph2["external_once"],
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("manifest:", len(manifest["results"]), "result files")
    print("freeze:", freeze["sha256"])


if __name__ == "__main__":
    main()
