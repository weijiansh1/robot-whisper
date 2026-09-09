"""Invariants the accounting must satisfy.  Run with ``python -m pytest tests``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import ledger_core as lc  # noqa: E402
import synth_core as sc  # noqa: E402

R = sc.RESULTS


@pytest.fixture(scope="module")
def cohorts():
    return sc.load_cohort("development_main"), sc.load_cohort("external_8b")


def test_cohort_shapes_and_risk_counts(cohorts):
    dev, ext = cohorts
    assert (dev.n, int(dev.risk.sum())) == (14800, 487)
    assert (ext.n, int(ext.risk.sum())) == (15600, 564)


def test_shared_detector_count(cohorts):
    dev, ext = cohorts
    shared = sc.shared_detectors(dev, ext)
    assert len(shared) == 419
    assert len([s for s in shared if not s.startswith("moe-v7")]) == 414


def test_physical_mode_join_is_total_and_exclusive(cohorts):
    """Every risk carries a physical failure mode and no safe episode does."""
    for c in cohorts:
        assert (c.mode[c.risk] != "").all()
        assert (c.mode[~c.risk] == "").all()


def test_length_is_a_negative_control_not_a_baseline(cohorts):
    """Risk is defined as not finishing before the cap, so an alarm on
    length-at-cap recalls every risk by construction."""
    for c in cohorts:
        alarm = sc.negative_control_length(c) >= 0
        assert alarm[c.risk].all()
    census = pd.read_csv(R / "detector_census.csv")
    neg = census[census.bundle == "_negative_control"]
    assert len(neg) == 2
    assert (neg.recall == 1.0).all()
    assert (~neg.is_baseline.astype(bool)).all()


def test_in_window_deadline(cohorts):
    for c in cohorts:
        for suite, cap in sc.CAP.items():
            m = c.suite == suite
            if m.any():
                assert np.allclose(c.deadline[m], 0.65 * cap)


def test_family_partition_covers_every_live_detector():
    z = np.load(R / "family_labels.npz", allow_pickle=False)
    assert len(z["names_dev"]) == len(z["labels_dev"])
    fams = json.loads((R / "families.json").read_text())
    assert fams["development_shared419"]["n_families_headline"] == len(
        set(z["labels_dev"].tolist()))
    # the rate-matched null must not reproduce the family structure
    assert (fams["development_shared419"]["n_families_null_headline"]
            > 10 * fams["development_shared419"]["n_families_headline"])


def test_family_representatives_respect_the_development_budget(cohorts):
    dev, _ = cohorts
    names, labels = lc.load_family_labels()
    for budget in (0.005, lc.REFERENCE_BUDGET, 0.02):
        reps = lc.representatives(dev, names, labels, budget)
        assert (reps.dev_fpr <= budget + 1e-12).all()
        assert reps.family.is_unique


def test_selection_never_used_external_outcomes(cohorts):
    """Representatives are a function of development only: recomputing them
    after permuting the external labels must give the same detectors."""
    dev, ext = cohorts
    names, labels = lc.load_family_labels()
    a = lc.representatives(dev, names, labels, lc.REFERENCE_BUDGET)
    rng = np.random.default_rng(0)
    ext.risk = rng.permutation(ext.risk)
    b = lc.representatives(dev, names, labels, lc.REFERENCE_BUDGET)
    assert a.detector.tolist() == b.detector.tolist()


def test_vote_decomposition_is_monotone_in_k():
    dec = pd.read_csv(R / "vote_decomposition.csv")
    d = dec[dec.suite != "_base"]
    for (_, _, _), g in d.groupby(["arm", "cohort", "suite"]):
        g = g.sort_values("k")
        assert (g.tp.diff().dropna() <= 0).all()
        assert (g.fp.diff().dropna() <= 0).all()


def test_union_recall_is_monotone_in_budget():
    cl = pd.read_csv(R / "coverage_ledger.csv")
    for coh, g in cl.groupby("cohort"):
        g = g.sort_values("budget")
        assert (g.ceiling_recall_all_admissible.diff().dropna() >= -1e-9).all()


def test_oracle_ceiling_is_not_beaten_by_an_honest_arm():
    """A ceiling an honest arm can beat is not a ceiling."""
    gap = pd.read_csv(R / "honesty_gap.csv")
    g = gap.dropna(subset=["oracle419_upper"])
    assert (g.honesty_gap_419_upper >= 0).all()
    assert (g.honesty_gap_622_upper >= 0).all()


def test_null_arm_is_far_below_the_observed_ceiling():
    """Only asserted inside the operating range.  At a 5,000 false alarm budget
    a rate-matched random bank flags a third of the corpus and picks up risks by
    volume alone, so the margin narrows there by construction; that is a fact
    about the budget, not about the detectors."""
    c = pd.read_csv(R / "headroom_ceiling.csv")
    obs = c[c.arm == "ORACLE_in_sample_419"].set_index("ext_fp_target").best_tp
    nul = c[c.arm == "NULL_rate_matched_419"].set_index("ext_fp_target").best_tp
    inrange = obs.index <= 2000
    assert (obs[inrange] > 2 * nul[inrange]).all()
    assert (obs > nul).all()


def test_key_numbers_match_the_tables():
    k = json.loads((R / "key_numbers.json").read_text())
    cl = pd.read_csv(R / "coverage_ledger.csv")
    row = cl[(cl.cohort == "external_8b") & (cl.budget == lc.REFERENCE_BUDGET)].iloc[0]
    ref = [r for r in k["coverage"]["ladder_external"]
           if r["budget"] == lc.REFERENCE_BUDGET][0]
    assert ref["n_uncaught"] == row.n_uncaught
    assert abs(ref["union_recall"] - row.union_recall) < 1e-12
