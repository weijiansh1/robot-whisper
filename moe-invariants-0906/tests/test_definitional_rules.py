"""The definitional filter is what separates a finding from a tautology, so it
gets its own tests.  Each case below is justified by the extractor source, not
by the data - see the module docstring of experiments/phase1_definitional.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))
from phase1_definitional import canonical, classify, family, is_forced  # noqa: E402

CASES = [
    # the known identity, exact in the code
    (["sp.load_entropy@L4[s0]", "sp.token_entropy@L4[s0]",
      "sp.token_differentiation@L4[s0]"], "definitional:entropy_identity"),
    # the same identity with one term taken at another denoising step
    (["sp.token_entropy@L14[s0]", "sp.load_entropy@L14[s0]",
      "sp.token_differentiation@L14[mean]"], "definitional:up_to_step_reduction"),
    # layer_graphs geometry IS step_profiles at step 9
    (["sp.conditional_energy@L3[s9]", "lg.conditional_energy@L3[agg]"],
     "cache_duplicate"),
    # flow_path = 9 * mean flow_speed
    (["sp.flow_speed@L2[mean]", "lg.flow_path@L2[agg]"], "cache_duplicate"),
    # the second-order Taylor pair
    (["sp.token_differentiation@L12[s0]", "sp.action_consensus@L12[s0]"],
     "definitional:smooth_equivalent"),
    # the Jensen envelope
    (["sp.conditional_energy@L4[mean]", "sp.state_action_alignment@L4[mean]"],
     "definitional:smooth_equivalent"),
    # X[mean] literally contains X[s9]
    (["sp.token_entropy@L3[mean]", "sp.token_entropy@L3[s9]"],
     "overlap:mean_contains_component"),
    # the two relations we actually want to be able to see
    (["sp.token_entropy@L3[s0]", "sp.token_entropy@L14[s0]"],
     "empirical:same_metric"),
    (["sp.token_entropy@L3[s0]", "sp.token_entropy@L3[s9]"],
     "empirical:same_metric"),
    (["sp.conditional_effective_rank@L5[s9]", "sp.flow_speed@L5[mean]"],
     "empirical:cross_metric"),
    # set_dwell is 95% zeros, quarantined
    (["ch.set_dwell@L4[agg]", "sp.token_entropy@L4[mean]"],
     "degenerate:constant_like"),
]


@pytest.mark.parametrize("names,expected", CASES,
                         ids=[c[1] + "|" + c[0][0] for c in CASES])
def test_classification(names, expected):
    assert classify(names) == expected


def test_state_mobility_reductions_collapse():
    a = canonical("mob.state_mobility@L2[s0]")
    b = canonical("mob.state_mobility@L2[s9]")
    assert a == b
    assert classify(["mob.state_mobility@L2[s0]",
                     "mob.state_mobility@L2[s9]"]) == "cache_duplicate"


def test_families_merge_only_what_the_code_forces():
    assert family("lg.flow_path@L2[agg]") == family("sp.flow_speed@L2[mean]")
    assert family("ch.hb_entropy_action@L5[agg]") == \
        family("sp.token_entropy@L5[mean]")
    # different layers must never merge
    assert family("sp.token_entropy@L2[mean]") != family("sp.token_entropy@L3[mean]")
    # unrelated functionals must never merge
    assert family("sp.flow_speed@L2[mean]") != family("mob.mobility@L2[mean]")


def test_is_forced_covers_every_nonempirical_label():
    for names, label in CASES:
        assert is_forced(label) == (not label.startswith("empirical")), label


def test_controls_confirm_the_rules_numerically():
    """The rules above are claims about the code; the control run is the
    numerical audit of those claims."""
    for cohort in ("development_main", "external_8b"):
        p = BUNDLE / "results" / f"phase1_controls_{cohort}.json"
        if not p.exists():
            pytest.skip("controls not run")
        d = json.loads(p.read_text())
        # E1 exact
        assert d["summary"]["C1_max_abs_residual_any_layer"] == 0.0
        # E2 exact to float32
        for lay in d["per_layer"].values():
            for m in lay["C2_layergraph_is_step9"].values():
                assert m["max_abs_delta"] < 1e-4
            # E3 exact
            assert lay["C3_flow_path"]["rel_resid"] < 1e-3
            # E6 is an envelope, never violated
            assert lay["C6_cond_energy_envelope"]["envelope_never_violated"]
        # A1 predicted slope 9/5
        lo, hi = d["summary"]["C7_fitted_slope_range"]
        assert 1.79 < lo < 1.81, lo
        assert hi < 1.95, hi
