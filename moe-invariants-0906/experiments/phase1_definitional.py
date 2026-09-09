"""Which relations are implied by the DEFINITIONS of the functionals.

Every rule below was established by reading the three extractor scripts, not by
looking at the data:

  moe-flow-semantics-0906/experiments/extract_flow_steps.py
  moe-unused-channels-0906/experiments/extract_discrete_channels.py
  moe-hb-front-back-0905/experiments/extract_layer_graphs_gpu.py

Notation: p_{q,l,s,t} is the 32-way router distribution, u = sqrt(p) a unit
vector, BC(a,b) = <u_a,u_b>, s_t = BC(token0, token t), G_tu = BC(t,u),
C = G - s s^T.  Token 0 is state, tokens 1..10 action.

EXACT identities in the code
  E1 token_differentiation := load_entropy - token_entropy      (literal
     subtraction, extract_flow_steps.py).  H(mean_t p_t) = mean_t H(p_t) + JSD.
  E2 layer_graphs {action_consensus, state_action_alignment, conditional_energy,
     conditional_effective_rank} are step_profiles' same metrics evaluated at
     denoising step 9 only (`terminal = root[:, :, -1]`).
  E3 flow_path = sum_{s=1..9} flow_speed_s = 9 * mean_s flow_speed.
  E4 expert_load_effective_rank = exp(load_entropy@s9)/32  (monotone, and
     locally linear because load_entropy spans < 0.02 nat).
  E5 hb_entropy_action = mean_s token_entropy  (float16 logging precision).
  E6 G_tt = ||u_t||^2 = 1, so C_tt = 1 - s_t^2 and
     conditional_energy = 1 - mean_t s_t^2 = 1 - state_action_alignment^2
                          - Var_t(s_t).
     A deterministic envelope, not an identity: the gap is Var_t(s_t).
  E7 X[mean] = (1/10) sum_s X[s]; it literally contains X[s0] and X[s9].

ASYMPTOTIC identity (near-uniform router; holds because action_consensus
~ 0.999 and token_entropy ~ log 32 - 0.005)
  A1 Writing p_t = pbar + d_t and A_tu = sum_e d_te d_ue / pbar_e,
       token_differentiation ~ m/2,   action_consensus ~ 1 - (5/18) m,
     with m = mean_t A_tt, hence
       token_differentiation ~ (9/5) (1 - action_consensus).
     Both are second-order measures of the same token dispersion m, so their
     agreement is a Taylor identity, not an empirical law.  The predicted slope
     -9/5 = -1.8 is a falsifiable consequence and is checked in
     phase1_controls.py.

EMPIRICAL DEGENERACY (a fact about this model, not about the definitions, but
it collapses three bank columns into one)
  Dg1 state_mobility is constant across the 10 denoising steps to ~2e-3 of its
      own scale: the state token's routing does not move under denoising.
"""

from __future__ import annotations

GEOMETRY_AT_STEP9 = ("action_consensus", "state_action_alignment",
                     "conditional_energy", "conditional_effective_rank",
                     "partial_edge_std")


def parse(name: str) -> tuple[str, str, str, str]:
    """'sp.token_entropy@L2[mean]' -> ('sp', 'token_entropy', 'L2', 'mean')."""
    cache, rest = name.split(".", 1)
    metric, rest = rest.split("@", 1)
    layer, red = rest.split("[", 1)
    return cache, metric, layer, red.rstrip("]")


def canonical(name: str) -> str:
    """Collapse variables that the code makes the same number (E2-E5, Dg1)."""
    cache, metric, layer, red = parse(name)
    if cache == "lg":
        if metric in GEOMETRY_AT_STEP9:
            return f"sp.{metric}@{layer}[s9]"
        if metric == "flow_path":
            return f"sp.flow_speed@{layer}[mean]"
        if metric == "expert_load_effective_rank":
            return f"sp.load_entropy@{layer}[s9]"
    if cache == "ch" and metric == "hb_entropy_action":
        return f"sp.token_entropy@{layer}[mean]"
    if metric == "state_mobility":
        return f"mob.state_mobility@{layer}[stepinvariant]"
    return name


ENTROPY_TRIO = {"load_entropy", "token_entropy", "token_dispersion"}

# Functionals that the definitions make smooth, strictly monotone functions of
# one another over the range this cache occupies.  Substituting one for another
# cannot turn a definitional relation into an empirical one, so classification
# is done AFTER this substitution.  Merging is the conservative choice: it can
# only move a relation out of the headline, never into it.
#   action_consensus  <-> token_differentiation      (A1, slope -9/5)
#   conditional_energy <-> state_action_alignment    (E6, ce = 1 - sa^2 - Var s)
#   expert_load_effective_rank <-> load_entropy      (E4, exp()/32)
#   hb_entropy_action <-> token_entropy              (E5)
#   flow_path <-> flow_speed                         (E3)
FAMILY_OF_METRIC = {
    "action_consensus": "token_dispersion",
    "token_differentiation": "token_dispersion",
    "conditional_energy": "state_alignment",
    "state_action_alignment": "state_alignment",
    "expert_load_effective_rank": "load_entropy",
    "hb_entropy_action": "token_entropy",
    "flow_path": "flow_speed",
}


def family(name: str) -> tuple[str, str, str]:
    """(family metric, layer, step reduction) after smooth-equivalence merging."""
    _, metric, layer, red = parse(canonical(name))
    return FAMILY_OF_METRIC.get(metric, metric), layer, red


def classify(names: list[str], _recursed: bool = False) -> str:
    """Label a candidate relation over `names` (target first).

    Returns one of
      cache_duplicate                   two columns are the same number
      definitional:<rule>               implied by the extractor definitions
      degenerate:constant_like          built on a near-degenerate channel
      empirical:same_metric             one functional, different layer/step
      empirical:cross_metric            the interesting case
    """
    canon = [canonical(n) for n in names]
    if len(set(canon)) < len(canon):
        return "cache_duplicate"
    if any(parse(c)[1] == "set_dwell" for c in canon):
        return "degenerate:constant_like"

    fam = [family(n) for n in names]
    if len(set(fam)) < len(fam):
        return "definitional:smooth_equivalent"
    metrics = {f[0] for f in fam}
    layers = {f[1] for f in fam}
    reds = {f[2] for f in fam}

    if len(layers) == 1 and len(reds) == 1 and metrics == ENTROPY_TRIO:
        return "definitional:entropy_identity"
    # E7: X[mean] = (1/10) sum_s X[s] literally CONTAINS X[s0] and X[s9].  This
    # only forces r >= 1/sqrt(10) (rel_resid <= 0.949) if the ten steps were
    # independent, so a much smaller residual here is an empirical statement
    # about step coherence -- but it is not a clean two-functional relation, so
    # it is quarantined and the honest version is the F4 step-axis analysis on
    # non-overlapping reductions.
    if len(metrics) == 1 and len(layers) == 1 and "mean" in reds and len(reds) > 1:
        return "overlap:mean_contains_component"
    # A relation that becomes definitional once the step reduction is ignored
    # is the definitional relation with one term replaced by a step-shifted
    # copy of itself.  Its residual measures step coherence, not a new law.
    blind = [f"{c.split('[')[0]}[X]" for c in canon]
    if not _recursed and blind != canon and len(set(blind)) == len(blind):
        if classify(blind, _recursed=True).startswith("definitional"):
            return "definitional:up_to_step_reduction"
    # A relation whose backbone is already forced is not a new relation.  Any
    # forced pair OR forced triple inside the set is enough: an 8-term
    # combination that contains {load_entropy, token_entropy,
    # token_differentiation} has lam_min = 0 for that reason alone.
    if len(names) > 2:
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                if is_forced(classify([names[a], names[b]])):
                    return "definitional:contains_forced_pair"
        if len(names) > 3:
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    for c in range(b + 1, len(names)):
                        if is_forced(classify([names[a], names[b], names[c]])):
                            return "definitional:contains_forced_triple"
    if len(metrics) == 1:
        return "empirical:same_metric"
    return "empirical:cross_metric"


def is_forced(tag: str) -> bool:
    """True if the definitions (or a cache duplication, or a degenerate
    channel, or a literal containment) already account for the relation."""
    return tag.startswith(("definitional", "cache_duplicate", "degenerate",
                           "overlap"))


def is_definitional(tag: str) -> bool:
    return tag.startswith(("definitional", "cache_duplicate", "degenerate"))
