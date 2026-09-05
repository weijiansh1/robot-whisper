#!/usr/bin/env python3
"""Export one query as a typed MoEAssuranceState JSON object."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "assurance"))

from profile_schema import AssuranceProfile  # noqa: E402
from state_types import MoEAssuranceState, OutcomeEstimate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=("main16x32", "grid50x8"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--query", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    args = parse_args()
    profile_path = BUNDLE / "results/profiles" / args.corpus / args.suite / f"{args.task}.npz"
    score_path = BUNDLE / "results/routes_1_4/scores" / args.corpus / args.suite / f"{args.task}.npz"
    posterior_path = BUNDLE / "results/route_7/posteriors" / args.corpus / args.suite / f"{args.task}.npz"
    profile = AssuranceProfile.load(profile_path)
    matched = np.flatnonzero(
        (profile.episode_id == args.episode) & (profile.query == args.query)
    )
    if len(matched) != 1:
        raise ValueError(f"query lookup returned {len(matched)} rows")
    row = int(matched[0])

    with np.load(score_path, allow_pickle=False) as archive:
        score_names = archive["score_names"].astype(str).tolist()
        score_values = np.asarray(archive["scores"][row], dtype=np.float64)
    scores = dict(zip(score_names, score_values))
    with np.load(posterior_path, allow_pickle=False) as archive:
        states = archive["state_names"].astype(str).tolist()
        belief = np.asarray(archive["posterior"][row], dtype=np.float64)

    layer_evidence = {}
    for layer_index, layer_name in enumerate(("L12", "L13", "L14", "L15")):
        layer_evidence[layer_name] = {
            name: float(profile.layer_features[row, layer_index, feature_index])
            for feature_index, name in enumerate(profile.layer_feature_names)
        }

    outcome_assurance = []
    if args.query == 0:
        tensor = pd.read_csv(BUNDLE / "results/route_8/assurance_tensor.csv")
        scene = int(profile.scene[row])
        selected = tensor[
            (tensor.source == "q0_committor")
            & (tensor.corpus == args.corpus)
            & (tensor.suite == args.suite)
            & (tensor.task == args.task)
            & (tensor.scene == scene)
        ]
        for item in selected.itertuples(index=False):
            outcome_assurance.append(
                OutcomeEstimate(
                    intervention=item.intervention,
                    horizon=item.horizon,
                    outcome=item.outcome,
                    posterior_mean=float(item.posterior_mean),
                    ci95_low=float(item.ci95_low),
                    ci95_high=float(item.ci95_high),
                    support=int(item.total),
                    posterior=item.posterior,
                )
            )

    recurrence = [finite_or_none(value) for value in profile.recurrence_distance[row]]
    unavailable = []
    if not outcome_assurance:
        unavailable.append("outcome probability at this query has no repeated physical forks")
    unavailable.extend(
        [
            "per-query infinitesimal perturbation robustness",
            "runtime-exact expert-output disagreement",
        ]
    )
    state = MoEAssuranceState(
        corpus=args.corpus,
        suite=args.suite,
        task=args.task,
        episode_id=args.episode,
        query=args.query,
        commitment=float(scores["r1_commitment"]),
        flow_coherence=float(1.0 - scores["r2_flow_instability"]),
        token_consensus=float(scores["r3_token_consensus"]),
        effective_rank=float(scores["r3_effective_rank"]),
        temporal_recurrence=recurrence,
        healthy_energy=float(scores["r4_healthy_energy"]),
        static_lockin_evidence=finite_or_none(scores["r3_static_lockin"]),
        layer_evidence=layer_evidence,
        mode_belief=dict(zip(states, belief.tolist())),
        outcome_assurance=outcome_assurance,
        unavailable=unavailable,
    )
    text = json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
