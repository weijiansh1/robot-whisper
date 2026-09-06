#!/usr/bin/env python3
"""Does the best circuit quantity beat its nearest published neighbour?

POST-HOC.  The pair was fixed by the label-free redundancy table (norm_fiedler
is 0.974 rank-correlated with partial_edge_std, its closest published
relative), but the decision to run this test was taken after the external AUC
table had been seen.  Reported on both cohorts.

Cluster bootstrap over episodes: resample episodes with replacement inside
each suite, recompute the survival-conditioned pooled AUC for both quantities
at a fixed layer, and take the paired difference of |AUC - 0.5|.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import protocol as P
from summarise_results import survival_auc


DEFAULT_OUTPUT = P.BUNDLE / "results/detectors"
PAIRS = (
    ("circuit", "norm_fiedler", "published", "partial_edge_std", "L12"),
    ("circuit", "norm_fiedler", "published", "mobility", "L12"),
    ("circuit", "gap_ratio", "published", "partial_edge_std", "L12"),
)
REPLICATES = 500
SEED = 20260906


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for cohort_name in ("development_main", "external_8b"):
        cohort = P.load_cohort(cohort_name)
        lookup = {(f, n): i for f, n, i in P.quantity_list(cohort)}
        layers = list(cohort["circuit"]["layer_names"].astype(str))
        circuit = cohort["circuit"]
        labels = pd.read_csv(P.LABEL_PATHS[cohort_name])
        risk = labels["original_failure"].to_numpy(bool)
        suite = P.suite_of(circuit)
        valid = circuit["valid"].astype(bool)
        priors = P.survival_prior(suite, circuit["length"].astype(int), risk)
        cut = {s: next((q for q, p in sorted(t.items()) if p >= P.LOW_PRIOR), None)
               for s, t in priors.items()}
        rng = np.random.default_rng(SEED)

        for fam_a, name_a, fam_b, name_b, layer in PAIRS:
            position = layers.index(layer)
            a = P.dense_values(cohort, fam_a, lookup[(fam_a, name_a)])[:, :, position]
            b = P.dense_values(cohort, fam_b, lookup[(fam_b, name_b)])[:, :, position]
            point_a = abs(survival_auc(a, valid, suite, risk, cut)[0] - 0.5)
            point_b = abs(survival_auc(b, valid, suite, risk, cut)[0] - 0.5)
            deltas = np.empty(args.replicates)
            for replicate in range(args.replicates):
                index = np.concatenate([
                    rng.choice(np.flatnonzero(suite == s), size=int((suite == s).sum()),
                               replace=True)
                    for s in np.unique(suite)
                ])
                ga = abs(survival_auc(a[index], valid[index], suite[index], risk[index], cut)[0] - 0.5)
                gb = abs(survival_auc(b[index], valid[index], suite[index], risk[index], cut)[0] - 0.5)
                deltas[replicate] = ga - gb
            results.append({
                "cohort": cohort_name, "layer": layer,
                "circuit_quantity": name_a, "reference_quantity": name_b,
                "circuit_auc_gap": point_a, "reference_auc_gap": point_b,
                "delta": point_a - point_b,
                "delta_lo95": float(np.quantile(deltas, 0.025)),
                "delta_hi95": float(np.quantile(deltas, 0.975)),
                "p_delta_le_0": float((deltas <= 0).mean()),
            })
            print(f"  {cohort_name} {name_a} vs {name_b}: {results[-1]['delta']:+.4f} "
                  f"[{results[-1]['delta_lo95']:+.4f}, {results[-1]['delta_hi95']:+.4f}]",
                  flush=True)

    payload = {
        "schema": "himoe.circuit_analogy.bootstrap_increment.v1",
        "post_hoc": True,
        "replicates": args.replicates,
        "seed": SEED,
        "bootstrap": "cluster over episodes within suite",
        "statistic": "|pooled survival-conditioned AUC - 0.5|, circuit minus reference",
        "results": results,
    }
    (args.output / "bootstrap_increment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
