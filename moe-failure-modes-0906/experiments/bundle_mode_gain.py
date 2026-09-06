#!/usr/bin/env python3
"""Q3/Q4 closing question: when a multi-head beats a single head, WHAT does it
catch that the single head missed -- is the extra yield spread evenly over the
physical failure modes, or concentrated in specific ones?

This is the only version of "is multi-head interpretable" that cannot be
circular: the extra detections are labelled by the simulator, not by any
detector.

Null for concentration: the episodes the single head missed are re-labelled by
permuting their physical mode WITHIN suite (and within task), 20,000 times; the
statistic is the mode composition of the extra catches.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C
from mode_specialisation import permutations

OUT = C.BUNDLE / "results"
B_PERM = 20000
SEED = 20260907


def or_alarm(arrays):
    out = np.full(len(arrays[0]), 1 << 14, dtype=np.int32)
    for a in arrays:
        a = np.asarray(a, int)
        fired = a >= 0
        out[fired] = np.minimum(out[fired], a[fired])
    out[out == (1 << 14)] = -1
    return out.astype(np.int16)


def main() -> None:
    ext_alarms = C.load_npz(OUT / "first_alarms_external.npz"); ext_alarms.pop("schema")
    frame = C.cohort_index("external_8b")
    risk = frame.risk.to_numpy(bool)
    suite = frame.suite.to_numpy(str)
    priors = C.survival_prior(suite, frame.length.to_numpy(int), risk)
    mode = frame.primary_failure_reason.fillna("").to_numpy(str)

    idx = np.flatnonzero(risk)
    mode_r, suite_r = mode[idx], suite[idx]
    task_r = frame.task_key.to_numpy(str)[idx]
    modes = list(pd.Series(mode_r).value_counts().index)
    onehot = np.stack([(mode_r == m).astype(np.float32) for m in modes], axis=1)
    rng = np.random.default_rng(SEED)
    perms = {"suite": permutations(suite_r, rng, B_PERM),
             "task": permutations(task_r, rng, B_PERM)}

    # Comparisons. Every bundle is an OR of frozen heads; every single is frozen.
    comparisons = {
        # descriptive frontier winners at matched external false alarms
        "global_fp57": {
            "single": ["expert_load_effective_rank|global"],
            "bundle": ["expert_load_effective_rank|global", "mobility|global",
                       "flow_path|global"],
        },
        "task_agnostic_fp100": {
            "single": ["v7_guard|task_agnostic"],
            "bundle": ["v7_guard|task_agnostic", "flow_endpoint|global",
                       "conditional_energy|global", "flow_path|global"],
        },
        # the interpretable two-head: absolute-anomaly head + task-relative head
        "interpretable_two_head": {
            "single": ["expert_load_effective_rank|per_task"],
            "bundle": ["expert_load_effective_rank|per_task", "mobility|global"],
        },
        "interpretable_two_head_global_base": {
            "single": ["mobility|global"],
            "bundle": ["mobility|global", "expert_load_effective_rank|per_task"],
        },
    }

    report = {"schema": "himoe.failure_modes_0906.bundle_mode_gain.v1",
              "permutations": B_PERM, "seed": SEED, "comparisons": {}}
    rows = []
    for label, spec in comparisons.items():
        single = or_alarm([ext_alarms[k] for k in spec["single"]])
        bundle = or_alarm([ext_alarms[k] for k in spec["bundle"]])
        s_score = C.score_candidate(np.asarray(single, int), risk,
                                    C.prior_of(np.asarray(single, int), suite, priors))
        b_score = C.score_candidate(np.asarray(bundle, int), risk,
                                    C.prior_of(np.asarray(bundle, int), suite, priors))
        s_fired = (np.asarray(single)[idx] >= 0)
        b_fired = (np.asarray(bundle)[idx] >= 0)
        extra = b_fired & ~s_fired
        missed = ~s_fired
        n_extra = int(extra.sum())

        observed = extra.astype(np.float32) @ onehot
        base = missed.astype(np.float32) @ onehot
        expected_uniform = base * (n_extra / max(int(missed.sum()), 1))

        stats = {}
        for name, perm in perms.items():
            # permute mode among the risks the single head missed, within stratum,
            # keeping the set of extra catches fixed
            null = np.empty((B_PERM, len(modes)), dtype=np.float32)
            e = extra.astype(np.float32)
            for b in range(0, B_PERM, 2000):
                chunk = perm[b:b + 2000]
                null[b:b + 2000] = e[chunk] @ onehot
            mean = null.mean(axis=0)
            sd = null.std(axis=0)
            p = (np.abs(null - mean) >= np.abs(observed - mean)[None, :] - 1e-9).mean(axis=0)
            var = np.where(sd > 0, sd ** 2, np.inf)
            t_obs = float((((observed - mean) ** 2) / var).sum())
            t_null = (((null - mean[None, :]) ** 2) / var[None, :]).sum(axis=1)
            stats[name] = {"expected": mean, "p": p,
                           "omnibus_p": float((t_null >= t_obs - 1e-9).mean())}

        for position, m in enumerate(modes):
            rows.append({
                "comparison": label, "mode": m, "mode_short": C.MODE_SHORT.get(m, m),
                "n_mode": int(onehot[:, position].sum()),
                "single_caught": int((s_fired & (mode_r == m)).sum()),
                "bundle_caught": int((b_fired & (mode_r == m)).sum()),
                "extra": int(observed[position]),
                "expected_extra_suite": float(stats["suite"]["expected"][position]),
                "expected_extra_task": float(stats["task"]["expected"][position]),
                "expected_extra_uniform_over_misses": float(expected_uniform[position]),
                "p_suite": float(stats["suite"]["p"][position]),
                "p_task": float(stats["task"]["p"][position]),
            })
        report["comparisons"][label] = {
            "single_heads": spec["single"], "bundle_heads": spec["bundle"],
            "single_external": s_score, "bundle_external": b_score,
            "extra_true_positives": n_extra,
            "extra_false_alarms": int(((np.asarray(bundle) >= 0) & ~risk).sum()
                                      - ((np.asarray(single) >= 0) & ~risk).sum()),
            "omnibus_p_suite": stats["suite"]["omnibus_p"],
            "omnibus_p_task": stats["task"]["omnibus_p"],
        }

    pd.DataFrame(rows).to_csv(OUT / "bundle_mode_gain.csv", index=False)
    (OUT / "bundle_mode_gain.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    for label, block in report["comparisons"].items():
        s, b = block["single_external"], block["bundle_external"]
        print(f"{label}: single tp={s['tp']} fp={s['fp']} -> bundle tp={b['tp']} fp={b['fp']} "
              f"| extra TP={block['extra_true_positives']} extra FP={block['extra_false_alarms']} "
              f"| concentration p suite={block['omnibus_p_suite']:.4f} "
              f"task={block['omnibus_p_task']:.4f}")


if __name__ == "__main__":
    main()
