"""Census of every saved alarm vector: what it fires on, what it catches.

No detector is built and no feature is re-derived.  Each first-alarm vector is
read as published; this script only scores it against the canonical risk label
under the in-window deadline and the *task*-matched survival prior, and records
which vectors are bit-identical to which others.

Conventions fixed here and used by every downstream script:
  * in-window  == first alarm chunk < 0.65 x suite cap  (caps 30/52/28/22)
  * FP         == in-window alarm on a non-risk episode; FPR == FP / n_nonrisk
  * lift       == precision / mean task-matched survival prior at the alarm chunk
  * every count threshold is inclusive (>= / <=), never strict
  * length is a negative control (is_baseline = False), never a ranked entry
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import synth_core as sc


def score_matrix(c: sc.Cohort, alarms: np.ndarray, names: list[str]) -> pd.DataFrame:
    iw = (alarms >= 0) & (alarms < c.deadline[None, :])
    risk = c.risk
    n_risk = int(risk.sum())
    n_safe = int((~risk).sum())
    tp = (iw & risk).sum(1)
    fp = (iw & ~risk).sum(1)
    fired_any = (alarms >= 0).sum(1)

    prior_t = np.full(alarms.shape, np.nan)
    prior_s = np.full(alarms.shape, np.nan)
    for d in range(alarms.shape[0]):
        prior_t[d] = sc.prior_of(alarms[d], c.task, c.priors_task)
        prior_s[d] = sc.prior_of(alarms[d], c.suite, c.priors_suite)
    with np.errstate(invalid="ignore"):
        mean_pt = np.array(
            [np.nanmean(prior_t[d][iw[d]]) if iw[d].any() else np.nan
             for d in range(alarms.shape[0])]
        )
        mean_ps = np.array(
            [np.nanmean(prior_s[d][iw[d]]) if iw[d].any() else np.nan
             for d in range(alarms.shape[0])]
        )
    alarms_iw = tp + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(alarms_iw > 0, tp / np.maximum(alarms_iw, 1), np.nan)
        task_lift = precision / mean_pt
        suite_lift = precision / mean_ps

    rows = {
        "detector": names,
        "bundle": [d.split("::", 1)[0] for d in names],
        "name": [d.split("::", 1)[1] for d in names],
        "cohort": c.name,
        "alarms_any": fired_any,
        "alarms_in_window": alarms_iw,
        "tp": tp,
        "fp": fp,
        "recall": tp / n_risk,
        "fpr": fp / n_safe,
        "precision": precision,
        "task_prior": mean_pt,
        "task_lift": task_lift,
        "suite_prior": mean_ps,
        "suite_lift": suite_lift,
        "is_baseline": True,
    }
    df = pd.DataFrame(rows)
    for s in sorted(set(c.suite)):
        m = c.suite == s
        nr = int(risk[m].sum())
        ns = int((~risk[m]).sum())
        df[f"tp_{s}"] = (iw[:, m] & risk[m]).sum(1)
        df[f"fp_{s}"] = (iw[:, m] & ~risk[m]).sum(1)
        df[f"recall_{s}"] = df[f"tp_{s}"] / nr
        df[f"fpr_{s}"] = df[f"fp_{s}"] / ns
    return df


def duplicate_groups(alarms: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Bit-identical alarm vectors.  The bluntest possible test of whether the
    detector names overstate the number of distinct objects."""
    seen: dict[bytes, list[str]] = {}
    for i, nm in enumerate(names):
        seen.setdefault(alarms[i].tobytes(), []).append(nm)
    rows = []
    for gid, (_, members) in enumerate(sorted(seen.items(), key=lambda kv: -len(kv[1]))):
        for m in members:
            rows.append({"group": gid, "size": len(members), "detector": m,
                         "representative": members[0]})
    return pd.DataFrame(rows)


def main() -> None:
    sc.RESULTS.mkdir(parents=True, exist_ok=True)
    ext = sc.load_cohort("external_8b")
    dev = sc.load_cohort("development_main")
    shared = sc.shared_detectors(ext, dev)

    frames, dup_frames = [], []
    meta = {"schema": "himoe.method_synthesis.census.v1"}
    for c in (ext, dev):
        df = score_matrix(c, c.alarms, c.detectors)
        df["shared_cohorts"] = df.detector.isin(shared)

        # Negative control.  Risk is *defined* as failing to finish before the
        # cap, so a length-at-cap alarm recalls 100% of risks by construction.
        # It is recorded, labelled, and excluded from every ranking.
        neg = sc.negative_control_length(c)
        ndf = score_matrix(c, neg[None, :], ["_negative_control::length_at_cap"])
        ndf["is_baseline"] = False
        ndf["shared_cohorts"] = True
        frames.append(pd.concat([df, ndf], ignore_index=True))

        d = duplicate_groups(c.alarms, c.detectors)
        d["cohort"] = c.name
        dup_frames.append(d)
        meta[c.name] = {
            "n_episodes": int(c.n),
            "n_risk": int(c.risk.sum()),
            "n_detectors": int(len(c.detectors)),
            "n_distinct_alarm_vectors": int(d.group.nunique()),
            "n_detectors_in_duplicate_groups": int((d["size"] > 1).sum()),
            "negative_control_recall": float(ndf.recall.iloc[0]),
            "negative_control_fp": int(ndf.fp.iloc[0]),
            "union_recall_all_detectors": float(
                ((c.alarms >= 0) & (c.alarms < c.deadline[None, :]) & c.risk).any(0).sum()
                / c.risk.sum()
            ),
            "union_fp_all_detectors": int(
                ((c.alarms >= 0) & (c.alarms < c.deadline[None, :]) & ~c.risk).any(0).sum()
            ),
            "physical_modes": {
                k: int(v) for k, v in
                pd.Series(c.mode[c.risk]).value_counts().items()
            },
        }
    meta["n_shared_detectors"] = len(shared)
    meta["n_shared_excluding_v7"] = len([s for s in shared if not s.startswith("moe-v7")])
    meta["note_negative_control"] = (
        "length_at_cap recalls every risk by construction (risk == did not finish "
        "before the cap); is_baseline=False and it is excluded from all rankings."
    )

    census = pd.concat(frames, ignore_index=True)
    census.to_csv(sc.RESULTS / "detector_census.csv", index=False)
    pd.concat(dup_frames, ignore_index=True).to_csv(
        sc.RESULTS / "duplicate_groups.csv", index=False
    )
    (sc.RESULTS / "census.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2)[:4000])


if __name__ == "__main__":
    main()
