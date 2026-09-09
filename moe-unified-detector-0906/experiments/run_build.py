#!/usr/bin/env python3
"""Fit the unified detector on development, seal it, and score every cohort once.

Run order is fixed and single-pass:

    development_main   -> gated chunks, terms, weights, thresholds, frontier
    external_8b        -> scored once at every frozen frontier point
    legacy_main16x32   -> scored once at every frozen frontier point

Arms produced
    unified              headline; portable channels, all three normalisations
    unified_extended     same recipe over the step-resolved channels (no legacy)
    ablate_no_self       normalisations restricted to {raw, pop}
    ablate_no_pop        normalisations restricted to {raw, self}
    ablate_raw_only      normalisations restricted to {raw}
    ablate_no_gating     every in-window chunk is a gate
    null_white           identical pipeline on iid noise channels
    null_episode_const   identical pipeline on episode-constant noise channels

Temporal integration is not an arm: it is the `rule` axis inside every arm, so
"drop temporal integration" is read off by restricting to `single_chunk`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import protocol as P
import arms
import auc_table
import detector as D
import sweep as S


RESULTS = P.RESULTS
DRIFT_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_arm(
    name: str,
    frames: dict,
    values: dict,
    columns: list[str],
    table: pd.DataFrame,
    info: pd.DataFrame,
    **kwargs,
) -> tuple[D.Detector, dict, pd.DataFrame]:
    arm_label = kwargs.pop("arm_label", "real")
    dev = frames[P.FIT_COHORT]
    det, diag = D.fit(
        dev, values[P.FIT_COHORT], columns, table, info, name=name, **kwargs
    )
    scores = {
        cohort: D.score(det, frames[cohort], values[cohort], columns)
        for cohort in values
    }
    prof = diag["gate_profile"]
    best_chunk = {}
    for suite in P.SUITES:
        sub = prof[(prof["suite"] == suite) & prof["gated"]]
        best_chunk[suite] = int(
            sub.sort_values(["top3", "top1"], ascending=False)["chunk"].iloc[0]
        )
    gmask = D.gate_mask(dev, det.gated)
    pool = scores[P.FIT_COHORT][gmask]
    drifts = tuple(
        float(np.quantile(pool[np.isfinite(pool)], q)) for q in DRIFT_QUANTILES
    )
    curve = S.run(
        scores, frames, det.gated, best_chunk, name, arm_label, drifts=drifts
    )
    det.meta["best_chunk"] = best_chunk
    det.meta["drifts"] = list(drifts)
    det.meta["drift_quantiles"] = list(DRIFT_QUANTILES)
    return det, {"diag": diag, "scores": scores, "best_chunk": best_chunk}, curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--skip-extended", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "auc").mkdir(exist_ok=True)

    log("loading cohorts")
    frames = {c: P.load_cohort(c) for c in P.COHORTS}
    meta = P.column_frame(frames[P.FIT_COHORT]["columns"])
    portable_cols = meta.loc[meta["is_portable"] | meta["is_control"], "column"].tolist()
    extended_cols = list(frames[P.FIT_COHORT]["columns"])
    for cohort in P.COHORTS:
        assert set(portable_cols) <= set(frames[cohort]["columns"]), cohort

    dev = frames[P.FIT_COHORT]
    log(f"development risks={int(dev['risk'].sum())}  "
        f"external={int(frames['external_8b']['risk'].sum())}  "
        f"legacy={int(frames['legacy_main16x32']['risk'].sum())}")
    assert int(dev["risk"].sum()) == 487
    assert int(frames["external_8b"]["risk"].sum()) == 564
    assert int(frames["legacy_main16x32"]["risk"].sum()) == 307

    log("within-episode information")
    info = D.within_episode_info(dev, dev["values"])
    info.to_csv(RESULTS / "within_episode_information.csv", index=False)

    def slice_cols(cols):
        idx = {c: j for j, c in enumerate(frames[P.FIT_COHORT]["columns"])}
        return {
            cohort: frames[cohort]["values"][
                :, :, [frames[cohort]["columns"].index(c) for c in cols]
            ]
            for cohort in frames
            if set(cols) <= set(frames[cohort]["columns"])
        }

    log("AUC table: real / extended")
    path = RESULTS / "auc/development_real_extended.csv.gz"
    if path.exists():
        table_ext = pd.read_csv(path)
    else:
        table_ext = auc_table.build(dev, dev["values"], extended_cols, workers=args.workers)
        table_ext.to_csv(path, index=False)
    table_port = table_ext[table_ext["column"].isin(set(portable_cols))].copy()

    curves, detectors, artefacts = [], {}, {}

    portable_values = slice_cols(portable_cols)
    log(f"portable arm: {len(portable_cols)} columns, cohorts={list(portable_values)}")

    specs = [
        ("unified", dict(scope_portable=True, norms=P.NORMALISATIONS)),
        ("ablate_no_self", dict(scope_portable=True, norms=("raw", "pop"))),
        ("ablate_no_pop", dict(scope_portable=True, norms=("raw", "self"))),
        ("ablate_raw_only", dict(scope_portable=True, norms=("raw",))),
        ("ablate_no_gating",
         dict(scope_portable=True, norms=P.NORMALISATIONS, use_all_chunks=True)),
        ("unified_plus_prior",
         dict(scope_portable=True, norms=P.NORMALISATIONS, use_prior=True)),
    ]
    for name, kw in specs:
        log(f"fitting {name}")
        det, art, curve = build_arm(
            name, frames, portable_values, portable_cols, table_port, info, **kw
        )
        curves.append(curve)
        detectors[name] = det
        artefacts[name] = art
        log(f"  gated={det.gated}  terms={det.terms}")

    for arm in ("white", "episode_const"):
        log(f"null arm {arm}: AUC table")
        null_values = {
            cohort: arms.arm_values(
                frames[cohort], arm, portable_cols, arms.arm_seed(arm, cohort)
            )
            for cohort in portable_values
        }
        tpath = RESULTS / f"auc/development_{arm}_portable.csv.gz"
        if tpath.exists():
            ntable = pd.read_csv(tpath)
        else:
            ntable = auc_table.build(
                dev, null_values[P.FIT_COHORT], portable_cols, workers=args.workers
            )
            ntable.to_csv(tpath, index=False)
        log(f"null arm {arm}: fitting")
        det, art, curve = build_arm(
            f"null_{arm}", frames, null_values, portable_cols, ntable, info,
            scope_portable=True, norms=P.NORMALISATIONS, arm_label=arm,
        )
        curves.append(curve)
        detectors[f"null_{arm}"] = det
        artefacts[f"null_{arm}"] = art
        del null_values

    if not args.skip_extended:
        log("extended arm")
        ext_values = slice_cols(extended_cols)
        det, art, curve = build_arm(
            "unified_extended", frames, ext_values, extended_cols, table_ext, info,
            scope_portable=False, norms=P.NORMALISATIONS,
        )
        curves.append(curve)
        detectors["unified_extended"] = det
        artefacts["unified_extended"] = art
        del ext_values

    log("baselines")
    curves.append(S.survival_curve(frames))
    curves.append(S.length_leak_row(frames))

    table = pd.concat(curves, ignore_index=True)
    table.to_csv(RESULTS / "operating_points.csv.gz", index=False)
    log(f"wrote operating_points.csv.gz  rows={len(table)}")

    fronts = []
    for (name, arm), sub in table.groupby(["detector", "arm"]):
        if name in ("survival_prior", "length_leak_NOT_a_baseline"):
            continue
        f = S.frontier(sub)
        f["detector"], f["arm"] = name, arm
        fronts.append(f)
    front = pd.concat(fronts, ignore_index=True)
    front.to_csv(RESULTS / "development_frontier.csv.gz", index=False)
    log(f"wrote development_frontier.csv.gz  rows={len(front)}")

    payload = {
        name: {
            "terms": det.terms,
            "sign": det.sign.tolist(),
            "weight": det.weight.tolist(),
            "gated": det.gated,
            "meta": det.meta,
        }
        for name, det in detectors.items()
    }
    P.write_json(RESULTS / "detectors.json", payload)

    np.savez_compressed(
        RESULTS / "scores_dev.npz",
        **{name: art["scores"][P.FIT_COHORT].astype(np.float32)
           for name, art in artefacts.items()},
    )
    with open(RESULTS / "artefacts.pkl", "wb") as fh:
        pickle.dump(
            {
                "detectors": detectors,
                "scores": {n: a["scores"] for n, a in artefacts.items()},
                "best_chunk": {n: a["best_chunk"] for n, a in artefacts.items()},
                "gate_profile": {n: a["diag"]["gate_profile"] for n, a in artefacts.items()},
                "term_stats": {n: a["diag"]["term_stats"] for n, a in artefacts.items()},
            },
            fh,
        )
    for name, art in artefacts.items():
        art["diag"]["gate_profile"].to_csv(
            RESULTS / f"gate_profile_{name}.csv", index=False
        )
    log("done")


if __name__ == "__main__":
    main()
