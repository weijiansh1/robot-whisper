#!/usr/bin/env python3
"""Rebuild the 24 frozen frame-survey detectors on the development cohort.

`moe-hb-front-back-0905` only sealed external first-alarm arrays. The physical
labels cover both runs, so the development side is needed for any dev -> external
stability claim and for selecting a multi-head bundle without touching external.

The configuration (representation, direction, quantile, mode) is read verbatim
from the frozen `external_detectors.csv`; nothing is re-selected here. The alarm
machinery is the same one that produced the external arrays:

    trailing mean width 4 -> orient -> 4 consecutive confirmations -> threshold

Thresholds on development:
    per_task  leave-one-init-state-out ("crossfit") same-task quantile, i.e. the
              episode never calibrates its own threshold.
    global    one pooled quantile over development_main + development_extra peaks,
              exactly as the frozen script computed its development candidates.

As a self-check the script also rebuilds the external arrays and asserts they are
bit-identical to the sealed ones.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C
import evaluate_layerwise_alarm_development as dev

OUT = C.BUNDLE / "results"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = pd.read_csv(C.EXTERNAL_DETECTORS)
    config = config[config["feasible"].fillna(False).astype(bool)]

    graphs = {n: C.load_graph(n)
              for n in ("development_main", "development_extra", "external_8b")}
    mobilities = {n: C.load_npz(C.MOBILITY_PATHS[n]) for n in graphs}
    for name in graphs:
        if not np.array_equal(graphs[name]["episode"], mobilities[name]["episode"]):
            raise ValueError(f"{name}: graph and mobility caches are not aligned")

    dev_task_index = graphs["development_main"]["task_index"].astype(int)
    dev_init = graphs["development_main"]["init_state_id"].astype(int)
    dev_valid = graphs["development_main"]["valid"].astype(bool)
    ext_valid = graphs["external_8b"]["valid"].astype(bool)
    ext_task = C.task_of(graphs["external_8b"])
    reference_task = {n: C.task_of(graphs[n])
                      for n in ("development_main", "development_extra")}

    sealed = C.load_npz(C.EXTERNAL_ALARMS)

    dev_alarms: dict[str, np.ndarray] = {}
    ext_alarms: dict[str, np.ndarray] = {}
    checks = []

    for _, row in config.iterrows():
        quantity = str(row["quantity"])
        mode = str(row["mode"])
        representation = str(row["representation"])
        direction = str(row["direction"])
        quantile = float(row["quantile"])
        key = f"{quantity}|{mode}"

        caches = {n: C.quantity_cache(graphs[n], mobilities[n], quantity) for n in graphs}
        reprs = {n: {k: v for k, (v, _) in dev.representations(caches[n]).items()}
                 for n in caches}

        dev_or = C.oriented(reprs["development_main"][representation], direction)
        dev_persistent = dev.persistent_score(dev_or, C.CONFIRMATIONS)
        ext_persistent = dev.persistent_score(
            C.oriented(reprs["external_8b"][representation], direction), C.CONFIRMATIONS
        )
        pooled_peak = np.concatenate(
            [dev.row_max(C.oriented(reprs[n][representation], direction))
             for n in ("development_main", "development_extra")]
        )

        if mode == "global":
            line = dev.quantile_higher(pooled_peak, quantile)
            dev_first = dev.first_query(
                np.isfinite(dev_persistent) & (dev_persistent > line) & dev_valid
            )
            ext_first = dev.first_query(
                np.isfinite(ext_persistent) & (ext_persistent > line) & ext_valid
            )
        else:
            crossfit = dev.crossfit_thresholds(dev.row_max(dev_or), dev_task_index, dev_init)
            position = list(dev.QUANTILES).index(quantile)
            dev_first = dev.first_query(
                np.isfinite(dev_persistent)
                & (dev_persistent > crossfit[:, position][:, None])
                & dev_valid
            )
            ext_first = np.full(len(ext_persistent), -1, dtype=np.int16)
            for task in np.unique(ext_task):
                take = np.flatnonzero(ext_task == task)
                peaks = [
                    dev.row_max(
                        C.oriented(
                            reprs[n][representation][reference_task[n] == task], direction
                        )
                    )
                    for n in ("development_main", "development_extra")
                    if (reference_task[n] == task).any()
                ]
                line = dev.quantile_higher(np.concatenate(peaks), quantile)
                ext_first[take] = dev.first_query(
                    np.isfinite(ext_persistent[take])
                    & (ext_persistent[take] > line)
                    & ext_valid[take]
                )

        dev_alarms[key] = dev_first.astype(np.int16)
        ext_alarms[key] = ext_first.astype(np.int16)
        identical = bool(np.array_equal(ext_first.astype(np.int16), sealed[key]))
        checks.append({"detector": key, "external_reproduced_exactly": identical,
                       "disagreements": int((ext_first != sealed[key]).sum())})
        print(f"  {key}: external reproduced exactly = {identical}", flush=True)

    v7 = C.load_npz(C.V7_ALARMS)
    for head in ("freeze", "acceleration", "periodicity", "turbulence", "guard"):
        dev_alarms[f"v7_{head}|task_agnostic"] = v7[f"main_{head}"].astype(np.int16)
        ext_alarms[f"v7_{head}|task_agnostic"] = v7[f"external_{head}"].astype(np.int16)

    np.savez_compressed(
        OUT / "first_alarms_development.npz",
        schema=np.asarray("himoe.failure_modes_0906.alarms.development.v1"),
        **dev_alarms,
    )
    np.savez_compressed(
        OUT / "first_alarms_external.npz",
        schema=np.asarray("himoe.failure_modes_0906.alarms.external.v1"),
        **ext_alarms,
    )
    (OUT / "alarm_rebuild_audit.json").write_text(
        json.dumps(
            {
                "schema": "himoe.failure_modes_0906.alarm_rebuild.v1",
                "detectors": len(dev_alarms),
                "external_reproduction": checks,
                "all_external_reproduced": all(c["external_reproduced_exactly"] for c in checks),
                "development_thresholds": {
                    "per_task": "leave-one-init-state-out same-task quantile (crossfit)",
                    "global": "pooled quantile over development_main + development_extra peaks",
                },
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print("all external arrays reproduced:",
          all(c["external_reproduced_exactly"] for c in checks))


if __name__ == "__main__":
    main()
