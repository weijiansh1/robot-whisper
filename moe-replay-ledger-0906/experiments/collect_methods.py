#!/usr/bin/env python3
"""Collect every method's first-alarm array for one cohort and check it.

Each entry is either read from a bundle's sealed array or rebuilt from the raw
per-chunk scores. Whichever it is, the assembled TP/FP is compared with the
count the bundle published; a disagreement is recorded, never silently resolved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger_core import (  # noqa: E402
    CIRCUIT,
    COMBO,
    FLOW,
    FRAMES,
    HB,
    Method,
    STATECH,
    TOKGEO,
    TWOTIER,
    V4,
    V7,
    Cohort,
    _npz,
    and_alarm,
    build_layer_rules,
    counts,
    k_of_n_alarm,
    or_alarm,
)


FROZEN_COMBOS = {
    "combo:G-A": dict(
        family="cascade", mode="global", pool="all12", level="0.99",
        level_loose="0.85", k=2, window=-1, frames=2,
    ),
    "combo:G-B": dict(
        family="quorum", mode="global", pool="all12", level="0.99",
        level_loose="0.99", k=1, window=0, frames=1,
    ),
    "combo:P-C": dict(
        family="quorum", mode="per_task", pool="dedup11", level="0.8",
        level_loose="0.8", k=4, window=-1, frames=1,
    ),
}


def _published(frame: pd.DataFrame, key_cols: dict) -> tuple[int, int] | None:
    block = frame
    for col, val in key_cols.items():
        block = block[block[col] == val]
    if len(block) != 1:
        return None
    return int(block.iloc[0]["tp"]), int(block.iloc[0]["fp"])


def collect(cohort: Cohort) -> tuple[list[Method], list[dict], list[dict]]:
    """Return (methods, discrepancies, skipped)."""
    methods: list[Method] = []
    discrepancies: list[dict] = []
    skipped: list[dict] = []
    ext = cohort.name == "external_8b"
    n = cohort.n

    def add(m: Method) -> None:
        assert len(m.first) == n, f"{m.name}: array length {len(m.first)} != {n}"
        assert m.first.max() < cohort.queries, f"{m.name}: alarm chunk beyond cache"
        fired = m.first >= 0
        # an alarm can only land on a chunk the episode actually ran
        bad = int((fired & (m.first >= cohort.length)).sum())
        assert bad == 0, f"{m.name}: {bad} alarms after the episode ended"
        tp, fp = counts(m.first, cohort.risk)
        if m.published is not None and (tp, fp) != m.published:
            discrepancies.append(
                {
                    "cohort": cohort.name,
                    "method": m.name,
                    "assembled_tp": tp,
                    "assembled_fp": fp,
                    "published_tp": m.published[0],
                    "published_fp": m.published[1],
                    "source": m.source,
                }
            )
        methods.append(m)

    def skip(name: str, reason: str) -> None:
        skipped.append({"cohort": cohort.name, "method": name, "reason": reason})

    # ---------------------------------------------------------------- v4 -----
    layer = build_layer_rules(cohort.name)
    pub_layer = (
        {
            "front_lock": (240, 36),
            "back_lock": (409, 146),
            "all_lock": (382, 76),
            "L5_switching": (34, 5),
            "v4_dual_regime": (410, 81),
        }
        if ext
        else {
            # moe-v4-0904/results/cache_new_v4/evaluation_summary.json,
            # cohort development_main: tp = risk_n - fn
            "all_lock": (349, 61),
            "L5_switching": (26, 7),
            "v4_dual_regime": (372, 67),
        }
    )
    for key, arr in layer.items():
        add(
            Method(
                name=f"v4:{key}" if key != "v4_dual_regime" else "v4_dual_regime",
                family="v4_layer_lock",
                threshold_mode="per_task",
                runtime_task_identity=True,
                source=str(
                    (HB / "experiments/evaluate_layer_survival_baseline.py").relative_to(
                        HB.parent
                    )
                ),
                config={
                    "front_lock": "front_median | low | w4 | k4 | q0.75",
                    "back_lock": "back_median | low | w4 | k4 | q0.75",
                    "all_lock": "all_median | low | w4 | k4 | q0.75",
                    "L5_switching": "L5 | high | w4 | k8 | q0.80",
                    "v4_dual_regime": "all_median lock q0.75 OR L5 instability q0.80",
                }[key],
                first=arr,
                published=pub_layer.get(key),
            )
        )

    # cross-check against the sealed v4 array
    sealed = _npz(V4 / "results/cache_new_v4/sealed_first_alarms.npz")
    prefix = "external" if ext else "main"
    for key, arr_name in (("all_lock", "lock"), ("L5_switching", "instability"), ("v4_dual_regime", "dual")):
        stored = sealed[f"{prefix}_{arr_name}"].astype(np.int16)
        rebuilt = layer[key]
        if not np.array_equal(stored, rebuilt):
            agree = int((stored == rebuilt).sum())
            discrepancies.append(
                {
                    "cohort": cohort.name,
                    "method": f"v4:{key} vs sealed_first_alarms",
                    "assembled_tp": agree,
                    "assembled_fp": n - agree,
                    "published_tp": n,
                    "published_fp": 0,
                    "source": "moe-v4-0904/results/cache_new_v4/sealed_first_alarms.npz "
                    "(columns are rows-agreeing / rows-differing)",
                }
            )

    # ---------------------------------------------------------------- v7 -----
    v7 = _npz(V7 / "results/intrinsic_guard_v7/sealed_first_alarms.npz")
    v7_metrics = pd.read_csv(V7 / "results/intrinsic_guard_v7/outcome_metrics.csv")
    v7_metrics = v7_metrics[
        (v7_metrics["group"] == "all") & (v7_metrics["cohort"] == cohort.name)
    ]
    v7_map = {
        "freeze": "relative_freeze",
        "acceleration": "flow_acceleration",
        "periodicity": "recurrence_loss",
        "turbulence": "confirmed_turbulence",
        "guard": "intrinsic_guard_v7",
    }
    for key, detector in v7_map.items():
        arr = v7[f"{prefix}_{key}"].astype(np.int16)
        add(
            Method(
                name="v7_guard" if key == "guard" else f"v7:{key}",
                family="v7_intrinsic_guard",
                threshold_mode="task_agnostic",
                runtime_task_identity=False,
                source="moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz",
                config=f"sealed branch '{detector}', pooled 16k-corpus thresholds",
                first=arr,
                published=_published(v7_metrics, {"detector": detector}),
            )
        )
    # is the guard the OR of its branches?
    or3 = or_alarm(
        v7[f"{prefix}_freeze"], v7[f"{prefix}_acceleration"], v7[f"{prefix}_turbulence"]
    )
    guard_is_or3 = bool(np.array_equal(or3, v7[f"{prefix}_guard"].astype(np.int16)))

    # ------------------------------------------------------ frame survey -----
    survey = pd.read_csv(HB / "results/frame_survey/external_detectors.csv")
    if ext:
        arrays = _npz(HB / "results/frame_survey/external_first_alarms.npz")
    else:
        arrays = _npz(TWOTIER / "results/development_first_alarms.npz")
    survey_specs: dict[str, dict] = {}
    for _, row in survey.iterrows():
        if not bool(row["feasible"]):
            skip(f"{row['quantity']}|{row['mode']}", "frame survey marked infeasible")
            continue
        key = f"{row['quantity']}|{row['mode']}"
        survey_specs[key] = {
            "representation": row["representation"],
            "direction": row["direction"],
            "quantile": float(row["quantile"]),
        }
        if key not in arrays:
            skip(key, f"array absent from the {cohort.name} alarm file")
            continue
        add(
            Method(
                name=key,
                family=f"frame_survey/{FRAMES[row['quantity']]}",
                threshold_mode=str(row["mode"]),
                runtime_task_identity=row["mode"] == "per_task",
                source=(
                    "moe-hb-front-back-0905/results/frame_survey/external_first_alarms.npz"
                    if ext
                    else "moe-two-tier-0906/results/development_first_alarms.npz"
                ),
                config=f"{row['representation']} | {row['direction']} | w4 | k4 | q{row['quantile']:g}",
                first=arrays[key].astype(np.int16),
                published=(int(row["tp"]), int(row["fp"])) if ext else None,
            )
        )

    # the two-tier bundle republished the same 24 heads on external; check agreement
    if ext:
        twotier_heads = _npz(TWOTIER / "results/external_first_alarms.npz")
        for key in survey_specs:
            if key in twotier_heads and key in arrays:
                if not np.array_equal(
                    twotier_heads[key].astype(np.int16), arrays[key].astype(np.int16)
                ):
                    agree = int((twotier_heads[key] == arrays[key]).sum())
                    discrepancies.append(
                        {
                            "cohort": cohort.name,
                            "method": f"{key} frame_survey vs two-tier copy",
                            "assembled_tp": agree,
                            "assembled_fp": n - agree,
                            "published_tp": n,
                            "published_fp": 0,
                            "source": "row-wise array comparison",
                        }
                    )

    if not ext:
        # the newer bundles ship external arrays only; the shared machinery is
        # still exercised on development for the rules that can be rebuilt.
        for label in (
            "flow-semantics step_alarm",
            "circuit-analogy detectors",
            "token-geometry detection",
            "state-channel detectors",
            "two-tier WATCH/ACT",
        ):
            skip(label, "bundle ships external_8b arrays only; no development twin")
        sys.path.insert(0, str(COMBO / "experiments"))
        import combination_core as cc  # noqa: E402

        combo_cohort = cc.Cohort(cohort.name)
        assert np.array_equal(combo_cohort.risk, cohort.risk)
        dev_shortlist = pd.read_csv(COMBO / "results/development_rules.csv")
        for name, rule in FROZEN_COMBOS.items():
            key = cc.rule_key(rule)
            if rule["family"] == "quorum":
                cache = cc.WindowCache(
                    combo_cohort, rule["pool"], rule["mode"], rule["level"]
                )
                first = cc.quorum_alarm(cache, rule["k"], rule["window"], rule["frames"])
            else:
                strict = cc.WindowCache(
                    combo_cohort, rule["pool"], rule["mode"], rule["level"]
                )
                loose = cc.WindowCache(
                    combo_cohort, rule["pool"], rule["mode"], rule["level_loose"]
                )
                first = cc.cascade_alarm(
                    strict, loose, rule["k"], rule["window"], rule["frames"]
                )
            pub = dev_shortlist[dev_shortlist["rule"] == key] if "rule" in dev_shortlist else []
            add(
                Method(
                    name=name,
                    family="combination_rules",
                    threshold_mode=rule["mode"],
                    runtime_task_identity=rule["mode"] == "per_task",
                    source="moe-combination-rules-0906/results/alarms/"
                    "development_main_alarms.npz (rebuilt through combination_core)",
                    config=key,
                    first=first,
                    published=(int(pub.iloc[0]["tp"]), int(pub.iloc[0]["fp"]))
                    if len(pub) >= 1
                    else None,
                )
            )
        # constructed combos, development twin
        for name, cfg, heads, op in (
            (
                "build:OR_crossframe_global",
                "mobility|global OR flow_path|global OR expert_load_effective_rank|global",
                ["mobility|global", "flow_path|global", "expert_load_effective_rank|global"],
                "or",
            ),
            (
                "build:AND_mobility_expertload_global",
                "mobility|global AND expert_load_effective_rank|global",
                ["mobility|global", "expert_load_effective_rank|global"],
                "and",
            ),
            (
                "build:AND_mobility_flowsettling_per_task",
                "mobility|per_task AND flow_settling_log_ratio|per_task",
                ["mobility|per_task", "flow_settling_log_ratio|per_task"],
                "and",
            ),
        ):
            if not all(h in arrays for h in heads):
                skip(name, "member head missing from the development alarm file")
                continue
            members = [arrays[h].astype(np.int16) for h in heads]
            add(
                Method(
                    name=name,
                    family="constructed",
                    threshold_mode="per_task" if "per_task" in cfg else "global",
                    runtime_task_identity="per_task" in cfg,
                    source="built here from moe-two-tier-0906 development head arrays",
                    config=cfg,
                    first=or_alarm(*members) if op == "or" else and_alarm(*members),
                    published=None,
                )
            )
        from reconcile_prior_replay import join_prior_alarm, length_only_methods

        prior_methods, join_report = join_prior_alarm(cohort)
        for m in prior_methods:
            add(m)
        for m in length_only_methods(cohort, "suite") + length_only_methods(
            cohort, "task"
        ):
            add(m)
        return methods, discrepancies, skipped, {"prior_replay_join": join_report}

    # ------------------------------------------------------- flow semantics --
    path = FLOW / "results/step_alarm/external_first_alarms.npz"
    flow_det = pd.read_csv(FLOW / "results/step_alarm/external_detectors.csv")
    flow_arrays = _npz(path)
    wanted = [f"mobility_s{i}" for i in range(10)] + ["mobility_step_range"]
    for quantity in wanted:
        for mode in ("per_task", "global"):
            key = f"{quantity}|{mode}"
            if key not in flow_arrays:
                skip(key, "array absent from moe-flow-semantics-0906 step_alarm npz")
                continue
            spec = flow_det[
                (flow_det["quantity"] == quantity) & (flow_det["mode"] == mode)
            ]
            cfg = (
                f"{spec.iloc[0]['representation']} | {spec.iloc[0]['direction']} | "
                f"w4 | k4 | q{float(spec.iloc[0]['quantile']):g}"
                if len(spec) == 1
                else "unspecified"
            )
            add(
                Method(
                    name=f"flow:{key}",
                    family="flow_semantics/denoising_step",
                    threshold_mode=mode,
                    runtime_task_identity=mode == "per_task",
                    source="moe-flow-semantics-0906/results/step_alarm/external_first_alarms.npz",
                    config=cfg,
                    first=flow_arrays[key].astype(np.int16),
                    published=_published(flow_det, {"quantity": quantity, "mode": mode}),
                )
            )

    # ------------------------------------------------------ circuit analogy --
    circ_arrays = _npz(CIRCUIT / "results/detectors/external_first_alarms.npz")
    circ_det = pd.read_csv(CIRCUIT / "results/detectors/external_detectors.csv")
    for mode in ("global", "per_task"):
        key = f"circuit|norm_fiedler|{mode}"
        if key not in circ_arrays:
            skip(key, "array absent from moe-circuit-analogy-0906 detectors npz")
            continue
        spec = circ_det[
            (circ_det["quantity"] == "norm_fiedler") & (circ_det["mode"] == mode)
        ]
        add(
            Method(
                name=f"circuit:norm_fiedler|{mode}",
                family="circuit_analogy/graph_spectrum",
                threshold_mode=mode,
                runtime_task_identity=mode == "per_task",
                source="moe-circuit-analogy-0906/results/detectors/external_first_alarms.npz",
                config=(
                    f"{spec.iloc[0]['representation']} | {spec.iloc[0]['direction']} | "
                    f"w4 | k4 | q{float(spec.iloc[0]['quantile']):g}"
                    if len(spec) == 1
                    else "unspecified"
                ),
                first=circ_arrays[key].astype(np.int16),
                published=_published(
                    circ_det, {"quantity": "norm_fiedler", "mode": mode}
                ),
            )
        )
    # The reproducibility problem. Three quantities are published by
    # moe-hb-front-back-0905 and moe-circuit-analogy-0906 at different operating
    # points under a selection rule both bundles call identical. Both variants
    # are carried in the ledger under distinct names; neither is picked.
    pes_note = {}
    for quantity in ("partial_edge_std", "conditional_effective_rank", "partial_query_d1"):
        c_row = circ_det[
            (circ_det["quantity"] == quantity) & (circ_det["mode"] == "global")
        ]
        key = f"published|{quantity}|global"
        if len(c_row) != 1 or key not in circ_arrays:
            continue
        arr = circ_arrays[key].astype(np.int16)
        tp, fp = counts(arr, cohort.risk)
        fs_row = survey[(survey["quantity"] == quantity) & (survey["mode"] == "global")]
        pes_note[quantity] = {
            "frame_survey": f"q{float(fs_row.iloc[0]['quantile']):g} -> "
            f"{int(fs_row.iloc[0]['tp'])}/{int(fs_row.iloc[0]['fp'])}",
            "circuit_analogy": f"q{float(c_row.iloc[0]['quantile']):g} -> "
            f"{int(c_row.iloc[0]['tp'])}/{int(c_row.iloc[0]['fp'])}",
            "assembled_from_circuit_array": [tp, fp],
            "arrays_identical": bool(
                np.array_equal(arr, arrays[f"{quantity}|global"].astype(np.int16))
            ),
        }
        add(
            Method(
                name=f"circuitcopy:{quantity}|global",
                family="reproducibility_variant",
                threshold_mode="global",
                runtime_task_identity=False,
                source="moe-circuit-analogy-0906/results/detectors/external_first_alarms.npz",
                config=f"L3 | high | w4 | k4 | q{float(c_row.iloc[0]['quantile']):g} "
                "(circuit-analogy's selection of a quantity the frame survey "
                "froze at a different quantile)",
                first=arr,
                published=(int(c_row.iloc[0]["tp"]), int(c_row.iloc[0]["fp"])),
            )
        )

    # -------------------------------------------------------- token geometry --
    tg_arrays = _npz(TOKGEO / "results/detection/external_first_alarms.npz")
    tg_det = pd.read_csv(TOKGEO / "results/detection/external_detectors.csv")
    for quantity in ("procrustes_prev", "bandedness", "lam1_share"):
        for mode in ("per_task", "global"):
            key = f"{quantity}|{mode}"
            if key not in tg_arrays:
                skip(key, "array absent from moe-token-geometry-0906 detection npz")
                continue
            spec = tg_det[(tg_det["quantity"] == quantity) & (tg_det["mode"] == mode)]
            add(
                Method(
                    name=f"tokgeo:{key}",
                    family=f"token_geometry/{spec.iloc[0]['frame'] if len(spec) else 'unknown'}",
                    threshold_mode=mode,
                    runtime_task_identity=mode == "per_task",
                    source="moe-token-geometry-0906/results/detection/external_first_alarms.npz",
                    config=(
                        f"{spec.iloc[0]['representation']} | {spec.iloc[0]['direction']} | "
                        f"w4 | k4 | q{float(spec.iloc[0]['quantile']):g}"
                        if len(spec) == 1
                        else "unspecified"
                    ),
                    first=tg_arrays[key].astype(np.int16),
                    published=_published(tg_det, {"quantity": quantity, "mode": mode}),
                )
            )

    # --------------------------------------------------------- state channel --
    sc_arrays = _npz(STATECH / "results/detectors/external_first_alarms.npz")
    sc_det = pd.read_csv(STATECH / "results/detectors/external_detectors.csv")
    # verify the metadata this bundle ships alongside its alarms
    assert np.array_equal(sc_arrays["risk"].astype(bool), cohort.risk), (
        "state-channel npz risk vector is not row-aligned with the label file"
    )
    assert np.array_equal(sc_arrays["length"].astype(int), cohort.length)
    assert np.array_equal(sc_arrays["suite"].astype(str), cohort.suite)
    assert np.array_equal(sc_arrays["episode"].astype(int), cohort.episode)
    assert np.array_equal(sc_arrays["task"].astype(str), cohort.task)
    for quantity in ("td_slope", "state_mobility_s9", "state_minus_action_mobility"):
        for mode in ("per_task", "global"):
            key = f"{quantity}|{mode}"
            if key not in sc_arrays:
                skip(key, "array absent from moe-state-channel-0906 detectors npz")
                continue
            spec = sc_det[(sc_det["quantity"] == quantity) & (sc_det["mode"] == mode)]
            add(
                Method(
                    name=f"state:{key}",
                    family="state_channel",
                    threshold_mode=mode,
                    runtime_task_identity=mode == "per_task",
                    source="moe-state-channel-0906/results/detectors/external_first_alarms.npz",
                    config=(
                        f"{spec.iloc[0]['representation']} | {spec.iloc[0]['direction']} | "
                        f"w4 | k4 | q{float(spec.iloc[0]['quantile']):g}"
                        if len(spec) == 1
                        else "unspecified"
                    ),
                    first=sc_arrays[key].astype(np.int16),
                    published=_published(sc_det, {"quantity": quantity, "mode": mode}),
                )
            )
    state_mobility_duplicate = bool(
        np.array_equal(
            sc_arrays["state_mobility_s9|global"], sc_arrays["state_mobility_mean|global"]
        )
        and np.array_equal(
            sc_arrays["state_mobility_s9|per_task"],
            sc_arrays["state_mobility_mean|per_task"],
        )
    )

    # ------------------------------------------------------ combination rules --
    sys.path.insert(0, str(COMBO / "experiments"))
    import combination_core as cc  # noqa: E402

    combo_cohort = cc.Cohort(cohort.name)
    assert np.array_equal(combo_cohort.risk, cohort.risk)
    shortlist = pd.read_csv(COMBO / "results/external_shortlist.csv")
    for name, rule in FROZEN_COMBOS.items():
        key = cc.rule_key(rule)
        if rule["family"] == "quorum":
            cache = cc.WindowCache(combo_cohort, rule["pool"], rule["mode"], rule["level"])
            first = cc.quorum_alarm(cache, rule["k"], rule["window"], rule["frames"])
        else:
            strict = cc.WindowCache(combo_cohort, rule["pool"], rule["mode"], rule["level"])
            loose = cc.WindowCache(
                combo_cohort, rule["pool"], rule["mode"], rule["level_loose"]
            )
            first = cc.cascade_alarm(
                strict, loose, rule["k"], rule["window"], rule["frames"]
            )
        pub = shortlist[shortlist["rule"] == key]
        add(
            Method(
                name=name,
                family="combination_rules",
                threshold_mode=rule["mode"],
                runtime_task_identity=rule["mode"] == "per_task",
                source="moe-combination-rules-0906/results/alarms/external_8b_alarms.npz "
                "(rebuilt through combination_core)",
                config=key,
                first=first,
                published=(int(pub.iloc[0]["tp"]), int(pub.iloc[0]["fp"]))
                if len(pub) >= 1
                else None,
            )
        )

    # -------------------------------------------------------------- two tier --
    tt = _npz(TWOTIER / "results/external_first_alarms.npz")
    assert np.array_equal(tt["risk"].astype(bool), cohort.risk), (
        "two-tier npz risk vector is not row-aligned with the label file"
    )
    assert np.array_equal(tt["length"].astype(int), cohort.length)
    assert np.array_equal(tt["suite"].astype(str), cohort.suite)
    tt_rules = pd.read_csv(TWOTIER / "results/external_rules.csv")

    def tt_pub(pool: str, mode: str, role: str) -> tuple[int, int] | None:
        block = tt_rules[
            (tt_rules["pool"] == pool)
            & (tt_rules["mode"] == mode)
            & (tt_rules["role"] == role)
        ]
        if len(block) != 1:
            return None
        return int(block.iloc[0]["tp"]), int(block.iloc[0]["fp"])

    watch_a3 = ["expert_load_effective_rank|global", "mobility|global",
                "partial_edge_std|global", "flow_path|global"]
    watch_a1 = ["conditional_query_d1|global", "expert_load_effective_rank|global"]
    act_b = ["action_consensus|global", "conditional_query_d1|global", "mobility|global"]
    add(
        Method(
            name="twotier:WATCH_a3",
            family="two_tier",
            threshold_mode="global",
            runtime_task_identity=False,
            source="moe-two-tier-0906/results/external_first_alarms.npz (OR rebuilt)",
            config="OR of " + " + ".join(watch_a3),
            first=or_alarm(*[tt[h].astype(np.int16) for h in watch_a3]),
            published=tt_pub("primary", "global", "watch_a3"),
        )
    )
    add(
        Method(
            name="twotier:WATCH_a1",
            family="two_tier",
            threshold_mode="global",
            runtime_task_identity=False,
            source="moe-two-tier-0906/results/external_first_alarms.npz (OR rebuilt)",
            config="OR of " + " + ".join(watch_a1),
            first=or_alarm(*[tt[h].astype(np.int16) for h in watch_a1]),
            published=tt_pub("primary", "global", "watch_a1"),
        )
    )
    add(
        Method(
            name="twotier:ACT_b",
            family="two_tier",
            threshold_mode="global",
            runtime_task_identity=False,
            source="moe-two-tier-0906/results/external_first_alarms.npz (2-of-3 rebuilt)",
            config="2-of-3 (latching) of " + " + ".join(act_b),
            first=k_of_n_alarm([tt[h].astype(np.int16) for h in act_b], 2),
            published=tt_pub("primary", "global", "act_b"),
        )
    )

    # -------------------------------------------------- constructed combos ----
    fs = arrays
    add(
        Method(
            name="build:OR_crossframe_global",
            family="constructed",
            threshold_mode="global",
            runtime_task_identity=False,
            source="cross-frame OR built here from the frame-survey arrays",
            config="mobility|global OR flow_path|global OR expert_load_effective_rank|global",
            first=or_alarm(
                fs["mobility|global"].astype(np.int16),
                fs["flow_path|global"].astype(np.int16),
                fs["expert_load_effective_rank|global"].astype(np.int16),
            ),
            published=(381, 54),  # controller reference, republished in two-tier
        )
    )
    add(
        Method(
            name="build:AND_mobility_expertload_global",
            family="constructed",
            threshold_mode="global",
            runtime_task_identity=False,
            source="AND built here from the frame-survey arrays",
            config="mobility|global AND expert_load_effective_rank|global",
            first=and_alarm(
                fs["mobility|global"].astype(np.int16),
                fs["expert_load_effective_rank|global"].astype(np.int16),
            ),
            published=(69, 3),
        )
    )
    add(
        Method(
            name="build:AND_mobility_flowsettling_per_task",
            family="constructed",
            threshold_mode="per_task",
            runtime_task_identity=True,
            source="AND built here from the frame-survey arrays",
            config="mobility|per_task AND flow_settling_log_ratio|per_task",
            first=and_alarm(
                fs["mobility|per_task"].astype(np.int16),
                fs["flow_settling_log_ratio|per_task"].astype(np.int16),
            ),
            published=(187, 8),
        )
    )

    # -------------------------------------------------- prior replay bundle ---
    from reconcile_prior_replay import (
        join_prior_alarm,
        length_only_methods,
        transferred_length_only,
    )

    prior_methods, join_report = join_prior_alarm(cohort)
    for m in prior_methods:
        add(m)
    for m in (
        length_only_methods(cohort, "suite")
        + length_only_methods(cohort, "task")
        + transferred_length_only(cohort)
    ):
        add(m)

    notes = {
        "v7_guard_is_or_of_three_branches": guard_is_or3,
        "v7_guard_is_or_of_freeze_and_turbulence": bool(
            np.array_equal(
                or_alarm(v7[f"{prefix}_freeze"], v7[f"{prefix}_turbulence"]),
                v7[f"{prefix}_guard"].astype(np.int16),
            )
        ),
        "state_mobility_s9_equals_state_mobility_mean": state_mobility_duplicate,
        "bundle_disagreement_heads": pes_note,
        "prior_replay_join": join_report,
    }
    return methods, discrepancies, skipped, notes  # type: ignore[return-value]
