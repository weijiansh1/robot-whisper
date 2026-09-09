"""Calibrate and evaluate fixed relative constraints on existing routing data."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from relative_direction_models import COMPONENTS, METHODS, PRIMARY, SCHEMA, WINDOW, make_components, model_scores, reference_evidence
from guard import ALPHAS, KINDS, PeakBank, first_trigger
from model_tables import alarm_tables, auroc_tables
from run_guard_experiment import BASE, HERE, RUN_B, verified_inputs
from run_analysis import digest, state_splits, write_json

BASELINE = BASE / "state_action_confirmation_20260909"
PREVIOUS = BASE / "causal_routing_models_20260909"
CONTROL_NAMES = ("old_ac", "previous_absolute_acc", "previous_leaky_state_trap", "previous_persistent_ac")


def inputs():
    frame, features, valid, hashes = verified_inputs()
    for directory, names in ((PREVIOUS, ("predictions.npz", "query_auroc.csv")),):
        manifest = json.loads((directory / "final_verification.json").read_text())
        for name in names:
            path = directory / name
            assert digest(path) == manifest["artifacts"][name], path
            hashes[str(path)] = digest(path)
    components = make_components(features, valid)
    return frame, valid, components, hashes


def calibrate(frame, valid, components, out):
    profiles, thresholds = [], []
    baseline_manifest = json.loads((BASELINE / "final_verification.json").read_text())["artifacts"]
    for fold, ref, cal, test in state_splits(frame):
        ref_success = ref[~frame.iloc[ref].failure.to_numpy(bool)]
        success = ~frame.iloc[cal].failure.to_numpy(bool)
        refs = {name: PeakBank(np.where(np.isfinite(components[name][ref_success]), components[name][ref_success], -np.inf).max(1))
                for name in COMPONENTS}
        scores = model_scores(reference_evidence({name: value[cal] for name, value in components.items()}, refs), valid[cal])
        index, groups = pd.factorize(pd.MultiIndex.from_frame(frame.iloc[cal].loc[success, ["task", "init_state_id"]]), sort=True)
        banks = {kind: {} for kind in KINDS}
        for name, values in scores.items():
            current = values[success]
            peaks = np.where(np.isfinite(current), current, -np.inf).max(1)
            grouped = np.full(len(groups), -np.inf)
            np.maximum.at(grouped, index, peaks)
            for kind, units in zip(KINDS, (peaks, grouped)):
                bank = PeakBank(units)
                banks[kind][name] = bank.to_list()
                for alpha in ALPHAS:
                    threshold, rank = bank.threshold(alpha)
                    thresholds.append(dict(fold=fold, calibration=kind, method=name, alpha=alpha,
                                           threshold=threshold, rank=rank, units=len(units),
                                           no_exposure_units=int(np.isneginf(units).sum()), attainable=rank<=len(units)))
        base_path = BASELINE / f"fold_{fold}_profile.json"
        assert digest(base_path) == baseline_manifest[base_path.name]
        profile = dict(schema=SCHEMA, fold=fold, methods=list(METHODS), primary=PRIMARY, window=WINDOW,
                       base_profile=os.path.relpath(base_path, out), base_profile_sha256=digest(base_path),
                       references={name: bank.to_list() for name, bank in refs.items()}, banks=banks,
                       reference_rows=ref, reference_success_rows=ref_success, calibration_rows=cal,
                       calibration_success_rows=cal[success], test_rows=test, retrospective=True, no_classifier_training=True)
        write_json(out / f"fold_{fold}_profile.json", profile)
        np.savez_compressed(out / f"fold_{fold}_calibration.npz", rows=cal, success=success, valid=valid[cal],
                            methods=np.asarray(METHODS), scores=np.stack([scores[name] for name in METHODS]))
        profiles.append(json.loads((out / f"fold_{fold}_profile.json").read_text()))
        print(f"CALIBRATED relative models fold={fold}", flush=True)
    pd.DataFrame(thresholds).to_csv(out / "calibration_thresholds.csv", index=False)
    files = [out / "execution_contract.json", out / "calibration_thresholds.csv", *out.glob("fold_*")]
    write_json(out / "calibration_seal.json", dict(phase="before_B_models", artifacts={p.name: digest(p) for p in files}))
    return profiles


def predict(frame, valid, components, profiles, out):
    rows_b = np.flatnonzero(frame.run_id.eq(RUN_B))
    b = frame.iloc[rows_b].reset_index(drop=True)
    assert len(b) == 16000 and b.failure.sum() == 564
    inverse = np.full(len(frame), -1, int)
    inverse[rows_b] = np.arange(len(b))
    scores = np.full((len(METHODS), len(b), valid.shape[1]), np.nan)
    tails = np.full((len(KINDS), *scores.shape), np.nan)
    first = np.full((len(KINDS), len(ALPHAS), len(METHODS), len(b)), -2, np.int16)
    assigned = np.zeros(len(b), bool)
    for profile in profiles:
        rows = np.asarray(profile["test_rows"])
        ix = inverse[rows]
        assert not assigned[ix].any()
        assigned[ix] = True
        refs = {name: PeakBank.from_list(bank) for name, bank in profile["references"].items()}
        current = model_scores(reference_evidence({name: value[rows] for name, value in components.items()}, refs), valid[rows])
        for mi, name in enumerate(METHODS):
            scores[mi, ix] = current[name]
            for ki, kind in enumerate(KINDS):
                tail = PeakBank.from_list(profile["banks"][kind][name]).tail(current[name])
                tails[ki, mi, ix] = tail
                for ai, alpha in enumerate(ALPHAS):
                    first[ki, ai, mi, ix] = first_trigger(tail, valid[rows], alpha)
    assert assigned.all() and (first >= -1).all()
    with np.load(PREVIOUS / "predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(rows_b, z["global_rows"])
        old_index = list(z["methods"]).index("ac_reference")
        np.testing.assert_array_equal(scores[METHODS.index("ac_reference")], z["scores"][old_index])
        np.testing.assert_array_equal(first[:, :, METHODS.index("ac_reference")], z["first"][:, :, old_index])
        np.testing.assert_array_equal(tails[:, METHODS.index("ac_reference")], z["tails"][:, old_index])
        controls = np.stack([z["control_first"][:, :, list(z["control_names"]).index("old_ac")],
                             *[z["first"][:, :, list(z["methods"]).index(name)]
                               for name in ("absolute_acc", "leaky_state_trap", "persistent_ac")]], axis=2)
        frozen, old_ac_rank, v82 = z["frozen_first"], z["old_ac_rank"], z["raw_v82_scores"]
    saved = dict(scores=scores, tails=tails, first=first, frozen_first=frozen, control_first=controls,
                 control_names=np.asarray(CONTROL_NAMES), old_ac_rank=old_ac_rank, raw_v82_scores=v82,
                 global_rows=rows_b, valid=valid[rows_b], methods=np.asarray(METHODS), kinds=np.asarray(KINDS), alphas=np.asarray(ALPHAS))
    np.savez_compressed(out / "predictions.npz", **saved)
    b.to_csv(out / "test_index.csv", index=False)
    exposure = []
    for mi, name in enumerate(METHODS):
        observed = np.isfinite(scores[mi])
        for label in (False, True):
            selected = observed[b.failure.eq(label)]
            q = np.flatnonzero(selected.any(0))
            exposure.append(dict(method=name, failure=label, episodes=int(len(selected)),
                                 exposed_episodes=int(selected.any(1).sum()), queries=int(selected.sum()),
                                 first_query=int(q[0]) if len(q) else -1,
                                 zero_score_queries=int((scores[mi, b.failure.eq(label)][selected] == 0).sum())))
    pd.DataFrame(exposure).to_csv(out / "exposure.csv", index=False)
    print("PREDICTED: AC reference exactly reproduces previous scores, tails and first alarms", flush=True)
    return b, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "relative_direction_models_20260909")
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frame, valid, components, hashes = inputs()
    names = ("relative_direction_models.py", "run_relative_direction_models.py", "model_tables.py", "RELATIVE_DIRECTIONS_PROTOCOL_ZH.md",
             "encoder.py", "guard.py", "state_action.py")
    write_json(out / "execution_contract.json", dict(retrospective=True, no_classifier_training=True,
        primary=dict(method=PRIMARY, calibration="task_init", alpha=.01), methods=METHODS,
        raw_routing_only=True, duration_baseline_computed=False, inherited_v82_query_slope=.0015,
        B_parameter_selection=False, direction_hypotheses_informed_by_previous_B=True, inputs=hashes, sources={name: digest(HERE / name) for name in names}))
    profiles = calibrate(frame, valid, components, out)
    b, saved = predict(frame, valid, components, profiles, out)
    alarm_tables(b, saved, out, "ac_reference")
    auroc_tables(b, saved, out, "ac_reference")
    for name, expected in json.loads((out / "calibration_seal.json").read_text())["artifacts"].items():
        assert digest(out / name) == expected
    write_json(out / "experiment_verification.json", dict(all_checks_passed=True,
        every_B_episode_once=True, reference_alarm_checks=2*4*len(b), calibration_seal_unchanged=True,
        artifacts={p.name: digest(p) for p in out.iterdir() if p.is_file()}))
    metrics = pd.read_csv(out / "alarm_metrics.csv")
    print(metrics.loc[metrics.scope.eq("all") & (metrics.alpha.eq(.01) | metrics.calibration.eq("original")),
                      ["calibration", "method", "tp", "fp", "recall", "fpr"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
