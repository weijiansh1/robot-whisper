from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BUNDLE = Path(__file__).resolve().parents[1]
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "experiments"))
sys.path.insert(0, str(BUNDLE / "method"))

import loso_folds  # noqa: E402
from evaluate_intrinsic_guard_v7 import (  # noqa: E402
    EXTERNAL_FEATURE,
    EXTERNAL_LAYER,
    EXTRA_FEATURE,
    EXTRA_LAYER,
    LABEL_ROOT,
    MAIN_FEATURE,
    MAIN_LAYER,
    aligned_labels,
    combine_reference,
    feature,
    load_npz,
    sha256,
)


RESULT = BUNDLE / "results/intrinsic_guard_v7"
LOSO = BUNDLE / "results/loso_validation"

EXPECTED_CALIBRATION = {
    "libero_goal": 12_000,
    "libero_long": 12_000,
    "libero_object": 12_000,
    "libero_spatial": 12_000,
}
EXPECTED_DEVELOPMENT = {
    "libero_goal": 10_800,
    "libero_long": 12_000,
    "libero_object": 10_800,
    "libero_spatial": 10_800,
}
EXPECTED_EVALUATION = {
    "libero_goal": 4_000,
    "libero_long": 4_000,
    "libero_object": 3_600,
    "libero_spatial": 4_000,
}

PINNED_L1_RECALL = 0.623
PINNED_L1_PRECISION = 0.8716
PINNED_L2_RECALL = 0.6293
PINNED_L2_PRECISION = 0.8804


def full_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return combine_reference(
        load_npz(MAIN_LAYER),
        load_npz(EXTRA_LAYER),
        load_npz(MAIN_FEATURE),
        load_npz(EXTRA_FEATURE),
    )


def published_constants() -> loso_folds.FoldConstants:
    mobility, acceleration, periodicity = full_reference()
    scale = loso_folds.periodicity_scale_of(periodicity)
    peaks = loso_folds.reference_peaks(mobility, acceleration, periodicity, scale)
    return loso_folds.constants_from_peaks(
        peaks, scale, *loso_folds.PUBLISHED_QUANTILES
    )


def test_fold_sizes_match_the_spec_table() -> None:
    main_layer = load_npz(MAIN_LAYER)
    extra_layer = load_npz(EXTRA_LAYER)
    external_layer = load_npz(EXTERNAL_LAYER)
    reference_suite = np.concatenate(
        (loso_folds.suite_of(main_layer), loso_folds.suite_of(extra_layer))
    )
    main_suite = loso_folds.suite_of(main_layer)
    external_suite = loso_folds.suite_of(external_layer)

    assert set(loso_folds.SUITES) == set(np.unique(reference_suite))
    for held_out in loso_folds.SUITES:
        assert int((reference_suite != held_out).sum()) == EXPECTED_CALIBRATION[held_out]
        assert int((main_suite != held_out).sum()) == EXPECTED_DEVELOPMENT[held_out]
        assert int((external_suite == held_out).sum()) == EXPECTED_EVALUATION[held_out]


def test_calibration_never_contains_a_held_out_task() -> None:
    main_layer = load_npz(MAIN_LAYER)
    extra_layer = load_npz(EXTRA_LAYER)
    reference_task = np.concatenate(
        (loso_folds.task_of(main_layer), loso_folds.task_of(extra_layer))
    )
    reference_suite = np.concatenate(
        (loso_folds.suite_of(main_layer), loso_folds.suite_of(extra_layer))
    )
    for held_out in loso_folds.SUITES:
        calibration = set(reference_task[reference_suite != held_out])
        held = set(reference_task[reference_suite == held_out])
        assert calibration & held == set()
        assert len(held) > 0


def test_full_corpus_calibration_reproduces_the_published_profile() -> None:
    constants = published_constants()
    with np.load(RESULT / "global_profile.npz", allow_pickle=False) as published:
        for field in (
            "freeze_threshold",
            "acceleration_threshold",
            "periodicity_threshold",
            "periodicity_scale",
        ):
            np.testing.assert_array_equal(
                np.float32(getattr(constants, field)), published[field]
            )


def test_published_constants_reproduce_the_sealed_external_alarms() -> None:
    constants = published_constants()
    external_layer = load_npz(EXTERNAL_LAYER)
    external_feature = load_npz(EXTERNAL_FEATURE)
    alarms = loso_folds.guard_alarms(
        loso_folds.cohort_scores(
            external_layer["mobility"],
            feature(external_feature, "route_acceleration"),
            feature(external_feature, "lag_periodicity"),
            constants.periodicity_scale,
        ),
        external_layer["valid"].astype(bool),
        constants,
    )
    with np.load(RESULT / "sealed_first_alarms.npz", allow_pickle=False) as sealed:
        for name in ("freeze", "acceleration", "periodicity", "turbulence", "guard"):
            np.testing.assert_array_equal(alarms[name], sealed[f"external_{name}"])


def test_fold_selection_grid_is_640_points_and_uses_only_calibration_suites() -> None:
    held_out = "libero_object"
    main_layer = load_npz(MAIN_LAYER)
    main_feature = load_npz(MAIN_FEATURE)
    mobility, acceleration, periodicity = full_reference()
    keep = loso_folds.suite_of(main_layer) != held_out

    scale = loso_folds.periodicity_scale_of(periodicity)
    peaks = loso_folds.reference_peaks(mobility, acceleration, periodicity, scale)
    development = loso_folds.cohort_scores(
        main_layer["mobility"][keep],
        feature(main_feature, "route_acceleration")[keep],
        feature(main_feature, "lag_periodicity")[keep],
        scale,
    )
    labels = (
        aligned_labels(
            main_layer,
            LABEL_ROOT / "development_main_clean_labels.csv",
            "development_main",
        )
        .loc[keep]
        .reset_index(drop=True)
    )

    candidates, primary = loso_folds.select_fold_operating_point(
        peaks, development, main_layer["valid"].astype(bool)[keep], labels
    )
    assert len(candidates) == 640
    assert len(labels) == EXPECTED_DEVELOPMENT[held_out]
    assert set(labels["suite"]) == set(loso_folds.SUITES) - {held_out}
    assert primary is None or {
        "freeze_quantile",
        "acceleration_quantile",
        "periodicity_quantile",
    } <= set(primary.index)


def test_fold_selection_used_only_calibration_suite_episodes() -> None:
    selection = json.loads((LOSO / "fold_selection.json").read_text())
    for held_out, record in selection.items():
        assert record["development_episodes"] == EXPECTED_DEVELOPMENT[held_out]
        assert record["calibration_episodes"] == EXPECTED_CALIBRATION[held_out]
        assert held_out not in record["calibration_suites"]
        assert record["candidate_count"] == 640


def test_manifest_declares_the_seal_order() -> None:
    manifest = json.loads((LOSO / "loso_manifest.json").read_text())
    assert manifest["held_out_outcomes_used_for_calibration"] is False
    assert manifest["held_out_outcomes_used_for_selection"] is False
    assert manifest["external_outcomes_loaded_after_seal"] is True
    assert manifest["external_cohort_pristine_holdout"] is False
    assert sorted(manifest["folds"]) == sorted(loso_folds.SUITES)


def test_drift_table_covers_every_fold_level_and_constant() -> None:
    drift = pd.read_csv(LOSO / "threshold_drift.csv")
    constants = {
        "freeze_threshold",
        "acceleration_threshold",
        "periodicity_threshold",
        "periodicity_scale",
    }
    assert set(drift["constant"]) == constants
    assert set(drift["held_out_suite"]) == set(loso_folds.SUITES)
    assert drift["fold_value"].notna().all()
    for held_out in loso_folds.SUITES:
        block = drift[
            (drift["held_out_suite"] == held_out) & (drift["level"] == "loso_l1")
        ]
        assert set(block["constant"]) == constants


def test_published_level_reproduces_the_v7_per_suite_counts() -> None:
    new = pd.read_csv(LOSO / "loso_metrics.csv")
    new = new[new["level"] == "published"].set_index("held_out_suite")
    old = pd.read_csv(RESULT / "outcome_metrics_by_suite.csv")
    old = old[
        (old["cohort"] == "external_8b") & (old["detector"] == "intrinsic_guard_v7")
    ].set_index("group")
    for suite in loso_folds.SUITES:
        assert int(new.loc[suite, "tp"]) == int(old.loc[suite, "tp"])
        assert int(new.loc[suite, "fp"]) == int(old.loc[suite, "fp"])


def test_provenance_hashes_match_the_files_on_disk() -> None:
    manifest = json.loads((LOSO / "loso_manifest.json").read_text())
    artifacts = manifest["artifacts"]
    assert artifacts["fold_profiles_sha256"] == sha256(LOSO / "fold_profiles.npz")
    assert artifacts["threshold_drift_sha256"] == sha256(LOSO / "threshold_drift.csv")
    assert artifacts["fold_selection_sha256"] == sha256(LOSO / "fold_selection.json")
    assert artifacts["loso_metrics_sha256"] == sha256(LOSO / "loso_metrics.csv")
    assert artifacts["folds_module_sha256"] == sha256(
        BUNDLE / "experiments/loso_folds.py"
    )


def test_loso_macro_summary_is_pinned() -> None:
    macro = json.loads((LOSO / "loso_summary.json").read_text())["macro"]
    assert macro["loso_l1"]["folds"] == 4
    assert macro["loso_l2"]["folds"] == 4
    assert round(macro["loso_l1"]["risk_recall"], 4) == PINNED_L1_RECALL
    assert round(macro["loso_l1"]["precision"], 4) == PINNED_L1_PRECISION
    assert round(macro["loso_l2"]["risk_recall"], 4) == PINNED_L2_RECALL
    assert round(macro["loso_l2"]["precision"], 4) == PINNED_L2_PRECISION
