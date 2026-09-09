#!/usr/bin/env python3
"""Manifest: inputs read, code run, outputs written, and the checks that must hold."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import common as C

INPUTS = (
    "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
    "moe-v4-0904/results/layerwise_mobility/external_8b.npz",
    "moe-v4-0904/results/cache16x32_v4/layerwise_mobility.npz",
    "moe-v4-0904/results/cache16x32_v4/episode_alarms.csv",
    "moe-hb-front-back-0905/results/layer_graphs/development_main.npz",
    "moe-hb-front-back-0905/results/layer_graphs/external_8b.npz",
    "moe-hb-front-back-0905/results/layer_graphs/legacy_main16x32.npz",
    "moe-unused-channels-0906/results/channels/development_main_index.npz",
    "moe-unused-channels-0906/results/channels/development_main_quantities.npy",
    "moe-unused-channels-0906/results/channels/external_8b_index.npz",
    "moe-unused-channels-0906/results/channels/external_8b_quantities.npy",
    "moe-flow-semantics-0906/results/step_profiles/development_main_index.npz",
    "moe-flow-semantics-0906/results/step_profiles/development_main_metrics.npy",
    "moe-flow-semantics-0906/results/step_profiles/development_main_mobility.npy",
    "moe-flow-semantics-0906/results/step_profiles/development_main_state_mobility.npy",
    "moe-flow-semantics-0906/results/step_profiles/external_8b_index.npz",
    "moe-flow-semantics-0906/results/step_profiles/external_8b_metrics.npy",
    "moe-flow-semantics-0906/results/step_profiles/external_8b_mobility.npy",
    "moe-flow-semantics-0906/results/step_profiles/external_8b_state_mobility.npy",
    "double-selete/trainfree/results/timeout_extension_plus10/development_main_clean_labels.csv",
    "double-selete/trainfree/results/timeout_extension_plus10/external_8b_clean_labels.csv",
    "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv",
)
ORDER = (
    "within_episode_audit.py",
    "information_map.py",
    "reachability.py",
    "upper_bound.py",
    "frontier.py",
    "budget_allocation.py",
    "summarise.py",
    "make_figure.py",
)


def sha256(path: Path, limit: int = 1 << 30) -> str:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while size < limit:
            block = handle.read(1 << 22)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest()


def describe(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256_first_1GiB": sha256(path),
    }


def main() -> None:
    project = C.PROJECT
    bundle = C.BUNDLE
    inputs = []
    for rel in INPUTS:
        path = project / rel
        inputs.append(
            describe(path, project) if path.exists() else {"path": rel, "missing": True}
        )
    code = [describe(bundle / "experiments" / name, bundle) for name in ORDER]
    code.append(describe(bundle / "experiments" / "common.py", bundle))
    outputs = [
        describe(p, bundle)
        for p in sorted((bundle / "results").iterdir())
        if p.is_file() and p.name != "manifest.json"
    ]
    try:
        versions = subprocess.run(
            ["python", "-c",
             "import numpy,scipy,pandas,sklearn,matplotlib;"
             "print(numpy.__version__,scipy.__version__,pandas.__version__,"
             "sklearn.__version__,matplotlib.__version__)"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        versions = "unavailable"

    C.write_json(
        bundle / "results" / "manifest.json",
        {
            "bundle": "moe-early-window-0906",
            "question": (
                "inside the early window, conditioned on task and on elapsed time, "
                "does MoE routing carry information about eventual failure, and is "
                "80% risk recall at alarm phase <= 0.65 reachable"
            ),
            "run_order": list(ORDER),
            "library_versions_numpy_scipy_pandas_sklearn_matplotlib": versions,
            "seed": C.SEED,
            "cohorts": {
                "development_main": "14800 episodes, 487 risks, 37 tasks",
                "external_8b": "15600 episodes, 564 risks, 39 tasks",
                "legacy_main16x32": (
                    "2560 episodes, 307 risks, 5 tasks; carried through the "
                    "information map but produces zero admissible cells because no "
                    "suite in it has 4 tasks, so it supports no task-stratified "
                    "verdict and is excluded from every headline"
                ),
            },
            "definitions": {
                "risk": "original_failure, i.e. did not finish before the original horizon cap",
                "phase": "(q + 1) / episode_length; risk episodes have length == cap",
                "window": "phase <= 0.65, i.e. chunk <= " + str(C.WINDOW_65),
                "timely_fpr": "false alarms / non-risk episodes, matching the frozen protocol",
                "admissible_cell": "at least 4 contributing tasks and at least 5000 pairs",
            },
            "checks_that_must_hold": {
                "ctrl_const_elapsed_auc": (
                    "exactly 0.5 with variance exactly 0 in every stratum; asserted at "
                    "run time in information_map.py"
                ),
                "leak_full_length_auc": "median 1.0 (definitional leak), see estimator_checks.csv",
                "leak_full_length_recall_at_budget": (
                    "1.0 in all four suites, see reachability_checks.json; recovers the "
                    "brief's statement that any sub-cap length threshold catches every "
                    "failure by construction"
                ),
                "delong_se_vs_permutation_sd": "median ratio 0.970 over 97 probed cells",
                "within_episode_constant_channels": (
                    "excluded from the candidate set, see within_episode_verdict.json"
                ),
            },
            "inputs_read_only": inputs,
            "code": code,
            "outputs": outputs,
            "not_modified": (
                "no file outside moe-early-window-0906/ is written; no git command is run"
            ),
        },
    )
    print("manifest written")


if __name__ == "__main__":
    main()
