"""Validate a paired HB5/d0 v4 store and report frozen mechanism effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from himoe_hb5_intervention import ARM_TO_CODE
from himoe_intervention_store import (
    ZarrHB5InterventionReader,
    paired_mechanism_metrics,
    validate_paired_store,
)


def validate(path: str | Path) -> dict:
    reader = ZarrHB5InterventionReader(str(path))
    summary = validate_paired_store(reader)
    pair = np.asarray(reader["pair_id"][:], dtype=np.int64)
    draw = np.asarray(reader["draw_id"][:], dtype=np.int64)
    arm = np.asarray(reader["arm"][:], dtype=np.int64)
    original = np.asarray(reader["original_routed"][:], dtype=np.float64)
    executed = np.asarray(reader["executed_routed"][:], dtype=np.float64)
    delta = np.asarray(reader["intervention_delta"][:], dtype=np.float64)
    raw = np.asarray(reader["original_selected_expert_raw"][:], dtype=np.float64)
    weights = np.asarray(reader["original_selected_expert_weight"][:], dtype=np.float64)
    slots = np.asarray(reader["dropped_slot"][:], dtype=np.int64)

    max_identity_error = float(np.max(np.abs(executed - original - delta)))
    baseline_rows = arm == ARM_TO_CODE["baseline"]
    max_baseline_noop_error = float(
        max(
            np.max(np.abs(executed[baseline_rows] - original[baseline_rows])),
            np.max(np.abs(delta[baseline_rows])),
        )
    )
    max_drop_formula_error = 0.0
    max_random_norm_error = 0.0
    max_random_radial_error = 0.0
    max_random_post_norm_error = 0.0
    keys = sorted(set(zip(pair, draw, strict=True)))
    for key in keys:
        selected = np.flatnonzero((pair == key[0]) & (draw == key[1]))
        by_arm = {int(arm[index]): int(index) for index in selected}
        drop_index = by_arm[ARM_TO_CODE["drop"]]
        random_index = by_arm[ARM_TO_CODE["random"]]
        drop_slot = slots[drop_index]
        dropped_vector = np.take_along_axis(
            raw[drop_index], drop_slot[:, None, None], axis=1
        ).squeeze(1)
        dropped_weight = np.take_along_axis(
            weights[drop_index], drop_slot[:, None], axis=1
        ).squeeze(1)
        expected_drop = (
            original[drop_index] - dropped_weight[:, None] * dropped_vector
        ) / (1.0 - dropped_weight[:, None])
        max_drop_formula_error = max(
            max_drop_formula_error,
            float(np.max(np.abs(executed[drop_index] - expected_drop))),
        )
        target = delta[drop_index]
        control = delta[random_index]
        r = original[drop_index]
        max_random_norm_error = max(
            max_random_norm_error,
            float(
                np.max(
                    np.abs(
                        np.linalg.norm(control, axis=-1)
                        - np.linalg.norm(target, axis=-1)
                    )
                )
            ),
        )
        max_random_radial_error = max(
            max_random_radial_error,
            float(np.max(np.abs(np.sum((control - target) * r, axis=-1)))),
        )
        max_random_post_norm_error = max(
            max_random_post_norm_error,
            float(
                np.max(
                    np.abs(
                        np.linalg.norm(r + control, axis=-1)
                        - np.linalg.norm(r + target, axis=-1)
                    )
                )
            ),
        )
    return {
        **summary,
        "drop_policy": reader.meta["drop_policy"],
        "max_original_runtime_reconstruction_error": float(
            np.max(reader["original_reconstruction_max_abs_error"][:])
        ),
        "max_executed_equals_original_plus_delta_error": max_identity_error,
        "max_baseline_noop_error": max_baseline_noop_error,
        # This audit reconstructs from fp16 stored v_e; the live intervention
        # and runtime reconstruction were computed from fp32 tensors.
        "max_drop_formula_error_from_stored_fp16_raw": max_drop_formula_error,
        "max_random_delta_norm_match_error": max_random_norm_error,
        "max_random_dot_r_match_error": max_random_radial_error,
        "max_random_post_routed_norm_match_error": max_random_post_norm_error,
        "mechanism_effects": paired_mechanism_metrics(reader),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = validate(args.store)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
