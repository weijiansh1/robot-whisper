#!/usr/bin/env python3
"""Stages 1 and 2: does the action chunk follow the object at all, and after closure?

Runs the frozen policy once per counterfactual row and reduces the chunks to the
readouts ``PROTOCOL.md`` registered.  No patching, no internals: this establishes
the behaviour that later stages try to explain, and it is the gate on whether
those stages are worth running.

Stage 1 asks whether displacing the target before the grasp moves the command
toward it.  If it does not, there is no working evidence-to-action computation in
this behaviour and the study stops.

Stage 2 measures the same policy's response, on the same scale, to the target
being absent from the gripper after closure.  The registered expectation is that
it is small relative to the shift dose-response; the reportable quantity is the
ratio, not a significance test against a null.

Two checks run alongside and are hard validity gates rather than results:

* determinism -- a row re-queried with the same fixed flow noise must return the
  same action, otherwise every paired contrast is confounded by sampling;
* historical agreement -- ``held`` re-queried from the restored state against the
  action the capture recorded.  This is a diagnostic, not a gate: RGB is
  regenerated rather than replayed, so exact agreement was never expected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from readout import (  # noqa: E402
    chunk_distance,
    cluster_bootstrap,
    direction_cosine,
    gripper_mean,
    tracking_cosine,
    translation_norm,
)


SCHEMA = "himoe.coupling_action_response.v1"
IMAGE_KEY = "observation/image"
WRIST_IMAGE_KEY = "observation/wrist_image"
STATE_KEY = "observation/state"
PROMPT_KEY = "prompt"
ACTION_KEY = "actions"
FLOW_NOISE_KEY = "flow/noise"

DETERMINISM_ROWS = 8
DETERMINISM_TOLERANCE = 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_policy(args: argparse.Namespace) -> Any:
    if args.allow_cpu:
        # moevla's create_trained_policy picks its device from
        # torch.cuda.is_available() and ignores require_cuda, so hiding the GPUs
        # is the only way to force CPU. Must happen before torch initialises.
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from himoe_libero_bridge.policies import HiMoEPolicy

    if not args.allow_cpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; pass --allow-cpu only for smoke tests")
    return HiMoEPolicy(
        checkpoint_dir=str(args.checkpoint_dir),
        suite=args.suite,
        upstream_root=str(args.upstream_root),
        require_cuda=not args.allow_cpu,
        libero_wrist_layout=args.wrist_layout,
    )


def query(policy: Any, row: dict[str, np.ndarray], prompt: str) -> np.ndarray:
    response = policy.infer(
        {
            IMAGE_KEY: row["image"],
            WRIST_IMAGE_KEY: row["wrist_image"],
            STATE_KEY: row["state"],
            PROMPT_KEY: prompt,
            FLOW_NOISE_KEY: row["flow_noise"],
        }
    )
    action = np.asarray(response[ACTION_KEY], np.float32)
    if action.shape != (10, 7):
        raise RuntimeError("unexpected action shape %s" % (action.shape,))
    return action


def per_row_readouts(
    action: np.ndarray, eef_xyz: np.ndarray, target_xyz: np.ndarray
) -> dict[str, float]:
    return {
        "B_dir": direction_cosine(action, eef_xyz, target_xyz),
        "B_grip": gripper_mean(action),
        "B_mag": translation_norm(action),
    }


def contrasts(
    rows: list[dict[str, Any]], actions: np.ndarray, shift_direction: dict[int, np.ndarray]
) -> list[dict[str, Any]]:
    """One record per (unit, non-held condition), paired against that unit's held row."""
    by_unit: dict[int, dict[str, int]] = {}
    for index, row in enumerate(rows):
        by_unit.setdefault(int(row["unit_id"]), {})[str(row["condition"])] = index

    records = []
    for unit, conditions in sorted(by_unit.items()):
        if "held" not in conditions:
            raise RuntimeError("unit %d has no held row" % unit)
        held_index = conditions["held"]
        held_action = actions[held_index]
        held_row = rows[held_index]
        held_readout = per_row_readouts(
            held_action, held_row["eef_xyz"], held_row["target_xyz"]
        )
        for condition, index in sorted(conditions.items()):
            if condition == "held":
                continue
            row = rows[index]
            action = actions[index]
            readout = per_row_readouts(action, row["eef_xyz"], row["target_xyz"])
            record = {
                "unit_id": unit,
                "anchor": str(row["anchor"]),
                "condition": condition,
                "episode_index": int(row["episode_index"]),
                "query_index": int(row["query_index"]),
                "chunk_distance": chunk_distance(action, held_action),
            }
            for name, value in readout.items():
                record[name] = value
                record["%s_held" % name] = held_readout[name]
                record["delta_%s" % name] = value - held_readout[name]
            direction = shift_direction.get(unit)
            record["B_track"] = (
                tracking_cosine(action, held_action, direction)
                if direction is not None
                else float("nan")
            )
            records.append(record)
    return records


def summarise(records: list[dict[str, Any]], *, resamples: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    conditions = sorted({record["condition"] for record in records})
    for condition in conditions:
        selected = [record for record in records if record["condition"] == condition]
        clusters = [record["episode_index"] for record in selected]
        entry: dict[str, Any] = {
            "n_units": len(selected),
            "anchor": sorted({record["anchor"] for record in selected}),
        }
        for quantity in ("delta_B_dir", "delta_B_grip", "delta_B_mag", "B_track", "chunk_distance"):
            entry[quantity] = cluster_bootstrap(
                [record[quantity] for record in selected],
                clusters,
                resamples=resamples,
                seed=seed,
            )
        out[condition] = entry
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.inputs, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    audit = json.loads(args.inputs_audit.read_text(encoding="utf-8"))
    prompt = str(audit["prompt"])
    n_rows = int(arrays["image"].shape[0])
    if args.limit_units is not None:
        # Smoke path: keep whole units, never a partial one, so every retained
        # condition still has its held row to pair against.
        keep = sorted({int(u) for u in arrays["unit_id"]})[: args.limit_units]
        mask = np.isin(arrays["unit_id"], keep)
        arrays = {name: value[mask] if len(np.shape(value)) and np.shape(value)[0] == n_rows
                  else value for name, value in arrays.items()}
        n_rows = int(arrays["image"].shape[0])

    rows: list[dict[str, Any]] = []
    for index in range(n_rows):
        rows.append(
            {
                "unit_id": int(arrays["unit_id"][index]),
                "condition": str(arrays["condition"][index]),
                "anchor": str(arrays["anchor"][index]),
                "episode_index": int(arrays["episode_index"][index]),
                "query_index": int(arrays["query_index"][index]),
                "target_xyz": arrays["target_xyz"][index],
                "eef_xyz": arrays["eef_xyz"][index],
            }
        )

    shift_direction: dict[int, np.ndarray] = {}
    for unit in sorted({row["unit_id"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["unit_id"] == unit]
        held = [i for i in indices if rows[i]["condition"] == "held"]
        shifted = [i for i in indices if rows[i]["condition"].startswith("shift")]
        if held and shifted:
            # Every shift of a unit is along one direction; take the largest,
            # which has the best-conditioned difference.
            largest = max(shifted, key=lambda i: int(rows[i]["condition"][5:]))
            delta = np.asarray(rows[largest]["target_xyz"], np.float64) - np.asarray(
                rows[held[0]]["target_xyz"], np.float64
            )
            norm = float(np.linalg.norm(delta[:2]))
            if norm > 0:
                shift_direction[unit] = delta / norm

    # Check the noise contract before paying two minutes to load 7.6 GB of
    # weights.  A builder that hardcoded the wrong internal action dim produced
    # (10, 32) noise and only failed deep inside the first forward pass.
    from himoe_libero_bridge.protocol import FLOW_NOISE_SHAPE

    noise_shape = tuple(int(value) for value in arrays["flow_noise"].shape[1:])
    if noise_shape != tuple(int(value) for value in FLOW_NOISE_SHAPE):
        raise RuntimeError(
            "inputs carry flow noise of shape %s, protocol requires %s; rebuild "
            "the counterfactuals" % (noise_shape, tuple(FLOW_NOISE_SHAPE))
        )

    policy = build_policy(args)
    started = time.perf_counter()
    actions = np.zeros((n_rows, 10, 7), np.float32)
    for index in range(n_rows):
        actions[index] = query(
            policy,
            {
                "image": arrays["image"][index],
                "wrist_image": arrays["wrist_image"][index],
                "state": arrays["state"][index],
                "flow_noise": arrays["flow_noise"][index],
            },
            prompt,
        )
        if (index + 1) % 25 == 0 or index + 1 == n_rows:
            print("queried %d/%d" % (index + 1, n_rows), flush=True)
    elapsed = time.perf_counter() - started

    repeat_indices = list(range(0, n_rows, max(1, n_rows // DETERMINISM_ROWS)))[:DETERMINISM_ROWS]
    repeat_errors = []
    for index in repeat_indices:
        again = query(
            policy,
            {
                "image": arrays["image"][index],
                "wrist_image": arrays["wrist_image"][index],
                "state": arrays["state"][index],
                "flow_noise": arrays["flow_noise"][index],
            },
            prompt,
        )
        repeat_errors.append(float(np.abs(again - actions[index]).max()))
    determinism_ok = bool(max(repeat_errors) <= DETERMINISM_TOLERANCE) if repeat_errors else False

    historical_errors = [
        float(np.abs(actions[index] - arrays["historical_action"][index]).max())
        for index in range(n_rows)
        if rows[index]["condition"] == "held"
    ]

    records = contrasts(rows, actions, shift_direction)
    summary_by_condition = summarise(records, resamples=args.resamples, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "actions.npz",
        actions=actions,
        unit_id=arrays["unit_id"],
        condition=arrays["condition"],
        anchor=arrays["anchor"],
        episode_index=arrays["episode_index"],
    )
    fields = sorted({key for record in records for key in record})
    with (args.out_dir / "contrasts.csv").open("w", encoding="utf-8") as stream:
        stream.write(",".join(fields) + "\n")
        for record in records:
            stream.write(",".join(str(record.get(field, "")) for field in fields) + "\n")

    summary = {
        "schema": SCHEMA,
        "protocol": "moe-coupling-circuit-0906/PROTOCOL.md",
        "inputs": str(args.inputs.resolve()),
        "inputs_sha256": sha256_file(args.inputs),
        "task_key": audit["task_key"],
        "prompt": prompt,
        "checkpoint_dir": str(args.checkpoint_dir),
        "wrist_layout": args.wrist_layout,
        "allow_cpu": bool(args.allow_cpu),
        "policy_metadata": jsonable(getattr(policy, "metadata", {})),
        "rows": n_rows,
        "units": int(len({row["unit_id"] for row in rows})),
        "units_by_anchor": {
            anchor: len({row["unit_id"] for row in rows if row["anchor"] == anchor})
            for anchor in sorted({row["anchor"] for row in rows})
        },
        "seconds_per_query": elapsed / max(1, n_rows),
        "validity": {
            "determinism_rows": len(repeat_indices),
            "determinism_max_abs_error": max(repeat_errors) if repeat_errors else None,
            "determinism_passed": determinism_ok,
            "historical_agreement_max_abs_error": max(historical_errors) if historical_errors else None,
            "historical_agreement_median_abs_error": (
                float(np.median(historical_errors)) if historical_errors else None
            ),
            "historical_agreement_is_a_gate": False,
        },
        "by_condition": summary_by_condition,
        "limitations": [
            "Behaviour only. Nothing here localises a computation.",
            "Edited states are restorations, not states the simulator stepped into.",
            "A small response is not evidence of no response; the reportable quantity "
            "is its size against the shift dose-response.",
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(jsonable({"validity": summary["validity"], "by_condition": summary_by_condition}), indent=2, sort_keys=True))
    if not determinism_ok:
        raise RuntimeError(
            "determinism check failed (max abs error %r); paired contrasts are not "
            "interpretable" % (max(repeat_errors) if repeat_errors else None)
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--wrist-layout", default="paper-right")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--limit-units", type=int, help="smoke tests only")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "smoke tests only, and it does not currently reach an action: this "
            "checkpoint's AS gate casts its input to bfloat16 while the weight "
            "stays float32, which only works under CUDA autocast "
            "(modeling_moe.py:132). It still validates loading and the "
            "observation and flow-noise contracts. A CPU run would not be a "
            "valid result in any case and is recorded as such."
        ),
    )
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
