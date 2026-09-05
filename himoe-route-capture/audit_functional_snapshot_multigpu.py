#!/usr/bin/env python3
"""Load Long once, migrate it across GPUs, and run the rich no-op gate per GPU."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
CAPTURE_ROOT = ROOT / "himoe-route-capture"


def parse_gpus(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("gpus must be comma-separated integers") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("gpus must be a non-empty unique list")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("2,3,4,5,6,7"))
    parser.add_argument("--suite", choices=("goal", "spatial", "object", "long"), default="long")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--sketch-dim", type=int, default=16)
    parser.add_argument("--prompt", default="put the red mug on the plate")
    parser.add_argument(
        "--checkpoint-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge/checkpoints"),
    )
    parser.add_argument(
        "--upstream-root",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA"
        ),
    )
    parser.add_argument("--libero-wrist-layout", default="checkpoint-right")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=CAPTURE_ROOT / "runs" / "functional-noop-multigpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
    sys.path.insert(0, str(CAPTURE_ROOT))

    import torch

    from audit_functional_snapshot_noop import _load_policy, run

    if torch.cuda.device_count() != len(args.gpus):
        raise RuntimeError(
            "expected %d visible devices, found %d"
            % (len(args.gpus), torch.cuda.device_count())
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    load_args = argparse.Namespace(**vars(args), gpu=args.gpus[0])
    policy, load_audit = _load_policy(load_args)
    records = []
    previous_device = 0
    try:
        for logical_gpu, physical_gpu in enumerate(args.gpus):
            if logical_gpu:
                policy._policy.model.to(torch.device("cuda:%d" % logical_gpu))
                torch.cuda.synchronize(logical_gpu)
                torch.cuda.empty_cache()
            torch.cuda.set_device(logical_gpu)
            run_args = argparse.Namespace(
                **{
                    **vars(args),
                    "gpu": physical_gpu,
                    "snapshot": args.out_dir / ("gpu%d_snapshot.npz" % physical_gpu),
                    "out": args.out_dir / ("gpu%d_audit.json" % physical_gpu),
                }
            )
            result = run(
                run_args,
                loaded_policy=policy,
                load_audit=load_audit,
            )
            records.append(result)
            if logical_gpu:
                with torch.cuda.device(previous_device):
                    torch.cuda.empty_cache()
            previous_device = logical_gpu
            print(
                "gpu%d passed=%s device=%s action=%s"
                % (
                    physical_gpu,
                    result["passed"],
                    result["model_device"],
                    result["action_sha256"]["C"][:16],
                ),
                flush=True,
            )
    finally:
        del policy
        for logical_gpu in range(torch.cuda.device_count()):
            with torch.cuda.device(logical_gpu):
                torch.cuda.empty_cache()

    action_hashes = {row["action_sha256"]["C"] for row in records}
    manifest = {
        "schema": "himoe.functional_snapshot.multigpu_audit.v1",
        "passed": bool(records and all(row["passed"] for row in records)),
        "physical_gpus": list(args.gpus),
        "power_limit_note": "All H20-3e devices report the hardware maximum 500 W limit.",
        "same_fixed_input_action_across_gpus": len(action_hashes) == 1,
        "records": [
            {
                "physical_gpu": row["physical_gpu"],
                "model_device": row["model_device"],
                "passed": row["passed"],
                "action_sha256": row["action_sha256"]["C"],
                "snapshot": row["snapshot"],
                "snapshot_file_bytes": row["snapshot_file_bytes"],
                "ranges": row["ranges"],
                "timing_seconds": row["timing_seconds"],
            }
            for row in records
        ],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("passed=%s; wrote %s" % (manifest["passed"], manifest_path), flush=True)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
