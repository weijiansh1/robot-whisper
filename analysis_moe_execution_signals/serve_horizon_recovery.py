#!/usr/bin/env python3
"""Serve one LIBERO-long policy on each selected GPU for recovery collection."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ONLINE_SERVERS = ROOT / "online-servers"
if str(ONLINE_SERVERS) not in sys.path:
    sys.path.insert(0, str(ONLINE_SERVERS))

import serve_model_bundle as bundle  # noqa: E402
import serve_model_matrix as matrix  # noqa: E402


def _ports(value: str) -> tuple[int, ...]:
    try:
        ports = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ports must be comma-separated integers"
        ) from error
    if (
        not ports
        or len(ports) != len(set(ports))
        or any(not 1 <= port <= 65535 for port in ports)
    ):
        raise argparse.ArgumentTypeError("ports must be unique values in 1..65535")
    return ports


def ensure_max_power(gpus: tuple[int, ...]) -> dict[int, tuple[float, float]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,power.limit,power.max_limit",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    observed = {}
    for line in output.splitlines():
        index, current, maximum = (part.strip() for part in line.split(","))
        observed[int(index)] = (float(current), float(maximum))
    selected = {gpu: observed[gpu] for gpu in gpus}
    below = {gpu: values for gpu, values in selected.items() if values[0] != values[1]}
    if below:
        for gpu, (_current, maximum) in below.items():
            subprocess.run(
                ["nvidia-smi", "-i", str(gpu), "--power-limit", str(maximum)],
                check=True,
            )
        return ensure_max_power(gpus)
    return selected


def build_parser() -> argparse.ArgumentParser:
    home = Path("/home/jovyan")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus", type=matrix.parse_gpus, default=matrix.parse_gpus("0,1,2")
    )
    parser.add_argument("--ports", type=_ports, default=_ports("8930,8931,8932"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--libero-cache", default=str(home / ".cache/himoe-libero-bridge")
    )
    parser.add_argument(
        "--upstream-root",
        default=str(home / ".cache/himoe-libero-bridge/upstream/HiMoE-VLA"),
    )
    parser.add_argument(
        "--calvin-cache", default=str(home / ".cache/himoe-calvin-alignment")
    )
    parser.add_argument(
        "--calvin-root",
        default=str(home / ".cache/himoe-calvin-alignment/upstream/calvin"),
    )
    parser.add_argument(
        "--bridge-src", default=str(home / "work/himoe-libero-wrist-fix/src")
    )
    parser.add_argument(
        "--route-capture-src", default=str(ROOT / "himoe-route-capture")
    )
    parser.add_argument(
        "--calvin-src", default=str(home / "work/himoe-calvin-alignment/src")
    )
    parser.add_argument(
        "--openpi-src",
        default=str(
            home
            / ".cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src"
        ),
    )
    parser.add_argument("--libero-wrist-layout", default="paper-right")
    parser.add_argument("--calvin-wrist-layout", default="released-left")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.gpus) != len(args.ports):
        raise ValueError("--gpus and --ports must have equal lengths")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
    power = ensure_max_power(args.gpus)
    bundle._install_source_paths(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        force=True,
    )
    logging.info("verified maximum GPU power limits: %s", power)
    records = tuple(
        matrix.MatrixEndpoint(
            physical_gpu=physical_gpu,
            logical_gpu=logical_gpu,
            model=bundle.ModelEndpoint(name="long", kind="libero", port=port),
        )
        for logical_gpu, (physical_gpu, port) in enumerate(zip(args.gpus, args.ports))
    )
    policies = matrix._load_matrix(args, records)
    return matrix._serve_matrix(args, records, policies)


if __name__ == "__main__":
    raise SystemExit(main())
