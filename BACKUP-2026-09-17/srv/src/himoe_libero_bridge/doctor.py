"""Read-only prerequisite checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any, Dict, Optional

from himoe_libero_bridge.suites import get_suite

def run_doctor(
    libero_root: Optional[str], upstream_root: Optional[str], checkpoint_dir: Optional[str], suite: str = "goal"
) -> int:
    spec = get_suite(suite)
    report = {
        "python": sys.version.split()[0],
        "imports": {},
        "paths": {},
        "gpu": {},
    }  # type: Dict[str, Any]
    for module in ("numpy", "msgpack", "websockets", "PIL", "imageio"):
        report["imports"][module] = importlib.util.find_spec(module) is not None
    for name, value, marker in (
        ("libero_root", libero_root, "libero/libero"),
        ("upstream_root", upstream_root, "src/moevla"),
    ):
        if value:
            path = pathlib.Path(value).expanduser().resolve()
            report["paths"][name] = {"path": str(path), "valid": (path / marker).exists()}
    if checkpoint_dir:
        path = pathlib.Path(checkpoint_dir).expanduser().resolve()
        weights = path / "pytorch_model.pth"
        norm_stats = path / spec.normalization_asset / "meta" / "stats.json"
        actual_bytes = weights.stat().st_size if weights.is_file() else None
        actual_sha256 = None
        if actual_bytes == spec.weights_bytes:
            digest = hashlib.sha256()
            with weights.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
        report["paths"]["checkpoint_dir"] = {
            "path": str(path),
            "suite": spec.key,
            "weights_bytes": actual_bytes,
            "expected_weights_bytes": spec.weights_bytes,
            "expected_weights_sha256": spec.weights_sha256,
            "weights_sha256": actual_sha256,
            "normalization_asset": spec.normalization_asset,
            "normalization_stats": norm_stats.is_file() and norm_stats.stat().st_size > 0,
            "valid": actual_sha256 == spec.weights_sha256
            and norm_stats.is_file()
            and norm_stats.stat().st_size > 0,
        }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report["gpu"]["nvidia_smi"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError) as error:
        report["gpu"]["error"] = str(error)
    print(json.dumps(report, indent=2, sort_keys=True))
    values = list(report["imports"].values())
    values.extend(item["valid"] for item in report["paths"].values())
    return 0 if all(values) else 1
