from __future__ import annotations

import pathlib
import sys

import pytest


MODULE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
import verify_model_bundles as verify  # noqa: E402


def metadata(logical_gpu: int) -> dict:
    return {
        "bundle_schema": "himoe-vla-five-model-gpu-bundle/v1",
        "bundle_model": "goal",
        "bundle_physical_gpu": 7,
        "bundle_logical_gpu": logical_gpu,
        "bundle_port": 8870,
        "checkpoint_load_audit": {
            "mode": "torch_mmap_meta_assign",
            "target_device": "cuda:%d" % logical_gpu,
        },
    }


def test_bundle_identity_accepts_matrix_logical_gpu() -> None:
    verify._verify_bundle_identity(metadata(1), 8870, 7, 1, "goal")


def test_bundle_identity_rejects_wrong_matrix_logical_gpu() -> None:
    with pytest.raises(RuntimeError, match="bundle metadata mismatch"):
        verify._verify_bundle_identity(metadata(1), 8870, 7, 0, "goal")
