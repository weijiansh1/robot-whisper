"""Immutable LIBERO suite, checkpoint, and normalization pairings."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Dict


CHECKPOINT_BYTES = 8_138_322_389


@dataclasses.dataclass(frozen=True)
class SuiteSpec:
    key: str
    benchmark: str
    model_repo: str
    checkpoint_name: str
    train_config: str
    dataset_config: str
    normalization_asset: str
    weights_sha256: str
    weights_bytes: int = CHECKPOINT_BYTES

    def checkpoint_dir(self, cache_root: str) -> pathlib.Path:
        return pathlib.Path(cache_root).expanduser().resolve() / "checkpoints" / self.checkpoint_name


SUITES = {
    "goal": SuiteSpec(
        key="goal",
        benchmark="libero_goal",
        model_repo="ZhiyingDu/HiMoE-VLA-Libero-Goal",
        checkpoint_name="HiMoE-VLA-Libero-Goal",
        train_config="libero_goal_no_noops_lerobot_finetune",
        dataset_config="libero_goal_no_noops_lerobot",
        normalization_asset="libero_goal_no_noops",
        weights_sha256="98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953",
    ),
    "spatial": SuiteSpec(
        key="spatial",
        benchmark="libero_spatial",
        model_repo="ZhiyingDu/HiMoE-VLA-Libero-Spatial",
        checkpoint_name="HiMoE-VLA-Libero-Spatial",
        train_config="libero_spatial_no_noops_lerobot_finetune",
        dataset_config="libero_spatial_no_noops_lerobot",
        normalization_asset="libero_spatial_no_noops",
        weights_sha256="1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137",
    ),
    "object": SuiteSpec(
        key="object",
        benchmark="libero_object",
        model_repo="ZhiyingDu/HiMoE-VLA-Libero-Object",
        checkpoint_name="HiMoE-VLA-Libero-Object",
        train_config="libero_object_no_noops_lerobot_finetune",
        dataset_config="libero_object_no_noops_lerobot",
        normalization_asset="libero_object_no_noops",
        weights_sha256="f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa",
    ),
    "long": SuiteSpec(
        key="long",
        benchmark="libero_10",
        model_repo="ZhiyingDu/HiMoE-VLA-Libero-10",
        checkpoint_name="HiMoE-VLA-Libero-10",
        train_config="libero_10_no_noops_lerobot_finetune",
        dataset_config="libero_10_no_noops_lerobot",
        normalization_asset="libero_10_no_noops",
        weights_sha256="cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256",
    ),
}  # type: Dict[str, SuiteSpec]

SUITE_NAMES = tuple(SUITES)


def get_suite(name: str) -> SuiteSpec:
    try:
        return SUITES[name]
    except KeyError as error:
        raise ValueError("Unknown suite %r; choose one of %s" % (name, ", ".join(SUITE_NAMES))) from error
