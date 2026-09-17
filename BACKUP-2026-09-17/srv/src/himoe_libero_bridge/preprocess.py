"""LIBERO observation preprocessing matched to the upstream HiMoE example."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from PIL import Image

from himoe_libero_bridge.protocol import (
    IMAGE_KEY,
    IMAGE_SHAPE,
    PROMPT_KEY,
    STATE_KEY,
    WRIST_IMAGE_KEY,
    validate_observation,
)


def resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    image = np.asarray(image)
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 image, got %s" % (image.shape,))
    if np.issubdtype(image.dtype, np.floating):
        image = (255.0 * image).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)

    pil_image = Image.fromarray(image)
    current_width, current_height = pil_image.size
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = pil_image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    padded = Image.new(resized.mode, (width, height), 0)
    padded.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
    return np.asarray(padded, dtype=np.uint8)


def quat_to_axis_angle(quaternion: Any) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise ValueError("Quaternion shape must be (4,), got %s" % (quat.shape,))
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray((quat[:3] * 2.0 * math.acos(quat[3])) / denominator, dtype=np.float32)


def frame_from_observation(observation: Dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(observation["agentview_image"])[::-1, ::-1], dtype=np.uint8)


def build_policy_observation(observation: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    base_image = frame_from_observation(observation)
    wrist_image = np.ascontiguousarray(
        np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1], dtype=np.uint8
    )
    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    target_height, target_width, _ = IMAGE_SHAPE
    return validate_observation(
        {
            IMAGE_KEY: resize_with_pad(base_image, target_height, target_width),
            WRIST_IMAGE_KEY: resize_with_pad(wrist_image, target_height, target_width),
            STATE_KEY: state,
            PROMPT_KEY: str(prompt),
        }
    )
