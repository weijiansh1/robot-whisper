#!/usr/bin/env python3
"""Check an actual LIBERO camera on one explicitly allowed EGL device."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import time
import uuid

import numpy as np

from benchmarks.run_benchmarks import ALLOWED_GPUS, load_suite, variants


def verify_egl_device(gpu):
    if gpu not in ALLOWED_GPUS or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(gpu):
        raise ValueError("Use the benchmark CPU-only environment with explicit EGL selection")
    from mujoco.egl import egl_ext as EGL

    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(EGL.eglGetProcAddress(b"eglQueryDeviceBinaryEXT"))
    value, size = (ctypes.c_ubyte * 16)(), ctypes.c_int()
    if not query(EGL.eglQueryDevicesEXT()[gpu], 0x335C, 16, value, ctypes.byref(size)) or size.value != 16:
        raise ValueError("EGL device UUID unavailable")
    actual = "GPU-" + str(uuid.UUID(bytes=bytes(value)))
    expected = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
                              check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if actual != expected:
        raise ValueError("EGL UUID differs from requested physical GPU")
    return actual


def run(args):
    device_uuid = verify_egl_device(args.gpu)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from himoe_libero_bridge.preprocess import build_policy_observation
    row = next(row for row in variants("pro") if row["variant_id"] == "960b5f115fadb96ce14c5ac5")
    suite = load_suite(row["registry"], {})
    task = suite.get_task(row["registry_index"])
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    random.seed(20260908)
    np.random.seed(20260908)
    started = time.monotonic()
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=224, camera_widths=224,
                            render_gpu_device_id=args.gpu)
    try:
        env.seed(20260908)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(row["registry_index"])[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
        request = build_policy_observation(obs, str(env.language_instruction))
        cameras = {key: dict(std=float(np.std(value)), minimum=int(np.min(value)), maximum=int(np.max(value)))
                   for key, value in request.items() if key in ("observation/image", "observation/wrist_image")}
        result = dict(gpu=args.gpu, egl_device_uuid=device_uuid,
                      elapsed_seconds=time.monotonic() - started, cameras=cameras,
                      passed=len(cameras) == 2 and all(value["std"] >= 1 for value in cameras.values()))
        print(json.dumps(result), flush=True)
        return 0 if result["passed"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=ALLOWED_GPUS, required=True)
    raise SystemExit(run(parser.parse_args()))
