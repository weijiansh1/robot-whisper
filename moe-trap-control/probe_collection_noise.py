#!/usr/bin/env python3
"""Compare complete upstream/accelerated noise outputs and global RNG state."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from collection_noise import glass_blur_exact, install
from collection_storage import atomic_json


def run(args):
    from libero.libero.envs import env_wrapper

    original = env_wrapper.glass_blur
    metadata = install()
    rows = []
    for seed in (712001, 712002):
        image = Image.fromarray(np.random.default_rng(seed).integers(0, 256, (224, 224, 3), dtype=np.uint8))
        for severity in range(1, 11):
            np.random.seed(seed + severity)
            tick = time.monotonic()
            expected = original(image, severity)
            original_seconds = time.monotonic() - tick
            expected_state = np.random.get_state()
            np.random.seed(seed + severity)
            tick = time.monotonic()
            actual = glass_blur_exact(image, severity)
            accelerated_seconds = time.monotonic() - tick
            actual_state = np.random.get_state()
            np.testing.assert_array_equal(actual, expected)
            for before, after in zip(expected_state, actual_state):
                np.testing.assert_array_equal(before, after)
            rows.append(dict(seed=seed, severity=severity, output_exact=True, rng_exact=True,
                             original_seconds=original_seconds, accelerated_seconds=accelerated_seconds))
    original_seconds = sum(row["original_seconds"] for row in rows)
    accelerated_seconds = sum(row["accelerated_seconds"] for row in rows)
    report = dict(status="passed", cases=len(rows), metadata=metadata, original_seconds=original_seconds,
                  accelerated_seconds=accelerated_seconds, speedup=original_seconds / accelerated_seconds, rows=rows)
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
