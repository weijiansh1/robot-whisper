"""CPU acceleration of the installed Plus glass-noise pixel loop."""

import hashlib
from pathlib import Path

import numpy as np
from numba import njit
from skimage.filters import gaussian

UPSTREAM_SHA256 = "e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c"
PARAMETERS = ((.5, 1, 3), (.7, 1, 3), (.9, 2, 3), (1., 2, 2), (1.1, 3, 2),
              (1.3, 3, 2), (1.5, 4, 2), (1.8, 4, 2), (2.2, 5, 1), (2.5, 5, 1))


@njit(cache=True)
def overwrite_neighbours(pixels, offsets, delta, iterations):
    cursor = 0
    for _ in range(iterations):
        for h in range(pixels.shape[0] - delta, delta, -1):
            for w in range(pixels.shape[1] - delta, delta, -1):
                dx, dy = offsets[cursor, 0], offsets[cursor, 1]
                cursor += 1
                # Preserve the upstream NumPy-view assignment, including its overwrite.
                for channel in range(pixels.shape[2]):
                    pixels[h, w, channel] = pixels[h + dy, w + dx, channel]
                    pixels[h + dy, w + dx, channel] = pixels[h, w, channel]


def glass_blur_exact(x, severity=1):
    sigma, delta, iterations = PARAMETERS[severity - 1]
    pixels = np.uint8(gaussian(np.array(x) / 255., sigma=sigma, channel_axis=-1) * 255)
    if pixels.ndim != 3:
        raise ValueError("Collection glass noise expects a color image")
    count = iterations * max(pixels.shape[0] - 2 * delta, 0) * max(pixels.shape[1] - 2 * delta, 0)
    offsets = np.random.randint(-delta, delta, size=(count, 2))
    overwrite_neighbours(pixels, offsets, delta, iterations)
    return np.clip(gaussian(pixels / 255., sigma=sigma, channel_axis=-1), 0, 1) * 255


def install():
    from libero.libero.envs import env_wrapper

    source = Path(env_wrapper.__file__)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != UPSTREAM_SHA256:
        raise ValueError("Plus noise source changed; fast path requires a new exactness audit")
    overwrite_neighbours(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((36, 2), dtype=np.int64), 1, 1)
    env_wrapper.glass_blur = glass_blur_exact
    return dict(enabled=True, upstream_sha256=actual, random_source="numpy_global_MT19937", jit_fastmath=False)
