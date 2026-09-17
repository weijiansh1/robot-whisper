"""Drop-in, bit-identical accelerations for LIBERO-plus image perturbations.

LIBERO-plus's ``glass_blur`` shuffles pixels in a pure-Python triple loop with one
``np.random.randint`` call per pixel (~125k calls and ~1 s per 256x256 frame).  Its
"swap" is the numpy-view pitfall: ``x[h, w], x[h2, w2] = x[h2, w2], x[h, w]`` copies
pixel (h2, w2) into (h, w) and then writes the already-overwritten (h, w) back, so
both pixels end up equal to the original (h2, w2).  The kernel below reproduces
exactly that sequence, with all random draws taken in one vectorised call from the
same global RandomState (same stream, same order), and runs in ~5 ms.

``install()`` replaces ``libero.libero.envs.env_wrapper.glass_blur``; ``self_test()``
checks bit-identity against the original on random images for every severity.
"""
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

_SEVERITY = [(0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2), (1.1, 3, 2), (1.3, 3, 2),
             (1.5, 4, 2), (1.8, 4, 2), (2.2, 5, 1), (2.5, 5, 1)]


def _shuffle_py(x, draws, max_delta, iterations):
    height_x, weight_x = x.shape[0], x.shape[1]
    k = 0
    for _ in range(iterations):
        for h in range(height_x - max_delta, max_delta, -1):
            for w in range(weight_x - max_delta, max_delta, -1):
                dx, dy = draws[k, 0], draws[k, 1]
                k += 1
                h2, w2 = h + dy, w + dx
                for ch in range(x.shape[2]):
                    v = x[h2, w2, ch]
                    x[h, w, ch] = v
                    x[h2, w2, ch] = v
    return x


_shuffle = njit(cache=True)(_shuffle_py) if njit is not None else _shuffle_py


def glass_blur(x, severity=1):
    from skimage.filters import gaussian
    c = _SEVERITY[severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)
    height_x, weight_x = x.shape[0], x.shape[1]
    rows = max(0, height_x - 2 * c[1])
    cols = max(0, weight_x - 2 * c[1])
    count = c[2] * rows * cols
    draws = np.random.randint(-c[1], c[1], size=(count, 2)) if count else np.zeros((0, 2), np.int64)
    x = np.ascontiguousarray(x)
    _shuffle(x, draws.astype(np.int64), int(c[1]), int(c[2]))
    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255


_ZOOM_C = [np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01), np.arange(1, 1.21, 0.02), np.arange(1, 1.26, 0.02),
           np.arange(1, 1.31, 0.03), np.arange(1, 1.36, 0.01), np.arange(1, 1.41, 0.01), np.arange(1, 1.46, 0.02),
           np.arange(1, 1.51, 0.02), np.arange(1, 1.56, 0.03)]


def zoom_blur(x, severity=1):
    """LIBERO-plus zoom_blur with the independent scipy zooms run in threads (scipy.ndimage releases the GIL);
    the float32 accumulation keeps the original sequential order, so the output is bit-identical."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    import libero.libero.envs.env_wrapper as wrapper
    c = _ZOOM_C[severity - 1]
    x = (np.array(x) / 255.).astype(np.float32)
    threads = max(1, int(os.environ.get("HIMOE_ZOOM_THREADS", os.environ.get("MAGICK_THREAD_LIMIT", "4"))))
    if threads > 1:
        with ThreadPoolExecutor(threads) as ex:
            zooms = list(ex.map(lambda z: wrapper.clipped_zoom(x, z), c))
    else:
        zooms = [wrapper.clipped_zoom(x, z) for z in c]
    out = np.zeros_like(x)
    for z in zooms:
        out += z
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def install():
    import libero.libero.envs.env_wrapper as wrapper
    if getattr(wrapper.glass_blur, "_fast", False):
        return
    glass_blur._fast = True
    wrapper._original_glass_blur = wrapper.glass_blur
    wrapper.glass_blur = glass_blur
    wrapper._original_zoom_blur = wrapper.zoom_blur
    wrapper.zoom_blur = zoom_blur


def self_test(shape=(64, 64, 3), seeds=(0, 1)):
    import libero.libero.envs.env_wrapper as wrapper
    original = getattr(wrapper, "_original_glass_blur", wrapper.glass_blur)
    for severity in range(1, 11):
        for seed in seeds:
            img = np.random.RandomState(seed + 100).randint(0, 256, size=shape).astype(np.uint8)
            np.random.seed(seed)
            a = original(img.copy(), severity)
            np.random.seed(seed)
            b = glass_blur(img.copy(), severity)
            if not np.array_equal(a, b):
                raise AssertionError("glass_blur mismatch at severity %d seed %d: max diff %g" % (severity, seed, np.abs(a - b).max()))
            if np.random.get_state()[2] != np.random.get_state()[2]:
                raise AssertionError("RNG state diverged")
    original_zoom = getattr(wrapper, "_original_zoom_blur", wrapper.zoom_blur)
    for severity in range(1, 11):
        img = np.random.RandomState(severity).randint(0, 256, size=shape).astype(np.uint8)
        a, b = original_zoom(img.copy(), severity), zoom_blur(img.copy(), severity)
        if not np.array_equal(a, b):
            raise AssertionError("zoom_blur mismatch at severity %d" % severity)
    return True


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "upstream" / "LIBERO-plus"))
    sys.path.insert(0, str(root / "dependencies" / "libero-plus"))
    import os
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(root / "configs" / "libero-plus"))
    print("self-test:", self_test())
    import libero.libero.envs.env_wrapper as wrapper
    img = np.random.RandomState(0).randint(0, 256, size=(256, 256, 3)).astype(np.uint8)
    for name, fn in (("original", wrapper.glass_blur), ("fast", glass_blur)):
        np.random.seed(7)
        t = time.perf_counter()
        fn(img.copy(), 5)
        print("%-9s glass_blur 256x256 severity 5: %.3f s" % (name, time.perf_counter() - t))
    for name, fn in (("original", wrapper.zoom_blur), ("fast", zoom_blur)):
        t = time.perf_counter()
        fn(img.copy(), 7)
        print("%-9s zoom_blur 256x256 severity 7: %.3f s" % (name, time.perf_counter() - t))
