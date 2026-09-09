"""Format benchmark for MoE routing traces, on real captured data.

Size alone does not decide this.  The trace is written once, append-only, while
a rollout is running (so write cost must stay well under the 717 ms inference
step), and then read many times in two very different patterns:

  full      stream every control step once -- analysis passes, PCA, decoding
  by-step   pull a handful of scattered control steps -- "show me episode 37"
  by-layer  pull one layer across all steps -- per-layer statistics

A format that wins on bytes but forces a full decompress for a single control
step is the wrong choice here, so all three are measured.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import time

import numpy as np

SCRATCH = pathlib.Path("/tmp/fmt-bench")


def _clear() -> pathlib.Path:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    return SCRATCH


def _size(p: pathlib.Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _timeit(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench(x: np.ndarray, steps: list[int], layer: int) -> list[dict]:
    """x: [T, L, D, S, E] float16 -- the router softmax array."""
    out = []

    def record(name, path, wr, rd_full, rd_step, rd_layer, note=""):
        out.append({
            "name": name, "bytes": _size(path), "write_s": wr,
            "full_s": rd_full, "step_s": rd_step, "layer_s": rd_layer, "note": note,
        })
        print("  %-34s %7.1f MB  写 %5.2fs  全读 %5.2fs  抽步 %6.3fs  抽层 %6.3fs %s"
              % (name, _size(path) / 1e6, wr, rd_full, rd_step, rd_layer, note))

    # ---- zarr v3 ---------------------------------------------------------
    import zarr
    from zarr.codecs import ZstdCodec

    for level, chunk_t in ((3, 4), (9, 4), (19, 4), (19, 32)):
        d = _clear() / f"zarr_zstd{level}_c{chunk_t}.zarr"
        chunks = (chunk_t,) + x.shape[1:]

        def w(level=level, chunks=chunks, d=d):
            z = zarr.create_array(str(d), shape=x.shape, dtype=x.dtype, chunks=chunks,
                                  compressors=[ZstdCodec(level=level)], overwrite=True)
            z[:] = x
        wr = _timeit(w, 1)
        z = zarr.open_array(str(d), mode="r")
        record(f"zarr v3 zstd-{level} chunk_t={chunk_t}", d, wr,
               _timeit(lambda: np.asarray(z[:])),
               _timeit(lambda: [np.asarray(z[s]) for s in steps]),
               _timeit(lambda: np.asarray(z[:, layer])))

    # ---- HDF5 ------------------------------------------------------------
    import h5py

    for name, kw in (("gzip-4", dict(compression="gzip", compression_opts=4)),
                     ("lzf", dict(compression="lzf"))):
        d = _clear() / f"h5_{name}.h5"

        def w():
            with h5py.File(d, "w") as f:
                f.create_dataset("p", data=x, chunks=(4,) + x.shape[1:], **kw)
        wr = _timeit(w, 1)
        f = h5py.File(d, "r")
        z = f["p"]
        record(f"HDF5 {name}", d, wr,
               _timeit(lambda: z[:]),
               _timeit(lambda: [z[s] for s in steps]),
               _timeit(lambda: z[:, layer]))
        f.close()

    # ---- blosc2 nd -------------------------------------------------------
    try:
        import blosc2
        d = _clear() / "blosc2.b2nd"

        def w():
            blosc2.asarray(np.ascontiguousarray(x), urlpath=str(d), mode="w",
                           chunks=(4,) + x.shape[1:],
                           cparams={"codec": blosc2.Codec.ZSTD, "clevel": 9})
        wr = _timeit(w, 1)
        z = blosc2.open(str(d))
        record("blosc2 b2nd zstd-9", d, wr,
               _timeit(lambda: z[:]),
               _timeit(lambda: [z[s] for s in steps]),
               _timeit(lambda: z[:, layer]))
    except Exception as exc:  # pragma: no cover - optional dependency
        print("  blosc2 skipped:", exc)

    # ---- flat baselines --------------------------------------------------
    import zlib
    d = _clear() / "raw.npy"
    wr = _timeit(lambda: np.save(d, x), 1)
    record("裸 .npy (无压缩, mmap)", d, wr,
           _timeit(lambda: np.array(np.load(d, mmap_mode="r"))),
           _timeit(lambda: [np.array(np.load(d, mmap_mode="r")[s]) for s in steps]),
           _timeit(lambda: np.array(np.load(d, mmap_mode="r")[:, layer])),
           "无压缩")

    d = _clear() / "arr.npz"
    wr = _timeit(lambda: np.savez_compressed(d, p=x), 1)
    record("npz (deflate)", d, wr,
           _timeit(lambda: np.load(d)["p"]),
           _timeit(lambda: [np.load(d)["p"][s] for s in steps]),
           _timeit(lambda: np.load(d)["p"][:, layer]),
           "无局部读取")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="runs/within64-s24/routes.zarr")
    ap.add_argument("--array", default="hb_router_probs")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    import zarr
    g = zarr.open(args.zarr, mode="r")
    x = np.asarray(g[args.array][: args.steps])
    print("数据: %s %s %s  原始 %.1f MB\n" % (args.array, x.shape, x.dtype, x.nbytes / 1e6))
    rng = np.random.default_rng(0)
    steps = sorted(rng.choice(x.shape[0], size=32, replace=False).tolist())
    bench(x, steps, layer=2)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
