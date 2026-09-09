"""Measure the corpus storage codec on real routing arrays, not on intuition.

The 08-10 report recommends Blosc(zstd, clevel=1, shuffle) over the status-quo
plain Zstd, but every array in the store has a different dtype and every stored
batch so far used plain Zstd, so the recommendation has never actually been
applied end to end.  This re-encodes a real `routes.zarr` under each candidate
and reports bytes and encode time per array.

Run in the model env (needs zarr 3).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import time

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ZstdCodec


def _dir_bytes(path: pathlib.Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def candidates(itemsize: int):
    """name -> compressor list, for one array's itemsize.

    The status quo is ``zstd3`` (every batch stored so far).  The 08-10 report
    recommended ``blosc(clevel=1, shuffle)``; it is kept here so the comparison
    stays honest rather than being quietly dropped once it lost.
    """
    def blosc(clevel, shuffle="shuffle"):
        return [BloscCodec(cname="zstd", clevel=clevel, shuffle=shuffle, typesize=itemsize)]

    return {
        "A_zstd3_statusquo": [ZstdCodec(level=3)],
        "zstd5": [ZstdCodec(level=5)],
        "zstd9": [ZstdCodec(level=9)],
        "blosc1_shuffle": blosc(1),
        "blosc5_shuffle": blosc(5),
        "blosc5_noshuffle": blosc(5, "noshuffle"),
        "blosc7_shuffle": blosc(7),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="runs/within64-s24/routes.zarr")
    ap.add_argument("--scratch", default="/tmp/codec-bench")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    source = zarr.open_group(args.source, mode="r")
    scratch = pathlib.Path(args.scratch)

    names = sorted(source.array_keys())
    data = {name: np.asarray(source[name][:]) for name in names}
    chunks = {name: source[name].chunks for name in names}
    print("source: %s" % args.source)
    for name in names:
        print("  %-18s %-22s %-9s raw=%7.1f MB"
              % (name, data[name].shape, data[name].dtype, data[name].nbytes / 2**20))

    totals = {}
    for name in names:
        itemsize = data[name].dtype.itemsize
        print("\n%s (itemsize=%d)" % (name, itemsize))
        for label, compressors in candidates(itemsize).items():
            best = None
            for _ in range(args.repeats):
                path = scratch / label / name
                if path.exists():
                    shutil.rmtree(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                array = zarr.create_array(
                    store=str(path), shape=data[name].shape, chunks=chunks[name],
                    dtype=data[name].dtype, compressors=compressors, overwrite=True,
                )
                array[:] = data[name]
                elapsed = time.perf_counter() - start
                size = _dir_bytes(path)
                best = (elapsed, size) if best is None else (min(best[0], elapsed), size)
            totals.setdefault(label, [0, 0.0])
            totals[label][0] += best[1]
            totals[label][1] += best[0]
            print("  %-28s %8.2f MB   %6.0f ms" % (label, best[1] / 2**20, 1000 * best[0]))

    print("\n=== totals (best of %d) ===" % args.repeats)
    base = totals["A_zstd3_statusquo"]
    for label, (size, elapsed) in totals.items():
        print("%-28s %8.2f MB (%+5.1f%%)   %7.0f ms (%+5.1f%%)"
              % (label, size / 2**20, 100 * (size / base[0] - 1),
                 1000 * elapsed, 100 * (elapsed / base[1] - 1)))
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
