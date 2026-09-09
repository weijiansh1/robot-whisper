"""Measure on-disk size for the routing trace across storage configurations."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

sys.path.insert(0, str(Path(__file__).parent))
from himoe_route_store import ZarrRouteReader  # noqa: E402

SRC = "/tmp/routes.zarr"


def du(path: str | Path) -> int:
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def write_zarr(path, data, chunks, codecs_for):
    if Path(path).exists():
        shutil.rmtree(path)
    root = zarr.create_group(store=path, overwrite=True)
    for name, arr in data.items():
        ch = (chunks, *arr.shape[1:])
        root.create_array(
            name=name,
            shape=arr.shape,
            chunks=ch,
            dtype=arr.dtype,
            compressors=codecs_for(arr.dtype),
        )[:] = arr
    return du(path)


def main() -> int:
    r = ZarrRouteReader(SRC)
    n = len(r)
    data = {k: np.asarray(r[k][:]) for k in r.names}
    raw = sum(a.nbytes for a in data.values())
    print(f"{n} control steps | uncompressed {raw:,} B = {raw / n / 1024:.1f} KiB/step\n")
    for k, a in sorted(data.items(), key=lambda kv: -kv[1].nbytes):
        print(f"  {k:18s} {str(a.dtype):8s} {str(a.shape):24s} {a.nbytes:>10,} B")

    print(f"\n{'config':38s} {'bytes':>11s} {'ratio':>7s} {'KiB/step':>9s}")
    print("-" * 68)

    def row(label, size):
        print(f"{label:38s} {size:>11,} {raw / size:>6.2f}x {size / n / 1024:>8.2f}")

    # --- npz baselines ---
    np.savez("/tmp/routes.npz", **data)
    row("npz (uncompressed)", du("/tmp/routes.npz"))
    np.savez_compressed("/tmp/routes_c.npz", **data)
    row("npz_compressed (ZIP deflate)", du("/tmp/routes_c.npz"))

    # --- zarr variants ---
    for lvl in (1, 3, 5, 9):
        size = write_zarr(
            f"/tmp/z_zstd{lvl}", data, 64, lambda dt: [ZstdCodec(level=lvl)]
        )
        row(f"zarr chunk=64  zstd-{lvl}  (no shuffle)", size)

    for lvl in (1, 3, 5, 9):
        size = write_zarr(
            f"/tmp/z_bs{lvl}",
            data,
            64,
            lambda dt: [
                BloscCodec(
                    cname="zstd", clevel=lvl, shuffle="bitshuffle", typesize=dt.itemsize
                )
            ],
        )
        row(f"zarr chunk=64  blosc-zstd-{lvl} bitshuf", size)

    for chunk in (16, 64, 256, 1024):
        size = write_zarr(
            f"/tmp/z_ch{chunk}",
            data,
            chunk,
            lambda dt: [
                BloscCodec(
                    cname="zstd", clevel=3, shuffle="bitshuffle", typesize=dt.itemsize
                )
            ],
        )
        row(f"zarr chunk={chunk:<4d} blosc-zstd-3 bitshuf", size)

    # --- what the naive schema would have cost ---
    print("\n=== schema overhead avoided ===")
    cw = data["hb_selected_prob"]  # same shape/dtype as combine_weight
    as_full_bytes = n * 4 * 10 * 51 * (1 + 3 * 2)
    print(f"  storing combine_weight too : +{cw.nbytes:,} B  (+{cw.nbytes / raw * 100:.0f}% raw)")
    print(f"  AS without collapse        : +{as_full_bytes:,} B")
    print(f"  full router_probs [.,32]   : +{n * 8 * 10 * 51 * 32 * 2:,} B  ({n * 8 * 10 * 51 * 32 * 2 / raw:.1f}x raw)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
