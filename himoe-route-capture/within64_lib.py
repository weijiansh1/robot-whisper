"""Load a serve_with_recorder.py capture and segment it back into episodes.

The server writes one flat routes.zarr with a running control-step counter; it
never learns episode boundaries because the LIBERO client does not send an
``episode_id`` (serve_with_recorder.py:78 falls back to a constant).  The
boundaries live in the client's summaries.json instead, as ``inference_calls``
per episode.  Segmentation is therefore a cumsum, and it is exact: the total
must equal the trace length or the two sides disagree about what was captured.

Layout of the stored arrays (verified against himoe_router_recorder.stack):

    hb_router_probs   [T, L=8, D=10, S=11, E=32]  float16
    hb_expert_ids     [T, L=8, D=10, S=11, K=4]   uint8
    hb_entropy        [T, L=8, D=10, S=11]        float16

with L over HB layers [2,3,4,5,12,13,14,15], D over flow denoising steps, and S
over suffix tokens.  Token 0 is the state token; tokens 1..10 are the action
tokens -- moevla.py:483 keeps ``suffix_out[:, -n_action_steps:]``, so only the
latter reach the action head.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import zarr

HB_LAYERS = [2, 3, 4, 5, 12, 13, 14, 15]
N_EXPERTS = 32
TOP_K = 4
STATE_TOKEN = 0
ACTION_TOKENS = slice(1, None)


@dataclass
class Episode:
    index: int
    init_state_id: int
    flow_noise_seed: int
    success: bool
    action_steps: int
    n_control: int
    start: int  # offset into the flat trace
    probs: np.ndarray  # [T, 8, 10, 11, 32] float32
    ids: np.ndarray  # [T, 8, 10, 11, 4] uint8
    entropy: np.ndarray  # [T, 8, 10, 11] float32


@dataclass
class Run:
    label: str
    layout: str
    checkpoint_sha256: str
    episodes: list[Episode]
    summaries: list[dict]

    @property
    def n_success(self) -> int:
        return sum(1 for e in self.episodes if e.success)


def _check_axes(
    probs: np.ndarray, ids: np.ndarray, entropy: np.ndarray, meta: dict
) -> None:
    """Pin the axis order of the stored routing arrays.

    Axis 2 is the flow denoising step (``config.num_steps``) and axis 3 is the
    suffix token (``config.n_action_steps + 1``).  These are separate config
    fields that are 10 and 11 on this checkpoint and 10 and 51 on the upstream
    default, so a transpose is only self-evident while they differ.  Captures
    written after 2026-08-12 carry both values in server_metadata.json; assert
    against those when present rather than trusting the shape.
    """
    if probs.ndim != 5:
        raise RuntimeError(f"hb_router_probs should be 5-D [T,L,D,S,E], got {probs.shape}")
    _, n_layer, n_denoise, n_suffix, n_expert = probs.shape
    if n_layer != len(HB_LAYERS) or n_expert != N_EXPERTS:
        raise RuntimeError(
            f"expected {len(HB_LAYERS)} HB layers and {N_EXPERTS} experts, "
            f"got shape {probs.shape}"
        )
    if ids.shape[:4] != probs.shape[:4] or ids.shape[4] != TOP_K:
        raise RuntimeError(f"hb_expert_ids shape {ids.shape} disagrees with {probs.shape}")
    if entropy.shape != probs.shape[:4]:
        raise RuntimeError(f"hb_entropy shape {entropy.shape} disagrees with {probs.shape}")

    want_denoise = meta.get("route_axis_n_denoise", meta.get("num_steps"))
    want_suffix = meta.get("route_axis_n_suffix")
    if want_suffix is None and meta.get("n_action_steps") is not None:
        want_suffix = int(meta["n_action_steps"]) + 1
    if want_denoise is not None and n_denoise != int(want_denoise):
        raise RuntimeError(
            f"axis 2 is {n_denoise} but the server recorded num_steps={want_denoise}; "
            "the denoise and suffix axes are transposed"
        )
    if want_suffix is not None and n_suffix != int(want_suffix):
        raise RuntimeError(
            f"axis 3 is {n_suffix} but the server recorded n_action_steps+1={want_suffix}; "
            "the denoise and suffix axes are transposed"
        )
    if want_denoise is None or want_suffix is None:
        print(
            f"[axes] server_metadata.json predates the axis fields; assuming "
            f"denoise={n_denoise}, suffix={n_suffix} from shape alone"
        )


def load_run(
    server_dir: str | pathlib.Path,
    client_dir: str | pathlib.Path,
    allow_partial: bool = False,
) -> Run:
    """Load a capture and cut it back into episodes.

    ``allow_partial`` keeps only the episodes whose control steps are already on
    disk, for inspecting a run that is still going.  The server's Zarr writer
    buffers ``chunk_steps`` control steps before flushing, so a live trace always
    trails the client's summaries.json by up to one chunk.
    """
    server_dir = pathlib.Path(server_dir)
    client_dir = pathlib.Path(client_dir)
    summaries = json.loads((client_dir / "summaries.json").read_text())
    meta = json.loads((client_dir / "server_metadata.json").read_text())

    group = zarr.open(str(server_dir / "routes.zarr"), mode="r")
    probs = np.asarray(group["hb_router_probs"][:], dtype=np.float32)
    ids = np.asarray(group["hb_expert_ids"][:])
    entropy = np.asarray(group["hb_entropy"][:], dtype=np.float32)
    control_step = np.asarray(group["control_step"][:])

    if not (np.diff(control_step) == 1).all():
        raise RuntimeError("control_step is not contiguous; the capture dropped a chunk")

    _check_axes(probs, ids, entropy, meta)

    available = int(probs.shape[0])
    total = int(sum(s["inference_calls"] for s in summaries))
    if total != available:
        if not allow_partial:
            raise RuntimeError(
                f"client reports {total} inference calls but the trace holds "
                f"{available} control steps; the two runs are not the same run"
            )
        keep, used = [], 0
        for s in summaries:
            if used + int(s["inference_calls"]) > available:
                break
            keep.append(s)
            used += int(s["inference_calls"])
        print(f"[partial] {len(keep)}/{len(summaries)} episodes complete on disk "
              f"({used}/{available} captured control steps)")
        summaries = keep

    episodes: list[Episode] = []
    cursor = 0
    for summary in summaries:
        n = int(summary["inference_calls"])
        episodes.append(
            Episode(
                index=int(summary["episode_index"]),
                init_state_id=int(summary["init_state_id"]),
                flow_noise_seed=int(summary["flow_noise_seed"]),
                success=bool(summary["success"]),
                action_steps=int(summary["action_steps"]),
                n_control=n,
                start=cursor,
                probs=probs[cursor : cursor + n],
                ids=ids[cursor : cursor + n],
                entropy=entropy[cursor : cursor + n],
            )
        )
        cursor += n

    return Run(
        label=str(summaries[0].get("label", "?")),
        layout=str(meta.get("libero_wrist_layout", "?")),
        checkpoint_sha256=str(meta.get("checkpoint_sha256", "")),
        episodes=episodes,
        summaries=summaries,
    )


def action_token_probs(episode: Episode) -> np.ndarray:
    """[T, 8, 10, 32] -- router distribution averaged over the action tokens."""
    return episode.probs[:, :, :, ACTION_TOKENS, :].mean(axis=3)


def state_token_probs(episode: Episode) -> np.ndarray:
    """[T, 8, 10, 32] -- router distribution at the state token."""
    return episode.probs[:, :, :, STATE_TOKEN, :]


def selection_counts(episode: Episode, layer: int) -> np.ndarray:
    """[T, 32] -- how many of the 10*11*4 slots picked each expert, per step."""
    sel = episode.ids[:, layer].reshape(episode.n_control, -1)
    out = np.zeros((episode.n_control, N_EXPERTS), dtype=np.int32)
    for t in range(episode.n_control):
        out[t] = np.bincount(sel[t], minlength=N_EXPERTS)
    return out


def onehot_sets(episode: Episode) -> np.ndarray:
    """[T, 8, 10, 11, 32] bool -- the top-4 set at every individual routing site.

    A "site" is one (control step, layer, denoising step, token).  Comparing sets
    per site is the meaningful churn measure; pooling a whole control step is not,
    because 10 denoising steps x 11 tokens x 4 slots = 440 draws from a
    near-uniform router touch almost all 32 experts, so any two control steps look
    ~90% identical no matter what the router did.
    """
    out = np.zeros(episode.ids.shape[:-1] + (N_EXPERTS,), dtype=bool)
    np.put_along_axis(out, episode.ids.astype(np.int64), True, axis=-1)
    return out


def jaccard_from_intersection(inter: np.ndarray) -> np.ndarray:
    """Jaccard for two top-k sets given |A & B|; both sets have exactly TOP_K members."""
    return inter / (2 * TOP_K - inter)
