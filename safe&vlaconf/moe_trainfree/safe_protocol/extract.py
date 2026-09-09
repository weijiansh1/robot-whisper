"""Extract fixed full-trajectory features directly from the two complete hub runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import zarr

from core import ROOT, RUNS, digest, write_json, route_features, behavior_features

HERE = Path(__file__).resolve().parent
HUB = ROOT / "VLA_MUI_HUB"


def extract_source(source, destination):
    run = HUB / source
    target = destination / (hashlib.sha256(source.encode()).hexdigest()[:20] + ".npz")
    summary_path = run / "client/summaries.json"
    metadata_path = run / "client/server_metadata.json"
    meta_path = run / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    meta = json.loads(meta_path.read_text())
    summaries = sorted(json.loads(summary_path.read_text()), key=lambda item: item["episode_index"])
    if len(summaries) != 400 or not meta["sampling"]["complete"]:
        raise ValueError(f"incomplete source {source}")
    expected = {"source": source, "summary_sha256": digest(summary_path),
                "metadata_sha256": digest(metadata_path), "meta_sha256": digest(meta_path),
                "extractor_sha256": digest(Path(__file__)), "core_sha256": digest(HERE / "core.py")}
    if target.exists():
        with np.load(target, allow_pickle=False) as archive:
            audit = json.loads(str(archive["audit"]))
            if any(audit.get(k) != v for k, v in expected.items()):
                raise ValueError(f"stale extracted cache {source}")
        return audit
    started = time.perf_counter()
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    lengths = np.asarray([s["inference_calls"] for s in summaries], dtype=np.int16)
    episodes = np.asarray([s["episode_index"] for s in summaries], dtype=np.int16)
    ids, steps = np.asarray(store["episode_id"][:]), np.asarray(store["control_step"][:])
    if not np.array_equal(ids, np.repeat(episodes, lengths)) or not np.all(np.diff(steps) == 1):
        raise ValueError(f"route/episode alignment {source}")
    raw = np.asarray(store["hb_router_probs"][:, :, 9, :, :])
    if raw.shape != (int(lengths.sum()), 8, 11, 32):
        raise ValueError("unexpected routing shape")
    qmax = 52
    output = {name: np.full((400, qmax, size), np.nan, np.float32)
              for name, size in (("load", 256), ("stats", 32), ("history", 40), ("behavior", 4), ("direct", 6))}
    rows, cursor = [], 0
    client_hashes = hashlib.sha256()
    sample_replays = []
    for i, summary in enumerate(summaries):
        length = int(lengths[i])
        ep = int(episodes[i])
        chunk = raw[cursor:cursor + length]
        feature = route_features(chunk)
        client = run / f"client/episode_{ep:02d}.npz"
        client_hashes.update(f"{ep}:{digest(client)}\n".encode())
        with np.load(client, allow_pickle=False) as saved:
            state, actions = saved["state"], saved["actions"]
        if state.shape != (length, 8) or actions.shape != (length, 10, 7):
            raise ValueError(f"client/route alignment {source}/{ep}")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("nonfinite client input")
        behavior, behavior_direct = behavior_features(state, actions, metadata["normalization_action_std"])
        for name in ("load", "stats", "history"):
            output[name][i, :length] = feature[name]
        output["behavior"][i, :length] = behavior
        output["direct"][i, :length] = np.concatenate((feature["direct"], behavior_direct), axis=-1)
        if i in (0, 199, 399):
            for stop in sorted(set((1, min(5, length), min(9, length), length))):
                replay = route_features(chunk[:stop])
                for name in feature:
                    if not np.allclose(feature[name][:stop], replay[name], equal_nan=True):
                        raise AssertionError(f"future-dependent feature {name}")
                b, d = behavior_features(state[:stop], actions[:stop], metadata["normalization_action_std"])
                if not np.allclose(b, behavior[:stop], equal_nan=True) or not np.allclose(d, behavior_direct[:stop], equal_nan=True):
                    raise AssertionError("future-dependent behavior")
                sample_replays.append([ep, stop])
        rows.append({"source": source, "run_id": run.name, "suite": run.parent.parent.name,
                     "task": run.parent.parent.name + "/" + run.parent.name, "episode": ep,
                     "init_state_id": int(summary["init_state_id"]), "noise_seed": int(summary["flow_noise_seed"]),
                     "length": length, "checkpoint": metadata["checkpoint_sha256"]})
        cursor += length
    valid = np.arange(qmax)[None] < lengths[:, None]
    for name in ("load", "stats", "behavior"):
        if not np.isfinite(output[name][valid]).all():
            raise ValueError(f"missing feature {name}")
    audit = dict(expected, episodes=400, queries=int(lengths.sum()), checkpoint=metadata["checkpoint_sha256"],
                 route_final_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                 client_hashes_sha256=client_hashes.hexdigest(), prefix_checks=sample_replays,
                 raw_alignment_verified=True, output=target.name, seconds=time.perf_counter() - started)
    np.savez_compressed(target, **output, valid=valid, index=np.asarray(json.dumps(rows)), audit=np.asarray(json.dumps(audit)))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round3_safe")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    destination = output / "features"
    destination.mkdir(parents=True, exist_ok=True)
    sources = sorted(str(p.parent.relative_to(HUB)) for p in (HUB / "cache_new/HiMoE-VLA").glob("libero_*/*/*/meta.json") if p.parent.name in RUNS)
    if len(sources) != 80:
        raise ValueError(f"expected 80 complete sources, found {len(sources)}")
    audits = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(extract_source, source, destination): source for source in sources}
        for future in as_completed(jobs):
            audit = future.result()
            audits.append(audit)
            print(f"EXTRACT {len(audits)}/80 {audit['source']}: {audit['queries']} queries, {audit['seconds']:.1f}s", flush=True)
    index = []
    for audit in sorted(audits, key=lambda x: x["source"]):
        with np.load(destination / audit["output"], allow_pickle=False) as archive:
            index.extend(json.loads(str(archive["index"])))
    frame = pd.DataFrame(index)
    if frame.duplicated(["task", "run_id", "episode"]).any():
        raise ValueError("duplicate episode key")
    for suite, part in frame.groupby("suite"):
        if part.checkpoint.nunique() != 1 or part.task.nunique() != 10:
            raise ValueError(f"checkpoint/task mismatch {suite}")
    frame.to_csv(output / "index.csv", index=False)
    write_json(output / "extraction_audit.json", sorted(audits, key=lambda x: x["source"]))
    print(f"EXTRACTED {len(frame)} trajectories, {frame.length.sum()} queries", flush=True)


if __name__ == "__main__":
    main()
