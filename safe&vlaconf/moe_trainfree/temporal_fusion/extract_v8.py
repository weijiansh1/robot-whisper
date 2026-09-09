"""Align existing v8 inputs and complete missing episodes from raw routing."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from fusion import HERE, ROOT, flow_speed, raw_v8_features
from core import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round7_temporal_fusion")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    frame = pd.read_csv(HERE.parent / "results/round5_knn/index.csv")
    lookup = {(r.task, r.run_id, int(r.episode)): i for i, r in enumerate(frame.itertuples())}
    values = np.full((len(frame), 52, 2), np.nan, np.float32)
    filled = np.zeros(len(frame), bool)
    sources = {}
    for cohort, cache in (("development_main", "main_reference.npz"), ("external_8b", "external_8b.npz")):
        meta_path = ROOT / "moe-v4-0904/results/layerwise_mobility" / cache
        speed_path = ROOT / "moe-v8-0906/results" / f"{cohort}_flow_speed.npz"
        with np.load(meta_path) as meta, np.load(speed_path) as data:
            tasks = meta["task_names"].astype(str)[meta["task_index"]]
            rows = np.asarray([lookup[(task, str(meta["run_id"]), int(ep))] for task, ep in zip(tasks, meta["episode"])])
            for field, key in (("length", "length"), ("init_state_id", "init_state_id"), ("noise_seed", "flow_noise_seed")):
                np.testing.assert_array_equal(frame.iloc[rows][field], meta[key])
            speed = data["flow_speed"]
            np.testing.assert_array_equal(np.isfinite(speed).all(axis=(-2, -1)), meta["valid"])
            values[rows] = raw_v8_features(speed)
            filled[rows] = True
        sources[str(meta_path.relative_to(ROOT))] = digest(meta_path)
        sources[str(speed_path.relative_to(ROOT))] = digest(speed_path)
    reused = int(filled.sum())
    replay_rows = set()
    for _, group in frame.groupby(["suite", "run_id"]):
        replay_rows.update(int(x) for x in group.index[[0, len(group) // 2, -1]])
    selected = frame.loc[~filled | frame.index.isin(replay_rows)]
    raw_checks = []
    for source, group_rows in selected.groupby("source"):
        path = ROOT / "VLA_MUI_HUB" / source
        summary_path = path / "client/summaries.json"
        summaries = sorted(json.loads(summary_path.read_text()), key=lambda x: x["episode_index"])
        offsets, begin = {}, 0
        for summary in summaries:
            length = int(summary["inference_calls"])
            offsets[int(summary["episode_index"])] = (begin, begin + length)
            begin += length
        group = zarr.open_group(str(path / "server/routes.zarr"), mode="r")
        raw = group["hb_router_probs"]
        assert raw.shape[0] == begin
        ids = np.asarray(group["episode_id"][:])
        for record in group_rows.itertuples():
            lo, hi = offsets[int(record.episode)]
            assert hi - lo == record.length
            assert np.all(ids[lo:hi] == int(record.episode))
            extracted = raw_v8_features(flow_speed(np.asarray(raw[lo:hi])))
            if filled[record.Index]:
                np.testing.assert_allclose(values[record.Index, :record.length], extracted, rtol=2e-4, atol=3e-6)
                raw_checks.append(dict(global_row=int(record.Index), queries=int(record.length)))
            else:
                values[record.Index, :record.length] = extracted
                filled[record.Index] = True
        sources[str(summary_path.relative_to(ROOT))] = digest(summary_path)
        print(f"V8 INPUT {source}: {len(group_rows)} episodes", flush=True)
    assert filled.all()
    valid = np.arange(52)[None] < frame.length.to_numpy()[:, None]
    np.testing.assert_array_equal(np.isfinite(values).all(-1), valid)
    np.savez_compressed(output / "v8_inputs.npz", raw=values, valid=valid)
    frame.to_csv(output / "index.csv", index=False)
    write_json(output / "input_audit.json", dict(reused_episodes=reused, completed_episodes=len(frame)-reused,
        raw_replays=raw_checks, inputs=sources, index_sha256=digest(output / "index.csv"),
        output_sha256=digest(output / "v8_inputs.npz"), extractor_sha256=digest(Path(__file__)),
        feature_source_sha256=digest(HERE / "fusion.py")))
    print("V8 INPUTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
