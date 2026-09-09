"""Join audited all-task routing caches and fill missing per-layer flow paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import zarr

from encoder import query_readout

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v82_validation"))
from run_analysis import BASE, ROOT, archive, digest, historical_map, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "routing_dynamics_20260908")
    parser.add_argument("--reuse-readouts", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = HERE / "PROTOCOL_ZH.md"
    write_json(output / "execution_contract.json", dict(
        protocol_sha256=digest(protocol), encoder_sha256=digest(HERE / "encoder.py"),
        raw_online_input_only=True, duration_used_as_score=False, time_only_baseline=False,
        new_rollouts=False, detector_training=False, outcome_dependent_feature_selection=False,
        retrospective=True, primary_feature="decoupling", reference_queries=[1, 2, 3, 4],
        smooth_width=3, history_width=4, epsilon=1e-6))
    if args.reuse_readouts is not None:
        previous_output = args.reuse_readouts.resolve()
        prior_path = previous_output / "input_verification.json"
        prior = json.loads(prior_path.read_text())
        for name in ("index.csv", "readouts.npz"):
            assert digest(previous_output / name) == prior["artifacts"][name]
            shutil.copyfile(previous_output / name, output / name)
        prior.update(reused_from=str(previous_output), reused_manifest_sha256=digest(prior_path),
                     source_sha256=digest(__file__),
                     artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()})
        write_json(output / "input_verification.json", prior)
        print("READOUTS REUSED with hashes verified", flush=True)
        return
    previous = BASE / "v82_validation_20260908"
    index_path = previous / "index.csv"
    expected = json.loads((previous / "input_verification.json").read_text())["inputs"]
    old_analysis = json.loads((previous / "analysis_verification.json").read_text())
    assert digest(index_path) == old_analysis["artifacts"]["index.csv"]
    frame = pd.read_csv(index_path)
    assert len(frame) == 32000 and np.array_equal(frame.global_row, np.arange(len(frame)))
    inputs = {str(index_path): digest(index_path)}
    cache_path = BASE / "round3_safe/v7/v7_inputs.npz"
    assert digest(cache_path) == expected[str(cache_path.relative_to(ROOT))]
    cached = archive(cache_path)
    valid = cached["valid"]
    np.testing.assert_array_equal(valid, np.arange(52)[None] < frame.length.to_numpy()[:, None])
    values = np.full((32000, 52, 25), np.nan, np.float32)
    values[..., :8] = cached["mobility"]
    values[..., 16] = cached["acceleration"]
    inputs[str(cache_path)] = digest(cache_path)
    audits = json.loads((BASE / "round3_safe/extraction_audit.json").read_text())
    by_source = {a["source"]: a for a in audits}
    for source, part in frame.groupby("source", sort=True):
        path = BASE / "round3_safe/features" / by_source[source]["output"]
        inputs[str(path)] = digest(path)
        assert inputs[str(path)] == expected[str(path.relative_to(ROOT))]
        with np.load(path, allow_pickle=False) as z:
            saved = pd.DataFrame(json.loads(str(z["index"])))
            for name in ("episode", "init_state_id", "noise_seed", "length"):
                np.testing.assert_array_equal(part[name], saved[name])
            values[part.index, :, 17:25] = z["stats"].reshape(400, 52, 8, 4)[..., 2]
    filled = np.zeros(32000, bool)
    speed_audit = json.loads((BASE / "round7_temporal_fusion/input_audit.json").read_text())["inputs"]
    for metadata_name, speed_name in (("main_reference.npz", "development_main_flow_speed.npz"),
                                       ("external_8b.npz", "external_8b_flow_speed.npz")):
        metadata_path = ROOT / "moe-v4-0904/results/layerwise_mobility" / metadata_name
        speed_path = ROOT / "moe-v8-0906/results" / speed_name
        for path in (metadata_path, speed_path):
            inputs[str(path)] = digest(path)
            assert inputs[str(path)] == speed_audit[str(path.relative_to(ROOT))]
        meta = archive(metadata_path)
        rows = historical_map(frame, meta)
        with np.load(speed_path, allow_pickle=False) as z:
            speed = z["flow_speed"]
        np.testing.assert_array_equal(np.isfinite(speed).all(axis=(-2, -1)), valid[rows])
        assert not filled[rows].any()
        values[rows, :, 8:16] = speed.sum(-1)
        filled[rows] = True
        del speed, meta
    replay_rows = set()
    for _, part in frame.groupby(["suite", "run_id"]):
        replay_rows.add(int(part.index[0]))
        replay_rows.add(int(part.loc[part.failure].index[0]))
    s05 = "libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
    replay_rows.update(frame.index[frame.task.eq(s05) & frame.episode.isin([77, 96, 174, 190])].tolist())
    to_read = frame.loc[~filled | frame.index.isin(replay_rows)]
    queries_checked, episodes_completed, replayed = 0, 0, []
    for source, part in to_read.groupby("source", sort=True):
        run = ROOT / "VLA_MUI_HUB" / source
        summary_path = run / "client/summaries.json"
        inputs[str(summary_path)] = digest(summary_path)
        assert inputs[str(summary_path)] == by_source[source]["summary_sha256"]
        summaries = sorted(json.loads(summary_path.read_text()), key=lambda s: s["episode_index"])
        offsets = np.r_[0, np.cumsum([s["inference_calls"] for s in summaries])]
        store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        for row in part.itertuples():
            lo, hi = offsets[row.episode:row.episode + 2]
            assert hi - lo == row.length
            assert np.all(store["episode_id"][lo:hi] == row.episode)
            raw = np.asarray(store["hb_router_probs"][lo:hi])
            previous_final, extracted = None, []
            for query in raw:
                result, previous_final = query_readout(query, previous_final)
                extracted.append(result)
            extracted = np.asarray(extracted, np.float32)
            ours = values[row.Index, :row.length]
            for columns in (slice(0, 8), slice(16, 17), slice(17, 25)):
                np.testing.assert_allclose(extracted[:, columns], ours[:, columns], rtol=2e-4, atol=3e-6, equal_nan=True)
            if filled[row.Index]:
                np.testing.assert_allclose(extracted[:, 8:16], ours[:, 8:16], rtol=2e-4, atol=3e-6)
            else:
                values[row.Index, :row.length, 8:16] = extracted[:, 8:16]
                filled[row.Index] = True
                episodes_completed += 1
            queries_checked += row.length
            if row.Index in replay_rows:
                replayed.append(int(row.Index))
        print(f"READOUTS {source}: {len(part)} raw episodes", flush=True)
    assert filled.all()
    assert np.isfinite(values[..., 8:][valid]).all()
    assert np.isfinite(values[:, 1:, :8][valid[:, 1:]]).all()
    path = values[..., 8:16]
    v8_path = BASE / "round7_temporal_fusion/v8_inputs.npz"
    inputs[str(v8_path)] = digest(v8_path)
    assert inputs[str(v8_path)] == expected[str(v8_path.relative_to(ROOT))]
    with np.load(v8_path, allow_pickle=False) as z:
        np.testing.assert_allclose(-np.log(path[..., :4].mean(-1) / path[..., 4:].mean(-1)),
                                   z["raw"][..., 0], rtol=2e-4, atol=3e-6, equal_nan=True)
    np.savez_compressed(output / "readouts.npz", values=values, valid=valid, global_rows=frame.global_row.to_numpy())
    frame.to_csv(output / "index.csv", index=False)
    write_json(output / "input_verification.json", dict(
        all_checks_passed=True, episodes=32000, queries=int(valid.sum()), sources=80,
        cached_flow_episodes=32000 - episodes_completed, completed_flow_episodes=episodes_completed,
        raw_queries_checked=queries_checked, representative_replay_rows=replayed, inputs=inputs,
        source_sha256=digest(__file__), artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
    print(f"READOUTS COMPLETE: {valid.sum()} queries, {episodes_completed} completed flow episodes", flush=True)


if __name__ == "__main__":
    main()
