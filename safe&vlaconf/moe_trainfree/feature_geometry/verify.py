"""Verify geometry sources, sample identities, raw replay, and rendered assets."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

from analyze import ROOT, SEED, REPRESENTATIONS, digest, dynamics, load_suite, write_json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "safe_protocol"))
from v7_adapter import raw_episode, extract_episode
from core import route_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round4_geometry")
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    root, parent = args.input.resolve(), args.parent.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    hash_checks = 0
    for category, prefix in (("sources", ROOT), ("inputs", ROOT), ("artifacts", root)):
        for name, expected in manifest[category].items():
            if digest(prefix / name) != expected:
                raise AssertionError(f"changed {category}: {name}")
            hash_checks += 1
    random_audit = json.loads((root / "random_init/audit.json").read_text())
    for name, expected in random_audit["artifacts"].items():
        assert digest(root / "random_init" / name) == expected
        hash_checks += 1
    for name, expected in random_audit["inputs"].items():
        assert digest(root / name) == expected
        hash_checks += 1
    assert digest(HERE / "random_init.py") == random_audit["source_sha256"]
    assert digest(HERE / "RANDOM_INIT_ZH.md") == random_audit["protocol_sha256"]
    frame = pd.read_csv(parent / "outcome_alignment.csv")
    cases = json.loads((root / "cases.json").read_text())
    scaling = json.loads((root / "scaling.json").read_text())
    with np.load(parent / "v7/v7_inputs.npz", allow_pickle=False) as z:
        mobility, acceleration, periodicity = (z[k] for k in ("mobility", "acceleration", "periodicity"))
    replay_checks, coordinate_checks, points_checked = [], 0, 0
    for suite in sorted(frame.suite.unique()):
        for representation in REPRESENTATIONS:
            kind = representation.split("_", 1)[1]
            meta = pd.read_csv(root / "projections" / f"{suite}_{kind}_points.csv")
            assert not meta.duplicated(["global_row", "query"]).any()
            assert ((meta["query"] >= 0) & (meta["query"] < meta.length)).all()
            if kind == "matched":
                assert (meta["query"] >= 7).all()
            expected = frame.iloc[meta.global_row]
            for column in ("task", "episode", "failure", "length", "checkpoint", "init_state_id", "noise_seed"):
                np.testing.assert_array_equal(expected[column], meta[column])
            np.testing.assert_allclose(meta.safe_color, np.where(meta.failure, meta["query"] / (meta.length - 1), 0.))
            seeds = (SEED,) if kind == "full" else (SEED, SEED + 1, SEED + 2)
            for seed in seeds:
                with np.load(root / "projections" / f"{suite}_{representation}_seed{seed}.npz", allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["point_global_rows"], meta.global_row)
                    np.testing.assert_array_equal(z["point_queries"], meta["query"])
                    assert np.isfinite(z["xy"]).all() and (z["xy"].std(0) > 0).all()
                    coordinate_checks += 1
            points_checked += len(meta)
        for case in [c for c in cases if c["suite"] == suite]:
            row = frame.iloc[case["global_row"]]
            raw = raw_episode(row)
            m, a, p = extract_episode(raw)
            for stored, rebuilt in ((mobility[case["global_row"], :case["length"]], m),
                                     (acceleration[case["global_row"], :case["length"]], a),
                                     (periodicity[case["global_row"], :case["length"]], p)):
                np.testing.assert_allclose(stored, rebuilt, rtol=2e-4, atol=3e-6, equal_nan=True)
            route = route_features(raw[:, :, 9])["load"]
            normalized = (route - np.asarray(scaling[suite]["routing"]["center"])) / np.asarray(scaling[suite]["routing"]["scale"])
            meta = pd.read_csv(root / "projections" / f"{suite}_full_points.csv")
            mask = meta.global_row == case["global_row"]
            np.testing.assert_array_equal(meta.loc[mask, "query"], np.arange(case["length"]))
            with np.load(root / "projections" / f"{suite}_full_vectors.npz", allow_pickle=False) as z:
                np.testing.assert_allclose(normalized, z["routing"][mask], rtol=1e-5, atol=1e-5)
            replay_checks.append({"global_row": case["global_row"], "queries": len(raw), "raw_route_and_v7_match": True})
    # Independently sort direct float64 distances for selected diagnostic rows.
    neighbors = pd.read_csv(root / "neighbor_points.csv")
    assert not neighbors.duplicated(["representation", "global_row", "query"]).any()
    oracle_checks = []
    for suite in sorted(frame.suite.unique()):
        part, arrays = load_suite(parent, frame, suite, {})
        g = part.global_row.to_numpy()
        d, _ = dynamics(mobility[g], acceleration[g], periodicity[g], scaling[suite]["periodicity_scale"])
        arrays["dynamics"] = d
        for name in ("routing", "dynamics"):
            arrays[name] = ((arrays[name] - np.asarray(scaling[suite][name]["center"])) /
                            np.asarray(scaling[suite][name]["scale"])).astype(np.float32)
        for q in (7, 14, 21):
            eligible = (part.run_id == "right-50x8b-20260903").to_numpy() & (part.length.to_numpy() > q)
            eligible &= np.isfinite(arrays["dynamics"][:, q]).all(-1)
            rows = np.flatnonzero(eligible)
            current = part.iloc[rows]
            for name in ("routing", "dynamics"):
                x = arrays[name][rows, q].astype(np.float64)
                for failed in (False, True):
                    choices = np.flatnonzero(current.failure.to_numpy() == failed)
                    if not len(choices):
                        continue
                    i = int(choices[len(choices) // 2])
                    allowed = current.task.to_numpy() != current.iloc[i].task
                    distance = np.square(x - x[i]).sum(-1)
                    distance[~allowed] = np.inf
                    found = np.argsort(distance)[:20]
                    expected = float(current.failure.to_numpy()[found].mean())
                    actual = neighbors.loc[(neighbors.global_row == int(current.iloc[i].global_row)) &
                                           (neighbors.representation == name) & (neighbors["query"] == q)].iloc[0]
                    np.testing.assert_allclose(actual.neighbor_failure_rate, expected, atol=1e-12)
                    np.testing.assert_allclose(actual.candidate_failure_rate, current.failure.to_numpy()[allowed].mean(), atol=1e-12)
                    oracle_checks.append({"suite": suite, "representation": name, "query": q,
                                          "global_row": int(current.iloc[i].global_row)})
    render = json.loads((root / "render_verification.json").read_text())
    assert digest(HERE / "render_states.py") == render["source_sha256"]
    rendered_frames = 0
    for case in render["cases"]:
        assert digest(ROOT / case["source"]) == case["source_sha256"]
        for name, expected in case["image_sha256"].items():
            assert digest(root / name) == expected
            with Image.open(root / name) as image:
                pixels = np.asarray(image.convert("RGB"))
                assert pixels.shape == (320, 320, 3) and pixels.std() > 5
            rendered_frames += 1
    figures = json.loads((root / "figure_manifest.json").read_text())
    assert digest(HERE / "plot.py") == figures["plotter_sha256"]
    image_checks = []
    for name, expected in figures["files"].items():
        assert digest(root / name) == expected
        if name.endswith((".png", ".gif")):
            with Image.open(root / name) as image:
                assert np.asarray(image.convert("RGB")).std() > 5
                if name.endswith(".gif"):
                    suite = Path(name).name.replace("_recorded_path.gif", "")
                    case = next(c for c in cases if c["suite"] == suite and c["role"] == "failure")
                    assert image.n_frames == case["length"]
                    before = np.asarray(image.convert("RGB")).astype(float)
                    image.seek(image.n_frames - 1)
                    assert np.abs(np.asarray(image.convert("RGB")) - before).mean() > .5
                image_checks.append({"name": name, "size": list(image.size), "frames": getattr(image, "n_frames", 1)})
    write_json(root / "final_verification.json", {"hash_checks": hash_checks, "coordinate_checks": coordinate_checks,
        "annotation_rows_checked": points_checked, "raw_episode_replays": replay_checks,
        "independent_neighbor_oracles": oracle_checks, "rendered_frames_checked": rendered_frames,
        "image_checks": image_checks, "verifier_sha256": digest(Path(__file__))})
    print(f"VERIFIED: {coordinate_checks} projections, {len(replay_checks)} raw episodes, "
          f"{len(oracle_checks)} independent neighbor checks, {rendered_frames} rendered frames", flush=True)


if __name__ == "__main__":
    main()
