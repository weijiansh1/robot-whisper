"""Render recorded states with the pinned collection simulator (Python 3.8)."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from himoe_libero_bridge.libero_runtime import EpisodeConfig, _load_task
from himoe_libero_bridge.preprocess import frame_from_observation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round4_geometry")
    args = parser.parse_args()
    output = args.output.resolve()
    cases = json.loads((output / "cases.json").read_text())
    audits = []
    for case in cases:
        source = ROOT / "VLA_MUI_HUB" / case["source"]
        meta = json.loads((source / "meta.json").read_text())
        summaries = json.loads((source / "client/summaries.json").read_text())
        summary = next(s for s in summaries if int(s["episode_index"]) == case["episode"])
        path = source / ("client/episode_%02d.npz" % case["episode"])
        with np.load(path, allow_pickle=False) as z:
            states = z["sim_state"].astype(np.float64)
            robot_states = z["state"]
        if len(states) != case["length"] or len(states) != int(summary["inference_calls"]):
            raise ValueError("case length does not match saved states")
        config = EpisodeConfig(
            libero_root=str(ROOT / "himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO"),
            task_suite=meta["benchmark"], task_id=int(meta["task_id"]),
            init_state_id=case["init_state_id"], seed=int(summary.get("seed", 7)),
            max_steps=int(meta["max_steps"]), render_size=320,
        )
        env, _, task, _ = _load_task(config)
        if str(task.name) != case["task"].split("/", 1)[1]:
            raise ValueError("rendered task does not match case")
        destination = output / "frames" / str(case["global_row"])
        destination.mkdir(parents=True, exist_ok=True)
        checks, files = [], {}
        try:
            for q, state in enumerate(states):
                observation = env.set_init_state(state)
                restored = np.asarray(env.get_sim_state())
                nq = int(env.sim.model.nq)
                qpos_error = float(np.max(np.abs(restored[1:1 + nq] - state[1:1 + nq])))
                eef_error = float(np.max(np.abs(observation["robot0_eef_pos"] - robot_states[q, :3])))
                eef_site = env.sim.data.site_xpos[env.env.robots[0].eef_site_id]
                observable_kinematics_error = float(np.max(np.abs(eef_site - observation["robot0_eef_pos"])))
                # Match the existing physical audit's 5 mm observation tolerance.
                if qpos_error > 1e-6 or eef_error > 5e-3:
                    raise AssertionError("recorded state mismatch: qpos=%g eef=%g" % (qpos_error, eef_error))
                image = frame_from_observation(observation)
                if image.shape != (320, 320, 3) or image.std() < 5:
                    raise AssertionError("blank or malformed simulator image")
                filename = destination / ("q%03d.png" % q)
                Image.fromarray(image).save(filename)
                files[str(filename.relative_to(output))] = digest(filename)
                checks.append({"query": q, "qpos_error": qpos_error, "eef_error_m": eef_error,
                               "observable_kinematics_error_m": observable_kinematics_error,
                               "pixel_std": float(image.std()), "sim_time": float(state[0])})
        finally:
            env.close()
        audits.append({"global_row": case["global_row"], "suite": case["suite"], "role": case["role"],
                       "source": str(path.relative_to(ROOT)), "source_sha256": digest(path),
                       "checks": checks, "image_sha256": files})
        print("RENDERED %s %s: %d states" % (case["suite"], case["role"], len(states)), flush=True)
    (output / "render_verification.json").write_text(json.dumps({
        "renderer": "CPU OSMesa; recorded sim_state restoration", "new_rollouts": False,
        "original_rgb_recording": False, "frame_orientation": "collection frame_from_observation",
        "qpos_tolerance": 1e-6, "recorded_eef_tolerance_m": 5e-3,
        "source_sha256": digest(Path(__file__)), "cases": audits,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
