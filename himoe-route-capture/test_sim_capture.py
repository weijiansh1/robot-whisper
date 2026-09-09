"""Exercise the patched capture path without a model server.

A stub client returns a fixed, protocol-valid action chunk, so the whole
``run_one`` path runs for real -- env, settle, replan, npz layout -- and the one
thing under test is whether the new ``sim_state`` column is captured, aligned and
saved correctly.

The alignment assertion is the point: ``sim_state[i]`` must be the state the
policy saw at control step ``i``, not the state after that chunk executed.  It is
checked by re-deriving the robot's finger positions from the sim state and
comparing them against ``observation/state``, which the policy built at the same
instant.

Run in the LIBERO env, no GPU and no policy needed.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from himoe_libero_bridge.libero_runtime import EpisodeConfig
from himoe_libero_bridge.protocol import ACTION_CHUNK_STEPS, ACTION_DIM, ACTION_KEY

from rollout_with_routes import run_one


class StubClient:
    """Returns a small constant reach-down action so the drawer joint stays put
    but the arm actually moves; enough to make the alignment check meaningful."""

    def __init__(self):
        self.calls = 0

    def infer(self, request):
        self.calls += 1
        actions = np.zeros((ACTION_CHUNK_STEPS, ACTION_DIM), dtype=np.float32)
        actions[:, 2] = -0.25          # descend
        actions[:, 6] = -1.0           # gripper open
        return {ACTION_KEY: actions}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--out", default="/tmp/sim-capture-test")
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()

    config = EpisodeConfig(
        task_suite="libero_goal", task_id=0, init_state_id=24, seed=7,
        host="127.0.0.1", port=0, libero_root=args.libero_root,
        output_root=args.out, settle_steps=10, max_steps=args.max_steps,
        replan_steps=10,
    )
    client = StubClient()
    summary, ids, w, states, actions, sim_states, layers, layout = run_one(
        config, client, flow_noise_seed=1000, no_capture=True)

    n = summary["inference_calls"]
    print("control steps      :", n)
    print("state      shape   :", states.shape)
    print("sim_state  shape   :", sim_states.shape, "(expected [%d, %d])"
          % (n, layout["state_dim"]))
    assert sim_states.shape == (n, layout["state_dim"]), "sim_state shape mismatch"
    assert states.shape[0] == n, "proprio/sim row count disagree"

    by_name = {j["joint"]: j for j in layout["joints"]}
    fingers = [j for j in by_name if j.startswith("gripper0_")]
    print("gripper joints     :", fingers)

    # alignment: proprio dims 6,7 are the two finger positions; they must match
    # the same quantity read out of the sim state at the same row
    f_lo = by_name[fingers[0]]["state_lo"]
    f_hi = by_name[fingers[-1]]["state_hi"]
    sim_fingers = sim_states[:, f_lo:f_hi]
    err = np.abs(sim_fingers - states[:, 6:8]).max()
    print("max |sim fingers - observation/state[6:8]| = %.3e" % err)
    assert err < 1e-5, "sim_state is not aligned with the policy observation"

    drawer = by_name["wooden_cabinet_1_middle_level"]
    d = sim_states[:, drawer["state_lo"]:drawer["state_hi"]].ravel()
    print("drawer joint per control step:", np.round(d, 4))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "episode_00.npz", expert_ids=ids, expert_weights=w,
                        state=states, actions=actions, sim_state=sim_states,
                        layer_indices=layers)
    (out / "sim_layout.json").write_text(json.dumps(layout, indent=2))
    with np.load(out / "episode_00.npz") as z:
        assert "sim_state" in z.files, "sim_state missing from npz"
        print("npz keys           :", sorted(z.files))
        print("npz sim_state      :", z["sim_state"].shape, z["sim_state"].dtype)
    print("\nOK: sim state captured, aligned, and round-trips through the npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
