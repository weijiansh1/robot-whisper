"""Smoke test for the object-state capture, before touching the capture script.

The proprio control used in pc_meaning6/7 covers the robot only -- eef pose and
the two fingers.  On ``open_the_middle_drawer_of_the_cabinet`` the drawer's own
joint is the obvious uncontrolled progress variable, and it is not in
``observation/state``.  This checks three things without needing the model server:

  * ``get_sim_state()`` exists on the env the capture actually builds and returns
    a flat vector
  * the drawer joint can be located inside it by name, so the analysis can slice
    the object DOFs out of the robot ones
  * that entry actually moves when the drawer is pulled

Run in the LIBERO env, no GPU and no policy needed.
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from himoe_libero_bridge.libero_runtime import EpisodeConfig, _load_task, LIBERO_DUMMY_ACTION

    config = EpisodeConfig(
        task_suite=args.benchmark,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        seed=args.seed,
        host="127.0.0.1",
        port=0,
        libero_root=args.libero_root,
        output_root="/tmp",
    )
    env, _obs, task, prompt = _load_task(config)
    print("task   :", task.name)
    print("prompt :", prompt)

    s0 = np.asarray(env.get_sim_state(), np.float64)
    sim = env.sim
    nq, nv = int(sim.model.nq), int(sim.model.nv)
    print("sim state dim = %d   (nq=%d, nv=%d, so layout is [time] + qpos + qvel)"
          % (s0.shape[0], nq, nv))

    # joint name -> slice inside the flattened state (qpos starts at index 1)
    layout = []
    for name in sim.model.joint_names:
        addr = sim.model.get_joint_qpos_addr(name)  # int for 1-DOF, (lo, hi) for free/ball
        lo, hi = (int(addr[0]), int(addr[1])) if isinstance(addr, tuple) else (int(addr), int(addr) + 1)
        layout.append({"joint": name, "qpos_lo": lo, "qpos_hi": hi})
    print("\n%d joints; the ones that are not the robot arm/gripper:" % len(layout))
    for e in layout:
        if e["joint"].startswith(("robot0_", "gripper0_")):
            continue
        print("  %-42s qpos[%3d:%3d]  init=%s"
              % (e["joint"], e["qpos_lo"], e["qpos_hi"],
                 np.round(s0[1 + e["qpos_lo"]: 1 + e["qpos_hi"]], 4)))

    # drive the dummy action for a while; report which non-robot DOFs moved
    for _ in range(30):
        env.step(LIBERO_DUMMY_ACTION.tolist())
    s1 = np.asarray(env.get_sim_state(), np.float64)
    print("\nafter 30 settle steps, largest |delta| among non-robot qpos entries:")
    moved = []
    for e in layout:
        if e["joint"].startswith(("robot0_", "gripper0_")):
            continue
        d = np.abs(s1[1 + e["qpos_lo"]: 1 + e["qpos_hi"]]
                   - s0[1 + e["qpos_lo"]: 1 + e["qpos_hi"]]).max()
        moved.append((d, e["joint"]))
    for d, name in sorted(moved, reverse=True)[:6]:
        print("  %-42s %.6f" % (name, d))

    print("\nobj_of_interest:", getattr(env, "obj_of_interest", None))
    print("check_success  :", env.check_success())
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
