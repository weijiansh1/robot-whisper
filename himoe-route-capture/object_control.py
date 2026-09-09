"""Redo the residual decode with the object state added to the control.

pc_meaning7 showed routing beating proprioception at control steps 10-12, peaking
at step 11 (0.852 raw, 0.714 after a k-NN proprio residual, while proprio alone is
at chance).  That control covers the robot only.  On the drawer tasks the cabinet's
slide joint is the obvious uncontrolled progress variable, and it is visible to the
policy through the cameras but absent from ``observation/state``.

This script runs the same sweep against three controls:

  robot    observation/state, 8 dims -- what pc_meaning7 used
  object   task object DOFs sliced out of the captured MuJoCo state
  both     the union, which is the control the step-11 claim has to survive

Needs a capture made with the patched rollout_with_routes.py (``sim_state`` in the
episode npz plus ``sim_layout.json``); it exits with a clear message otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from within64_lib import action_token_probs, load_run
from within64_analyze import decode, reduce_dims
from pc_meaning6 import knn_residual


def robot_qpos_features(sim: np.ndarray, layout: dict) -> np.ndarray:
    """The arm's actual joint angles plus the two fingers, 9 dims.

    The policy's ``observation/state`` gives the end effector's pose, which a 7-DOF
    arm can reach in a one-parameter family of joint configurations -- and it is the
    joint configuration, not the eef pose, that the cameras see.  Controlling on the
    joints closes that gap without the dimensionality blow-up of matching on all 41.
    """
    cols = [sim[:, j["state_lo"]:j["state_hi"]] for j in layout["joints"] if j["is_robot"]]
    if not cols:
        raise RuntimeError("no robot joints in sim_layout.json")
    return np.column_stack(cols)


def object_features(sim: np.ndarray, layout: dict, full: bool = False) -> np.ndarray:
    """Task-object DOFs from the flattened MuJoCo state.

    Free joints contribute their 3 position components only: the quaternion adds
    four correlated columns whose sign is arbitrary, which a nearest-neighbour
    control handles badly.  ``full`` keeps every non-robot qpos entry instead.
    """
    cols = []
    for j in layout["joints"]:
        if j["is_robot"]:
            continue
        lo, hi = j["state_lo"], j["state_hi"]
        cols.append(sim[:, lo:hi] if (full or hi - lo != 7) else sim[:, lo:lo + 3])
    if not cols:
        raise RuntimeError("no non-robot joints in sim_layout.json")
    return np.column_stack(cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--sim-dir", help="replay_sim_state.py sidecar; defaults to --client-dir "
                                      "for captures written by the patched rollout script")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-step", type=int, default=17)
    ap.add_argument("--full-object-qpos", action="store_true")
    ap.add_argument("--robot-qpos", action="store_true",
                    help="control the robot on its 9 joint positions rather than on "
                         "the policy's 8-dim eef pose; the tightest control that "
                         "still covers the arm configuration the cameras see")
    ap.add_argument("--all-qpos", action="store_true",
                    help="control on every joint position in the scene, robot arm "
                         "included, rather than on the policy's 8-dim eef state. "
                         "With fixed cameras and a deterministic renderer the "
                         "images are a function of qpos, so this is the closest "
                         "available stand-in for controlling the observation "
                         "itself -- at the cost of a 41-dim nearest-neighbour "
                         "match, which under-removes and so flatters the routing.")
    args = ap.parse_args()

    client = pathlib.Path(args.client_dir)
    sim_dir = pathlib.Path(args.sim_dir) if args.sim_dir else client
    layout_path = sim_dir / "sim_layout.json"
    if not layout_path.exists():
        sys.exit("no sim_layout.json in %s -- either re-run the capture with the "
                 "patched rollout_with_routes.py, or retrofit this one with "
                 "replay_sim_state.py and pass --sim-dir" % sim_dir)
    layout = json.loads(layout_path.read_text())

    run = load_run(args.server_dir, args.client_dir)

    # a retrofitted capture only has trustworthy object state where the replay
    # tracked the recorded run; drop the rest, and report whether the drop is
    # outcome-biased, because that would bias everything downstream
    keep = None
    check_path = sim_dir / "replay_check.json"
    if check_path.exists():
        checks = json.loads(check_path.read_text())
        keep = {c["episode"] for c in checks if c.get("ok", True)}
        dropped = [c for c in checks if not c.get("ok", True)]
        n_ok_drop = sum(1 for c in dropped if c["success_recorded"])
        kept = [c for c in checks if c.get("ok", True)]
        print("replay gate: keeping %d/%d episodes; kept success %d/%d, dropped success %d/%d"
              % (len(keep), len(checks),
                 sum(1 for c in kept if c["success_recorded"]), len(kept),
                 n_ok_drop, len(dropped)))

    feats, t, y, ep, robot, sim = [], [], [], [], [], []
    for e in run.episodes:
        if keep is not None and e.index not in keep:
            continue
        feats.append(action_token_probs(e).reshape(e.n_control, -1))
        t += list(range(e.n_control))
        y += [int(e.success)] * e.n_control
        ep += [e.index] * e.n_control
        with np.load(client / ("episode_%02d.npz" % e.index)) as z:
            robot.append(z["state"])
            s = z["sim_state"] if "sim_state" in z.files else None
        if s is None:
            side = sim_dir / ("sim_state_%02d.npy" % e.index)
            if not side.exists():
                sys.exit("no sim state for episode %d (neither in the npz nor at %s)"
                         % (e.index, side))
            s = np.load(side)
        if s.shape[0] != e.n_control:
            sys.exit("episode %d: sim_state has %d rows but %d control steps"
                     % (e.index, s.shape[0], e.n_control))
        sim.append(s)
    x = np.concatenate(feats).astype(np.float64)
    t, y, ep = np.array(t), np.array(y), np.array(ep)
    robot = np.concatenate(robot).astype(np.float64)
    sim_all = np.concatenate(sim).astype(np.float64)
    obj = object_features(sim_all, layout, args.full_object_qpos or args.all_qpos)
    if args.all_qpos or args.robot_qpos:
        # Under --all-qpos the "both" column has to be every joint position exactly
        # once.  Taking it as the raw sim slice instead put the object DOFs in twice
        # -- once inside the slice and once again from ``obj`` -- which made that
        # column a 57-dim mixture rather than the scene's 41 qpos, and so not
        # comparable with the other three control specifications.  Splitting the
        # slice by joint gives the same 41 columns (verified: robot 9 + object 32),
        # and keeps the robot/object columns individually meaningful.
        robot = robot_qpos_features(sim_all, layout)
    both = np.column_stack([robot, obj])
    if args.all_qpos and both.shape[1] != layout["nq"]:
        raise RuntimeError("--all-qpos should give exactly nq=%d control dims, got %d"
                           % (layout["nq"], both.shape[1]))
    print("control dims: robot=%d  object=%d  both=%d" % (robot.shape[1], obj.shape[1], both.shape[1]))
    names = [j["joint"] for j in layout["joints"] if not j["is_robot"]]
    print("object joints:", ", ".join(names))

    residuals = {name: knn_residual(x, c, ep)
                 for name, c in (("robot", robot), ("object", obj), ("both", both))}

    # "every episode still alive" is relative to the episodes that survived the
    # replay gate, not to the capture's original count
    n_eps = int(np.unique(ep).size)
    rows = []
    for s in range(args.max_step + 1):
        m = t == s
        if m.sum() < n_eps:
            continue
        r = {"step": int(s), "n": int(m.sum()), "n_success": int(y[m].sum())}
        d = decode(reduce_dims(x[m]), y[m])
        r["raw"], r["raw_p"] = round(d["balanced_accuracy"], 3), round(d["permutation_p"], 4)
        for name, res in residuals.items():
            d = decode(reduce_dims(res[m]), y[m])
            r["res_" + name] = round(d["balanced_accuracy"], 3)
            r["res_%s_p" % name] = round(d["permutation_p"], 4)
        for name, c in (("robot", robot), ("object", obj), ("both", both)):
            d = decode(c[m], y[m])
            r["ctrl_" + name] = round(d["balanced_accuracy"], 3)
            r["ctrl_%s_p" % name] = round(d["permutation_p"], 4)
        rows.append(r)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "object_joints": names}, indent=2))

    print("\n%-5s | %-15s | %-38s | %-38s" % ("", "routing", "routing 扣掉对照后", "对照本身"))
    print("%-5s | %-15s | %-12s %-12s %-12s | %-12s %-12s %-12s"
          % ("step", "raw", "-robot", "-object", "-both", "robot", "object", "both"))
    for r in rows:
        print("%-5d | %-15s | %-12s %-12s %-12s | %-12s %-12s %-12s"
              % (r["step"], "%.3f p=%.3f" % (r["raw"], r["raw_p"]),
                 "%.3f p=%.3f" % (r["res_robot"], r["res_robot_p"]),
                 "%.3f p=%.3f" % (r["res_object"], r["res_object_p"]),
                 "%.3f p=%.3f" % (r["res_both"], r["res_both_p"]),
                 "%.3f" % r["ctrl_robot"], "%.3f" % r["ctrl_object"],
                 "%.3f" % r["ctrl_both"]))
    print("\ngo/no-go: 第 10-12 步的 -both 列若仍显著，第 11 步窗口成立；若归零，"
          "则那个窗口只是 8 维 proprio 看不见的进度读出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
