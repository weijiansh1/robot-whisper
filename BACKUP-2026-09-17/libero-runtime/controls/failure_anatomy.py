"""Replay one recorded episode and log, at every query boundary: end-effector position, gripper command,
target-object position, whether the object is lifted / in hand.  Used to classify failure modes and to place
the v8.2 alarm relative to the first grasp attempt.  Writes <out>/<tag>.json.  No policy calls."""
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "srv" / "src"))


def main(parent, tag, first_alarm, out):
    summary = json.loads((pathlib.Path(parent) / "summary.json").read_text())
    trace = np.load(pathlib.Path(parent) / "episode-trace.npz")
    source = pathlib.Path(summary["benchmark_source"])
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "configs" / ("libero-" + summary["benchmark_variant"]))
    sys.path.insert(0, str(source))
    if summary["benchmark_variant"] == "plus":
        sys.path.insert(0, str(ROOT / "dependencies" / "libero-plus"))
        sys.path.insert(0, str(ROOT))
        import fast_perturbations
    from himoe_libero_bridge.libero_runtime import EpisodeConfig, LIBERO_DUMMY_ACTION, _load_task
    config = EpisodeConfig(libero_root=str(source), output_root="/tmp", task_suite=summary["task_suite"], task_id=int(summary["task_id"]),
                           init_state_id=int(summary["init_state_id"]), seed=int(summary["seed"]), max_steps=520, render_size=128)
    env, obs, task, prompt = _load_task(config)
    if summary["benchmark_variant"] == "plus":
        fast_perturbations.install()
    for _ in range(config.settle_steps):
        obs, *_ = env.step(LIBERO_DUMMY_ACTION.tolist())
    inner = env.env
    sim = inner.sim
    goal = inner.parsed_problem["goal_state"]
    objs = list(inner.obj_of_interest)

    def body_pos(name):
        for cand in (name + "_main", name):
            try:
                return np.array(sim.data.body_xpos[sim.model.body_name2id(cand)])
            except Exception:
                continue
        return None
    first = goal[0] if goal else []
    target = next((o for o in objs if any(o in str(x) for x in first)), objs[0] if objs else None)
    init_pos = {o: body_pos(o) for o in objs}
    rows = []
    Q = len(trace["predicted_actions"])
    for q in range(Q):
        ee = np.array(obs["robot0_eef_pos"])
        tp = body_pos(target) if target else None
        grip_cmd = float(np.sign(trace["predicted_actions"][q][:, 6].mean()))
        gq = obs.get("robot0_gripper_qpos")
        opening = float(abs(gq[0] - gq[1])) if gq is not None else None
        row = dict(q=q, step=q * 10, ee=ee.tolist(), gripper_cmd=grip_cmd, gripper_opening=opening,
                   target=target, target_pos=tp.tolist() if tp is not None else None,
                   ee_target_dist=float(np.linalg.norm(ee - tp)) if tp is not None else None,
                   target_lift=float(tp[2] - init_pos[target][2]) if tp is not None and init_pos[target] is not None else None,
                   objects={o: (body_pos(o) - init_pos[o]).tolist() for o in objs if init_pos[o] is not None})
        rows.append(row)
        for a in trace["predicted_actions"][q][: int(trace["executed_lengths"][q])]:
            obs, *_ = env.step(a.tolist())
    env.close()
    # derived events
    def first_q(cond):
        return next((r["q"] for r in rows if cond(r)), None)
    close_q = first_q(lambda r: r["gripper_cmd"] > 0)
    inhand_q = first_q(lambda r: r["ee_target_dist"] is not None and r["ee_target_dist"] < 0.04 and r["gripper_cmd"] > 0)
    lift_q = first_q(lambda r: r["target_lift"] is not None and r["target_lift"] > 0.02)
    drop_q = None
    if lift_q is not None:
        drop_q = next((r["q"] for r in rows if r["q"] > lift_q and r["target_lift"] is not None and r["target_lift"] < 0.01), None)
    # stuck: EE moved < 1 cm over 5 consecutive queries
    stuck_q = None
    for i in range(5, len(rows)):
        ees = np.array([rows[j]["ee"] for j in range(i - 5, i + 1)])
        if np.linalg.norm(ees.max(0) - ees.min(0)) < 0.01:
            stuck_q = rows[i - 5]["q"]
            break
    events = dict(tag=tag, success=summary["success"], steps=summary["action_steps"], queries=Q, first_alarm=first_alarm, target=target,
                  first_close_q=close_q, first_inhand_q=inhand_q, first_lift_q=lift_q, drop_q=drop_q, stuck_q=stuck_q,
                  min_ee_target_dist=min((r["ee_target_dist"] for r in rows if r["ee_target_dist"] is not None), default=None),
                  final_target_lift=rows[-1]["target_lift"], other_objects_moved={o: float(np.linalg.norm(v)) for o, v in rows[-1]["objects"].items()})
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    json.dump(dict(events=events, rows=rows), open(pathlib.Path(out) / (tag + ".json"), "w"))
    print(json.dumps(events))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], None if sys.argv[3] == "None" else int(sys.argv[3]), sys.argv[4])
