"""Routing reachability on wrong-target (Pro swap) failures.

For each alarmed Pro-swap failure, restore the simulator at alarm-8 to read the true position of the
task's target object and of the end-effector, then query the policy with many forced routing
configurations (all 81 AS combinations, plus random HB top-4 sets per layer) and ask whether ANY
configuration produces a chunk whose end-effector displacement points toward the true target
rather than the (swapped) wrong location.  No environment actions are executed.
"""
import argparse
import itertools
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "srv" / "src"))
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
    os.environ.pop(k, None)


def target_positions(parent, query):
    """Restore the env at `query` (replay) and return EE position, target-object position, and the other object's position."""
    summary = json.loads((pathlib.Path(parent) / "summary.json").read_text())
    trace = np.load(pathlib.Path(parent) / "episode-trace.npz")
    source = pathlib.Path(summary["benchmark_source"])
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "configs" / ("libero-" + summary["benchmark_variant"]))
    sys.path.insert(0, str(source))
    from himoe_libero_bridge.libero_runtime import EpisodeConfig, LIBERO_DUMMY_ACTION, _load_task
    config = EpisodeConfig(libero_root=str(source), output_root="/tmp", task_suite=summary["task_suite"], task_id=int(summary["task_id"]),
                           init_state_id=int(summary["init_state_id"]), seed=int(summary["seed"]), max_steps=520, render_size=256)
    env, obs, task, prompt = _load_task(config)
    for _ in range(config.settle_steps):
        obs, *_ = env.step(LIBERO_DUMMY_ACTION.tolist())
    for q in range(query):
        for a in trace["predicted_actions"][q][: int(trace["executed_lengths"][q])]:
            obs, *_ = env.step(a.tolist())
    inner = env.env
    goal = inner.parsed_problem["goal_state"]
    objs = inner.obj_of_interest
    sim = inner.sim
    def body_pos(name):
        for cand in (name + "_main", name):
            try:
                return np.array(sim.data.body_xpos[sim.model.body_name2id(cand)])
            except Exception:
                continue
        return None
    ee = np.array(obs["robot0_eef_pos"])
    positions = {o: body_pos(o) for o in objs}
    env.close()
    return dict(ee=ee.tolist(), goal=goal, objects={k: (v.tolist() if v is not None else None) for k, v in positions.items()},
                prompt=prompt, images=trace["images"][query], wrist=trace["wrist_images"][query], state=trace["states"][query],
                noise=trace["flow_noises"][query])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", default=str(ROOT / "controls" / "parents-proswap.json"))
    ap.add_argument("--port", type=int, default=9571)
    ap.add_argument("--max-parents", type=int, default=12)
    ap.add_argument("--hb-samples", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "controls" / "reachability-proswap.json"))
    a = ap.parse_args()
    from himoe_libero_bridge.client import PolicyClient
    from himoe_libero_bridge.protocol import unpackb
    parents = [p for p in json.load(open(a.parents)) if not p["success"] and p["first_alarm"] is not None][: a.max_parents]
    client = PolicyClient("127.0.0.1", a.port, connect_timeout=60, inference_timeout=300)
    rng = np.random.default_rng(0)
    as_combos = list(itertools.product(range(3), repeat=4))                       # 81
    hb_sets = [[sorted(rng.choice(32, 4, replace=False).tolist()) for _ in range(8)] for _ in range(a.hb_samples)]
    results = []
    for p in parents:
        q = max(1, p["first_alarm"] - 8)
        info = target_positions(p["dir"], q)
        # the task's target object = first object mentioned in the first goal predicate
        first = info["goal"][0] if info["goal"] else None
        tgt_name = next((o for o in info["objects"] if first and any(o in str(x) for x in first)), None)
        if tgt_name is None or info["objects"].get(tgt_name) is None:
            results.append(dict(tag=p["tag"], skipped="no target position", goal=info["goal"]))
            continue
        tgt = np.array(info["objects"][tgt_name]); ee = np.array(info["ee"])
        others = [np.array(v) for k, v in info["objects"].items() if k != tgt_name and v is not None]
        base = {"observation/image": info["images"], "observation/wrist_image": info["wrist"], "observation/state": info["state"].astype(np.float32),
                "prompt": info["prompt"], "episode_id": -1, "capture/tag": "reach"}
        def query(as_list=None, hb_list=None, K=None):
            req = dict(base)
            K = K or (len(as_list) if as_list is not None else len(hb_list))
            req["candidates/noises"] = np.repeat(info["noise"][None], K, axis=0).astype(np.float32)
            if as_list is not None:
                req["candidates/as_experts"] = np.asarray(as_list, np.int16)
            if hb_list is not None:
                req["candidates/hb_experts"] = np.asarray(hb_list, np.int16)
            client._connection.send(client._packer.pack(req))
            return np.asarray(unpackb(client._connection.recv(timeout=300))["candidates/actions"], np.float32)
        native = query(as_list=[[-1, -1, -1, -1]])[0]
        acts_as = query(as_list=[list(c) for c in as_combos])
        acts_hb = np.concatenate([query(hb_list=hb_sets[i:i + 32]) for i in range(0, len(hb_sets), 32)])
        def score(chunk):
            d = chunk[:, :3].sum(0) * 0.05                                     # approx. displacement in metres over the chunk
            to_t = tgt - ee; to_t = to_t / (np.linalg.norm(to_t) + 1e-9)
            cos_t = float(d @ to_t / (np.linalg.norm(d) + 1e-9))
            cos_o = max((float(d @ ((o - ee) / (np.linalg.norm(o - ee) + 1e-9)) / (np.linalg.norm(d) + 1e-9)) for o in others), default=float("nan"))
            return cos_t, cos_o, float(np.linalg.norm(d))
        nat = score(native)
        rows_as = [score(c) for c in acts_as]; rows_hb = [score(c) for c in acts_hb]
        spread_as = float(np.abs(acts_as[:, :, :6] - native[None, :, :6]).mean() / (np.abs(native[:, :6]).mean() + 1e-9))
        spread_hb = float(np.abs(acts_hb[:, :, :6] - native[None, :, :6]).mean() / (np.abs(native[:, :6]).mean() + 1e-9))
        results.append(dict(tag=p["tag"], task=p["task_name"], target=tgt_name, ee_to_target_m=float(np.linalg.norm(tgt - ee)),
                            native_cos_target=nat[0], native_cos_other=nat[1], native_disp_m=nat[2],
                            as_best_cos_target=max(r[0] for r in rows_as), as_frac_toward_target=float(np.mean([r[0] > 0.5 for r in rows_as])),
                            hb_best_cos_target=max(r[0] for r in rows_hb), hb_frac_toward_target=float(np.mean([r[0] > 0.5 for r in rows_hb])),
                            as_rel_change=spread_as, hb_rel_change=spread_hb, n_as=len(acts_as), n_hb=len(acts_hb)))
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in results[-1].items() if k not in ("task",)}), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
