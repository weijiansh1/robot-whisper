"""Branch a recorded LIBERO episode at a query and continue it under a control strategy.

The parent episode (run_benchmark artifact: summary.json + episode-trace.npz) is replayed
action-for-action up to the branch query, the simulator state is checked against the recorded
one, and the remaining budget is executed with candidate action chunks sampled from the policy
(K flow noises in one batched request) and a selection rule:

  native          parent's own noise sequence (control arm; = the parent if numerics agree)
  random          one fresh noise per query (plain resampling)
  boundary        candidate farthest from the candidate mean in normalised action space
  small_cluster   medoid of the smallest cluster (average-linkage, 3 clusters) of candidate chunks
  big_cluster     medoid of the largest cluster
  motion_max      candidate with the largest end-effector displacement over the chunk
  route_exit      candidate whose HB routing state is farthest from the parent's recent routing states
  route_stay      ... nearest
  flow_straight   candidate whose denoising path is straightest (net / path length)
  flow_curved     ... most curved
  v82_min         candidate that minimises the v8.2 freeze score if its routing were appended
  kick            execute a fixed recovery primitive (open gripper, lift) for one chunk, then native

``--control-queries N`` applies the rule for N queries after the branch (then native sampling);
``--control-queries -1`` keeps applying it until the v8.2 alarm is no longer latched (max 12).
"""
import argparse
import copy
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "srv" / "src"))
sys.path.insert(0, str(ROOT / "dependencies" / "libero-plus"))
sys.path.insert(0, str(ROOT.parent / "coding" / "robot-whisper-0909" / "moe-trap-control"))
sys.path.insert(0, str(ROOT))

ROUTE_STRATEGIES = ("route_exit", "route_stay", "v82_min")
FLOW_STRATEGIES = ("flow_straight", "flow_curved")


def _chunk(move, gripper, hold_gripper=None, n_move=5):
    rows = [list(move) + [gripper]] * n_move + [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, hold_gripper if hold_gripper is not None else gripper]] * (10 - n_move)
    return np.asarray(rows, np.float32)


# recovery primitives: one 10-action chunk executed instead of the policy's chunk
KICKS = {
    "kick": _chunk((0, 0, 0.6, 0, 0, 0), -1.0),                 # open gripper, lift (the exp-02 winner)
    "kick_open": _chunk((0, 0, 0, 0, 0, 0), -1.0),              # just open and hold
    "kick_lift_closed": _chunk((0, 0, 0.6, 0, 0, 0), 1.0),      # lift keeping the gripper closed
    "kick_high": _chunk((0, 0, 1.0, 0, 0, 0), -1.0, n_move=8),  # bigger lift
    "kick_retreat": _chunk((-0.6, 0, 0.5, 0, 0, 0), -1.0),      # back and up, open
    "kick_wiggle": np.asarray([[0.5, 0, 0.3, 0, 0, 0, -1], [-0.5, 0, 0.3, 0, 0, 0, -1], [0, 0.5, 0.3, 0, 0, 0, -1], [0, -0.5, 0.3, 0, 0, 0, -1],
                               [0.5, 0, 0, 0, 0, 0, -1], [-0.5, 0, 0, 0, 0, 0, -1], [0, 0.5, 0, 0, 0, 0, -1], [0, -0.5, 0, 0, 0, 0, -1],
                               [0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, -1]], np.float32),
    "kick_random": None,                                        # random direction push (drawn per branch), open gripper
    "idle_hold": _chunk((0, 0, 0, 0, 0, 0), 1.0),               # do nothing, keep the gripper closed (wait)
    "idle_open": _chunk((0, 0, 0, 0, 0, 0), -1.0),              # do nothing, gripper open (same as kick_open)
}
STRATEGIES = ("native", "random", "boundary", "small_cluster", "big_cluster", "motion_max", "route_exit", "route_stay",
              "flow_straight", "flow_curved", "v82_min") + tuple(KICKS) + ("kick_then_motion", "kick_then_random", "kick_if_open", "kick_if_stuck",
              "gain2", "gain3", "partial6", "partial7", "partial8", "partial9", "obs_perturb", "as_swap_motion", "as_swap_random")
KICK_CHUNK = KICKS["kick"]


def perturb_observation(obs, k, rng):
    """K deterministic observation variants: image shifts, brightness, wrist shifts, small state noise."""
    img = np.asarray(obs["observation/image"]); wrist = np.asarray(obs["observation/wrist_image"]); state = np.asarray(obs["observation/state"], np.float32)
    out = {}
    if k == 0:
        return out
    variant = (k - 1) % 7
    if variant == 0:
        out["observation/image"] = np.roll(img, 8, axis=1)
    elif variant == 1:
        out["observation/image"] = np.roll(img, -8, axis=1)
    elif variant == 2:
        out["observation/image"] = np.roll(img, 8, axis=0)
    elif variant == 3:
        out["observation/image"] = np.clip(img.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
    elif variant == 4:
        out["observation/image"] = np.clip(img.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    elif variant == 5:
        out["observation/wrist_image"] = np.roll(wrist, 10, axis=1)
    else:
        out["observation/state"] = (state + rng.normal(0, 0.02, size=state.shape)).astype(np.float32)
    return out


def route_points(probs):
    """Hellinger embedding of [N, 8, 10, 11, 32] routing probabilities (back layers, action tokens)."""
    p = np.asarray(probs, np.float64)[:, 4:, :, 1:]
    p = p / p.sum(-1, keepdims=True)
    cells = int(np.prod(p.shape[1:-1]))
    return np.sqrt(p).reshape(len(p), -1) / np.sqrt(2 * cells)


class Selector:
    def __init__(self, strategy, parent_probs, rng):
        self.strategy = strategy
        self.rng = rng
        self.reference = route_points(parent_probs[-8:]) if parent_probs is not None and len(parent_probs) else None
        self.monitor = None
        self.steps = []
        self._last_pt = None
        self.inputs = []
        self.scales = None
        if parent_probs is not None:
            for p in parent_probs:
                self.commit_route_only(p)
        if strategy == "v82_min" or True:
            from v82_closed_loop import V82Monitor
            self.monitor = V82Monitor()
            for p in (parent_probs if parent_probs is not None else []):
                self.monitor.update(np.asarray(p, np.float32))

    def choose(self, actions, probs=None, flow=None):
        K = len(actions)
        s = self.strategy
        info = {}
        if s in ("native", "random") or s in KICKS or s in ("kick_then_random", "kick_if_open", "kick_if_stuck") or s.startswith(("gain", "partial")) or s == "as_swap_random":
            return 0, info
        if s in ("obs_perturb", "as_swap_motion"):
            s = "motion_max"
        if s == "kick_then_motion":
            s = "motion_max"
        flat = actions.reshape(K, -1)
        std = flat.std(0) + 1e-6
        z = (flat - flat.mean(0)) / std
        if s == "boundary":
            d = np.linalg.norm(z, axis=1)
            info["dist_to_mean"] = d.tolist()
            return int(np.argmax(d)), info
        if s in ("small_cluster", "big_cluster"):
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import pdist, squareform
            D = squareform(pdist(z))
            labels = fcluster(linkage(pdist(z), method="average"), t=min(3, K), criterion="maxclust")
            sizes = {int(l): int((labels == l).sum()) for l in set(labels)}
            target = min(sizes, key=lambda l: (sizes[l], l)) if s == "small_cluster" else max(sizes, key=lambda l: (sizes[l], -l))
            members = np.flatnonzero(labels == target)
            medoid = members[np.argmin(D[np.ix_(members, members)].sum(1))]
            info.update(cluster_sizes=sizes, chosen_cluster=int(target))
            return int(medoid), info
        if s == "motion_max":
            disp = np.abs(actions[:, :, :3]).sum(axis=(1, 2))
            info["displacement"] = disp.tolist()
            return int(np.argmax(disp)), info
        if s in ("route_exit", "route_stay"):
            if probs is None or self.reference is None:
                return 0, {"fallback": "no routing"}
            pts = route_points(probs)
            d = np.sqrt(((pts[:, None, :] - self.reference[None, :, :]) ** 2).sum(-1)).min(1)
            info["min_dist_to_reference"] = d.tolist()
            return int(np.argmax(d) if s == "route_exit" else np.argmin(d)), info
        if s in ("flow_straight", "flow_curved"):
            if flow is None:
                return 0, {"fallback": "no flow path"}
            f = flow.reshape(K, flow.shape[1], -1)                 # [K, 11, 240]
            path = np.linalg.norm(np.diff(f, axis=1), axis=2).sum(1)
            net = np.linalg.norm(f[:, -1] - f[:, 0], axis=1)
            ratio = net / np.maximum(path, 1e-9)
            info["straightness"] = ratio.tolist()
            return int(np.argmax(ratio) if s == "flow_straight" else np.argmin(ratio)), info
        if s == "v82_min":
            if probs is None or self.monitor is None:
                return 0, {"fallback": "no routing"}
            scores = []
            for k in range(K):
                m = copy.deepcopy(self.monitor)
                st = m.update(np.asarray(probs[k], np.float32))
                scores.append(float(st.get("freeze_score") if st.get("freeze_score") is not None else np.nan))
            info["freeze_scores"] = scores
            scores = np.where(np.isfinite(scores), scores, np.inf)
            return int(np.argmin(scores)), info
        raise ValueError(s)

    def commit(self, probs_k):
        if probs_k is not None:
            p = np.asarray(probs_k, np.float64)[4:, :, :1]
            p = p / p.sum(-1, keepdims=True)
            pt = np.sqrt(p).reshape(-1) / np.sqrt(2 * np.prod(p.shape[:-1]))
            if getattr(self, "_last_pt", None) is not None:
                self.steps.append(float(np.linalg.norm(pt - self._last_pt)))
            self._last_pt = pt
        if self.monitor is not None and probs_k is not None:
            st = self.monitor.update(np.asarray(probs_k, np.float32))
            return bool(st.get("v82_alarm", st.get("alarm")))
        return None

    def commit_route_only(self, probs_k):
        p = np.asarray(probs_k, np.float64)[4:, :, :1]
        p = p / p.sum(-1, keepdims=True)
        pt = np.sqrt(p).reshape(-1) / np.sqrt(2 * np.prod(p.shape[:-1]))
        if self._last_pt is not None:
            self.steps.append(float(np.linalg.norm(pt - self._last_pt)))
        self._last_pt = pt

    def commit_input(self, x):
        """x: [4, 10, 1024] MoE input state of the executed query; window diameter with prefix-calibrated per-layer scales."""
        self.inputs.append(np.asarray(x, np.float32))
        if len(self.inputs) == 12:
            X = np.stack(self.inputs)
            self.scales = np.maximum(np.sqrt((X ** 2).mean(axis=(0, 2, 3))), 1e-12)

    def diameter(self, window=14):
        if getattr(self, "scales", None) is None or len(self.inputs) < 12:
            return None
        X = np.stack(self.inputs[-window:]) / self.scales[None, :, None, None]
        Xn = X.reshape(len(X), 4, -1)
        d = np.sqrt(((Xn[:, None] - Xn[None]) ** 2).mean(-1).mean(-1))
        return float(d.max())

    def load_bank(self, path, own_tag):
        z = np.load(path)
        keep = z["owner"] != own_tag          # leave the parent's own routing states out of the reference
        self.bank = z["bank"][keep].astype(np.float32)
        self.bank_sq = (self.bank ** 2).sum(1)
        self.knn_hist = []

    def knn_distance(self, probs_k, k=20):
        if getattr(self, "bank", None) is None:
            return None
        p = np.asarray(probs_k, np.float64)[4:, :, 1:]
        p = p / p.sum(-1, keepdims=True)
        x = (np.sqrt(p).reshape(-1) / np.sqrt(2 * np.prod(p.shape[:-1]))).astype(np.float32)
        d = np.sqrt(np.maximum((x ** 2).sum() + self.bank_sq - 2 * self.bank @ x, 0))
        val = float(np.sort(d)[:k].mean())
        self.knn_hist.append(val)
        return val

    def route_step_mean(self, window=13):
        return float(np.mean(self.steps[-window:])) if len(self.steps) >= 6 else None


def load_parent_probs(tag, capture_root, upto):
    files = sorted(glob.glob("%s/server-p*/%s/q*.npz" % (capture_root, tag)), key=lambda p: int(p.split("/q")[-1][:3]))[:upto]
    out = []
    for f in files:
        with np.load(f) as z:
            out.append(z["hb_router_probs"].astype(np.float32))
    return np.stack(out) if out else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True, help="episode artifact directory (summary.json + episode-trace.npz)")
    ap.add_argument("--tag", required=True, help="capture tag of the parent (for routing history)")
    ap.add_argument("--capture-root", default=str(ROOT.parent / "moe-capture" / "topo-20260916c"))
    ap.add_argument("--branch-query", type=int, required=True)
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--noise-scale", type=float, default=1.0, help="std of the candidate flow noises (1 = the policy's own prior)")
    ap.add_argument("--control-queries", type=int, default=1)
    ap.add_argument("--route-step-threshold", type=float, default=0.058, help="online early trigger: windowed mean routing step (state token, back layers) below this from q>=12")
    ap.add_argument("--diam-threshold", type=float, default=1.20, help="online early trigger: 14-query window diameter of the MoE input state (back layers, last flow) below this from q>=12")
    ap.add_argument("--knn-threshold", type=float, default=0.0447, help="online kNN trigger: mean distance to the 20 nearest success-reference routing states, 2 consecutive queries above")
    ap.add_argument("--knn-bank", default=str(ROOT / "controls" / "knn-bank-success.npz"))
    ap.add_argument("--stuck-threshold", type=float, default=2.0, help="kick_if_stuck: kick only when the policy's own chunk |xyz| displacement is below this")
    ap.add_argument("--retrigger-cooldown", type=int, default=0, help="if >0, re-arm the trigger this many queries after control ended (repeated interventions)")
    ap.add_argument("--trigger", default="branch", choices=("branch", "v82", "route_step", "either", "diam", "diam_or_v82", "knn", "knn_or_v82"),
                    help="branch: control starts at --branch-query (hindsight); v82: run from --branch-query and start control when the online v8.2 alarm latches")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parent = pathlib.Path(args.parent)
    summary = json.loads((parent / "summary.json").read_text())
    trace = np.load(parent / "episode-trace.npz")
    variant = summary["benchmark_variant"]
    source = pathlib.Path(summary["benchmark_source"])
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "configs" / ("libero-" + variant))
    sys.path.insert(0, str(source))
    from himoe_libero_bridge.client import PolicyClient
    from himoe_libero_bridge.libero_runtime import (EpisodeConfig, LIBERO_DUMMY_ACTION, _load_task, _sim_state,
                                                    build_policy_observation)
    from himoe_libero_bridge.protocol import unpackb
    from libero.libero import benchmark  # noqa: F401
    if variant == "plus":
        import fast_perturbations
        fast_perturbations.install()

    config = EpisodeConfig(libero_root=str(source), output_root=args.out, port=args.port, task_suite=summary["task_suite"],
                           task_id=int(summary["task_id"]), init_state_id=int(summary["init_state_id"]), seed=int(summary["seed"]),
                           max_steps=args.max_steps, render_size=256, inference_timeout=180.0)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = dict(parent=str(parent), tag=args.tag, strategy=args.strategy, branch_query=args.branch_query, candidates=args.candidates,
                  control_queries=args.control_queries, seed=args.seed, noise_scale=args.noise_scale, trigger=args.trigger, parent_success=summary["success"],
                  parent_steps=summary["action_steps"], status="failed", success=False, action_steps=0, decisions=[])
    started = time.perf_counter()
    environment = None
    client = None
    try:
        environment, observation, task, prompt = _load_task(config)
        for _ in range(config.settle_steps):
            observation, *_ = environment.step(LIBERO_DUMMY_ACTION.tolist())
        # replay the parent's executed actions up to the branch query
        steps = 0
        for q in range(args.branch_query):
            for action in trace["predicted_actions"][q][: int(trace["executed_lengths"][q])]:
                observation, reward, done, info = environment.step(action.tolist())
                steps += 1
        expected = trace["sim_states_before"][args.branch_query]
        actual = _sim_state(environment)
        result["replay_state_max_abs_diff"] = float(np.abs(actual - expected).max())
        result["replay_steps"] = steps
        result["action_steps"] = steps
        client = PolicyClient(host="127.0.0.1", port=args.port, connect_timeout=120.0, inference_timeout=180.0)
        parent_probs = load_parent_probs(args.tag, args.capture_root, args.branch_query)
        selector = Selector(args.strategy, parent_probs, np.random.default_rng([args.seed, args.branch_query]))
        if args.trigger in ("knn", "knn_or_v82"):
            selector.load_bank(args.knn_bank, args.tag)
        rng = np.random.default_rng([args.seed, args.branch_query, 7])
        query = args.branch_query
        controlled = 0
        alarm_latched = True
        triggered = False
        rearm_at = 0
        deferrals = 0
        success = False
        while steps < args.max_steps and not success:
            policy_observation = build_policy_observation(observation, prompt)
            request = dict(policy_observation)
            request["episode_id"] = -1
            request["capture/tag"] = "branch/%s/%s/q%02d/s%d" % (args.tag, args.strategy, args.branch_query, args.seed)
            request["capture/step"] = query
            if args.retrigger_cooldown > 0 and triggered and controlled >= max(1, args.control_queries) and query >= rearm_at:
                triggered = False   # allow another intervention later
                controlled = 0
                result.setdefault("retriggers", 0)
            armed = args.trigger == "branch" or triggered
            controlling = armed and args.strategy not in ("native",) and (
                (args.control_queries >= 0 and controlled < args.control_queries) or (args.control_queries < 0 and alarm_latched and controlled < 12))
            decision = dict(query=query, step=steps, controlling=bool(controlling), armed=bool(armed))
            kick_now = controlling and (args.strategy in KICKS or (args.strategy.startswith("kick_then") and controlled == 0))
            deferred = False
            if controlling and args.strategy in ("kick_if_open", "kick_if_stuck"):
                probe_req = dict(request)
                probe_req["candidates/noises"] = rng.standard_normal((1, 10, 24)).astype(np.float32)
                probe_req["capture/return_probs"] = True
                if args.trigger in ("diam", "diam_or_v82"):
                    probe_req["capture/return_input"] = True
                client._connection.send(client._packer.pack(probe_req))
                raw = client._connection.recv(timeout=180.0)
                probe_resp = unpackb(raw) if not isinstance(raw, str) else {}
                own = np.asarray(probe_resp["candidates/actions"], np.float32)[0]
                gripper_open = float(own[:, 6].mean()) < 0
                displacement = float(np.abs(own[:, :3]).sum())
                decision["gripper_open"] = bool(gripper_open)
                decision["own_displacement"] = displacement
                condition = gripper_open if args.strategy == "kick_if_open" else displacement < args.stuck_threshold
                if condition:
                    kick_now = True
                    actions = KICKS["kick"]
                    decision["chosen"] = "kick"
                    probs_k = np.asarray(probe_resp["candidates/hb_probs"], np.float32)[0] if "candidates/hb_probs" in probe_resp else None
                    if "candidates/input_back_last" in probe_resp:
                        selector.commit_input(np.asarray(probe_resp["candidates/input_back_last"], np.float32)[0])
                else:
                    deferred = deferrals < 6
                    deferrals += 1 if deferred else 0
                    controlling = False if deferred else controlling
                    actions = own
                    decision["chosen"] = "deferred" if deferred else "gave_up"
                    probs_k = np.asarray(probe_resp["candidates/hb_probs"], np.float32)[0] if "candidates/hb_probs" in probe_resp else None
                    if "candidates/input_back_last" in probe_resp:
                        selector.commit_input(np.asarray(probe_resp["candidates/input_back_last"], np.float32)[0])
                    if not deferred:
                        controlled += 1   # consume the intervention without kicking
            if args.strategy in ("kick_if_open", "kick_if_stuck") and (kick_now or deferred or decision.get("chosen") == "gave_up"):
                pass
            elif kick_now:
                if args.strategy == "kick_random" or (args.strategy.startswith("kick_then") and False):
                    direction = rng.standard_normal(3)
                    direction = 0.7 * direction / np.linalg.norm(direction)
                    actions = _chunk(tuple(direction) + (0, 0, 0), -1.0)
                else:
                    actions = KICKS["kick"] if args.strategy.startswith("kick_then") else KICKS[args.strategy]
                decision["chosen"] = "kick"
                probs_k = None
                if args.trigger != "branch" or args.control_queries < 0:
                    # still need this query's routing for the monitor: ask for one native candidate without executing it
                    request["candidates/noises"] = rng.standard_normal((1, 10, 24)).astype(np.float32)
                    request["capture/return_probs"] = True
                    if args.trigger in ("diam", "diam_or_v82"):
                        request["capture/return_input"] = True
                    client._connection.send(client._packer.pack(request))
                    raw = client._connection.recv(timeout=180.0)
                    response = unpackb(raw) if not isinstance(raw, str) else {}
                    if "candidates/hb_probs" in response:
                        probs_k = np.asarray(response["candidates/hb_probs"], np.float32)[0]
                    if "candidates/input_back_last" in response:
                        selector.commit_input(np.asarray(response["candidates/input_back_last"], np.float32)[0])
            elif args.strategy in ("kick_if_open", "kick_if_stuck") and decision.get("chosen") in ("kick", "deferred", "gave_up"):
                pass
            else:
                if args.strategy == "native" and query < len(trace["flow_noises"]):
                    noises = trace["flow_noises"][query][None]
                elif controlling and args.strategy.startswith(("gain", "partial")):
                    noises = rng.standard_normal((1, 10, 24)).astype(np.float32)          # the policy's own draw
                elif controlling and args.strategy.startswith("as_swap"):
                    noise1 = rng.standard_normal((1, 10, 24)).astype(np.float32)
                    noises = np.repeat(noise1, 4, axis=0)                                 # native + 3 forced AS experts, same noise
                    request["candidates/as_experts"] = np.array([-1, 0, 1, 2], np.int16)
                elif controlling and args.strategy not in ("random", "kick_then_random", "obs_perturb"):
                    noises = (args.noise_scale * rng.standard_normal((args.candidates, 10, 24))).astype(np.float32)
                elif controlling:
                    noises = (args.noise_scale * rng.standard_normal((1, 10, 24))).astype(np.float32)
                else:
                    noises = rng.standard_normal((1, 10, 24)).astype(np.float32)
                request["candidates/noises"] = noises
                if args.strategy.startswith("partial"):
                    request["capture/return_flow"] = True
                if args.strategy in ROUTE_STRATEGIES or args.control_queries < 0 or args.trigger != "branch":
                    request["capture/return_probs"] = True
                if args.trigger in ("diam", "diam_or_v82"):
                    request["capture/return_input"] = True
                if args.strategy in FLOW_STRATEGIES:
                    request["capture/return_flow"] = True
                if controlling and args.strategy == "obs_perturb":
                    merged = None
                    for k in range(args.candidates):
                        req_k = dict(request)
                        req_k.update(perturb_observation(policy_observation, k, rng))
                        client._connection.send(client._packer.pack(req_k))
                        raw = client._connection.recv(timeout=180.0)
                        if isinstance(raw, str):
                            raise RuntimeError("server error: " + raw[:500])
                        r_k = unpackb(raw)
                        if merged is None:
                            merged = {key: [] for key in r_k if key.startswith("candidates/")}
                        for key in merged:
                            merged[key].append(np.asarray(r_k[key])[0])
                    response = {key: np.stack(v) for key, v in merged.items()}
                else:
                    client._connection.send(client._packer.pack(request))
                    raw = client._connection.recv(timeout=180.0)
                    if isinstance(raw, str):
                        raise RuntimeError("server error: " + raw[:500])
                    response = unpackb(raw)
                cands = np.array(response["candidates/actions"], dtype=np.float32, copy=True)
                if controlling and args.strategy.startswith("gain"):
                    g = float(args.strategy[4:])
                    scaled = cands.copy()
                    scaled[:, :, :6] = np.clip(scaled[:, :, :6] * g, -1.0, 1.0)
                    decision["gain"] = g
                    cands = scaled
                if controlling and args.strategy.startswith("partial"):
                    step = int(args.strategy[7:])
                    cands = np.array(response["candidates/flow_path_actions"], dtype=np.float32, copy=True)[:, step]
                    cands[:, :, :6] = np.clip(cands[:, :, :6], -1.0, 1.0)
                    decision["denoise_step"] = step
                if controlling and args.strategy == "as_swap_random":
                    ids = np.asarray(response.get("candidates/as_expert_ids", [[0]]))
                    native_e = int(np.asarray(ids[0]).reshape(-1)[0])
                    choices = [e for e in (0, 1, 2) if e != native_e]
                    k_forced = 1 + int(rng.choice(choices))
                    decision["native_as_expert"] = native_e
                    decision["forced_as_expert"] = k_forced - 1
                    cands = cands[[k_forced]]
                if controlling and args.strategy == "as_swap_motion":
                    ids = np.asarray(response.get("candidates/as_expert_ids", [[0]]))
                    native_e = int(np.asarray(ids[0]).reshape(-1)[0])
                    decision["native_as_expert"] = native_e
                    decision["forced_displacements"] = np.abs(cands[1:, :, :3]).sum(axis=(1, 2)).tolist()
                    decision["native_displacement"] = float(np.abs(cands[0, :, :3]).sum())
                    cands = cands[[1 + e for e in (0, 1, 2) if e != native_e]]
                probs = np.asarray(response["candidates/hb_probs"], np.float32) if "candidates/hb_probs" in response else None
                flow = np.asarray(response["candidates/flow_path"], np.float32) if "candidates/flow_path" in response else None
                if controlling and len(cands) > 1:
                    k, info = selector.choose(cands, probs, flow)
                    decision.update(chosen=int(k), **info)
                else:
                    k = 0
                actions = cands[k]
                probs_k = probs[k] if probs is not None else None
                if "candidates/input_back_last" in response:
                    selector.commit_input(np.asarray(response["candidates/input_back_last"], np.float32)[k])
            if controlling and not (args.strategy in ("kick_if_open", "kick_if_stuck") and decision.get("chosen") == "gave_up"):
                controlled += 1
            latched = selector.commit(probs_k)
            if latched is not None:
                alarm_latched = latched
                decision["v82_alarm"] = latched
                rs = selector.route_step_mean()
                early = rs is not None and query >= 12 and rs < args.route_step_threshold
                decision["route_step_mean"] = rs
                dm = selector.diameter()
                decision["diameter"] = dm
                low = dm is not None and query >= 12 and dm < args.diam_threshold
                kd = selector.knn_distance(probs_k) if args.trigger in ("knn", "knn_or_v82") and probs_k is not None else None
                decision["knn_distance"] = kd
                knn_fire = kd is not None and query >= 6 and len(selector.knn_hist) >= 2 and all(v > args.knn_threshold for v in selector.knn_hist[-2:])
                fire = ((args.trigger == "v82" and latched) or (args.trigger == "route_step" and early) or (args.trigger == "either" and (latched or early))
                        or (args.trigger == "diam" and low) or (args.trigger == "diam_or_v82" and (low or latched))
                        or (args.trigger == "knn" and knn_fire) or (args.trigger == "knn_or_v82" and (knn_fire or latched)))
                if fire and not triggered and query >= rearm_at:
                    triggered = True
                    result.setdefault("trigger_query", query)
                    result["retriggers"] = result.get("retriggers", -1) + 1
                    rearm_at = query + max(1, args.control_queries) + args.retrigger_cooldown
                    decision["triggered"] = True
            for action in actions:
                if steps >= args.max_steps:
                    break
                observation, reward, done, info = environment.step(action.tolist())
                steps += 1
                success = bool(environment.check_success())
                if success:
                    break
            decision["success_after"] = bool(success)
            result["decisions"].append(decision)
            query += 1
        result.update(status="completed", success=bool(success), action_steps=int(steps), controlled_queries=int(controlled))
    except BaseException as error:  # noqa: BLE001
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["seconds"] = round(time.perf_counter() - started, 1)
        for resource in (client, environment):
            try:
                if resource is not None:
                    resource.close()
            except BaseException:  # noqa: BLE001
                pass
        (out / "result.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ("strategy", "branch_query", "success", "action_steps", "replay_state_max_abs_diff", "seconds") if k in result}), flush=True)


if __name__ == "__main__":
    main()
