"""[v3: adds reprompt / noise2 / retrace arms] Closed-loop control experiment client: recurrence-triggered interventions against the HiMoE API servers.

Mirrors himoe_libero_bridge.libero_runtime.run_episode (same flow-noise RNG, same request keys) so the
``native`` arm reproduces the recorded topology runs bit for bit.  The trigger is the online recurrence
rate of the wire routing (top-4 expert ids/weights, back-path scope), calibrated offline
(samples/moe-recurrence-20260916/wire-route-calibration.json).  Arms:
  native            no intervention
  withdraw          on trigger: lift 4 cm, retreat up to 4 cm toward the previous position, hand back
  escalate          trigger 1/2/3: withdraw4 / side_plus / side_minus (escalates only if RR re-triggers)
  resample_escape   on trigger: for 6 queries, 8 parallel candidates (different flow noise) across the
                    servers; execute the candidate whose routing point is farthest from the recent recurrent set
  resample_random   same requests, random candidate (cost-matched control)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAL = ROOT / "samples/moe-recurrence-20260916/wire-route-calibration.json"
W, THEILER = 14, 3
ARMS = ("native", "withdraw", "escalate", "resample_escape", "resample_random", "hold16", "withdraw8", "persist", "withdraw_early", "reprompt", "noise2", "retrace")
REPROMPTS = ("please {p}", "{p} now", "carefully {p}")
NOISE_GAIN = 2.0
PERSIST_OPS = ("withdraw4", "side_plus", "side_minus", "withdraw8", "grip_cycle", "lift8")
COOLDOWN_QUERIES, MAX_INTERVENTIONS, RESAMPLE_QUERIES, CANDIDATES = 6, 3, 6, 8
MAX_INTERVENTIONS_V2 = 6
PHYS_STEP_M, PHYS_STEPS_MAX = 0.03, 10


def bootstrap(benchmark):
    source = ROOT / "upstream" / ("LIBERO-plus" if benchmark == "plus" else "LIBERO-PRO")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT.parent / "srv" / "src"))
    sys.path.insert(0, str(ROOT.parent / "srv" / "packages" / "openpi-client" / "src"))
    sys.path.insert(0, str(source))
    if benchmark == "plus":
        sys.path.insert(0, str(ROOT / "dependencies" / "libero-plus"))
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "configs" / ("libero-" + benchmark))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    return source


def route_point(ids, weights):
    """wire arrays (10 flow, 8 layer, 10 token, 4) -> back-path sqrt-prob vector, identical to the offline calibration."""
    import numpy as np
    ids = np.asarray(ids).transpose(1, 0, 2, 3)[4:]            # (4 layers, 10 flow, 10 tok, 4)
    w = np.asarray(weights, np.float64).transpose(1, 0, 2, 3)[4:]
    w = w / np.maximum(w.sum(-1, keepdims=True), 1e-12)
    dense = np.zeros(ids.shape[:-1] + (32,))
    np.put_along_axis(dense, ids.astype(np.int64), w, axis=-1)
    cells = int(np.prod(dense.shape[:-1]))
    return np.sqrt(dense).reshape(-1) / np.sqrt(2 * cells)


class RecurrenceDetector:
    def __init__(self, eps, theta, confirm):
        import numpy as np
        self.np = np
        self.eps, self.theta, self.confirm = eps, theta, confirm
        self.points, self.rr, self.diam, self.run = [], [], [], 0

    def push(self, point):
        np = self.np
        self.points.append(point)
        n = len(self.points)
        if n < W:
            self.rr.append(float("nan")); self.diam.append(float("nan"))
            return float("nan")
        P = np.stack(self.points[-W:])
        S = np.sqrt(np.maximum(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1), 0.0))
        J = np.arange(W)
        m = (J[None, :] - J[:, None]) >= THEILER
        rr = float((S[m] <= self.eps).mean())
        self.rr.append(rr); self.diam.append(float(S.max()))
        self.run = self.run + 1 if rr >= self.theta else 0
        return rr

    def triggered(self):
        return self.run >= self.confirm

    def reset_confirmation(self):
        self.run = 0

    def escape_distance(self, point):
        np = self.np
        if not self.points:
            return float("inf")
        P = np.stack(self.points[-W:])
        return float(np.sqrt(((P - point[None, :]) ** 2).sum(-1)).min())


def bounded(v, radius):
    import numpy as np
    v = np.asarray(v, float)
    return v * min(1.0, radius / max(float(np.linalg.norm(v)), 1e-12))


def waypoints(operator, initial, history):
    """returns (targets, gripper commands per target or None) ; gripper None = keep last command"""
    import numpy as np
    initial = np.asarray(initial, float)
    history = np.asarray(history, float)
    lift = initial + np.array([0.0, 0.0, 0.04])
    if operator == "withdraw4":
        xy = bounded(history[-1, :2] - initial[:2], 0.04)
        return [lift, lift + np.r_[xy, 0.0]]
    if operator == "withdraw8":
        lift8 = initial + np.array([0.0, 0.0, 0.08])
        xy = bounded(history[0, :2] - initial[:2], 0.08)
        return [lift8, lift8 + np.r_[xy, 0.0]]
    if operator == "lift8":
        return [initial + np.array([0.0, 0.0, 0.08])]
    if operator == "retrace":
        back = initial + bounded(history[0] - initial, 0.08)
        return [lift, np.r_[back[:2], lift[2]], np.r_[back[:2], max(back[2], initial[2])]]
    if operator in ("hold16", "grip_cycle"):
        return [initial]
    direction = initial[:2] - history[-1, :2]
    perp = np.array([-direction[1], direction[0]])
    if np.linalg.norm(perp) < 1e-5:
        perp = np.array([1.0, 0.0])
    perp = perp / np.linalg.norm(perp)
    detour = lift + np.r_[perp * (0.04 if operator == "side_plus" else -0.04), 0.0]
    return [lift, detour, lift]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", choices=("plus", "pro"), required=True)
    ap.add_argument("--perturbation", default="none")
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--task-name")
    ap.add_argument("--init-state-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--flow-noise-seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--render-size", type=int, default=256)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", default="9510", help="comma separated; first = primary, all used for candidates")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--theta", type=float, default=0.4)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--candidate-seed", type=int, default=20260916)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare-trace", type=Path, default=None, help="episode-trace.npz of the native run for bitwise checks")
    args = ap.parse_args()

    source = bootstrap(args.benchmark)
    import numpy as np
    from himoe_libero_bridge.client import PolicyClient
    from himoe_libero_bridge.libero_runtime import EpisodeConfig, LIBERO_DUMMY_ACTION, _load_task, _sim_state
    from himoe_libero_bridge.preprocess import build_policy_observation
    from himoe_libero_bridge import protocol as P
    from libero.libero import benchmark as lb
    if args.benchmark == "plus" and os.environ.get("HIMOE_FAST_PERTURB", "1") != "0":
        import fast_perturbations
        fast_perturbations.install()

    FLOW_NOISE_KEY = getattr(P, "FLOW_NOISE_KEY", "flow/noise")
    CAPTURE_KEY = getattr(P, "ROUTING_CAPTURE_KEY", "routing/capture")
    IDS_KEY = getattr(P, "ROUTING_EXPERT_IDS_KEY", "routing/expert_ids")
    WTS_KEY = getattr(P, "ROUTING_EXPERT_WEIGHTS_KEY", "routing/expert_weights")
    ACTION_KEY = getattr(P, "ACTION_KEY", "actions")
    NOISE_SHAPE = tuple(getattr(P, "FLOW_NOISE_SHAPE", (10, 24)))
    cal = json.loads(CAL.read_text())
    eps = float(cal["eps"])

    base_suite = "libero_10"
    task_suite = base_suite if (args.benchmark == "plus" or args.perturbation == "none") else base_suite + "_" + args.perturbation
    suite = lb.get_benchmark_dict()[task_suite]()
    if args.task_name is not None:
        matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == args.task_name]
        if len(matches) != 1:
            raise SystemExit("task name matches %d tasks" % len(matches))
        task_id = matches[0]
    else:
        task_id = args.task_id or 0
    config = EpisodeConfig(libero_root=str(source), output_root=str(args.out), host=args.host, port=int(args.ports.split(",")[0]),
                           task_suite=task_suite, task_id=task_id, init_state_id=args.init_state_id, seed=args.seed,
                           max_steps=args.max_steps, render_size=args.render_size, inference_timeout=120, flow_noise_seed=args.flow_noise_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    ports = [int(p) for p in args.ports.split(",")]
    def connect(port):
        last = None
        for attempt in range(8):
            try:
                return PolicyClient(args.host, port, connect_timeout=120.0, inference_timeout=300.0)
            except Exception as error:   # busy server: metadata handshake can time out under load
                last = error
                time.sleep(5 + 5 * attempt)
        raise RuntimeError("could not connect to port %d: %s" % (port, last))

    class LazyClients:
        """primary connection at start; candidate connections only when a resample window needs them"""
        def __init__(self):
            self._c = {0: connect(ports[0])}
        def __getitem__(self, i):
            if i not in self._c:
                self._c[i] = connect(ports[i])
            return self._c[i]
        def __len__(self):
            return len(ports)
        def __iter__(self):
            return iter(list(self._c.values()))

    clients = LazyClients()
    primary = clients[0]
    compare = np.load(args.compare_trace) if args.compare_trace else None

    if args.arm == "withdraw_early":
        args.theta = min(args.theta, 0.3)
    detector = RecurrenceDetector(eps, args.theta, args.confirm)
    trace_rng = np.random.default_rng(args.flow_noise_seed)
    cand_rng = np.random.default_rng(np.random.SeedSequence([args.candidate_seed, task_id, args.init_state_id, ARMS.index(args.arm)]))
    summary = dict(status="failed", success=False, arm=args.arm, benchmark=args.benchmark, perturbation=args.perturbation,
                   task_suite=task_suite, task_id=task_id, init_state_id=args.init_state_id, seed=args.seed,
                   flow_noise_seed=args.flow_noise_seed, theta=args.theta, confirm=args.confirm, eps=eps,
                   action_steps=0, inference_calls=0, candidate_calls=0, physical_steps=0, interventions=[],
                   first_trigger_query=None, bitwise_matches=0, bitwise_checked=0, ports=ports)
    per_query = []   # (query index, step, rr, diam, chosen candidate, escape distance, mode)
    states, actions_log = [], []
    started = time.perf_counter()
    environment = None
    try:
        environment, observation, task, prompt = _load_task(config)
        summary.update(task_name=str(task.name), prompt=prompt)
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())
        try:
            output_scale = np.asarray(environment.env.robots[0].controller.output_max, float)[:3]
        except Exception:
            output_scale = np.array([0.05, 0.05, 0.05])
        summary["output_scale"] = output_scale.tolist()

        def step_env(action):
            nonlocal observation
            observation, reward, done, _ = environment.step(np.asarray(action, np.float32).tolist())
            summary["action_steps"] += 1
            summary["success"] = bool(environment.check_success())
            states.append(np.asarray(observation["robot0_eef_pos"], float).copy())
            actions_log.append(np.asarray(action, np.float32).copy())
            return summary["success"] or summary["action_steps"] >= config.max_steps

        def infer(client, obs, noise):
            request = dict(obs)
            request[FLOW_NOISE_KEY] = noise
            request[CAPTURE_KEY] = True
            response = client.infer(request)
            acts = np.asarray(response[ACTION_KEY], np.float32)
            point = route_point(response[IDS_KEY], response[WTS_KEY])
            return acts, point

        boundary_positions = []
        last_gripper = -1.0
        interventions_left = MAX_INTERVENTIONS_V2 if args.arm in ("hold16", "withdraw8", "persist", "withdraw_early", "reprompt", "noise2", "retrace") else MAX_INTERVENTIONS
        cooldown = 0
        resample_left = 0
        api_left = 0
        query = 0
        mode = "policy"
        while summary["action_steps"] < config.max_steps and not summary["success"]:
            active_prompt = prompt
            if api_left > 0 and args.arm == "reprompt":
                active_prompt = REPROMPTS[(len(summary["interventions"]) - 1) % len(REPROMPTS)].format(p=prompt)
            obs = build_policy_observation(observation, active_prompt)
            boundary_positions.append(np.asarray(observation["robot0_eef_pos"], float).copy())
            native_noise = trace_rng.standard_normal(NOISE_SHAPE).astype(np.float32)
            if api_left > 0 and args.arm == "noise2":
                native_noise = (native_noise * NOISE_GAIN).astype(np.float32)
            if api_left > 0:
                api_left -= 1
                mode_override = args.arm
            else:
                mode_override = None
            chosen, escape = 0, float("nan")
            if resample_left > 0 and args.arm in ("resample_escape", "resample_random"):
                noises = [native_noise] + [cand_rng.standard_normal(NOISE_SHAPE).astype(np.float32) for _ in range(CANDIDATES - 1)]
                with ThreadPoolExecutor(max_workers=CANDIDATES) as pool:
                    results = list(pool.map(lambda i: infer(clients[i % len(clients)], obs, noises[i]), range(CANDIDATES)))
                summary["candidate_calls"] += CANDIDATES
                dists = [detector.escape_distance(pt) for _, pt in results]
                chosen = int(np.argmax(dists)) if args.arm == "resample_escape" else int(cand_rng.integers(CANDIDATES))
                escape = dists[chosen]
                acts, point = results[chosen]
                resample_left -= 1
                mode = "resample"
            else:
                acts, point = infer(primary, obs, native_noise)
                mode = mode_override or "policy"
            summary["inference_calls"] += 1
            if compare is not None and query < compare["predicted_actions"].shape[0] and mode == "policy" and not summary["interventions"]:
                summary["bitwise_checked"] += 1
                summary["bitwise_matches"] += int(np.array_equal(acts, compare["predicted_actions"][query]))
            rr = detector.push(point)
            per_query.append((query, summary["action_steps"], rr, detector.diam[-1], chosen, escape, mode))
            query += 1
            # execute the chosen chunk
            stop = False
            for a in acts[: config.replan_steps]:
                last_gripper = 1.0 if float(a[6]) > 0 else -1.0
                if step_env(a):
                    stop = True
                    break
            if stop:
                break
            if cooldown > 0:
                cooldown -= 1
            # trigger logic
            if args.arm != "native" and detector.triggered() and cooldown == 0 and interventions_left > 0 and query >= W:
                if summary["first_trigger_query"] is None:
                    summary["first_trigger_query"] = query - 1
                n_done = (MAX_INTERVENTIONS_V2 if args.arm in ("hold16", "withdraw8", "persist", "withdraw_early", "reprompt", "noise2", "retrace") else MAX_INTERVENTIONS) - interventions_left
                record = dict(query=query - 1, step=summary["action_steps"], rr=rr, index=n_done)
                if args.arm in ("reprompt", "noise2"):
                    record.update(operator=args.arm)
                    api_left = RESAMPLE_QUERIES
                    summary["interventions"].append(record)
                    interventions_left -= 1
                    cooldown = COOLDOWN_QUERIES + RESAMPLE_QUERIES
                    detector.reset_confirmation()
                    continue
                if args.arm in ("withdraw", "escalate", "hold16", "withdraw8", "persist", "withdraw_early", "retrace"):
                    operator = {"withdraw": "withdraw4", "withdraw_early": "withdraw4", "hold16": "hold16", "withdraw8": "withdraw8", "retrace": "retrace"}.get(args.arm)
                    if args.arm == "escalate":
                        operator = ("withdraw4", "side_plus", "side_minus")[n_done]
                    elif args.arm == "persist":
                        operator = PERSIST_OPS[min(n_done, len(PERSIST_OPS) - 1)]
                    hist = np.stack(boundary_positions[-3:]) if len(boundary_positions) >= 3 else np.repeat(boundary_positions[-1][None], 3, 0)
                    initial = np.asarray(observation["robot0_eef_pos"], float)
                    record.update(operator=operator, start_pos=initial.tolist())
                    phys = 0
                    if operator in ("hold16", "grip_cycle"):
                        # hold16: 16 zero-motion steps; grip_cycle: 8 steps with the gripper toggled, then 8 steps restored
                        for i in range(16):
                            action = np.zeros(7, np.float32)
                            action[6] = (-last_gripper if (operator == "grip_cycle" and i < 8) else last_gripper)
                            phys += 1
                            summary["physical_steps"] += 1
                            if step_env(action):
                                stop = True
                                break
                    else:
                        steps_max = PHYS_STEPS_MAX * (2 if operator in ("withdraw8", "lift8", "retrace") else 1)
                        for target in waypoints(operator, initial, hist):
                            for _ in range(steps_max):
                                pos = np.asarray(observation["robot0_eef_pos"], float)
                                delta = bounded(target - pos, PHYS_STEP_M)
                                if np.linalg.norm(target - pos) < 0.005:
                                    break
                                action = np.zeros(7, np.float32)
                                action[:3] = np.clip(delta / output_scale, -1, 1)
                                action[6] = last_gripper
                                phys += 1
                                summary["physical_steps"] += 1
                                if step_env(action):
                                    stop = True
                                    break
                            if stop:
                                break
                    record.update(physical_steps=phys, end_pos=np.asarray(observation["robot0_eef_pos"], float).tolist())
                else:
                    record.update(operator=args.arm)
                    resample_left = RESAMPLE_QUERIES
                summary["interventions"].append(record)
                interventions_left -= 1
                cooldown = COOLDOWN_QUERIES
                detector.reset_confirmation()
                if stop:
                    break
        summary["status"] = "completed"
    except BaseException as error:
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        raise
    finally:
        for c in clients:
            try:
                c.close()
            except Exception:
                pass
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass
        summary["duration_seconds"] = time.perf_counter() - started
        summary["queries"] = len(per_query)
        summary["max_rr"] = float(np.nanmax([r[2] for r in per_query])) if per_query and np.isfinite([r[2] for r in per_query]).any() else None
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        np.savez_compressed(args.out / "control.npz",
                            per_query=np.array([(q, s, rr, d, c, e) for q, s, rr, d, c, e, _ in per_query], float),
                            modes=np.array([m for *_, m in per_query]),
                            eef_positions=np.array(states, float), actions=np.array(actions_log, np.float32),
                            route_points=np.array(detector.points, np.float32) if detector.points else np.zeros((0, 0), np.float32))
        print(json.dumps({k: summary[k] for k in ("arm", "status", "success", "action_steps", "inference_calls", "candidate_calls",
                                                   "physical_steps", "first_trigger_query", "bitwise_matches", "bitwise_checked")}), flush=True)


if __name__ == "__main__":
    main()
