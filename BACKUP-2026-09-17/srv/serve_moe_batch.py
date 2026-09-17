"""Batched WebSocket HiMoE policy server with optional per-request MoE capture.

Same wire protocol as serve_with_recorder.py / serve_moe_capture.py, but requests from all
connected clients are queued and executed as one batched ``sample_actions`` call (up to
``--max-batch`` requests, waiting at most ``--max-wait-ms`` for the batch to fill).  On an
H20 a batch of 32 gives ~8x the requests/second of batch-1 inference for ~2 GB extra memory.

Numerics: batched bf16 inference is not bit-identical to batch-1 (observed max action
difference ~1e-2), so runs made with this server are comparable with each other but not
bit-for-bit with serve_moe_capture.py runs.

Capture (``--capture-out``): for every request, the full HB routing probabilities (recorder,
batch-split) and the HB MoE input/shared/total port tensors (batch-split) are saved to
<out>/<capture/tag>/q<step>.npz, identical layout to serve_moe_capture.py.
Requests carrying ``routing/capture`` fall back to the single-request production path.
"""
import argparse
import asyncio
import json
import os
import pathlib
import queue
import signal
import sys
import threading
import time
import zipfile

import numpy as np

HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
FIELDS = ("input", "shared", "total")
ROUTING_CAPTURE_KEY = "routing/capture"


class MoEPortCapture:
    """Forward hooks on the eight HB MoE modules, batch-aware (values are [B, 11, 1024])."""

    def __init__(self, model):
        self.layers = model.paligemma_with_expert.gemma_expert.layers
        self.handles = []
        self.records = {field: [] for field in FIELDS}

    def hook(self, field, layer):
        def capture(module, inputs, output):
            value = inputs[0] if field == "input" else output
            if value.ndim != 3 or tuple(value.shape[1:]) != (11, 1024):
                raise RuntimeError("Unexpected HB MoE port shape: %s %s" % (field, tuple(value.shape)))
            self.records[field].append((layer, value.detach().clone()))
        return capture

    def __enter__(self):
        try:
            for index in HB_LAYERS:
                module = self.layers[index].mlp
                if type(module).__name__ != "HBMoE":
                    raise RuntimeError("HB MoE module identity changed")
                for field in FIELDS:
                    port = module.shared_experts if field == "shared" else module
                    self.handles.append(port.register_forward_hook(self.hook(field, index)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def response(self, batch_size):
        import torch
        result = {}
        for field, rows in self.records.items():
            if [i for i, _ in rows] != list(HB_LAYERS) * 10:
                raise RuntimeError("Incomplete MoE port capture: " + field)
            values = torch.stack([value for _, value in rows]).float().cpu().numpy()   # [80, B, 11, 1024]
            values = values.reshape(10, 8, batch_size, 11, 1024).transpose(2, 1, 0, 3, 4)  # [B, 8, 10, 11, 1024]
            if not np.isfinite(values).all():
                raise RuntimeError("Nonfinite MoE port")
            result["mechanism/" + field] = np.ascontiguousarray(values)
        return result


def save_npz(path, payload, level):
    """np.savez with a configurable deflate level (numpy's savez_compressed pins level 6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED if level else zipfile.ZIP_STORED, compresslevel=level or None) as zf:
        for key, value in payload.items():
            with zf.open(key + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
    os.replace(tmp, path)


class Writer:
    def __init__(self, threads, level):
        self.q = queue.Queue(maxsize=2048)
        self.level = level
        self.errors = []
        self.threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(threads)]
        for t in self.threads:
            t.start()

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            path, payload = item
            try:
                save_npz(path, payload, self.level)
            except Exception as error:  # noqa: BLE001
                self.errors.append("%s: %r" % (path, error))
                print("WRITE ERROR %s: %r" % (path, error), flush=True)
            finally:
                self.q.task_done()

    def close(self):
        self.q.join()
        for _ in self.threads:
            self.q.put(None)


class ASForce:
    """Per-sample routing overrides.
    forced_as[b]: int (same expert at all 4 AS layers), or list of 4 ints (per AS layer), -1 = model's choice.
    forced_hb[b]: None or list of 8 lists of 4 expert ids (per HB layer, in recorder layer order); weights are the
    gate's own softmax scores of the forced experts, renormalised to sum 1 (HB) / raw score (AS, top-1)."""

    def __init__(self, gates, forced_as, forced_hb=None):
        self.as_gates = [g for g in gates if g.kind == "AS"]
        self.hb_gates = [g for g in gates if g.kind == "HB"]
        self.forced_as = forced_as
        self.forced_hb = forced_hb
        self.handles = []

    def __enter__(self):
        import torch
        import torch.nn.functional as F

        def make_hook(layer_pos, kind):
            def hook(module, args, output):
                topk_idx, topk_weight, aux = output
                hs = args[0].to(torch.bfloat16)
                B, seq, h = hs.shape
                spec = [(f[layer_pos] if isinstance(f, (list, tuple)) else f) for f in self.forced_as] if kind == "AS" else \
                       [(f[layer_pos] if f is not None else None) for f in self.forced_hb]
                if all((x is None) or (isinstance(x, int) and x < 0) for x in spec):
                    return None
                scores = F.linear(hs.view(-1, h), module.weight, None).softmax(dim=-1).view(B, seq, -1)
                idx = topk_idx.view(B, seq, -1).clone()
                w = topk_weight.view(B, seq, -1).clone()
                for b, x in enumerate(spec):
                    if x is None or (isinstance(x, int) and x < 0):
                        continue
                    ids = torch.as_tensor(list(x) if isinstance(x, (list, tuple)) else [int(x)], device=hs.device)
                    idx[b] = ids.view(1, -1).expand(seq, -1).to(idx.dtype)
                    sc = scores[b][:, ids]
                    if kind == "HB":
                        sc = sc / (sc.sum(-1, keepdim=True) + 1e-20)
                    w[b] = sc.to(w.dtype)
                return idx.view_as(topk_idx), w.view_as(topk_weight), aux
            return hook
        for pos, g in enumerate(self.as_gates):
            self.handles.append(g.module.register_forward_hook(make_hook(pos, "AS")))
        if self.forced_hb is not None and any(f is not None for f in self.forced_hb):
            for pos, g in enumerate(self.hb_gates):
                self.handles.append(g.module.register_forward_hook(make_hook(pos, "HB")))
        return self

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def infer_batch(policy, observations, want_flow_path=False, as_forced=None, gates=None, hb_forced=None):
    """Batched twin of moevla.policies.policy.Policy.infer (per-sample transforms, one sample_actions)."""
    import torch
    from moevla.models.model import from_dict, preprocess_observation_and_to_device
    inner = policy._policy
    per_sample, noises = [], []
    for obs in observations:
        inputs = dict(obs)
        noise = inputs.pop("flow/noise", None)
        inputs = inner._input_transform(inputs)
        per_sample.append((inputs, from_dict(inputs)))
        noises.append(noise)

    def stack(key_fn):
        first = key_fn(per_sample[0])
        if isinstance(first, dict):
            return {k: stack(lambda s, k=k: key_fn(s)[k]) for k in first}
        return torch.stack([torch.as_tensor(key_fn(s)) for s in per_sample])

    inputs = stack(lambda s: s[0])
    observation = stack(lambda s: s[1])
    observation = preprocess_observation_and_to_device(observation, train=False)
    device = observation["state"].device
    flow_noise = None if any(n is None for n in noises) else torch.tensor(np.stack(noises), dtype=torch.float32, device=device)
    path = []
    model = inner.model
    if want_flow_path:
        original = model.denoise_step

        def recording_denoise_step(state, ppm, pam, kv, dm, x_t, t):
            if not path:
                path.append(x_t.detach().float().cpu().numpy().copy())
            v = original(state, ppm, pam, kv, dm, x_t, t)
            path.append((x_t + (-1.0 / model.config.num_steps) * v).detach().float().cpu().numpy().copy())
            return v
        model.denoise_step = recording_denoise_step
    def _active(x):
        return x is not None and not (isinstance(x, int) and x < 0)
    need = gates is not None and ((as_forced is not None and any(_active(x) for x in as_forced)) or (hb_forced is not None and any(x is not None for x in hb_forced)))
    force_ctx = ASForce(gates, as_forced if as_forced is not None else [-1] * len(observations), hb_forced) if need else None
    try:
        if force_ctx is not None:
            force_ctx.__enter__()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions = inner._sample_actions(observation["images"], observation["image_masks"], observation["tokenized_prompt"],
                                            observation["tokenized_prompt_mask"], observation["state"], observation["data_mask"],
                                            noise=flow_noise)
    finally:
        if force_ctx is not None:
            force_ctx.__exit__(None, None, None)
        if want_flow_path:
            model.denoise_step = original
    outputs = []
    flow_path = np.stack(path, axis=1) if path else None   # [B, 11, 10, 24] (noise, then after each Euler step)
    for i in range(len(observations)):
        state_i = inputs["state"][i].cpu().numpy()
        out = inner._output_transform({"state": state_i, "actions": actions[i].cpu().numpy()})
        if flow_path is not None:
            out["flow/path"] = flow_path[i]
            # the same output transform applied to every intermediate denoising state -> executable partial-denoise chunks
            out["flow/path_actions"] = np.stack([inner._output_transform({"state": state_i, "actions": flow_path[i][s]})["actions"]
                                                 for s in range(flow_path.shape[1])]).astype(np.float32)
        outputs.append(out)
    return outputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--suite", default="long")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--libero-wrist-layout", default="released-left")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-wait-ms", type=float, default=15.0, help="how long to wait for more requests before running a partial batch")
    ap.add_argument("--capture-out", default=None, help="if set, save HB routing + MoE port tensors per request under this directory")
    ap.add_argument("--writer-threads", type=int, default=6)
    ap.add_argument("--compress-level", type=int, default=1, help="deflate level for capture files; 0 = store")
    args = ap.parse_args()

    import hashlib
    import torch
    torch.set_num_threads(int(os.environ.get("HIMOE_SERVER_THREADS", "4")))
    from himoe_libero_bridge.server import create_policy
    from himoe_libero_bridge.protocol import Packer, server_metadata, unpackb, validate_action_response, validate_observation

    policy = create_policy("himoe", args.checkpoint_dir, args.upstream_root, args.gpu, args.suite, args.libero_wrist_layout)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from himoe_router_recorder import HiMoERouteRecorder

    recorder = None
    writer = None
    out = None
    if args.capture_out:
        out = pathlib.Path(args.capture_out)
        out.mkdir(parents=True, exist_ok=True)
        recorder = HiMoERouteRecorder(policy._policy.model, store_full_probs=True).attach()
        if list(recorder.hb_layers) != list(HB_LAYERS):
            raise SystemExit("HB layer set changed: %s" % recorder.hb_layers)
        writer = Writer(args.writer_threads, args.compress_level)
        index = (out / "index.jsonl").open("a")
        index_lock = threading.Lock()
    core = policy._policy.model
    cfg = core.config
    probe = recorder if recorder is not None else HiMoERouteRecorder(core, store_full_probs=True).attach()

    metadata = server_metadata(getattr(policy, "backend_name", "himoe"))
    metadata.update(getattr(policy, "metadata", {}))
    metadata.update({"server_instance_id": os.urandom(16).hex(), "server_pid": os.getpid(),
                     "server_started_unix_ns": time.time_ns(), "batched_inference": True, "max_batch": args.max_batch,
                     "episode_id_key": "episode_id", "capture_tag_key": "capture/tag",
                     "moe_capture": "hb-ports-v1" if out else None, "n_action_steps": int(cfg.n_action_steps),
                     "num_steps": int(cfg.num_steps)})
    if out:
        (out / "server-metadata.json").write_text(json.dumps(
            {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool, list, type(None)))}, indent=2, default=str))

    stats = {"calls": 0, "batches": 0, "infer_s": 0.0, "capture_s": 0.0, "batch_hist": {}}
    per_episode = {}
    pending = queue.Queue()

    def run_batch(items):
        """items: list of (observation, meta) -> list of responses; runs in the worker thread."""
        observations = [{k: v for k, v in o.items() if not (k.startswith("capture/") or k.startswith("intervene/"))} for o, _ in items]
        def _as(o):
            v = o.get("intervene/as_expert", -1)
            return [int(x) for x in np.asarray(v).reshape(-1)] if np.ndim(v) else int(v)
        as_forced = [_as(o) for o, _ in items]
        hb_forced = [None if o.get("intervene/hb_experts") is None else [[int(e) for e in row] for row in np.asarray(o["intervene/hb_experts"]).reshape(8, -1)] for o, _ in items]
        want_flow = any(o.get("capture/return_flow") for o, _ in items)
        want_input = any(o.get("capture/return_input") for o, _ in items)
        want_probs = any(o.get("capture/return_probs") for o, _ in items) or want_input
        t0 = time.perf_counter()
        if recorder is not None or want_probs:
            if recorder is not None or want_input:
                (recorder or probe).begin_control_step(episode_id=-1, control_step=stats["batches"])
                with MoEPortCapture(probe.core) as capture:
                    outs = infer_batch(policy, observations, want_flow, as_forced, probe.gates, hb_forced)
                rec = (recorder or probe).end_control_step()
                ports = capture.response(len(items))
            else:
                probe.begin_control_step(episode_id=-1, control_step=stats["batches"])
                outs = infer_batch(policy, observations, want_flow, as_forced, probe.gates, hb_forced)
                rec = probe.end_control_step()
        else:
            outs = infer_batch(policy, observations, want_flow, as_forced, probe.gates, hb_forced)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        responses = []
        for i, ((obs, meta), out_i) in enumerate(zip(items, outs)):
            response = validate_action_response({"actions": out_i["actions"]})
            if "flow/noise" in obs:
                response["flow/noise_sha256"] = hashlib.sha256(np.ascontiguousarray(obs["flow/noise"], dtype=np.float32).tobytes()).hexdigest()
            response["server/inference_ms"] = np.float32(1000.0 * (t1 - t0) / len(items))
            response["server/batch_size"] = len(items)
            obs = items[i][0]
            if obs.get("capture/return_probs"):
                response["routing/hb_probs"] = np.asarray(rec.hb_router_probs[i], dtype=np.float16)
            if obs.get("capture/return_input"):
                response["moe/input_back_last"] = np.asarray(ports["mechanism/input"][i][4:, -1, 1:, :], dtype=np.float16)   # [4, 10, 1024]
            if obs.get("capture/return_flow") and "flow/path" in out_i:
                response["flow/path"] = np.asarray(out_i["flow/path"], dtype=np.float32)
                response["flow/path_actions"] = np.asarray(out_i["flow/path_actions"], dtype=np.float32)
            if obs.get("capture/return_probs"):
                response["routing/as_expert_ids"] = np.asarray(rec.as_expert_ids[i]).astype(np.int16)
            if recorder is not None:
                episode_id, tag, step = meta
                payload = {k: v[i] for k, v in ports.items()}
                payload.update({
                    "hb_router_probs": rec.hb_router_probs[i], "hb_expert_ids": rec.hb_expert_ids[i],
                    "hb_selected_prob": rec.hb_selected_prob[i], "hb_entropy": rec.hb_entropy[i],
                    "as_expert_ids": rec.as_expert_ids[i], "as_probs": rec.as_probs[i],
                    "as_collapsed": np.array(bool(rec.as_collapsed)),
                    "actions": np.asarray(response["actions"], dtype=np.float32),
                    "episode_id": np.array(episode_id), "step": np.array(int(step)), "control_step": np.array(stats["calls"] + i),
                    "batch_size": np.array(len(items)),
                })
                if "flow/noise_sha256" in response:
                    payload["flow_noise_sha256"] = np.array(str(response["flow/noise_sha256"]))
                path = out / tag / ("q%03d.npz" % int(step))
                writer.q.put((path, payload))
                with index_lock:
                    index.write(json.dumps({"episode_id": episode_id, "tag": tag, "step": int(step), "control_step": stats["calls"] + i,
                                            "batch_size": len(items), "path": str(path.relative_to(out)),
                                            "inference_ms": round(1000 * (t1 - t0), 1), "unix": time.time()}) + "\n")
                    index.flush()
            responses.append(response)
        t2 = time.perf_counter()
        stats["calls"] += len(items)
        stats["batches"] += 1
        stats["infer_s"] += t1 - t0
        stats["capture_s"] += t2 - t1
        stats["batch_hist"][len(items)] = stats["batch_hist"].get(len(items), 0) + 1
        if stats["batches"] % 25 == 0:
            print("served %d requests in %d batches (mean batch %.1f, %.0f ms/batch, capture %.0f ms/batch, queue %d)"
                  % (stats["calls"], stats["batches"], stats["calls"] / stats["batches"], 1000 * stats["infer_s"] / stats["batches"],
                     1000 * stats["capture_s"] / stats["batches"], writer.q.qsize() if writer else 0), flush=True)
        return responses

    def worker(loop):
        while True:
            first = pending.get()
            if first is None:
                return
            items = [first]
            deadline = time.perf_counter() + args.max_wait_ms / 1000.0
            while len(items) < args.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    items.append(pending.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                responses = run_batch([(obs, meta) for obs, meta, _ in items])
                for (_, _, fut), response in zip(items, responses):
                    loop.call_soon_threadsafe(fut.set_result, response)
            except Exception as error:  # noqa: BLE001
                for _, _, fut in items:
                    loop.call_soon_threadsafe(fut.set_exception, error)

    async def handler(websocket, *unused):
        packer = Packer()
        await websocket.send(packer.pack(metadata))
        loop = asyncio.get_running_loop()
        while True:
            try:
                request = await websocket.recv()
                if isinstance(request, str):
                    raise ValueError("Inference request must be a binary msgpack frame")
                observation = validate_observation(unpackb(request))
                episode_id = int(observation.pop("episode_id", -1))
                tag = str(observation.pop("capture/tag", "episode-%d" % episode_id))
                step = observation.pop("capture/step", None)
                if step is None:
                    step = per_episode.get(episode_id, 0)
                per_episode[episode_id] = int(step) + 1
                if observation.get(ROUTING_CAPTURE_KEY):
                    response = await asyncio.to_thread(policy.infer, observation)   # single-request production path
                elif "candidates/noises" in observation:
                    noises = np.asarray(observation.pop("candidates/noises"), dtype=np.float32)
                    if noises.ndim != 3 or noises.shape[1:] != (10, 24):
                        raise ValueError("candidates/noises must be [K, 10, 24]")
                    futures = []
                    as_list = observation.pop("candidates/as_experts", None)
                    hb_list = observation.pop("candidates/hb_experts", None)
                    for k in range(len(noises)):
                        item = dict(observation)
                        item["flow/noise"] = np.ascontiguousarray(noises[k])
                        if as_list is not None:
                            a = np.asarray(as_list)
                            item["intervene/as_expert"] = a[k] if a.ndim == 2 else int(a.reshape(-1)[k])
                        if hb_list is not None:
                            item["intervene/hb_experts"] = np.asarray(hb_list)[k]
                        fut = loop.create_future()
                        pending.put((item, (episode_id, "%s/cand%02d" % (tag, k), step), fut))
                        futures.append(fut)
                    results = await asyncio.gather(*futures)
                    response = {"actions": results[0]["actions"], "candidates/actions": np.stack([r["actions"] for r in results]),
                                "server/batch_size": results[0]["server/batch_size"],
                                "server/inference_ms": results[0]["server/inference_ms"]}
                    if "routing/hb_probs" in results[0]:
                        response["candidates/hb_probs"] = np.stack([r["routing/hb_probs"] for r in results])
                    if "moe/input_back_last" in results[0]:
                        response["candidates/input_back_last"] = np.stack([r["moe/input_back_last"] for r in results])
                    if "flow/path" in results[0]:
                        response["candidates/flow_path"] = np.stack([r["flow/path"] for r in results])
                        response["candidates/flow_path_actions"] = np.stack([r["flow/path_actions"] for r in results])
                    if "routing/as_expert_ids" in results[0]:
                        response["candidates/as_expert_ids"] = np.stack([r["routing/as_expert_ids"] for r in results])
                else:
                    fut = loop.create_future()
                    pending.put((observation, (episode_id, tag, step), fut))
                    response = await fut
                await websocket.send(packer.pack(response))
            except Exception as error:  # noqa: BLE001
                if error.__class__.__name__ in ("ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError"):
                    return
                import traceback
                payload = {"error": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                try:
                    await websocket.send(json.dumps(payload))
                    await websocket.close(code=1011, reason="Inference failed")
                finally:
                    return

    def shutdown(*_):
        print("shutting down: %d requests, %d batches, batch histogram %s" % (stats["calls"], stats["batches"], stats["batch_hist"]), flush=True)
        if writer is not None:
            writer.close()
            index.close()
            (out / "capture-summary.json").write_text(json.dumps(dict(stats, write_errors=writer.errors), indent=2))
        if recorder is not None:
            recorder.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    async def serve():
        try:
            from websockets.asyncio.server import serve as ws_serve
        except ImportError:
            from websockets.server import serve as ws_serve
        loop = asyncio.get_running_loop()
        threading.Thread(target=worker, args=(loop,), daemon=True).start()
        async with ws_serve(handler, args.host, args.port, compression=None, max_size=None) as server:
            print("serving on ws://%s:%d  (batched, max batch %d, capture -> %s)" % (args.host, args.port, args.max_batch, out), flush=True)
            await server.serve_forever()

    try:
        asyncio.run(serve())
    finally:
        shutdown()


if __name__ == "__main__":
    main()
