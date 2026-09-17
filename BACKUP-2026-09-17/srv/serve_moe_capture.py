"""WebSocket HiMoE policy server that also captures HB MoE port tensors per request.

Same policy, same protocol and same actions as serve_with_recorder.py.  For every
inference request it additionally saves one compressed npz with:

  hb_router_probs   [8, 10, 11, 32] float16   full HB routing probabilities (recorder)
  hb_expert_ids     [8, 10, 11, 4]  uint8     executed top-4 expert ids
  hb_selected_prob  [8, 10, 11, 4]  float16
  as_expert_ids / as_probs                     AS routing (collapsed when loop-invariant)
  mechanism/input   [8, 10, 11, 1024] float32  HB MoE module input   (P3j port capture)
  mechanism/shared  [8, 10, 11, 1024] float32  shared-expert output
  mechanism/total   [8, 10, 11, 1024] float32  aggregated MoE output
  actions           the returned action chunk

Requests are attributed by the client-supplied ``episode_id`` (int), ``capture/tag``
(str, becomes the directory name) and ``capture/step`` (int, becomes the file name).
Those keys are stripped before the policy sees the observation.  Files are written by
background threads so inference latency is not paid for compression.
"""
import argparse
import json
import os
import pathlib
import queue
import signal
import sys
import threading
import time

import numpy as np

HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
FIELDS = ("input", "shared", "total")


class MoEPortCapture:
    """Forward hooks on the eight HB MoE modules; identical to P3j's moe_compute_capture."""

    def __init__(self, model):
        self.layers = model.paligemma_with_expert.gemma_expert.layers
        self.handles = []
        self.records = {field: [] for field in FIELDS}

    def hook(self, field, layer):
        def capture(module, inputs, output):
            value = inputs[0] if field == "input" else output
            if tuple(value.shape) != (1, 11, 1024):
                raise RuntimeError("Unexpected HB MoE port shape: " + field)
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

    def response(self):
        import torch
        result = {}
        for field, rows in self.records.items():
            if [i for i, _ in rows] != list(HB_LAYERS) * 10:
                raise RuntimeError("Incomplete MoE port capture: " + field)
            values = torch.stack([value for _, value in rows]).float().cpu().numpy()
            values = np.ascontiguousarray(values.reshape(10, 8, 11, 1024).transpose(1, 0, 2, 3))
            if not np.isfinite(values).all():
                raise RuntimeError("Nonfinite MoE port")
            result["mechanism/" + field] = values
        return result


class Writer:
    def __init__(self, threads=3):
        self.q = queue.Queue(maxsize=256)
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
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **payload)
                os.replace(tmp, path)
            except Exception as error:  # noqa: BLE001
                self.errors.append("%s: %r" % (path, error))
                print("WRITE ERROR %s: %r" % (path, error), flush=True)
            finally:
                self.q.task_done()

    def put(self, path, payload):
        self.q.put((path, payload))

    def close(self):
        self.q.join()
        for _ in self.threads:
            self.q.put(None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--suite", default="long")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--libero-wrist-layout", default="released-left")
    ap.add_argument("--out", required=True, help="capture root; one sub-directory per capture/tag")
    ap.add_argument("--writer-threads", type=int, default=3)
    args = ap.parse_args()

    from himoe_libero_bridge.server import PolicyServer, create_policy

    import torch
    # several server processes share each GPU and the host; unbounded intra-op threads (one per core)
    # spin during the CPU-side parts of every request and burn ~2 cores per process for nothing
    torch.set_num_threads(int(os.environ.get("HIMOE_SERVER_THREADS", "4")))
    policy = create_policy("himoe", args.checkpoint_dir, args.upstream_root, args.gpu,
                           args.suite, args.libero_wrist_layout)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from himoe_router_recorder import HiMoERouteRecorder

    core = policy._policy.model
    recorder = HiMoERouteRecorder(core, store_full_probs=True).attach()
    print("discovered %d gates; HB layers %s" % (len(recorder.gates), recorder.hb_layers), flush=True)
    if list(recorder.hb_layers) != list(HB_LAYERS):
        raise SystemExit("HB layer set changed: %s" % recorder.hb_layers)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(args.writer_threads)
    index_lock = threading.Lock()
    index = (out / "index.jsonl").open("a")
    stats = {"calls": 0, "infer_s": 0.0, "capture_s": 0.0}
    per_episode = {}
    inner = policy.infer

    def infer(observation):
        episode_id = int(observation.pop("episode_id", -1))
        tag = str(observation.pop("capture/tag", "episode-%d" % episode_id))
        step = observation.pop("capture/step", None)
        if step is None:
            step = per_episode.get(episode_id, 0)
        per_episode[episode_id] = int(step) + 1
        recorder.begin_control_step(episode_id=episode_id, control_step=stats["calls"])
        t0 = time.perf_counter()
        with MoEPortCapture(recorder.core) as capture:
            response = inner(observation)
        t1 = time.perf_counter()
        rec = recorder.end_control_step()
        payload = capture.response()
        payload.update({
            "hb_router_probs": rec.hb_router_probs[0],
            "hb_expert_ids": rec.hb_expert_ids[0],
            "hb_selected_prob": rec.hb_selected_prob[0],
            "hb_entropy": rec.hb_entropy[0],
            "as_expert_ids": rec.as_expert_ids[0],
            "as_probs": rec.as_probs[0],
            "as_collapsed": np.array(bool(rec.as_collapsed)),
            "actions": np.asarray(response["actions"], dtype=np.float32),
            "episode_id": np.array(episode_id),
            "step": np.array(int(step)),
            "control_step": np.array(stats["calls"]),
        })
        if "flow/noise_sha256" in response:
            payload["flow_noise_sha256"] = np.array(str(response["flow/noise_sha256"]))
        path = out / tag / ("q%03d.npz" % int(step))
        writer.put(path, payload)
        t2 = time.perf_counter()
        with index_lock:
            index.write(json.dumps({
                "episode_id": episode_id, "tag": tag, "step": int(step), "control_step": stats["calls"],
                "path": str(path.relative_to(out)), "inference_ms": round(1000 * (t1 - t0), 1),
                "unix": time.time(),
            }) + "\n")
            index.flush()
        stats["calls"] += 1
        stats["infer_s"] += t1 - t0
        stats["capture_s"] += t2 - t1
        if stats["calls"] % 50 == 0:
            print("captured %d requests  (infer %.0f ms, capture %.0f ms, queue %d)"
                  % (stats["calls"], 1000 * stats["infer_s"] / stats["calls"],
                     1000 * stats["capture_s"] / stats["calls"], writer.q.qsize()), flush=True)
        return response

    policy.infer = infer
    policy.metadata["moe_capture"] = "hb-ports-v1"
    policy.metadata["moe_capture_fields"] = list(FIELDS)
    policy.metadata["episode_id_key"] = "episode_id"
    policy.metadata["capture_tag_key"] = "capture/tag"
    cfg = policy._policy.model.config
    policy.metadata["n_action_steps"] = int(cfg.n_action_steps)
    policy.metadata["num_steps"] = int(cfg.num_steps)
    (out / "server-metadata.json").write_text(json.dumps(
        {k: v for k, v in policy.metadata.items() if isinstance(v, (str, int, float, bool, list, type(None)))},
        indent=2, default=str))

    def shutdown(*_):
        print("shutting down: flushing %d queued captures" % writer.q.qsize(), flush=True)
        writer.close()
        index.close()
        (out / "capture-summary.json").write_text(json.dumps(dict(stats, write_errors=writer.errors), indent=2))
        recorder.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print("serving on ws://%s:%d  (capture -> %s)" % (args.host, args.port, out), flush=True)
    try:
        PolicyServer(policy, args.host, args.port, backend=getattr(policy, "backend_name", "himoe")).serve_forever()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
