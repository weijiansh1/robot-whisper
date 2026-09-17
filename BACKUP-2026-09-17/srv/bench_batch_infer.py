"""Feasibility benchmark for batched HiMoE inference.

Loads the policy exactly like the servers do, takes real observations from saved
episode traces, and compares
  * batch-1 inference through the production path (HiMoEPolicy.infer)
  * batched inference: per-sample input transforms, stacked tensors, one
    sample_actions call with stacked per-request flow noise, per-sample output transforms
for throughput (requests/second) and numerical difference of the returned actions.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "packages" / "openpi-client" / "src"))


def load_observations(count):
    traces = sorted(glob.glob("/home/swj/data/libero-runtime/simulations/topo-all-f/*/episode-*/episode-trace.npz"))
    obs = []
    for trace in traces:
        summary = json.loads((Path(trace).parent / "summary.json").read_text())
        with np.load(trace) as z:
            for q in (0, len(z["images"]) // 2):
                obs.append({"observation/image": z["images"][q].astype(np.uint8),
                            "observation/wrist_image": z["wrist_images"][q].astype(np.uint8),
                            "observation/state": z["states"][q].astype(np.float32),
                            "prompt": summary["selected_task_prompt"],
                            "flow/noise": z["flow_noises"][q].astype(np.float32)})
                if len(obs) >= count:
                    return obs
    return obs


def infer_batch(policy, observations):
    """Batched twin of moevla.policies.policy.Policy.infer."""
    from moevla.models.model import from_dict, preprocess_observation_and_to_device
    from moevla.policies.policy import tree_map
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
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actions = inner._sample_actions(observation["images"], observation["image_masks"], observation["tokenized_prompt"],
                                        observation["tokenized_prompt_mask"], observation["state"], observation["data_mask"],
                                        noise=flow_noise)
    outputs = []
    for i in range(len(observations)):
        out = {"state": inputs["state"][i].cpu().numpy(), "actions": actions[i].cpu().numpy()}
        outputs.append(inner._output_transform(out))
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    from himoe_libero_bridge.server import create_policy
    data = "/home/swj/data"
    policy = create_policy("himoe", data + "/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/checkpoints/HiMoE-VLA-Libero-10",
                           data + "/srv", a.gpu, "long", "released-left")
    observations = load_observations(32)
    print("observations: %d from real traces" % len(observations), flush=True)

    # reference: production batch-1 path (twice, to measure run-to-run determinism)
    ref = [policy.infer(dict(o))["actions"] for o in observations]
    ref2 = [policy.infer(dict(o))["actions"] for o in observations]
    print("batch-1 run-to-run max|diff| = %.2e" % max(np.abs(x - y).max() for x, y in zip(ref, ref2)), flush=True)
    # my batched path at B=1 must equal the production path
    b1 = [infer_batch(policy, [o])[0]["actions"] for o in observations[:8]]
    print("batched path at B=1 vs production batch-1: max|diff| = %.2e" % max(np.abs(x - y).max() for x, y in zip(ref[:8], b1)), flush=True)

    torch.cuda.synchronize()
    t = time.perf_counter()
    for o in observations[:8]:
        policy.infer(dict(o))
    torch.cuda.synchronize()
    single = (time.perf_counter() - t) / 8
    print("production batch-1: %.3f s/request -> %.2f req/s" % (single, 1 / single), flush=True)

    rows = []
    for B in [int(x) for x in a.batches.split(",")]:
        batch = [observations[i % len(observations)] for i in range(B)]
        infer_batch(policy, batch)  # warm-up for this shape
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(a.repeats):
            outs = infer_batch(policy, batch)
        torch.cuda.synchronize()
        per_batch = (time.perf_counter() - t) / a.repeats
        diff = max(np.abs(outs[i]["actions"] - ref[i % len(observations)]).max() for i in range(B))
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        rows.append((B, per_batch, B / per_batch, diff, mem))
        print("B=%2d: %.3f s/batch  %.2f req/s  (%.1fx of batch-1)  max|diff| vs batch-1 %.2e  peak mem %.1f GB"
              % (B, per_batch, B / per_batch, (B / per_batch) * single, diff, mem), flush=True)
        torch.cuda.reset_peak_memory_stats()
    json.dump([dict(batch=B, seconds=s, req_per_s=r, max_diff=float(d), peak_gb=m) for B, s, r, d, m in rows],
              open("/tmp/bench_batch_infer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
