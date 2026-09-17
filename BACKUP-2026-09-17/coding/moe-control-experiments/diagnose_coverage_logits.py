"""One extra frozen-request replay to inspect pre-softmax BF16 arithmetic."""

import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from gate_runtime import infer_isolated, load_isolated
from gate_capture import DtypeScopeCapture, GateProbePolicy
from collection_routes import CAPTURE_KEY, FullASCapture
from scope_bias_control import BIAS_KEY
from run_gate_experiment import FULL_KEYS, assert_equal, save_json, sha256_file, source_for


class LogitCapture(DtypeScopeCapture):
    def __init__(self, layers, bias):
        super().__init__(layers, bias)
        self.logit_records = []

    def _hook(self, layer):
        upstream = super()._hook(layer)

        def capture(gate, inputs, output):
            import torch
            from torch.nn import functional as F

            step, slot = divmod(len(self.records), 8)
            result = upstream(gate, inputs, output)
            logits = F.linear(inputs[0].reshape(-1, inputs[0].shape[-1]), gate.weight, None)
            bias = torch.as_tensor(self.bias[slot, step], device=logits.device, dtype=logits.dtype)
            effective_logits = logits + bias
            record = self.records[-1]
            if not torch.equal(logits.softmax(-1), record[1]) or not torch.equal(effective_logits.softmax(-1), record[7]):
                raise RuntimeError("Recorded logits do not reproduce actual gate probabilities")
            self.logit_records.append((logits.detach().clone(), effective_logits.detach().clone(), bias.detach().clone()))
            return result

        return capture

    def response(self):
        import torch

        result = super().response()
        if len(self.logit_records) != 80:
            raise RuntimeError("Incomplete raw-logit capture")
        for index, name in enumerate(("native_logits_fp32", "effective_logits_fp32", "applied_bias_fp32")):
            value = torch.stack([row[index] for row in self.logit_records])
            result["diagnostic/" + name] = value.reshape(10, 8, 11, 32).permute(1, 0, 2, 3).float().cpu().numpy().copy()
        result["diagnostic/logit_dtypes"] = sorted({str(row[0].dtype) for row in self.logit_records})
        result["diagnostic/probability_dtypes"] = sorted({str(row[1].dtype) for row in self.records})
        result["diagnostic/gpu_exact_logit_softmax"] = True
        return result


class LogitPolicy(GateProbePolicy):
    def infer(self, observation):
        request = dict(observation)
        bias = request.pop(BIAS_KEY)
        if not request.pop(CAPTURE_KEY) or not request.get("routing/capture"):
            raise ValueError("Diagnostic requires full capture")
        with LogitCapture(self.policy._routing_layers, bias) as capture, FullASCapture(self.as_layers) as as_capture:
            result = self.policy.infer(request)
        result.update(capture.response())
        result.update(as_capture.response())
        return result


def main():
    root = Path(sys.argv[1]).resolve()
    config = json.loads((root / "config.json").read_text())
    collection = json.loads((root / "collection.json").read_text())
    row, = [r for r in collection["rows"] if r["parent"] == config["parents"][0]["parent"] and r["kind"] == "candidate"
            and r["spec"] == dict(kind="candidate", generator="state_gate", pool=0, candidate=0)]
    directory = root / "logit-diagnostic"
    directory.mkdir()
    result = dict(passed=False, extra_model_calls_attempted=0, extra_model_calls_completed=0,
                  formal_reference_path=row["path"], formal_reference_sha256=row["sha256"])
    save_json(directory / "protocol.json", dict(purpose="Post-collection arithmetic audit, not a new candidate",
                                                max_extra_model_calls=1, source_row=row,
                                                script_sha256=sha256_file(Path(__file__))))
    started = time.monotonic()
    try:
        wrapped, loaded = load_isolated()
        save_json(directory / "model-load.json", loaded)
        parent = config["parents"][0]
        _, _, request = source_for(parent)
        with np.load(root / row["path"], allow_pickle=False) as original:
            request["flow/noise"] = original["gate_probe/request_noise"].copy()
            request[BIAS_KEY] = original["gate_probe/request_bias"].copy()
            request[CAPTURE_KEY] = True
            result["extra_model_calls_attempted"] += 1
            response, resource = infer_isolated(LogitPolicy(wrapped.policy), request)
            result["extra_model_calls_completed"] += 1
            assert_equal(response, original, FULL_KEYS, "Logit diagnostic changed the original candidate")
        arrays = {k: v for k, v in response.items() if isinstance(v, np.ndarray)}
        arrays["gate_probe/request_bias"] = request[BIAS_KEY]
        arrays["gate_probe/request_noise"] = request["flow/noise"]
        path = directory / "response.npz"
        np.savez_compressed(path, **arrays)
        native = arrays["diagnostic/native_logits_fp32"].astype(float)
        effective = arrays["diagnostic/effective_logits_fp32"].astype(float)
        bias = request[BIAS_KEY].astype(float)
        result.update(passed=True, original_candidate_full_output_exact=True, resource=resource,
                      raw_response_sha256=sha256_file(path), gpu_exact_logit_softmax=response["diagnostic/gpu_exact_logit_softmax"],
                      logit_dtypes=response["diagnostic/logit_dtypes"], probability_dtypes=response["diagnostic/probability_dtypes"],
                      max_requested_vs_effective_logit_delta=float(np.max(np.abs(effective - native - bias))),
                      new_environment_actions=0)
    except BaseException as error:
        result.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        save_json(directory / "result.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
