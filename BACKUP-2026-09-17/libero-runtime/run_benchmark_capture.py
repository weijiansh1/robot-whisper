"""run_benchmark.py wrapper that stamps every inference request with an episode id and tag.

The capture server (srv/serve_moe_capture.py) reads `episode_id` and `capture/tag` from the
observation, records HB MoE port tensors for the request and saves them under <out>/<tag>/.
Everything else is identical to run_benchmark.py (same policy, same simulator, same artifacts).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "srv" / "src"))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--capture-tag", required=True)
    args, rest = parser.parse_known_args()

    from himoe_libero_bridge import client as client_module

    original_infer = client_module.PolicyClient.infer
    counter = {"step": 0}

    def stamped_infer(self, observation):
        stamped = dict(observation)
        stamped["episode_id"] = int(args.episode_id)
        stamped["capture/tag"] = str(args.capture_tag)
        stamped["capture/step"] = int(counter["step"])
        counter["step"] += 1
        return original_infer(self, stamped)

    client_module.PolicyClient.infer = stamped_infer
    sys.argv = [str(ROOT / "run_benchmark.py")] + rest
    import runpy
    runpy.run_path(str(ROOT / "run_benchmark.py"), run_name="__main__")


if __name__ == "__main__":
    main()
