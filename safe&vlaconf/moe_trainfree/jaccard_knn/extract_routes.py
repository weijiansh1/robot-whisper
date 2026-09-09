"""Cache final-flow aligned routing without reading B success/failure labels."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import zarr

from metrics import HERE, ROOT
from core import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round3_safe")
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round6_jaccard_knn")
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    destination = output / "route_cache"
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(parent / "index.csv")
    previous = {a["source"]: a for a in json.loads((parent / "extraction_audit.json").read_text())}
    audits = []
    for source, part in frame.groupby("source", sort=True):
        start = time.perf_counter()
        filename = hashlib.sha256(source.encode()).hexdigest()[:20] + ".npz"
        target = destination / filename
        run = ROOT / "VLA_MUI_HUB" / source
        expected = {"source": source, "route_final_sha256": previous[source]["route_final_sha256"],
                    "index_sha256": digest(parent / "index.csv"), "extractor_sha256": digest(Path(__file__))}
        if target.exists():
            with np.load(target, allow_pickle=False) as z:
                audit = json.loads(str(z["audit"]))
                assert all(audit.get(k) == v for k, v in expected.items()), source
            audit["cache_sha256"] = digest(target)
            audits.append(audit)
            continue
        store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        lengths, episodes = part.length.to_numpy(), part.episode.to_numpy()
        np.testing.assert_array_equal(store["episode_id"][:], np.repeat(episodes, lengths))
        assert (np.diff(store["control_step"][:]) == 1).all()
        final = np.asarray(store["hb_router_probs"][:, :, 9, :, :])
        assert hashlib.sha256(final.tobytes()).hexdigest() == expected["route_final_sha256"], source
        ids = np.asarray(store["hb_expert_ids"][:, :, 9, 1:, :])
        assert ids.shape == (len(final), 8, 10, 4)
        assert not (np.diff(np.sort(ids, axis=-1), axis=-1) == 0).any()
        assert ids.max() < 32
        queries = np.concatenate([np.arange(length) for length in lengths])
        rows = np.repeat(part.index.to_numpy(), lengths)
        valid = queries >= 7
        probabilities = final[valid, :, 1:].reshape(-1, 80, 32)
        ids = ids[valid].reshape(-1, 80, 4)
        selected = np.take_along_axis(probabilities, ids.astype(int), axis=-1)
        top_values = np.partition(probabilities, -4, axis=-1)[..., -4:]
        np.testing.assert_array_equal(np.sort(selected, axis=-1), np.sort(top_values, axis=-1))
        audit = dict(expected, output=filename, episodes=len(part), points=int(valid.sum()),
            reference_query_min=7, top4_selected_values_verified=True,
            expert_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
        np.savez_compressed(target, probabilities=probabilities, expert_ids=ids,
            global_rows=rows[valid], queries=queries[valid].astype(np.int16), audit=np.asarray(json.dumps(audit)))
        audit["cache_sha256"] = digest(target)
        audits.append(audit)
        print(f"EXTRACT {len(audits)}/80 {source}: {valid.sum()} points, {time.perf_counter()-start:.1f}s", flush=True)
    write_json(output / "extraction_audit.json", audits)
    print(f"EXTRACTED {sum(a['points'] for a in audits)} query points", flush=True)


if __name__ == "__main__":
    main()
