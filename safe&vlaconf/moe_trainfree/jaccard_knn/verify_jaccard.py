"""Verify saved routing scores with direct formulas and raw-query replay."""

import argparse
import json
from pathlib import Path
import time

import numba
import numpy as np
import pandas as pd
import zarr

from metrics import HERE, ROOT, METHODS, JaccardMonitor
from core import digest, write_json
from run_jaccard import load


def raw_episode(frame, row):
    record = frame.iloc[row]
    part = frame.loc[frame.source == record.source]
    position = int(np.flatnonzero(part.index.to_numpy() == row)[0])
    begin = int(part.iloc[:position].length.sum())
    end = begin + int(record.length)
    store = zarr.open_group(str(ROOT / "VLA_MUI_HUB" / record.source / "server/routes.zarr"), mode="r")
    np.testing.assert_array_equal(store["episode_id"][begin:end], np.repeat(record.episode, record.length))
    return np.asarray(store["hb_router_probs"][begin:end]), np.asarray(store["hb_expert_ids"][begin:end])


def independent_scores(probability, ids, profile):
    query = probability.astype(np.float32)
    query = query / query.sum(-1, keepdims=True)
    bank = profile["reference_probabilities"].astype(np.float32)
    bank = bank / bank.sum(-1, keepdims=True)
    query_mask = np.zeros((80,32), bool)
    bank_mask = np.zeros(bank.shape, bool)
    np.put_along_axis(query_mask, ids.astype(int), True, axis=-1)
    np.put_along_axis(bank_mask, profile["reference_ids"].astype(int), True, axis=-1)
    intersection = (bank_mask & query_mask).sum(-1)
    union = (bank_mask | query_mask).sum(-1)
    ja = (1 - intersection / union).mean(1)
    low = np.minimum(bank.astype(float), query.astype(float)).sum(-1)
    high = np.maximum(bank.astype(float), query.astype(float)).sum(-1)
    wj_site = (1 - low / high).mean(1)
    wj_aligned = 1 - low.sum(1) / high.sum(1)
    hellinger = np.sqrt(np.square(np.sqrt(bank).astype(float) - np.sqrt(query).astype(float)).sum((1,2))/160)
    distances = np.column_stack((ja,wj_site,wj_aligned,hellinger))
    return np.sort(distances, axis=0)[:20].mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round6_jaccard_knn")
    parser.add_argument("--baseline", type=Path, default=HERE.parent / "results/round5_knn")
    args = parser.parse_args()
    output, baseline = args.input.resolve(), args.baseline.resolve()
    numba.set_num_threads(8)
    sealed = json.loads((output / "sealed_manifest.json").read_text())
    hashes = 0
    for category, prefix in (("sources",ROOT),("inputs",ROOT),("artifacts",output)):
        for name,expected in sealed[category].items():
            assert digest(prefix/name)==expected,name
            hashes+=1
    frame = pd.read_csv(output / "outcome_alignment.csv")
    oracles, replays, timing = [], [], []
    calibration_checks = 0
    for info in sealed["folds"]:
        fold=info["fold"]
        data=load(output / "predictions" / f"{fold}.npz")
        path=output / "profiles" / f"{fold}.npz"
        profile=load(path)
        old=load(baseline / "profiles" / f"{fold}.npz")
        for key in ("success_global_rows","success_queries"):
            np.testing.assert_array_equal(profile[key],old[key])
        assert not frame.iloc[profile["success_global_rows"]].failure.any()
        np.testing.assert_array_equal(data["calibration_labels"],frame.iloc[data["calibration_rows"]].failure.astype(int))
        np.testing.assert_array_equal(profile["thresholds"],data["thresholds"])
        success=data["calibration_labels"]==0
        cal_frame=frame.iloc[data["calibration_rows"]].loc[success]
        groups=(cal_frame.task+"|"+cal_frame.init_state_id.astype(str)).to_numpy()
        for mi,method in enumerate(METHODS):
            cal=data["calibration_scores"][mi,success]
            peaks=np.where(np.isfinite(cal),cal,-np.inf).max(1)
            grouped=np.array([peaks[groups==key].max() for key in sorted(set(groups))])
            for ki,units in enumerate((peaks,grouped)):
                for ai,alpha in enumerate(data["alphas"]):
                    rank=int(np.ceil((len(units)+1)*(1-alpha)))
                    threshold=np.sort(units)[rank-1] if rank<=len(units) else np.inf
                    np.testing.assert_equal(data["thresholds"][ki,ai,mi],threshold)
                    crossed=np.isfinite(data["scores"][mi]) & (data["scores"][mi]>threshold)
                    first=np.where(crossed.any(1),crossed.argmax(1),-1)
                    np.testing.assert_array_equal(first,data["first"][ki,ai,mi])
                    calibration_checks+=1
        for kind in ("calibration","test"):
            rows=data[f"{kind}_rows"]
            for selection,position in enumerate((0,len(rows)//2,len(rows)-1)):
                row=int(rows[position])
                length=int(frame.iloc[row].length)
                if length<=7:
                    continue
                raw,ids=raw_episode(frame,row)
                query=(7,max(7,length//2),length-1)[selection]
                expected=independent_scores(raw[query,:,9,1:].reshape(80,32),ids[query,:,9,1:].reshape(80,4),profile)
                actual=data["calibration_scores" if kind=="calibration" else "scores"][:,position,query]
                np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-7)
                oracles.append({"fold":fold,"kind":kind,"global_row":row,"query":query,
                    "methods":len(METHODS),"max_abs_error":float(np.max(np.abs(actual-expected)))})
                if kind=="test":
                    monitor=JaccardMonitor(path,str(profile["checkpoint"]))
                    observed=[]
                    for q in range(length):
                        started=time.perf_counter()
                        result=monitor.update(raw[q],ids[q])
                        if q>=7:
                            timing.append(1000*(time.perf_counter()-started))
                        observed.append(result["score"])
                    mi=METHODS.index("route_wj_site_k20")
                    np.testing.assert_allclose(observed,data["scores"][mi,position,:length],rtol=2e-6,atol=2e-7,equal_nan=True)
                    ai=list(data["alphas"]).index(.05)
                    assert result["first_alarm_query"]==data["first"][1,ai,mi,position]
                    replays.append({"fold":fold,"global_row":row,"queries":length,
                        "method":"route_wj_site_k20","first_alarm_query":result["first_alarm_query"]})
        print(f"VERIFIED {fold}",flush=True)
    write_json(output / "verification.json", {"hash_checks":hashes,"calibration_rank_checks":calibration_checks,
        "direct_distance_oracles":oracles,"raw_stream_replays":replays,
        "all_reference_identities_match_round5":True,"all_alarm_times_match":True,
        "monitor_update_ms_median":float(np.median(timing)),"monitor_update_ms_p95":float(np.quantile(timing,.95)),
        "timed_queries":len(timing),"latency_scope":"CPU 8-thread configuration; computes all four distance methods; excludes VLA and disk",
        "verifier_sha256":digest(Path(__file__)),"methods_sha256":digest(HERE / "metrics.py")})
    print(f"PASS: {hashes} hashes, {calibration_checks} ranks, {4*len(oracles)} direct scores, {len(replays)} raw replays",flush=True)


if __name__ == "__main__":
    main()
