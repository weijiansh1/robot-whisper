"""Exact, position-aligned routing distances and success-reference kNN."""

from pathlib import Path
import sys

import numpy as np
from numba import njit, prange

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "boundary_knn"))
from knn import ROOT, ALPHAS
from core import conformal_threshold, first_alarm, trajectory_peak

METHODS = ("route_ja_k20", "route_wj_site_k20", "route_wj_aligned_k20", "route_hellinger_aligned_k20")


def prepare(probabilities, expert_ids):
    p = np.asarray(probabilities, dtype=np.float32)
    ids = np.asarray(expert_ids)
    if p.ndim != 3 or p.shape[1:] != (80, 32) or ids.shape != (*p.shape[:2], 4):
        raise ValueError("expected [query,80,32] probabilities and [query,80,4] expert IDs")
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
        raise ValueError("probabilities must be finite, nonnegative and have positive site mass")
    if not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (ids >= 32).any():
        raise ValueError("expert IDs must be integers in [0,32)")
    if (np.diff(np.sort(ids, axis=-1), axis=-1) == 0).any():
        raise ValueError("Top-4 IDs must be distinct at every site")
    p = np.ascontiguousarray(p / p.sum(-1, keepdims=True))
    masks = np.bitwise_or.reduce(np.left_shift(np.uint32(1), ids.astype(np.uint32)), axis=-1)
    return p, np.sqrt(p), p.sum(-1, dtype=np.float64), np.ascontiguousarray(masks)


@njit(inline="always")
def popcount32(value):
    value = value - ((value >> 1) & 0x55555555)
    value = (value & 0x33333333) + ((value >> 2) & 0x33333333)
    value = (value + (value >> 4)) & 0x0F0F0F0F
    return ((value * 0x01010101) >> 24) & 0xFF


@njit(parallel=True, cache=True, fastmath=True)
def distance_matrix(p, root, mass, masks, bank_p, bank_root, bank_mass, bank_masks):
    result = np.empty((len(p), len(bank_p), 4), dtype=np.float64)
    for i in prange(len(p)):
        for j in range(len(bank_p)):
            ja, wj_site, total_l1, total_mass, squared_root = 0., 0., 0., 0., 0.
            for site in range(80):
                intersection = popcount32(masks[i, site] & bank_masks[j, site])
                ja += 1. - intersection / (8. - intersection)
                l1 = 0.
                for expert in range(32):
                    l1 += abs(float(p[i, site, expert]) - float(bank_p[j, site, expert]))
                    delta = float(root[i, site, expert]) - float(bank_root[j, site, expert])
                    squared_root += delta * delta
                site_mass = mass[i, site] + bank_mass[j, site]
                # For nonnegative vectors, 1 - sum(min)/sum(max) = 2*L1/(mass+L1).
                wj_site += 2. * l1 / (site_mass + l1)
                total_l1 += l1
                total_mass += site_mass
            result[i, j, 0] = ja / 80.
            result[i, j, 1] = wj_site / 80.
            result[i, j, 2] = 2. * total_l1 / (total_mass + total_l1)
            result[i, j, 3] = np.sqrt(squared_root / 160.)
    return result


class JaccardScorer:
    def __init__(self, reference_probabilities, reference_ids, batch_size=64):
        self.bank = prepare(reference_probabilities, reference_ids)
        if len(self.bank[0]) < 20:
            raise ValueError("at least 20 reference points required")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

    def score(self, probabilities, expert_ids):
        output = np.empty((len(METHODS), len(probabilities)), dtype=np.float32)
        for start in range(0, len(probabilities), self.batch_size):
            stop = min(start + self.batch_size, len(probabilities))
            query = prepare(probabilities[start:stop], expert_ids[start:stop])
            distances = distance_matrix(*query, *self.bank)
            nearest = np.partition(distances, 19, axis=1)[:, :20]
            output[:, start:stop] = nearest.mean(1).T
        return output


def calibrate(scores, labels, frame, test_scores):
    success = np.asarray(labels) == 0
    groups = (frame.loc[success, "task"] + "|" + frame.loc[success, "init_state_id"].astype(str)).to_numpy()
    thresholds = np.empty((2, len(ALPHAS), len(METHODS)))
    first = np.empty((*thresholds.shape, test_scores.shape[1]), np.int16)
    records = []
    for mi, method in enumerate(METHODS):
        peaks = trajectory_peak(scores[mi, success])
        units = np.asarray([peaks[groups == group].max() for group in sorted(set(groups))])
        for ki, (kind, values) in enumerate((("episode", peaks), ("task_init", units))):
            for ai, alpha in enumerate(ALPHAS):
                threshold, rank = conformal_threshold(values, alpha)
                thresholds[ki, ai, mi] = threshold
                first[ki, ai, mi] = first_alarm(test_scores[mi], threshold)
                records.append({"method": method, "kind": kind, "alpha": alpha, "rank": rank,
                    "calibration_units": len(values), "threshold": threshold,
                    "exceedances": int((values > threshold).sum()),
                    "success_calibration_episodes": int(success.sum())})
    return thresholds, first, records


class JaccardMonitor:
    def __init__(self, profile_path, checkpoint, method="route_wj_site_k20", alpha=.05, grouped=True):
        with np.load(profile_path, allow_pickle=False) as z:
            self.profile = {key: z[key] for key in z.files}
        if str(self.profile["checkpoint"]) != checkpoint:
            raise ValueError("reference checkpoint mismatch")
        self.method_index = list(self.profile["methods"].astype(str)).index(method)
        ai = list(self.profile["alphas"]).index(alpha)
        self.threshold = float(self.profile["thresholds"][int(grouped), ai, self.method_index])
        self.scorer = JaccardScorer(self.profile["reference_probabilities"], self.profile["reference_ids"])
        self.reset()

    def reset(self):
        self.query, self.first_alarm_query = 0, -1

    def update(self, probabilities, expert_ids):
        p, ids = np.asarray(probabilities), np.asarray(expert_ids)
        if p.shape != (8, 10, 11, 32) or ids.shape != (8, 10, 11, 4):
            raise ValueError("expected complete all-flow routing probabilities and Top-4 IDs")
        if self.query >= 52:
            raise ValueError("query exceeds the evaluated 52-query horizon")
        score = np.nan
        if self.query >= 7:
            score = float(self.scorer.score(p[:, 9, 1:].reshape(1, 80, 32),
                ids[:, 9, 1:].reshape(1, 80, 4))[self.method_index, 0])
        trigger = bool(np.isfinite(score) and score > self.threshold)
        if trigger and self.first_alarm_query < 0:
            self.first_alarm_query = self.query
        result = {"query": self.query, "score": score, "threshold": self.threshold,
                  "trigger_now": trigger, "alarm": self.first_alarm_query >= 0,
                  "first_alarm_query": self.first_alarm_query}
        self.query += 1
        return result
