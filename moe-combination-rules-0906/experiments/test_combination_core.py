#!/usr/bin/env python3
"""Deterministic checks that the quorum machinery means what it claims.

Run before any selection. Failures here invalidate every downstream number.
"""

from __future__ import annotations

import sys

import numpy as np

import combination_core as core


def check(label: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label}", flush=True)
    if not ok:
        sys.exit(1)


def audit_rows(votes: np.ndarray) -> np.ndarray:
    """Deterministic audit subset: every episode with two or more votes, plus a
    fixed stride sample of the rest."""
    busy = np.flatnonzero((votes >= 0).sum(axis=0) >= 2)
    stride = np.arange(0, votes.shape[1], 37)
    return np.unique(np.concatenate([busy[:400], stride]))


def brute_force(
    votes: np.ndarray,
    k: int,
    window: int,
    frames: int,
    rows: np.ndarray,
    queries: int = 52,
) -> np.ndarray:
    """Independent loop implementation of the predeclared quorum definition."""
    names = core.POOLS["all12"]
    out = np.full(len(rows), -1, dtype=np.int16)
    for episode_position, episode in enumerate(rows):
        column = votes[:, episode]
        for query in range(queries):
            low = -10**9 if window < 0 else query - window
            inside = [
                position
                for position in range(len(column))
                if column[position] >= 0 and low <= column[position] <= query
            ]
            if len(inside) >= k and len({core.FRAMES[names[p]] for p in inside}) >= frames:
                out[episode_position] = query
                break
    return out


def main() -> None:
    cohort = core.Cohort("development_main")

    reference = core.prior_of(
        cohort.alarms["mobility|global|selected"], cohort.suite, cohort.priors
    )
    fast = cohort.prior_of(cohort.alarms["mobility|global|selected"])
    check(
        "vectorised survival prior matches select_early_lock.prior_of",
        np.allclose(reference, fast, equal_nan=True),
    )

    for mode in core.MODES:
        votes = cohort.votes("all12", mode, "selected")
        cache = core.WindowCache(cohort, "all12", mode, "selected")
        check(
            f"k=1 quorum is plain OR ({mode})",
            np.array_equal(
                core.quorum_alarm(cache, 1, -1, 1), core.pool_or(votes)
            ),
        )
        check(
            f"k=1 quorum does not depend on the window ({mode})",
            np.array_equal(
                core.quorum_alarm(cache, 1, 0, 1), core.quorum_alarm(cache, 1, 8, 1)
            ),
        )
        check(
            f"latching k=12 quorum is the AND of all twelve ({mode})",
            np.array_equal(
                core.quorum_alarm(cache, 12, -1, 1),
                np.where(
                    (votes >= 0).all(axis=0), votes.max(axis=0), -1
                ).astype(np.int16),
            ),
        )

        # A two-member pool is not in POOLS, so build the pairwise case by hand.
        names = core.POOLS["all12"]
        left = names.index("mobility")
        right = names.index("expert_load_effective_rank")
        pair = votes[[left, right]]
        grid = core._vote_grid(pair, cohort.queries)
        total = core._window_count(grid, -1).sum(axis=0)
        check(
            f"latching k=2 on a pair is the pairwise AND ({mode})",
            np.array_equal(
                core._first_true(total >= 2),
                core.pairwise_and(pair[0], pair[1]),
            ),
        )

        earliest = core.pool_or(votes)
        rows = audit_rows(votes)
        for window in (0, 1, 2, 4, 8, -1):
            for k, frames in ((2, 1), (2, 2), (3, 1)):
                first = core.quorum_alarm(cache, k, window, frames)
                check(
                    f"vectorised quorum matches the brute force definition "
                    f"k={k} W={window} f={frames} ({mode})",
                    np.array_equal(first[rows], brute_force(votes, k, window, frames, rows)),
                )
                fired = first >= 0
                check(
                    f"quorum never fires before the pool's first vote "
                    f"k={k} W={window} f={frames} ({mode})",
                    bool(np.all(first[fired] >= earliest[fired])),
                )
            if window < 0:
                continue
            first = core.quorum_alarm(cache, 2, window, 1)
            wider = core.quorum_alarm(cache, 2, window + 1 if window < 8 else -1, 1)
            check(
                f"a wider window can only fire earlier or equally early W={window} ({mode})",
                bool(
                    np.all(
                        np.where(wider >= 0, wider, np.iinfo(np.int16).max)
                        <= np.where(first >= 0, first, np.iinfo(np.int16).max)
                    )
                ),
            )

        strict = core.WindowCache(cohort, "all12", mode, "0.99")
        loose = core.WindowCache(cohort, "all12", mode, "0.9")
        check(
            f"cascade with k=1 reduces to the strict quorum ({mode})",
            np.array_equal(
                core.cascade_alarm(strict, loose, 1, -1, 1),
                core.quorum_alarm(strict, 1, -1, 1),
            ),
        )

    votes = cohort.votes("all12", "global", "selected")
    for position in range(votes.shape[0]):
        fired = votes[position] >= 0
        check(
            f"every alarm chunk of pool member {position} is inside its episode",
            bool(np.all(votes[position][fired] < cohort.length[fired])),
        )

    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
