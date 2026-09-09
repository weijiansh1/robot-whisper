# Legacy K32 MoE routing-consensus audit

This is a retrospective endpoint-proxy audit, not confirmatory evidence of an action-value selector.

## Data and estimand

- `20` exact snapshots x `32` candidates = `640` candidate chunks.
- Each selector chooses one candidate from the complete K32 pool. The random expectation is the exact within-snapshot candidate mean.
- Every reported contrast is reduced within snapshot first, then bootstrapped with snapshots receiving equal weight.
- A positive contrast means a larger raw proxy. Only `drawer_progress_after_continuation` and `success_in_window` have a declared higher-is-better direction.

## Full-pool selectors

The table reports `selected - random expectation`; parentheses contain the snapshot-bootstrap 95% interval.

| selector | chunk drawer delta | continuation drawer progress | success | object qpos moved |
|---|---:|---:|---:|---:|
| action medoid | -0.000000 ([-0.000002, +0.000001]) | +0.003171 ([-0.001879, +0.010248]) | +0.014063 ([+0.000000, +0.042188]) | +0.000856 ([+0.000018, +0.001746]) |
| full HB probability | -0.000034 ([-0.000099, +0.000000]) | +0.002457 ([-0.001274, +0.006533]) | +0.014063 ([+0.000000, +0.042188]) | -0.000245 ([-0.001458, +0.000919]) |
| top-4 expert ID | -0.000034 ([-0.000099, +0.000000]) | +0.001117 ([-0.002645, +0.005115]) | +0.014063 ([+0.000000, +0.042188]) | +0.000369 ([-0.000714, +0.001444]) |

The full HB probability medoid and action medoid choose the same candidate in `10.0%` of snapshots. The expert-ID and full-probability medoids agree in `40.0%`.

## Denoising-prefix route medoids

| completed denoise rounds | matches full route medoid | matches action medoid | drawer progress minus random | success minus random |
|---:|---:|---:|---:|---:|
| 1 | 50.0% | 5.0% | +0.000408 | +0.014063 |
| 2 | 55.0% | 5.0% | +0.000408 | +0.014063 |
| 3 | 60.0% | 5.0% | +0.000408 | +0.014063 |
| 4 | 65.0% | 5.0% | +0.000408 | +0.014063 |
| 5 | 70.0% | 5.0% | +0.000448 | +0.014063 |
| 6 | 70.0% | 5.0% | +0.000448 | +0.014063 |
| 7 | 80.0% | 5.0% | +0.003303 | +0.014063 |
| 8 | 75.0% | 5.0% | +0.003273 | +0.014063 |
| 9 | 85.0% | 10.0% | +0.002507 | +0.014063 |
| 10 | 100.0% | 10.0% | +0.002457 | +0.014063 |

## Outcome support

| proxy | mixed snapshots | degenerate snapshots |
|---|---:|---:|
| `drawer_delta_chunk` | 4 | 16 |
| `drawer_delta_after_continuation` | 14 | 6 |
| `drawer_progress_after_continuation` | 14 | 6 |
| `success_in_window` | 1 | 19 |
| `object_qpos_moved` | 20 | 0 |

The all-20-snapshot contrasts above correctly include zero-information snapshots, but this makes the effective outcome support explicit: binary success is mixed in only one snapshot, chunk drawer displacement in only four, and continuation drawer displacement in fourteen. The JSON also reports mixed-only effects; a one-snapshot interval is deliberately unavailable.

## Decision

No MoE routing-consensus advantage is established. The full-probability route medoid's opening-oriented continuation effect versus random is `+0.002457` with interval `[-0.001274, +0.006533]`; its effect relative to the action medoid is `-0.000714` with interval `[-0.004846, +0.002004]`. Both include zero.

The positive success contrast cannot be interpreted inferentially: only one snapshot contains both success and failure candidates, so its mixed-only confidence interval is unavailable. Denoise-prefix results are descriptive and were inspected jointly; no prefix is a separately confirmed selector.

## Interpretation limits

- All 20 snapshots come from one LIBERO-Goal task and five source episodes; there is no held-out task.
- Each candidate has one shared-CRN continuation realization, not repeated-continuation Q.
- The continuation route rows are not selector inputs; only each candidate's initial generation route is used.
- The signed drawer proxies are stage-dependent physical coordinates. Only the explicitly negated drawer-progress proxy has a higher-is-better interpretation.
- Most snapshots are outcome-degenerate for success and chunk drawer motion, so pair count cannot repair the small effective snapshot count.
- This reused exploratory sample cannot establish that routing consensus improves closed-loop task success.
