# cache_new frozen confirmation

Primary family: 2 signals x 2 event definitions x 2 leads; 5000 group sign flips, one maxT family.

## Admission

- Episodes: 800; initial states: 50; seeds: 16.
- Success: 511/800; Trap/static proxy events: 263/151.
- This is an untouched holdout relative to the A/B feature and lead selection; its onset remains a query proxy.

## Primary F0-residual results

| event | signal | lead | direction | det AUC | 95% CI of event-high AUC | point p | maxT p |
|---|---|---:|---|---:|---:|---:|---:|
| static | `query_exec_churn` | -4 | event_low | 0.785 | [0.173, 0.263] | 0.0002 | 0.0002 |
| static | `query_support_churn` | -4 | event_low | 0.786 | [0.172, 0.259] | 0.0002 | 0.0002 |
| static | `query_exec_churn` | -2 | event_low | 0.726 | [0.232, 0.316] | 0.0002 | 0.0002 |
| static | `query_support_churn` | -2 | event_low | 0.728 | [0.230, 0.315] | 0.0002 | 0.0002 |
| trap | `query_exec_churn` | -4 | event_low | 0.717 | [0.222, 0.349] | 0.0002 | 0.0002 |
| trap | `query_support_churn` | -4 | event_low | 0.715 | [0.223, 0.354] | 0.0002 | 0.0002 |
| trap | `query_exec_churn` | -2 | event_low | 0.676 | [0.261, 0.395] | 0.0002 | 0.0050 |
| trap | `query_support_churn` | -2 | event_low | 0.676 | [0.261, 0.390] | 0.0002 | 0.0050 |

## Decision

Confirmed event-low cells: 8/8.
A confirmed result means actual cross-query execution support freezes before the event even after removing the two old F0 loop signals. It remains observational and does not show that forcing churn, rerouting, or truncating a chunk improves behavior.
