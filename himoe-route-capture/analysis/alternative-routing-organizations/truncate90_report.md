# Terminal-truncation audit

All views are rebuilt on ten relative-phase anchors spanning 0.50 to 0.90. The last 10% of each rollout is absent.

| representation | method | K | noise | failures in noise | outcome excess | task NMI | length NMI | ARI to full |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| peer_rank | hdbscan | 2 | 0.864 | 233/307 | 0.023 | 0.018 | 0.039 | 0.593 |
| peer_rank | louvain | 9 | 0.000 | 0/307 | 0.013 | 0.079 | 0.136 | 0.266 |
| lag_spectrum | hdbscan | 3 | 0.232 | 306/307 | 0.141 | 0.611 | 0.535 | 0.059 |
| lag_spectrum | louvain | 19 | 0.000 | 0/307 | 0.087 | 0.667 | 0.776 | 0.807 |
| path_signature | hdbscan | 0 | 1.000 | 307/307 | 0.000 | 0.000 | 0.000 | 1.000 |
| path_signature | louvain | 7 | 0.000 | 0/307 | 0.030 | 0.365 | 0.293 | 0.231 |
| route_topology | hdbscan | 7 | 0.311 | 288/307 | 0.063 | 0.554 | 0.555 | 0.414 |
| route_topology | louvain | 49 | 0.000 | 0/307 | 0.048 | 0.583 | 0.675 | 0.305 |
| layer_wave | hdbscan | 0 | 1.000 | 307/307 | 0.000 | 0.000 | 0.000 | 1.000 |
| layer_wave | louvain | 9 | 0.000 | 0/307 | 0.023 | 0.567 | 0.414 | 0.570 |

## Reading

- Lag-spectrum still places almost every failure outside its dense clusters after the terminal segment is removed.
- Peer-rank retains two small cross-task density islands, but most episodes remain noise and outcome effect size stays below 0.05.
- Path-signature and layer-wave still have no HDBSCAN density cluster at the frozen scale.
- Truncation reduces endpoint leakage; it does not equalize the physical meaning of relative phase across completion and timeout trajectories.
