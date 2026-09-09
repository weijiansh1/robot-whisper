# Amendment: superseded before outcome analysis

Time: `2026-08-23T13:22:36Z` (UTC).

The original protocol in this directory incorrectly made future path energy
the primary endpoint and used RMS as the primary amplitude. It is superseded by
[`../raw-internal-activation-correction/PREREGISTRATION.md`](../raw-internal-activation-correction/PREREGISTRATION.md),
which had already been frozen independently.

Before this amendment, the feature-only reconstruction command completed and
wrote `features.npz`. No feature values, metadata values, predictive outcomes,
or raw `m_e` summaries were opened or inspected, and no predictive analysis was
run. The superseded cache schema is rejected by the amended analyzer.

The amended primary observational endpoint is held-out prediction of
`flatten10x7(live7(x10-x1))`. The primary raw magnitude is absolute
`L2(m_e)` in checkpoint units. RMS is reported only as the identity
`L2/sqrt(1024)`. Future path energy is excluded from the claim gate.
