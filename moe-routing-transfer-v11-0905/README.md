# HiMoE-VLA Routing Transfer Field

This experiment tests a train-free mechanism hypothesis: early HB-MoE routing suppresses
noise-specific variation, while late HB-MoE routing organizes token-relative structure.
Unlike marginal graph summaries, it models the coupled front-to-back transformation over
both flow steps and closed-loop queries.

The feature builder never loads outcomes. Event labels are used only by the separate matched
evaluation script.

The main conclusions, exact AUCs, negative results, and reproduction commands are in
[`REPORT_ZH.md`](REPORT_ZH.md). The key distinction is between a robust population-level
front-contraction/back-differentiation law and the query-local transfer instability that is
actually useful for trap recognition.
