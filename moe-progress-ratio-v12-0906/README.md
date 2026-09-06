# 进度率保护器 v12

判据：`R(q, W) = d(z_q, z_{q-W}) / sum_{k=1..W} d(z_{q-k+1}, z_{q-k}) < r*`，连续确认 K 次。

`R` 是同单位之比，无量纲，因此不需要 episode 基线也不需要语料常数定尺度。
低 `R` 同时覆盖冻结（路径短且净位移小）与兜圈（路径长但净位移小）；
高 `R` 否决慢速成功，那是当前误报的主体。

设计见 `docs/superpowers/specs/2026-09-06-progress-ratio-guard-design.md`。
主指标是相对存活先验的提升倍数，不是 episode 级精确度。

判决与全部结果见 [`REPORT_ZH.md`](REPORT_ZH.md)。复现顺序见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

**作为检测器它两轮预注册都未通过。** 保留下来的是两件别的东西：召回与精确度在本语料上
无法区分任何监视器（纯时钟召回 1.000 @ 0.42% 误报），以及由此得到的可部署组合规则
——时钟支 ∨ v7 relative_freeze @q0.98，external 召回 1.000 / 精度 0.887 / 31% 的失败
提前 110 个环境步。
