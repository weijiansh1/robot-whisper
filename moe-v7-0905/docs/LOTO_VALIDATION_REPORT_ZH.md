# v7 内在保护器的 Leave-One-Task-Out 验证

日期：2026-09-05
配套报告：`LOSO_VALIDATION_REPORT_ZH.md`（未见 suite）

## 1. 结论

留出单个任务后，pooled 结果为 **444 TP / 81 FP、78.72% recall、84.57% precision**（loto_l1），
与全语料的 439/80、77.84%、84.59% 实质无差别。重选 operating point 的 loto_l2 为
420/80、74.47%、84.00%，略差于不重选。

**未见任务本身不是问题。** 把这一结论和 LOSO 并排看，才是本轮两个实验的真正发现：

| 实验 | 移除的语料 | 是否改变 horizon 配方 | freeze 阈值漂移 std | pooled recall | pooled precision |
|---|---|:---:|---:|---:|---:|
| published | 无 | — | — | 77.84% | 84.59% |
| **LOTO** | 1 个任务（2.5%） | 否 | **2.95%** | 78.72% | 84.57% |
| **LOSO** | 1 个 suite（25%） | 是 | **18.68%** | 73.40% | 78.26% |

伤害 v7 的不是"没见过这个任务"，而是"校准语料的构成变了"。去掉一个任务，剩下 39 个任务的
horizon 混合比例几乎不动，pooled trajectory-peak 分位数因此几乎不动；去掉一个 suite，长短轨迹
的比例整体改变，判定线随之摆动 44.5 pp。

这把 LOSO 报告 §8 的结论收得更紧：需要重构的不是"任务无关性"，而是**判定线对语料配方的依赖**。

## 2. 设置

40 折，每折留出一个 reference 任务的 400 条轨迹：

- 校准 reference：15,600 条（16,000 − 400），每折固定
- development 标签：14,400 条（若留出任务在 main）或 14,800 条（若在 extra 的 3 个 long 任务中）
- 评估：该任务在 external 的 400 条

**1 折无外部回放**：`libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket`
在 reference 的 40 个任务里，但不在 external 的 39 个里。该折照常产出 profile 与漂移记录，
但不参与评分，已记入 `loto_manifest.json` 的 `folds_without_external_replay`。
**0 折 infeasible**——640 点网格在每一折都有满足预注册约束的候选。

LOTO 的自然估计量是 pooled：把 39 折各自的 400 条留出预测拼回一个 15,600 的向量，每条 episode
都由一个没见过它任务的 profile 打分。这个数与 v7 的 439/80 直接可比。

## 3. 结果

### 3.1 pooled（15,600 episodes，每条都由未见其任务的 profile 打分）

| 级别 | TP / FP | risk recall | early-4 | precision | timely FPR |
|---|---:|---:|---:|---:|---:|
| published | 439 / 80 | 77.84% | 58.69% | 84.59% | 0.532% |
| **loto_l1** | **444 / 81** | **78.72%** | **59.75%** | **84.57%** | **0.539%** |
| loto_l2 | 420 / 80 | 74.47% | 56.03% | 84.00% | 0.532% |

`published` 行复现 v7 报告的 439/80、77.84%、84.59%、58.69%，逐位一致。

`loto_l1` 的 recall 比全语料**略高**（+0.89 pp）。原因是把留出任务的 400 条从 reference 拿掉，
它们对上分位数的贡献一并消失，阈值略微下移，报警略微更容易触发。这是 2.5% 语料扰动的量级，
不构成任何实质差异。

`loto_l2` 反而比 `loto_l1` 差 4.25 pp。每折在 39/40 个任务的 development outcomes 上重选
operating point，选出的点在留出任务上不如直接沿用已发布的分位数——这是 operating-point 选择
轻度过拟合 development 的证据，与 LOSO 中"重选只找回 0.64 pp"的观察方向一致。

### 3.2 macro over 39 tasks

| 级别 | risk recall | precision | timely FPR |
|---|---:|---:|---:|
| published | 72.30% | 78.14% | 0.583% |
| loto_l1 | 72.33% | 80.70% | 0.601% |
| loto_l2 | 71.04% | 80.65% | 0.587% |

单任务的 risk n 普遍在个位到几十，逐任务点估计噪声极大，macro 仅作参考，主结果以 pooled 为准。

## 4. 阈值漂移：LOTO vs LOSO

`l1` 级（分位数水平锁死，只换语料切片）四个常量的相对漂移：

| 常量 | LOTO min / max / std | LOSO min / max / std |
|---|---|---|
| freeze_threshold | −12.18% / +2.98% / **2.95%** | −19.29% / +25.17% / **18.68%** |
| acceleration_threshold | −4.26% / +1.44% / **2.06%** | −27.21% / +17.30% / **20.82%** |
| periodicity_threshold | −3.09% / +2.29% / 1.47% | −8.60% / +12.27% / 9.04% |
| periodicity_scale | −0.64% / +0.75% / 0.42% | −5.18% / +3.71% / 3.83% |

LOSO 的漂移标准差是 LOTO 的 **6-10 倍**。freeze 与 acceleration 两个阈值在 LOTO 下基本不动
（std < 3%），在 LOSO 下摆动近 20%。

LOTO 里 freeze 的 min 为 −12.18%，是唯一稍大的单折——它来自留出某个长任务时，该任务贡献的高
peak 一次性消失。即便如此也只有 LOSO 极值的一半。

## 5. 限制

- external 在 v3-v6 中已被反复查看，LOTO 同样不能把它变成 pristine holdout。
- LOTO 只覆盖"未见任务、已见 suite"。跨 suite 见 `LOSO_VALIDATION_REPORT_ZH.md`；
  跨模型 / 跨本体仍不在范围内。
- 逐任务 risk n 太小（个位到几十），`loto_metrics_by_task.csv` 的单任务数字不应单独引用。
- `loto_l2` 的 operating point 仍由 development outcomes 选出，只是标签不再来自留出任务。
  三个级别下阈值**数值**都是无标签 order statistic。
- 40 折共用同一批 reference 轨迹，折与折之间校准集重叠 97.5%，因此各折结果高度相关，
  不能当作 40 个独立重复。

## 6. 对实验 B 的影响

方向不变但目标更明确：B 要消除的是**判定线对语料配方的依赖**，不是"对任务身份的依赖"——后者
v7 已经解决了（L0），而且 LOTO 证明它解决得很好。

具体地，freeze 与 acceleration 的 pooled trajectory-peak 分位数是长度敏感的：语料里长轨迹占比
一变，切点就跟着变（LOSO §4 已给出单调对应关系）。B 应把这两个切点换成对轨迹长度自适应、
且不含任何语料常数的**前缀内秩统计量**。`periodicity_scale` 在 LOTO 和 LOSO 下都是最稳的
（std 0.42% / 3.83%），优先级最低。

## 7. 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-v7-0905
python experiments/evaluate_loto_task.py
```

40 折约 24 秒，纯 CPU。产物在 `results/loto_validation/`：
`loto_metrics.csv`、`loto_metrics_by_task.csv`、`threshold_drift.csv`、`fold_selection.json`、
`fold_profiles.npz`、`pooled_first_alarms.npz`、`loto_manifest.json`、`loto_summary.json`。
