# 长度分层重加权：修好了阈值，但需要一个可事前验证的前提

日期：2026-09-05
前置：`LOSO_VALIDATION_REPORT_ZH.md`、`CALIBRATION_VARIANTS_REPORT_ZH.md`

## 1. 结论

变体实验证明 v7 的 pooled trajectory-peak 切点**不能被替换**——它同时承担多重检验校正、绝对
尺度参照、非存活条件基准三件工作。本实验因此不动切点的形状，只改语料的**有效构成**：把参考
轨迹按长度重加权，使其 rollout 长度直方图匹配目标分布，再取同样的峰值分位数。在线监视器完全
不变，仍然是每流一个标量、仍然只读 `hb_router_probs`。

结果分两层，必须分开说：

**阈值层面：有效。** 相对同分布 oracle（用留出 suite 自己的 4,000 条参考校准），重加权平均补回
**freeze 86.2%、acceleration 69.1%** 的阈值差距，四折全部为正。这与 LOSO 测得的长度敏感度
完全吻合：这两个流的跨折漂移 std 是 18.7% 和 20.8%。periodicity 的漂移只有 9.0%，按长度重加权
对它是纯粹加噪声（补回率 −675.6%），因此增设 `loso_reweighted_fa` 只对前两个流加权。

**指标层面：三折成功，一折崩溃，聚合数字被拖垮。**

| 折 | ESS | recall: l1 → rw → oracle | early-4: l1 → rw → oracle |
|---|---:|---|---|
| libero_goal | 2,801 | 0.576 → **0.859** → 0.840 | 0.406 → **0.793** → 0.755 |
| libero_spatial | 2,451 | 0.629 → **0.893** → 0.900 | 0.129 → **0.343** → 0.393 |
| libero_object | 1,974 | 0.386 → **0.477** → 0.477 | 0.296 → **0.432** → 0.432 |
| libero_long | **381** | 0.902 → 0.248 → 0.526 | 0.890 → 0.223 → 0.496 |

在有长度支撑的三折上，重加权**几乎精确复现同分布 oracle**——`libero_object` 的 recall 与
early-4 与 oracle 完全相同，`libero_spatial` 差 0.007。early-4 recall 相对 `loso_l1` 提升
2–3 倍。这直接证实了 LOSO §4 的机制判断：判定线确实由校准语料的长度构成决定，把长度构成对齐
就能把线拉回正确位置。

`libero_long` 崩溃，原因可以事前诊断：complement 语料的长度 p10/50/90 是 9/12/15，目标是
21/25/39；**80.3% 的目标长度在参考语料中不足 50 条**，30.4% 完全没有支撑，有效样本量从 12,000
塌到 381。在 97.5 分位的尾部用 381 个有效样本估计，结果不可用（periodicity 阈值 1.061，
oracle 0.348，差 3 倍；`_fa` 版把它排除后 long 的 recall 从 0.248 回到 0.358，仍远低于 oracle
的 0.526）。

## 2. 最有用的产出：一个事前可验证的部署前提

`length_weights` 返回的三个诊断量**只用无标签的 rollout 长度就能算，不需要 outcome，也不需要
跑完评估**：

```
effective_sample_size   加权后的有效样本量
uncovered_target_mass   目标长度分布中参考语料完全没有支撑的质量
clipped_target_mass     被权重上限截断的质量
```

四折实测：

| 折 | ESS | 未覆盖质量 | 截断质量 | 重加权是否可用 |
|---|---:|---:|---:|:---:|
| libero_goal | 2,801 | 0.000 | 0.031 | 是 |
| libero_spatial | 2,451 | 0.000 | 0.143 | 是 |
| libero_object | 1,974 | 0.000 | 0.000 | 是 |
| libero_long | 381 | **0.304** | **0.472** | 否 |

**这三个数正确地、事前地标出了唯一失败的那一折。** 因此可以写成部署规则：

> 迁移到新任务分布前，先用少量无标签 rollout 估计其长度直方图，计算 ESS 与未覆盖质量。
> ESS 充足且未覆盖质量接近 0 时，长度重加权可以把校准拉到接近同分布水平；否则参考语料在该
> horizon 区间根本没有样本，任何重加权都无法弥补，必须在新长度区间补采参考轨迹。

## 3. 一个反直觉的对照：同分布 oracle 并不一致地更好

`oracle_same` 用留出 suite 自己的 4,000 条参考校准，是"如果有同分布参考"的上界。它的 micro
结果是 380 TP / 102 FP、67.38% recall、78.84% precision，对比 `loso_l1` 的
413/125、73.23%、76.77%：

- precision +2.07 pp、timely FPR −0.153 pp、macro early-4 +8.89 pp：**更好**
- recall −5.85 pp：**更差**

`loso_l1` 在 `libero_long` 上 0.902 的高 recall 不是本事，是阈值被短 horizon 语料压得过低的
副产物（同折 precision 0.675、FPR 3.19%）。校准正确不等于指标好看。

`libero_object` 是另一个方向的例子：它的 risk 患病率只有 44/3,600 = 1.2%，此时**过保守的阈值
反而给出更好的 precision**。published（全语料，含 long，阈值高）precision 0.759；oracle_same
（正确校准）只有 0.284；重加权 0.071。三者 recall 分别是 0.500 / 0.477 / 0.477。低患病率下
precision 对阈值极度敏感，补回 73% 的阈值差距仍会把 FP 从 53 变成 275——**指标在阈值空间里
不是 Lipschitz 的**，这也是为什么"阈值补回 86%"没有翻译成"指标改善 86%"。

## 4. 聚合数字

| 级别 | macro recall | macro precision | macro early-4 | micro TP/FP | micro recall | micro precision |
|---|---:|---:|---:|---:|---:|---:|
| published | 70.54% | 87.30% | 49.05% | 439 / 80 | 77.84% | 84.59% |
| loso_l1 | 62.30% | 87.16% | 43.00% | 413 / 125 | 73.23% | 76.77% |
| loso_reweighted | 61.92% | 67.73% | 44.74% | 305 / 320 | 54.08% | 48.80% |
| loso_reweighted_fa | 63.48% | 66.26% | 45.79% | 330 / 327 | 58.51% | 50.23% |
| oracle_same | 68.56% | 72.60% | **51.89%** | 380 / 102 | 67.38% | 78.84% |

**聚合上重加权是负结果**，被 `libero_long` 的崩溃和 `libero_object` 的低患病率放大效应拖垮。
只看这张表会得出"重加权没用"的错误结论；§1 的逐折表和 §2 的诊断才是可用的部分。

`_fa`（只对 freeze 与 acceleration 加权）比全加权好一点（micro recall +4.43 pp），主要来自修好
long 的 periodicity 阈值，但不足以救回该折。两个版本都保留在产物中。

## 5. 有效性边界

- **本实验的声明弱于 LOSO。** 目标长度直方图取自留出 suite 的 external rollout。长度不含
  outcome，但确实是留出 cohort 的信息。`reweight_manifest.json` 中
  `held_out_lengths_used: true`、`held_out_outcomes_used: false` 显式声明了这一点。实际部署时
  这个直方图应来自新分布上的少量无标签试跑；脚本的 `--target-sample N` 可以模拟只用 N 条估计。
- `oracle_same` 只有 4,000 条参考，比 complement 的 12,000 少，其阈值本身也更noisy，作为上界
  是保守的。
- 权重上限固定为 20。`libero_spatial` 有 14.3%、`libero_long` 有 47.2% 的目标质量被截断，
  说明上限在起作用；未做上限敏感性扫描。
- 仍然只有 4 折，且与 LOSO 共用同一批数据，结论不能当作独立验证。
- external 在 v3-v6 中已被反复查看，不是 pristine holdout。

## 6. 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-v7-0905
python experiments/evaluate_length_reweighting.py
# 只用 40 条无标签 rollout 估计目标长度分布：
python experiments/evaluate_length_reweighting.py --target-sample 40
```

约 6 秒，纯 CPU。产物在 `results/length_reweighting/`：`reweight_summary.csv`、
`reweight_fold_metrics.csv`、`fold_thresholds.csv`、`sealed_first_alarms.npz`、
`reweight_manifest.json`（含四折的权重诊断）。
