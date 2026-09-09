# 预注册:expert-activation v2(同 token、异内部——种子干净重做)

冻结时间:2026-08-22(在任何 v2 特征提取或分析运行之前写定)。
脚本:`analyze_expert_activation_v2.py`。本文件此后只追加"偏离记录",不改正文。

## 动机

v1(`analysis/expert-activation-future/long-t08`)检验了重算的专家贡献能否预示首
query 的动作 basin 与整条 rollout 成败。08-22 复审(review 文档 B2/B6/B7)与用户
08-22 提案("同 token、异内部")指出四个缺陷,本次全部修正:

1. **种子模板泄漏**:v1 动作表用 scene-LOSO,而 16 个场景共享 32 个 seed
   (1000–1031),同日晚间 early-action-head 证明 seed 模板 Spearman 0.919、
   种子双开后塌到 0.013。v1 的 +0.015/+0.060/+0.092 增量全部暴露。
2. **token 聚合**:v1 丢掉 state token(token 0),并把 10 个 action token 的特征
   求 mean/std——违反"ruler+gate,永不平均 suffix"。
3. **缺 D**:方向性分歧 Σα‖v−r‖²/Σα‖v‖² 从未计算(只有范数 CV)。
4. **只有距离、没有水平**:提案的机制假设("高张力→脆弱→分叉")预言 pair 的
   D/C/Q **均值水平**有信息,v1 只造了差异型特征。

## 数据(与 v1 逐位相同,除 token 轴)

- t08 right-16x32:16 场景 × 32 seed,首 policy query,296/512 成功。
- 层 {2,5,12,15},全部 10 个 denoise 轮,**全部 11 个 suffix token**(v1 取 1–10)。
- checkpoint `cdc2b21f…`;噪声由 seed 精确重建(`default_rng(seed).standard_normal((10,24))`);
  首 query 动作块 `actions[0]` 按 checkpoint std 归一化。
- 专家输出重算:v_e = E_e(h) 不加权,r = Σ α_e v_e,α = 存储 top-4 概率重归一;
  **存储 ids 为权威**;fp32 CPU 计算(v1 为 bf16,门控复算一致率照常报告)。

## 特征(每 rollout × 层 × 轮 × token)

8 个标量:`input_rms`、`expert_mass`(Σα‖v‖ᵣₘₛ)、`routed_rms`、
**`D = 1 − ‖r‖²/Σα‖v‖²`**(等价 Σα‖v−r‖²/Σα‖v‖²,因 Σα=1)、
**`C = 1 − ‖r‖/Σα‖v‖`**(v1 的 cancellation 取反)、`expert_norm_cv`、
`routed_over_shared`、**`Q = 1 − cos(r,s)`**。
向量:r(routed)、s(shared)逐 token 存 fp16;h 直接取 hidden.zarr。

## pair 特征(场景内,每 层 × 轮 一个格)

- 标量 f:|f_i−f_j| 与 (f_i+f_j)/2,各取 {state token, action token 逐 token 后平均}
  → 每标量 4 个特征。action 池化定义:先逐 token 算,再对 token 1–10 求均值。
- 方向:1−cos 逐 token,取 {state, action 池化} → 每向量族 2 个特征
  (routed / shared / h 三族)。
- router:v1 的 overlap 距离,但**按层**,{state, action 池化} → 2 个特征。
- 噪声基线 **B1(12 维)**:noise7 RMS 距离、noise24 RMS 距离、逐 token noise7
  距离(10)。

## 目标与切分

**主任务**:同 basin pair 标签,v1 定义(场景内动作块 RMS 距离 ≤ 场景中位数)。
**主切分:种子双开 4 折**——seed 排序后连续 8 个一组
(1000–07 / 1008–15 / 1016–23 / 1024–31);训练 pair = 两端都在 24 个训练 seed
(276/场景 ×16=4416);测试 pair = 两端都在留出 8 seed(28/场景 ×16=448)。
模型:LogisticRegression(C=0.1, lbfgs) + StandardScaler。
**统计量**:每折 = 16 个场景内测试 AUC 的均值(某场景单类则记 nan 跳过);
格值 = 4 折均值。
**稳健性(仅真数据)**:场景+种子双重双开(4 场景折 × 4 种子折 = 16 格,
训练 = 训练场景 × 训练 seed pair,测试 = 留出场景 × 留出 seed pair)。

**次任务(纯描述,预承诺不得重开 selector 线、不出任何头条)**:成败,
v1 的按 seed GroupKFold(4) 条件 AUC 协议,模型
{scene, scene+noise, +size, +DCQ, +shared_size, +final_action}(PCA12 同 v1)。

## 通道(每个 = B1 + 通道特征)

| 通道 | 特征 | 维数 |
|---|---|---|
| c0 | B1 单独 | 12 |
| c1 | +router overlap | 2 |
| c2 | +size {expert_mass, routed_rms} | 8 |
| c3 | +DCQ {D, C, Q} | 12 |
| c4 | +routed 方向 | 2 |
| c5 | +shared 方向(对照臂) | 2 |
| c6 | +h 方向(对照臂) | 2 |
| c7 | +全内部 c2∪c3∪c4 | 22 |

另报无拟合的逐场景 direct AUC(与 v1 表可比;v1 的 direct 列无训练故无泄漏,
受泄漏影响的只是 LOSO 增量列)。

## 判定规则(冻结)

- **R1(内部状态在噪声基线之外可见)**:发现族 = {c1,c2,c3,c4,c7} × 4 层 ×
  10 轮 = 200 格。ΔAUC = 格值(通道) − 格值(B1)。零分布:场景内 rollout↔内部
  特征重排(每场景一个置换,统一作用于所有内部数组;噪声/动作/标签不动),
  **200 次置换,max-stat FWER,p<0.05**。
- **R2a(专家特有)**:R1 通过格上,ΔAUC(通道) − ΔAUC(c5 shared) > 0 且场景
  jackknife CI 不含 0。
- **R2b(相对原始 h 的提取优势)**:同上,对照 c6。
- **措辞阶梯**:仅 R1 → "内部计算状态可见,分支通用接口";+R2a → "专家分解
  特有";+R2b → "VQ/分解预提取超过同容量原始 h 读出"。任何结果**不得**表述为
  超出完整 hidden state 的信息(确定性所致,提取差距框架)。
- **止损**:R1 无一通过 → basin 线与 selector 线一并关闭,只剩脆弱性臂(A 臂,
  扰动重放测 F)。

## 冻结预测

1. v1 的 LOSO 增量在种子双开下明显缩水;routed 方向 +0.060 → 预测 < +0.03。
2. shared ≥ routed 续存(分支通用,非专家特有)。
3. c6(原始 h)≳ 所有分解通道。
4. c3 的增量 ≤ 方向通道;若"张力"假设有任何真东西,应出现在 pair 均值
   (水平)特征上——这是唯一真正未测过的格。
5. state token 通道非零(注意力把噪声混进 state token)但晚轮弱于 action token。

## 描述性输出(无 FWER、无结论)

逐 token direct AUC 曲线(11 token × 层 × 轮);匹配子集回声(noise7 距离 ≤
场景中位数的 pair 上,c7 与 B1 的折 AUC);成败次任务表。

## 计算

/opt/conda/bin/python,纯 CPU(禁 GPU 0 与 4g.71gb;本实验不用 GPU),
fp32,torch 48 线程,置换分析 joblib 48 进程,全局 seed=0,置换数 200。

## 偏离记录(冻结后追加,只登记不改正文)

- D1(实现期,分析未跑):特征文件额外存第 9 个标量 `shared_rms`,仅用于次任务
  (成败)的 `+shared_size` 分组(正文"次任务"已冻结该分组);它不进入
  c1–c7 任何 pair 通道,发现族维持 200 格不变。
