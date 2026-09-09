# HiMoE-VLA 的 HB-MoE 路由：三层拆解与关闭结论

日期：2026-08-15
操作点：`checkpoint-right`（正确 wrist layout），LIBERO-Goal / Spatial / Object
代码基线：`himoe-libero-wrist-fix` bridge + upstream `HiMoE-VLA`
论文：arXiv 2512.05693v2（源码在 `/home/jovyan/work/himoe-vla/arXiv-2512.05693v2.tar.gz`，单文件 `example.tex`）

---

## 摘要

本轮回答的问题是：**HiMoE-VLA 的 MoE 路由状态能否作为 action proposal 的选择信号？**

结论是**否**，而且不是"没测出显著性"，是机制上讲不通。把 MoE 的输出

$$y(h)=E_{\text{shared}}(h)+\sum_{e\in S(h)}\tilde p_e(h)\,E_e(h)$$

拆成三层分别测量，三层全部关闭：

| 层 | 问题 | 结论 | 关键数字 |
|---|---|---|---|
| ① $S(h)$ | 选了哪些专家 | **死** | 23.8% 的位点 $\Delta_{4,5}$ 精确为 0；随机换专家仅动作 4% |
| ② $\tilde p_e(h)$ | 权重是多少 | **死** | $M_4=0.140$，仅比均匀的 0.125 高 0.015 |
| ③ $E_e(h)$ | 专家算出了什么 | **专家有差异且不可互换**，但分支太小传不出去 | $\overline{\cos(E_i,E_j)}\approx0.04$；全换四个 $\Delta y_{\text{MoE}}=31.3\%$ 而 $\Delta A=6.0\%$ |

合起来的**准确**表述是：**在这个 checkpoint 上，HB router 处于负载均衡、近并列、对量化敏感的稀疏分配工作区间，不是一个稳定可读的候选级决策变量。**

**不要**写成"稀疏随机特征集成"。专家**不可互换**：把选中的四个全部换成随机四个，MoE 块输出变化 **31.3%**（见 §4.4）。gate 不可读，与它所选的东西无关紧要，是两回事——限制损害传播的是 routed 分支只占残差流约 9%，不是专家彼此等价。

这条机制解释了此前**所有**的 null：step-11 路由移植的"有信息无权威"、routed 分支只占残差流约 9%、以及"路由是镜子不是方向盘"。

---

## 一、先纠正三处前提

### 1.1 这个 checkpoint 是 11 个 suffix token，不是 51

类默认是 `chunk_size=50, n_action_steps=50`（`moevla.py:89-90`），但 LIBERO 训练配置全部显式覆盖为 `MoEVLAConfig(n_action_steps=10)`（`training/config.py:2476` 及另外 5 处）。该值同时决定：

- 噪声张量形状 `(bsize, n_action_steps, max_action_dim)`（`:402`）——**初始噪声就只有 10 行**，不存在生成 50 个再截断
- 注意力掩码长度 `[1] + [0]*(n_action_steps-1)`（`:320`）
- 输出切片 `suffix_out[:, -n_action_steps:]`（`:382`/`:483`）

`chunk_size=50` 仍是默认值，但在推理路径上完全不参与，只出现在 `:134` 的校验与 `:163` 的 `action_delta_indices`（训练数据加载窗口）。

bridge 不设定这个值，它是读出来再断言：`policies.py:460-464` 检查 `n_action_steps == 10`，不符就拒绝服务。

**后果**：chunk = 10、replan = 10，**两次 query 之间零重叠**，所以不存在 old-tail / new-prefix 一致性这个研究对象。要造出重叠必须显式设 `replan_steps < 10`，那是一个新的控制干预，不是当前默认机制。

早剪枝的相对算力公式**不受影响**，它只依赖 $T=10$：

$$C_{\text{rel}}=\frac{sK+(T-s)M}{TK},\qquad K=16,M=4,s=4\ \Rightarrow\ 0.55$$

受影响的是绝对开销（每次 forward 只有 11 个 suffix token）。

### 1.2 论文从未给动作视野赋值

`example.tex:129-130` 定义 $A_t=[a_t,\ldots,a_{t+H-1}]$，$H$ 在全文再无数值。`horizon` 的全部 7 处出现除此之外都是 "long-horizon"（benchmark 设定）。流积分步数同样未提。

**所以 50 和 10 都只能引代码，不能引论文。**

论文说了并且核对无误的：$N=32$、top-$K=4$（`:217`, `:595`）；24 维统一动作向量 = 8 EEF + 16 关节（`:591`）；AS 在最外层、HB 相邻、中间 dense（`:142`）；$\lambda_{\text{AS}}=0.002$、$\lambda_{\text{HB}}=0.001$。

### 1.3 paper/code 差异：AS-MoE 实际没有 shared expert，且 AS 是 $N{=}3,K{=}1$

论文 `:144`："Each MoE block uses top-$K$ routing over $N$ experts and **includes a shared expert**"。

代码：`ASMoEConfig.n_shared_experts = None`，`HBMoEConfig.n_shared_experts = 1`（`himoe.py:85-112`）。`modeling_moe.py:276` 显式 `if self.config.n_shared_experts is not None` 才相加。

**以 checkpoint 代码为准：AS-MoE 无 shared expert。**

第二处相关歧义：论文实现部分笼统写 $N{=}32, K{=}4$（`:217`, `:595`），但公开代码里那是 **HB** 的配置，**AS 实际是 $N{=}3, K{=}1$**（`ASMoEConfig`）。论文主方法描述没有区分两者的部署配置。

**复现歧义（不作断言）**：论文的 "w/o shared expert" 消融没有说明删除的是哪些层的 shared expert。若严格按公开代码复现，它只能删除 HB 的 shared 分支，因为 AS 根本没有这一支。作者内部版本是否与发布代码相同，公开材料无法确认——这应记为复现歧义，而不是断言。

---

## 二、语料与可分性（正确操作点）

`corpus/libero30-right-v1/`：30 任务 × 50 init state = 1500 集，18,384 控制步，0.78 GB。成功率对表论文 Section 4.1 三套各差 ≤1 集（98.0 / 94.2 / 96.8 vs 97.8 / 94.2 / 96.6）。

**约束**：失败只有 55/1500（3.7%），13 个任务零失败。**这批语料做不了结局预测**——RAD 那条线仍需 `released-left` 批次（28–70% 成功）。

**分片混淆已排除**：采集把 task 0–4 放 2g.35gb、5–9 放 1g.35gb（三套一致），slice 与 task_id 完全共线。`corpus/slice-control-v1/` 在另一分片重跑 goal/t00 与 object/t00，跨片 JS 为 0.0055 / 0.0187，任务间是 0.119——十分之一。（`analyze_slice_control.py`）

**可分性**：加 `--equal-steps 192` 后 within 底线 0.0348 → 0.0425，between/within **3.36 → 2.80**。引用 2.80。

副产品：跨片距离**低于**同任务 split-half 底线，因为跨片比的是同一批 init state。所以那条"噪声底线"主要是 **init-state 方差**。

---

## 三、路由的表观性质

### 3.1 AS-MoE 在单一动作空间内恒定

30 个任务只有 **1 种** AS 路由：层0→e2、层1→e0、层2→e0、层3→e1，全是 1.0/0.0。

AS gate 输入是 24 维 `data_mask`（`ASMoEConfig.condition_dim = 24`），LIBERO 内不变，所以选择不变。

**这不与论文矛盾**：论文的 AS 主张（`:795`）是跨 CALVIN 关节角 / CALVIN EEF / LIBERO EEF 三个 **source** 聚类，即跨动作空间。测试它需要 `data_mask` 变化的部署，不是不同的 LIBERO 任务。

### 3.2 Router audit：均衡是训练目标，不是发现

`router_audit.py`，全部从存储的完整 32 维 softmax 算：

| token | $H_{\text{load}}$ | $H_{\text{token}}$ | $M_4$（均匀 0.125） | $D_{45}$ |
|---|---|---|---|---|
| action，层 0–7 | 0.9997–0.9999 | **0.9989–0.9992** | **0.139–0.141** | 3.1e-4 |
| state，层 0–3 | 0.86–0.98 | **0.83–0.95** | **0.29–0.43** | 3.2e-3–5.1e-3 |
| state，层 4–7 | 0.997–0.998 | 0.988–0.991 | 0.184–0.193 | 1.5e-3 |

论文 `:795` 自己写明 HB-Reg "converges close to its theoretical lower bound (corresponding to near-uniform expert utilization)"。所以均匀是设计目标的达成，实测是确认而非挑战。

**跨 checkpoint 复现**（三套发布的 LIBERO checkpoint，action token）：

| | $H_{\text{token}}$ | $M_4$（均匀 0.125） | ids[...,0] 是真 argmax |
|---|---|---|---|
| goal | 0.9989–0.9992 | 0.139–0.141（1.12×） | 44.7% |
| spatial | 0.9962–0.9974 | 0.150–0.158（**1.27×**） | 39.7% |
| object | 0.9977–0.9984 | 0.145–0.149（1.17×） | 42.5% |

**不是 Goal 单点现象。** spatial 最不简并，$M_4$ 也只有均匀的 1.27 倍。

near-tie 结构同样跨套复现，spatial 约为 goal 的一半强度：$\Delta_{4,5}$ 恰好为 0 的比例 13.6% vs 23.8%，model-vs-argsort 分歧 7.5% vs 13.4%——仍是每 7 个位点就有 1 个的边界完全任意。

**尾部检查（排除 sharp islands）**，240,000 个 action-token 位点：

- $\Pr[M_4>0.2]=\mathbf{0.0000\%}$，每层、每个动作位置都是 0
- $M_4$ 最大值 0.1844
- $H_{\text{token}}$ 的 **P1** = 0.9969；$\Pr[H_{\text{token}}<0.99]=0.0000\%$

**简并不是均值假象，整条分布都平。**

对照 state token：$\Pr[M_4>0.2]=57.7\%$，$H_{\text{token}}$ 的 P1 = 0.683。

### 3.3 结构性死结

routing-cloud 批次（同一真实观测 64 次独立噪声抽样）：

| | 观测值 | 随机基线 | 倍数 |
|---|---:|---:|---:|
| **state token** top-4 集合相同 | 99.41% | 0.00278% | 35748× |
| **action token** top-1 相同 | 34.17% | 3.125% | 11× |
| action top-4 无序集合相同 | 5.46% | 0.00278% | **1963×** |
| action 平均 Jaccard | 0.369 | 0.0667 | 5.5× |

> **唯一路由锐利的 token（state）对候选零判别力；唯一能分出候选的 token（action）路由器近简并。**

state token 的零判别力是**计算图决定的**，不是经验巧合：它只由 `state_proj(state)` 产生（`moevla.py:283`），timestep 只融进 action token（`:306`），注意力掩码让它看得到 prefix 和自己但看不到 action token（`:294`/`:320` + `make_att_2d_masks` 的 cumsum 规则 `:77-78`），而 prefix KV 在去噪前就固定缓存。

实测确认：state token 的路由概率在 **10 个去噪轮上逐位相同**（最大差恰为 0.0，100% 的 (控制步, 层) 对）；action token 为 0%。

同时注意：action-token 路由**主要由观测决定**（换噪声后 top-1 仍有 34% 不变），噪声只改动其中一部分。

---

## 四、三层拆解（主结果）

### 层 ①：选了哪些专家 —— 死

`probe_near_tie.py`。gate 的 `forward` 被逐行镜像替换以截获 `topk` 实际消费的张量；镜像经原 gate 校验（top-4 集合一致），干预的 `none` 臂精确复现基线动作（0.0000e+00）。

**scores 是 bfloat16，且大量打平：**

| | |
|---|---|
| 32 个 score 里的不同取值 | 均值 **23.9**，中位 24，P5 = 19 |
| 4th 位被 >1 个专家共享 | **39.1%** |
| $\Delta_{4,5}$ **恰好为 0** | **23.8%** |
| model topk vs 同一组数的确定性 argsort | 13.4%（纯 tie-breaking） |
| 确定性 bf16 argsort vs 真 fp32 shadow | 13.6%（纯量化） |
| model topk vs fp32 shadow | 14.2% |

平均 8 个专家在 gate 的 dtype 下与别的专家数值不可区分。近四分之一的位点，第 4 与第 5 名 score 字面相等。

**功能性干预**（换掉 4 个选中专家里最弱的那个，880 个位点，权重不动）：

| | max\|ΔA\| | 占 \|A\| |
|---|---:|---:|
| none | 0.0000e+00 | 0.00% |
| nearest（第 5 名） | 6.46e-3 | **3.03%** |
| random（28 个未选中里均匀抽） | 8.68e-3 | **4.08%** |

$$\text{nearest}/\text{random}=0.744$$

换成**完全随机**的另一个专家只动 4%，而"换成打平的那个"仅比随机小 26%——**切割线以下的排序几乎不携带功能**。

### 层 ②：权重 —— 死

$M_4=0.140$ 对均匀的 0.125，仅高 0.015。四个被选中专家的归一化权重近乎相等。

### 层 ③：专家算出了什么 —— 专家有差异，选择无差异

`probe_expert_contribution.py`。`HBMoE.forward` 逐实例镜像以精确分离 routed 与 shared 两支；镜像经动作逐位一致校验。

**分支幅度**（每 token 范数的中位数）：

| block | routed/shared | routed/identity | routed/(routed+shared) |
|---|---:|---:|---:|
| layers 2–5 | 0.40–0.58 | 0.081–0.089 | 0.28–0.37 |
| layers 12–15 | 0.13–0.37 | 0.080–0.112 | 0.12–0.27 |

**引用时必须写清分母**：这是 MoE 块输出的 12–37%，但只是**残差流**的 8–11%。旧报告的"routed 分支占 5.6%"用的是另一个分母，两者不冲突。

**专家多样性**（在采样 token 上跑全部 32 个专家）：

$$\overline{\cos(E_i,E_j)}=+0.029\sim+0.050,\qquad[\text{p5},\text{p95}]\approx[-0.036,+0.134]$$

$d=1024$ 下两个随机单位向量的 cosine 均值为 0、标准差 $1/\sqrt{1024}=0.031$。**32 个专家的输出彼此近乎正交**，只带约 1 个标准差的弱共同分量。它们不是在算同一个函数。

**但选择没有特殊性：**

$$\frac{\|\overline{E}_{\text{selected 4}}-\overline{E}_{\text{all 32}}\|}{\|\overline{E}_{\text{all 32}}\|}\Big/\frac{\|\overline{E}_{\text{random 4}}-\overline{E}_{\text{all 32}}\|}{\|\overline{E}_{\text{all 32}}\|}=\mathbf{1.112}$$

八个 block 全部 ≥ 1.04。采样 token 数提高 3 倍后为 1.112（原 1.119），已收敛。

### 4.4 全组替换：专家不可互换（此处推翻了本报告初稿的强表述）

`probe_expert_swap.py`，4 个观测，四个权重不动、只动身份，同时量三处：

| arm | $\Delta y_{\text{MoE}}$ | $\Delta v$ | $\Delta A$ |
|---|---:|---:|---:|
| none | 0.0000 | 0.0000 | 0.0000 |
| swap1_near（最弱的换成第 5 名） | 0.1139 | 0.0324 | 0.0131 |
| swap1_random | 0.1293 | 0.0276 | 0.0094 |
| swap4_next（全换成 5–8 名） | 0.2972 | 0.0858 | 0.0534 |
| **swap4_random** | **0.3132** | 0.0955 | **0.0598** |

**把选中的四个全部换成随机四个，MoE 块输出变化 31.3%。专家不可互换。** 连"退而求其次换成下四名"也有 29.7%，排序顺序不提供保护。

**这一对数字本身是本轮最强的机制事实**：第 5–8 名在 router score 上只以 bf16 尺度的 margin 落后于 Top-4，但在专家函数空间里**并不比随机专家更接近**。

$$|g_e(h)-g_{e'}(h)|\approx 0 \quad\text{而}\quad \|E_e(h)-E_{e'}(h)\|\gg 0$$

> Near-tied routing scores are not calibrated to counterfactual expert substitutability: replacing the selected experts with ranks 5–8 is nearly as disruptive as replacing them with random experts.

四条证据由此接成一条链：① bf16 尺度的 near-tie 决定谁进 Top-4；② 边界两侧的专家功能上并不相近；③ 因此极小的数值扰动触发不连续的专家函数切换；④ 残差流把扰动衰减（$\Delta y$ 31–50% → $\Delta A$ 6–19%）但没有抹掉。

一句话：**路由边界在数值上极薄，在功能上却很厚**（locally fragile, globally attenuated）。

最终动作只变 6.0%，原因不是专家等价，而是 **routed 分支只占残差流约 9%**。

数量级自洽：专家近正交 ⟹ 全换四个使 routed 分支变约 $\sqrt2$；乘以 routed/(routed+shared) ≈ 0.25 ⟹ 预测 $\Delta y\approx35\%$，实测 31.3%。

**在 spatial checkpoint 上复现（relative L2）：**

| arm | goal $\Delta y$ | goal $\Delta A$ | spatial $\Delta y$ | spatial $\Delta A$ |
|---|---:|---:|---:|---:|
| swap1_random | 12.9% | 0.94% | 18.4% | 0.69% |
| swap4_next | 29.7% | 5.34% | 48.1% | 18.3% |
| **swap4_random** | 31.3% | 5.98% | **50.1%** | **19.2%** |

spatial 的效应**更大**（$\Delta y$ 50.1%，$\Delta A$ 19.2%），结论方向一致。

**指标警告**：`probe_near_tie.py` 报的是 $\max|\Delta A|/\overline{|A|}$，即 chunk 上的最坏单元素除以平均幅度，对离群元素极敏感——它在 spatial 上给出 36.2%，而同一干预的 relative L2 只有 0.69%。**以 relative L2 为准，废弃 max 指标。**

**结论强度分层（这一条要写进任何对外表述）：**

> **强结论**：离散 HB routing 身份与权重不是可用的 rollout 选择信号。
> **不成立**：专家函数随机或功能冗余——已被本节直接证伪。

---

## 五、方法学记录（可复用）

这几条都是本轮踩过并修掉的，复用这套工具时会再遇到。

1. **`torch.topk(..., sorted=False)`**（`modeling_moe.py:82`）——`hb_expert_ids[...,0]` 只有 44.7% 是真 argmax。集合类统计（共选、Jaccard）不受影响；任何需要真 top-1 的量必须从 `hb_router_probs` 重算。
2. **float16 存储不是第二次量化**——float16 有 10 位尾数，多于 bf16 的 7 位，在 $1/32$ 附近无损。存储 id 集合与对存储概率 argsort 的 4.5% 分歧来自 **bf16 下的精确打平**，不是存储精度。
3. **autocast 区域内的 "fp32 shadow" 不是 fp32**——autocast 拦截 `F.linear`，不管输入 cast 成什么。必须 `torch.autocast(..., enabled=False)`。不加的话 shadow 等于部署计算本身并与它 100% 一致，那个 0.00% 就是 bug 自报。
4. **包 `sample_actions` 无效**——`policy.py:41` 在构造时绑定 `self._sample_actions`。`denoise_step` 可包，因为它在 `sample_actions` 里是 `self.denoise_step(...)` 动态查的。
5. **flow trace 必须 clone**——`x_t += dt*v_t` 是原地的，存引用会让 10 份都是最终值。
6. **只抓入口 $x_t$ 会丢最后一次 Euler 更新**——$T$ 次调用给 $x^{(0)}\ldots x^{(T-1)}$，用 $x^{(T-1)}$ 当"最终 chunk" 会让 $\tau=T-1$ 变成自比较（症状：rho 精确 +1.000）。必须同时抓 $v_t$ 并外推 $x^{(T)}=x^{(T-1)}+\Delta t\,v^{(T-1)}$。
7. **Euler 残差不会是 0**——$v_t$ 是 bf16，残差落在 $|dt|\times\text{ulp}_{\text{bf16}}(v)$ 的格点上，约 1e-3，是信号的 0.01%。挂错张量会给出 $O(|x|)$ 的残差，容差要按这个量级设。
8. **recorder 的数组是 `[B,L,D,S,K]`**，不是 `[L,D,S,K]`。
9. **零对照必须内建**——state token 对候选零判别力是从注意力掩码先验推出的，实测 AUC 在全部 10 步精确 0.500。距离函数、配对、AUC、轴序任何一处接错都不会正好 0.500。
10. **`pgrep -f <脚本名>` 会自匹配**，用 PID 文件。

---

## 六、正式关闭的范围

关闭的对象要写得精确：

$$\boxed{\text{AS/HB gate 的 expert ID 或 gate 权重}\ \not\Rightarrow\ \text{候选级 rollout 排序}}$$

具体包括：HB Top-4 expert ID、Top-4 内归一化权重、`ids[...,0]`、route histogram、route-based motion basin、route-based early pruning。

关闭理由不是"样本不够"，而是三类 router 在候选级上遭遇**互补**的结构性失败：

$$\text{AS gate：稳定，但不看 candidate（输入是 data\_mask）}$$
$$\text{HB state gate：锐利，但按计算图不可能看到 candidate noise}$$
$$\text{HB action gate：看 candidate，但近简并且 Top-K 边界数值不稳}$$

**当前 checkpoint 中不存在一个同时满足"随候选变化、局部锐利、数值稳定、领先动作形成"的 routing observable。**

**不能扩大成**："MoE 架构本身没用"。MoE 仍可能通过增加条件容量、减少异质数据负迁移，或形成不可读的分布式计算而有效——§4.4 的 31.3% 正是它在做实事的直接证据。被否定的只是它的 gate state 适合充当 RAD 式 verifier 这一点。

对外可直接引用的表述：

> On the released HiMoE-VLA checkpoint, candidate-level routing signals are structurally unavailable. AS-MoE routing is fixed by the action-space mask and is therefore invariant across sibling action proposals. HB-MoE state-token routing is sharp but likewise invariant to proposal noise, while HB-MoE action-token routing varies across proposals but operates in a near-uniform, quantisation-sensitive regime and fails to predict final action basins. We therefore discontinue routing-based rollout selection. This conclusion concerns the interpretability and selection utility of router identities and probabilities; it does **not** imply that the underlying expert functions are random or interchangeable — replacing all four selected experts moves the MoE block output by 31%.

---

## 七、还活着的两条（都是独立方向，不是原主线的延续）

### 7.1 state 表示驱动的 query 级自适应 $K$

state token 路由锐利（$M_4$ 达 3.5× 均匀）但对候选零判别，所以只能做**预算分配**，不能做候选排序。这是结构结论。

**但它缺一块关键拼图**：多采样本身不产生收益，除非绑定一个**非路由的** downstream selector（action medoid / density / 外部 value model / safety filter / simulator lookahead）。若最后仍固定执行第一条，另外 $K-1$ 条只是浪费算力。

所以要预测的不是单纯的 action dispersion，而是在固定 selector $\mathcal S$ 下的实际边际收益：

$$G_K^{\mathcal S}(s)=V_{\mathcal S,K}(s)-V_{\mathcal S,1}(s)$$

同时报告 $D_K(s)=\operatorname{median}_{i<j}d(A_i,A_j)$（分叉程度）与 $G_K^{\mathcal S}(s)$（可兑现收益）——两者不一定一致：候选很多样但 selector 分不出好坏时，自适应采样只会浪费。

**归因要求**：state routing 是 state hidden state 的确定性投影，必须与原始 proprio、state-token hidden state、VLM prefix pooled、state router logits 在**匹配的降维与 probe 容量**下比较。若不优于普通 hidden state，只能称为 "query-state adaptive compute"，不能称为 "routing-aware"。

正确的名字是 **state-conditioned adaptive rollout budgeting**，不是 routing-based selection。

#### 第一轮实测（同一批 44 个 state，留一状态 ridge，共同 PCA 预算）

预测 $D_K(s)=\operatorname{median}_{i<j}\|A_i-A_j\|$：

| 预测量 | $R^2$ | Spearman |
|---|---:|---:|
| state routing（逐层 $M_4$ + 熵，16 维） | **−0.055** | +0.240 |
| proprio state（基线） | **+0.067** | +0.342 |
| proprio + routing | −0.027 | +0.257 |

**路由不如原始本体感受，加进去反而更差。** 与 $I(D;g(h))\le I(D;h)$ 一致：路由是 state 的确定性投影，读不出 state 里没有的东西。

#### 预算收益分解（`analyze_budget_gain.py`，零模拟器成本，用已有的 fork pilot）

把"多采样是否有用"拆成三项，$\mathcal S$ = 10 步归一化 action medoid（gripper 单独降权），子集随机抽取而非取前 $K$ 条：

| $K$ | $O_K$（oracle 机会） | $G_K^{\mathrm{med}}$（实际兑现）[95% CI] | $R_K^{\mathrm{med}}$（regret） | 兑现比 |
|---:|---:|---|---:|---:|
| 1 | +0.0004 | +0.0004 [−0.0003,+0.0012] | +0.0000 | +1.00 |
| 4 | +0.0121 | −0.0008 [−0.0024,+0.0002] | +0.0130 | −0.07 |
| 8 | +0.0170 | **−0.0018 [−0.0042,−0.0002]** | +0.0187 | −0.11 |
| 32 | **+0.0216** | **−0.0032** | +0.0247 | **−0.15** |

$O_{32}=+0.0216$ 相当于**每状态候选值标准差的 5.0 倍**——机会是真实且随 $K$ 单调增长的。但 medoid **不但拿不到，还比随机抽一条更差**（$K=8$ 的 CI 显著为负）。regret 把机会吃光还超出。

**这正是您列的情况 B：候选池里有好动作，可部署的 selector 找不到。** 按您的决策规则，**不应训练 adaptive $K$，应先换 selector**。

作用域：14 个可用快照（20 个里 6 个标签退化），单任务单 init state，`released-left` 操作点，价值代理是 drawer 行程而非成功。二元 `success_in_window` 只有 **1 个**快照有方差，太稀疏，无法分解——这也正是 fork pilot 当初选连续标签的原因。

**功效限制要写清**：44 个 state 全部来自**一个 init state 的 4 条 episode**，是同几条轨迹上的相邻点，不是 44 个多样状态。所有 $R^2$ 都贴着 0，差异在噪声内。这一轮只能说"没有证据支持 routing 优于 proprio"，不能说 $D_K$ 不可预测。要真做这条线，需要跨任务、跨 init state 的几十个独立状态，外加固定 selector 的 $G_K^{\mathcal S}$ 标签。

### 7.2 conditional lead test：现在是归因实验，不是找信号

真实观测下 $\rho(d_\varepsilon,d_A)=0.384$ / AUC 0.689（`within64-s24`，64 sibling），说明 flow 对初始噪声几何做了实质重构，残差

$$r_{ij}=d_A(i,j)-\hat g\big(d_\varepsilon(i,j)\big)$$

是合理的研究对象。但按本轮结果，解释它的**不太可能是路由**。所以问题要改成：

> action expert 的**哪一层内部表示**最早编码了 flow 对噪声几何施加的额外变形，而 gate 最多保留了其中多少？

逐层嵌套：

$$\mathcal M_0:[d_\varepsilon,d_{x^\tau}]\ \to\ \mathcal M_1:+d_{h_{\text{pre-MoE}}}\ \to\ \mathcal M_2:+d_{\text{router logits}}\ \to\ \mathcal M_3:+d_{\text{routed/shared out}}\ \to\ \mathcal M_4:+d_{h_{\text{post-MoE}}}$$

有一个硬的信息论约束：$g(h)=\operatorname{softmax}(Wh)$ 是 $h$ 的确定性函数，所以理想解码器下

$$I(r;g(h))\le I(r;h)$$

**router 不可能包含 hidden state 中原本不存在的信息**，最多是把某些方向压成更易被简单 probe 读出的低维表示。因此 hidden 与 gate 必须用**相同维数的 PCA、相同正则、相同 train/test split**，否则"gate 更好预测"会被误读成"gate 信息更多"。

**统计**：pair 不独立（$K$ 个候选有 $K(K-1)/2$ 个 pair 但只有 $K$ 个抽样单位），所有置信区间按 **query state** 自助。

**停止条件**：在 20–50 个真实 state 上，若 router/expert 特征相对基线的平均增益 $\Delta\mathrm{AUC}<0.01\sim0.02$ 且跨 state 不稳定，就不必再采更大的 routing cloud。

#### 实测（2026-08-15，44 个真实 query state × K=16，CPU 采集 77.9 min）

| $\tau$ | AUC($d_\varepsilon$) | AUC($d_{x^\tau}$) | AUC($d_R$) | $\mathcal M_0$ | $\mathcal M_1$ | $\Delta$AUC [95% CI] | 置换 | **净** |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.913 | 0.913 | 0.546 | 0.913 | 0.942 | +0.0289 [+0.0237,+0.0347] | +0.0015 | **+0.0274** |
| 1 | 0.913 | 0.937 | 0.559 | 0.994 | 0.994 | +0.0002 | +0.0003 | −0.0001 |
| 5 | 0.913 | 0.991 | 0.569 | 0.997 | 0.997 | +0.0005 | +0.0004 | +0.0001 |
| 9 | 0.913 | 0.999 | 0.657 | 1.000 | 1.000 | +0.0001 | +0.0001 | +0.0000 |

**state-token 零对照在全部 10 步精确为 0.0000**（管线在真实观测上再次验证）。

置换对照（同一组路由距离在 pair 间打乱）量出样本内乐观，$\tau\ge1$ 的"显著"增益**全部**是它：净增益 $-0.0002\sim+0.0002$。唯一真实的增益在 $\tau=0$（+0.027，18 倍于乐观），那里 $x^{(0)}$ 就是噪声本身，路由提供了原始 L2 读不出的非线性读数；而 $\tau=1$ 时 provisional action 单独已达 0.994，**没有余量**（ceiling effect）。

**$\Delta\mathrm{AUC}\le0.0002 \ll 0.01$，停止条件达成**：不必再采更大的 routing cloud。

**一处对我自己早先数字的纠正**：之前用 `within64-s24` 报的 $\mathrm{AUC}(d_\varepsilon\to d_A)=0.689$ 是拿 24 维原始噪声比 7 维后处理动作、且未做有效维掩码。同空间带掩码的干净测量是 **0.913**。"真实观测下缺口很大"这个论证大部分是测量假象——缺口是 0.913→1.000。

---

## 八、作用域与待复核

- 层 ①、③ 与全组替换的测量在 **CPU bf16**（autocast shim）+ **合成观测** + 2–4 次 query 上完成。
- 会迁移到 CUDA 的：tie **结构**（唯一值数、$\Delta_{4,5}=0$ 比例）是 dtype 与 logit 分布的性质；专家 cosine 是权重的性质。
- **不会**迁移的：tie **打破的选择**是 kernel 相关的。同卡重复 / 不同 MIG / 不同设备的 top-4 集合对比必须上 GPU。
- 剂量曲线只有 1 个和 4 个两点，中间未测。
- ~~未在第二个公开 checkpoint 上复制 audit~~ **已做**：router audit 在 spatial + object 上复现，near-tie 与 expert-swap 在 spatial 上复现（§3.2、§4.4）。CALVIN-D 未测。
- 分支比例与 1.112 需在真实观测上复现。

## 九、三处纠错的完整轨迹

保留它们不是自曝瑕疵，而是展示管线如何变得可信。共同教训：**当一个复杂系统给出异常干净的数字时，先怀疑索引、对象身份、数值路径和比较空间，再考虑是否发现了强规律。**

### 错误 1：把 $x^{(9)}$ 当成最终 chunk

1. **原始异常**：$\tau=9$ 的 action rho 精确为 **+1.000**，AUC 精确 1.000。
2. **当时为何看似合理**："动作在最后一步完全成熟"是符合直觉的叙事。
3. **什么暴露了它**：rho 精确到小数点后三位全是 1——确定性系统里的完美相关，只可能是自比较。
4. **修复后**：$\tau=9$ 为 rho +0.998 / AUC 0.997。`denoise_step` 只在入口抓 $x_t$，$T$ 次调用给 $x^{(0)}\ldots x^{(T-1)}$，最后一次 Euler 更新在返回之后；改为同时抓 $v_t$ 并外推 $x^{(T)}=x^{(T-1)}+\Delta t\,v^{(T-1)}$。
5. **加入的自动检查**：`euler_residual()` 校验 $x^{(\tau+1)}=x^{(\tau)}+\Delta t\,v^{(\tau)}$，容差按 bf16 舍入量级（0.05）设定——挂错张量会给出 $O(|x|)$ 的残差，比这大两个数量级。

### 错误 2：autocast 区内的名义 fp32 shadow

1. **原始异常**：确定性 argsort 与"fp32 shadow"的 top-4 集合分歧率 **0.00%**。
2. **当时为何看似合理**：可以读成"量化不改变偏好"，一个干净的正结论。
3. **什么暴露了它**：0.00% 在有 24% 精确打平的数据上不可能——两个不同精度的排序不该完全一致。
4. **修复后**：argsort vs 真 fp32 = 13.60%。autocast 会拦截 `F.linear`，不管输入 cast 成什么 dtype，所以 shadow 实际就是部署计算本身。
5. **加入的自动检查**：shadow 必须在 `torch.autocast(..., enabled=False)` 内计算，并**断言** `shadow.dtype == torch.float32`。

### 错误 3：跨表示空间的距离比较

1. **原始异常**：真实观测下 $\mathrm{AUC}(d_\varepsilon\to d_A)=0.689$，与合成观测的 0.877–0.934 差距很大。
2. **当时为何看似合理**："真实观测留下了大缺口"正好为下一步采集提供了立项理由。
3. **什么暴露了它**：同空间、带有效维掩码的干净测量给出 **0.913**，与合成观测一致。原来的 0.689 是拿 **24 维原始噪声**比 **7 维后处理动作**、且未掩掉 padding 维。
4. **修复后**：缺口是 0.913→1.000，不是 0.689→1.000。立项理由不成立（采集本身仍产出了"净增益为零"这个硬结论）。
5. **加入的自动检查**：任何距离比较必须显式声明 **representation / mask / normalization / 样本单位**四项。`analyze_flow_lead.py` 用数据驱动的有效维掩码（跨候选真正有方差的维），并在文档字符串里写明四项。

### 持续零对照

state token 对候选**按计算图**零判别力（只 attend prefix、不接 timestep），所以它的 $\Delta$AUC 必须恒为 0。实测在合成与真实观测的全部 10 个 flow step 上都精确为 **0.500 / 0.0000**。任何一处索引、配对、轴序或距离函数接错，它都不会正好命中。**这是本轮最有效的单项检查，应作为常设项保留。**

---

## 十、最终收束（2026-08-15）

### 9.1 关闭三条

$$\boxed{\text{AS/HB routing}\ \rightarrow\ \text{candidate ranking}}$$
$$\boxed{\text{action-cloud consensus or geometry}\ \rightarrow\ \text{candidate ranking}}$$
$$\boxed{\text{state routing}\ \rightarrow\ \text{adaptive }K\ \text{（在当前 selector 族下）}}$$

**不应再尝试**：第五种 action distance、更复杂的 medoid kernel、调 commitment/jerk 权重、在同一组特征上换更大的 ranker、或先训 adaptive $K$ 再指望以后补 selector。这些都补不回缺失的状态与任务语义。

### 9.2 决策价值的完整证据（`analyze_budget_gain.py`，512 次随机子集，逐 snapshot bootstrap）

$K{=}1$ 按定义锚定为 $O_1=R_1=G_1\equiv0$；实测 MC 误差中位 $8\times10^{-5}$、最大 $1.6\times10^{-3}$。

| selector | $G_8$ | $G_{16}$ [95% CI] | $G_{32}$ | $K{=}32$ 兑现比 |
|---|---:|---|---:|---:|
| medoid | −0.0018 | −0.0024 [−0.0057,+0.0003] | −0.0032 | −0.15 |
| anti-medoid | −0.0015 | −0.0033 [−0.0097,+0.0025] | −0.0050 | −0.23 |
| commit | +0.0006 | +0.0008 [−0.0101,+0.0109] | +0.0007 | +0.03 |
| PCM | +0.0007 | +0.0003 [−0.0033,+0.0037] | −0.0012 | −0.06 |

对照 $O_4=0.0121$、$O_8=0.0170$、$O_{32}=0.0216$（≈5 个候选价值标准差）。监督版 action-only ranker 的答案已在案：`fork-pilot-n32` 的 LOSO 给出 $\rho_{\text{action}}=-0.034\ (p=.77)$，而组内是 $+0.264\ (p=.0005)$。

**作用域（必须与上表同段引用）**：14 个可用快照、单任务、单 initial state、`released-left` 操作点、**short-horizon drawer-progress proxy** 作为价值代理（**不是**任务成功价值）、二元 `success_in_window` 仅 1 个快照有方差、medoid 价值百分位偏离 0.5 约 $z=-2.0$。

因此**不能**写 "Action-cloud selectors generally fail for VLA rollout selection"，只能写：

> In this forked released-left pilot, action-cloud geometry failed to recover the available best-of-$K$ opportunity across held-out snapshots.

### 9.3 组内有效、LOSO 失效说明的是什么

**措辞必须限定在证据覆盖的范围内。** 被否定的**不是**所有 $f(A_i,\{A_j\})$——本 pilot 没有穷尽高容量 action-only 函数，一个足够大的序列网络理论上仍可能从十步动作的细节中推出隐藏任务阶段的某种代理。被否定的是**所测试的**那一族：状态无关的 action-cloud 几何、动力学摘要，以及低容量监督组合。

> 组内相关与跨快照失效的分裂表明，**所测试的**状态无关 action-cloud 函数不能作为可迁移价值模型；结果与"候选价值依赖状态—动作交互"的解释**一致**，因此下一步需要显式读取观测、状态、任务阶段和候选动作的价值模型。

> The within-snapshot/LOSO split rules out **the tested** state-agnostic action-cloud selectors as transferable value models. It is **consistent with** candidate value depending on state–action interaction, motivating a critic of the form $\hat Q(o_s,x_s,\mathrm{phase}_s,A_i)$.

$Q(s,A_i)=f(o_s,x_s,\text{phase}_s,A_i)$ 目前是**受支持的假设**，不是已验证的正结果——读取观测的 critic 还没训练过，更没验证过。同一段末端位移在不同状态下可能是"接近把手""已接触施力""即将脱离""沿正确轴拉动""因姿态产生无效侧向运动"。**在所测的特征族里没有找到跨状态共享的价值坐标系**——固定状态时局部关系可见，换快照就变。这也解释了为什么监督 action-only ranker 并不比手工几何规则更有希望。

### 9.4 两个精确结论（这才是本轮的产出）

$$\boxed{\text{MoE routing can be causally consequential without being a readable decision state}}$$
$$\boxed{\text{Best-of-}K\text{ opportunity can exist without being recoverable from candidate-cloud geometry}}$$

不是"routing 没用"。第一条由 §4.4 的 $\Delta y_{\text{MoE}}=31\text{–}50\%$ 直接支撑；第二条由 $O_{32}=5$ 个标准差与四个 selector 全部失败共同支撑。

### 9.5 下一步是新项目，不是本项目的延续

必须直接估计 $\widehat Q(o,s,\text{instruction},A)$ 而非 $\widehat Q(A,\mathcal C)$：state-conditioned value critic、读取视觉/语言/候选动作的 verifier、或显式任务进度模型。数据规模完全不同——14 个快照够否定廉价 selector，不够训练 critic；新数据需覆盖多 initial state、多任务阶段、多接触状态、多任务与 benchmark，划分仍按 state/episode/task，绝不按候选拆分。

`commit` 的微弱非负趋势可留作未来 critic 的输入先验或正则项，**不再作为待优化的方法**。

### 9.6 $\tau=0$ hidden 抽头：不再作为主线

它最多回答"第一次 denoise forward 内部哪一层最早编码最终 action geometry"，而当前瓶颈已经是 $\text{final action geometry}\not\Rightarrow\text{candidate value}$。且 $\tau=1$ 的外部 provisional action 已达 0.994，提前量有限。仅在将来单独写"flow geometry 如何在层间形成"的机制分析时才值得只在 $\tau=0$ 重采一次。

---

## 附录：产物清单

```
analysis/router-audit/summary.json
analysis/near-tie/summary.json
analysis/expert-contribution/summary.json, summary_n11.json
analysis/slice-control-v1/slice_control.json
analysis/libero30-right-v1-eq192/summary.json
analysis/graph-separability/summary.json
analysis/graph-phase/summary.json
analysis/flow-lead-cpu/lead_curve.json
```

```
analysis/budget-gain/{summary,medoid,anti_medoid,commit,pcm,success}.json
analysis/value-location/summary.json
analysis/flow-lead/real_obs.json
analysis/adaptive-k/summary.json
runs/flow-lead-cpu-t0s24/          44 states x 16 candidates, 11 part files
```

脚本：`router_audit.py`、`probe_near_tie.py`、`probe_expert_swap.py`、`probe_expert_contribution.py`、`probe_uniformity.py`、`analyze_slice_control.py`、`probe_routing_graph.py`、`probe_graph_reliability.py`、`analyze_graph_separability.py`、`probe_graph_phase.py`、`serve_flow_trace.py`、`rollout_flow_lead.py`、`probe_flow_lead_cpu.py`、`analyze_flow_lead.py`、`analyze_adaptive_k.py`、`analyze_budget_gain.py`、`analyze_value_location.py`


---
