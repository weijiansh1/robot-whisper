可以。下面我把“MoE 语法学习”写成一套**可直接落地、可做论文消融、在线严格只读 MoE**的完整方案。

先给出最核心的定义：

> **健康 MoE 语法不是学习一个平均健康向量，而是学习：一次 action chunk 的 routing 过程通常怎样形成，以及连续 action chunks 之间通常怎样转移、持续和结束。**

整套系统建议分成五层：

$$
\boxed{
\text{MoE routing}
\rightarrow
\text{query 表征}
\rightarrow
\text{routing word}
\rightarrow
\text{healthy grammar}
\rightarrow
\text{异常与语义}
}
$$

对应地：

* 一次 flow step 的局部 routing 状态可以理解成“字母”；
* 一次 query 内完整的 10-step flow routing 是“单词”；
* 连续多个 query 是“句子”；
* 健康句子的条件转移概率是“语法”；
* 句子未来进入 loop、static、恢复或完成的概率是“语义”。

不过工程上，我不建议第一版真的做两次离散化——先把 flow step 离散成字母，再把字母序列离散成单词。这样误差会层层累积。

**第一版最稳的做法是：把一次完整 query 直接压成一个 routing word，再学习跨 query 的语法。**“字母”只用于后续解释单词内部发生了什么。

---

# 一、你真正要学习的数学对象

第 \(i\) 条 episode、第 \(q\) 次 VLA 重规划的完整 HB-MoE routing 为：

$$
P^{(i)}_q
=
\left\{
p^{(i)}_{q,l,f,u,e}
\right\}
\in[0,1]^{L\times F\times U\times E}.
$$

按你当前数据：

$$
L=8,\qquad F=10,\qquad U=11,\qquad E=32.
$$

其中：

* \(l\)：HB-MoE layer；
* \(f\)：flow step；
* \(u\)：state/action token；
* \(e\)：expert；
* \(p_{q,l,f,u,e}\)：完整 gate softmax。

每个成功 episode 最终形成一条序列：

$$
\mathcal R_i=
(P^{(i)}_1,P^{(i)}_2,\ldots,P^{(i)}_{T_i}).
$$

目标不是重建 \(P_q\)，而是学习健康序列分布：

$$
p_H
\left(
P_q\mid P_{1:q-1}
\right).
$$

在线时，若：

$$
-\log p_H(P_q\mid P_{1:q-1})
$$

持续很高，就说明当前 routing 序列不符合健康语法。

但直接对 \(8\times10\times11\times32\) 的高维张量建模太重，也难解释，所以需要先得到 query-level 表征。

---

# 二、数据如何划分：这一步比模型更重要

你必须避免兄弟分支和相同 snapshot 泄漏。

建议按以下单位分组：

$$
\boxed{
\text{task}
+
\text{init-state}
+
\text{root episode/snapshot}
}
$$

来自同一个 root trajectory 的所有 rolling branches、Snapshot-Forks、noise seeds，必须全部进入同一个 split。

建议划成四部分：

| 数据集                  | 用途                      | 是否需要 Trap 标签 |
| -------------------- | ----------------------- | -----------: |
| Healthy Train        | 学 query tokenizer 和健康语法 |            否 |
| Healthy Calibration  | 定阈值、控制健康误报              |            否 |
| Semantic Calibration | 将语法异常映射成 loop/static 概率 |         少量需要 |
| Blind Test           | 最终检测、提前量、跨语料评估          |        只用于评价 |

Healthy Train 应包括：

$$
\text{clean success}
+
\text{recovered success}
+
\text{slow success}.
$$

不要只放最短、最顺的成功轨迹。否则正常纠错和慢速推进都会被模型当成“病句”。

如果当前还没有 slow success，可以先训练第一版语法，但暂时不要主张它能识别“健康但被 timeout 截断”的轨迹。

---

# 三、第一层：从原始 routing 提取 flow-step 表征

为了避免 expert-ID 置换问题，主模型应优先使用**置换不变的概率几何特征**，而不是 Expert 7、Expert 19 这样的编号。

对每个 \(q,l,f\)，构造 token–expert gate matrix：

$$
G_{q,l,f}
=
\begin{bmatrix}
p_{q,l,f,1,1}&\cdots&p_{q,l,f,1,E}\\
\vdots&&\vdots\\
p_{q,l,f,U,1}&\cdots&p_{q,l,f,U,E}
\end{bmatrix}
\in\mathbb R^{U\times E}.
$$

建议每个 layer、flow step 计算以下量。

## 1. 平均归一化 gate entropy

$$
H_{q,l,f}
=
\frac1U
\sum_u
\frac{
-\sum_ep_{q,l,f,u,e}\log p_{q,l,f,u,e}
}{
\log E
}.
$$

## 2. Top-1/Top-2 margin

$$
M_{q,l,f}
=
\frac1U
\sum_u
\left(
p_{(1)}-p_{(2)}
\right).
$$

## 3. Top-1 或 Top-4 mass

$$
S_{q,l,f}^{(K)}
=
\frac1U
\sum_u
\sum_{k=1}^Kp_{(k)}.
$$

## 4. 跨 token soft consensus

先定义 Hellinger 距离：

$$
d_H(p_i,p_j)
=
\frac1{\sqrt2}
\left\|
\sqrt{p_i}-\sqrt{p_j}
\right\|_2.
$$

然后：

$$
C_{q,l,f}^{tok}
=
1-
\frac{2}{U(U-1)}
\sum_{i<j}
d_H(p_i,p_j).
$$

它比离散 Top-4 occupancy 更稳，因为不会受到 top-\(K\) 截断边界过度放大。

## 5. Gate matrix effective rank

设 \(G\) 的奇异值为 \(\sigma_j\)，定义：

$$
\pi_j=
\frac{\sigma_j}{\sum_k\sigma_k},
$$

$$
r_{q,l,f}^{eff}
=
\frac{
\exp(-\sum_j\pi_j\log\pi_j)
}{
\min(U,E)
}.
$$

它衡量跨 token routing 有多少有效自由度。

## 6. 跨层一致性

将每层平均 gate distribution 记为：

$$
\bar p_{q,l,f}
=
\frac1U\sum_up_{q,l,f,u}.
$$

定义：

$$
D_{q,f}^{layer}
=
\frac{2}{L(L-1)}
\sum_{l<l'}
d_H
\left(
\bar p_{q,l,f},
\bar p_{q,l',f}
\right).
$$

## 7. Flow routing velocity

$$
V_{q,l,f}
=
\frac1U
\sum_u
d_H
\left(
p_{q,l,f,u},
p_{q,l,f-1,u}
\right).
$$

## 8. Route acceleration

使用平方根概率坐标：

$$
x_{q,l,f,u}=\sqrt{p_{q,l,f,u}},
$$

$$
a_{q,l,f,u}
=
x_{q,l,f+1,u}
-
2x_{q,l,f,u}
+
x_{q,l,f-1,u}.
$$

再聚合：

$$
A_{q,l,f}
=
\frac1U\sum_u\|a_{q,l,f,u}\|_2.
$$

最后，每个 flow step 得到一个局部表征：

$$
x_{q,f}
=
[
H,M,S,C^{tok},r^{eff},D^{layer},V,A
]_{l=1}^{L}.
$$

若逐层拼接，维度约为：

$$
8\text{ 个指标}\times8\text{ 层}=64.
$$

这是很小的。

---

# 四、第二层：将一次完整 query 编码成一个“单词”

一次 query 有：

$$
x_{q,0},x_{q,1},\ldots,x_{q,9}.
$$

最直接的连续 query descriptor 是：

$$
g_q
=
\operatorname{concat}
\left[
x_{q,0:9},
\Delta x_{q,1:9},
\Delta^2x_{q,2:9}
\right].
$$

它保留：

* flow 前期到后期的变化；
* 一阶 routing dynamics；
* 二阶 routing dynamics；
* 终端是否收敛；
* gate 是否先尖后平；
* token consensus 是否在后期突然变化。

维度可能几百，但没有关系。只使用 Healthy Train 进行：

1. 中位数/MAD robust scaling；
2. PCA 压到 16–32 维；
3. 在压缩空间中学习健康 query codebook。

这里所有 normalizer 和 PCA 都只能在 Healthy Train 上拟合。

---

## 推荐的单词 tokenizer：GMM codebook

拟合一个 \(K\) 分量 Gaussian Mixture：

$$
p(g_q)
=
\sum_{k=1}^{K}
\pi_k
\mathcal N(g_q;\mu_k,\Sigma_k).
$$

每个 query 的 routing word 为：

$$
w_q
=
\arg\max_k
P(k\mid g_q).
$$

但在线评分时不要只保留 hard ID，也保存 soft posterior：

$$
\gamma_{q,k}=P(k\mid g_q).
$$

建议测试：

$$
K\in\{8,16,32,64\}.
$$

选择 \(K\) 时不能看 Trap AUC，而应使用：

* held-out healthy negative log-likelihood；
* codebook cluster stability；
* 最小 cluster support；
* 健康序列的 next-word predictability。

如果某个 query 在整个健康 GMM 下的密度低于健康 calibration 的极端分位数，可以额外输出：

$$
w_q=\langle UNK\rangle.
$$

这样，一个 query 自身从未在健康数据出现过，就是“词汇异常”。

VQ-VAE 也可以学习离散 code，但它引入编码器、解码器和 codebook optimization；对于你们当前这种低维结构化 routing feature，GMM 或 k-means 更容易解释、调试和做消融。VQ-VAE 更适合作为后续更大规模版本。([arXiv][1])

---

# 五、为什么不直接把十个 flow-step 字母拼成单词？

因为假设字母表大小为 12，一次 query 有 10 个 flow steps，理论组合数量是：

$$
12^{10}.
$$

绝大多数具体组合只出现一两次，概率语法会非常稀疏。

所以第一版采用：

$$
\boxed{
\text{完整 flow trajectory}
\rightarrow
\text{一个 query word}
}
$$

最稳妥。

后续为了可解释，可以再查看某个 word 内部对应的 flow pattern：

* 前期切换、后期收敛；
* 尖化—切换；
* flattening—shared support；
* late-flow acceleration；
* cross-token synchronization。

也就是说，字母层用于解释，单词层用于建模。

---

# 六、第三层：学习健康“句法”

每个健康 episode 变成：

$$
\langle BOS\rangle,
w_1,w_2,\ldots,w_T,
\langle END\rangle.
$$

这里 `<END>` 很重要。它将来允许你估计：

> 当前健康句子距离完成还有多少 query。

## 推荐第一版：Probabilistic Suffix Tree

PST/variable-order Markov model 的核心是：不同上下文可以使用不同长度的历史，不必统一固定成一阶或五阶 Markov。某个 query 可能只依赖前一个 word，而 pre-loop 类句型可能需要看前 3–5 个 words。([arXiv][2])

对任意后缀上下文：

$$
c_q=(w_{q-k},\ldots,w_{q-1}),
$$

统计：

$$
N(c_q,w).
$$

带 Dirichlet smoothing 的 next-word distribution：

$$
P_H(w\mid c)
=
\frac{
N(c,w)+\alpha
}{
N(c)+\alpha|\mathcal V|
}.
$$

其中：

* \(\mathcal V\)：word vocabulary，包括 `<UNK>` 和 `<END>`；
* \(\alpha\) 可先取 \(0.5\) 或 \(1\)；
* 最大上下文长度建议：

$$
K_{\max}\in\{4,6,8\}.
$$

在线使用**最长且有足够训练支持的后缀**：

$$
c_q^*
=
\max
\left\{
c:
N(c)\ge n_{\min}
\right\}.
$$

例如：

$$
n_{\min}=20.
$$

还可以增加一个 pruning 条件：只有当子上下文的 next-word distribution 与其父上下文有足够 KL 差异时，才保留长上下文：

$$
D_{KL}
\left(
P(\cdot\mid c)
\|
P(\cdot\mid suffix(c))
\right)
>\delta.
$$

这样不会把偶然出现一次的长句型硬记进语法。

---

# 七、不要把 hard word assignment 的误差传给语法

在线拿到 \(g_q\) 时，GMM 给出每个 word 的 emission likelihood：

$$
p(g_q\mid w=k)
=
\mathcal N(g_q;\mu_k,\Sigma_k).
$$

健康语法根据历史给出 word prior：

$$
P_H(w=k\mid c_q).
$$

因此当前 query 在健康语法下的完整预测概率是：

$$
\boxed{
p_H(g_q\mid c_q)
=
\sum_{k=1}^{K}
P_H(w=k\mid c_q)
p(g_q\mid w=k)
}
$$

这比简单先取：

$$
w_q=\arg\max_kP(k\mid g_q)
$$

再计算单个 word 概率更稳。

它同时利用：

* 当前 query 本身像哪个健康 word；
* 当前上下文预计下一个 word 应该是什么。

---

# 八、把异常拆成四类，而不是一个黑箱分数

## 1. 词汇异常：这一次 query 本身就不像任何健康 word

忽略上下文，使用健康全局 word prior：

$$
p_H^{lex}(g_q)
=
\sum_k
\pi_k
p(g_q\mid w=k).
$$

定义：

$$
A_q^{lex}
=
-\log p_H^{lex}(g_q).
$$

高 \(A^{lex}\) 说明：

> 这一次 flow routing 过程本身就是一个陌生“单词”。

例如 pre-loop 的异常 late-flow acceleration 可能形成词汇异常。

---

## 2. 句法异常：单词见过，但出现在不合适的上下文

$$
A_q^{ctx}
=
-\log p_H(g_q\mid c_q).
$$

为了去除“这个词本身很少见”的影响，可以定义纯顺序惩罚：

$$
\boxed{
A_q^{order}
=
A_q^{ctx}-A_q^{lex}
}
$$

若：

$$
A_q^{lex}\text{ 正常},
\qquad
A_q^{order}\text{ 高},
$$

说明：

> 这个 routing word 本身健康，但此时不应该接在前面那句话之后。

这最适合发现：

* 错误顺序；
* phase regression；
* 普通模式以异常顺序组合；
* mixed Trap。

---

## 3. 持续时间异常：合法单词说得太久

设当前 macro-word 已持续 \(d_q\) 个 queries。

从健康数据学习：

$$
P_H(D=d\mid w,c).
$$

第一版可使用平滑经验 histogram 或 negative-binomial distribution。

定义：

$$
A_q^{dur}
=
-\log
P_H(D\ge d_q\mid w_q,c_q).
$$

这里用 survival probability 比点概率更合理，因为你关心的是：

> 一个状态持续至少这么久，在健康语法中有多罕见？

Static 很可能属于：

$$
A^{lex}\text{ 不一定高},
\quad
A^{order}\text{ 不一定高},
\quad
A^{dur}\text{ 持续升高}.
$$

也就是：

> 单词合法、语序合法，但一个词被重复得太久。

若需要更原则化的 duration 建模，可以进一步换成 HSMM。HSMM 与普通 HMM 的主要区别是显式建模隐状态持续时间，非常适合区分 1–2 query 的正常纠错与持续多 query 的 Trap。([ScienceDirect][3])

---

## 4. 返回异常：纠错后没有回到健康句型

健康成功轨迹中可能有：

$$
\text{Progress}
\rightarrow
\text{Correction}
\rightarrow
\text{Return}
\rightarrow
\text{Progress}.
$$

而 Trap 可能是：

$$
\text{Progress}
\rightarrow
\text{Correction}
\rightarrow
\text{Correction}
\rightarrow
\text{Lock}.
$$

可以从 PST transition graph 计算，给定当前 context，在未来 \(h\) 个 query 内到达某个“核心健康 context 集合” \(\mathcal H_{\text{core}}\) 的概率：

$$
P_H
\left(
T_{\mathcal H_{\text{core}}}\le h
\mid c_q
\right).
$$

定义：

$$
A_q^{ret}(h)
=
1-
P_H
\left(
T_{\mathcal H_{\text{core}}}\le h
\mid c_q
\right).
$$

第一版如果不想显式定义核心健康状态，可以先不加入这个分量，使用 CUSUM 对异常持续性进行替代。

---

# 九、总异常分数怎样组合？

不要直接把四个 raw NLL 相加，因为尺度不同、相关性也很强。

先在 Healthy Calibration 上将每个分数变成健康百分位：

$$
r_j(q)
=
\widehat F_j^{H}
\left(
A_q^j
\right),
$$

其中：

$$
j\in
\{
lex,order,dur,ret
\}.
$$

然后使用：

$$
S_q^{grammar}
=
\max_jr_j(q).
$$

这样一旦出现：

* 陌生单词；
* 非法语序；
* 超长持续；
* 无法正常返回；

任意一种异常，score 都会升高。

在线还应返回解释：

```text
grammar_anomaly = 0.97

lexical_surprise = 0.42
order_surprise   = 0.98
duration_surprise= 0.73
return_failure   = 0.91
```

---

# 十、单点异常不等于 Trap：加顺序累积器

正常纠错可能产生一个高异常 query，所以不能单点报警。

使用 CUSUM：

$$
G_q
=
\max
\left(
0,\,
G_{q-1}
+
S_q^{grammar}
-
\kappa
\right).
$$

当：

$$
G_q>\tau_{\mathrm{on}}
$$

进入异常状态。

只有当：

$$
G_q<\tau_{\mathrm{off}},
\qquad
\tau_{\mathrm{off}}<\tau_{\mathrm{on}},
$$

才退出异常状态。

这能区分：

### 短暂纠错

$$
S_q\uparrow
\rightarrow
S_{q+1}\downarrow
\rightarrow
G_q\text{ 回落}.
$$

### 持续 Trap

$$
S_q\uparrow
\rightarrow
S_{q+1}\uparrow
\rightarrow
S_{q+2}\uparrow
\rightarrow
G_q>\tau.
$$

阈值应在 Healthy Calibration 上使用**每条 episode 的最大 CUSUM**校准：

$$
\tau_\alpha
=
Q_{1-\alpha}
\left(
\max_qG_q^{(1)},
\ldots,
\max_qG_q^{(n)}
\right).
$$

这样控制的是：

> 一条健康 episode 中至少误报一次的概率。

而不是较弱的“随机 query 误报率”。

---

# 十一、加入 `<END>` 后，可以识别“正确但被 timeout 截断”

每条成功 episode 的末尾加入：

$$
\langle END\rangle.
$$

PST 的 context node 本身可以被看作状态。从当前 context \(c\)，定义未来 \(\Delta\) 个 query 内到达 `<END>` 的概率：

$$
E_q(\Delta)
=
P_H
\left(
T_{\langle END\rangle}\le\Delta
\mid c_q
\right).
$$

可通过动态规划计算：

$$
E_0(c)
=
\mathbf1[c=\langle END\rangle],
$$

$$
E_{h+1}(c)
=
P_H(\langle END\rangle\mid c)
+
\sum_{w\neq\langle END\rangle}
P_H(w\mid c)
E_h
\left(
next(c,w)
\right).
$$

于是到官方 timeout 时：

### 健康但被截断

$$
S_H^{grammar}\text{ 低},
\qquad
G_H\text{ 低},
\qquad
E_H(\Delta)\text{ 高}.
$$

含义是：

> 当前句子仍符合健康语法，而且从当前句子前缀出发，健康数据中经常在额外 \(\Delta\) 个 query 内结束。

### Trap-like timeout

$$
G_H\text{ 高}
$$

或者：

$$
A_H^{dur}\text{ 高},
\qquad
E_H(\Delta)\text{ 低}.
$$

含义是：

> 单纯多给时间更可能继续重复当前异常句型。

需要强调：\(E_H(\Delta)\) 起初只是**健康语法下的 completion hitting probability**。要将其称为真实“延长后成功概率”，仍需使用真实 late-success rollout 做校准。

---

# 十二、加入潜在“方言”，解决不同任务的健康语法不同

严格 MoE-only 时，在线不能直接输入 task ID。但全局只有一套语法可能太宽泛。

可以建立 mixture of grammars：

$$
p_H(g_q\mid history)
=
\sum_{m=1}^{M}
P(m\mid g_{1:q})
p_{H,m}(g_q\mid history).
$$

其中 \(m\) 是潜在 routing dialect：

* 不同任务；
* 不同内部阶段；
* 不同机器人风格；
* 不同初始状态。

最小实现：

1. 用 episode 前 3–5 个 query 的 word posterior 和 transition histogram 聚类；
2. 每个 cluster 拟合独立 PST；
3. 在线根据当前 prefix 更新各 grammar posterior；
4. 计算混合预测概率。

建议先从：

$$
M\in\{1,2,4,8\}
$$

测试。只有全局 grammar 健康误报很高时，再引入 dialect mixture。

---

# 十三、第四层：给语法附加“语义”

只用健康数据能学到：

$$
P_H(\text{routing sentence})
$$

但不能自动知道低概率句子具体意味着 loop、static 还是一种未见但合理的新行为。

语义需要少量外部 outcome 数据。

## 1. 预测语义

对一个 suffix-tree context \(c\)，保存未来 \(h\) 个 query 的结果计数：

$$
n_{c,h}
=
[
n_{\mathrm{loop}},
n_{\mathrm{static}},
n_{\mathrm{return}},
n_{\mathrm{success}},
n_{\mathrm{other}}
].
$$

使用 Dirichlet 后验：

$$
\theta_{c,h}
\sim
Dirichlet
\left(
n_{c,h}+\alpha
\right).
$$

得到：

$$
\mu(c,h)
=
\begin{bmatrix}
P(\text{loop within }h\mid c)\\
P(\text{static within }h\mid c)\\
P(\text{return healthy}\mid c)\\
P(\text{success}\mid c)
\end{bmatrix}.
$$

当长 context 样本不足时，沿 suffix tree 回退到短 context。

这就是句子的预测语义。

---

## 2. 干预语义

Snapshot-Fork 后，对不同干预 \(u\) 统计：

$$
n_{c,u,h,y}.
$$

得到：

$$
\mu_{\mathrm{int}}(c,u,h)
=
P(Y=y\mid c,do(u)).
$$

例如：

```text
当前 sentence prefix:
DESYNC → CONFIDENT_SWITCH

continue:
    loop     0.72
    recover  0.14

fresh noise:
    loop     0.68
    recover  0.17

short chunk:
    loop     0.31
    recover  0.52
```

这时，语法模型不再只说“这是一句病句”，而开始说：

> 这类句子通常需要什么回应。

但这必须建立在多个独立 trunks 的干预数据上；你们当前 fresh-noise 结果还不足以学习可靠的 recovery semantics。

---

# 十四、推荐的数据结构：Semantic Suffix Trie

每个 PST 节点可以存：

```text
GrammarNode:
    context: tuple[word_id]
    support_count: int

    next_word_counts[K + BOS/END/UNK]
    next_word_probs[K + BOS/END/UNK]

    duration_histogram[word_id]
    duration_survival[word_id]

    healthy_nll_distribution

    outcome_counts[horizon][outcome]
    intervention_counts[action][horizon][outcome]

    child_nodes
    suffix_parent
```

这样同一棵树同时承担：

* 健康语法；
* next-word prediction；
* duration modeling；
* completion hitting；
* outcome semantics；
* intervention semantics；
* 长 context 到短 context 的 backoff。

---

# 十五、在线算法完整流程

```python
def update_moe_grammar(router_probs):
    # 1. 从当前 MoE routing 提取纯 MoE query descriptor
    flow_features = extract_flow_features(router_probs)
    query_vector = build_query_descriptor(flow_features)
    query_vector = healthy_scaler.transform(query_vector)
    query_vector = healthy_pca.transform(query_vector)

    # 2. 健康 query-word emission likelihood
    emissions = codebook.component_likelihoods(query_vector)

    # 3. 根据已有 word history 选择 PST 最长可靠 context
    context = pst.longest_supported_suffix(word_history)

    # 4. 健康 grammar 对下一 word 的先验
    word_prior = pst.next_word_distribution(context)

    # 5. 当前 query 的健康条件似然
    conditional_likelihood = (word_prior * emissions).sum()

    # 6. 无上下文健康似然
    lexical_likelihood = (global_word_prior * emissions).sum()

    lexical_surprise = -log(lexical_likelihood)
    contextual_surprise = -log(conditional_likelihood)
    order_surprise = contextual_surprise - lexical_surprise

    # 7. 解码当前最可能 word
    posterior = normalize(word_prior * emissions)
    word = posterior.argmax()

    # 8. 更新 word duration
    duration = update_run_length(word)
    duration_surprise = pst.duration_surprise(
        context, word, duration
    )

    # 9. 映射到健康 calibration percentile
    grammar_score = max(
        healthy_cdf_lex(lexical_surprise),
        healthy_cdf_order(order_surprise),
        healthy_cdf_duration(duration_surprise),
    )

    # 10. 时序累积，区分瞬时 correction 和持久异常
    cusum.update(grammar_score)

    # 11. 计算从当前语法 context 到 END 的有限时域概率
    p_finish_extra = pst.end_hitting_probability(
        context, horizon=extra_queries
    )

    # 12. 可选：读取少量 outcome-calibrated semantics
    p_loop_2 = semantic_map.loop_within_2(context)
    p_static = semantic_map.static_now(context)

    word_history.append(word)

    return {
        "word": word,
        "lexical_surprise": lexical_surprise,
        "order_surprise": order_surprise,
        "duration_surprise": duration_surprise,
        "grammar_anomaly": cusum.value,
        "p_finish_with_extra_queries": p_finish_extra,
        "p_loop_within_2": p_loop_2,
        "p_static": p_static,
    }
```

整个在线过程只需要当前和历史 MoE routing。

---

# 十六、怎样证明“真的存在语法”？

这是整个方向的生死线。你不能因为使用了 PST 就说 MoE 有语法。

至少需要以下六个实验。

## 1. Unigram vs Markov vs PST

比较 held-out healthy NLL：

$$
\text{Unigram}
$$

$$
\text{First-order Markov}
$$

$$
\text{Fixed-order }k
$$

$$
\text{PST}
$$

$$
\text{PST + duration}.
$$

如果历史上下文不能明显降低 held-out NLL，说明只有词频，没有语法。

---

## 2. Ordered vs Bag-of-Words

保持窗口内 word 计数完全相同，打乱顺序。

比较：

$$
\mathrm{NLL}_{ordered}
<
\mathrm{NLL}_{shuffled}.
$$

如果有序句子和 bag-of-words 一样，说明顺序没有信息，不能讲句法。

---

## 3. Query-order shuffle

对健康 episode 打乱 query 顺序，但保持每个 query word 不变。

真实顺序应比打乱顺序更符合语法。

---

## 4. Flow-order shuffle

在每个 query 内打乱 10 个 flow steps，再重新生成 query descriptor 和 word。

如果 word codebook 真正在捕捉 flow 形成过程，打乱 flow 后：

* query lexical likelihood 应下降；
* word assignment 应改变；
* pre-loop 检测性能应下降。

否则 query word 可能只依赖 flow-step 边际统计，而不依赖 flow dynamics。

---

## 5. Letter → Word → Sentence 层级增益

比较：

$$
\text{single flow feature}
$$

$$
\text{single query descriptor}
$$

$$
\text{single routing word}
$$

$$
\text{bag of recent words}
$$

$$
\text{ordered routing sentence}.
$$

真正支持语言类比的结果应当是：

$$
\boxed{
\mathrm{Perf}_{sentence}
>
\mathrm{Perf}_{bag}
>
\mathrm{Perf}_{single\ query}.
}
$$

如果 sentence 没有增益，就应该保留“routing dynamics signal”，不要主张“grammar”。

---

## 6. 上下文消歧

筛选都包含高 volatility/acceleration word 的窗口。

比较：

$$
\text{Switch}
\rightarrow
\text{Return}
$$

和：

$$
\text{Desync}
\rightarrow
\text{Switch}
\rightarrow
\text{Sync}.
$$

如果前者主要属于正常纠错，后者主要进入 loop，说明同一个 word 的意义取决于上下文。

---

# 十七、Trap 检测实验怎样设计？

训练阶段：

$$
\boxed{\text{只用 healthy episodes}}
$$

测试阶段包括：

* held-out healthy；
* recovered success；
* slow success；
* loop；
* static；
* mixed Trap；
* 其他失败；
* 新任务；
* 新 init-state。

报告：

* healthy episode false alarm rate；
* event-level AUC/AUPRC；
* loop lead time；
* static detection delay；
* transient correction false positive；
* unknown Trap recall；
* sentence vs single-query improvement；
* lexical/order/duration 异常占比。

已知的 loop/static detector 可以作为并行锚点：

$$
(V^{late},A^{route})\rightarrow\text{loop precursor},
$$

$$
\text{lag periodicity}\rightarrow\text{static detector}.
$$

健康 grammar 负责开放集部分：

$$
\text{known motif 未触发}
+
\text{grammar anomaly 持续}
\rightarrow
\text{unknown persistent anomaly}.
$$

---

# 十八、怎样验证“健康但被截断”？

必须真正延长原 timeout 轨迹，而不能只根据 grammar 猜。

设原 horizon 为 \(H_0\)，继续运行到：

$$
H_1=1.5H_0
\quad\text{或}\quad
2H_0.
$$

得到：

* late success；
* self-recovery；
* persistent loop；
* persistent static；
* still censored。

在 \(H_0\) 时冻结 grammar 输出，比较：

1. 全部停止；
2. 全部延长；
3. 随机延长相同数量；
4. grammar-selected extension。

选择延长的条件可写为：

$$
G_{H_0}<\tau_G
$$

且：

$$
E_{H_0}(\Delta)>\tau_E.
$$

报告：

$$
\text{Extension Precision}
=
\frac{
\text{被延长且最终成功}
}{
\text{被延长总数}
}
$$

和：

$$
\text{Success Gain per Extra Action}.
$$

这会直接验证：

> 健康语法能否识别“句子还没说完”，而不是“句子已经坏了”。

---

# 十九、跨模型怎样做？

建议同时建立两套 tokenizer。

## Raw-ID grammar

保留 expert identity、具体 layer 和具体 flow index。

它是同模型性能上界，但几乎必然模型特定。

## Abstract phenotype grammar

只使用：

* normalized entropy；
* margin；
* consensus；
* effective rank；
* volatility；
* acceleration；
* recurrence；
* relative layer depth；
* normalized flow time。

换模型后只重新学习：

$$
\text{healthy scaler}
+
\text{query codebook}.
$$

再测试原 grammar 是否迁移。

比较：

1. raw zero-shot；
2. abstract zero-shot；
3. 新模型只重新学习 tokenizer；
4. few-shot grammar adaptation；
5. target-model full relearning。

最理想的结果是：

> 原始专家词汇不迁移，但抽象的 switching–flattening–synchronization 语法可以迁移。

---

# 二十、计算和存储成本

你当前每 query 的完整 softmax包含：

$$
8\times10\times11\times32=28160
$$

个数。

fp16 大约：

$$
28160\times2=56320\text{ bytes}
$$

约 55 KiB/query。

如果最终只保存：

$$
32
$$

维 query descriptor，fp32 只需：

$$
32\times4=128\text{ bytes/query}.
$$

压缩约：

$$
440\times.
$$

训练成本：

* robust scaler/PCA：CPU 即可；
* GMM codebook：CPU 或单张普通 GPU；
* PST：本质是计数和 trie；
* duration model：计数或低维拟合；
* semantic calibration：小型统计模型。

PST、Context Tree、HSMM 都远小于 VLA，PST 还是典型的可变阶离散序列预测模型；HSMM 则显式表示状态持续时间。([arXiv][2])

真正昂贵的仍然是：

$$
\boxed{\text{收集覆盖充分的健康和 late-success rollout}}
$$

而不是语法模型训练。

---

# 二十一、与现有 VLA failure monitoring 的区别

SAFE 使用成功和失败 rollout 的 VLA 内部特征训练 failure detector；VLAConf 则用冻结的 VLA representations 建立 one-class confidence，再进行任务成功概率校准。([arXiv][4])

你们真正可能产生区别的地方不是“又做 one-class anomaly detection”，而是：

1. **只读取 MoE routing，而不是 dense hidden state；**
2. **建模 query 内 flow 轨迹与 query 间转移两个嵌套时间尺度；**
3. **区分词汇异常、语序异常和持续时间异常；**
4. **识别每个局部 routing 状态都正常、但整体顺序错误的组合异常；**
5. **将 loop/static 视为已知句型，同时保留 unknown persistent sentence；**
6. **通过 `<END>` hitting probability 区分病句和未说完的健康句子。**

VLAConf 已经证明“冻结 VLA 表示 + one-class confidence + 小型校准器”是合理范式，因此你们必须通过“ordered routing sentence 是否优于单点或 bag-of-words”证明 grammar 的新增价值。([arXiv][5])

---

# 二十二、我最推荐的最小可行版本

暂时不要做 VQ-VAE、Transformer、HDP-HSMM，也不要一次建所有 Trap 语义。

先做：

$$
\boxed{
\text{Robust feature}
+
\text{PCA}
+
\text{GMM words}
+
\text{PST grammar}
+
\text{duration}
}
$$

具体超参数先固定为：

$$
K_{\mathrm{word}}\in\{16,32\},
$$

$$
K_{\max}^{context}=6,
$$

$$
n_{\min}=20,
$$

$$
\alpha=0.5.
$$

第一轮只回答四个问题：

1. ordered healthy sequence 是否比 shuffled 更可预测？
2. PST 是否优于 unigram、bigram 和 bag-of-words？
3. sentence score 是否比单 query 的 volatility/acceleration 更好地区分 transient correction 与 pre-loop？
4. 加入 duration 是否改善 static 和 persistent anomaly 检测？

只有这四项成立，才继续：

* outcome semantics；
* `<END>` extendability；
* intervention suffix；
* cross-model shared grammar。

---

# 二十三、建议的工程目录

```text
moe_grammar/
├── data/
│   ├── episode_index.parquet
│   ├── healthy_train.txt
│   ├── healthy_calibration.txt
│   ├── semantic_calibration.txt
│   └── blind_test.txt
│
├── features/
│   ├── flow_features.zarr
│   ├── query_vectors.zarr
│   └── word_posteriors.zarr
│
├── tokenizer/
│   ├── robust_scaler.pkl
│   ├── pca.pkl
│   ├── gmm_codebook.pkl
│   └── tokenizer_config.yaml
│
├── grammar/
│   ├── suffix_trie.pkl
│   ├── duration_model.pkl
│   ├── end_hitting_cache.pkl
│   └── grammar_config.yaml
│
├── semantics/
│   ├── outcome_counts.pkl
│   ├── intervention_counts.pkl
│   └── calibrator.pkl
│
└── evaluation/
    ├── ordered_vs_shuffle.py
    ├── letter_word_sentence.py
    ├── trap_detection.py
    ├── late_success.py
    └── cross_model_transfer.py
```

---

# 最终版本

整套 MoE 语法学习可以压缩为：

$$
\boxed{
P_{q,l,f,u,e}
\rightarrow
g_q
\rightarrow
w_q
\rightarrow
P_H(w_q\mid w_{<q})
}
$$

在线异常分解为：

$$
\boxed{
\text{Lexical Surprise}
+
\text{Order Surprise}
+
\text{Duration Surprise}
+
\text{Persistence}
}
$$

再通过少量 outcome 数据赋予：

$$
\boxed{
\text{loop/static/return/success semantics}
}
$$

并通过 `<END>` hitting probability估计：

$$
\boxed{
\text{当前是病句，
还是一个仍然健康但尚未完成的句子。}
}
$$

最值得先实现的不是一个复杂神经网络，而是 **GMM query words + PST + duration**。它成本低、解释性强，而且能最直接回答这个方向的生死问题：

$$
\boxed{
\text{MoE 的跨 query 顺序本身，
是否包含超越单点 routing signal 的信息？}
}
$$

只要 ordered sentence 稳定优于 bag-of-words、single-query 和 shuffled controls，“机器人的低语具有语法”就从一个好听的比喻，变成了一个有实验证据的科学结论。

[1]: https://arxiv.org/abs/1711.00937?utm_source=chatgpt.com "Neural Discrete Representation Learning"
[2]: https://arxiv.org/pdf/1107.0051?utm_source=chatgpt.com "On Prediction Using Variable Order Markov Models"
[3]: https://www.sciencedirect.com/science/article/pii/S0004370209001416?utm_source=chatgpt.com "Hidden semi-Markov models"
[4]: https://arxiv.org/abs/2506.09937?utm_source=chatgpt.com "SAFE: Multitask Failure Detection for Vision-Language-Action Models"
[5]: https://arxiv.org/html/2605.29605v1?utm_source=chatgpt.com "VLAConf: Calibrated Task-Success Confidence for Vision- ..."
