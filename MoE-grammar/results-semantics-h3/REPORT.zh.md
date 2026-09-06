# 第四层：语义（`idea.md` §十三）

## 为什么单独做这一层

`idea.md` 第 28–29 行把两件事分开：

> 健康句子的条件转移概率是「**语法**」；
> 句子未来进入 loop、static、恢复或完成的概率是「**语义**」。

此前所有失败监测结果，都是拿语法层的 surprisal 直接回归最终 episode outcome。那既不是
语法该做的事，也不是语义的定义。本层按原文实现：给 suffix-tree context 挂上未来 h 个
query 的 outcome 计数 $n_{c,h}$，取 Dirichlet 后验 $\mu(c,h)$，长 context 支持不足时沿
suffix 回退。

判定标准只有一个：**context-conditioned $\mu(c,h)$ 能否胜过把 context 去掉的同一个模型**。
若不能，则语法 context 不携带 outcome 语义，这一层就是空的。

## 协议

- 语料：full-40，fold 0，state-blocked（训练 states 与测试 states 不相交）。
- outcome 字母表：`return` / `persist` / `end_success` / `end_failure`。原文的 `loop` 与
  `static` 未纳入——本语料没有可靠 loop 标注，`static` 只存在于 Scene8 anchor。
- 「当前偏离」定义为语法自身 surprisal 的校准上尾（train-success 经验 CDF ≥ 0.8）。
  只有偏离中的 query 才被打标：不偏离时问「是否返回」没有意义。
- context 严格取当前 query 之前的词；outcome 取之后 h 个 query。max_order=3，
  min_support=30，Dirichlet α=1。
- 对照 root 模型是同一估计器令 max_order=0，除 context 外一切相同。

## 结果

fold 0 测试集共 24,187 个偏离 query（return 14,699 / end_success 5,729 / persist 3,182 /
end_failure 577）。log-loss，越低越好；CI 为 init-state 分块 bootstrap。

| h | 全字母表 context | root | 差 | 95% CI | states |
|---:|---:|---:|---:|---|---:|
| 1 | 0.9188 | 1.0584 | **−0.1387** | [−0.152, −0.126] | 10/10 |
| 3 | 0.8073 | 1.0002 | **−0.1915** | [−0.209, −0.177] | 10/10 |
| 5 | 0.7740 | 0.9875 | **−0.2120** | [−0.232, −0.192] | 10/10 |

backoff 分布（h=3）：order-2 占 51.6%、order-3 占 18.9%、order-1 占 27.3%，仅 2.2% 退到根。
即 context 普遍有支持，不是靠回退撑起来的。

### 混淆控制：return vs persist

全字母表的增益里可能混有「这个 context 预示 episode 快结束了」。剔除 `end_*`、只在
`return` 与 `persist` 之间重归一化，可分离出原文真正关心的
Correction→Return 与 Correction→Persist：

| h | context | root | 差 | 95% CI | states |
|---:|---:|---:|---:|---|---:|
| 1 | 0.6507 | 0.6934 | **−0.0420** | [−0.051, −0.034] | 10/10 |
| 3 | 0.4130 | 0.4689 | **−0.0543** | [−0.071, −0.038] | 10/10 |
| 5 | 0.2836 | 0.3288 | **−0.0434** | [−0.062, −0.027] | 10/10 |

约 72% 的全字母表增益确实来自 episode 长度信号，但**剩下的纯 return/persist 语义在三个
horizon 上都显著、且 10/10 states 一致**。

作为量级参照：语法层最强的有序证据 `ordered − bag` 是 0.005 bits/query，而 h=3 的
return/persist 语义增益是 0.054 nats ≈ 0.078 bits/query，约为其 **15 倍**。

## 干预语义

`idea.md` §十三.2 要求 $n_{c,u,h,y}$ 与 $\mu_{int}(c,u,h)=P(Y\mid c, do(u))$。
fork 语料只有 **10 个独立 trunk**（全部 trunk 失败），因此条件化到 routing context 不可估计，
这里只做 arm 层，并按 trunk 分块 bootstrap——同一 fork 内的 candidate 共享前缀，不独立。

| arm | success | trunk-blocked 95% CI | 减 control | 95% CI |
|---|---:|---|---:|---|
| `control` | 0.020 | [0.000, 0.000] | — | — |
| `triggered` | 0.320 | [0.025, 0.583] | **+0.3125** | [+0.025, +0.583] |
| `delayed_+4` | 0.220 | [0.000, 0.357] | +0.2083 | [+0.000, +0.357] |
| `delayed_+8` | 0.118 | [0.000, 0.250] | +0.0938 | [+0.000, +0.250] |

只有 `triggered` 的对照区间排除零，且区间极宽。`delayed_+4` / `delayed_+8` 都触零。
这与 `idea.md` 第 1092 行的预判一致：现有 fork 数据不足以学可靠的 recovery semantics。

## 边界

- `return` / `persist` 由**语法自身的 surprisal 百分位**定义，不是物理标签。因此本层证明的是
  「context 能预测这条 routing 轨迹会不会回到健康流形」，不是「能预测机器人会不会脱困」。
  物理有效性仍须以独立 stasis onset 为准，而那条路已被证否。
- 干预语义只到 arm 层，10 个 trunk。要支撑 $\mu_{int}(c,u,h)$ 需要显著更多独立 trunk。
- 结果来自 fold 0；五折复算尚未做。
