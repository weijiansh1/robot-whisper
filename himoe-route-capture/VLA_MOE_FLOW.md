# HiMoE-VLA 从视觉与语言到机器人动作的完整流程

本文说明一次 HiMoE-VLA 控制步中，视觉输入 `V`、语言指令 `L` 和机器人状态 `S` 如何经过条件编码、10 轮 flow 迭代与 MoE 路由，最终生成一段机器人动作。

示例输入：

```text
V：主摄像头和腕部摄像头看到桌上的红杯与托盘
L："把红杯放进托盘"
S：机械臂末端位置、姿态和夹爪状态
```

> `V + L` 还不足以闭环控制机器人；实际推理还需要当前机器人状态 `S`。

## 完整控制步

```mermaid
flowchart TD
    V["V：视觉观测<br/>主摄像头 + 腕部摄像头"]
    L["L：语言指令<br/>把红杯放进托盘"]
    S["S：当前机器人状态<br/>位置 3 + 旋转 3 + 夹爪 2"]

    V --> VP["图像预处理<br/>翻转、缩放与填充到 224 x 224"]
    L --> LP["PaliGemma tokenizer<br/>得到语言 token"]
    S --> SP["归一化并补齐<br/>8 维状态变为 48 维内部表示"]

    VP --> VE["SigLIP 视觉编码器<br/>得到 visual tokens"]
    LP --> LE["语言 embedding<br/>得到 language tokens"]
    VE --> PREFIX["拼接条件前缀<br/>visual tokens + language tokens"]
    LE --> PREFIX
    PREFIX --> PG["普通 PaliGemma 主干<br/>每层编码一次"]
    PG --> KV["缓存每层的 K/V<br/>整个控制步内保持不变"]

    N["高斯噪声 epsilon<br/>形状 10 x 24"] --> X["初始动作轨迹 x_t<br/>t = 1.0"]

    KV --> FLOW
    SP --> FLOW
    X --> FLOW

    subgraph LOOP["Flow 迭代：共 10 轮"]
        FLOW["输入当前 x_t、时间 t、状态 S 和条件 K/V"]
        FLOW --> ST["state projection<br/>生成 1 个 state token"]
        FLOW --> AT["action projection + time embedding<br/>生成 10 个 action tokens"]
        ST --> SUFFIX
        AT --> SUFFIX["后缀序列<br/>1 个 state token + 10 个 action tokens"]
        SUFFIX --> EXPERT["18 层 Expert Gemma<br/>Attention 读取 V/L 的 K/V<br/>MoE 或 Dense MLP 处理后缀"]
        EXPERT --> HEAD["action output projection"]
        HEAD --> VEL["预测速度场 v_t<br/>形状 10 x 24"]
        VEL --> EULER["Euler 更新<br/>x_(t-0.1) = x_t - 0.1 * v_t"]
        EULER --> CHECK{"是否已完成 10 轮？"}
        CHECK -- "否：t 减少 0.1" --> FLOW
    end

    CHECK -- "是：得到 x_0" --> RAW["内部动作块<br/>形状 10 x 24"]
    RAW --> OUT["去除填充维度并反归一化"]
    OUT --> ACTIONS["可执行动作块<br/>10 x 7"]
    ACTIONS --> ENV["依次调用 environment.step<br/>最多执行 10 个动作"]
    ENV --> DONE{"任务成功或达到上限？"}
    DONE -- "是" --> END["episode 结束"]
    DONE -- "否" --> NEXT["读取新的视觉 V' 和状态 S'<br/>语言指令 L 通常不变"]
    NEXT --> CONTROL["开始下一个控制步<br/>重新推理和规划"]
```

## 一轮 Flow 内部的 MoE 计算

下面展开主图中的一轮 `Expert Gemma`。MoE 不负责决定 flow 的迭代次数；它负责在每轮迭代中计算速度场 `v_t`。

```mermaid
flowchart TD
    KV["条件 K/V<br/>来自视觉 V 和语言 L"]
    XT["当前动作轨迹 x_t"]
    T["当前时间 t"]
    S["state token"]

    XT --> FUSE["action projection"]
    T --> TIME["正余弦时间编码"]
    FUSE --> MLP["融合动作与时间的 MLP"]
    TIME --> MLP
    MLP --> A["10 个 action tokens"]
    S --> TOKENS["11 个后缀 tokens"]
    A --> TOKENS

    TOKENS --> ATT["当前 Decoder 层的 Attention"]
    KV --> ATT
    ATT --> H["融合 V、L、S、x_t、t 的 hidden state h"]
    H --> TYPE{"当前层类型"}

    TYPE -- "AS-MoE<br/>层 0、1、16、17" --> AS["由 data_mask 路由<br/>3 个专家中选 top-1"]
    TYPE -- "HB-MoE<br/>层 2-5、12-15" --> ROUTER["router_probs = softmax W_router h"]
    TYPE -- "Dense<br/>层 6-11" --> DENSE["普通 MLP"]

    ROUTER --> TOPK["32 个专家中选 top-4<br/>并归一化选择权重"]
    TOPK --> E1["专家 e1"]
    TOPK --> E2["专家 e2"]
    TOPK --> E3["专家 e3"]
    TOPK --> E4["专家 e4"]
    E1 --> MIX["按路由权重加权求和"]
    E2 --> MIX
    E3 --> MIX
    E4 --> MIX
    SHARED["始终执行的 shared expert"] --> SUM["相加"]
    MIX --> SUM

    AS --> ASOUT["选中 AS expert 输出<br/>加上 AS shared expert 输出"]
    ASOUT --> RES["残差连接"]
    DENSE --> RES
    SUM --> RES
    RES --> MORE{"还有下一 Decoder 层？"}
    MORE -- "是" --> ATT
    MORE -- "否" --> ACTIONTOK["取最后 10 个 action hidden states"]
    ACTIONTOK --> HEAD["线性 action head"]
    HEAD --> VEL["本轮速度场 v_t"]
```

HB-MoE 路由可以概括为：

```text
h = ExpertGemmaSuffix(S, x_t, t | KV(V, L))
p = softmax(W_router * h)
top4_ids, top4_weights = TopK(p, 4)
moe_out = sum(top4_weights[i] * Expert[top4_ids[i]](h))
          + SharedExpert(h)
v_t = ActionHead(h_action)
x_(t-0.1) = x_t - 0.1 * v_t
```

## 三种容易混淆的“10”

| 名称 | 数量 | 含义 |
|---|---:|---|
| Flow 迭代 | 10 轮 | 同一段动作轨迹在模型内部被整体修正 10 次 |
| Action token | 10 个 | 一次推理同时预测的 10 个未来动作位置 |
| 环境动作步 | 最多 10 步 | 推理完成后，客户端实际调用 `environment.step()` 的次数 |

Flow 迭代发生在一次 `policy.infer()` 内部，期间环境不会前进。只有 10 轮全部完成后，客户端才开始执行动作。

## 路由捕获的数据层级

一次控制步内，HB-MoE 的主要路由张量为：

```text
hb_expert_ids[8 个 HB 层, 10 个 flow 迭代, 11 个后缀 token, top-4]
```

如果包含多个控制步，则最外层再增加控制步轴 `T`：

```text
hb_expert_ids[T, 8, 10, 11, 4]
```

因此一个控制步包含：

```text
8 层 x 10 轮 x 11 个 token = 880 个 HB 路由位置
每个位置从 32 个专家中选择 4 个专家
```

HB 路由器直接读取当前 hidden state。该 hidden state 已通过 Attention 融合 `V/L`，并随 `S`、`x_t` 和 `t` 改变，所以不同视觉、指令、控制步或 flow 轮次都可能产生不同路由。

AS-MoE 路由只由 `data_mask` 决定。在当前 LIBERO 配置中，它通常跨 flow 轮次和 token 保持相同，因此采集器会验证后将其折叠存储。

## 对应代码

| 阶段 | 代码位置 |
|---|---|
| 构建视觉、状态与语言观测 | `rollout_with_routes.py::run_one`、桥接层 `preprocess.py::build_policy_observation` |
| V/L 前缀编码 | 上游 `moevla.py::embed_prefix` |
| state/action/time 后缀编码 | 上游 `moevla.py::embed_suffix` |
| 10 轮 flow 与 Euler 更新 | 上游 `moevla.py::sample_actions` |
| AS/HB-MoE 路由和专家合并 | 上游 `modeling_moe.py` |
| 每个控制步的路由捕获 | `himoe_router_recorder.py` |
| 路由数组的存储结构 | `himoe_route_store.py`、`within64_lib.py` |
| 动作块在环境中的执行 | `rollout_with_routes.py::run_one` |
