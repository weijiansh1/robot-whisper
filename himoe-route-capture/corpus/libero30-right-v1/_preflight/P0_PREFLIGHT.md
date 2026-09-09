# P0 前检核对表 — libero30-right-v1

执行时间：2026-08-13 11:0x
结论：**全部通过，可进 P1**

## 1. checkpoint 完整性（本次最大的首运行风险）

用 bridge 自带的 `himoe_libero_bridge.cli doctor`（逐块 sha256，不是自己搓的），
对 `suites.py::SuiteSpec.weights_sha256` 校验。三套全部逐位吻合：

| suite | 大小 | sha256（实测 == 期望） | norm stats |
|---|---|---|---|
| goal | 8,138,322,389 B | `98ee29d0…df1b953` | ✅ |
| spatial | 8,138,322,389 B | `1029d082…ee87c137` | ✅ |
| object | 8,138,322,389 B | `f9c56615…b2bdaafa` | ✅ |

`spatial` / `object` 此前从未运行过，现已确认**权重字节完好、归一化 `meta/stats.json` 存在**。
剩余首运行风险收敛到「能否正确加载并产出正确 prompt」，由 P3 冒烟覆盖。

原始报告：`doctor-{goal,spatial,object}.json`。三次 `exit=0`。

## 2. 运行环境

| 项 | 实测 |
|---|---|
| model env zarr | 3.0.6 / numcodecs 0.15.1 ✅（容器重启后需重装，本次在） |
| imports（numpy/msgpack/websockets/PIL/imageio） | 全部 ✅ |
| `libero_root` / `upstream_root` | 均有效 ✅ |

## 3. GPU

| 分片 | UUID | 空闲显存 | 用途 |
|---|---|---|---|
| 2g.35gb | `MIG-ed0ef408-…d96a` | 32.4 GB | ✅ 本次采集 A 道 |
| 1g.35gb | `MIG-60ef5cbe-…78d3` | 32.4 GB | ✅ 本次采集 B 道 |
| 4g.71gb | `MIG-63b1c8d1-…5eba` | — | ❌ 不使用 |

**4g.71gb 上原有的 `routing-cloud-k11-s24` 作业已停**：确认是我们自己的进程
（同容器、我们的脚本、写入 `runs/routing-cloud-k11-s24`），已等其跑完最后一个 recipient
（16/16 全部落盘，客户端自行退出）后 SIGTERM 服务器。
另有一个 138 GB 的计算进程属于**其他租户**（`Insufficient Permissions`），未触碰。

## 4. 磁盘

| 口径 | 值 |
|---|---|
| `df` Avail | 123.2 GB（**本卷 df 报的是 xfs project quota，不可信**） |
| `statvfs ffree*512` | **62.3 GB**（以此为准） |
| 本批预估占用 | ≈ 1.2 GB |

余量充裕。

## 5. 溯源指纹（写入 MANIFEST）

| 项 | 值 |
|---|---|
| `himoe-libero-wrist-fix` git rev | `d779741` |
| `himoe-route-capture` | **非 git 仓库**，改用目录内容指纹 |
| route-capture 树 sha256（71 个 `*.py`） | `df15fdd73b3d6fb3c5e066ebe85ec396d2f8abe40746f063eb37483b94ec1e4e` |

树指纹会在 P2 改代码后变化，**以 P4 开跑那一刻的值为准**写进 MANIFEST。
