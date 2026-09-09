# expert-activation v2:同 token、异内部(种子干净重做)

预注册:`prereg.md`(先冻结后运行)。v1 对照:`../expert-activation-future/long-t08`。

## 对齐与量级

门控复算(fp32,对全部 11 token):L2 set-match 0.896 MAE 6.9e-05; L5 set-match 0.872 MAE 7.5e-05; L12 set-match 0.853 MAE 8.3e-05; L15 set-match 0.757 MAE 1.3e-04;

## 主任务:同 basin pair(种子双开 4 折,B1=12 维噪声基线)

B1 基线折-场景 AUC = 0.5836;置换零分布 max-ΔAUC 分位:q50 0.0151 / q95 0.0213 / q99 0.0237。

发现族 200 格(5 通道 x 4 层 x 10 轮)中 FWER p<0.05 的格数:21。

各通道最佳格(ΔAUC vs B1,附 FWER p):

| 通道 | 最佳格 | ΔAUC | FWER p | 未校正 p |
|---|---|---:|---:|---:|
| c1_router | l2_t1 | +0.0165 | 0.348 | 0.005 |
| c2_size | l0_t4 | +0.0350 | 0.005 | 0.005 |
| c3_dcq | l1_t2 | +0.0416 | 0.005 | 0.005 |
| c4_routed_dir | l1_t3 | +0.0126 | 0.841 | 0.005 |
| c7_all | l0_t6 | +0.0463 | 0.005 | 0.005 |

对照臂(不属于发现族)最佳 ΔAUC:

- c5_shared_dir:+0.0646(l2_t0)
- c6_h_dir:+0.0930(l2_t7)

## 描述性 direct AUC(无拟合,与 v1 direct 列可比)

noise7 0.674 / noise24 0.565。

v1 风格(action token 均值向量、4 层池化余弦)按轮:

| tau | routed_dir | shared_dir |
|---:|---:|---:|
| 0 | 0.699 | 0.708 |
| 1 | 0.682 | 0.713 |
| 2 | 0.677 | 0.718 |
| 3 | 0.675 | 0.724 |
| 4 | 0.659 | 0.727 |
| 5 | 0.664 | 0.724 |
| 6 | 0.665 | 0.704 |
| 7 | 0.665 | 0.680 |
| 8 | 0.671 | 0.684 |
| 9 | 0.615 | 0.724 |

(v1 报告的对应列:routed 0.699 / shared 0.708(tau0),shared 峰值 0.727(tau4)。)

## 双重双开稳健性(场景+种子)

B1 = 0.5884;c4_routed_dir_l0_t0 0.5820; c4_routed_dir_l1_t0 0.5921; c4_routed_dir_l2_t0 0.5877; c4_routed_dir_l3_t0 0.5863; c5_shared_dir_l0_t0 0.6122; c5_shared_dir_l1_t0 0.5884; c5_shared_dir_l2_t0 0.6470; c5_shared_dir_l3_t0 0.5885; c6_h_dir_l0_t0 0.5942; c6_h_dir_l1_t0 0.6195; c6_h_dir_l2_t0 0.6337; c6_h_dir_l3_t0 0.6167; c7_all_l0_t0 0.5364; c7_all_l1_t0 0.6041; c7_all_l2_t0 0.5714; c7_all_l3_t0 0.5665;

## 匹配子集回声(噪声距离低于场景中位数的测试 pair)

B1 = 0.4908;c7_all 各格见 summary.json(matched_echo)。

## 次任务(纯描述):成败,按 seed 4 折条件 AUC

| tau | scene+noise | +size | +DCQ | +shared_size | +final_action |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.467 | 0.453 | 0.547 | 0.519 | 0.451 |
| 1 | 0.467 | 0.484 | 0.549 | 0.509 | 0.451 |
| 2 | 0.467 | 0.444 | 0.507 | 0.488 | 0.451 |
| 3 | 0.467 | 0.498 | 0.477 | 0.491 | 0.451 |
| 4 | 0.467 | 0.439 | 0.420 | 0.427 | 0.451 |
| 5 | 0.467 | 0.455 | 0.498 | 0.415 | 0.451 |
| 6 | 0.467 | 0.465 | 0.488 | 0.434 | 0.451 |
| 7 | 0.467 | 0.446 | 0.512 | 0.439 | 0.451 |
| 8 | 0.467 | 0.474 | 0.561 | 0.420 | 0.451 |
| 9 | 0.467 | 0.441 | 0.556 | 0.418 | 0.451 |

(v1 对应:+expert size 在 tau0 达 0.585,基线 0.467。预承诺:该表不得重开 selector 线。)

## 判定(按 prereg 规则)

R1 通过格:c2_size_l0_t4, c2_size_l0_t5, c2_size_l0_t6, c2_size_l0_t7, c2_size_l1_t7, c2_size_l1_t8, c3_dcq_l1_t0, c3_dcq_l1_t2, c3_dcq_l1_t3, c3_dcq_l1_t8, c3_dcq_l3_t6, c3_dcq_l3_t8, c7_all_l0_t4, c7_all_l0_t5, c7_all_l0_t6, c7_all_l1_t2, c7_all_l1_t3, c7_all_l1_t4, c7_all_l1_t8, c7_all_l3_t8, c7_all_l3_t9。
c2_size_l0_t4 vs c5_shared_dir:Δ=-0.0043 CI[-0.0166,+0.0080] → R2 未通过。
c2_size_l0_t4 vs c6_h_dir:Δ=+0.0284 CI[+0.0137,+0.0431] → R2 通过(CI>0)。
c2_size_l0_t5 vs c5_shared_dir:Δ=-0.0208 CI[-0.0338,-0.0078] → R2 未通过。
c2_size_l0_t5 vs c6_h_dir:Δ=+0.0213 CI[+0.0060,+0.0366] → R2 通过(CI>0)。
c2_size_l0_t6 vs c5_shared_dir:Δ=-0.0050 CI[-0.0184,+0.0085] → R2 未通过。
c2_size_l0_t6 vs c6_h_dir:Δ=+0.0227 CI[+0.0116,+0.0339] → R2 通过(CI>0)。
c2_size_l0_t7 vs c5_shared_dir:Δ=+0.0194 CI[+0.0066,+0.0321] → R2 通过(CI>0)。
c2_size_l0_t7 vs c6_h_dir:Δ=+0.0147 CI[+0.0020,+0.0274] → R2 通过(CI>0)。
c2_size_l1_t7 vs c5_shared_dir:Δ=+0.0190 CI[+0.0070,+0.0310] → R2 通过(CI>0)。
c2_size_l1_t7 vs c6_h_dir:Δ=+0.0041 CI[-0.0082,+0.0164] → R2 未通过。
c2_size_l1_t8 vs c5_shared_dir:Δ=+0.0214 CI[+0.0095,+0.0332] → R2 通过(CI>0)。
c2_size_l1_t8 vs c6_h_dir:Δ=+0.0016 CI[-0.0101,+0.0132] → R2 未通过。
c3_dcq_l1_t0 vs c5_shared_dir:Δ=+0.0208 CI[+0.0115,+0.0302] → R2 通过(CI>0)。
c3_dcq_l1_t0 vs c6_h_dir:Δ=-0.0164 CI[-0.0274,-0.0054] → R2 未通过。
c3_dcq_l1_t2 vs c5_shared_dir:Δ=+0.0363 CI[+0.0239,+0.0487] → R2 通过(CI>0)。
c3_dcq_l1_t2 vs c6_h_dir:Δ=+0.0027 CI[-0.0088,+0.0141] → R2 未通过。
c3_dcq_l1_t3 vs c5_shared_dir:Δ=+0.0294 CI[+0.0208,+0.0379] → R2 通过(CI>0)。
c3_dcq_l1_t3 vs c6_h_dir:Δ=-0.0056 CI[-0.0147,+0.0035] → R2 未通过。
c3_dcq_l1_t8 vs c5_shared_dir:Δ=+0.0310 CI[+0.0186,+0.0434] → R2 通过(CI>0)。
c3_dcq_l1_t8 vs c6_h_dir:Δ=+0.0111 CI[-0.0010,+0.0233] → R2 未通过。
c3_dcq_l3_t6 vs c5_shared_dir:Δ=+0.0109 CI[+0.0021,+0.0197] → R2 通过(CI>0)。
c3_dcq_l3_t6 vs c6_h_dir:Δ=-0.0398 CI[-0.0534,-0.0262] → R2 未通过。
c3_dcq_l3_t8 vs c5_shared_dir:Δ=+0.0274 CI[+0.0161,+0.0388] → R2 通过(CI>0)。
c3_dcq_l3_t8 vs c6_h_dir:Δ=-0.0508 CI[-0.0709,-0.0307] → R2 未通过。
c7_all_l0_t4 vs c5_shared_dir:Δ=-0.0030 CI[-0.0162,+0.0101] → R2 未通过。
c7_all_l0_t4 vs c6_h_dir:Δ=+0.0296 CI[+0.0119,+0.0474] → R2 通过(CI>0)。
c7_all_l0_t5 vs c5_shared_dir:Δ=-0.0130 CI[-0.0286,+0.0026] → R2 未通过。
c7_all_l0_t5 vs c6_h_dir:Δ=+0.0291 CI[+0.0093,+0.0490] → R2 通过(CI>0)。
c7_all_l0_t6 vs c5_shared_dir:Δ=+0.0107 CI[-0.0040,+0.0254] → R2 未通过。
c7_all_l0_t6 vs c6_h_dir:Δ=+0.0384 CI[+0.0248,+0.0520] → R2 通过(CI>0)。
c7_all_l1_t2 vs c5_shared_dir:Δ=+0.0395 CI[+0.0281,+0.0510] → R2 通过(CI>0)。
c7_all_l1_t2 vs c6_h_dir:Δ=+0.0059 CI[-0.0053,+0.0171] → R2 未通过。
c7_all_l1_t3 vs c5_shared_dir:Δ=+0.0344 CI[+0.0234,+0.0454] → R2 通过(CI>0)。
c7_all_l1_t3 vs c6_h_dir:Δ=-0.0005 CI[-0.0122,+0.0111] → R2 未通过。
c7_all_l1_t4 vs c5_shared_dir:Δ=+0.0253 CI[+0.0127,+0.0378] → R2 通过(CI>0)。
c7_all_l1_t4 vs c6_h_dir:Δ=-0.0006 CI[-0.0127,+0.0114] → R2 未通过。
c7_all_l1_t8 vs c5_shared_dir:Δ=+0.0279 CI[+0.0150,+0.0407] → R2 通过(CI>0)。
c7_all_l1_t8 vs c6_h_dir:Δ=+0.0080 CI[-0.0045,+0.0206] → R2 未通过。
c7_all_l3_t8 vs c5_shared_dir:Δ=+0.0277 CI[+0.0173,+0.0382] → R2 通过(CI>0)。
c7_all_l3_t8 vs c6_h_dir:Δ=-0.0505 CI[-0.0691,-0.0319] → R2 未通过。
c7_all_l3_t9 vs c5_shared_dir:Δ=+0.0301 CI[+0.0178,+0.0425] → R2 通过(CI>0)。
c7_all_l3_t9 vs c6_h_dir:Δ=+0.0008 CI[-0.0122,+0.0139] → R2 未通过。
措辞阶梯:仅 R1 → 分支通用接口;+R2a(胜 shared)→ 专家分解特有;+R2b(胜 h)→ 分解预提取。任何结果均为提取差距,不得表述为超出完整 hidden state 的信息。
