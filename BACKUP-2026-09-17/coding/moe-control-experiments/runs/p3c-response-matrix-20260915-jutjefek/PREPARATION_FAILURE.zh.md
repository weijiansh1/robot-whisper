# 未启动的准备记录

本目录的准备阶段已写出固定输入和偏置表，但在打印状态 JSON 时遇到 NumPy bool_ 序列化错误。
没有启动 collect，没有模型前向，也没有环境动作。本目录保留，不作为正式实验结果使用。

修正只是在 run_response_matrix_experiment.py 的 prepare() 最后打印时套用已有 jsonable()。
原方向、预算、输入和配置的写入逻辑均未修改。修正后的正式实验使用另一个新目录，不覆盖此处。
本目录 config 中的 runner 源哈希对应修正前版本，故不应再对本目录执行 collect。
