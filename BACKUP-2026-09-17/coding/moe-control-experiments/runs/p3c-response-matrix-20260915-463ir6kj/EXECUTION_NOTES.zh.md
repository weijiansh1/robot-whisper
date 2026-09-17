# 执行记录

正式目录为本目录，实验设计、方向和模型预算在前向之前冻结。

保留两个没有任何模型前向的记录，不覆盖、不把它们当成实验重复：

1. `../p3c-response-matrix-20260915-jutjefek`：准备末尾打印状态时 NumPy bool_ 无法序列化；只修正控制台输出的 jsonable 转换。
2. `../p3c-response-matrix-20260915-7lwwr1gs`：准备成功，但加载前另一任务占用 GPU，资源检查退出。failure.json 的 model_attempts / model_completed 均为 0。

未停止、重启或修改另一个任务。其自然释放 GPU 后，在本目录启动同一实验。
启动前已核对本目录与第二个目录的 parents、probes、source_hashes 完全相同，包含相同输入和偏置银行 SHA256。
因此不存在观察响应后换方向、重抽随机数或换样本；第一条模型前向发生在本目录。
