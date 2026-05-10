# ReGenNet AI 上下文记忆

## InterHuman-AS Table 4

- Table 4 主指标是 `FID / Acc. / Div. / Multimod.`，不是 `R Precision / MM Dist`。
- Table 4 主路径应使用 ST-GCN recognition feature evaluator；text-motion evaluator 只作为后续扩展。
- 当前本地 InterHuman 数据没有 text/caption 目录，也没有 InterHuman recognition checkpoint。
- 当前实现先走 SMPL reproduction attempt；若要精确对齐论文的 SMPL-X 口径，需要单独补 SMPL-X 转换或确认官方 evaluator 口径。

## 已完成进度

- 已新增 InterHuman-AS H5 预处理脚本：`preprocess/interhuman_as.py`。
- 已生成冻结 H5：`dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5`。
- 已为 `data_loaders/a2m/interhuman.py` 增加 H5 读取路径，同时保留在线 `.pkl` loader。
- H5 输出与在线 loader 抽样对齐，最大误差为 0。
- H5 loader 最小训练 smoke 已通过。
- H5 full smoke 已通过：`save/interhuman/h5_full_1000_smoke_noworkers/model000001000.pt`。
- InterHuman 训练当前固定 `num_workers=0`，原因是 PyTorch 1.7 多 worker 在提前按 `num_steps` 停止时会阻塞进程退出。
- 已实现 P3 `--grad_accum_steps`；`num_steps` 表示 optimizer steps，`effective_batch_size=batch_size*world_size*grad_accum_steps`。
- `grad_accum_steps=1` 和 `grad_accum_steps=2` 的短 smoke 已通过，checkpoint 可加载。
- `num_steps=1000, grad_accum_steps=16` 中等 smoke 已通过，耗时约 7 分钟，最终 checkpoint：`save/interhuman/p3_grad_accum_16_1000_smoke/model000001000.pt`。
- 论文尺寸模型 smoke 已通过：`layers=8, latent_dim=512, lambda_orient/body/transl=1, grad_accum_steps=4, num_steps=100`。
- 已修复 InterHuman H5 显式 translation loss 的 NaN：translation slot 使用最后一个 joint slot，而不是硬编码 index 55。

## 下一步

- 验证论文 batch size 近似口径：`layers=8, latent_dim=512, batch_size=1, grad_accum_steps=64, num_steps=1000`。
- P5 前必须确认 InterHuman 类别标签来源和 recognition checkpoint 路线。
