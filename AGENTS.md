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

## 下一步

- 实现 `--grad_accum_steps`。
- P5 前必须确认 InterHuman 类别标签来源和 recognition checkpoint 路线。
