# P8 Official SoMoFormer XYZ 骨架视频计划

## 目标

- 使用 P8 `official_somoformer_xyz` seed0 checkpoint 生成 InterHuman test split 的双人 24-joint xyz 骨架视频。
- 视频展示三种颜色并带图例：
  - observed 30 帧
  - ground truth future 120 帧
  - predicted future 120 帧
- 输出到 `results/forecasting/interhuman/p8_official_somoformer_xyz_videos/`。

## 输入

- checkpoint：`save/forecasting/interhuman/p8_official_somoformer_xyz_h256_l6_dct30_s0_5000/model000005000.pt`
- dataset：`dataset/interhuman/smpl/conditioned`
- split：`test`
- 协议：`window_len=150, obs_len=30, pred_len=120`

## 实现决策

- 新增脚本 `sample/visualize_forecasting_xyz.py`，放在 ReGenNet 项目内，因为 P8 dataset、model loader 和 checkpoint 均属于 ReGenNet。
- 不使用 `render/crendermotion.py`，因为该渲染链路需要 rot6d/SMPL-X 参数；P8 输出为 `[T,2,24,3]` xyz joint positions。
- 用 matplotlib Agg 后端和 `FuncAnimation` 生成 mp4，避免依赖显示环境。
- 使用 SMPL 24-joint 常规骨架连接；两个人都绘制，同一种轨迹语义用同一颜色。
- 自动按逐样本指标选择 `success/close/failure/boundary` 各 2 个样本，避免只展示成功样本。

## 验收

- `python -m compileall sample/visualize_forecasting_xyz.py` 通过。
- 用 `--max_samples 16 --num_samples 4` 能生成最小 smoke 视频。
- 默认 seed0 完整 test inference 能生成 8 个样本视频、对应 `.npy`、`selection.json/csv` 和 `summary.md`。
