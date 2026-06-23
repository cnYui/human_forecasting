# P8 Official SoMoFormer XYZ 骨架视频结果

## 改动

- 新增 `sample/visualize_forecasting_xyz.py`。
- 使用 P8 `official_somoformer_xyz` seed0 checkpoint 对 InterHuman test split 全量推理。
- 自动选择 8 个 qualitative 样本：success / close / failure / boundary 各 2 个。
- 输出双人 24-joint xyz 骨架 mp4，并在视频中标注三色图例：
  - Observed：`#2F6BFF`
  - GT future：`#2CA02C`
  - Pred future：`#FF7F0E`

## 命令

编译：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m compileall sample/visualize_forecasting_xyz.py
```

smoke：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m sample.visualize_forecasting_xyz \
  --max_samples 16 \
  --num_samples 4 \
  --batch_size 8 \
  --fps 10 \
  --dpi 80 \
  --save_dir results/forecasting/interhuman/p8_official_somoformer_xyz_videos_smoke
```

正式输出：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m sample.visualize_forecasting_xyz
```

## 输出

主目录：

```text
results/forecasting/interhuman/p8_official_somoformer_xyz_videos/
```

关键文件：

- `run_config.json`
- `metrics_per_sample_all.json`
- `metrics_per_sample_all.csv`
- `selection.json`
- `selection.csv`
- `summary.md`
- `videos/*.mp4`
- `arrays/*.npy`
- `preview/success_0304_frame.png`

生成视频：

```text
videos/success_0304.mp4
videos/success_0245.mp4
videos/close_0398.mp4
videos/close_0257.mp4
videos/failure_0237.mp4
videos/failure_0094.mp4
videos/boundary_0243.mp4
videos/boundary_0292.mp4
```

## 验证

- `compileall` 通过。
- smoke 生成 4 个 mp4，元数据正常。
- 正式输出生成 8 个 mp4，均为 `640x640`、`20fps`、`7.5s`。
- 8 个 `.npy` 数组均为 finite。
- 数组 shape：
  - `obs_xyz`: `[30, 2, 24, 3]`
  - `gt_xyz`: `[120, 2, 24, 3]`
  - `pred_xyz`: `[120, 2, 24, 3]`
  - `repeat_xyz`: `[120, 2, 24, 3]`
- 已抽取 `preview/success_0304_frame.png` 做视觉检查：图例存在，三色清晰，标题不被遮挡。

## 边界

- 这是 xyz joint skeleton 视频，不是 SMPL/SMPL-X mesh 渲染视频。
- `render/crendermotion.py` 不适用于 P8 xyz 输出，因为它需要 rot6d/SMPL-X 参数。
