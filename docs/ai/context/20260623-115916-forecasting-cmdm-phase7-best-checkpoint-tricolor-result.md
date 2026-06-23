# ForecastingCMDM Phase 7 最佳 checkpoint 三色 skeleton 结果

## 执行边界

用户确认：

```text
使用最佳 checkpoint
真实 source label 生成
只做 skeleton 可视化
三种颜色展示输入前、生成的和正确的
先生成 8 个视频
```

本次未做：

```text
不做 label swap
不做 mesh render
不声明双人互动视频
不声明动作语义控制成功
```

## 使用 checkpoint

```text
save/forecasting/ntu120_label/phase6d_high_noise_onestep_ft_h256_l4_s0_5100/model000005500.pt
```

选择原因：

```text
这是当前 DDIM50 full test 的最佳最终生成 checkpoint。
5600 已出现反弹。
```

## 训练集覆盖核对

微调命令使用：

```text
max_samples = -1
```

即没有人为截断训练集。

训练集 `xsub.train.h5` 在 `window_len=60` 过滤后：

```text
raw_count = 4273
kept_count = 1956
skipped_too_short = 2317
covered_labels = 26
missing_labels = []
min_class_count = 2
handshaking_count = 170
length_min = 60
length_max = 203
length_mean = 76.5046
```

label counts：

```text
[82, 76, 121, 137, 38, 113, 205, 214, 170, 205, 183, 10, 6, 5, 8, 5, 3, 2, 16, 100, 10, 70, 9, 21, 91, 56]
```

结论：

```text
26 个动作标签都训练到了；
但类别非常不均衡，不能说每个动作训练得一样充分。
```

## Source-label 生成

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m eval.eval_label_forecasting_distance \
  --checkpoint save/forecasting/ntu120_label/phase6d_high_noise_onestep_ft_h256_l4_s0_5100/model000005500.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/phase7_best5500_source_label_ddim50_8 \
  --mode ddim50 \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 8 --sample_batch_size 8 \
  --max_samples 8 --seed 0 \
  --save_arrays --overwrite
```

输出目录：

```text
results/forecasting/ntu120_label/phase7_best5500_source_label_ddim50_8
```

整体指标：

```text
num_samples = 8
generated_mse = 0.043046879
generated_mae = 0.084308897
copy_last_mse = 0.044734246
copy_last_mae = 0.066929109
```

判断：

```text
这 8 个样本上 generated MSE 略优于 copy-last，但 MAE 仍高于 copy-last。
```

## 视频生成

新增脚本：

```text
sample/visualize_label_forecasting_tricolor.py
```

颜色：

```text
蓝色 = input obs20
橙色 = generated future40
绿色 = real future40
```

转换：

```text
Rotation2xyz_x
[56,6,T] -> [55,3,T]
num_person = 1
body_only = true
```

正式命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m sample.visualize_label_forecasting_tricolor \
  --source_dir results/forecasting/ntu120_label/phase7_best5500_source_label_ddim50_8 \
  --save_dir results/forecasting/ntu120_label/phase7_best5500_tricolor_skeleton_videos \
  --num_videos 8 \
  --fps 20 --dpi 120 \
  --body_only \
  --overwrite
```

输出目录：

```text
results/forecasting/ntu120_label/phase7_best5500_tricolor_skeleton_videos
```

输出文件：

```text
run_config.json
selection.json
selection.csv
summary.md
conversion_probe.json
arrays/case0000_A001.npz ... arrays/case0007_A008.npz
frames/case0000_A001_first.png ... frames/case0007_A008_first.png
videos/case0000_A001.mp4 ... videos/case0007_A008.mp4
```

生成动作：

```text
A001, A002, A003, A004, A005, A006, A007, A008
```

## 验收

代码检查：

```text
py_compile passed
```

视频检查：

```text
num_videos = 8
all_video_nonempty = true
all_frame_nonempty = true
```

转换检查：

```text
obs_xyz_shape = [20,55,3]
real_future_xyz_shape = [40,55,3]
generated_xyz_shape = [40,55,3]
所有 xyz finite = true
```

视觉抽查：

```text
首帧正常显示蓝色输入 skeleton。
future frame 正常显示橙色 generated 和绿色 real 的叠加。
```

## 边界说明

这些视频只能用于人为定性判断：

```text
生成轨迹和真实轨迹是否接近
是否明显抖动、冻结、漂移
哪些动作 case 更差
```

不能据此声称：

```text
双人互动生成成功
动作语义控制成功
```
