# ForecastingCMDM Phase 7 最佳 checkpoint 三色 skeleton 计划

## 用户确认边界

本次执行边界：

```text
checkpoint: 当前最佳 model000005500.pt
生成口径: 使用真实 source label，不做 label swap
可视化: skeleton only
数量: 8 个 case
颜色: 输入前 20 帧、生成 future40、真实 future40 三色区分
用途: 人为定性判断生成和真实差多少
```

## 使用 checkpoint

```text
save/forecasting/ntu120_label/phase6d_high_noise_onestep_ft_h256_l4_s0_5100/model000005500.pt
```

选择原因：

```text
DDIM50 full test MSE=0.030736648，优于原始 5000 和 copy-last；
5600 已出现反弹。
```

## 训练数据覆盖回答

刚才微调使用：

```text
max_samples = -1
data_path = dataset/ntu120/smplx/conditioned/xsub.train.h5
```

即没有人为截断训练集。

在 `window_len=60` 过滤后，训练集：

```text
kept_count = 1956
covered_labels = 26
min_class_count = 2
handshaking train = 170
```

因此：

```text
每个动作标签都训练到了；
但类别数量不均衡，不能说每个动作标签训练得同样充分。
```

## 执行步骤

### 1. 生成 8 个 source-label matched 样本

使用：

```text
eval/eval_label_forecasting_distance.py
mode = ddim50
max_samples = 8
save_arrays = true
```

输出目录：

```text
results/forecasting/ntu120_label/phase7_best5500_source_label_ddim50_8
```

保存：

```text
generated_future40.npy [8,56,6,40]
real_future40.npy      [8,56,6,40]
obs_motion.npy         [8,56,6,20]
sample_metrics.jsonl
metadata.json
metrics.json
```

### 2. 三色 skeleton 可视化

新增入口：

```text
sample/visualize_label_forecasting_tricolor.py
```

输入：

```text
source_dir = phase7_best5500_source_label_ddim50_8
```

转换：

```text
Rotation2xyz_x
rot6d [1,56,6,T] -> xyz [1,55,3,T]
num_person = 1
jointstype = smplx
```

颜色：

```text
observed obs20       = blue
generated future40   = orange
real future40        = green
```

时间：

```text
frame 0..19: 画 observed
frame 20..59: 同时画 generated 和 real
```

输出目录：

```text
results/forecasting/ntu120_label/phase7_best5500_tricolor_skeleton_videos
```

输出：

```text
videos/caseXXXX_Axxx.mp4
frames/caseXXXX_Axxx_first.png
arrays/caseXXXX_Axxx.npz
selection.json
summary.md
run_config.json
```

## 不声明的内容

不声明：

```text
双人互动视频
动作语义控制成功
mesh render
论文最终定性结果
```

原因：

```text
当前可安全转换的是 [56,6,T] -> 单个 SMPL-X 55-joint skeleton。
视频只用于人为检查 generated future 和真实 future 的贴合程度。
```
