# Forecasting P1 Dataset + Normalizer 结果记录

## 文档定位

本文记录 P1 实现与验收结果，引用以下上游文档：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260603-190529-forecasting-p1-plan.md
```

P1 目标是建立 InterHuman forecasting 的唯一数据协议，不进入 P2 metrics / repeat baseline。

## 实现文件

新增文件：

```text
utils/forecasting_motion.py
data_loaders/forecasting/__init__.py
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
eval/eval_forecasting.py
```

未修改旧主路径：

```text
train/train_mdm.py
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
data_loaders/tensors.py::ccollate
```

## 已实现内容

### Active vector

实现：

```text
extract_active_motion(motion_h5) -> [T,2,147]
restore_active_motion(active) -> [T,25,12]
```

映射：

```text
actor:   motion[:, :24, 0:6] + motion[:, 24, 0:3]
reactor: motion[:, :24, 6:12] + motion[:, 24, 6:9]
```

restore 只还原有效 channel，其他 channel 保持 0。

### Dataset

实现：

```text
InterHumanForecastDataset
```

输出：

```text
obs:    [30,2,147]
target: [120,2,147]
meta: sample_id / start / length
```

采样规则：

```text
train: random crop 150 frames
val/test: center crop 150 frames
T < 150: 过滤
```

### Collate

实现：

```text
forecasting_collate(batch) -> obs, target, meta
```

不做 padding。

### Normalizer

实现：

```text
ForecastingNormalizer
compute_forecasting_normalizer
load_forecasting_normalizer
```

统计来源：

```text
train split only
T>=150 train sequences
每条可用序列的全部帧
active vector original scale
```

保存：

```text
save/forecasting/interhuman/p1_dataset_smoke/normalizer.pt
save/forecasting/interhuman/p1_dataset_smoke/normalizer.json
```

## 验收命令

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode dataset_smoke \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 4 \
  --num_workers 0 \
  --save_dir save/forecasting/interhuman/p1_dataset_smoke
```

## 验收结果

命令已成功退出。

数据集长度：

```text
train: 2910
val:   226
test:  508
```

batch shape：

```text
obs:    [4,30,2,147]
target: [4,120,2,147]
```

roundtrip：

```text
active -> h5-like -> active max_abs_error = 0.0
normalize -> denormalize max_abs_error = 1.1920928955078125e-07
```

start 规则：

```text
train 同一样本多次读取 start 可变化。
val/test 同一样本多次读取 start 固定。
```

normalizer 摘要：

```text
num_train_sequences_used: 2910
num_train_frames_used: 1481944
person_dim: 147
eps: 1e-6
```

输出文件：

```text
save/forecasting/interhuman/p1_dataset_smoke/dataset_smoke_summary.json
save/forecasting/interhuman/p1_dataset_smoke/normalizer.json
save/forecasting/interhuman/p1_dataset_smoke/normalizer.pt
```

## 结论

P1 已完成，允许进入 P2：

```text
Metrics + Repeat Baseline
```

P2 必须继续使用本 P1 的 `InterHumanForecastDataset`、`forecasting_collate`、active vector 和 normalizer，不得重新定义数据协议。
