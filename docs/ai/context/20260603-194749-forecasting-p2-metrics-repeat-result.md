# Forecasting P2 Metrics + Repeat Baseline 结果记录

## 文档定位

本文记录 P2 实现与验收结果，引用以下上游文档：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260603-194249-forecasting-p2-plan.md
docs/ai/context/20260603-191712-forecasting-p1-dataset-result.md
```

P2 目标是在训练模型前完成 original-scale evaluator 和 repeat baseline 闭环。

## 实现文件

新增：

```text
utils/forecasting_metrics.py
```

修改：

```text
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

### Metrics

实现：

```text
compute_forecasting_metrics(pred, target, obs)
```

输入：

```text
pred:   [B,120,2,147]
target: [B,120,2,147]
obs:    [B,30,2,147]
```

固定输出 key：

```text
future_mse
rotation_mse
translation_mse
short_mse
mid_mse
long_mse
relative_root_distance_error
relative_orientation_error
inter_person_distance_consistency
```

指标均在 original scale 上计算，不使用 normalized space。

### 相对朝向误差

使用项目已有函数：

```text
utils.rotation_conversions.rotation_6d_to_matrix
```

实现细节：

```text
root rot6d -> rotation matrix
R_rel = R_A^T R_B
R_err = R_rel_pred^T R_rel_target
angle = acos(clamp((trace(R_err)-1)/2, -1, 1))
```

与计划文档相比，最终实现没有把上界 clamp 到 `1 - eps`。原因是 `pred == target` 时如果强制上界为 `1 - eps`，完美预测也会产生非零角度，破坏 P2 sanity check。当前实现仍保留 clamp 防止 NaN，并且只有 root rot6d 完全相同时才把对应 angle 置为 0，避免吞掉真实的微小朝向误差。

### Repeat baseline

实现：

```text
pred[:, t] = obs[:, -1]
```

评估入口：

```text
eval/eval_forecasting.py --mode repeat
```

### Eval modes

当前 `eval/eval_forecasting.py` 支持：

```text
dataset_smoke
metrics_sanity
repeat
checkpoint
```

其中 `checkpoint` 仅保留为 `NotImplementedError`，等待 P3/P4 接入。

### 结果落盘

P2 同时保存：

```text
json
yaml
```

YAML 不新增依赖。若环境已有 `yaml` 包则使用 `safe_dump`，否则走纯文本 fallback。

## 验收命令

metrics sanity：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode metrics_sanity \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 64 \
  --num_workers 0 \
  --save_dir save/forecasting/interhuman/p2_metrics_sanity
```

repeat baseline：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode repeat \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 64 \
  --num_workers 0 \
  --save_dir save/forecasting/interhuman/repeat_150_30_120
```

## 验收结果

### 编译检查

命令：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall utils/forecasting_metrics.py eval/eval_forecasting.py
```

结果：

```text
通过
```

### Metrics sanity

结果文件：

```text
save/forecasting/interhuman/p2_metrics_sanity/metrics_sanity.json
save/forecasting/interhuman/p2_metrics_sanity/metrics_sanity.yaml
```

数据：

```text
split: test
num_samples: 508
checked_batch_size: 64
```

sanity 指标：

```text
future_mse: 0.0
rotation_mse: 0.0
translation_mse: 0.0
short_mse: 0.0
mid_mse: 0.0
long_mse: 0.0
relative_root_distance_error: 0.0
relative_orientation_error: 0.0
inter_person_distance_consistency: 0.0
```

### Repeat baseline test

结果文件：

```text
save/forecasting/interhuman/repeat_150_30_120/metrics_test.json
save/forecasting/interhuman/repeat_150_30_120/metrics_test.yaml
```

数据：

```text
split: test
num_samples: 508
batch_size: 64
num_workers: 0
```

repeat 指标：

```text
future_mse: 0.036892867478446695
rotation_mse: 0.03583101653970602
translation_mse: 0.08786168225168244
short_mse: 0.019088565562595063
mid_mse: 0.04046128798774847
long_mse: 0.05112874942032371
relative_root_distance_error: 0.255221389058068
relative_orientation_error: 0.5552304635836384
inter_person_distance_consistency: 0.006041959892430409
```

## 结论

P2 已完成，允许进入 P3：

```text
Independent / Concat Baselines
```

P3 必须复用 `utils/forecasting_metrics.py` 和 `eval/eval_forecasting.py` 中的 evaluator，不允许为可训练 baseline 另写一套指标。
