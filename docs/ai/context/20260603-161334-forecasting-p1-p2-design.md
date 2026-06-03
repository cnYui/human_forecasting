# Forecasting P1/P2 详细设计

## 目标

本设计只覆盖第一阶段的最小闭环：

```text
P1: InterHuman forecasting dataset + active vector + normalizer
P2: forecasting metrics + repeat / zero-velocity baseline evaluator
```

不修改当前代码实现，不启动训练。

## 为什么先做 P1/P2

论文主张是：

```text
显式建模双人交互关系可以改善双人未来动作联合预测。
```

但在 relation-aware model 前必须先确认：

```text
1. 数据协议明确。
2. 输入输出 shape 正确。
3. 指标能反映 future motion 和 inter-person relation。
4. repeat baseline 能作为 sanity check。
```

如果没有 P1/P2，后续模型结果无法判断是模型有效，还是数据切分、padding、指标或尺度处理错误。

## 固定协议

第一阶段协议：

```text
dataset: InterHuman
source: dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5
window_len: 150
obs_len: 30
pred_len: 120
input: 前 30 帧双人动作
target: 后 120 帧双人动作
```

长度规则：

```text
T >= 150: 可用
T < 150: 过滤
```

不能 padding，原因：

```text
padding 会把不存在的未来动作变成重复帧，污染 forecasting 指标。
```

## H5 到 active vector

InterHuman H5 原始格式：

```text
motion: [T, 25, 12]
```

slot 含义：

```text
actor_rot6d:        motion[:, :24, 0:6]
actor_translation:  motion[:, 24, 0:3]
reactor_rot6d:      motion[:, :24, 6:12]
reactor_translation:motion[:, 24, 6:9]
```

active vector 格式：

```text
person_dim = 24 * 6 + 3 = 147
two_person_dim = 2 * 147 = 294
```

单人向量：

```text
person = concat(rot6d.reshape(24 * 6), translation)
```

双人样本：

```text
motion_active: [T, 2, 147]
motion_active[:, 0] = actor
motion_active[:, 1] = reactor
```

设计理由：

```text
H5 的 [25,12] 包含为两个人共用 layout 而留下的 zero channel。
直接对 [25,12] 做 MSE 会把无效 zero channel 纳入指标。
active vector 只保留真实监督维度。
```

## Dataset 输出

建议 dataset item：

```text
{
  "obs": Tensor[30, 2, 147],
  "target": Tensor[120, 2, 147],
  "sample_id": str,
  "start": int,
  "length": int,
}
```

训练模式：

```text
对每条 T >= 150 的序列随机选择 start。
window = motion[start:start+150]
obs = window[:30]
target = window[30:]
```

验证/测试模式：

```text
start = floor((T - 150) / 2)
使用 center crop，保证确定性。
```

后续可扩展：

```text
multi-window eval
固定 stride eval
不同 window_len / obs_ratio 协议
```

但第一版不做，避免评估口径过早复杂化。

## Collate 设计

由于 P1 固定窗口，batch 内不需要 padding。

建议 batch：

```text
obs:    Tensor[B, 30, 2, 147]
target: Tensor[B, 120, 2, 147]
meta:   list[dict]
```

不要复用 `data_loaders/tensors.py::ccollate`，原因：

```text
ccollate 的语义是 actor condition / reactor target。
forecasting 的语义是 observed prefix / future suffix。
```

## Normalizer 设计

训练时需要归一化，否则 rotation 和 translation 的尺度差异会影响 loss。

统计来源：

```text
只使用 train split。
只统计 T >= 150 的样本。
统计 active vector 全部时间帧。
```

建议统计维度：

```text
mean: [1, 1, 2, 147]
std:  [1, 1, 2, 147]
```

使用方式：

```text
normalized = (value - mean) / std
original = normalized * std + mean
```

std 防护：

```text
std < eps 的维度设为 1.0
eps = 1e-6
```

保存格式建议：

```text
save/forecasting/.../normalizer.pt
```

也可同时保存摘要：

```text
normalizer.json
```

摘要包含：

```text
dataset
data_path
window_len
obs_len
pred_len
num_train_sequences_used
person_dim
eps
```

## Metrics 输入输出

评估函数输入：

```text
pred:   Tensor[B, 120, 2, 147]
target: Tensor[B, 120, 2, 147]
obs:    Tensor[B, 30, 2, 147]  # 部分交互指标需要 obs[-1]
```

要求：

```text
指标在 original scale 上计算。
不要在 normalized space 报告论文指标。
```

输出：

```text
{
  "future_mse": float,
  "rotation_mse": float,
  "translation_mse": float,
  "short_mse": float,
  "mid_mse": float,
  "long_mse": float,
  "relative_root_distance_error": float,
  "relative_orientation_error": float,
  "inter_person_distance_consistency": float,
}
```

## 指标定义

### future_mse

```text
mean((pred - target)^2)
```

作用：

```text
整体重建误差。
```

### rotation_mse

只取每人的前 144 维：

```text
rot = person[:, :144]
```

定义：

```text
mean((pred_rot - target_rot)^2)
```

### translation_mse

只取每人的最后 3 维：

```text
trans = person[:, 144:147]
```

定义：

```text
mean((pred_trans - target_trans)^2)
```

### short / mid / long mse

对 pred_len=120 做三段：

```text
short: frames 0..39
mid:   frames 40..79
long:  frames 80..119
```

作用：

```text
验证 relation-aware model 是否真正改善长期预测。
```

### relative_root_distance_error

取两人的 root translation：

```text
trans_A = person_A[..., 144:147]
trans_B = person_B[..., 144:147]
dist = norm(trans_A - trans_B)
```

定义：

```text
mean(abs(pred_dist - target_dist))
```

### relative_orientation_error

取两人的 root rot6d：

```text
root_A = person_A[..., 0:6]
root_B = person_B[..., 0:6]
```

转换为 rotation matrix：

```text
R_A = rot6d_to_matrix(root_A)
R_B = rot6d_to_matrix(root_B)
R_rel = R_A^T R_B
```

误差：

```text
angle(R_rel_pred^T R_rel_target)
```

报告单位建议：

```text
radian
```

后续论文表格可额外换算为 degree。

### inter_person_distance_consistency

第一版定义为相对 root distance 的时间变化误差：

```text
delta_dist[t] = dist[t] - dist[t-1]
mean(abs(pred_delta_dist - target_delta_dist))
```

作用：

```text
惩罚两人关系变化趋势错误，而不只看绝对距离。
```

## Repeat / Zero-Velocity Baseline

定义：

```text
pred[:, t] = obs[:, -1]
```

shape：

```text
obs[-1]: [B, 2, 147]
pred:    [B, 120, 2, 147]
```

用途：

```text
1. 验证 evaluator 能跑完整 test split。
2. 提供最低 baseline。
3. 捕获数据切分错误：如果 repeat baseline 异常好，可能发生 target 泄漏或窗口切错。
```

## Sanity Checks

P1 必须通过：

```text
dataset train/val/test 长度等于 T>=150 的样本数。
obs shape = [30,2,147]。
target shape = [120,2,147]。
所有 obs/target 数值有限。
train 模式同一样本多次读取 start 可变化。
eval 模式同一样本多次读取 start 固定。
```

P2 必须通过：

```text
pred == target 时所有 MSE 类指标为 0。
relative_root_distance_error 为 0。
relative_orientation_error 接近 0。
repeat baseline 可完整评估 test split。
metrics 可保存为 yaml/json。
```

## 文件设计

后续实现建议新增：

```text
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
utils/forecasting_motion.py
utils/forecasting_metrics.py
eval/eval_forecasting.py
```

职责划分：

```text
interhuman.py: H5 读取、长度过滤、窗口裁剪。
tensors.py: forecasting collate。
forecasting_motion.py: active vector extract/restore、normalizer。
forecasting_metrics.py: 指标计算。
eval_forecasting.py: repeat baseline 与模型评估入口。
```

本轮只写设计，不创建这些代码文件。

## 验收命令草案

后续实现后应支持类似命令：

```text
python -m eval.eval_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --baseline repeat \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 64 \
  --split test
```

预期输出：

```text
save/forecasting/interhuman/repeat_test_metrics.yaml
```

## 当前不做

```text
不改 train/train_mdm.py。
不改 model/cmdm.py。
不改 diffusion/gaussian_diffusion.py。
不接入 ReGenNet diffusion。
不做 relation-aware model。
不做自然语言或动作标签条件。
不做 SMPL-X 转换。
```

## 下一步

确认本设计后，再进入 P1 实现：

```text
1. active vector extract/restore。
2. InterHumanForecastDataset。
3. forecasting collate。
4. normalizer 统计。
5. shape/finite smoke。
```

P1 完成并验证后，再进入 P2 repeat baseline evaluator。
