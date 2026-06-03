# 双人未来预测数据集选择

## 当前任务

当前阶段选择：

```text
给定双人动作前 20% 帧，预测后 80% 帧。
```

暂不继续追 ReGenNet Table 4 完整复现，也暂不做自然语言或动作标签条件预测。

## 本地三套数据状态

### NTU120-AS

路径：

```text
dataset/ntu120/smplx/conditioned/xsub.train.h5
dataset/ntu120/smplx/conditioned/xsub.test.h5
```

统计：

```text
train: 4273
test: 3845
shape: [T, 56, 6]
train frame min/mean/median/max: 18 / 59.17 / 57 / 203
test  frame min/mean/median/max: 17 / 53.24 / 49 / 174
```

优势：

- SMPL-X 表示，和 ReGenNet 原始 NTU/Chi3D 路径一致。
- 有 26 类双人动作类别，可做后续 action-conditioned forecasting。
- 本地已有 `recognition_training/ntu_smplx/checkpoint_0100.pth.tar`。
- 样本数较大，数据形态规整。

不足：

- 序列偏短，README 训练口径是 60 帧；前 20% 只有约 12 帧，后 80% 约 48 帧。
- 动作类别更偏 RGB-D action recognition，交互丰富度不如 InterHuman。

### Chi3D

路径：

```text
dataset/chi3d/smplx/conditioned/chi3d_smplx_train.h5
dataset/chi3d/smplx/conditioned/chi3d_smplx_test.h5
```

统计：

```text
train: 293
test: 74
shape: [T, 56, 6]
train frame min/mean/median/max: 105 / 171.86 / 170 / 346
test  frame min/mean/median/max: 111 / 179.54 / 180 / 317
```

优势：

- SMPL-X 表示，和原项目路径一致。
- 长度适合前 20% 到后 80% 的预测设定。
- 本地已有 `recognition_training/chi3d_smplx/checkpoint_0060.pth.tar`。

不足：

- 样本太少，单独作为主数据集发论文风险高。
- 更适合作为小规模验证或跨数据集补充实验。

### InterHuman

路径：

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5
```

统计：

```text
train: 6021
val: 580
test: 1175
shape: [T, 25, 12]
train frame min/mean/median/max: 6 / 283.08 / 144 / 8231
val   frame min/mean/median/max: 9 / 326.73 / 115 / 5154
test  frame min/mean/median/max: 6 / 301.19 / 124 / 9097
```

优势：

- 样本数最大。
- 序列平均长度最长，最适合“前 20% 观测，后 80% 预测”。
- 数据语义更接近双人交互动作。
- 当前项目已经为 InterHuman 实现 H5 预处理和 loader，并完成多个 smoke。

不足：

- 当前本地是 SMPL 25-slot 表示，不是 NTU/Chi3D 的 SMPL-X 56-slot 表示。
- 本地没有 text/caption 目录，暂时不能做自然语言条件。
- 没有 InterHuman recognition checkpoint；但当前 forecasting 阶段以 MSE 类指标为主，不依赖该 checkpoint。
- 帧长分布很宽，需要在 forecasting loader 中明确最小帧数、裁剪和采样策略。

## 推荐选择

主数据集：

```text
InterHuman
```

理由：

```text
InterHuman 最大、最长、交互语义最匹配当前 forecasting 任务，并且本地已有可用 H5 链路。
```

辅助数据集：

```text
NTU120-AS
```

用途：

```text
后续做 action label 条件预测，或作为更规整的 SMPL-X 对照实验。
```

不建议作为主数据集：

```text
Chi3D
```

理由：

```text
样本量太小，不适合作为第一阶段主线；可作为 qualitative 或 cross-dataset generalization 补充。
```

## 当前边界

第一阶段只做：

```text
InterHuman unconditioned two-person forecasting
input:  前 20% 双人动作
target: 后 80% 双人动作
metric: MSE / rotation MSE / translation MSE
```

暂不做：

```text
Table 4 复现
自然语言条件预测
动作标签条件预测
SMPL-X 转换
ST-GCN recognition evaluator
```

后续如果第一阶段稳定，再考虑：

```text
NTU120-AS action-conditioned forecasting
Chi3D 小规模泛化验证
InterHuman 文本数据补齐后的 language-conditioned forecasting
```
