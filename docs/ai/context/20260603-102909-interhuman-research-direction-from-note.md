# InterHuman 后续研究方向推断

## 输入线索

`docs/note.md` 记录：

```text
MSE 评价指标
frame number 前20%帧数作为观测数据，预测后80%的双人动作
关节数 维度
动作标签或者自然语言的条件
```

## 判断

导师大概率不是只要求继续做 ReGenNet Table 4 的 1:1 复现，而是建议把当前工程转成一个更明确、可控、可投稿的任务：

```text
给定双人交互动作序列前 20% 帧，预测后 80% 帧的双人未来动作。
```

核心任务更接近：

```text
two-person human interaction motion forecasting
interaction-aware future motion prediction
conditioned two-person motion prediction
```

这和当前 ReGenNet 的 actor-conditioned reaction generation 有关系，但评价口径更简单：优先用 MSE/MPJPE 类重建误差，而不是先卡在 Table 4 的 ST-GCN evaluator、类别标签、recognition checkpoint 和 SMPL-X 口径。

## 最可能的投稿方向

### 方向 A：双人交互动作未来预测基准

定义输入输出：

```text
input:  前 20% 帧，包含 person A 和 person B
output: 后 80% 帧，预测 person A 和 person B
metric: MSE / MPJPE / root translation error / rotation error
```

优点：

- 和导师笔记完全一致。
- 不依赖 InterHuman-AS Table 4 的 recognition checkpoint。
- 当前 H5 loader 和训练链路可以复用。
- 容易做 ablation：是否使用交互 loss、是否使用文本或动作标签、不同观测比例。

风险：

- 只用 MSE 容易被认为贡献偏工程，需要有模型创新或任务设定创新。

### 方向 B：条件增强的双人动作预测

在方向 A 基础上加入条件：

```text
condition 1: action label
condition 2: natural language caption
condition 3: actor/reactor role
```

问题定义：

```text
observe first 20% frames + condition -> predict future 80% two-person motion
```

优点：

- 能把 InterHuman 的文本优势用起来。
- 比单纯 MSE 预测更像一篇生成/多模态论文。
- 可以比较无条件、动作标签条件、自然语言条件三种设置。

风险：

- 当前本地数据还没有 text/caption 目录，需要补数据。
- 如果只有 action label，没有自然语言，也可以先做弱条件版本。

### 方向 C：在线反应预测

更贴近 ReGenNet：

```text
input: actor 前 20% 或 actor 当前/历史动作
output: reactor 后 80% 反应动作
```

或者：

```text
input: 双人前 20% + actor 后续动作
output: reactor 后续动作
```

优点：

- 和 ReGenNet 的 action-reaction synthesis 对齐。
- 任务有明确应用意义：一个人的动作触发另一个人的反应。

风险：

- 如果预测目标只包含 reactor，和 note 中“预测后80%的双人动作”不完全一致。
- online 设置需要严格定义模型可见哪些未来信息。

## 推荐路线

最稳妥路线是：

```text
先做方向 A，保证能完整训练、评估、画图；
再扩展方向 B，把 action label 或 natural language 作为条件；
最后把 ReGenNet 作为 baseline 或 backbone，而不是把论文复现当成最终目标。
```

建议论文题目方向：

```text
Conditioned Forecasting of Two-Person Interactive Human Motion
Interaction-Aware Future Motion Prediction from Partial Dyadic Observations
Language/Action Conditioned Two-Person Motion Forecasting
```

## 当前项目需要转向的工作

1. 明确定义 forecasting dataset：
   - 输入帧：前 20%。
   - 预测帧：后 80%。
   - 人数：2。
   - 表示：SMPL rot6d + root translation 或 joint xyz。

2. 新增 forecasting loader：
   - 从 H5 读取完整序列。
   - 输出 `obs_motion` 和 `target_motion`。
   - 记录 `joint_count`、`feature_dim`、`obs_len`、`pred_len`。

3. 新增简单 baseline：
   - zero velocity / repeat last frame。
   - GRU/Transformer deterministic predictor。
   - ReGenNet/diffusion predictor。

4. 新增 MSE 类评估：
   - 全局 MSE。
   - rotation MSE。
   - translation MSE。
   - MPJPE，如果可转 joint xyz。
   - 按时间段报告，例如短期、中期、长期。

5. 再考虑条件：
   - action label 来源。
   - natural language caption 来源。
   - 是否把 condition 作为文本 embedding 或类别 embedding。

## 和当前 ReGenNet 复现的关系

当前 InterHuman Table 4 复现不是白做，它可以作为：

- 数据预处理基础。
- InterHuman H5 冻结格式。
- 生成模型 backbone。
- interaction loss baseline。
- 论文 related work 和 baseline。

但如果目标是发论文，下一步不应该继续只追 Table 4 1:1 复现。更合适的是把复现结果转化为一个新的、可评估的 forecasting 任务。
