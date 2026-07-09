# 双人预测视觉拟合训练规则分层设计

## 目标

用户当前关注点不是单纯指标变好，而是：

```text
生成/预测出来的双人 future 在视觉上更贴近真实结果。
```

因此训练规则不能只依赖 `xyz_mse/mpjpe`，也不能只引入原 ReGenNet 的 FID。FID 适合看分布是否像真实动作，但不保证单条样本贴近对应 ground truth。

## 建议分层

### 1. 单样本 paired fitting

回答：

```text
预测 future 是否贴近这一条样本的真实 future。
```

训练规则：

```text
xyz loss
mpjpe loss
root translation loss
local pose loss
short / mid / long horizon loss
final frame loss
```

要点：

```text
root 和 local body 分开约束，避免整个人位置对了但姿态错，或姿态对了但两人整体漂移。
```

### 2. 时间动态

回答：

```text
动作是否连续、速度是否接近真实、是否抖动或过度平滑。
```

训练规则：

```text
first-step continuity loss
velocity matching loss
acceleration matching loss
jerk / smoothness diagnostic
```

要点：

```text
velocity / acceleration 应对齐真实 future，而不是只惩罚模型自身变化，否则容易变成静止或过度平滑。
```

### 3. 双人互动关系

回答：

```text
两个人之间的相对位置、距离变化和接触关系是否合理。
```

训练规则：

```text
relative root distance loss
relative root velocity loss
inter-person distance change loss
key joint pair distance loss
contact event consistency loss
collision / penetration penalty
```

要点：

```text
只看 root distance 不够。handshake / hug / push 等动作需要手-手、手-身体、身体-身体等关键关节距离约束。
```

### 4. 动作语义

回答：

```text
预测动作是否像输入 label 对应的动作类别。
```

训练/评估规则：

```text
action classifier feature loss
action recognition accuracy
class-wise accuracy
class-wise FID
```

要点：

```text
动作语义指标只能作为辅助。它说明像某个动作类别，不说明逐帧贴近真实 future。
```

### 5. 分布质量

回答：

```text
整体生成动作是否像真实动作分布，是否模式坍缩。
```

评估规则：

```text
FID
Diversity
```

要点：

```text
当前 deterministic predictor 不适合把 Multimodality 当核心指标。只有支持同一 obs/action 多次采样后，Multimodality 才更合理。
```

### 6. 可视化分层验收

回答：

```text
指标改善是否真的反映到视频观感。
```

验收规则：

```text
固定 case 可视化
按动作类别采样可视化
失败类型标注：漂移、跳帧、接触错位、过度静止、长期发散
```

要点：

```text
8 个样本视频不能替代 full test 指标，但必须作为错误诊断入口。
```

## 优先级

第一优先级：

```text
short/mid/long horizon error
final frame error
root/local split loss
velocity + acceleration target matching
key joint pair relation loss
```

第二优先级：

```text
action recognition accuracy
FID
class-wise metrics
```

第三优先级：

```text
stochastic multi-sample prediction
multimodality
APD / ADE / FDE / MMADE / MMFDE
```

## 关键边界

如果目标是视觉上贴近正确答案，主优化方向仍应是 paired forecasting loss 和 interaction-aware loss。

原 ReGenNet 的 FID / Accuracy / Diversity 可以补充说明生成质量，但不能作为解决视觉不拟合的主要训练目标。
