# 双人关系推理网络方向

## 背景

当前任务边界：

```text
InterHuman
输入：双人前 20% 动作
输出：双人后 80% 动作
暂不加入自然语言条件
暂不追 ReGenNet Table 4
```

用户补充的方向：

```text
用一个神经网络去思考这两个人之间的关系。
```

## 判断

这是比“加入自然语言条件”更合理的论文创新点。原因是它和自动驾驶、机器人、人机共处、行人交互预测等后续应用一致：

```text
真实系统可观察到两个人的历史姿态、距离、朝向、速度和交互趋势；
真实系统通常不会提前获得自然语言描述。
```

因此创新点应从可观测交互关系中学习，而不是依赖 oracle 文本。

## 关系推理网络要学习什么

可以显式建模以下关系：

```text
相对位置：person A root - person B root
相对速度：person A velocity - person B velocity
相对朝向：root orientation difference
身体距离：关键关节间距离或最小距离
交互阶段：靠近、接触、远离、同步、对抗
影响方向：A 影响 B、B 影响 A、双向影响
```

这些关系不需要人工标签，可以从前 20% 观测动作中自监督提取。

## 可行模型形式

### 方案 1：Relation Encoder

在 forecasting 模型前增加一个关系编码器：

```text
obs_motion -> relation_encoder -> relation_embedding
obs_motion + relation_embedding -> future_predictor
```

优点：

- 实现最简单。
- 容易做 ablation：去掉 relation encoder 看性能下降。
- 适合第一版。

### 方案 2：Person Graph / Joint Graph

把两个人建成图：

```text
节点：关节或人
边：人体骨架边 + 跨人交互边
```

跨人边可以由距离、注意力或可学习权重决定。

优点：

- 更像“神经网络思考两人关系”。
- 可解释性更强，可以可视化 cross-person attention。

风险：

- 工程量更大。
- 如果设计太复杂，第一版容易失控。

### 方案 3：Cross-Attention Interaction Module

分别编码 person A 和 person B，然后做双向 cross-attention：

```text
A_obs -> A_tokens
B_obs -> B_tokens
A attends to B
B attends to A
interaction_tokens -> future predictor
```

优点：

- 和 Transformer/diffusion backbone 容易结合。
- 能表达非对称影响关系。

风险：

- 需要设计清楚 token 粒度：按人、按关节、按时间，或三者组合。

## 推荐第一版

第一版建议采用：

```text
Relation Encoder + Future Predictor
```

不要一开始做过大的图网络。最小可行关系特征：

```text
root relative translation
root relative velocity
root orientation difference
inter-person joint distance summary
```

然后用一个小 Transformer/MLP/GRU 编码成：

```text
relation_embedding
```

再注入 forecasting 模型。

## 论文表述

主张不应写成：

```text
我们额外加入一个条件。
```

而应写成：

```text
双人未来动作预测的关键不是单人运动外推，而是对两人交互关系的动态建模。
```

可验证假设：

```text
显式建模双人关系可以降低长期预测误差，并改善交互一致性。
```

## 实验设计

至少需要比较：

```text
Baseline 1: repeat last frame
Baseline 2: no-relation forecasting model
Ours: relation-aware forecasting model
```

指标：

```text
future MSE
rotation MSE
translation MSE
long-horizon MSE
relative distance error
collision / penetration proxy
diversity 或 best-of-K，若后续做多模态预测
```

核心 ablation：

```text
去掉 relation encoder
只用 person A
只用 person B
使用双人但不使用跨人关系
不同观测比例：10% / 20% / 30% / 50%
```

## 当前结论

当前论文方向应修正为：

```text
Interaction-aware two-person human motion forecasting
```

第一阶段实现边界：

```text
InterHuman
前 20% 双人动作 -> 后 80% 双人动作
先建立 no-relation baseline
再加入 relation encoder
```
