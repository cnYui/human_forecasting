# SoMoFormer 是否是时序输入的澄清

## 用户问题

用户指出：

```text
SoMoFormer 不是时序输入的，它只是关节输入的。
按这句话回论文核对：这个理解是否正确？
```

## 核对结论

用户的判断在“Transformer token / query 组织方式”这个层面是正确的。

更精确表述：

```text
SoMoFormer 不是 time-step token / time-sequence query 输入；
SoMoFormer 是 joint-coordinate trajectory token 输入。
```

但不能说 SoMoFormer 完全没有时间序列信息，因为每个 joint token 内部包含一整段时间轨迹：

```text
每个 token = 某个人某个关节某个 XYZ 坐标在 obs+future padding 上的 trajectory
```

论文原文明确写：

```text
reformulate ... from predicting a time-sequence series of poses to predicting a joint-sequence series of trajectories
```

并说明：

```text
one input query for each joint, rather than for each timestep
N × J × 3 input queries
each input query is a vector of size (t + T) containing one cartesian dimension of a joint trajectory over all timesteps
```

所以最准确的理解是：

```text
时间维度被装进每个 joint-coordinate token 的特征向量里；
Transformer 的 attention 维度是 joints/persons，不是 timesteps。
```

## 和 T2P 的对比修正

之前把 T2P 简称为“时序输入”容易误导。更准确是：

```text
T2P 的任务输入是 observed pose history sequence；
但内部也不是简单 time-step flatten。
```

T2P 内部：

```text
trajectory branch: past hip trajectory time sequence
pose branch: local pose displacement -> DCT/body-part/intra-agent attention
traj-pose fusion: agent-wise graph attention
```

所以对比应写成：

```text
SoMoFormer：关节坐标轨迹作为主 token，attention 直接在 joints/persons 之间做。
T2P：时间历史仍作为任务输入，但把 root trajectory 和 local pose 分支建模；关节先变成 local body-part/pose embedding，再参与 trajectory conditioning。
```

## 推荐术语

以后避免说：

```text
SoMoFormer 不是时序输入
```

因为这可能被理解为它没有使用历史时间序列。建议写：

```text
SoMoFormer 不是按时间步组织 token，而是按 joint-coordinate trajectory 组织 token。
```

中文短句：

```text
SoMoFormer 的 token 是关节轨迹，不是时间帧。
```

## 对当前项目的含义

当前 `model/forecasting.py` 的输入组织是：

```text
obs [B,T,2,147] -> GRU over T -> hidden -> full future
```

这是真正的 time-sequence encoding。它和 SoMoFormer 的差别不是有没有时间信息，而是：

```text
当前模型 attention/encoding 主轴是时间；
SoMoFormer attention 主轴是 joint/person token，时间轨迹被编码到 token feature 中。
```

因此，如果要借鉴 SoMoFormer，需要改的是 tokenization：

```text
从 [T, person, active_dim] 时间序列
改为 [person, joint/root/type] 的 trajectory token
```

## 使用来源

- `docs/download/2022-somoformer-multi-person-pose-forecasting-transformers.pdf`
- `docs/download/2024-multi-agent-long-term-3d-human-pose-forecasting-trajectory-conditioning.pdf`
- `docs/ai/context/20260609-112253-somoformer-vs-t2p-joint-attention-relation.md`
