# SoMoFormer 与 T2P 的关节建模关系

## 问题

用户指出：

```text
SoMoFormer 是把关节之间做 attention。
T2P 似乎也用上了关节，而不仅仅是时间序列。
两者是什么关系？
```

对比对象：

```text
docs/download/2022-somoformer-multi-person-pose-forecasting-transformers.pdf
docs/download/2024-multi-agent-long-term-3d-human-pose-forecasting-trajectory-conditioning.pdf
```

## 结论

两篇都反对把 pose 当成简单时间序列 flatten 输入：

```text
[T, people, joints, xyz] -> flatten -> temporal model
```

但它们使用“关节”的层次不同：

```text
SoMoFormer：joint trajectory token 是主建模对象，attention 直接发生在 joints / people 之间。
T2P：joint/local pose 是辅助结构，先被编码成 per-agent pose embedding，再用于 global trajectory prediction 和 trajectory-conditioned local pose decoding。
```

换句话说：

```text
SoMoFormer 是 joint-token-first。
T2P 是 trajectory-first + local-pose-conditioned。
```

## SoMoFormer 的关节 attention

SoMoFormer 明确把问题从：

```text
time-sequence series of poses
```

改写成：

```text
joint-sequence series of trajectories
```

具体输入 token 是：

```text
每个人、每个关节、每个坐标维度的一整段轨迹
```

对一个 `N` 人、`J` 关节、XYZ 坐标的场景：

```text
token 数 = N * J * 3
每个 token = 长度为 obs_len + pred_len 的一维 coordinate trajectory
```

未来部分先 constant padding，再做 DCT：

```text
[observed trajectory || padded future trajectory]
-> DCT coefficients
-> Transformer encoder
-> residual / completed trajectory
-> IDCT
```

所以 SoMoFormer 的 attention 直接发生在这些 token 之间：

```text
actor left wrist x-trajectory
reactor pelvis z-trajectory
actor right knee y-trajectory
...
```

它用 learnable embeddings 补充 token 语义：

```text
joint type embedding
person identity embedding
global grid position embedding
```

这让模型能直接学：

```text
同一个人的关节依赖
不同人的关节依赖
相近人的跨人交互
```

## T2P 的关节使用方式

T2P 也不是纯时间序列模型，但它没有像 SoMoFormer 一样把每个 joint coordinate trajectory 作为主 token。

T2P 先把 motion 分成：

```text
global trajectory = hip joint 的全局轨迹
local pose = joints 相对 hip 的局部坐标
```

T2P 的局部关节处理在 pose encoder 中：

```text
local pose displacement
-> DCT
-> body-part sequence / MPBP
-> intra-agent attention
-> per-agent pose embedding
```

这里的 attention 更接近：

```text
一个人内部的 body parts / local joints 之间关系
```

然后 T2P 把 pose embedding 和 trajectory embedding 融合：

```text
trajectory embedding + pose embedding
-> agent-wise graph attention
-> traj-pose embedding
```

注意这个 graph attention 的节点是 agent，不是每个 joint。T2P 的跨人交互主要发生在：

```text
agent-wise trajectory scale
```

而不是：

```text
all joints of all people directly attend to each other
```

最后 T2P 用 trajectory query 条件化 local pose decoder：

```text
future hip trajectory proposals
-> condition local pose prediction
-> global pose = hip trajectory + local pose
```

## 两者关系

### 共同点

两篇都认为简单 temporal sequence 不够：

```text
只按时间帧建模，会忽略人体结构。
多人场景里，只 flatten people/joints 会让交互学习很难。
长期预测需要非递归或整段预测，减少误差累积。
```

两篇都用：

```text
3D joint positions / XYZ
DCT 或 frequency-domain motion representation
Transformer / attention
多人 joint/agent interaction
```

两篇都可以被归到：

```text
structured multi-person pose forecasting
```

### 关键区别

| 维度 | SoMoFormer | T2P |
|---|---|---|
| 主 token | joint-coordinate trajectory | agent trajectory + per-agent pose embedding |
| attention 粒度 | joints / persons 直接 attention | local body-part attention + agent-wise graph attention |
| global/local 解耦 | 移除 global translation，用 grid embedding 补位置 | 显式分解 hip trajectory 和 local pose |
| 预测形式 | trajectory completion / residual refinement | 先 global trajectory，再 trajectory-conditioned local pose |
| 多模态 | 不是主轴 | F 个 trajectory modes，best-of-K |
| 强项 | 保留 joint/person 细粒度结构 | 长时多人场景中高效建模 global intention |
| 风险 | token 多，依赖 embedding / augmentation | local pose 仍不是跨人 joint-level direct attention |

## 对当前项目的含义

当前 `model/forecasting.py` 的 relation 模型是：

```text
两个单人 GRU hidden + root relation GRU hidden
-> single hidden
-> full future [120,2,147]
```

它既没有 SoMoFormer 的：

```text
joint/person token attention
future padding + residual refinement
DCT trajectory completion
```

也没有 T2P 的：

```text
root trajectory / local pose 显式解耦
trajectory-conditioned local pose decoder
```

所以它能赢 concat，但输 independent，并不意外。

## 推荐设计吸收方式

不要把两篇看成互斥路线。更合理的是组合它们的核心思想：

```text
SoMoFormer 负责“怎么保留 joint/person 细粒度结构”
T2P 负责“怎么把 global trajectory 和 local pose 解耦”
```

对 InterHuman SMPL active-vector，可以设计为：

```text
1. independent base predictor
2. root trajectory branch: 预测两个人 root translation / orientation future
3. local pose branch: body rot6d tokens 或 joint tokens 做 attention
4. trajectory conditioning: root future 作为 local pose decoder condition
5. gated residual: pred = independent_base + gate * interaction_delta
```

第一版可以不直接做全量 SoMoFormer token Transformer，而是先做：

```text
independent base + trajectory-conditioned interaction residual + gate
```

然后再升级为：

```text
SMPL active-token Transformer
```

## 写作表达

可以这样写：

```text
SoMoFormer reformulates pose forecasting as joint-trajectory completion, allowing attention over joints and people. T2P takes a complementary route by decoupling global hip trajectories from local poses and conditioning local pose generation on predicted global intentions.
```

中文：

```text
SoMoFormer 的重点是把关节轨迹作为 token，让模型直接学习关节与人之间的 attention；T2P 的重点是把全局 root trajectory 和局部 pose 分开，先预测粗粒度全局意图，再用它条件化细粒度局部动作。
```

不能写：

```text
T2P 就是 SoMoFormer 的直接延续。
```

更准确是：

```text
T2P 和 SoMoFormer 都是对“纯时间序列 pose forecasting”的结构化修正，但一个是 joint-token reformulation，一个是 global-local trajectory conditioning。
```

## 使用来源

- SoMoFormer 本地 PDF：`docs/download/2022-somoformer-multi-person-pose-forecasting-transformers.pdf`
- SoMoFormer arXiv：https://arxiv.org/abs/2208.14023
- T2P 调研文档：`docs/ai/context/20260609-111650-trajectory-conditioning-paper-deep-review.md`
- SoMoFormer 既有分析：`docs/ai/context/20260609-095111-somoformer-architecture-analysis-for-regennet.md`
