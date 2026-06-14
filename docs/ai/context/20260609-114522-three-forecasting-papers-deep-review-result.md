# 三篇 Multi-person Forecasting 论文深度调研结果

## 调研对象

本次基于本地 PDF 正文抽取结果调研三篇论文：

```text
docs/download/2021-multi-person-3d-motion-prediction-multi-range-transformers.pdf
docs/download/2023-joint-relation-transformer-multi-person-motion-prediction.pdf
docs/download/2023-stochastic-multi-person-3d-motion-forecasting.pdf
```

对应论文：

```text
Multi-Person 3D Motion Prediction with Multi-Range Transformers
NeurIPS 2021

Joint-Relation Transformer for Multi-Person Motion Prediction
ICCV 2023

Stochastic Multi-Person 3D Motion Forecasting
ICLR 2023
```

## 一句话总览

这三篇都在 multi-person 3D motion / pose forecasting 大方向里，但主问题不同：

```text
MRT：看每个人自己的历史动作，再看全场其他人怎么影响他。
JRT：把所有人所有关节当 token，并显式加入关节距离、骨架连接、同人/跨人关系。
DuMMF：不只预测一个未来，而是生成多个合理未来，并兼顾个人动作真实、多人互动合理和多样性。
```

放到前面已经看过的 SoMoFormer / T2P 一起看：

```text
SoMoFormer：关节坐标轨迹 token，直接做 joints/persons attention。
JRT：在 SoMoFormer/MRT 这类 Transformer 上加显式 joint-relation 地图和 relation supervision。
T2P：先预测 global hip trajectory，再用 trajectory conditioning 预测 local pose。
DuMMF：把确定性预测扩展成 stochastic 多未来生成。
MRT：更早期的人级别 local/global Transformer，多人交互建模的基线工作。
```

## MRT：Multi-Range Transformers

### 它研究什么

MRT 研究 deterministic multi-person 3D motion prediction：

```text
输入：多人过去 1 秒 3D joint positions
输出：未来 3 秒多人 3D joint positions
```

它使用的是绝对世界坐标下的 3D joint positions：

```text
x_k^n ∈ R^{3J}
```

也就是每个人每一帧所有关节的 XYZ 坐标。论文特别强调不是把 pose center 固定到原点，而是直接使用 world-coordinate absolute joint positions，因此输入同时包含：

```text
global trajectory + local pose
```

这不是 SMPL/SMPL-X 参数，也不是每个关节的旋转表示。

### 它的方法核心

MRT 的结构可以用一句话概括：

```text
每个人先看自己怎么动，再看全场别人怎么动，最后用自己的当前姿势去问这些信息，生成未来。
```

模块拆开是：

```text
local-range Transformer encoder：
  每个人单独编码自己的历史 motion，保证个人动作连续、自然。

global-range Transformer encoder：
  把所有人、所有时间步放进去建模 social interaction。

spatial positional encoding：
  把 query person 和其他人之间的空间距离显式加进去，帮助模型知道谁离谁近。

Transformer decoder：
  用某个人当前 pose 作为 query，attend local/global features，预测这个人的未来 offset sequence。

motion discriminator：
  用对抗 loss 让长时预测更自然。
```

它不是 SoMoFormer 那种 joint-coordinate token 设计。MRT 的主轴更像：

```text
person-level local temporal motion + scene-level global interaction
```

### 数据、指标和 baseline

数据集：

```text
CMU-Mocap
MuPoTS-3D
3DPW
Mix1 / Mix2，多数据集合成的 9-15 人或 11 人场景
```

指标：

```text
MPJPE without aligning：同时反映 root trajectory 和 pose error
root error
pose error / aligned MPJPE
movement distribution
user study
```

baseline：

```text
LTD
HRI
SocialPool
MRT ablations：w/o Local, w/o Global, w/o discriminator, w/o SPE
```

### 大白话

MRT 像是在问：

```text
这个人自己过去怎么走？
旁边的人过去怎么动？
离他近的人会不会影响他？
那他接下来三秒该怎么动？
```

它的重点是“人和人之间的全局互动”，不是细到“左手腕和对方右膝盖之间的显式关系”。

## JRT：Joint-Relation Transformer

### 它研究什么

JRT 也是 deterministic multi-person motion prediction，但它把问题粒度推进到 joint-to-joint relation：

```text
输入：N 个人、J 个关节的历史 3D joint sequence
输出：未来所有人的 3D joint positions
```

论文定义：

```text
X_NJ ∈ R^{NJ × (Th × 3)}
Y_NJ ∈ R^{NJ × (Tf × 3)}
```

也就是说：

```text
每个 token = 某个人某个关节的一段历史 XYZ 轨迹
```

它使用 3D world coordinate joint positions，并额外拼接 velocity。它不是 SMPL/SMPL-X 参数，也不是旋转坐标。

### 它的方法核心

JRT 的核心判断是：

```text
Transformer attention 可以隐式学关系，但让模型自己猜骨架和关节关系不够。
```

所以它显式构造一条 relation stream：

```text
relative distance matrix D_X：
  每两个关节之间的历史距离。

adjacent matrix A：
  哪些关节有骨骼连接。

connectivity matrix C：
  哪些关节属于同一个人体骨架连通结构。
```

模型是 two-stream：

```text
joint stream：
  编码每个关节的历史位置和速度。

relation stream：
  编码每对关节之间的距离、骨架连接、同人连通约束。
```

关键模块是 relation-aware attention：

```text
普通 attention score = joint query 和 joint key 的相似度
JRT attention score = joint 相似度 + relation feature 给出的关系分数
```

同时它还预测未来 inter-joint distance，用 relation supervision 训练 relation stream。

### 数据、指标和 baseline

数据集：

```text
3DPW-SoMoF
3DPW-SoMoF/RC：作者去掉 camera drift 的修正版
CMU-Mocap
MuPoTS-3D
```

指标：

```text
VIM：SoMoF benchmark 上使用
MPJPE：CMU-Mocap / MuPoTS-3D 上使用
```

baseline：

```text
Zero Velocity
LTD
TRiPOD
DViTA
MRT
FutureMotion
SoMoFormer
```

实验边界：

```text
JRT 在 3DPW-SoMoF/RC、CMU-Mocap、MuPoTS-3D 上明显强于 MRT/SoMoFormer。
但在原始 3DPW-SoMoF 表上，SoMoFormer 的 AVG/900ms 比 JRT 略低；JRT 作者强调原始 3DPW 有 camera drift，修正版 SoMoF/RC 更适合建模互动。
```

### 大白话

JRT 像是在说：

```text
别只让 attention 自己猜。
我直接告诉模型：
哪些关节连着骨头，
哪些关节属于同一个人，
哪些关节和另一个人的关节距离在变化。
```

它和 SoMoFormer 很接近，但 SoMoFormer 是：

```text
关节轨迹 token 之间自己 attention。
```

JRT 是：

```text
关节轨迹 token attention + 显式关系图 + 未来关系监督。
```

## DuMMF：Stochastic Multi-Person 3D Motion Forecasting

### 它研究什么

DuMMF 的问题和 MRT/JRT 不完全一样。它研究的是 stochastic multi-person 3D motion forecasting：

```text
输入：多人历史 3D motion
输出：多个可能的未来 motion sequences
```

论文定义：

```text
输入 {X_n}_{n=1}^N
输出 M 个未来 {{Y_hat_n^m}_{n=1}^N}_{m=1}^M
```

主表示是绝对 3D joint coordinates：

```text
X_n[t], Y_hat_n^m[t] ∈ R^{V × 3}
```

论文也做了 SMPL-X representation / AMASS mesh 实验，但主表的 skeleton setting 仍是 3D joint positions；SMPL-X 是另一个表示实验，不应把整篇论文误读成只做 SMPL-X 参数预测。

### 它的方法核心

DuMMF 的核心不是“再设计一个确定性 Transformer”，而是：

```text
未来不是唯一答案。
多人动作预测应该输出多个合理未来。
```

它认为好的 stochastic multi-person 预测要同时满足三件事：

```text
single-person fidelity：
  每个人自己的动作要真实、连续、身体合理。

multi-person fidelity：
  多个人之间互动要合理，不能各动各的、碰撞或不协调。

overall diversity：
  多个预测结果要有差异，不能五个结果都一样。
```

方法叫 dual-level generative modeling：

```text
local level：
  每个人使用独立 intent code，鼓励个人动作真实和多样。

global level：
  所有人共享同一个 intent code，鼓励整群人的未来互动一致。
```

intent code 包含：

```text
learnable discrete intent codes
continuous intent codes
```

并且它是一个 framework，可以套在不同 generative models 和 deterministic predictors 上：

```text
CGAN
DDPM
SC-MPF
MRT
XIA
Transformer variants
```

### 数据、指标和 baseline

数据集：

```text
CMU-Mocap
MuPoTS-3D
SoMoF
AMASS / SMPL-X representation experiment
```

指标：

```text
ADE / FDE：Best-of-N accuracy
FPD：多样性
rootADE / rootFDE / rootFPD
poseADE / poseFDE / poseFPD
```

baseline：

```text
SC-MPF
Transformer + SC-MPF
XIA
MRT
上述 deterministic predictors 的 CGAN / DDPM stochastic variants
```

### 大白话

DuMMF 像是在说：

```text
你不能只预测一个未来。
同样一段过去，接下来可能左转、右转、停下、避让。
```

但如果每个人都自己随机生成，就会不协调。所以它分两层：

```text
每个人先有自己的想法，
整群人再共享一个社交意图，
这样未来既多样，又不像互相没看见。
```

## 三篇之间的核心差异

| 论文 | 预测类型 | 主要粒度 | 核心问题 | 直观说法 |
|---|---|---|---|---|
| MRT 2021 | deterministic | person/time + scene interaction | 多人长时确定性预测 | 看自己，再看别人 |
| JRT 2023 | deterministic | joint-to-joint relation | 显式注入关节关系 | 给 attention 一张人体关系地图 |
| DuMMF 2023 | stochastic | individual intent + social intent | 多个合理未来 | 每个人有想法，群体有共同社交意图 |

最重要的区别：

```text
MRT 和 JRT 都是“预测一个未来”的 deterministic 方法。
DuMMF 是“预测多个未来”的 stochastic 方法。
```

再细分：

```text
MRT 的交互主要在人/场景级别；
JRT 的交互推进到所有人所有关节之间；
DuMMF 的交互重点是未来分布和多样性，不是单一预测的 joint relation 最小误差。
```

## 和 SoMoFormer / T2P 放在一起怎么理解

### 五篇的方向地图

| 论文 | 大白话定位 |
|---|---|
| MRT | 人级别 local/global Transformer：每个人自己怎么动，别人怎么影响他 |
| SoMoFormer | joint-coordinate trajectory token：每个关节坐标轨迹之间直接 attention |
| JRT | SoMoFormer/MRT 之后的显式关系增强：attention 不只看相似度，还看骨架/距离/同人约束 |
| DuMMF | stochastic 多未来：不是问唯一答案，而是生成多个合理的多人未来 |
| T2P | trajectory-first：先预测人往哪走，再用轨迹条件化身体姿态 |

### 输入组织差异

```text
MRT：
  x_t^n ∈ R^{3J}，按人和时间组织，local encoder 看个人时间序列，global encoder 看多人场景。

SoMoFormer：
  token 是 joint-coordinate trajectory，不是 timestep；attention 主轴是 joints/persons。

JRT：
  token 是 person-joint historical sequence，同时新增 pairwise relation tensor。

DuMMF：
  输入也是多人 3D joint sequence，但额外引入 stochastic latent intents，输出多个未来。

T2P：
  显式拆 global hip trajectory 和 local pose，先 trajectory，再 pose。
```

### 它们是不是同一个问题

只能说是同一个大方向，不是完全同一个实验问题：

```text
MRT / SoMoFormer / JRT：
  更接近 deterministic multi-person 3D pose/motion prediction。

DuMMF：
  stochastic multi-person 3D motion forecasting，多未来生成，指标也换成 Best-of-N 和 diversity。

T2P：
  long-term multi-agent 3D pose forecasting，重点是 global trajectory conditioning，数据口径还包含 JRDB-GMP pseudo 3D pose。
```

所以不能直接说：

```text
谁绝对分数最高。
```

只能在同一 dataset、同一 split、同一指标下比较。

### baseline 是否一样

baseline 有重叠，但不完全一样：

```text
MRT：
  LTD / HRI / SocialPool。

JRT：
  Zero Velocity / LTD / TRiPOD / DViTA / MRT / FutureMotion / SoMoFormer。

DuMMF：
  SC-MPF / Transformer+SC-MPF / XIA / MRT，以及 CGAN/DDPM stochastic variants。

T2P：
  主要和 MRT / JRT / TBIFormer 等 long-term multi-agent baselines 比。
```

JRT 和后续 T2P 会把 MRT 当 baseline；DuMMF 也会把 MRT/XIA 作为 deterministic predictor 或 baseline。但因为 stochastic 评估指标不同，DuMMF 不能和 JRT/MRT 的 MPJPE/VIM 主表直接横向比分数。

## 对当前 InterHuman SMPL forecasting 的启发

当前项目的 150/30/120 InterHuman SMPL active-vector 主协议和这些论文不完全一样：

```text
当前项目：
  deterministic two-person forecasting
  InterHuman SMPL active-vector
  repeat / independent / concat / relation-aware / parameter-matched concat 同口径实证

这些论文：
  大多使用 3D joint positions
  有些是 multi-person 2-15 人
  有些是 stochastic 多未来
  有些使用 SoMoF / CMU / MuPoTS / 3DPW / JRDB-GMP
```

需要保留的论文边界：

```text
不能声称 multi-person forecasting 首创。
不能声称 interaction-aware 首创。
不能声称 explicit relation modeling 首创。
不能声称 long-horizon multi-agent pose forecasting 首创。
```

当前项目更稳的定位是：

```text
在 InterHuman SMPL active-vector 的 two-person deterministic forecasting 协议下，
做同口径 repeat / independent / concat / relation-aware / parameter-matched concat 对照，
证明轻量 root-level relation cues 相对 concat no-relation 有稳定收益，
同时诚实报告它不优于 independent，也不能替代更细粒度 joint-relation 方法。
```

技术上，下一版模型最值得吸收的不是继续堆一个 bigger hidden，而是：

```text
1. 先保留 independent base predictor，避免互动分支破坏单人动作质量。
2. 加 gated interaction residual，只让互动分支修正必要部分。
3. 如果要进一步追 SoMoFormer/JRT，改成 joint/person token 或 relation stream，而不是 single hidden 一次性解双人全未来。
4. 如果追 T2P，显式拆 root trajectory 和 local pose，再用 future trajectory condition local pose。
5. 如果研究问题转成多未来，才引入 DuMMF/CGAN/DDPM 这类 stochastic 设计。
```

## 最终大白话版本

```text
MRT：这个人自己过去怎么动，旁边人怎么影响他。

SoMoFormer：把每个关节坐标的一整段轨迹当 token，让关节和关节、人和人直接互相看。

JRT：SoMoFormer 让关节自己互相看，JRT 进一步把骨架连接、关节距离、同人/跨人关系直接告诉模型。

DuMMF：未来不是一个答案；让每个人有自己的未来意图，同时让整群人共享社交意图，生成多个合理未来。

T2P：先判断每个人往哪走，再根据这个轨迹补身体姿态。
```

对当前论文写作最关键的一句：

```text
我们的工作不能写成“第一次做多人交互预测”，只能写成“在 InterHuman SMPL 双人确定性长时预测协议下，做了严格同口径基线和轻量关系建模实证”。
```
