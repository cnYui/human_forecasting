# Forecasting 5 篇相关论文深度调查与当前创新点定位

## 调查问题

用户要求：

```text
深度调查已下载的 5 篇 multi-person / interaction-aware human motion forecasting 论文，以及它们的方向，
最后判断当前 InterHuman two-person forecasting 工作的创新点在哪里。
```

本文件基于本地 PDF 正文抽取，不是完整系统综述。调查对象为：

```text
docs/download/2021-multi-person-3d-motion-prediction-multi-range-transformers.pdf
docs/download/2022-somoformer-multi-person-pose-forecasting-transformers.pdf
docs/download/2023-joint-relation-transformer-multi-person-motion-prediction.pdf
docs/download/2023-stochastic-multi-person-3d-motion-forecasting.pdf
docs/download/2024-multi-agent-long-term-3d-human-pose-forecasting-trajectory-conditioning.pdf
```

当前项目对比对象：

```text
Interaction-aware joint forecasting of two-person human motion from partial observations
InterHuman SMPL reproduction active vector
window_len=150, obs_len=30, pred_len=120
repeat / independent / concat no-relation / relation-aware
P5 3-seed main table + ablation
P6 qualitative success/close/failure/boundary
```

## 总体判断

当前方向不是空白。

以下概念都已有成熟相关工作：

```text
multi-person human motion forecasting
interaction-aware forecasting
social interaction modeling
explicit relation modeling
long-horizon forecasting
stochastic / multi-modal future prediction
joint trajectory token Transformer
coarse-to-fine trajectory-conditioned pose prediction
```

当前工作不能声称：

```text
首次做 multi-person forecasting。
首次做 two-person forecasting。
首次把 inter-person relation 用于未来动作预测。
首次证明 interaction-aware modeling 有用。
提出 SOTA multi-person forecasting architecture。
```

当前较稳妥的创新定位是：

```text
在 InterHuman SMPL reproduction protocol 下，建立固定 150/30/120 的 deterministic two-person joint forecasting 任务；
以可复现的数据协议和统一指标，系统比较 repeat、independent、concat no-relation、relation-aware；
用 3-seed 主表、参数匹配 concat、relation feature/encoder 消融和 failure-inclusive qualitative，证明一个轻量显式 relation inductive bias 相对 concat no-relation 的稳定收益。
```

这更像一篇：

```text
protocol + empirical study + compact relation-aware baseline
```

不是一篇强 SOTA 架构论文。

## 五篇论文逐篇调查

### 1. Multi-Person 3D Motion Prediction with Multi-Range Transformers

本地文件：

```text
docs/download/2021-multi-person-3d-motion-prediction-multi-range-transformers.pdf
```

方向：

```text
deterministic multi-person 3D motion prediction
Transformer-based long-term forecasting
local individual motion + global social interaction
```

核心任务：

```text
给定多人过去 3D joint positions，预测多人未来 3D joint positions。
输入是 1 秒历史 motion，预测 3 秒未来 motion。
```

表示：

```text
每个人姿态用 J 个 skeleton joints 的 3D Cartesian coordinates 表示。
使用世界坐标下的 absolute joint positions。
```

模型：

```text
local-range Transformer encoder：每个人自己的历史动作。
global-range Transformer encoder：不同人之间的 social interactions。
Transformer decoder：按 person query 预测未来。
DCT/IDCT 编码时间序列。
带 motion discriminator。
```

数据：

```text
CMU-Mocap
MuPoTS-3D
3DPW
Panoptic mix
小人数设置 2-3 persons，大人数设置 9-15 persons。
```

指标：

```text
MPJPE
root error
pose error
movement distribution
1s / 2s / 3s future
```

与当前工作的重叠：

```text
重叠很大：都是多人未来动作预测，都明确建模 social interaction。
该文已经说明 independent single-person prediction 不够。
```

与当前工作的差异：

```text
该文使用 3D joint positions，不使用 SMPL rot6d active vector。
该文不是 InterHuman 数据集。
该文不是固定 150/30/120 protocol。
该文没有你的 repeat/independent/concat/relation-aware 四级同口径实验。
该文的 relation 建模是 Transformer global attention，不是显式 root-level relation features。
```

对当前创新点的影响：

```text
“多人预测需要 social interaction”不是你的创新。
“local individual + global social”也不是你的创新。
你的空间在于 InterHuman 协议和更可解释/更轻量的 relation feature empirical study。
```

### 2. SoMoFormer: Multi-Person Pose Forecasting with Transformers

本地文件：

```text
docs/download/2022-somoformer-multi-person-pose-forecasting-transformers.pdf
```

方向：

```text
multi-person 3D pose forecasting
joint-sequence tokenization
Transformer one-pass prediction
```

核心任务：

```text
给定多人历史 pose，预测多人未来 pose。
SoMoF 3DPW 设置约 16 frames input -> 14 frames future。
CMU/MuPoTS 设置为 15 frames / 1s history -> 45 frames / 3s future。
```

表示：

```text
3D joint coordinates。
每个 input token 是一个 joint coordinate trajectory 的 DCT 编码。
```

模型：

```text
Transformer encoder。
每个 joint coordinate trajectory 作为 token，而不是每个 timestep 作为 token。
使用 joint type embedding、person identity embedding、global position grid embedding。
一次预测全场所有人的未来 pose。
```

数据：

```text
SoMoF benchmark / 3DPW
AMASS
CMU-Mocap
MuPoTS-3D
```

指标：

```text
VIM
MPJPE
```

与当前工作的重叠：

```text
重叠很大：multi-person pose forecasting、长时预测、人与人关系。
```

与当前工作的差异：

```text
SoMoFormer 使用 joint trajectory token 和 full Transformer。
它通过 attention 隐式学习 joint/person 关系，没有手工 root relation feature。
不是 InterHuman SMPL 150/30/120。
没有当前项目这种 independent vs concat no-relation vs relation-aware 的清晰主张拆分。
```

对当前创新点的影响：

```text
“用 Transformer 在多人 pose 上做 forecasting”不是你的创新。
“attention 能学 joint/person relation”也不是你的创新。
当前项目若不实现 SoMoFormer/JRT 类强 baseline，则不能主张 SOTA，只能主张协议内的 controlled empirical evidence。
```

### 3. Joint-Relation Transformer for Multi-Person Motion Prediction

本地文件：

```text
docs/download/2023-joint-relation-transformer-multi-person-motion-prediction.pdf
```

方向：

```text
explicit relation-aware multi-person motion prediction
joint stream + relation stream
relation-aware attention
future relation supervision
```

这是五篇里与当前 relation-aware 主张最危险、最接近的一篇。

核心任务：

```text
给定多人历史 joint positions，预测多人未来 joint positions。
3DPW-SoMoF: historical 1030ms / 16 frames -> future 900ms / 14 frames。
CMU-Mocap: historical 1000ms / 15 frames -> future 3000ms / 45 frames。
```

表示：

```text
3D world coordinate joint positions。
joint matrix: N persons * J joints * time。
relation tensor: joint-to-joint relation。
```

模型：

```text
joint encoder
relation encoder
joint-relation fusion layer
relation-aware attention
joint-aware relation feature update
joint decoder
relation decoder
```

显式 relation 信息：

```text
relative distance matrix
bone adjacency matrix
connectivity matrix
intra-body skeleton constraints
inter-person constraints
```

监督：

```text
不仅监督未来 joints，还监督未来 inter-joint distances。
```

数据：

```text
3DPW-SoMoF
3DPW-SoMoF/RC
CMU-Mocap
MuPoTS-3D
AMASS pretraining / augmentation
```

指标：

```text
VIM
MPJPE
```

与当前工作的重叠：

```text
极强重叠。
它已经明确证明 explicit relation information 和 relation-aware attention 对多人 motion prediction 有帮助。
它也做 relation ablation 和 interaction modeling ablation。
```

与当前工作的差异：

```text
JRT 是 all-joint relation：N*J 到 N*J 的关系。
当前项目是 root-level compact relation features：relative root translation / velocity / distance / orientation。
JRT 使用 3D joint positions，不使用 SMPL rot6d active vector。
JRT 不使用 InterHuman 150/30/120。
JRT 的预测 horizon 在主 benchmark 上是 0.9s 或 3s；当前项目是 120 frames，按数据帧率口径属于固定 80% future protocol。
当前项目重点是 lightweight empirical protocol，不是大型 relation Transformer。
```

对当前创新点的影响：

```text
当前论文绝对不能写“提出显式关系建模用于多人动作预测”。
可以写“受显式关系建模工作启发，本文在 InterHuman two-person forecasting protocol 下研究轻量 root-level relation features 是否足以带来收益”。
```

### 4. Stochastic Multi-Person 3D Motion Forecasting

本地文件：

```text
docs/download/2023-stochastic-multi-person-3d-motion-forecasting.pdf
```

方向：

```text
stochastic multi-person 3D motion forecasting
multi-modal future prediction
dual-level local/global generative modeling
```

核心任务：

```text
给定多人过去动作，生成多个可能的未来动作。
目标不是单一 deterministic future，而是多样且真实的 future distribution。
```

表示：

```text
3D joint coordinates / skeleton representation。
部分实验也涉及 SMPL-X representations 转 skeleton 评估。
```

模型：

```text
DuMMF: Dual-level generative modeling framework。
local level：独立建模 individual motion，提升 single-person fidelity 和 diversity。
global level：建模 social interactions，提升 multi-person fidelity。
learnable latent codes 表示 future intent。
可接 GAN / diffusion / different predictors。
```

数据：

```text
CMU-Mocap
MuPoTS-3D
SoMoF benchmark
2-person / 3-person / more-person scenarios
```

指标：

```text
ADE
FDE
FPD
rootADE / rootFDE / poseADE / poseFDE
rootFPD / poseFPD
Best-of-N
diversity
```

与当前工作的重叠：

```text
都是 multi-person motion forecasting。
都区分 individual motion 与 social interaction。
该文也包含 2-person scenario。
```

与当前工作的差异：

```text
该文是 stochastic / multi-modal forecasting。
当前项目是 deterministic single prediction。
该文关注 diversity、best-of-N、intent codes、GAN/diffusion。
当前项目关注同口径 deterministic MSE 和 relation diagnostics。
```

额外重要风险：

```text
该文在 related work 中提到 Guo et al. 2022 使用 cross-attention 做 two-person motion forecasting，且只适用于 2-person scenarios。
这说明除了本轮 5 篇外，还存在更直接的 two-person forecasting 工作，后续必须补查。
```

对当前创新点的影响：

```text
“双人/多人预测”和“local-global social modeling”不是你的创新。
但 deterministic InterHuman protocol 与 stochastic multi-modal task 不同。
如果后续要做 diffusion / best-of-K / diversity，这篇会成为核心对照。
```

### 5. Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning

本地文件：

```text
docs/download/2024-multi-agent-long-term-3d-human-pose-forecasting-trajectory-conditioning.pdf
```

方向：

```text
long-term multi-agent human pose forecasting
interaction-aware trajectory-conditioned pose forecasting
coarse-to-fine prediction
new long-term multi-agent dataset
```

核心任务：

```text
给定多个 agent 的过去 3D pose，预测长时未来 pose。
重点解决 6+ agents、5s+ long-term、multi-modal global trajectory 与 local pose。
```

表示：

```text
global 3D joint positions。
hip joint trajectory 作为 global trajectory。
local pose = joint positions subtract hip joint。
```

模型：

```text
T2P / Trajectory2Pose。
先预测 multi-modal global trajectories。
再以 trajectory proposals 条件化预测 local poses。
使用 graph-based agent-wise interaction module。
coarse-to-fine: global intent -> local pose。
```

数据：

```text
CMU-Mocap / UMPM
3DPW
JRDB-GMP 新数据集
JRDB-GMP 包含 up to 24 agents，forecast up to 5 seconds。
```

指标：

```text
APE: local motion error
FDE: global trajectory final distance error
JPE: global + local joint error
```

与当前工作的重叠：

```text
重叠强：interaction-aware、long-term、multi-agent pose forecasting。
```

与当前工作的差异：

```text
该文是 multi-agent 6+ / 24 agents 复杂场景。
当前项目是固定 two-person。
该文显式拆 global trajectory 与 local pose。
当前项目在 SMPL active vector 中同时预测 rot6d 与 translation。
该文是 multi-modal trajectory-conditioned。
当前项目是 deterministic。
该文不是 InterHuman。
```

对当前创新点的影响：

```text
long-term interaction-aware forecasting 不是你的创新。
但当前项目可把 T2P 作为更复杂/更强的上位方向，说明本文第一阶段选择 deterministic two-person protocol 是更窄、更可控的实验设置。
```

## 横向矩阵

| 维度 | MRT | SoMoFormer | JRT | DuMMF | T2P | 当前项目 |
| --- | --- | --- | --- | --- | --- | --- |
| multi-person forecasting | 是 | 是 | 是 | 是 | 是 | 是 |
| two-person only | 否 | 否 | 否 | 否，但含 2-person | 否 | 是 |
| deterministic | 是 | 是 | 是 | 否，stochastic | 否/多模态 | 是 |
| explicit relation | 隐式 global attention | 隐式 attention | 强显式 all-joint relation | local/global social intents | graph interaction + trajectory conditioning | 轻量 root-level relation features |
| 数据集 | CMU/MuPoTS/3DPW/Panoptic mix | SoMoF/3DPW/AMASS/CMU/MuPoTS | 3DPW-SoMoF/CMU/MuPoTS | CMU/MuPoTS/SoMoF | CMU/UMPM/3DPW/JRDB-GMP | InterHuman |
| 表示 | 3D joint xyz | 3D joint xyz | 3D joint xyz + relation tensor | 3D joint xyz / skeleton | 3D joint xyz + trajectory/local pose | SMPL rot6d + root translation active vector |
| 主 horizon | 1s -> 3s | 1s -> 3s / SoMoF 0.93s | 1.03s -> 0.9s；1s -> 3s | 1/2/3s stochastic | up to 5s | 30 frames -> 120 frames |
| 指标 | MPJPE/root/pose | VIM/MPJPE | VIM/MPJPE | ADE/FDE/FPD | APE/FDE/JPE | MSE + relation diagnostics |
| 参数匹配消融 | 未作为核心 | 有组件消融 | 有 relation 消融 | 有 local/global/intent 消融 | 有模块消融 | 有 parameter-matched concat + feature/encoder 消融 |
| failure-inclusive qualitative | 有 qualitative | 有 qualitative | 有 qualitative | 有 qualitative | 有 qualitative | 明确 success/close/failure/boundary |

## 当前真正可成立的创新点

### 创新点 1：InterHuman 150/30/120 deterministic two-person forecasting protocol

当前项目把 InterHuman 转成：

```text
fixed window_len=150
obs_len=30
pred_len=120
input: 前 30 帧双人动作
target: 后 120 帧双人动作
representation: [T,2,147] SMPL active vector
```

五篇强相关论文没有使用这个 protocol。

这点的价值不是“发明 forecasting”，而是：

```text
把原本 text-driven / interaction generation 数据资产转成一个可复现的 deterministic two-person forecasting benchmark。
```

注意边界：

```text
如果只说 protocol，论文贡献偏工程/benchmark。
需要配合 relation-aware empirical finding 才能形成方法论文。
```

### 创新点 2：SMPL rot6d active-vector forecasting，而不是直接 3D joint coordinate forecasting

五篇主流工作大多直接预测：

```text
J joints * xyz
```

当前项目预测：

```text
24 joints * rot6d + root translation
```

即每人每帧：

```text
24*6 + 3 = 147
```

这点可能形成“表示层面”的差异：

```text
直接预测姿态参数，而不是先转成 global joint xyz。
```

但风险是：

```text
rot6d MSE 与文献 MPJPE/VIM 不直接可比。
如果写论文，最好补一个 SMPL forward / FK 后的 joint-space metric，至少报告 MPJPE-like 指标。
```

### 创新点 3：轻量、可解释的 root-level relation features

当前 relation features：

```text
relative root translation
relative root velocity
root distance
relative root orientation
```

相比 JRT：

```text
JRT 是 all-joint pairwise relation tensor，包含 joint-to-joint distance、bone adjacency、connectivity。
```

当前项目不是更强架构，而是更轻量：

```text
只用可解释的双人 root-level interaction cues，就能相对 concat no-relation 获得稳定收益。
```

这是可写的，但必须避免写成：

```text
我们首次提出 relation-aware forecasting。
```

应该写成：

```text
We study whether a compact set of interpretable inter-person relation cues is sufficient to improve deterministic two-person forecasting under the InterHuman protocol.
```

### 创新点 4：同口径 baseline 梯度很清楚

当前比较链条：

```text
repeat / zero velocity
independent predictor
concat no-relation predictor
relation-aware predictor
parameter-matched concat
relation no-encoder
single-feature relation variants
```

这点很重要。五篇强相关论文都有 baselines/ablations，但你的论文主张更收窄：

```text
relation-aware 不是只赢 repeat；
也不是只靠参数量；
而是在同 seed、同数据、同 evaluator 下稳定优于 concat no-relation。
```

当前 P5 结果支持：

```text
relation vs concat:
future_mse 3/3 seeds wins
long_mse 3/3 seeds wins
relative_root_distance_error 3/3 seeds wins

relation vs parameter-matched concat:
long_mse 3/3 seeds wins
relative_root_distance_error 3/3 seeds wins
```

这是当前最实在的贡献证据。

### 创新点 5：负结果和边界保留得比较诚实

当前 P5/P6 明确记录：

```text
relation-aware 不优于 independent。
repeat 在部分 relation-style metrics 上仍最低。
relative_orientation_error 和 inter_person_distance_consistency 不是当前 full model 优势。
P6 包含 failure 和 boundary samples。
```

这不一定是传统“性能创新”，但能形成一个更可信的 empirical paper：

```text
显式关系建模何时有效、何时不有效。
```

相比只报成功样本，这对 discussion 很有价值。

## 当前不能成立或很弱的创新点

### 不能声称：multi-person forecasting 是新任务

MRT、SoMoFormer、JRT、DuMMF、T2P 都已经做了 multi-person / multi-agent forecasting。

### 不能声称：interaction-aware 是新方向

五篇都围绕 interaction/social relation/agent interaction 展开。

### 不能声称：显式 relation 是新方法

JRT 已经把 explicit relation 做到很强：

```text
relative distance
bone adjacency
connectivity
relation-aware attention
future distance supervision
```

当前只能说：

```text
我们研究了更轻量、更可解释的 root-level relation cues。
```

### 不能声称：long-horizon forecasting 是新贡献

MRT/SoMoFormer/JRT/DuMMF 都有 3s 预测，T2P 做 5s。

当前的 120-frame future 是否“更长”，取决于 InterHuman 帧率和窗口协议，不能直接与 3s/5s 混比。

### 不能声称：当前方法全面优于 baselines

项目自身结果已经证明：

```text
independent future_mse / long_mse 优于 relation-aware。
```

因此主张必须是：

```text
relation-aware 相对 concat no-relation 和 parameter-matched concat 稳定改善。
```

不是：

```text
relation-aware 是最优预测器。
```

## 当前最合理的论文定位

推荐标题方向：

```text
Interaction-aware Deterministic Forecasting of Two-person Motion from Partial Observations on InterHuman
```

推荐贡献写法：

```text
1. We establish a deterministic two-person forecasting protocol on InterHuman using fixed 150-frame windows, where the first 30 frames are observed and the following 120 frames are forecast jointly.

2. We formulate the forecasting representation as a compact SMPL active vector, preserving per-person rot6d body pose and root translation, and provide a reproducible dataset/normalization/evaluation pipeline.

3. We conduct a controlled empirical study comparing repeat, independent, concat no-relation, and relation-aware predictors under a single evaluator, including 3-seed main results and parameter-matched ablation.

4. We show that compact relation cues improve long-horizon error and relative root distance over concat no-relation and parameter-matched concat, while also identifying clear limitations against independent prediction and relation-style metrics.
```

推荐摘要中的谨慎句：

```text
Rather than claiming a new forecasting paradigm, this work examines how far a compact relation-aware inductive bias can go under a controlled two-person InterHuman forecasting protocol.
```

## 如果想把创新点从“可写”提升到“更强”

### 必补 1：补查 XIA / Guo et al. 2022

DuMMF 正文提到：

```text
Guo et al. 2022 utilize cross-attention for two-person motion forecasting, applicable only to 2-person scenarios.
```

这可能是最接近当前 two-person forecasting 设置的工作之一。

建议单独下载并分析：

```text
Multi-Person Extreme Motion Prediction
```

如果该文与当前工作过近，当前 novelty 还需要进一步收窄为：

```text
InterHuman protocol + SMPL active vector + controlled relation feature ablation
```

### 必补 2：增加 joint-space metric

当前指标主要是 active-vector MSE 和 root relation diagnostics。

为了和五篇论文对齐，建议补：

```text
MPJPE-like joint position error
root trajectory error
maybe VIM-like flattened joint displacement
```

这需要把 rot6d + translation 恢复成 joint xyz。

### 必补 3：至少实现一个强文献 baseline 或文献启发 baseline

如果目标是方法论文，而不只是 protocol/empirical paper，建议实现：

```text
JRT-lite: relation-aware attention over active-vector joints/root
或 SoMoFormer-lite: joint/person token Transformer
或 XIA-like cross-interaction attention
```

当前 relation-aware GRU 能赢 concat，但赢不了 independent。强 baseline 会让论文更有说服力，即使结果显示边界。

### 可补 4：P5.4 observation ratio

当前只做 20% / 80%：

```text
30 -> 120
```

可增加：

```text
50 -> 100
75 -> 75
```

但必须先扩展 dynamic pred_len metrics，且保证原 `pred_len=120` 不漂移。

### 可补 5：把 negative finding 写成贡献

当前最有意思的结果之一是：

```text
independent 比 relation-aware 更低 MSE；
relation-aware 相对 concat 有稳定收益；
repeat 在部分 relation metrics 上最低。
```

这说明：

```text
关系建模不等于整体 MSE 最优；
relation metrics 也不等价于 forecasting accuracy。
```

如果 discussion 写得好，这反而可以成为论文的诚实洞察。

## 最终结论

当前预测工作的创新点不在：

```text
提出多人动作预测任务；
提出 interaction-aware forecasting；
提出显式 relation modeling；
提出 Transformer forecasting 架构；
刷新公开 benchmark SOTA。
```

当前预测工作的创新点在：

```text
1. InterHuman 上固定 150/30/120 的 deterministic two-person joint forecasting protocol。
2. 基于 SMPL active vector 的双人联合未来预测数据/评估闭环。
3. 以 repeat / independent / concat / relation-aware / parameter-matched concat 组成的严格同口径 empirical study。
4. 轻量 root-level relation cues 在 long_mse 和 relative root distance 上稳定优于 concat no-relation 的证据。
5. 明确报告 relation-aware 的失败边界，而不是包装成全面最优。
```

最稳妥的一句话定位：

```text
This work is a controlled empirical study of compact relation-aware inductive bias for deterministic two-person motion forecasting under a newly established InterHuman SMPL 150/30/120 protocol.
```

中文：

```text
本文不是开创多人动作预测方向，而是在 InterHuman SMPL 150/30/120 双人未来预测协议下，系统验证轻量显式关系归纳偏置相对无关系拼接基线的稳定收益与边界。
```
