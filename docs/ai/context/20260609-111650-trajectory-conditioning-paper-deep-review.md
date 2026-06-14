# Trajectory2Pose 论文深度调研

## 调研对象

本地 PDF：

```text
docs/download/2024-multi-agent-long-term-3d-human-pose-forecasting-trajectory-conditioning.pdf
```

论文：

```text
Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning
Jaewoo Jeong, Daehee Park, Kuk-Jin Yoon
CVPR 2024 Highlight, pp. 1617-1628
arXiv:2404.05218
代码：https://github.com/Jaewoo97/T2P
```

本次还检查了 CVF 官方页面、arXiv 页面、CVF supplementary zip 和官方 GitHub 仓库。计划文档见：

```text
docs/ai/context/20260609-110848-trajectory-conditioning-paper-research-plan.md
```

## 一句话结论

这篇论文的核心价值不是“用了交互关系”，而是把多人长时预测拆成：

```text
先预测多模态 global hip trajectory
再用每个 trajectory mode 条件化预测 local pose
最后 root trajectory + local pose 合成全局 3D pose
```

这个 coarse-to-fine / trajectory-conditioned 结构正好击中了当前 ReGenNet forecasting 模型的问题：当前 concat/relation 模型把双人 120 帧未来压进一个 single hidden 再一次性解码，缺少 root trajectory 和 local pose 的解耦，也缺少可退回 independent 的 base prediction。

## 论文要解决的问题

论文关注的是 long-term multi-agent 3D human pose forecasting：

```text
输入：多个人过去若干帧 3D pose
输出：多个人未来较长时间的 3D pose
```

作者认为已有方法有两个核心短板：

1. 预测 horizon 太短，多数不超过 3 秒。
2. 多人交互建模低效，把所有人所有 joints 一起建模会导致复杂度随 agents 和 joints 快速膨胀，在 6+ agents / 3s+ 场景中不稳。

论文同时从模型和数据集两个层面解决：

```text
模型：Trajectory2Pose / T2P
数据：JRDB-GMP，基于 JRDB 构造的长时多人 3D pose forecasting 数据集
```

## 方法拆解

### 表示拆解

论文把全局 motion 分成两部分：

```text
global trajectory: 每个人 hip joint 的全局轨迹
local pose: 每个人每个 joint 相对 hip joint 的局部坐标
```

预测目标不是直接生成所有 joints 的全局坐标，而是：

```text
global future pose = predicted global hip trajectory + predicted local pose
```

这个设计的第一性原理是：root / hip trajectory 更接近 global intention，局部肢体 motion 更接近 fine pose details。长时多人预测中，先学粗粒度 intention，再生成细粒度 pose，比直接预测所有 joint coordinates 更容易。

### Pose Encoder

Pose encoder 不在全局坐标中建模所有人的所有 joints，而是在每个人自己的 local hip coordinate 中编码局部动作。

主要结构：

```text
local pose displacement
-> DCT / frequency domain
-> Multi-Person Body-Part 模块
-> intra-agent body-part attention
-> per-agent pose embedding
```

这一块继承 TBIFormer 的 body-part / DCT 思路，但限制在 intra-agent 局部动作上，避免一开始就做全局多人 joint-level attention。

### Trajectory Module

Trajectory module 基于 HiVT 风格建模 agent trajectory interaction。

核心点：

```text
用 hip trajectory segment vector 表示移动方向
在 reference agent 坐标系中归一化邻居 motion
用 graph / message passing 建模 agent-agent interaction
```

这里的交互是 agent-wise trajectory scale，而不是 joint-wise scale。论文声称这样把复杂度从类似 `O(T^2 * N^2 * J^2)` 的全局 joint interaction 降到 `O(T * N^2)` 级别。

### Traj-Pose Module

Traj-Pose module 是论文最关键的交互融合模块：

```text
trajectory embedding + projected pose embedding
-> agent-wise graph attention
-> fused traj-pose embedding
```

直觉是：过去的手势、身体朝向、头身姿态可以暗示一个人的未来 global intention；反过来，未来 global trajectory 也能约束 local pose 的生成。

### Trajectory Decoder

Trajectory decoder 使用 temporal encoder、aggregator 和 MLP 生成多模态 future hip trajectories：

```text
fused embedding
-> temporal encoder
-> graph aggregator
-> MLP
-> F 个未来 trajectory proposals
```

正文主实验使用 `F=6` 个 prediction modes。

### Pose Decoder

Pose decoder 用 trajectory proposal 条件化 local pose：

```text
Q = MLP([pose query, trajectory query])
K/V = pose embedding
Transformer decoder
IDCT
future local pose
```

最终：

```text
future global pose = future trajectory + future local pose
```

这比当前项目的 concat/relation 直接从一个 hidden 输出 `[120,2,147]` 更有结构。

## 训练与评估口径

正文和补充材料描述：

```text
trajectory loss + local pose loss
对 F 个 modes 中和 GT 最接近的 mode 反传 L2 loss
```

补充材料写的是从 `F` 个 predictions 中选 minimum JPE mode 来计算所有 metrics。

但官方仓库当前代码有一个细微差异：

```text
训练：按 trajectory ADE 选择 mode，再计算 trajectory loss 和 joint displacement loss
评估：默认 sampling_method='ade'，按 trajectory ADE 选 mode，然后输出该 mode 的 full pose
```

这意味着论文描述和当前仓库实现存在可复现口径风险。做 related work 时可以引用论文结果，但不能把官方仓库当前实现当成完全等价的论文复现实验。

## 数据集 JRDB-GMP

JRDB-GMP 基于 JRDB 构造。JRDB 原本提供多相机图像、2D pose annotation 和 3D bounding boxes，但没有 3D human pose annotation。论文使用 BEV 这类 monocular 3D pose estimator 从图像中提取 3D pose，再用 2D pose 和 3D bbox 过滤、投影修正、对齐到全局坐标。

正文 Table 1 关键统计：

```text
JRDB-GMP 1s/2s: sample 1153, avg agents 6.8, median 5, max 24
JRDB-GMP 2s/5s: sample 4593, avg agents 6.8, median 5, max 22
```

补充材料还说明：

```text
只考虑 robot 4.5m 内的 agents
序列内至少需要多人存在
1s/2s 每 15 frames 采样，2s/5s 每 3 frames 采样
JRDB 场景包括 17 个 indoor 和 10 个 outdoor
```

重要限制：

```text
JRDB-GMP 不是 mocap 或 marker-based ground truth。
它是 BEV 估计 + annotation refinement 得到的 pseudo 3D pose。
补充材料承认单目估计和遮挡会留下 residual noise。
```

官方 GitHub README 还有一个后续变化：当前仓库上传的是更新后的 JRDB parser，写明 `3D joints => SMPL parameters for pose`，使用 SMPL theta `24x3`。这和 CVPR 正文的 joint-position 表述不是同一层数据口径，需要分开记录。

## 实验结果解读

### 主表 Table 2

对比方法：

```text
MRT
JRT
TBIFormer
Ours / T2P
```

数据集：

```text
CMU-Mocap (UMPM): 1s input / 2s output
3DPW: 0.8s input / 1.6s output
JRDB-GMP: 1s input / 2s output
JRDB-GMP: 2s input / 5s output
```

指标：

```text
JPE: global + local full joint error
APE: hip-aligned local joint error
FDE: hip trajectory final distance error
```

最重要的 JRDB-GMP 2s/5s @5s：

```text
JPE:
MRT 474.0
JRT 538.8
TBIFormer 481.3
T2P 390.4

APE:
MRT 101.9
JRT 120.2
TBIFormer 102.9
T2P 94.7

FDE:
MRT 454.8
JRT 497.2
TBIFormer 458.8
T2P 361.0
```

这说明 T2P 的优势主要体现在 long-horizon global trajectory 和 full global pose 上，local pose APE 也有收益，但幅度相对更小。

### Mode 数消融 Table 4

CMU-Mocap (UMPM) @2s：

```text
F=1 时 T2P APE 154.4, JPE 366.4
F=6 时 T2P APE 151.7, JPE 262.7
```

多模态 mode 对 JPE 的收益非常大。这个结果不能直接类比到当前 deterministic InterHuman 任务，因为当前主协议没有 best-of-K / oracle mode 评估。

### 结构消融 Table 5

JRDB-GMP @5s：

```text
baseline: JPE 471.4, APE 101.7, FDE 457.9
full model: JPE 391.2, APE 91.4, FDE 363.3
```

各组件的含义：

```text
trajectory encoder 使用 local pose embedding
trajectory encoder 建模 agent interaction
pose decoder 使用 trajectory conditioning
```

结论：

```text
local pose embedding 有助于预测 global intention
agent interaction 有助于 trajectory 和 pose
trajectory conditioning 对 APE/local pose 尤其关键
```

### 交互范围消融 Table 6

JRDB-GMP @5s JPE：

```text
T2P w/o interaction: 406.0
T2P interaction <2m: 403.5
T2P interaction <4m: 400.5
T2P interaction all: 390.4
```

这支持论文的主张：多人长时场景里，扩大 agent-wise interaction 范围能提升 global pose forecast。

### Failure Cases

补充材料承认：当 local motion 和 global motion intention 不一致时，T2P 也会失败。例子是边走边做随机 flapping arms 这类动作。

这点对当前项目很重要：root-level relation / trajectory conditioning 不是万能的，尤其无法解释所有 rot6d / limb-level motion。

## 可信度评价

### 强项

1. 结构动机清晰：把 global trajectory 和 local pose 解耦，符合长期预测的层级结构。
2. 交互建模尺度合理：agent-wise trajectory interaction 比 all-joint interaction 更适合多人长时场景。
3. 消融表覆盖关键组件：local pose embedding、agent interaction、trajectory conditioning、interaction range 都做了。
4. 新数据集确实补了 5s / 6+ agents 这类缺口。
5. 有官方代码和 supplementary，可追实现细节。

### 风险与边界

1. best-of-6 / oracle mode 评估会放大多模态方法优势。应用中如果只能输出一个未来，结果不能直接照搬。
2. 官方仓库当前实现和论文描述存在差异：LR、mode selection、JRDB parser 的 SMPL 参数化、FPS/窗口设置都有需要核对的地方。
3. JRDB-GMP 是 pseudo 3D pose 数据，不是严格 mocap ground truth；长期多人实验强在规模和现实性，弱在 3D pose 精度。
4. baseline 范围有限：没有把 SoMoFormer、Social Diffusion、stochastic multi-person 3D forecasting 等全部放进主表。
5. 论文主张长时多 agent，但对当前 two-person deterministic SMPL active-vector task 不是同口径证据。
6. GitHub 仓库没有显式 license，且有大量 hardcoded path；复现成本不低。

## 对当前 ReGenNet / InterHuman Forecasting 的影响

当前项目协议：

```text
InterHuman SMPL active vector
window_len=150, obs_len=30, pred_len=120
deterministic two-person forecasting
repeat / independent / concat / relation-aware
```

当前已知结果：

```text
relation-aware 稳定优于 concat no-relation
但不优于 independent
relation-style metrics 也不全面优于 repeat
```

T2P 对这个现象的解释很有价值：

```text
当前 concat/relation 模型缺少 global trajectory / local pose 解耦。
当前 relation branch 只提供 root-level relation hidden，但 decoder 仍用 single hidden 直接生成两个人全部未来。
模型不能自然退回 independent，因此交互信息噪声会伤害单人预测。
```

## 不建议直接照搬的部分

1. 不建议直接把 T2P 的 best-of-6 多模态评估塞进当前主协议。这样会改变论文问题定义，和 P1-P6 deterministic 结果不兼容。
2. 不建议直接换成 JRDB-GMP。当前论文主线是 InterHuman SMPL two-person forecasting，JRDB-GMP 是 in-the-wild pseudo 3D / multi-agent 数据，适合作为后续扩展，不适合替代第一阶段。
3. 不建议直接复刻官方 T2P repo。仓库当前是 post-paper parser，路径硬编码多，论文 joint-position 和当前 SMPL theta parser 口径不一致。
4. 不建议把 T2P 作为“interaction-aware 首创”来写。它本身证明这一方向已有强工作，当前项目必须收窄创新表述。

## 建议迁移的设计

### P7-A：Independent + Trajectory-conditioned Interaction Residual

优先级最高。

结构建议：

```text
base_pred = independent_model(obs)
root_traj_pred = trajectory_branch(obs_root, relation_features)
delta_local = local_pose_branch(obs_local, root_traj_pred, relation_features)
gate = sigmoid(relation_summary)
pred = base_pred + gate * compose(root_delta, local_delta)
```

目标：

```text
保住 independent 的单人预测能力
只在交互确实有帮助时加入 residual correction
```

验收：

```text
future_mse / long_mse 至少接近或不低于 independent
同 seed 至少在 interaction-heavy samples 改善 relative root distance 或 long-horizon error
```

### P7-B：Root Trajectory / Local Pose 解耦指标

当前 active-vector MSE 混合了：

```text
body rot6d
root orientation
root translation
```

建议补：

```text
root trajectory MSE / FDE-like
local pose MSE
joint-space MPJPE-like
interaction distance consistency
```

这样可以验证 relation 模型到底改善了 trajectory，还是只在 normalized active MSE 中产生小幅变化。

### P7-C：Active Token Transformer

如果要借鉴 SoMoFormer/T2P 的结构化建模，应把 active vector 拆开：

```text
person token
joint/root token
rot6d / translation type token
trajectory token over obs+future
```

再考虑：

```text
repeat future padding
residual correction
DCT/IDCT
trajectory-conditioned local pose decoder
```

这比当前 `[B,T,2,147] -> GRU -> hidden -> [B,120,2,147]` 更符合人体 motion 的结构。

## 论文写作定位

引用 T2P 时可以写：

```text
Trajectory2Pose demonstrates that long-term multi-agent pose forecasting benefits from decoupling global trajectory and local pose, using coarse global intention to condition fine local motion generation.
```

当前工作的差异应写成：

```text
Unlike T2P, which studies multi-modal long-term forecasting in multi-agent in-the-wild pseudo-3D pose sequences with best-of-K evaluation, our first-stage protocol focuses on deterministic two-person SMPL active-vector forecasting on InterHuman, with strict same-protocol comparisons among repeat, independent, concat, relation-aware, and parameter-matched concat baselines.
```

不能写：

```text
首次做 interaction-aware forecasting
首次做 multi-person forecasting
首次提出 global-local 解耦
全面优于现有 multi-agent forecasting SOTA
```

## 最终判断

这篇论文应该成为当前 related work 和 P7 设计的重要参考。它对当前项目最关键的启发是：

```text
交互不是简单拼 relation features；
长期预测需要先保住每个人自己的 motion，再把交互作为 trajectory-level / residual-level correction；
global trajectory 和 local pose 必须显式解耦，否则 joint model 很容易输给 independent。
```

当前最务实的下一步不是复刻完整 T2P，而是做一个轻量版本：

```text
independent base + trajectory-conditioned interaction residual + gate
```

如果这个模型不能至少接近 independent，再做更复杂的 token Transformer 或多模态 best-of-K 没有意义。

## 使用来源

- CVF official paper page: https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_Multi-agent_Long-term_3D_Human_Pose_Forecasting_via_Interaction-aware_Trajectory_Conditioning_CVPR_2024_paper.html
- CVF paper PDF: https://openaccess.thecvf.com/content/CVPR2024/papers/Jeong_Multi-agent_Long-term_3D_Human_Pose_Forecasting_via_Interaction-aware_Trajectory_Conditioning_CVPR_2024_paper.pdf
- CVF supplementary material: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Jeong_Multi-agent_Long-term_3D_CVPR_2024_supplemental.zip
- arXiv: https://arxiv.org/abs/2404.05218
- Official GitHub: https://github.com/Jaewoo97/T2P
- 当前项目对照：`model/forecasting.py`
- 当前项目相关记录：`docs/ai/context/20260609-095111-somoformer-architecture-analysis-for-regennet.md`
