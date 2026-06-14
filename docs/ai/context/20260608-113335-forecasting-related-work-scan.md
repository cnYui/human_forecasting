# Forecasting 方向相关工作快速检索

## 检索问题

用户询问当前方向是否已有现成思路和相关论文：

```text
基于部分观测的交互感知双人动作联合未来预测。
```

本次检索为 quick brief，不是完整 systematic review。

## 结论

该方向不是空白领域。已有工作覆盖：

```text
multi-person 3D motion prediction / forecasting
interaction-aware human pose forecasting
social motion forecasting
stochastic multi-person future motion forecasting
action-reaction synthesis
text-driven two-person interaction generation
```

当前项目不能声称“首次做人/双人未来预测”或“首次显式建模交互关系”。

当前可以保留的相对安全主张是：

```text
在 InterHuman SMPL reproduction protocol 下，构建固定 150/30/120 的 deterministic two-person joint forecasting protocol；
以 repeat / independent / concat no-relation / relation-aware 做同口径对照；
证明显式 relation features 相对 concat no-relation 和 parameter-matched concat 带来稳定 long_mse 与 relative root distance 改善。
```

## 强相关论文

### Joint-Relation Transformer for Multi-Person Motion Prediction

来源：

```text
ICCV 2023
https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Joint-Relation_Transformer_for_Multi-Person_Motion_Prediction_ICCV_2023_paper.html
https://arxiv.org/abs/2308.04808
```

相关性：

```text
极强。
同样强调 multi-person motion prediction 中个体历史动作和人与人交互依赖。
显式使用 relation information，包括 relative distance 和 intra-/inter-person physical constraints。
使用 relation-aware attention，并监督 future distance。
```

对当前项目的影响：

```text
当前 relation-aware predictor 的“显式关系特征改善多人预测”不是全新概念。
论文必须把该文作为核心 related work。
当前项目可强调更小、更可解释的 relation feature baseline 和 InterHuman two-person 150/30/120 protocol。
```

### Multi-Person 3D Motion Prediction with Multi-Range Transformers

来源：

```text
arXiv 2021
https://arxiv.org/abs/2111.12073
```

相关性：

```text
强。
该文明确指出人的动作依赖周围人的动作，不应孤立预测每个人。
模型包含 individual motion 的 local-range encoder 和 social interactions 的 global-range encoder。
可同时预测多人未来动作。
```

对当前项目的影响：

```text
independent vs joint/social modeling 的论证已有先例。
当前项目需要避免把“联合预测多人”包装为新概念。
```

### Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning

来源：

```text
CVPR 2024
https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_Multi-agent_Long-term_3D_Human_Pose_Forecasting_via_Interaction-aware_Trajectory_Conditioning_CVPR_2024_paper.html
```

相关性：

```text
强。
该文做 long-term multi-agent 3D human pose forecasting。
用 interaction-aware trajectory conditioning，先预测 multi-modal global trajectories，再条件化预测 local poses。
有 graph-based agent-wise interaction module。
```

对当前项目的影响：

```text
当前项目的 long-horizon 和 interaction-aware 主张已有强相关 CVPR 工作。
区别在于当前项目是 deterministic、两人、InterHuman SMPL active-vector 协议，而不是 multi-agent trajectory-conditioned multi-modal pose forecasting。
```

### Stochastic Multi-Person 3D Motion Forecasting

来源：

```text
ICLR 2023 / arXiv
https://arxiv.org/abs/2306.05421
```

相关性：

```text
强。
该文提出 stochastic multi-person 3D motion forecasting，强调 social properties、motion diversity 和 articulated motion complexity。
框架分 local individual motion 与 global social interactions 两层。
```

对当前项目的影响：

```text
当前 deterministic forecasting 是更窄版本。
后续如做 best-of-K / diversity / diffusion，可把该文作为扩展方向对照。
```

### SoMoFormer: Multi-Person Pose Forecasting with Transformers

来源：

```text
SoMoF benchmark entry / code
https://somof.stanford.edu/result/217/
https://github.com/evendrow/somoformer
```

相关性：

```text
强。
该文面向 multi-person 3D pose forecasting，使用 joint/person/global position embeddings，让模型学习 joints 与 people 之间的关系。
```

对当前项目的影响：

```text
可作为 transformer-based multi-person forecasting 相关工作。
当前项目不是 SoMoF benchmark，也不是 image/trajectory benchmark。
```

### Social-MAE

来源：

```text
arXiv 2024
https://arxiv.org/abs/2404.05578
```

相关性：

```text
中强。
该文做 multi-person motion representation learning，预训练后可用于 multi-person pose forecasting、social grouping、social action understanding。
```

对当前项目的影响：

```text
提示可以把 masked pretraining / representation learning 作为 P7+ 扩展。
当前 P1-P6 不需要引入预训练，否则会改变主线复杂度。
```

## 相邻但不同的方向

### ReGenNet

来源：

```text
CVPR 2024 / arXiv
https://arxiv.org/abs/2403.11882
```

相关性：

```text
相邻但不同。
ReGenNet 做 action-reaction synthesis：给 actor 动作生成 reactor 反应。
当前项目做 joint forecasting：给前 30 帧双人动作，同时预测两个人后 120 帧。
```

影响：

```text
ReGenNet 是当前项目历史工程资产和相邻任务，不是当前 P1-P6 的直接主任务。
```

### Interactive Humanoid

来源：

```text
3DV 2025 / OpenReview
https://openreview.net/forum?id=qFWfgadJVQ
```

相关性：

```text
相邻。
该文做 online full-body motion reaction synthesis，根据 human actor motion 生成 humanoid reaction，并使用 social affordance forecasting。
```

影响：

```text
说明 online reaction synthesis 方向也很活跃。
当前项目若写成 reaction synthesis 会被 ReGenNet / Interactive Humanoid 直接夹击；继续保持 joint forecasting 更清晰。
```

### InterHuman / InterGen 系列

来源：

```text
InterGen / InterHuman text-driven two-person interaction generation
```

相关性：

```text
相邻但不同。
大多是 text-to-two-person motion generation、interaction generation、motion inbetweening，不是固定 obs_len/pred_len 的 forecasting 主协议。
```

影响：

```text
可作为数据集和 human-human interaction generation 背景。
不能把它们当作 forecasting baseline，除非后续专门改造成 prediction protocol。
```

## 对当前论文写法的建议

不要写：

```text
We are the first to forecast multi-person human motion.
We are the first to model inter-person relations for forecasting.
```

建议写：

```text
Existing multi-person forecasting methods show that social interaction modeling is important.
However, two-person interaction forecasting under a fixed InterHuman SMPL protocol remains under-explored in this project setting.
We therefore establish a compact deterministic two-person forecasting protocol and evaluate relation-aware inductive bias against repeat, independent, concat no-relation, and parameter-matched baselines.
```

当前最危险的撞题点：

```text
Joint-Relation Transformer for Multi-Person Motion Prediction
Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning
Stochastic Multi-Person 3D Motion Forecasting
```

当前最稳妥的贡献边界：

```text
不是开创任务，而是在 InterHuman two-person forecasting protocol 下做一个严谨、可复现、消融充分的 relation-aware empirical study。
```
