# ReGenNet 长预测处理方式对 NTU xyz 的参考结论

## 用户问题

用户指出当前 NTU 双人 xyz 视频后段误差越来越大，要求参考当前 ReGenNet 项目中类似问题的解决方式。

## ReGenNet 现有做法

### 1. 不只看整段平均误差

InterHuman active-vector evaluator 明确拆分：

```text
short_mse: 1-40
mid_mse:   41-80
long_mse:  81-120
```

xyz evaluator 也对应拆分：

```text
short_joint_mse
mid_joint_mse
long_joint_mse
```

结论：

```text
ReGenNet 不是只用 full average 判断模型，而是把 long horizon 单独作为主指标。
当前 NTU pred_len=40 应迁移为 short/mid/long 或 per-frame/last10 指标。
```

### 2. 关系指标是硬监控项

ReGenNet active-vector evaluator 监控：

```text
relative_root_distance_error
relative_orientation_error
inter_person_distance_consistency
```

xyz evaluator 监控：

```text
root_translation_error
relative_root_distance_error
inter_person_distance_consistency_xyz
```

结论：

```text
双人视频后段看起来漂，通常是 root translation 或 relative root distance 漂；
这些指标必须进入 NTU full test 和 case-level 视频 summary。
```

### 3. T2P-style 结构把 root trajectory 和 local pose 分开

`T2PInterHumanXYZ.training_loss` 使用：

```text
xyz_loss
root_loss
local_loss
```

对应思想：

```text
先预测 root trajectory；
再用 trajectory condition 预测 local pose；
最后组合成 full xyz。
```

结论：

```text
NTU 当前 Transformer 直接输出 full xyz delta，root 漂移会被 55 joints 平均稀释；
应至少加 root/local auxiliary loss，更进一步可改成 T2P-style root/local 解耦。
```

### 4. JRT-style 结构显式学习 joint relation

`JointRelationTransformerXYZ.training_loss` 使用：

```text
pose_mse + relation_weight * future_relation_l1 + aux_weight * aux_losses
```

relation target 是 future joints 的：

```text
exp(-joint_distance)
```

结论：

```text
对双人互动视频，只压 joint MSE 不够；
NTU 可以先加 low-cost relation loss：root distance / pairwise root-joint distance；
若仍不稳，再迁移 JRT 的 joint-pair relation stream。
```

### 5. 原始 CMDM/ReGenNet diffusion 使用多个几何约束

`GaussianDiffusion.training_losses` 里除主 MSE 外，还有：

```text
rcxyz_mse
vel_xyz_mse / vel_mse
foot contact loss
relative orientation loss
relative body pose loss
relative root translation loss
```

结论：

```text
ReGenNet 的完整动作生成并不是单一 MSE；
当前 NTU xyz direct predictor 应迁移 velocity / root / relative relation 这些稳定项。
foot contact 暂时不迁移，因为 NTU SMPL-X 55 joint 脚索引和接触阈值需要额外验证。
```

## 对当前 NTU xyz 的具体修复方向

### 必做评估迁移

把 NTU pred_len=40 拆成：

```text
short:  1-13
mid:    14-26
long:   27-40
last10: 31-40
```

新增：

```text
short_mpjpe / mid_mpjpe / long_mpjpe / last10_mpjpe
short_xyz_mse / mid_xyz_mse / long_xyz_mse / last10_xyz_mse
per_frame_mpjpe[40]
per_frame_xyz_mse[40]
```

验收必须从：

```text
full average beats copy-last
```

升级为：

```text
full average beats copy-last
long/last10 beats copy-last
case-level video summary 不整体差于 copy-last
```

### 必做 loss 迁移

在 `NTULabelXYZTransformer.training_loss` 中迁移 ReGenNet 的思想：

```text
weighted_xyz_mse
root_loss
local_pose_loss
velocity_loss
acceleration_loss
relative_root_distance_loss
terminal_loss
```

其中：

```text
weighted_xyz_mse: 后段权重更高，linear 1.0 -> 3.0
terminal_loss: 最后 5 帧额外约束
root_loss: pred[:,:,:,0] vs target[:,:,:,0]
local_pose_loss: xyz - root
relative_root_distance_loss: ||root0-root1|| 的 MSE/L1
velocity_loss: 已有，但权重可上调
acceleration_loss: 新增，抑制轨迹趋势发散
```

### 可选结构迁移

如果加 loss 后后段仍明显漂：

```text
第一优先：T2P-style root/local 解耦预测
第二优先：JRT-style relation stream
第三优先：回到 diffusion，但必须先解决 direct predictor 的 long/last10 指标
```

理由：

```text
T2P 解耦直接针对 root 漂移和 local pose 变形；
JRT 对互动关系更强，但显存和实现复杂度更高；
diffusion 会重新引入采样目标与训练目标不一致的问题，不适合作为当前第一修复。
```

## 最小实验建议

从当前 1000-step checkpoint 微调：

```text
lr = 1e-4
num_steps = 3000
velocity_loss_weight = 0.5
mae_loss_weight = 0.1
horizon_weight_end = 3.0
terminal_loss_weight = 0.5
terminal_frames = 5
acceleration_loss_weight = 0.1
root_loss_weight = 1.0
local_loss_weight = 0.5
relative_root_loss_weight = 0.2
```

如果 `long/last10` 仍不过 copy-last，再进入 T2P-style root/local 模型改造。
