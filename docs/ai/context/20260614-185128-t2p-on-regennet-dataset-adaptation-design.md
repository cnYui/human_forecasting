# T2P 使用 ReGenNet 本地数据集的适配设计

## 问题

用户希望用本地数据集：

```text
/home/rpartx3080/CodeSpace/ReGenNet/dataset
```

来跑 T2P / Trajectory2Pose 思路。

## 本地数据状态

当前最适合第一阶段适配的是 InterHuman：

```text
dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5
shape: [T,25,12]
body_model: smpl
rotation: rot6d
actor_translation/reactor_translation 位于 joint slot 24
```

已有 ReGenNet forecasting pipeline 能从该 H5 生成：

```text
active: [T,2,147]
obs: [B,30,2,147]
target: [B,120,2,147]
xyz: [B,T,2,24,3]
```

NTU120 当前为：

```text
dataset/ntu120/smplx/conditioned/xsub.{train,test}.h5
shape: [T,56,6]
```

Chi3D 当前为：

```text
dataset/chi3d/smplx/conditioned/chi3d_smplx_{train,test}.h5
```

NTU120/Chi3D 是后续扩展候选，不建议作为 T2P 第一版适配入口。

## T2P 官方输入要求

官方 `T2P` forward 依赖的核心字段包括：

```text
input_seq: [B,N,input_time,J*3]
output_seq: [B,N,output_time+1,J*3]
y: [B*N,output_time,2] 或等价 root trajectory target
positions: [B*N,total_time,2]
rotate_mat: [B*N,3,3]
padding_mask: [B*N,total_time]
edge_index: [2,E]
bos_mask: [B*N,input_time]
```

T2P 原始 JRDB-GMP 设定是 multi-agent，不定人数，root trajectory 是 2D 平面轨迹，local pose 是 15 joints 的 3D 坐标或 SMPL theta 派生表示。

## 关键判断

不建议直接修改 `/home/rpartx3080/CodeSpace/T2P` 官方仓库来读取 ReGenNet 数据。

原因：

1. 官方仓库硬编码大量作者机器路径。
2. 官方默认 Hydra config 缺失或不匹配。
3. T2P 原协议是 JRDB-GMP / pseudo 3D / multi-agent / best-of-K；ReGenNet 主协议是 InterHuman / deterministic two-person。
4. 直接改官方仓库会形成第二套数据、训练、评估口径，难以和 P5/P7/P8 同口径比较。

更稳妥路线是在 ReGenNet 内新增 T2P-style adapter/model，复用当前数据与 evaluator。

## 推荐修改

### P9.1 数据 adapter

新增：

```text
data_loaders/forecasting/t2p_interhuman.py
```

职责：

1. 复用 `InterHumanForecastDataset`。
2. 将 active/xyz 样本转为 T2P-style batch。
3. 固定 `N=2`，无需 JRDB 不定人数 padding。
4. 生成 root trajectory 和 local pose：
   - `positions = xyz[:, :, :, 0, :2]`
   - `local_pose = xyz - root_xyz`
   - `input_seq/output_seq` 可使用 `[24,3]` flatten，或第一版裁成 T2P 兼容的 15 joints。
5. 生成全 False 的 `padding_mask`，因为 InterHuman 两人全程有效。
6. 生成 `edge_index = [[0,1],[1,0]]` per sample 或 batch 后按节点 offset 拼接。

### P9.2 模型适配

两个可选路线：

1. **推荐第一版：T2P-lite on InterHuman xyz**
   - 不迁移官方 JRDB preprocessing。
   - 用 root trajectory branch + local pose decoder 重写一个轻量模型。
   - 输出 `[B,120,2,24,3]`，复用 `eval_forecasting_xyz.py`。

2. **第二版：更接近官方 T2P**
   - 迁入 TBIFormer body-part encoder。
   - 迁入 HiVT-style trajectory branch，但改成固定两人、固定 graph。
   - 将 `num_joints` 改为 `24` 并重定义 body-part partition。
   - 将 best-of-K 先关闭，做 deterministic `num_modes=1`，保证和 ReGenNet 主协议一致；之后再单独做多模态扩展。

### P9.3 训练/eval 入口

不要复用 T2P 官方 `lightning_train.py`。

建议扩展现有：

```text
train/train_forecasting_xyz.py
eval/eval_forecasting_xyz.py
```

新增 model type：

```text
t2p_interhuman_xyz
```

保持同一组指标：

```text
joint_mse
mpjpe
long_joint_mse
root_translation_error
relative_root_distance_error
inter_person_distance_consistency_xyz
```

### P9.4 对比基线

第一版必须同口径比较：

```text
independent_pair_xyz
somoformer_xyz
official_somoformer_xyz
t2p_interhuman_xyz
```

不能把 T2P 官方 JRDB-GMP 表格结果拿来和 ReGenNet InterHuman 结果横比。

## 主要设计差异

T2P 官方：

```text
JRDB-GMP, N<=25, input 2s, output 5s, best-of-6, pseudo 3D, 15 joints
```

ReGenNet 适配：

```text
InterHuman, N=2, input 30 frames, output 120 frames, deterministic, SMPL xyz/active, 24 joints
```

因此第一版目标不是“复现论文数值”，而是验证 T2P 的结构思想在当前 InterHuman 协议下是否优于已有 P7/P8 baselines。

## 验收建议

1. dataset smoke：
   - `input_seq=[B,2,30,72]` 或 `[B,2,30,45]`
   - `output_seq=[B,2,121,72]` 或 `[B,2,121,45]`
   - `positions=[B*2,150,2]`
   - `padding_mask=[B*2,150]`
   - finite 全通过。
2. metrics sanity：
   - pred == target 时 xyz metrics 全 0。
3. 2-step train smoke：
   - checkpoint 可保存、可加载、可 eval。
4. seed0 5000-step：
   - 至少比较 `independent_pair_xyz` 和 `somoformer_xyz`。
5. 3-seed：
   - 只有 seed0 过线后再跑。

## 结论

第一版应在 ReGenNet 内做 `t2p_interhuman_xyz`，不要直接改官方 T2P 仓库。这样能复用现有 InterHuman split、normalizer、metrics、训练记录和 P7/P8 baseline，避免把 JRDB-GMP 的数据口径问题带进当前论文主协议。
