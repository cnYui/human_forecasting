# P9 T2P InterHuman XYZ 计划

## 已确认迁移边界

用户已确认 `docs/ai/context/20260614-185453-t2p-migration-boundary-confirmation.md` 中的 5 条边界：

1. 第一版只做 InterHuman 双人 xyz deterministic T2P-style baseline。
2. 第一版不使用 JRDB-GMP，不跑官方 T2P preprocessing。
3. 第一版不启用 best-of-6，`num_modes=1`；多模态只作为后续扩展。
4. 代码落在 ReGenNet 内，不直接改 `/home/rpartx3080/CodeSpace/T2P` 官方仓库。
5. 主排行只和 `independent_pair_xyz / somoformer_xyz / official_somoformer_xyz` 做同数据、同 split、同指标比较。

## 目标

新增 `t2p_interhuman_xyz`，把 T2P 的核心结构思想迁入当前 ReGenNet InterHuman 双人 forecasting 协议：

```text
obs_xyz [B,30,2,24,3]
-> root trajectory branch
-> local pose decoder conditioned on predicted trajectory
-> pred_xyz [B,120,2,24,3]
```

第一版目标是作为架构排行榜 baseline，而不是复现 T2P JRDB-GMP 论文分数。

## 固定协议

```text
dataset: InterHuman
data_root: dataset/interhuman/smpl/conditioned
window_len: 150
obs_len: 30
pred_len: 120
persons: 2
num_joints: 24
coord_dim: 3
evaluation: eval_forecasting_xyz.py 现有 xyz metrics
training budget: 默认沿用 P7/P8 5000 steps；seed0 先跑，过线后 seeds=0,1,2
```

## 实现范围

### P9.1 数据 adapter

新增：

```text
data_loaders/forecasting/t2p_interhuman.py
```

职责：

- 复用 `InterHumanForecastDataset`。
- 使用 `active_to_xyz` 生成 `obs_xyz/target_xyz`。
- 生成 T2P-style 中间字段：
  - `root_xy`
  - `local_pose`
  - `padding_mask`
  - `edge_index`
- 第一版固定两人有效，不引入 JRDB 不定人数 padding。

### P9.2 模型

新增：

```text
model/forecasting_t2p.py
```

第一版模型名：

```text
t2p_interhuman_xyz
```

保留 T2P 三个关键模块：

1. root trajectory predictor
2. two-person interaction/fusion
3. trajectory-conditioned local pose decoder

第一版不做：

- best-of-K
- BEV/ROMP
- SMPL theta decoder
- JRDB 15-joint body-part 口径

### P9.3 训练接入

扩展：

```text
train/train_forecasting_xyz.py
```

新增 `--model_type t2p_interhuman_xyz`。

训练 loss 第一版：

```text
loss = xyz_mse + root_loss_weight * root_mse + local_loss_weight * local_mse
```

默认建议：

```text
root_loss_weight=1.0
local_loss_weight=1.0
```

如第一版复杂度过高，可先用 `pred_xyz` MSE 闭环，再加 root/local 辅助 loss。

### P9.4 评估接入

扩展：

```text
eval/eval_forecasting_xyz.py
```

要求：

- checkpoint 可加载。
- 输出仍为 `[B,120,2,24,3]`。
- 不新增 metric key。
- 可参与现有 aggregate/summary 流程。

### P9.5 smoke

验收：

```text
dataset smoke: shape/finite
metrics sanity: pred==target 全 0
2-step train: checkpoint 可保存
checkpoint eval: 可加载且指标 finite
```

### P9.6 seed0 和 3-seed

先跑 seed0 5000-step：

```text
save/forecasting/interhuman/p9_t2p_interhuman_xyz_..._s0_5000
```

若 seed0 不明显失败，再跑 seeds=`0,1,2` 并汇总到：

```text
results/forecasting/interhuman/p9_t2p_interhuman_xyz_main/
```

## 主对比

P9 主表至少包括：

```text
independent_pair_xyz
somoformer_xyz
official_somoformer_xyz
t2p_interhuman_xyz
```

指标：

```text
joint_mse
mpjpe
long_joint_mse
root_translation_error
relative_root_distance_error
inter_person_distance_consistency_xyz
```

必须记录：

```text
num_params
batch_size
num_steps
lr
seed
train time
GPU memory if monitored
```

## 风险

1. T2P 的优势可能主要来自 JRDB 多人轨迹和 best-of-K；关闭 best-of-K 后收益可能变小。
2. InterHuman 的 root translation 已以 actor frame 0 为 origin，trajectory branch 必须尊重当前坐标系，不要重新引入世界坐标假设。
3. `24 joints` 不等同 T2P 原始 `15 joints`，body-part partition 不能硬套原论文索引。
4. 如果 `t2p_interhuman_xyz` 输给 `somoformer_xyz`，仍是有效结果，说明当前数据/协议下 joint-token completion 更强。

## 下一步

进入 P9.1/P9.2 实现前，先读取：

```text
data_loaders/forecasting/interhuman.py
utils/forecasting_xyz.py
model/forecasting_somoformer.py
train/train_forecasting_xyz.py
eval/eval_forecasting_xyz.py
```

然后按最小闭环实现 dataset smoke 和 2-step train。
