# T2P 迁移边界确认

## 背景

用户目标是固定 ReGenNet 本地数据集与评估协议，迁移不同论文项目的架构做同口径比较，观察最终分数排行。

本次确认对象是：

```text
Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning
T2P / Trajectory2Pose
```

相关前置文档：

```text
docs/ai/context/20260614-184859-t2p-source-data-reproduction-result.md
docs/ai/context/20260614-185128-t2p-on-regennet-dataset-adaptation-design.md
```

## 必须固定的实验协议

第一版 T2P 迁移必须固定在当前 ReGenNet 双人预测主协议下：

```text
dataset: InterHuman
window_len: 150
obs_len: 30
pred_len: 120
persons: 2
input source: dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5
primary comparison space: xyz [B,T,2,24,3]
evaluation: ReGenNet P7/P8 xyz metrics
seeds: 先 seed0 smoke/正式跑，过线后再 seeds 0/1/2
```

第一版不改变现有 P1-P8 的数据协议、split、metric key 或历史结果。

## 要迁移的内容

迁移 T2P 的架构思想，而不是迁移 JRDB 数据协议。

第一版保留：

1. `trajectory-first`：先预测每个人未来 root/global trajectory。
2. `global-local decoupling`：把 root/global motion 和 local pose 分开建模。
3. `trajectory conditioning`：用预测的未来 trajectory 条件化 local pose decoder。
4. `agent interaction`：在 root trajectory 或 fused trajectory-pose embedding 上建模两人交互。
5. `final composition`：`future_xyz = predicted_root + predicted_local_pose`。

## 不迁移的内容

第一版明确不迁移：

1. JRDB-GMP 数据集。
2. JRDB 原始 preprocessing。
3. BEV/ROMP 从图像估计 3D pose 的流程。
4. 官方仓库的作者机器绝对路径。
5. 官方缺失或不匹配的 Hydra config。
6. JRDB 的不定人数 `N<=25` padding 复杂度。
7. T2P 论文主表的 best-of-6 多模态评估。
8. 官方论文分数与 ReGenNet 分数的直接横比。

## 第一版必须关闭 best-of-K

T2P 原论文使用多模态 future trajectory proposals，常见为 `F=6`，评估时选最接近 GT 的 mode。

当前 ReGenNet P5/P7/P8 主协议是 deterministic。如果第一版直接用 best-of-6，会给 T2P baseline 额外优势，导致排行不公平。

因此第一版固定：

```text
num_modes = 1
sampling = deterministic
```

多模态 `num_modes=6` 只能作为后续单独扩展表，不能混入 deterministic 主排行。

## 第一版数据表示

从 InterHuman active/xyz 生成：

```text
obs_xyz: [B,30,2,24,3]
target_xyz: [B,120,2,24,3]
root_xy: xyz[..., root_joint=0, :2]
local_pose: xyz - root_xyz
padding_mask: 全 False
edge_index: 双人互连 0->1, 1->0
```

可选 joint 数：

1. 推荐第一版用 `24 joints`，保持 SMPL 全关节。
2. 若完全照 T2P body-part partition，可另做 `15 joints` ablation，但不作为第一版主结果。

## 第一版代码边界

推荐在 ReGenNet 内新增或扩展：

```text
data_loaders/forecasting/t2p_interhuman.py
model/forecasting_t2p.py
train/train_forecasting_xyz.py
eval/eval_forecasting_xyz.py
```

不直接修改：

```text
/home/rpartx3080/CodeSpace/T2P
```

官方 T2P 仓库只作为参考实现和结构来源。

## 第一版比较对象

主排行只比较同一数据、同一指标、同一 deterministic 协议下的模型：

```text
independent_pair_xyz
somoformer_xyz
official_somoformer_xyz
t2p_interhuman_xyz
```

若要更严谨，需要记录：

```text
num_params
batch_size
num_steps
lr
seed
GPU memory
train time
```

## 通过门槛

阶段推进建议：

1. dataset smoke 通过。
2. pred == target metrics sanity 全 0。
3. 2-step train checkpoint 可保存/加载/eval。
4. seed0 5000-step 与 P7/P8 baseline 同口径比较。
5. seed0 不明显失败后再跑 3-seed。

## 需要用户确认

请确认以下边界是否锁定：

1. 第一版只做 InterHuman 双人 xyz deterministic T2P-style baseline。
2. 第一版不使用 JRDB-GMP、不跑官方 T2P preprocessing。
3. 第一版不启用 best-of-6，多模态只作为后续扩展。
4. 代码落在 ReGenNet 内，不直接改官方 T2P 仓库。
5. 主排行只和 `independent_pair_xyz / somoformer_xyz / official_somoformer_xyz` 同口径比较。
