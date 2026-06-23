# JRT 使用 ReGenNet 本地数据集的适配设计

## 背景

用户希望用 `/home/rpartx3080/CodeSpace/ReGenNet/dataset` 下的数据跑 JRT。当前本地数据主要包括：

```text
dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5
dataset/chi3d/smplx/conditioned/chi3d_smplx_{train,test}.h5
dataset/interhuman/motions/*.pkl
dataset/interhuman/annotations_interhuman/interhuman_label.json
dataset/chi3d/annotations_chi3d/chi3d_label.json
```

现有 ReGenNet forecasting 主路径已经完成：

```text
InterHuman H5 [T,25,12]
-> extract_active_motion
-> active [T,2,147]
-> active_to_xyz
-> xyz [B,T,2,24,3]
```

JRT 官方代码期望的是：

```text
3DPW poseData.pkl: split -> list of [2,30,39]
reshape -> [N=2,T=30,J=13,3]
obs_len=16, pred_len=14
relation = exp(-joint_distance) + adjacency + connectivity
```

## 核心判断

不建议直接修改 `/home/rpartx3080/CodeSpace/JRTransformer` 官方仓库去读取 ReGenNet 的 `dataset/`。

原因：

1. JRT 官方仓库是复现资产，应保持干净，方便和论文结果对照。
2. ReGenNet 已有 InterHuman forecasting split、active-vector 数据协议、xyz evaluator 和 P7/P8 训练链路；重复在 JRT 仓库写一套 H5/SMPL loader 会产生双份口径。
3. JRT 的 relation 语义依赖 3D joint positions；ReGenNet 的原始 H5 是 SMPL rot6d + translation，不是直接的 joint xyz。
4. 当前论文主协议是 InterHuman 150/30/120，不是 JRT 官方 3DPW 16/14；硬改官方 JRT 会混淆协议。

推荐做法是在 ReGenNet 内新增 `jrt_xyz` baseline，复用 P7/P8 joint-space pipeline。

## 第一版范围

第一版只支持 InterHuman：

```text
dataset = interhuman
data_path = dataset/interhuman/smpl/conditioned
obs_len = 30
pred_len = 120
num_persons = 2
num_joints = 24
coord_dim = 3
```

暂不支持：

- Chi3D：本地是 SMPL-X H5，关节数、旋转/translation layout 和 InterHuman SMPL H5 不同，第一版不应混入。
- NTU120-AS：当前不是 forecasting H5 主路径，且 action-conditioned 语义不同。
- JRT 官方 CMU/MuPoTS：官方仓库未释放对应代码和预处理。

## 需要修改的模块

### 1. 新增 JRT xyz 模型

建议新增：

```text
model/forecasting_jrt.py
```

职责：

- 实现 `JointRelationTransformerXYZ`。
- 输入 `obs_xyz [B,30,2,24,3]`。
- 输出 `pred_xyz [B,120,2,24,3]`。
- 内部 token 为 `N*J = 48` 个 person-joint token。
- joint feature 可以使用：

```text
position sequence: [obs_len, 3]
velocity sequence: [obs_len, 3]
flatten 后输入 joint encoder
```

- relation feature 使用：

```text
exp(-distance history): [obs_len]
adjacency: [1]
connectivity / same-person: [1]
```

第一版 relation feature 维度：

```text
in_relation_size = obs_len + 2 = 32
```

### 2. 新增 relation 构造 helper

建议新增或放入同一模型文件：

```text
build_jrt_relation_features(obs_xyz, skeleton_edges)
```

输入：

```text
obs_xyz [B,T,2,24,3]
```

输出：

```text
relation [B,48,48,T+2]
```

细节：

- distance: 对每对 token 计算历史欧氏距离，再做 `exp(-distance)`，保持 JRT 官方口径。
- adjacency: 只在同一 person 内按 SMPL 24-joint skeleton edges 标记骨骼连接。
- connectivity: 同一 person 标记为 1，跨 person 标记为 0；如果要完全贴近 JRT，可改成“骨架路径连通”，但 SMPL 单人体内任意两个关节都连通，第一版 same-person 更直接。
- self relation 需要保留，不能丢掉对角线。

### 3. 接入模型 factory

修改：

```text
model/forecasting_xyz.py
```

新增：

```text
XYZ_FORECASTING_MODEL_TYPES += ("jrt_xyz",)
create_xyz_forecasting_model(... model_type="jrt_xyz")
create_xyz_forecasting_model_from_config(...)
```

这样可以复用：

```text
train/train_forecasting_xyz.py
eval/eval_forecasting_xyz.py
```

### 4. 训练 loss

第一版最低实现：

```text
loss = mse(pred_xyz, target_xyz)
```

更贴近 JRT 的实现：

```text
loss = pose_mse + relation_weight * future_distance_mse
```

推荐第一版直接实现 relation supervision，因为这是 JRT 的核心，不然只能叫 `relation-attention transformer`，不能算 JRT-style baseline。

需要模型 forward 返回：

```text
pred_xyz
pred_relation_distance
target_relation_distance 由 target_xyz 构造
```

为了不破坏现有训练器，可以给模型实现：

```text
training_loss(obs_xyz, target_xyz, relation_weight=1.0, aux_weight=...)
```

`train_forecasting_xyz.py` 已经支持有 `training_loss` 方法的模型。

### 5. 指标与评估

不新增指标口径，复用 P7/P8：

```text
joint_mse
mpjpe
short_joint_mse
mid_joint_mse
long_joint_mse
root_translation_error
relative_root_distance_error
inter_person_distance_consistency_xyz
```

原因：

- JRT 官方 VIM 是 3DPW/SoMoF 口径，不适合直接套到 InterHuman 150/30/120。
- 当前 ReGenNet 需要和 `independent_pair_xyz`、`somoformer_xyz`、`official_somoformer_xyz` 同口径比较。

### 6. CLI 建议

训练：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python train/train_forecasting_xyz.py \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --model_type jrt_xyz \
  --save_dir save/forecasting/interhuman/p9_jrt_xyz_h256_l4_s0_5000 \
  --obs_len 30 \
  --pred_len 120 \
  --window_len 150 \
  --hidden_dim 256 \
  --num_heads 8 \
  --num_layers 4 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --num_steps 5000 \
  --seed 0
```

评估：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python eval/eval_forecasting_xyz.py \
  --mode checkpoint \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --checkpoint save/forecasting/interhuman/p9_jrt_xyz_h256_l4_s0_5000/model000005000.pt \
  --save_dir results/forecasting/interhuman/p9_jrt_xyz_seed0
```

## 主要风险

1. 显存：JRT relation tensor 是 `[B,48,48,*]`，比 SoMoFormer token attention 更重；batch size 应从 `16` 开始，必要时降到 `8`。
2. 预测跨度：JRT 官方是 16->14，当前是 30->120，decoder 直接输出 120 帧可能比官方难很多。
3. skeleton edges：SMPL 24 joints 的 edge 定义必须固定并记录；不能沿用 JRT 3DPW 13-joint edges。
4. relation supervision 尺度：distance MSE 的尺度和 xyz MSE 不同，`relation_weight` 需要 smoke 后调节。
5. 与 official SoMoFormer 的比较要同 seed、同 train steps、同 batch/显存预算记录。

## 推荐阶段

P9.1：

- 新增 `jrt_xyz` 模型和 relation helper。
- 跑 `max_samples` smoke、shape/finite smoke、2-step training、checkpoint eval。

P9.2：

- seed0 5000-step，与 `independent_pair_xyz`、`somoformer_xyz`、`official_somoformer_xyz` 比较。

P9.3：

- 如果 seed0 有价值，再跑 seeds `0,1,2`。

P9.4：

- 消融：无 adjacency、无 connectivity、无 relation supervision。

## 论文表述边界

如果做成，它应写成：

```text
We adapt a JRT-style joint-relation baseline to our InterHuman SMPL joint-space forecasting protocol.
```

不能写成：

```text
We reproduce JRT on InterHuman exactly.
```

原因是数据集、关节定义、预测 horizon、指标和训练协议都已不同。
