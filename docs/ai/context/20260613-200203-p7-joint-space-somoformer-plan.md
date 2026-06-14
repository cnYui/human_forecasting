# P7.1 joint-space SoMoFormer baseline 计划

## 目标

先做第一种方案：把 ReGenNet InterHuman active-vector 数据转换为 SMPL 3D joints，再用 SoMoFormer-style joint/person token Transformer 做两人未来预测。

本阶段只验证 SoMoFormer 原始建模思想在 InterHuman 双人预测上的可行性，不进入当前 P5 active-vector 主表。

## 不做什么

- 不修改 `utils/forecasting_metrics.py::compute_forecasting_metrics`。
- 不改变 P1-P6 的 active-vector 数据协议。
- 不把 rot6d 硬 reshape 成 xyz。
- 不直接改 `/home/rpartx3080/CodeSpace/somoformer` 原仓库训练入口。
- 不声称复现 SoMoFormer 原论文结果。

## 当前输入输出

ReGenNet dataset 当前输出：

```text
obs_active:    [B,30,2,147]
target_active: [B,120,2,147]
```

P7.1 需要转换为：

```text
obs_xyz:    [B,30,2,24,3]
target_xyz: [B,120,2,24,3]
```

模型输出：

```text
pred_xyz: [B,120,2,24,3]
```

## 必要新增文件

### 1. `utils/forecasting_xyz.py`

职责：

- 提供 active-vector 到 SMPL xyz joints 的转换。
- 提供 joint-space metrics helper。
- 尽量复用 `model/rotation2xyz.py` / `model/smpl.py`，不重复造 SMPL forward。

建议函数：

```python
active_to_smpl_motion(active)
active_to_xyz(active, device, jointstype="smpl")
batch_active_to_xyz(value, device, jointstype="smpl")
compute_xyz_metrics(pred_xyz, target_xyz, obs_xyz)
```

shape contract：

```text
input active: [B,T,2,147] 或 [T,2,147]
output xyz:   [B,T,2,24,3]
```

注意：

- `active[..., :144]` 是 24 个 rot6d。
- `active[..., 144:147]` 是 root translation。
- 转给 `Rotation2xyz` 时需要恢复为旧代码常用 shape：`[B, joints, feats, frames]`。
- `num_person=2`，`pose_rep="rot6d"`，`translation=True`，`glob=True`，`jointstype="smpl"`，`vertstrans=True`。

验收：

```text
finite true
shape 等于 [B,T,2,24,3]
pred == target 时 xyz metrics 全 0
```

### 2. `model/forecasting_somoformer.py`

职责：

- 实现不依赖 SoMoFormer 原 dataset 的 SoMoFormer-style joint model。
- 可以参考 `/home/rpartx3080/CodeSpace/somoformer/src/model.py`，但要改成 ReGenNet 风格、中文错误信息、稳定 shape check。

建议类：

```python
class JointSpaceSoMoFormer(nn.Module)
```

输入输出：

```text
input obs_xyz: [B,30,2,24,3]
output pred_xyz: [B,120,2,24,3]
```

核心结构：

```text
obs_xyz
-> reshape [B,30,48,3]
-> last-pose padding to [B,150,48,3]
-> flatten coords [150,B,144]
-> DCT over time
-> tokens [144,B,dct_n]
-> Linear projection
-> person/joint/coord embeddings
-> TransformerEncoder
-> Linear to dct_n
-> residual on DCT coeffs
-> IDCT
-> output [B,150,48,3]
-> take future [B,120,2,24,3]
```

配置建议：

```text
obs_len=30
pred_len=120
num_persons=2
num_joints=24
coord_dim=3
dct_n=30 或 50
hidden_dim=256
num_heads=8
num_layers=4 或 6
dim_feedforward=1024
dropout=0.1
residual_connection=True
```

第一版可以暂时不实现 SoMoFormer 原始 grid location embedding；先用：

```text
person embedding
joint embedding
coord embedding
```

原因：

- ReGenNet 只有固定两人，且 root translation 已在 xyz 中。
- location embedding 后续可作为 ablation。

### 3. `train/train_forecasting_xyz.py`

职责：

- 复用 `InterHumanForecastDataset` 和 `forecasting_collate`。
- batch 内将 active obs/target 转为 xyz。
- 训练 `JointSpaceSoMoFormer`。
- 保存 checkpoint、args、train_log。

训练 loss：

```text
MSE(pred_xyz, target_xyz)
```

建议参数：

```text
--dataset interhuman
--data_path dataset/interhuman/smpl/conditioned
--save_dir ...
--window_len 150
--obs_len 30
--pred_len 120
--batch_size 16 或 32
--num_steps 5000
--dct_n 30
--hidden_dim 256
--num_layers 4
--num_heads 8
--lr 1e-4 或 3e-4
--weight_decay 1e-4
--seed 0
```

注意：

- active -> xyz 每 batch 做 SMPL forward，可能比 active-vector baseline 慢。
- 如果太慢，P7.1.1 再做 cached xyz H5，不作为第一步阻塞项。

### 4. `eval/eval_forecasting_xyz.py`

职责：

- 支持 dataset smoke、metrics sanity、checkpoint eval。
- 不复用 active-vector `compute_forecasting_metrics`。

建议 mode：

```text
--mode xyz_smoke
--mode metrics_sanity
--mode checkpoint
```

指标建议：

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

验收：

```text
metrics_sanity: pred == target 时所有误差为 0
checkpoint eval: 生成 metrics_test.json / metrics_test.yaml
```

### 5. 可选：`data_loaders/forecasting/xyz_tensors.py`

第一版不需要。直接在 train/eval 内转换即可。

如果后续需要 cache 或多 worker，可再引入。

## 需要小改的现有文件

### `AGENTS.md`

完成计划和结果后追加上下文记忆。

### 可选：`requirements` / 环境说明

当前默认 shell `python3` 缺 `torch`、`smplx`、`torch_dct`。实现前必须确认 ReGenNet 实际训练环境。

如果使用 SoMoFormer 原始 `torch-dct`，需要安装：

```text
torch-dct==0.1.5
```

也可以避免新依赖，复用 SoMoFormer 仓库中的 DCT matrix 思路，在本项目内用纯 torch matrix multiplication 实现。

## 不建议修改的现有文件

- `utils/forecasting_metrics.py`
- `eval/eval_forecasting.py`
- `train/train_forecasting.py`
- `model/forecasting.py`

原因：

P7.1 是 joint-space diagnostic baseline，不应污染当前 active-vector 主协议。

## 最小验收顺序

### Step 1：环境检查

```text
确认可用 python 环境中有 torch、smplx。
确认 body_models/smpl 可被 model/smpl.py 读取。
```

### Step 2：xyz adapter smoke

```text
加载 2-4 个 InterHuman samples。
obs_active -> obs_xyz。
target_active -> target_xyz。
检查 shape 和 finite。
```

### Step 3：metrics sanity

```text
compute_xyz_metrics(target_xyz, target_xyz, obs_xyz)
所有误差为 0。
```

### Step 4：model forward smoke

```text
obs_xyz [2,30,2,24,3] -> pred_xyz [2,120,2,24,3]
loss finite。
```

### Step 5：2-step training smoke

```text
num_steps=2
checkpoint 可保存 / 加载 / eval。
```

### Step 6：正式 seed0 baseline

```text
num_steps=5000
输出 metrics_test.json/yaml。
```

## 风险

### 1. SMPL forward 慢

active -> xyz 每 batch 要跑 SMPL。若训练太慢，后续缓存：

```text
dataset/interhuman/smpl/forecasting_xyz/interhuman_{train,val,test}.h5
shape per sample: [T,2,24,3]
```

### 2. joint-space 指标不能和 active-vector 主表直接比较

P7.1 只回答 SoMoFormer-style joint prediction 是否有效。若要进论文主表，必须继续 P7.2 active-vector 输出版本。

### 3. 关节集选择影响结果

第一版锁定 `jointstype="smpl"`，即 24 joints。不裁剪成 SoMoF 13 joints。

### 4. DCT 参数敏感

`seq_len=150` 比 SoMoFormer 原始 30 长。第一版建议 `dct_n=30` 起步，必要时对比 `dct_n=50`。

## 推荐结论

P7.1 需要新增 4 个核心文件：

```text
utils/forecasting_xyz.py
model/forecasting_somoformer.py
train/train_forecasting_xyz.py
eval/eval_forecasting_xyz.py
```

只需要轻微更新：

```text
AGENTS.md
docs/ai/context/*
```

第一阶段不要修改现有 P1-P6 active-vector 主路径。先把 joint-space baseline 跑通，看 SoMoFormer-style token Transformer 在 InterHuman 上是否有明显信号。
