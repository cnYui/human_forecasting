# NTU120 Label-conditioned ReGenNet Forecasting 最终目标设计

## 最终目标

训练一个 ReGenNet-style 条件扩散模型，实现：

```text
输入：前 30 帧双人动作 + 动作标签
输出：后 120 帧双人动作
数据：NTU120 2P
标签：26 个双人动作类，包含 handshaking
生成：同一 obs_30 更换标签时，future_120 应随标签发生可观察变化
```

示例：

```text
obs_30 + handshaking -> 未来动作应接近握手
obs_30 + hugging     -> 未来动作应接近拥抱
obs_30 + pushing     -> 未来动作应接近推人
```

## 当前硬约束

用户原始边界中有一条需要修正：

```text
只保留 T>=150 的 NTU120 样本
```

当前本地 NTU120 H5 严格过滤后只剩：

```text
train T>=150 = 5
test  T>=150 = 5
handshaking T>=150 = 0
```

所以该约束与“训练所有动作类”和“握手条件生成”冲突。

## 推荐正式协议

为保留 NTU120、全动作类和 handshaking 标签，推荐正式协议改为：

```text
对每条 NTU120 原始序列做时间重采样到 150 帧
obs_len = 30
pred_len = 120
window_len = 150
future_120 是 normalized sequence future
```

禁止使用简单 padding 作为主方案。原因是 padding 会把未来伪造成静止尾帧，破坏 forecasting 监督。

## 数据接口

输入 H5：

```text
dataset/ntu120/smplx/conditioned/xsub.train.h5
dataset/ntu120/smplx/conditioned/xsub.test.h5
```

原始 shape：

```text
[T,56,6]
```

其中当前 `Feeder` 会把最后一个 slot 视为 translation，把前面 slot 视为 pose。

新 dataset 输出建议：

```text
obs:         [B,56,6,30]
future:      [B,56,6,120]
label:       [B,1]
label_text:  list[str]
sample_id:   list[str]
```

训练时 diffusion target：

```text
x_start = future
x_t = q_sample(future, t)
```

条件：

```text
cond["y"]["obs_motion"] = obs
cond["y"]["action"] = label
cond["y"]["mask"] = future_mask
```

## 模型设计

模型名建议：

```text
ForecastingCMDM
```

来源：

```text
model/cmdm.py::CMDM
```

必须复用的 ReGenNet 机制：

- `TimestepEmbedder`
- `EmbedAction`
- classifier-free guidance 的 `cond_mask_prob` / `uncond` 路径
- Transformer Decoder/Encoder 或 MLP denoiser 主干
- rot6d + translation 表示
- `Rotation2xyz_x` 转 xyz 评估

需要新增/改造的部分：

```text
obs_encoder: encode obs_30 -> latent condition
future denoiser: denoise x_t future_120
condition fusion: timestep + action + obs
```

推荐第一版结构：

```text
obs_motion [B,56,6,30]
  -> InputProcess
  -> temporal pooling / Transformer encoder
  -> obs_cond [1,B,D]

action label [B,1]
  -> EmbedAction
  -> action_cond [1,B,D]

timestep
  -> TimestepEmbedder
  -> time_cond [1,B,D]

condition memory = time_cond + action_cond + obs_cond

x_t future [B,56,6,120]
  -> InputProcess
  -> sequence_pos_encoder
  -> Transformer Decoder / Encoder
  -> OutputProcess
  -> predicted future [B,56,6,120]
```

第一版不建议直接把 obs_30 拼到 future 时间轴再 inpaint，因为用户目标是“前缀条件预测未来”，显式 obs encoder 更清楚，也便于做 label swap。

## 训练 loss

基础 loss 复用 ReGenNet diffusion：

```text
rot_mse
rcxyz_mse
vel_mse
fc
```

双人交互项可复用/改造：

```text
relative root translation
relative body distance
relative orientation
```

第一版正式配置建议：

```text
lambda_vel = 0.5
lambda_rcxyz = 1.0
lambda_fc = 0.0 或 0.1
lambda_orient = 0.5
lambda_body = 0.5
lambda_transl = 1.0
cond_mask_prob = 0.1
```

实际权重以 smoke loss scale 为准。

## 训练阶段

### P0 数据 gate

输出：

```text
每类样本数
每类 handshaking 等关键类别样本数
原始帧长分布
重采样后 shape
```

### P1 Dataset smoke

检查：

```text
obs/future/action shape
label_text 是否正确
handshaking label 是否为 8
finite check
```

### P2 Model forward smoke

检查：

```text
ForecastingCMDM forward
输入 x_t future + obs + action + t
输出 shape = future shape
loss finite
```

### P3 2-step train smoke

命令形态：

```bash
python -m train.train_label_forecasting_diffusion \
  --dataset ntu \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --save_dir save/forecasting/ntu120_label/p3_smoke \
  --model_type forecasting_cmdm \
  --body_model smplx \
  --num_person 2 \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --resample_to_window \
  --batch_size 1 \
  --num_steps 2 \
  --save_interval 2 \
  --eval_interval 2 \
  --latent_dim 128 \
  --layers 2 \
  --cond_mask_prob 0.1 \
  --overwrite
```

### P4 Sampling smoke

固定一个 `obs_30`，分别输入：

```text
handshaking
hugging other person
pushing other person
```

输出：

```text
results/forecasting/ntu120_label/p4_label_swap/
```

需要保存：

```text
generated .npy
metrics json
可视化视频或骨架图
```

### P5 正式单 seed 训练

命令形态：

```bash
python -m train.train_label_forecasting_diffusion \
  --dataset ntu \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --save_dir save/forecasting/ntu120_label/forecasting_cmdm_h256_l4_s0_5000 \
  --model_type forecasting_cmdm \
  --body_model smplx \
  --num_person 2 \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --resample_to_window \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --num_steps 5000 \
  --save_interval 1000 \
  --eval_interval 1000 \
  --latent_dim 256 \
  --layers 4 \
  --cond_mask_prob 0.1 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --seed 0
```

3080 显存不足时：

```text
batch_size = 2
grad_accum_steps = 8
latent_dim = 192
layers = 3
```

### P6 3-seed 训练

仅在 P5 指标和生成结果可用后运行：

```text
seed = 0,1,2
```

## 评估设计

基础预测指标：

```text
joint_mse
mpjpe
short/mid/long joint_mse
root_translation_error
relative_root_distance_error
inter_person_distance_consistency
```

标签一致性指标：

第一版可用已有动作标签训练一个轻量 action classifier 或复用已有 ST-GCN 评估结构，计算：

```text
generated future/action label classification accuracy
handshaking subset accuracy
label swap distinguishability
```

标签 swap 评估：

```text
同一 obs，输入不同 label
计算不同 label 生成结果之间的 motion distance
用 classifier 判断是否更接近指定 label
```

注意：如果只用 reconstruction metrics，无法证明“生成的是握手”；必须补 label consistency。

## 与现有 baseline 的关系

现有 `train/train_forecasting_xyz.py` 的 SoMoFormer/T2P/JRT 是 deterministic forecasting baseline，不满足“必须使用 ReGenNet 条件扩散”的导师要求。

本设计要保留这些 baseline 作为对比，但主模型应是：

```text
ReGenNet/CMDM-derived label-conditioned forecasting diffusion
```

## 论文表述边界

可以说：

```text
We adapt the ReGenNet conditional motion diffusion framework to label-conditioned two-person motion forecasting.
```

不应说：

```text
首次提出 multi-person forecasting
首次提出 interaction-aware forecasting
```

因为这些方向已有大量相关工作。

## 当前决策状态

已确认：

- 数据优先 NTU120 2P。
- 训练所有动作类。
- 使用动作标签做条件。
- 输出未来 120 帧。
- 使用 SMPL/SMPL-X rot6d + translation，评估时转 xyz。
- 必须复用 CMDM/ReGenNet 条件扩散结构。
- 第一阶段先 smoke，但必须保留最终正式训练设计。

待用户确认：

- 是否接受把“只保留 T>=150”改为“时间重采样到 150”。如果不接受，当前 NTU120 目标不可训练。
