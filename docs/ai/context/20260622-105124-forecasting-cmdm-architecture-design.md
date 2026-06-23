# ForecastingCMDM 模型架构设计

## 目标

本设计用于当前 NTU120 label-conditioned forecasting v2 协议：

```text
dataset = NTU120 2P
window_len = 60
obs_len = 20
pred_len = 40
condition = obs20 + action label
target = future40
model family = ReGenNet / CMDM-derived conditional diffusion
```

核心任务：

```text
x_t future40 + timestep + obs20 + action label -> denoised future40
```

## 为什么不能直接使用原始 CMDM.forward

原始 `model/cmdm.py::CMDM.forward()` 的关键语义：

```text
x      = noisy target motion
y["cmotion"] = condition motion
action = action label
```

但在 ReGenNet 原始训练中：

```text
x = reactor full motion
cmotion = actor full motion
任务 = actor full motion + label -> reactor full motion
```

当前任务是：

```text
x = noisy two-person future40
condition = two-person obs20 + action label
任务 = two-person prefix + label -> two-person future
```

因此不能直接把 `obs20` 塞进 `cmotion` 并沿用逐帧 `x + cmx`，因为 `obs20` 和 `future40` 时间长度不同，语义也不同。必须保留 CMDM 组件和条件扩散思想，但重写 condition path。

## 复用 CMDM 的组件

从 `model/cmdm.py` 复用：

```text
InputProcess
OutputProcess
PositionalEncoding
TimestepEmbedder
EmbedAction
mask_cond
classifier-free guidance 的 uncond 约定
TransformerEncoder / TransformerDecoder denoiser 风格
Rotation2xyz_x 评估支持
```

不直接复用：

```text
CMDM.forward 中 y["cmotion"] 与 x 的逐帧 add/concat 逻辑
原始 ccollate 的 actor/reactor 拆分语义
原始 train_mdm.py 的 actor->reactor 训练协议
```

## 建议类名与文件

建议新增：

```text
model/forecasting_cmdm.py
```

核心类：

```text
ForecastingCMDM(nn.Module)
```

辅助函数：

```text
create_forecasting_cmdm_model(...)
count_parameters(...)
ensure_future_shape(...)
```

训练入口：

```text
train/train_label_forecasting_diffusion.py
```

## 输入输出张量

Dataset batch：

```text
obs_motion: [B,56,6,20]
future:     [B,56,6,40]
action:     [B,1]
mask:       [B,1,1,40]
```

Diffusion：

```text
x_start = future
x_t = q_sample(x_start, t)
```

Model forward：

```text
forward(x_t, timesteps, y)
```

其中：

```text
x_t:                  [B,56,6,40]
timesteps:            [B]
y["obs_motion"]:      [B,56,6,20]
y["action"]:          [B,1]
y["mask"]:            [B,1,1,40]
y.get("uncond"):      bool，用于 CFG
```

输出：

```text
pred_future: [B,56,6,40]
```

## 架构总览

```text
obs_motion [B,56,6,20]
  -> obs_input_process
  -> obs_pos_encoder
  -> obs_encoder
  -> obs_pool
  -> obs_cond [1,B,D]

action [B,1]
  -> EmbedAction
  -> mask_cond
  -> action_cond [1,B,D]

timesteps [B]
  -> TimestepEmbedder
  -> time_cond [1,B,D]

x_t future40 [B,56,6,40]
  -> future_input_process
  -> future_pos_encoder
  -> future_tokens [40,B,D]

memory = time_cond + action_cond + obs_cond

future_tokens + memory
  -> denoiser
  -> output_process
  -> pred_future [B,56,6,40]
```

## Obs Encoder

第一版建议用轻量 TransformerEncoder：

```text
obs_tokens = InputProcess(obs_motion)          # [20,B,D]
obs_tokens = PositionalEncoding(obs_tokens)
obs_tokens = obs_encoder(obs_tokens)           # [20,B,D]
obs_cond = obs_tokens.mean(dim=0, keepdim=True) # [1,B,D]
```

取舍：

- mean pooling 简单稳定。
- 如果后续发现 label 控制弱，可把完整 obs token sequence 作为 decoder memory，而不是只池化成一个 token。

## Action Condition

复用 `EmbedAction`：

```text
action_cond = embed_action(y["action"]) # [B,D]
action_cond = mask_cond(action_cond, force_mask=y.get("uncond", False))
action_cond = action_cond.unsqueeze(0)  # [1,B,D]
```

训练时：

```text
cond_mask_prob = 0.1
```

用于 classifier-free guidance。

## Timestep Condition

复用 `TimestepEmbedder`：

```text
time_cond = embed_timestep(timesteps) # [1,B,D]
```

最终条件：

```text
global_cond = time_cond + action_cond + obs_cond
```

注意：

- `obs_cond` 是否在 `uncond=True` 时清零需要明确。第一版建议不清零 obs，只清零 action。
- 原因是采样时我们希望比较“同一 obs 下不同 label”的差异；uncond branch 应表示“无动作标签但仍有观测前缀”。

## Denoiser 方案

### 推荐第一版：Transformer Encoder 加 global token

```text
future_tokens = InputProcess(x_t) # [40,B,D]
future_tokens = PositionalEncoding(future_tokens)
xseq = cat([global_cond, future_tokens], dim=0) # [41,B,D]
output = TransformerEncoder(xseq)[1:] # [40,B,D]
pred = OutputProcess(output)
```

优点：

- 最接近原始 CMDM offline/trans_enc 风格。
- 实现简单。
- 适合 smoke。

缺点：

- `global_cond` 只有一个 token，obs 细节可能被压缩。

### 后续可选：Transformer Decoder

```text
memory = cat([global_cond, obs_tokens], dim=0)
future_tokens = decoder(tgt=future_tokens, memory=memory)
```

优点：

- obs20 全序列可被 future40 cross-attend。
- 更符合 prefix-conditioned forecasting。

缺点：

- 实现复杂一点。
- smoke 阶段不必上来就做。

## Classifier-free Guidance

训练：

```text
cond_mask_prob > 0
随机 mask action_cond
obs_cond 保留
```

采样：

```text
pred_cond = model(x_t, t, obs, action)
pred_uncond = model(x_t, t, obs, uncond=True)
pred = pred_uncond + scale * (pred_cond - pred_uncond)
```

需要一个 forecasting 专用 CFG wrapper，因为原始 `ClassifierFreeSampleModel` 假设 `cond_mode in ['text','action']`，且只转发 `y`；可以复用思路，不一定复用类本身。

## Loss 接口

第一版训练 loss：

```text
diffusion rot_mse: pred_future vs target_future
velocity_mse: temporal diff over 40 frames
root_translation_mse
```

可选 eval 或后续训练 loss：

```text
xyz_mse
mpjpe
relative_root_distance_error
inter_person_distance_consistency
relative body distance
```

建议 smoke 阶段：

- 训练只开 rot_mse + velocity_mse。
- xyz 转换只在 eval 小 batch 做。

原因：

- SMPL-X 转 xyz 成本高。
- 先保证 diffusion forward/backward 和 checkpoint/sampling 正常。

## 与 GaussianDiffusion 的关系

可以复用现有 `diffusion/gaussian_diffusion.py` 的 q_sample / p_sample_loop 思想，但不建议直接复用当前 `training_losses` 原函数而不改，因为它假设：

```text
model.model.module
y["cmotion"]
dataset.dataname
target shape 与原 CMDM 训练一致
```

推荐新增 forecasting 专用 loss wrapper：

```text
training_losses_forecasting_cmdm(model, diffusion, future, cond)
```

或在新训练入口内显式调用：

```text
t = schedule_sampler.sample(...)
x_t = diffusion.q_sample(future, t, noise)
pred = model(x_t, t, y)
loss = mse(pred, target)
```

这样能避免把原始 actor/reactor loss 逻辑误用到 prefix forecasting。

## 配置建议

Smoke：

```text
latent_dim = 128
num_layers = 2
num_heads = 4
ff_size = 512
dropout = 0.1
batch_size = 1
num_steps = 2
cond_mask_prob = 0.1
```

正式单 seed：

```text
latent_dim = 256
num_layers = 4
num_heads = 4
ff_size = 1024
dropout = 0.1
batch_size = 4
grad_accum_steps = 4
num_steps = 5000
lr = 1e-4
weight_decay = 1e-4
```

3080 保守：

```text
latent_dim = 192
num_layers = 3
batch_size = 2
grad_accum_steps = 8
```

## 需要保存的 checkpoint config

checkpoint 必须保存：

```text
model_type = forecasting_cmdm
dataset = ntu
window_len = 60
obs_len = 20
pred_len = 40
num_actions = 26
body_model = smplx
num_person = 2
latent_dim
num_layers
num_heads
ff_size
cond_mask_prob
```

否则后续 sampling/eval 无法可靠恢复模型。

## 主要风险

1. 类别不平衡：`T>=60` 后虽然覆盖 26 类，但某些类别样本极少。
2. Label 控制弱：如果 obs20 已经强烈决定未来，action label 可能被模型忽略。
3. Obs pooling 过强：mean pooling 可能丢掉交互细节。
4. 生成质量与 reconstruction loss 冲突：确定性重建可能压制扩散多样性。
5. SMPL-X xyz 转换慢：不能让 smoke 依赖大批量 xyz loss。

## 验证顺序

1. shape/finite forward。
2. 2-step train loss finite。
3. checkpoint load。
4. sampling 输出 future40。
5. 同一 obs 下 label swap 输出不完全相同。
6. 小 batch xyz metrics 可计算。
7. 可视化 handshaking/hugging/pushing 对比。

## 当前结论

本次“复用 CMDM 架构”应定义为：

```text
复用 CMDM 的条件扩散骨架和关键模块，
但新建 ForecastingCMDM，将 condition 从 y["cmotion"] 改为 obs20 + action label。
```

这是最符合导师“必须使用 ReGenNet 网络”要求、同时不破坏原始 ReGenNet actor->reactor 代码语义的实现边界。
