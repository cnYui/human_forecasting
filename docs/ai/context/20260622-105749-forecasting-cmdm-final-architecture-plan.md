# ForecastingCMDM-Decoder 最终模型架构计划

## 结论先行

最终模型不采用上一版文档中的“轻量 Transformer Encoder + global token”作为正式架构。该结构只适合 smoke 或 ablation。

当前最终目标模型确定为：

```text
ForecastingCMDM-Decoder
```

一句话定义：

```text
一个 CMDM/ReGenNet-derived 条件扩散模型，用 Transformer Decoder 对 noisy future40 去噪，并通过 cross-attention 显式读取 obs20 memory 与 action/timestep condition。
```

任务协议：

```text
dataset = NTU120 2P
window_len = 60
obs_len = 20
pred_len = 40
condition = obs20 + action label
target = future40
body_model = smplx
representation = rot6d + root translation
```

## 为什么最终必须用 Decoder

原始 CMDM 的核心不是简单 Transformer，而是：

```text
noisy target motion x_t
+ timestep embedding
+ condition motion
+ action label
-> denoised target motion
```

对当前任务，condition motion 不再是 actor full motion，而是：

```text
two-person observed prefix obs20
```

obs20 和 future40 长度不同，不能继续用原始 CMDM 的逐帧 `x + cmotion`。因此最终架构要把 CMDM 的 condition-motion 思想升级成：

```text
obs20 -> memory
future40 noisy tokens -> decoder target
decoder cross-attends obs memory
```

这才是完整利用 CMDM 的条件扩散思想，而不是只借几个模块。

## 与原始 CMDM 的对应关系

| 原始 CMDM | 本模型 |
|---|---|
| `x`: noisy reactor full motion | `x_t`: noisy two-person future40 |
| `y["cmotion"]`: actor full motion | `y["obs_motion"]`: two-person obs20 |
| `EmbedAction` | 复用，作为动作标签条件 |
| `TimestepEmbedder` | 复用，作为 diffusion timestep 条件 |
| `InputProcess` | 复用，分别编码 obs20 和 future40 |
| `OutputProcess` | 复用，输出 denoised future40 |
| `TransformerDecoder` online 架构 | 复用思想，改成 future tokens cross-attend obs memory |
| `ClassifierFreeSampleModel` 思想 | 复用，写 forecasting 专用 wrapper |
| actor->reactor 数据协议 | 不复用 |

## 最终模块组成

建议文件：

```text
model/forecasting_cmdm.py
```

核心类：

```text
ForecastingCMDMDecoder(nn.Module)
```

模块清单：

```text
obs_input_process      # CMDM InputProcess, 编码 obs20
future_input_process   # CMDM InputProcess, 编码 noisy future40
obs_pos_encoder        # 使用 CMDM PositionalEncoding/同类 sinusoidal PE，位置 0..19
future_pos_encoder     # 使用同一 PE 表，位置 20..59
obs_encoder            # TransformerEncoder, 构建 condition memory
embed_timestep         # CMDM TimestepEmbedder
embed_action           # CMDM EmbedAction
condition_fuse         # MLP/Linear，把 global condition 投影到 memory token
seqTransDecoder        # CMDM-style TransformerDecoder
output_process         # CMDM OutputProcess, 输出 future40
rot2xyz                # Rotation2xyz_x, 评估和可选几何 loss
```

## 张量协议

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
noise ~ N(0,I)
x_t = q_sample(x_start, t, noise)
```

Model forward：

```text
pred = model(x_t, timesteps, y)
```

其中：

```text
x_t:             [B,56,6,40]
timesteps:       [B]
y["obs_motion"]: [B,56,6,20]
y["action"]:     [B,1]
y["mask"]:       [B,1,1,40]
y["uncond"]:     optional bool
```

输出：

```text
pred_xstart: [B,56,6,40]
```

## 位置编码

最终模型必须区分 obs 和 future 在同一个 60 帧窗口中的绝对位置：

```text
obs positions    = 0..19
future positions = 20..59
```

实现建议：

```text
pe = PositionalEncoding(latent_dim)
obs_tokens += pe[0:20]
future_tokens += pe[20:60]
```

如果直接给 obs 和 future 都从 0 开始编码，会丢失“future 是 obs 之后”的时间关系。

## Obs Memory 构建

obs20 不是被池化成一个向量后丢掉细节，而是进入 memory：

```text
obs_tokens = obs_input_process(obs_motion) # [20,B,D]
obs_tokens = add_window_pos(obs_tokens, start=0)
obs_tokens = obs_encoder(obs_tokens)       # [20,B,D]
obs_summary = mean(obs_tokens, dim=0)      # [B,D]
```

obs_encoder：

```text
TransformerEncoderLayer(d_model=D, nhead=H, dim_feedforward=FFN, dropout=0.1, activation="gelu")
num_obs_layers = 2
```

最终 memory 不只包含 summary，而是包含完整 obs token：

```text
memory = concat([global_token, obs_tokens], dim=0) # [21,B,D]
```

## Global Condition Token

复用 CMDM 的 timestep/action 条件方式，但不再只把它加到未来 token 上，而是作为 decoder memory 的第一个 token。

```text
time_token = embed_timestep(timesteps) # [1,B,D]
action_token = embed_action(action).unsqueeze(0) # [1,B,D]
obs_summary_token = obs_summary.unsqueeze(0) # [1,B,D]
```

Action classifier-free dropout：

```text
if y.get("uncond", False):
    action_token = 0
elif training and cond_mask_prob > 0:
    action_token = action_token * Bernoulli(1 - cond_mask_prob)
```

注意：

```text
uncond 只移除 action，不移除 obs。
```

原因：

- 采样时我们比较的是同一 obs20 下不同 action label。
- 无条件分支应表示“给定 obs，但没有动作语义标签”。

融合：

```text
global_token = condition_fuse(time_token + action_token + obs_summary_token)
```

`condition_fuse` 第一版可为：

```text
LayerNorm(D) + Linear(D,D)
```

## Future Decoder

future40 是 diffusion target，decoder 的 target sequence 是 noisy future tokens：

```text
future_tokens = future_input_process(x_t)       # [40,B,D]
future_tokens = add_window_pos(future_tokens, start=20)
```

最终 denoiser：

```text
decoded = seqTransDecoder(
    tgt=future_tokens,
    memory=memory,
    tgt_mask=None,
    memory_mask=None,
)
```

关键决定：

```text
tgt_mask = None
```

原因：

- diffusion denoising 不是自回归预测。
- 模型在每个 diffusion step 同时去噪整段 future40。
- future tokens 之间应允许双向 self-attention，以建模整段动作一致性。

输出：

```text
pred_future = output_process(decoded) # [B,56,6,40]
```

## 完整 forward 伪代码

```python
def forward(self, x_t, timesteps, y):
    obs = y["obs_motion"]
    action = y["action"]
    force_uncond = y.get("uncond", False)

    obs_tokens = self.obs_input_process(obs)              # [20,B,D]
    obs_tokens = self.add_window_pos(obs_tokens, 0)
    obs_tokens = self.obs_encoder(obs_tokens)             # [20,B,D]
    obs_summary = obs_tokens.mean(dim=0, keepdim=True)    # [1,B,D]

    time_token = self.embed_timestep(timesteps)           # [1,B,D]
    action_token = self.embed_action(action).unsqueeze(0) # [1,B,D]
    action_token = self.mask_action(action_token, force_uncond)

    global_token = self.condition_fuse(
        time_token + action_token + obs_summary
    )                                                     # [1,B,D]
    memory = torch.cat([global_token, obs_tokens], dim=0) # [21,B,D]

    future_tokens = self.future_input_process(x_t)        # [40,B,D]
    future_tokens = self.add_window_pos(future_tokens, 20)

    decoded = self.seqTransDecoder(
        tgt=future_tokens,
        memory=memory,
        tgt_mask=None,
    )                                                     # [40,B,D]
    pred = self.output_process(decoded)                   # [B,56,6,40]
    return pred
```

## 与 CMDM online Decoder 的关系

原始 CMDM online：

```text
xseq = x + cmotion
output = TransformerDecoder(tgt=xseq, memory=emb)
```

本模型：

```text
tgt = noisy future40
memory = [timestep + action + obs_summary, obs20 tokens]
output = TransformerDecoder(tgt=tgt, memory=memory)
```

这是对 CMDM 的结构性复用：

- 仍然是 conditional diffusion。
- 仍然是 timestep/action-conditioned denoising。
- 仍然用 CMDM-style TransformerDecoder。
- 只是把 condition motion 从“逐帧 actor motion”改成“prefix obs memory”。

## 训练目标

模型预测：

```text
pred_xstart = clean future40
```

对应 diffusion 设置：

```text
model_mean_type = START_X
loss_type = MSE
```

第一阶段训练 loss：

```text
L = L_rot + λ_vel L_vel + λ_trans L_trans
```

建议：

```text
L_rot = mse(pred_future, target_future)
L_vel = mse(diff_t(pred_future), diff_t(target_future))
L_trans = mse(root_translation(pred_future), root_translation(target_future))
```

正式阶段可加：

```text
L_rel_root
L_inter_person_distance
L_xyz_small_batch
```

但 `L_xyz` 不作为 smoke 必选项，避免 SMPL-X 转换拖慢调试。

## Diffusion 采样

采样输入：

```text
obs_motion [B,56,6,20]
action [B,1]
shape = [B,56,6,40]
```

CFG wrapper：

```text
pred_cond = model(x_t, t, {"obs_motion": obs, "action": action})
pred_uncond = model(x_t, t, {"obs_motion": obs, "action": action, "uncond": True})
pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

推荐采样配置：

```text
train diffusion steps = 1000
smoke sample steps = 50 或直接 p_sample_loop 小 batch
正式可用 DDIM 50/100 做快速比较
guidance_scale = 1.0, 2.0, 3.0 做 sweep
```

## Label-conditioned generation 比较

固定同一 obs20：

```text
label 2: pushing other person
label 5: hugging other person
label 8: handshaking
label 17: high-five
```

输出：

```text
future40_push
future40_hug
future40_handshake
future40_highfive
```

需要比较：

```text
motion distance between generated futures
root trajectory difference
inter-person distance curve
action classifier consistency
可视化视频
```

如果 label swap 输出几乎相同，说明 action condition 被忽略，应优先检查：

1. action token 是否真的进入 memory。
2. CFG uncond 是否只 mask action。
3. 训练中 action dropout 是否过高或过低。
4. dataset 是否因 obs20 已经决定未来而削弱标签作用。
5. 是否需要把 action token 作为单独 memory token，而不是只与 time/obs summary 相加。

## 配置定稿

Smoke 同架构缩小版：

```text
latent_dim = 128
num_obs_layers = 1
num_decoder_layers = 2
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
num_obs_layers = 2
num_decoder_layers = 4
num_heads = 4
ff_size = 1024
dropout = 0.1
batch_size = 4
grad_accum_steps = 4
num_steps = 5000
lr = 1e-4
weight_decay = 1e-4
cond_mask_prob = 0.1
```

3080 保守配置：

```text
latent_dim = 192
num_obs_layers = 1
num_decoder_layers = 3
batch_size = 2
grad_accum_steps = 8
```

## 第一版之后的计划

### P0 Dataset + Gate

实现 `NTULabelForecastDataset`：

```text
T>=60
random crop train
center crop test
obs20/future40/action
```

### P1 Final Architecture Smoke

注意：smoke 也使用最终 Decoder 架构，只缩小层数和 hidden size，不再用 Encoder-only 临时架构。

检查：

```text
forward shape
loss finite
CFG wrapper forward
```

### P2 2-step Diffusion Train

跑：

```text
num_steps = 2
save checkpoint
load checkpoint
sample future40
```

### P3 Label Swap Smoke

同一 obs20，生成多个标签结果，至少保存 `.npy` 和可视化。

### P4 Single-seed Formal Train

使用正式配置跑 5000 steps。

输出：

```text
metrics_test.json
label_swap_summary.json
train_log.jsonl
checkpoint
```

### P5 Label Consistency Evaluator

训练或复用动作分类器：

```text
future40 -> action label
```

计算：

```text
classification accuracy
handshaking subset accuracy
label swap consistency
```

### P6 Ablation

至少做：

```text
no_action: obs20 only
no_obs: action only
encoder_only: 上一版轻量 Encoder 架构
decoder_final: 本最终架构
```

目的：

- 证明 action label 有效。
- 证明 obs prefix 有效。
- 证明最终 decoder memory 比轻量 encoder 更合理。

### P7 3-seed

仅当 P4/P5/P6 合理后运行：

```text
seed = 0,1,2
```

## 成功标准

模型层面：

```text
pred_future shape = [B,56,6,40]
loss finite
sampling finite
CFG 可用
```

任务层面：

```text
future40 reconstruction metrics 可计算
label swap 输出有明显差异
handshaking 生成可视化合理
action classifier consistency 高于 obs-only baseline
```

论文表述层面：

```text
可以称为 ReGenNet/CMDM-derived label-conditioned two-person forecasting diffusion。
不能称为原始 CMDM 直接训练 20->40，因为 condition path 已重构。
```

## 当前最终决策

后续实现以本架构为准：

```text
ForecastingCMDM-Decoder
obs20 memory + action/timestep global token
future40 noisy target decoder
predict clean future40
```

上一版 `ForecastingCMDM` 文档中的 Encoder-only 方案只作为 ablation 或极简 debug 方案，不再作为最终模型目标。
