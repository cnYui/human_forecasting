# ForecastingCMDM 最终目标架构 v3

## 目标结论

最终目标模型确定为：

```text
ForecastingCMDMDecoder
```

它不是“轻量 Transformer Encoder + global token”的临时 smoke 结构。Smoke 也必须使用同一个最终 Decoder 架构，只缩小 hidden size、层数和 batch。

最终任务协议固定为：

```text
dataset = NTU120 2P
split = NTU120 原始 xsub train / test
window_len = 60
obs_len = 20
pred_len = 40
condition = obs20 + action label
target = future40
body_model = smplx
representation = rot6d + root translation
input shape = [B,56,6,T]
num_actions = 26
handshaking = A009 = label 8
```

一句话架构：

```text
noisy future40 作为 TransformerDecoder 的 target；
obs20 编码成 memory；
timestep token、action token、obs summary token 与 obs20 token 一起作为 decoder memory；
输出 denoised future40。
```

## 为什么不能继续用轻量 Encoder 当最终架构

上一版轻量 Encoder 方案：

```text
global_cond = timestep + action + pooled obs
concat([global_cond, future_tokens]) -> TransformerEncoder -> future40
```

它的问题是：

```text
obs20 被压缩成一个向量，细节损失大；
action label 只和 timestep/obs summary 相加，容易被忽略；
没有显式 cross-attention，不像 CMDM online decoder 的条件生成结构；
只能作为 debug / ablation，不能作为最终模型目标。
```

最终模型必须让 future40 的每个 token 显式读取 obs20 的完整 token 序列，因此使用：

```text
TransformerDecoder(tgt=future40_noisy_tokens, memory=condition_memory)
```

## 从 CMDM 学到并复用的内容

这里复用的是 CMDM/ReGenNet 的条件扩散建模范式和可迁移组件，而不是直接复用原始 actor->reactor 数据协议。

源码对应关系：

| CMDM 源码组件 | 最终模型用法 |
|---|---|
| `InputProcess` | 分别编码 `obs20` 和 noisy `future40` |
| `OutputProcess` | 将 decoder 输出还原为 `[B,56,6,40]` |
| `PositionalEncoding` | 使用同一 sinusoidal PE，但 obs 用位置 `0..19`，future 用 `20..59` |
| `TimestepEmbedder` | 作为 diffusion timestep token |
| `EmbedAction` | 作为 action label token |
| `mask_cond` | 复用 classifier-free guidance 的 action dropout 语义 |
| `TransformerDecoder` online 思想 | target 是 noisy future，memory 是条件 token |
| `ClassifierFreeSampleModel` 思想 | 写 forecasting 专用 CFG wrapper，uncond 只 mask action |
| `Rotation2xyz_x` | SMPL-X rot6d 转 xyz，用于 MPJPE / joint MSE 评估 |
| `GaussianDiffusion` | 继续使用 ReGenNet diffusion 训练和采样框架 |

不能直接复用的部分：

```text
y["cmotion"] 逐帧 add/concat 到 x 的逻辑
actor full motion -> reactor full motion 的 collate 协议
原始 CMDM.forward 中要求 x 和 cmotion 同长度的假设
```

原因：

```text
原始 CMDM: noisy reactor full motion + actor full motion + label -> reactor full motion
当前任务: noisy two-person future40 + two-person obs20 + label -> two-person future40
```

`obs20` 和 `future40` 长度不同，不能再逐帧相加。正确迁移方式是把 `obs20` 编成 decoder memory。

## 模型文件与类

新增主文件：

```text
model/forecasting_cmdm.py
```

核心类：

```text
ForecastingCMDMDecoder(nn.Module)
ForecastingClassifierFreeSampleModel(nn.Module)
```

优先从 `model/cmdm.py` 直接 import 可复用组件：

```python
from model.cmdm import (
    InputProcess,
    OutputProcess,
    PositionalEncoding,
    TimestepEmbedder,
    EmbedAction,
)
```

如果后续发现循环依赖或 import 副作用，再把这些组件无行为改动地抽到 `model/cmdm_components.py`，并保持原始 `CMDM` 可用。

## 输入输出协议

Dataset batch：

```text
obs_motion: [B,56,6,20]
future:     [B,56,6,40]
action:     [B,1]
mask:       [B,1,1,40]
lengths:    [B] optional, 当前固定 40
sample_id:  List[str]
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
x_t:                  [B,56,6,40]
timesteps:            [B]
y["obs_motion"]:      [B,56,6,20]
y["action"]:          [B,1]
y["mask"]:            [B,1,1,40]
y.get("uncond"):      bool, CFG 采样时使用
```

输出：

```text
pred_xstart: [B,56,6,40]
```

## 最终架构图

```text
obs_motion [B,56,6,20]
  -> obs_input_process
  -> add absolute PE 0..19
  -> add obs_frame_type_embedding
  -> obs_encoder
  -> obs_tokens [20,B,D]
  -> mean pool
  -> obs_summary_token [1,B,D]

timesteps [B]
  -> TimestepEmbedder
  -> time_token [1,B,D]

action [B,1]
  -> EmbedAction
  -> action CFG mask/dropout
  -> action_token [1,B,D]

condition_memory =
  concat([
    time_token,
    action_token,
    obs_summary_token,
    obs_tokens
  ], dim=0)
  -> memory_norm / memory_proj
  -> [23,B,D]

x_t future40 [B,56,6,40]
  -> future_input_process
  -> add absolute PE 20..59
  -> add future_type_embedding
  -> future_tokens [40,B,D]

TransformerDecoder(
  tgt = future_tokens,
  memory = condition_memory,
  tgt_mask = None
)
  -> decoded_future_tokens [40,B,D]
  -> output_process
  -> pred_xstart [B,56,6,40]
```

关键决定：

```text
action token 单独作为 memory token，不再只与 timestep 相加。
tgt_mask = None，因为 diffusion 是整段 future40 同时去噪，不是自回归预测。
obs 不在 CFG uncond 时清零，uncond 只表示“没有动作标签”，仍然给定 obs20。
```

## 具体模块定义

核心参数：

```text
njoints = 56
nfeats = 6
input_feats = 336
latent_dim = 256 formal, 128 smoke
num_actions = 26
obs_len = 20
pred_len = 40
window_len = 60
```

模块清单：

```text
self.obs_input_process = InputProcess("rot6d", input_feats, latent_dim)
self.future_input_process = InputProcess("rot6d", input_feats, latent_dim)
self.output_process = OutputProcess("rot6d", input_feats, latent_dim, 56, 6)
self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
self.embed_action = EmbedAction(num_actions, latent_dim)

self.obs_encoder = nn.TransformerEncoder(...)
self.seqTransDecoder = nn.TransformerDecoder(...)

self.time_type = nn.Parameter(torch.zeros(1,1,D))
self.action_type = nn.Parameter(torch.zeros(1,1,D))
self.obs_summary_type = nn.Parameter(torch.zeros(1,1,D))
self.obs_frame_type = nn.Parameter(torch.zeros(1,1,D))
self.future_type = nn.Parameter(torch.zeros(1,1,D))

self.memory_norm = nn.LayerNorm(D)
self.memory_proj = nn.Linear(D,D)
self.future_norm = nn.LayerNorm(D)
self.rot2xyz = Rotation2xyz_x(...)
```

正式结构：

```text
obs_encoder_layers = 2
decoder_layers = 4
num_heads = 4
ff_size = 1024
dropout = 0.1
activation = "gelu"
cond_mask_prob = 0.1
```

3080 保守结构：

```text
latent_dim = 192
obs_encoder_layers = 1
decoder_layers = 3
batch_size = 2
grad_accum_steps = 8
```

Smoke 结构：

```text
latent_dim = 128
obs_encoder_layers = 1
decoder_layers = 2
num_heads = 4
ff_size = 512
batch_size = 1
num_steps = 2
```

注意：3080 保守结构和 smoke 结构只能缩小宽度和层数，不能换成 Encoder-only。

## 位置编码方案

必须使用 60 帧窗口内的绝对位置：

```text
obs positions = 0..19
future positions = 20..59
```

实现辅助函数：

```python
def add_window_pos(self, tokens, start):
    pe = self.sequence_pos_encoder.pe[start:start + tokens.shape[0]]
    return self.sequence_pos_encoder.dropout(tokens + pe)
```

不要让 obs 和 future 都从位置 0 开始，否则模型不知道 future 是 obs 后面的时间段。

## CFG 设计

训练时对 action 做 classifier-free dropout：

```text
cond_mask_prob = 0.1
```

mask 规则：

```text
训练随机 mask action token
采样 uncond=True 时强制 action token = 0
obs_motion 永远保留
timestep 永远保留
```

原因：

```text
目标是比较同一 obs20 下不同 action label 的生成差异；
无条件分支应表示“给定观测前缀，但没有动作语义标签”。
```

Forecasting CFG wrapper：

```python
pred_cond = model(x_t, t, y)
y_uncond = deepcopy(y)
y_uncond["uncond"] = True
pred_uncond = model(x_t, t, y_uncond)
pred = pred_uncond + scale * (pred_cond - pred_uncond)
```

推荐 sweep：

```text
guidance_scale = 1.0, 2.0, 3.0
```

如果 label swap 结果几乎相同，优先排查：

```text
action token 是否进入 memory
uncond 是否只 mask action
cond_mask_prob 是否过高
训练样本是否被大类主导
是否需要 class-balanced sampler
```

## Forward 伪代码

```python
def forward(self, x_t, timesteps, y):
    obs = y["obs_motion"]                  # [B,56,6,20]
    action = y["action"]                   # [B,1]
    force_uncond = y.get("uncond", False)

    obs_tokens = self.obs_input_process(obs)       # [20,B,D]
    obs_tokens = self.add_window_pos(obs_tokens, 0)
    obs_tokens = obs_tokens + self.obs_frame_type
    obs_tokens = self.obs_encoder(obs_tokens)      # [20,B,D]

    obs_summary = obs_tokens.mean(dim=0, keepdim=True)
    obs_summary = obs_summary + self.obs_summary_type

    time_token = self.embed_timestep(timesteps) + self.time_type

    action_emb = self.embed_action(action)         # [B,D]
    action_emb = self.mask_cond(action_emb, force_mask=force_uncond)
    action_token = action_emb.unsqueeze(0) + self.action_type

    memory = torch.cat(
        [time_token, action_token, obs_summary, obs_tokens],
        dim=0,
    )                                             # [23,B,D]
    memory = self.memory_proj(self.memory_norm(memory))

    future_tokens = self.future_input_process(x_t) # [40,B,D]
    future_tokens = self.add_window_pos(future_tokens, 20)
    future_tokens = self.future_norm(future_tokens + self.future_type)

    decoded = self.seqTransDecoder(
        tgt=future_tokens,
        memory=memory,
        tgt_mask=None,
        memory_mask=None,
    )                                             # [40,B,D]

    return self.output_process(decoded)            # [B,56,6,40]
```

## Diffusion 训练目标

第一版训练采用 CMDM/ReGenNet 常用的 clean motion prediction：

```text
model_mean_type = START_X
loss_type = MSE
target = x_start = clean future40
```

基础 loss：

```text
L_rot = mse(pred_future, target_future)
L_vel = mse(diff_t(pred_future), diff_t(target_future))
L_trans = mse(root_translation(pred_future), root_translation(target_future))
L_rel_root = mse(relative_root(pred_future), relative_root(target_future))
```

第一阶段权重：

```text
L = 1.0 * L_rot
  + 0.5 * L_vel
  + 0.5 * L_trans
  + 0.5 * L_rel_root
```

`L_xyz` 不作为 smoke 必选训练项。正式训练可加低频 xyz loss，例如每 N step 小 batch 计算：

```text
L_xyz = joint_mse(rot2xyz(pred), rot2xyz(target))
```

但主评估必须转 xyz 计算：

```text
MPJPE
joint MSE
root translation error
relative root distance error
inter-person distance consistency
```

## 数据与采样策略

Dataset：

```text
data_loaders/forecasting/ntu_label.py
class NTULabelForecastDataset
```

过滤：

```text
只保留 T >= 60
不 padding
不重采样
```

Train crop：

```text
从每条 T>=60 序列随机裁剪连续 60 帧
obs = frames[0:20]
future = frames[20:60]
```

Test crop：

```text
center crop 连续 60 帧
```

类别不平衡处理：

```text
必须记录每类 train/test 数量
正式训练默认启用 class-balanced sampler 或 per-class repeat sampling
所有指标同时报告 overall 和 per-class
```

## 评估闭环

### 预测误差

对 test split 计算：

```text
rot_mse
future_joint_mse_xyz
future_mpjpe
short/mid/long MPJPE: 1-13, 14-26, 27-40
root_translation_error
relative_root_distance_error
inter_person_distance_consistency
```

### 标签一致性

必须补一个动作分类一致性评估，否则无法证明“输入 handshaking 后输出就是 handshaking”。

分类器建议：

```text
eval/action_consistency_classifier.py
input = future40, 可用 rot6d 或 xyz
model = temporal transformer / temporal CNN
train = real future40 from NTU120 train split
test = real future40 from NTU120 test split
then evaluate generated future40
```

生成结果指标：

```text
classifier_consistency_acc
handshaking_consistency_acc
per_class_consistency_acc
label_swap_classifier_acc
```

### Label swap

固定同一个 `obs20`，替换动作标签：

```text
label 2: pushing other person
label 5: hugging other person
label 8: handshaking
label 17: high-five
```

保存：

```text
generated_future40.npy
metadata.json
xyz.npy
mp4 visualization
label_swap_summary.json
```

判断：

```text
同一 obs20 下，不同 label 的 future40 应有可观察差异；
handshaking label 的生成结果应在分类器上更容易被判为 handshaking；
不能只看 MSE，因为 MSE 会奖励平均动作。
```

## 第一版之后的完整计划

### P0 数据 gate

产物：

```text
T>=60 train/test 总数
每类 train/test 数量
handshaking train/test 数量
obs/future/action shape
finite 检查
```

通过标准：

```text
train/test 均覆盖 26 类
handshaking train/test 非零
batch shape = [B,56,6,20] 和 [B,56,6,40]
```

### P1 最终 Decoder 架构 smoke

使用 `ForecastingCMDMDecoder` 缩小版。

通过标准：

```text
forward output shape = [B,56,6,40]
loss finite
backward finite
CFG cond/uncond forward finite
```

### P2 2-step diffusion train smoke

产物：

```text
checkpoint
train_log.jsonl
metrics_smoke.json
sample_future40.npy
```

通过标准：

```text
能训练 2 step
能保存和加载 checkpoint
能从噪声采样 future40
输出没有 NaN/Inf
```

### P3 Label swap smoke

用 P2 checkpoint 或短训 checkpoint 生成：

```text
same obs20 + labels [2,5,8,17]
```

通过标准：

```text
采样流程跑通
每个 label 的输出文件完整
可视化可生成
先不要求语义一定正确
```

### P4 动作分类器 gate

先在 real future40 上训练/验证动作分类器。

通过标准：

```text
real test classification accuracy 明显高于 random 1/26
handshaking real test 可被识别
```

如果分类器在真实数据上都失败，不能用它评价生成结果。

### P5 正式单 seed 训练

正式命令目标：

```text
save/forecasting/ntu120_label/forecasting_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000
```

配置：

```text
latent_dim = 256
decoder_layers = 4
obs_encoder_layers = 2
num_heads = 4
batch_size = 4
grad_accum_steps = 4
num_steps = 5000
lr = 1e-4
weight_decay = 1e-4
cond_mask_prob = 0.1
```

3080 OOM 时只降为：

```text
latent_dim = 192
decoder_layers = 3
obs_encoder_layers = 1
batch_size = 2
grad_accum_steps = 8
```

不能改回 Encoder-only。

### P6 正式评估

产物：

```text
metrics_test.json
metrics_per_class.json
label_consistency.json
label_swap_summary.json
visualizations/
```

必须包括：

```text
MPJPE / joint MSE
handshaking subset metrics
classification consistency
label swap 对比
```

### P7 Ablation

至少做：

```text
decoder_final: 完整模型
no_action: mask action，只给 obs20
no_obs_tokens: 只给 obs summary，不给 obs20 token memory
encoder_only: 旧轻量 Encoder 结构，只作为 ablation
no_cfg: 不做 action dropout / guidance
```

目的：

```text
证明 action label 对生成有影响
证明 obs20 token memory 比只池化更合理
证明最终 Decoder 不是随便换结构
```

### P8 3-seed

当 P5/P6 合理后运行：

```text
seed = 0,1,2
```

报告：

```text
mean ± std
overall + per-class
handshaking subset
```

## 推荐入口命令形态

Smoke：

```bash
python -m train.train_label_forecasting_diffusion \
  --dataset ntu120_2p \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke \
  --model_type forecasting_cmdm_decoder \
  --body_model smplx \
  --window_len 60 \
  --obs_len 20 \
  --pred_len 40 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_steps 2 \
  --latent_dim 128 \
  --decoder_layers 2 \
  --obs_encoder_layers 1 \
  --num_heads 4 \
  --cond_mask_prob 0.1 \
  --num_workers 0 \
  --seed 0 \
  --overwrite
```

正式单 seed：

```bash
python -m train.train_label_forecasting_diffusion \
  --dataset ntu120_2p \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/forecasting_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000 \
  --model_type forecasting_cmdm_decoder \
  --body_model smplx \
  --window_len 60 \
  --obs_len 20 \
  --pred_len 40 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --eval_batch_size 4 \
  --num_steps 5000 \
  --save_interval 1000 \
  --eval_interval 1000 \
  --latent_dim 256 \
  --decoder_layers 4 \
  --obs_encoder_layers 2 \
  --num_heads 4 \
  --cond_mask_prob 0.1 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --seed 0
```

## 成功标准

工程层面：

```text
dataset gate 通过
final decoder smoke 通过
2-step diffusion train 通过
checkpoint load/sample 通过
label swap sample 通过
```

模型层面：

```text
loss finite 且下降趋势正常
MPJPE / joint MSE 可计算
生成结果没有 NaN/Inf 和明显爆炸
```

任务层面：

```text
同一 obs20 下不同 label 生成 future40 有可观察差异
handshaking label 的输出在可视化和分类一致性上更接近 handshaking
classification consistency 高于 no_action baseline
decoder_final 优于或至少不弱于 encoder_only ablation
```

论文/汇报表述：

```text
可以称为 ReGenNet/CMDM-derived label-conditioned two-person motion forecasting diffusion。
不能称为原始 CMDM.forward 原封不动用于 20->40，因为 condition path 已按 forecasting 重构。
可以说复用了 CMDM 的条件扩散、timestep/action conditioning、TransformerDecoder denoiser、CFG 和 SMPL-X xyz 评估链路。
```

## 当前最终决定

后续实现以本文件为准：

```text
最终模型 = ForecastingCMDMDecoder
最终结构 = future noisy target decoder + obs20/action/timestep memory
第一版 smoke = 同架构缩小版
旧 Encoder-only = 仅保留为 ablation/debug
```
