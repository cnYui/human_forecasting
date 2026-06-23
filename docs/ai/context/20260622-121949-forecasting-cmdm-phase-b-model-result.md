# ForecastingCMDMDecoder 阶段 B 模型实现结果

## 结论

阶段 B 模型 gate 已完成并通过。

本阶段新增：

```text
model/forecasting_cmdm.py
```

本阶段新增结果记录：

```text
docs/ai/context/20260622-121949-forecasting-cmdm-phase-b-model-result.md
```

未改动：

```text
model/cmdm.py
model/cfg_sampler.py
train/train_mdm.py
train/*
eval/*
sample/*
```

`model.cmdm` 组件可直接 import，因此未抽取 `model/cmdm_components.py`。

## 实现内容

新增类：

```text
ForecastingCMDMDecoder
ForecastingClassifierFreeSampleModel
```

新增工具函数：

```text
count_parameters
_ensure_finite
_normalize_action
_normalize_timesteps
```

复用组件：

```text
InputProcess
OutputProcess
PositionalEncoding
TimestepEmbedder
EmbedAction
```

核心结构：

```text
obs_motion [B,56,6,20]
  -> InputProcess
  -> PE 0..19
  -> obs frame type
  -> TransformerEncoder
  -> obs_tokens

time_token / action_token / obs_summary / obs_tokens
  -> concat memory
  -> LayerNorm + Linear

x_t future [B,56,6,40]
  -> InputProcess
  -> PE 20..59
  -> future type + LayerNorm
  -> TransformerDecoder(tgt=future_tokens, memory=memory, tgt_mask=None)
  -> OutputProcess
  -> pred [B,56,6,40]
```

CFG 语义：

```text
ForecastingClassifierFreeSampleModel 跑 cond/uncond 两次 forward
uncond 只设置 y["uncond"] = True
ForecastingCMDMDecoder.mask_action 只 mask action embedding
obs_motion 始终保留
```

`rot2xyz`：

```text
init_rot2xyz 默认 False
阶段 B smoke 未初始化 Rotation2xyz_x
self.rot2xyz = None
```

原因：

```text
阶段 B 只验证模型 forward/backward，不让 SMPL-X 资产或评估组件阻塞模型 gate。
```

## 参数量

Smoke 配置：

```text
latent_dim = 128
obs_encoder_layers = 1
decoder_layers = 2
num_heads = 4
ff_size = 512
cond_mask_prob = 0.1
init_rot2xyz = False
```

参数量：

```text
trainable_parameters = 911056
```

## 验证环境

```text
python_executable = /home/rpartx3080/.local/micromamba/envs/regennet/bin/python
torch_version = 1.7.1
```

## 验证结果

### 1. CMDM 组件 import

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
from model.cmdm import InputProcess, OutputProcess, PositionalEncoding, TimestepEmbedder, EmbedAction
print('cmdm components import ok')
PY
```

结果：

```text
cmdm components import ok
exit_code = 0
```

### 2. 静态编译

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile model/forecasting_cmdm.py
```

结果：

```text
exit_code = 0
```

### 3. 随机输入 forward/backward/CFG smoke

结果：

```text
ok 911056
exit_code = 0
```

通过内容：

```text
forward output shape = [1,56,6,40]
loss.backward() 通过
ForecastingClassifierFreeSampleModel 输出 shape = [1,56,6,40]
CFG 输出 finite
```

### 4. Dataset batch smoke

使用：

```text
dataset/ntu120/smplx/conditioned/xsub.train.h5
max_samples = 4
batch_size = 2
```

结果：

```text
dataset batch ok (2, 56, 6, 40)
exit_code = 0
```

通过内容：

```text
batch["future"] 可作为 x_t
batch["obs_motion"] / batch["action"] / batch["mask"] 可直接作为 y
输出 shape = [2,56,6,40]
输出 finite
loss.backward() 通过
```

### 5. Shape guard 负例

结果：

```text
bad_future_len ok ValueError x_t 后三维必须是 (56, 6, 40)，当前为 (56, 6, 39)
bad_obs_len ok ValueError obs_motion 必须是 (1, 56, 6, 20)，当前为 (1, 56, 6, 19)
bad_action ok ValueError action 必须在 [0,25] 内
missing_obs ok ValueError y['obs_motion'] 不能为空
float_timestep ok ValueError timesteps 必须是整数 index；当前模型要求 diffusion rescale_timesteps=False
shape guard ok
exit_code = 0
```

### 6. CFG 结构语义 smoke

验证：

```text
model.eval()
uncond=True 时，同 obs 不同 action 输出完全相同
uncond=True 时，不同 obs 输出不同
cond 时，同 obs 不同 action 输出不同
```

结果：

```text
cfg structure ok 0.16933289170265198 0.08417806029319763
exit_code = 0
```

解释：

```text
第一个数是 uncond 下不同 obs 的最大输出差异，说明 obs_motion 没有被清零。
第二个数是 cond 下不同 action 的最大输出差异，说明 action token 进入了模型。
```

## 阶段 B 退出条件核对

已满足：

```text
model/forecasting_cmdm.py 存在
ForecastingCMDMDecoder 可 import
ForecastingClassifierFreeSampleModel 可 import
py_compile 通过
随机输入 forward/backward/CFG 通过
dataset batch forward/backward 通过
action token 单独进入 memory
CFG uncond 只 mask action
obs PE 使用 0..19
future PE 使用 20..59
未修改 train/train_mdm.py
未把模型降级为 Encoder-only
```

## 下一阶段

允许进入下一阶段：

```text
train/train_label_forecasting_diffusion.py
2-step diffusion train smoke
checkpoint save/resume gate
```

仍不建议直接进入：

```text
label swap
正式训练
动作一致性分类器
```

这些需要等待 2-step diffusion train 和 checkpoint gate 通过。
