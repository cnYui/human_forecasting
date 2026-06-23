# ForecastingCMDMDecoder 阶段 B 模型实现 Plan

## 参考上下文

本 plan 依据：

```text
docs/ai/context/20260622-114122-forecasting-cmdm-phase-a-data-gate-result.md
docs/ai/context/20260622-114726-forecasting-cmdm-phase-b-model-design.md
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
docs/ai/context/20260622-105946-forecasting-cmdm-final-target-architecture-v3.md
```

## 阶段 B 定义

本文中的阶段 B 是模型 gate：

```text
新增 model/forecasting_cmdm.py
实现 ForecastingCMDMDecoder
实现 ForecastingClassifierFreeSampleModel
完成 py_compile、随机输入 forward/backward/CFG、dataset batch forward/backward
```

它不是采样生成闭环。以下内容不进入本阶段：

```text
train/train_label_forecasting_diffusion.py
checkpoint save/resume
sample/sample_label_forecasting_diffusion.py
label swap
eval/action_consistency_classifier.py
formal metrics
```

## 当前输入条件

阶段 A 已通过：

```text
train kept_count = 1956
test kept_count = 1253
train/test 均覆盖 26 类
handshaking label 8 train/test 非零
batch obs_motion = [B,56,6,20]
batch future = [B,56,6,40]
batch action = [B,1]
batch mask = [B,1,1,40]
obs_motion/future finite scan PASS
```

阶段 B 直接依赖：

```python
x_t = batch["future"]
y = {
    "obs_motion": batch["obs_motion"],
    "action": batch["action"],
    "mask": batch["mask"],
}
```

验证命令统一使用：

```text
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python
```

## 修改范围

本阶段只新增：

```text
model/forecasting_cmdm.py
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-b-model-result.md
```

原则上不修改：

```text
model/cmdm.py
model/cfg_sampler.py
data_loaders/forecasting/ntu_label.py
train/*
eval/*
sample/*
```

只有当 `from model.cmdm import ...` 明确出现循环依赖或副作用时，才新增：

```text
model/cmdm_components.py
```

如果发生该情况，必须先补一份新 context 文档说明抽取原因和无行为改动边界。

## 实现步骤

### 1. 新建模型文件骨架

创建：

```text
model/forecasting_cmdm.py
```

文件内导入：

```python
import torch
import torch.nn as nn
from copy import deepcopy

from model.cmdm import (
    InputProcess,
    OutputProcess,
    PositionalEncoding,
    TimestepEmbedder,
    EmbedAction,
)
```

暂不导入 `clip`，不复用 `CMDM` 主类。

### 2. 实现工具函数

需要实现：

```text
count_parameters(model)
_ensure_finite(name, tensor)
_normalize_action(action, batch_size, num_actions, device)
```

目的：

```text
让模型 gate 的错误信息明确
避免 shape 错误延迟到 Transformer 内部才爆炸
保持后续训练入口能记录参数量
```

### 3. 实现 ForecastingCMDMDecoder.__init__

必要属性：

```text
self.model_type = "forecasting_cmdm_decoder"
self.njoints = 56
self.nfeats = 6
self.input_feats = 336
self.num_actions = 26
self.obs_len = 20
self.pred_len = 40
self.window_len = 60
self.data_rep = "rot6d"
self.cond_mode = "action"
self.cond_mask_prob = 0.1
self.translation = True
self.pose_rep = "rot6d"
self.glob = True
self.glob_rot = None
self.body_model = "smplx"
self.dataset = "ntu120_2p"
```

必要模块：

```text
obs_input_process
future_input_process
output_process
sequence_pos_encoder
embed_timestep
embed_action
obs_encoder
seqTransDecoder
time_type
action_type
obs_summary_type
obs_frame_type
future_type
memory_norm
memory_proj
future_norm
```

PyTorch 1.7.1 兼容要求：

```text
nn.TransformerEncoderLayer 不使用 batch_first
nn.TransformerDecoderLayer 不使用 batch_first
不使用 norm_first
activation 使用 "gelu"
输入输出统一保持 [T,B,D]
```

`rot2xyz` 策略：

```text
init_rot2xyz 默认 False
init_rot2xyz=False 时 self.rot2xyz = None
init_rot2xyz=True 时再导入并初始化 Rotation2xyz_x
```

原因：

```text
阶段 B 只测模型 forward/backward，不应被 SMPL-X 资产或评估组件阻塞。
```

### 4. 实现位置编码

实现：

```python
def add_window_pos(self, tokens, start):
    pe = self.sequence_pos_encoder.pe[start:start + tokens.shape[0]]
    return self.sequence_pos_encoder.dropout(tokens + pe)
```

强制约定：

```text
obs start = 0
future start = 20
```

不能让 obs 和 future 都从 0 开始。

### 5. 实现 shape guard

Forward 开始时检查：

```text
x_t.dim() == 4
x_t.shape == [B,56,6,40]
timesteps.shape == [B]
timesteps dtype 可作为 PE index 使用，优先 long
y 是 dict
y["obs_motion"].shape == [B,56,6,20]
y["action"].shape == [B,1] 或 [B]
0 <= action < 26
x_t 和 obs_motion finite
```

注意：

```text
CMDM 的 TimestepEmbedder 用 timesteps 索引 positional encoding。
阶段 B 假设 diffusion rescale_timesteps=False。
如果后续训练传入 float timestep，必须另做连续 timestep embedding，不能在本阶段偷改。
```

### 6. 实现 action mask

实现：

```python
def mask_action(self, action_emb, force_mask=False):
    if force_mask:
        return torch.zeros_like(action_emb)
    if self.training and self.cond_mask_prob > 0.0:
        mask = torch.bernoulli(
            torch.ones(action_emb.shape[0], device=action_emb.device) * self.cond_mask_prob
        ).view(-1, 1)
        return action_emb * (1.0 - mask)
    return action_emb
```

语义：

```text
训练时随机 drop action
CFG uncond 时强制 drop action
obs_motion 永远保留
timestep 永远保留
```

### 7. 实现 forward

Forward 数据流：

```text
obs_motion -> obs_input_process -> PE 0..19 -> obs_frame_type -> obs_encoder
obs_tokens mean -> obs_summary_token
timesteps -> time_token
action -> EmbedAction -> mask_action -> action_token
memory = [time_token, action_token, obs_summary_token, obs_tokens]
memory -> memory_norm -> memory_proj
x_t future40 -> future_input_process -> PE 20..59 -> future_type -> future_norm
TransformerDecoder(tgt=future_tokens, memory=memory, tgt_mask=None)
OutputProcess -> [B,56,6,40]
```

强制检查：

```text
action token 必须单独进入 memory
tgt_mask 必须为 None
uncond 不得清零 obs_motion
输出 shape 必须等于 x_t.shape
```

### 8. 实现 config / parameters_wo_clip

`config()` 返回：

```text
model_type
njoints
nfeats
num_actions
obs_len
pred_len
window_len
latent_dim
obs_encoder_layers
decoder_layers
num_heads
ff_size
dropout
activation
cond_mask_prob
data_rep
body_model
dataset
init_rot2xyz
```

`parameters_wo_clip()`：

```python
return list(self.parameters())
```

原因：

```text
本模型不加载 CLIP。
```

### 9. 实现 ForecastingClassifierFreeSampleModel

Wrapper 行为：

```text
持有 self.model
暴露 rot2xyz / translation / njoints / nfeats / data_rep / cond_mode 等兼容属性
要求 model.cond_mask_prob > 0
forward 内分别跑 cond 和 uncond
uncond 只设置 y_uncond["uncond"] = True
scale 优先取 y["scale"]，否则取构造参数 guidance_scale
```

核心公式：

```text
out = out_uncond + scale * (out_cond - out_uncond)
```

不要修改原始 `model/cfg_sampler.py`。

## 验证顺序

### Gate 1: 静态编译

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile model/forecasting_cmdm.py
```

通过标准：

```text
exit_code = 0
```

### Gate 2: 随机输入模型 smoke

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
import torch
from model.forecasting_cmdm import ForecastingCMDMDecoder, ForecastingClassifierFreeSampleModel

model = ForecastingCMDMDecoder(
    njoints=56, nfeats=6, num_actions=26, obs_len=20, pred_len=40,
    latent_dim=128, obs_encoder_layers=1, decoder_layers=2,
    num_heads=4, ff_size=512, dropout=0.1, cond_mask_prob=0.1,
    init_rot2xyz=False,
)
x = torch.randn(1, 56, 6, 40)
y = {"obs_motion": torch.randn(1, 56, 6, 20), "action": torch.tensor([[8]])}
t = torch.tensor([10], dtype=torch.long)
out = model(x, t, y)
assert out.shape == x.shape
loss = out.square().mean()
loss.backward()
cfg = ForecastingClassifierFreeSampleModel(model, guidance_scale=2.0)
out_cfg = cfg(x.detach(), t, y)
assert out_cfg.shape == x.shape
assert torch.isfinite(out_cfg).all()
print("ok")
PY
```

通过标准：

```text
forward shape = [1,56,6,40]
loss finite
backward 不报错
CFG 输出 finite
```

### Gate 3: dataset batch smoke

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
import torch
from torch.utils.data import DataLoader
from data_loaders.forecasting.ntu_label import NTULabelForecastDataset, ntu_label_forecasting_collate
from model.forecasting_cmdm import ForecastingCMDMDecoder

dataset = NTULabelForecastDataset(
    "dataset/ntu120/smplx/conditioned/xsub.train.h5",
    split="train",
    window_len=60,
    obs_len=20,
    pred_len=40,
    max_samples=4,
    seed=0,
)
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=ntu_label_forecasting_collate,
)
batch = next(iter(loader))
model = ForecastingCMDMDecoder(
    njoints=56, nfeats=6, num_actions=26, obs_len=20, pred_len=40,
    latent_dim=128, obs_encoder_layers=1, decoder_layers=2,
    num_heads=4, ff_size=512, dropout=0.1, cond_mask_prob=0.1,
    init_rot2xyz=False,
)
t = torch.tensor([10, 20], dtype=torch.long)
y = {
    "obs_motion": batch["obs_motion"],
    "action": batch["action"],
    "mask": batch["mask"],
}
out = model(batch["future"], t, y)
assert out.shape == batch["future"].shape
assert torch.isfinite(out).all()
out.square().mean().backward()
print("dataset batch ok")
PY
```

通过标准：

```text
dataset batch 可直接进入模型
输出 shape = [2,56,6,40]
输出 finite
backward 不报错
```

### Gate 4: shape guard 负例 smoke

至少检查：

```text
错误 future 长度 39 应抛 ValueError
错误 obs 长度 19 应抛 ValueError
action = 26 应抛 ValueError
缺少 y["obs_motion"] 应抛 KeyError 或 ValueError
```

该 gate 可以用短脚本人工执行，不必新增测试文件。

## 完成后记录

实现完成后新增结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-b-model-result.md
```

结果文档必须记录：

```text
实际新增/修改文件
是否改动 model/cmdm.py
是否初始化 rot2xyz
参数量
py_compile 结果
随机输入 smoke 结果
dataset batch smoke 结果
shape guard 负例结果
是否满足进入下一阶段条件
```

## 阶段 B 退出条件

全部满足才允许进入下一阶段：

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

下一阶段应是：

```text
train/train_label_forecasting_diffusion.py
2-step diffusion train smoke
checkpoint save/resume gate
```

不是直接进入 label swap 或正式训练。

## 风险控制

### import 副作用

`model/cmdm.py` 顶层导入 `clip`。当前环境如果缺少 `clip`，直接 import 可能失败。优先实测；若失败，不应安装新依赖作为 workaround，而应抽取无 CLIP 依赖的公共组件到 `model/cmdm_components.py`，并让 `model/cmdm.py` 和 `model/forecasting_cmdm.py` 共享该文件。

### SMPL-X 资产

`Rotation2xyz_x` 可能依赖本地 SMPL-X 模型资产。阶段 B 默认 `init_rot2xyz=False`，防止模型 gate 被评估资产阻塞。

### timestep 类型

`TimestepEmbedder` 使用 positional encoding index。阶段 B 只接受整数 timestep。后续 diffusion 若开启 `rescale_timesteps=True`，必须先改 timestep embedding 设计。

### label 条件弱化

阶段 B 只能保证 action 进入模型结构，不能证明语义有效。语义有效性必须等后续 label swap 和动作一致性分类器 gate。
