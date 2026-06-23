# ForecastingCMDMDecoder 阶段 B 模型设计

## 参考文档

本设计依据以下上下文：

```text
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
docs/ai/context/20260622-114122-forecasting-cmdm-phase-a-data-gate-result.md
docs/ai/context/20260622-105946-forecasting-cmdm-final-target-architecture-v3.md
docs/ai/context/20260622-105749-forecasting-cmdm-final-architecture-plan.md
```

## 阶段命名说明

本文中的阶段 B 指 `20260622-114122-forecasting-cmdm-phase-a-data-gate-result.md` 里允许进入的下一步：

```text
新增 model/forecasting_cmdm.py
实现 ForecastingCMDMDecoder
实现 ForecastingClassifierFreeSampleModel
只做 forward / backward / CFG shape smoke
```

它对应总提交计划里的 Commit 2 / P1 模型 gate，不是总计划“阶段 B: 生成闭环”。采样、label swap、metrics 和 checkpoint load 必须等模型 gate 与后续 2-step train gate 通过后再进入。

## 当前已满足条件

阶段 A 数据 gate 已通过：

```text
train kept_count = 1956
test kept_count = 1253
train/test 均覆盖 26 类
handshaking label 8 train/test 非零
obs_motion = [B,56,6,20]
future = [B,56,6,40]
action = [B,1]
mask = [B,1,1,40]
obs_motion/future 全量 finite
```

阶段 B 可以直接依赖 batch contract：

```python
x_start = batch["future"]
y = {
    "obs_motion": batch["obs_motion"],
    "action": batch["action"],
    "mask": batch["mask"],
}
```

运行验证优先使用项目环境：

```text
python_executable = /home/rpartx3080/.local/micromamba/envs/regennet/bin/python
torch_version = 1.7.1
```

## 阶段 B 目标

新增正式模型文件：

```text
model/forecasting_cmdm.py
```

新增核心类：

```text
ForecastingCMDMDecoder
ForecastingClassifierFreeSampleModel
```

阶段 B 只证明模型接口可靠：

```text
随机 x_t + obs20 + action 可 forward
输出 shape 等于 noisy future40 输入 shape
loss.backward() finite
CFG cond/uncond forward finite
dataset batch 可 forward/backward
```

## 阶段 B 不做

本阶段不接入：

```text
train/train_label_forecasting_diffusion.py
diffusion loss
checkpoint save/resume
sample/sample_label_forecasting_diffusion.py
eval/eval_label_forecasting_diffusion.py
eval/action_consistency_classifier.py
label swap
formal metrics
```

本阶段不改：

```text
model/cmdm.py 的原始 CMDM.forward 语义
train/train_mdm.py
data_loaders/forecasting/ntu_label.py 的 batch contract
```

不能把阶段 B 降级为 Encoder-only。Smoke 也必须使用最终 Decoder 架构，只缩小 `latent_dim`、层数和 batch。

## 文件与 import 策略

模型文件放在：

```text
model/forecasting_cmdm.py
```

优先直接复用 `model/cmdm.py` 组件：

```python
from model.cmdm import (
    InputProcess,
    OutputProcess,
    PositionalEncoding,
    TimestepEmbedder,
    EmbedAction,
)
```

当前检查结果：

```text
InputProcess: [B,J,F,T] -> [T,B,D]
OutputProcess: [T,B,D] -> [B,J,F,T]
TimestepEmbedder: timesteps [B] -> [1,B,D]
EmbedAction: action [B,1] -> [B,D]
PositionalEncoding.pe: [max_len,1,D]
```

只有在直接 import 产生明确循环依赖或副作用时，才无行为改动抽公共组件到：

```text
model/cmdm_components.py
```

阶段 B 默认不做该抽取。

## 模型接口

构造参数第一版固定为显式参数，避免训练入口后续依赖隐式全局配置：

```text
model_type = "forecasting_cmdm_decoder"
njoints = 56
nfeats = 6
num_actions = 26
obs_len = 20
pred_len = 40
window_len = 60
data_rep = "rot6d"
body_model = "smplx"
dataset = "ntu120_2p"
latent_dim = 128 smoke / 256 formal
obs_encoder_layers = 1 smoke / 2 formal
decoder_layers = 2 smoke / 4 formal
num_heads = 4
ff_size = 512 smoke / 1024 formal
dropout = 0.1
activation = "gelu"
cond_mask_prob = 0.1
```

兼容属性：

```text
self.model_type
self.njoints
self.nfeats
self.num_actions
self.obs_len
self.pred_len
self.window_len
self.data_rep
self.cond_mode = "action"
self.cond_mask_prob
self.translation = True
self.pose_rep = "rot6d"
self.glob = True
self.glob_rot = None
self.body_model
self.dataset
```

`rot2xyz` 是评估需要的属性，但阶段 B forward gate 不应被 SMPL-X 资产加载阻塞。建议实现为可选初始化：

```text
init_rot2xyz = False by default for smoke
init_rot2xyz = True when evaluation/training code needs xyz conversion
self.rot2xyz = None if disabled
```

如果后续训练循环硬依赖 `model.rot2xyz`，再在训练入口显式打开，不能让纯模型 smoke 因评估组件失败。

## 架构设计

最终模型是 decoder denoiser：

```text
noisy future40 作为 TransformerDecoder target
obs20 token memory + timestep/action/obs_summary token 作为 memory
输出 denoised future40
```

模块清单：

```text
self.obs_input_process = InputProcess("rot6d", 336, D)
self.future_input_process = InputProcess("rot6d", 336, D)
self.output_process = OutputProcess("rot6d", 336, D, 56, 6)
self.sequence_pos_encoder = PositionalEncoding(D, dropout)
self.embed_timestep = TimestepEmbedder(D, self.sequence_pos_encoder)
self.embed_action = EmbedAction(26, D)
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
```

`TransformerEncoderLayer` 和 `TransformerDecoderLayer` 使用 PyTorch 1.7 兼容写法，不使用 `batch_first=True` 或新版 `norm_first` 参数。

## 位置编码

必须保留 60 帧窗口内的绝对时间关系：

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

不能让 obs 和 future 都从位置 0 开始编码，否则模型不知道 future 是 obs 后面的时间段。

## Forward 协议

输入：

```text
x_t: [B,56,6,40]
timesteps: [B]
y["obs_motion"]: [B,56,6,20]
y["action"]: [B,1]
y["mask"]: [B,1,1,40], optional in model forward
y.get("uncond"): bool, CFG 采样时使用
```

输出：

```text
pred_xstart: [B,56,6,40]
```

Forward 伪代码：

```python
def forward(self, x_t, timesteps, y):
    self._check_future(x_t)
    obs = self._check_obs(y["obs_motion"], batch_size=x_t.shape[0])
    action = self._normalize_action(y["action"], batch_size=x_t.shape[0])
    force_uncond = bool(y.get("uncond", False))

    obs_tokens = self.obs_input_process(obs)
    obs_tokens = self.add_window_pos(obs_tokens, 0)
    obs_tokens = obs_tokens + self.obs_frame_type
    obs_tokens = self.obs_encoder(obs_tokens)

    obs_summary = obs_tokens.mean(dim=0, keepdim=True)
    obs_summary = obs_summary + self.obs_summary_type

    time_token = self.embed_timestep(timesteps) + self.time_type

    action_emb = self.embed_action(action)
    action_emb = self.mask_action(action_emb, force_mask=force_uncond)
    action_token = action_emb.unsqueeze(0) + self.action_type

    memory = torch.cat(
        [time_token, action_token, obs_summary, obs_tokens],
        dim=0,
    )
    memory = self.memory_proj(self.memory_norm(memory))

    future_tokens = self.future_input_process(x_t)
    future_tokens = self.add_window_pos(future_tokens, 20)
    future_tokens = self.future_norm(future_tokens + self.future_type)

    decoded = self.seqTransDecoder(
        tgt=future_tokens,
        memory=memory,
        tgt_mask=None,
        memory_mask=None,
    )
    return self.output_process(decoded)
```

关键点：

```text
action token 单独进入 memory，不能只加到 timestep 上
tgt_mask = None，因为 diffusion 是整段 future40 同时去噪，不是自回归
obs_motion 在 uncond 时仍保留
mask 当前不参与 Transformer attention，只作为后续 loss/eval 的有效帧合同
```

## Shape guard

阶段 B 实现必须显式报错，不依赖 PyTorch 内部 matmul 报错：

```text
x_t.dim() == 4
x_t.shape[1:] == (56,6,40)
obs_motion.dim() == 4
obs_motion.shape == (B,56,6,20)
action.shape == (B,1) 或 action.shape == (B,)
0 <= action < 26
timesteps.shape == (B,)
all finite: x_t / obs_motion
```

`timesteps` 当前按 CMDM `TimestepEmbedder` 作为 PE index 使用，因此 diffusion 必须保持 `rescale_timesteps=False`。如果后续需要浮点 rescaled timesteps，应另写连续 timestep embedding，不能直接把 float timestep 传入 `sequence_pos_encoder.pe[timesteps]`。

## CFG 设计

阶段 B 的 CFG wrapper 使用 forecasting 专用类：

```text
ForecastingClassifierFreeSampleModel
```

它与现有 `model/cfg_sampler.py` 思路一致，但语义固定为：

```text
uncond 只 mask action
uncond 不清零 obs_motion
scale 可从 wrapper 构造参数 guidance_scale 读取，也可优先读取 y["scale"]
```

推荐实现：

```python
def forward(self, x, timesteps, y=None):
    if y is None:
        raise ValueError("CFG 需要 y")
    y_cond = dict(y)
    y_uncond = dict(y)
    y_uncond["uncond"] = True

    out_cond = self.model(x, timesteps, y_cond)
    out_uncond = self.model(x, timesteps, y_uncond)
    scale = y.get("scale", self.guidance_scale)
    if not torch.is_tensor(scale):
        scale = torch.ones(x.shape[0], device=x.device, dtype=x.dtype) * float(scale)
    return out_uncond + scale.view(-1, 1, 1, 1) * (out_cond - out_uncond)
```

`ForecastingCMDMDecoder.mask_action`：

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

## 配置与参数统计

阶段 B 模型需要提供：

```text
config()
count_parameters(model)
parameters_wo_clip()
```

`parameters_wo_clip()` 直接返回 `self.parameters()`，因为本模型不加载 CLIP。

`config()` 至少返回：

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

## 最小测试

### 静态编译

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile model/forecasting_cmdm.py
```

通过标准：

```text
exit_code = 0
```

### 随机输入 forward/backward/CFG smoke

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
forward shape 正确
loss/backward finite
CFG cond/uncond forward finite
```

### Dataset batch -> model smoke

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
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=ntu_label_forecasting_collate)
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
输出 shape = [B,56,6,40]
无 NaN/Inf
backward 可执行
```

## 阶段 B 退出条件

全部满足后才进入 diffusion 2-step train：

```text
model/forecasting_cmdm.py 新增完成
py_compile 通过
随机输入 forward/backward/CFG smoke 通过
dataset batch forward/backward smoke 通过
CFG uncond 确认只 mask action，不 mask obs_motion
action token 确认单独进入 memory
obs/future 位置编码确认为 0..19 / 20..59
```

## 风险与 TODO

### `rot2xyz` 初始化风险

`Rotation2xyz_x` 会初始化 SMPL-X layer。阶段 B 不依赖 xyz 评估，因此默认延后初始化。后续 eval 或 xyz loss 需要时再显式开启。

### 类别不平衡

阶段 B 不处理 sampler。正式训练前需要在训练入口处理 class-balanced sampler 或 per-class repeat，并至少报告 per-class 与 handshaking subset。

### Label 被忽略

阶段 B 只能证明 action token 进入网络，不能证明语义有效。后续必须通过 label swap 和 action consistency classifier gate 检查。

### `mask` 未进入 attention

当前协议固定 pred_len=40 且不 padding，mask 暂不进入 decoder attention。若后续支持可变长度或 padding，必须把 mask 传入 loss 和 attention key padding mask。

## 下一步

阶段 B 设计完成后，下一次实现只应修改：

```text
model/forecasting_cmdm.py
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-b-model-result.md
```

如果实现中发现 `model/cmdm.py` import 有副作用或循环依赖，再新建单独设计/结果文档说明是否抽取 `model/cmdm_components.py`。
