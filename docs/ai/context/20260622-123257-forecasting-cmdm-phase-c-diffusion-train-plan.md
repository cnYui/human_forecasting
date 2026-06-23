# ForecastingCMDMDecoder 阶段 C Diffusion 训练实施 Plan

## 参考上下文

本 plan 依据：

```text
docs/ai/context/20260622-122332-forecasting-cmdm-phase-c-diffusion-train-design.md
docs/ai/context/20260622-121949-forecasting-cmdm-phase-b-model-result.md
docs/ai/context/20260622-114122-forecasting-cmdm-phase-a-data-gate-result.md
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
```

本阶段实现前提：

```text
阶段 A 数据 gate 已通过
阶段 B 模型 gate 已通过
当前模型文件 = model/forecasting_cmdm.py
当前数据文件 = data_loaders/forecasting/ntu_label.py
```

## 阶段 C 定义

本文中的阶段 C 是训练 gate：

```text
新增 train/train_label_forecasting_diffusion.py
完成 START_X diffusion 训练闭环
完成 2-step smoke
完成 checkpoint save/resume
```

它不是生成闭环，也不是标签语义评估。以下内容不进入本阶段：

```text
sample/sample_label_forecasting_diffusion.py
label swap
eval/eval_label_forecasting_diffusion.py
eval/action_consistency_classifier.py
正式 5000-step 训练
```

## 固定协议

数据协议：

```text
dataset = ntu120_2p
train_path = dataset/ntu120/smplx/conditioned/xsub.train.h5
test_path = dataset/ntu120/smplx/conditioned/xsub.test.h5
window_len = 60
obs_len = 20
pred_len = 40
num_actions = 26
```

batch contract：

```text
future      = [B,56,6,40]
obs_motion  = [B,56,6,20]
action      = [B,1]
mask        = [B,1,1,40]
```

diffusion 协议：

```text
x_start = future
x_t = q_sample(x_start, t, noise)
model_output = pred_xstart
model_mean_type = START_X
loss_type = MSE
rescale_timesteps = False
```

训练条件：

```python
y = {
    "obs_motion": batch["obs_motion"],
    "action": batch["action"],
    "mask": batch["mask"],
}
```

CFG 训练：

```text
cond_mask_prob 默认 0.1
训练 step 必须保持 model.train()
uncond 只 mask action，不清零 obs_motion
```

## 修改范围

本阶段新增：

```text
train/train_label_forecasting_diffusion.py
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-c-diffusion-train-result.md
```

本阶段更新：

```text
AGENTS.md
```

原则上不修改：

```text
train/train_mdm.py
train/training_loop.py
model/cmdm.py
model/forecasting_cmdm.py
data_loaders/forecasting/ntu_label.py
diffusion/gaussian_diffusion.py
eval/*
sample/*
```

如果实现中发现阶段 B 模型或阶段 A 数据 contract 有 bug，先停止并新增 context 说明，不在阶段 C 里顺手改协议。

## 实施 Checklist

### 1. 新建训练入口骨架

文件：

```text
train/train_label_forecasting_diffusion.py
```

导入范围：

```python
import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_label import (
    NTULabelForecastDataset,
    ntu_label_forecasting_collate,
)
from diffusion import gaussian_diffusion as gd
from diffusion.resample import create_named_schedule_sampler
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.forecasting_cmdm import ForecastingCMDMDecoder, count_parameters
from utils.fixseed import fixseed
```

不导入：

```text
TrainLoop
create_model_and_diffusion
train_mdm
ClassifierFreeSampleModel
Rotation2xyz_x
```

原因：

```text
本阶段不复用旧 CMDM 训练协议，不做采样和 xyz 评估。
```

### 2. 实现基础工具函数

实现：

```text
_utc_now()
_device()
_write_json(path, value)
_append_train_log(args, record)
_prepare_save_dir(args)
```

`_prepare_save_dir` 规则：

```text
save_dir 不允许为空
目录不存在则创建
目录存在且非空，且未设置 --overwrite / --resume_checkpoint 时失败
--overwrite 允许清理本阶段 smoke 输出前的写入风险，但不删除目录外文件
```

实现时不要使用 `rm -rf`。如果 `--overwrite` 需要清理，只删除本脚本会写的固定文件：

```text
args.json
train_log.jsonl
model*.pt
opt*.pt
```

### 3. 实现 Dataset / DataLoader 构建

实现：

```text
_build_dataset(args, path, split)
_build_loader(args, path, split, shuffle, batch_size)
```

Dataset 参数：

```text
h5_path = path
split = train/test
window_len = args.window_len
obs_len = args.obs_len
pred_len = args.pred_len
max_samples = args.max_samples
seed = args.seed
strict = True
```

Loader 参数：

```text
batch_size = args.batch_size
shuffle = True for train
num_workers = args.num_workers
collate_fn = ntu_label_forecasting_collate
drop_last = False
```

阶段 C 不实现类别均衡采样。原因：

```text
本阶段只验证训练闭环；采样分布调整会影响后续结果解释，应单独设计。
```

### 4. 实现模型构建

实现：

```text
_build_model(args)
```

固定：

```text
model_type = forecasting_cmdm_decoder
njoints = 56
nfeats = 6
num_actions = 26
data_rep = rot6d
body_model = smplx
dataset = ntu120_2p
init_rot2xyz = False
```

从 CLI 读取：

```text
latent_dim
obs_encoder_layers
decoder_layers
num_heads
ff_size
dropout
cond_mask_prob
```

构建后记录：

```text
args.num_params = count_parameters(model)
args.model_config = model.config()
```

验收：

```text
model.config()["obs_len"] == args.obs_len
model.config()["pred_len"] == args.pred_len
model.config()["window_len"] == args.window_len
```

### 5. 实现 diffusion 构建

实现：

```text
_build_diffusion(args)
```

本阶段不直接调用 `utils.model_util.create_gaussian_diffusion(args)`。原因：

```text
create_gaussian_diffusion 的 args 集合来自旧 CMDM 入口；
阶段 C 只需要等价 diffusion 配置，不需要继承旧训练入口的参数耦合。
```

配置：

```text
steps = 1000
scale_beta = 1.0
noise_schedule = args.noise_schedule
timestep_respacing = args.timestep_respacing or [1000]
model_mean_type = START_X
model_var_type = FIXED_SMALL if args.sigma_small else FIXED_LARGE
loss_type = MSE
rescale_timesteps = False
data_rep = rot6d
num_person = 2
body_model = smplx
lambda_rcxyz = 0.0
lambda_vel = 0.0
lambda_fc = 0.0
lambda_orient = 0.0
lambda_body = 0.0
lambda_transl = 0.0
```

保存 diffusion_config：

```text
noise_schedule
timestep_respacing
sigma_small
steps
model_mean_type = START_X
loss_type = MSE
rescale_timesteps = False
```

### 6. 实现 loss helpers

实现：

```text
_masked_l2(a, b, mask)
_velocity_mse(pred, target, mask)
_root_translation_mse(pred, target, mask)
_relative_root_mse(pred, target, mask)
_compute_losses(pred, target, mask, args)
```

`_masked_l2` 返回 `[B]`：

```text
sum((a-b)^2 * mask) / valid_element_count
```

`_compute_losses` 第一版默认：

```text
terms["rot_mse"]
terms["loss"] = terms["rot_mse"]
```

可选权重：

```text
velocity_loss_weight
root_translation_loss_weight
relative_root_loss_weight
```

默认都为 0。

`relative_root_mse` 实现策略：

```text
如果 root slot 索引未确认，不实现该项，权重大于 0 时直接抛 NotImplementedError。
```

不要猜测双人 root slot。错误索引会比没有该项更危险。

### 7. 实现 train step

实现：

```text
_diffusion_train_step(model, diffusion, schedule_sampler, batch, args, device)
```

逻辑：

```python
future = batch["future"].to(device)
y = {
    "obs_motion": batch["obs_motion"].to(device),
    "action": batch["action"].to(device),
    "mask": batch["mask"].to(device),
}
t, weights = schedule_sampler.sample(future.shape[0], device)
noise = torch.randn_like(future)
x_t = diffusion.q_sample(future, t, noise=noise)
pred = model(x_t, t, y)
terms = _compute_losses(pred, future, y["mask"], args)
loss = (terms["loss"] * weights).mean()
```

返回：

```text
loss: scalar tensor
metrics: dict[str, float]
```

finite gate：

```text
future finite
obs_motion finite
pred finite
loss finite
```

失败时抛 `ValueError`，错误信息包含字段名。

### 8. 实现 checkpoint save/load

实现：

```text
_checkpoint_paths(args, step)
_save_checkpoint(args, model, optimizer, step)
_load_resume(args, model, optimizer, device)
_validate_resume_config(args, checkpoint_state)
```

保存文件：

```text
model000000002.pt
opt000000002.pt
```

模型 checkpoint 内容：

```text
model_state_dict
model_type
model_config
num_params
step
seed
diffusion_config
train_protocol
created_at
```

optimizer checkpoint 内容：

```text
optimizer_state_dict
step
```

resume 规则：

```text
检查 checkpoint 存在
检查 model_type 一致
检查核心 shape config 一致
加载 model_state_dict
如果同目录 opt*.pt 存在则加载 optimizer
返回 resume_step
```

核心 shape config：

```text
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
```

### 9. 实现训练主循环

实现：

```text
run_training(args)
```

流程：

```text
校验 dataset == ntu120_2p
校验 obs_len + pred_len == window_len
fixseed(args.seed)
准备 save_dir
构建 device
构建 train_loader
构建 model / diffusion / schedule_sampler
构建 AdamW
resume
while step < num_steps:
    遍历 train_loader
    grad_accum_steps
    clip_grad_norm
    optimizer.step()
    记录 train_log.jsonl
    按 save_interval 保存
结束时如果没有保存最终 step，则补存
```

日志记录：

```text
step
train_loss
rot_mse
velocity_mse 可选
root_translation_mse 可选
relative_root_mse 可选
lr
effective_batch_size
model_num_params
seed
checkpoint 可选
optimizer 可选
created_at
```

本阶段不运行 eval。`eval_interval` 保留但默认 0，若传入非 0 先只记录 warning 或直接报错。推荐直接要求：

```text
eval_interval == 0
```

原因：

```text
阶段 C 的 gate 是训练与 checkpoint，不是指标评估。
```

### 10. 实现 CLI

实现：

```text
build_arg_parser()
main()
```

必要参数：

```text
--dataset ntu120_2p
--data_path
--eval_data_path
--save_dir
--model_type forecasting_cmdm_decoder
--body_model smplx
--window_len 60
--obs_len 20
--pred_len 40
--batch_size
--eval_batch_size
--num_steps
--save_interval
--eval_interval 0
--latent_dim
--decoder_layers
--obs_encoder_layers
--num_heads
--ff_size
--dropout
--cond_mask_prob
--lr
--weight_decay
--grad_accum_steps
--clip_grad_norm
--max_samples
--num_workers
--seed
--resume_checkpoint
--overwrite
--noise_schedule cosine
--timestep_respacing ""
--sigma_small
--velocity_loss_weight 0.0
--root_translation_loss_weight 0.0
--relative_root_loss_weight 0.0
```

默认 smoke 友好：

```text
batch_size = 1
num_steps = 2
latent_dim = 128
decoder_layers = 2
obs_encoder_layers = 1
num_heads = 4
ff_size = 512
cond_mask_prob = 0.1
lr = 1e-4
```

## 验证顺序

### 1. 静态编译

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile train/train_label_forecasting_diffusion.py
```

通过标准：

```text
exit_code = 0
```

### 2. 入口 import

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
from train.train_label_forecasting_diffusion import build_arg_parser, run_training
print("train entry import ok")
PY
```

通过标准：

```text
train entry import ok
exit_code = 0
```

### 3. 2-step smoke

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m train.train_label_forecasting_diffusion \
  --dataset ntu120_2p \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke \
  --model_type forecasting_cmdm_decoder \
  --body_model smplx \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 1 --eval_batch_size 1 \
  --num_steps 2 --save_interval 2 --eval_interval 0 \
  --latent_dim 128 --decoder_layers 2 --obs_encoder_layers 1 \
  --num_heads 4 --ff_size 512 --cond_mask_prob 0.1 \
  --lr 1e-4 --weight_decay 1e-4 \
  --num_workers 0 --seed 0 --overwrite
```

通过标准：

```text
2 step 完成
loss finite
rot_mse finite
args.json 存在
train_log.jsonl 存在
model000000002.pt 存在
opt000000002.pt 存在
```

### 4. Checkpoint 内容检查

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
import torch
p = "save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt"
state = torch.load(p, map_location="cpu")
assert state["model_type"] == "forecasting_cmdm_decoder"
assert state["step"] == 2
assert state["model_config"]["obs_len"] == 20
assert state["model_config"]["pred_len"] == 40
assert state["train_protocol"]["mean_type"] == "START_X"
print("checkpoint ok")
PY
```

通过标准：

```text
checkpoint ok
exit_code = 0
```

### 5. Resume smoke

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m train.train_label_forecasting_diffusion \
  --dataset ntu120_2p \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke \
  --model_type forecasting_cmdm_decoder \
  --body_model smplx \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 1 --eval_batch_size 1 \
  --num_steps 3 --save_interval 1 --eval_interval 0 \
  --latent_dim 128 --decoder_layers 2 --obs_encoder_layers 1 \
  --num_heads 4 --ff_size 512 --cond_mask_prob 0.1 \
  --lr 1e-4 --weight_decay 1e-4 \
  --num_workers 0 --seed 0 \
  --resume_checkpoint save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt
```

通过标准：

```text
从 step=2 继续
step=3 完成
model000000003.pt 存在
opt000000003.pt 存在
train_log.jsonl 追加 step=3
```

## 阶段 C 退出条件

必须满足：

```text
新增 train/train_label_forecasting_diffusion.py
py_compile 通过
import smoke 通过
2-step smoke 通过
resume smoke 通过
loss / rot_mse finite
checkpoint 内容检查通过
args.json 记录完整 CLI 和协议
train_log.jsonl 可逐行 JSON 解析
未修改 train/train_mdm.py
未修改 train/training_loop.py
未修改 model/cmdm.py
未修改 diffusion/gaussian_diffusion.py
未引入 Encoder-only 主线
新增阶段 C result 文档
AGENTS.md 指向 result 文档或保留当前阶段状态
```

## 失败处理

### 1. loss NaN/Inf

处理顺序：

```text
检查 future / obs_motion finite
检查 mask sum 非 0
检查 pred finite
检查 q_sample 输出 finite
降低 lr 不作为第一反应
```

如果输入或模型输出已经非 finite，先修数据或模型 gate，不继续训练。

### 2. OOM

允许降级：

```text
batch_size = 1
latent_dim = 96
decoder_layers = 1
obs_encoder_layers = 1
ff_size = 384
```

不允许降级：

```text
不能换成 Encoder-only
不能取消 noisy future target
不能把 obs_motion 拼进 future 当输入
```

### 3. resume config mismatch

处理：

```text
直接报错
不要 silent load
不要 strict=False 绕过 shape mismatch
```

### 4. eval_interval 非 0

阶段 C 推荐直接报错：

```text
ValueError("阶段 C 不支持 eval_interval > 0")
```

原因：

```text
评估闭环属于后续阶段，不能在训练 gate 里混入未设计完成的指标。
```

## 阶段 C 结果文档模板

实现和 smoke 完成后新增：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-c-diffusion-train-result.md
```

至少记录：

```text
实现文件
未修改文件
运行环境
py_compile 结果
2-step smoke 命令和结果
resume smoke 命令和结果
checkpoint 文件列表
train_log 摘要
是否满足退出条件
下一阶段是否允许进入 sampling / label swap
```

## 下一阶段边界

阶段 C 通过后，下一阶段才开始：

```text
sample/sample_label_forecasting_diffusion.py
p_sample_loop / DDIM
same obs20 + labels [2,5,8,17] label swap
generated_future40.npy
metadata.json
metrics.json
label_swap_summary.json
```

仍然不直接进入动作一致性分类器。动作一致性需要先证明真实 future40 可分类，再用于生成一致性指标。
