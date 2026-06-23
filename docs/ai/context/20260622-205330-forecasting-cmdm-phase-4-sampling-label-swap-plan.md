# ForecastingCMDMDecoder 阶段 4 Sampling / Label Swap 实施 Plan

## 参考上下文

本 plan 依据：

```text
docs/ai/context/20260622-155322-forecasting-cmdm-phase-4-sampling-label-swap-design.md
docs/ai/context/20260622-124111-forecasting-cmdm-phase-c-diffusion-train-result.md
docs/ai/context/20260622-121949-forecasting-cmdm-phase-b-model-result.md
docs/ai/context/20260622-114122-forecasting-cmdm-phase-a-data-gate-result.md
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
```

## 阶段 4 定义

本文中的阶段 4 是生成闭环 gate：

```text
新增 sample/sample_label_forecasting_diffusion.py
加载阶段 C checkpoint
从 test H5 取 obs20
对同一 obs20 做 labels [2,5,8,17] label swap
保存 generated_future40.npy / metadata.json / metrics.json / label_swap_summary.json
验证 shape / finite / 不同 label 输出不完全相同
```

它不是：

```text
动作一致性分类器
正式 5000-step 训练
per-class 完整评估
视频可视化
论文结果整理
```

## 固定输入输出

输入：

```text
checkpoint = save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt
data_path = dataset/ntu120/smplx/conditioned/xsub.test.h5
window_len = 60
obs_len = 20
pred_len = 40
labels = 2 5 8 17
guidance_scale = 1.0
num_cases = 1
num_repetitions = 1
sample_index = 0
```

输出目录：

```text
results/forecasting/ntu120_label/p4_label_swap_smoke
```

输出文件：

```text
generated_future40.npy
obs_motion.npy
real_future40.npy
metadata.json
metrics.json
label_swap_summary.json
```

核心 shape：

```text
generated_future40.npy = [num_cases,num_labels,num_repetitions,56,6,40]
obs_motion.npy = [num_cases,56,6,20]
real_future40.npy = [num_cases,56,6,40]
```

## 修改范围

本阶段新增：

```text
sample/sample_label_forecasting_diffusion.py
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-4-sampling-label-swap-result.md
```

本阶段更新：

```text
AGENTS.md
```

原则上不修改：

```text
train/train_label_forecasting_diffusion.py
train/train_mdm.py
train/training_loop.py
model/forecasting_cmdm.py
model/cmdm.py
data_loaders/forecasting/ntu_label.py
diffusion/gaussian_diffusion.py
eval/*
```

本阶段暂不抽：

```text
utils/forecasting_cmdm.py
```

原因：

```text
当前只有采样脚本需要 checkpoint/model/diffusion 重建逻辑。
等后续 eval 也复用时再抽公共 util，避免过早固化 API。
```

## 实施 Checklist

### 1. 新建采样入口骨架

文件：

```text
sample/sample_label_forecasting_diffusion.py
```

导入范围：

```python
import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch

from data_loaders.forecasting.ntu_label import NTULabelForecastDataset
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.forecasting_cmdm import (
    ForecastingCMDMDecoder,
    ForecastingClassifierFreeSampleModel,
)
from utils.fixseed import fixseed
```

不导入：

```text
utils.model_util.load_model_wo_clip
utils.model_util.create_model_and_diffusion
train.train_label_forecasting_diffusion
Rotation2xyz_x
ClassifierFreeSampleModel 旧 wrapper
```

原因：

```text
阶段 C checkpoint 是 dict，不是旧 MDM/CMDM raw state_dict。
阶段 4 不做 xyz / MPJPE，不需要 SMPL-X 资产。
```

### 2. 实现基础工具函数

实现：

```text
_utc_now()
_device()
_write_json(path, value)
_prepare_save_dir(args)
_clear_stage_outputs(save_dir)
_ensure_finite(name, array_or_tensor)
_to_numpy(tensor)
```

`_prepare_save_dir` 规则：

```text
save_dir 不允许为空
目录不存在则创建
目录存在且非空，未传 --overwrite 时失败
--overwrite 只删除本脚本固定输出文件
```

固定输出文件：

```text
generated_future40.npy
obs_motion.npy
real_future40.npy
metadata.json
metrics.json
label_swap_summary.json
```

不使用 `rm -rf`。

### 3. 实现 checkpoint 加载

实现：

```text
_load_checkpoint(path, device)
_validate_checkpoint_state(state)
_build_model_from_checkpoint(state, device)
```

校验：

```text
state["model_type"] == "forecasting_cmdm_decoder"
state["train_protocol"]["mean_type"] == "START_X"
state["train_protocol"]["loss_type"] == "MSE"
state["model_config"] 存在
state["model_state_dict"] 存在
```

模型构建：

```python
model = ForecastingCMDMDecoder(**model_config)
model.load_state_dict(state["model_state_dict"], strict=True)
model.to(device)
model.eval()
```

如果 checkpoint 的 `model_config["init_rot2xyz"]` 是 `True`，采样脚本应强制覆盖为 `False` 后再构建，原因：

```text
阶段 4 不做 xyz 评估，不能让 SMPL-X 资产加载阻塞生成闭环。
```

### 4. 实现 diffusion 重建

实现：

```text
_sampling_diffusion_config(checkpoint_config, args)
_build_diffusion(config)
```

从 checkpoint 读取：

```text
noise_schedule
sigma_small
steps
data_rep
num_person
body_model
```

CLI 可覆盖：

```text
timestep_respacing
```

固定：

```text
model_mean_type = START_X
loss_type = MSE
rescale_timesteps = False
```

`_build_diffusion` 使用：

```text
gd.get_named_beta_schedule
space_timesteps
SpacedDiffusion
```

不调用：

```text
diffusion.training_losses
```

### 5. 实现 CLI 与参数校验

实现：

```text
build_arg_parser()
_validate_args(args, state)
main()
```

参数：

```text
--checkpoint
--data_path
--save_dir
--window_len 60
--obs_len 20
--pred_len 40
--labels 2 5 8 17
--guidance_scale 1.0
--batch_size 4
--num_cases 1
--num_repetitions 1
--sample_index 0
--seed 0
--use_ddim
--timestep_respacing
--progress
--overwrite
```

校验：

```text
checkpoint 存在
data_path 存在
labels 非空
labels 全部在 [0,25]
num_cases >= 1
num_repetitions >= 1
batch_size >= 1
guidance_scale >= 0
CLI window_len/obs_len/pred_len 与 checkpoint model_config 一致
```

如果不一致，直接报错，不 silent override。

### 6. 实现 test case 读取

实现：

```text
_build_dataset(args)
_select_cases(dataset, args)
```

Dataset：

```text
NTULabelForecastDataset(
    h5_path=args.data_path,
    split="test",
    window_len=args.window_len,
    obs_len=args.obs_len,
    pred_len=args.pred_len,
    max_samples=-1,
    seed=args.seed,
    strict=True,
)
```

选择规则：

```text
indices = range(sample_index, sample_index + num_cases)
越界直接报错
```

每个 case 保存：

```text
obs_motion
future
action
mask
sample_id
start
length
action_code
```

### 7. 实现 label swap batch 构建

实现：

```text
_build_label_swap_batch(cases, labels, num_repetitions, device)
```

输出：

```text
y
noise
batch_meta
```

shape：

```text
y["obs_motion"] = [N,56,6,20]
y["action"] = [N,1]
y["mask"] = [N,1,1,40]
noise = [N,56,6,40]
N = num_cases * num_labels * num_repetitions
```

共享 noise 规则：

```text
同一 case + repetition 下，不同 label 使用同一份 noise。
```

构造顺序固定为：

```text
case major -> label -> repetition
```

这样输出 reshape 为：

```text
[num_cases,num_labels,num_repetitions,56,6,40]
```

### 8. 实现分块采样

实现：

```text
_sample_in_batches(model, diffusion, y, noise, args, device)
```

流程：

```text
按 args.batch_size 切分 N
每块切 y 和 noise
sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
sample_fn(..., clip_denoised=False, model_kwargs={"y": y_chunk}, noise=noise_chunk)
拼接所有 chunk
```

CFG 规则：

```text
if guidance_scale == 1.0:
    sample_model = model
else:
    sample_model = ForecastingClassifierFreeSampleModel(model, guidance_scale)
    y_chunk["scale"] = torch.ones(chunk_size) * guidance_scale
```

`model.eval()` 必须保持。

禁止：

```text
gaussian_filter1d
clip_denoised=True
post smoothing
```

### 9. 实现基础 metrics

实现：

```text
_compute_rot_mse(generated, real_future)
_compute_root_translation_mse(generated, real_future)
_compute_label_swap_summary(generated, labels)
_write_outputs(...)
```

`generated` shape：

```text
[C,L,R,56,6,40]
```

`real_future` shape：

```text
[C,56,6,40]
```

rot MSE：

```text
((generated - real_future[:,None,None]) ** 2).mean(axis=(3,4,5))
```

输出：

```text
rot_mse_by_case_label_rep = [C,L,R]
rot_mse_by_label = [L]
rot_mse_mean = scalar
```

root translation MSE：

```text
最后一个 slot 的 0:3 维
```

label swap summary：

```text
对每个 case/repetition 的 label pair 计算 mean_abs_diff 和 max_abs_diff
pass_non_identical_check = 任意 max_abs_diff > 1e-7
```

metrics 必须标记：

```text
smoke_only = true
```

### 10. 实现输出写入

写入：

```text
generated_future40.npy
obs_motion.npy
real_future40.npy
metadata.json
metrics.json
label_swap_summary.json
```

JSON 使用：

```text
indent=2
sort_keys=True
ensure_ascii=False
```

metadata 必须包含：

```text
checkpoint
checkpoint_step
data_path
save_dir
seed
device
window_len
obs_len
pred_len
labels
label_action_codes
guidance_scale
use_ddim
timestep_respacing
num_cases
num_repetitions
generated_shape
source_meta
model_config
diffusion_config
sampling_diffusion_config
created_at
```

`label_action_codes`：

```text
label 2 -> A003
```

## 验证顺序

### 1. 静态编译

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile sample/sample_label_forecasting_diffusion.py
```

通过标准：

```text
exit_code = 0
```

### 2. Import smoke

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
from sample.sample_label_forecasting_diffusion import build_arg_parser, run_sampling
print("sample entry import ok")
PY
```

通过标准：

```text
sample entry import ok
exit_code = 0
```

### 3. p_sample_loop smoke

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m sample.sample_label_forecasting_diffusion \
  --checkpoint save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/p4_label_swap_smoke \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --labels 2 5 8 17 \
  --guidance_scale 1.0 \
  --batch_size 4 --num_cases 1 --num_repetitions 1 \
  --sample_index 0 --seed 0 --overwrite
```

通过标准：

```text
exit_code = 0
generated shape = [1,4,1,56,6,40]
输出 finite
pass_non_identical_check = true
```

### 4. 输出文件检查

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
import json
import numpy as np
base = "results/forecasting/ntu120_label/p4_label_swap_smoke"
generated = np.load(base + "/generated_future40.npy")
obs = np.load(base + "/obs_motion.npy")
future = np.load(base + "/real_future40.npy")
assert generated.shape == (1, 4, 1, 56, 6, 40)
assert obs.shape == (1, 56, 6, 20)
assert future.shape == (1, 56, 6, 40)
assert np.isfinite(generated).all()
with open(base + "/metadata.json") as f:
    metadata = json.load(f)
with open(base + "/metrics.json") as f:
    metrics = json.load(f)
with open(base + "/label_swap_summary.json") as f:
    summary = json.load(f)
assert metadata["labels"] == [2, 5, 8, 17]
assert metrics["finite"] is True
assert summary["pass_non_identical_check"] is True
print("phase4 outputs ok")
PY
```

通过标准：

```text
phase4 outputs ok
exit_code = 0
```

### 5. DDIM smoke 可选

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m sample.sample_label_forecasting_diffusion \
  --checkpoint save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/p4_label_swap_smoke_ddim50 \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --labels 2 5 8 17 \
  --guidance_scale 1.0 \
  --batch_size 4 --num_cases 1 --num_repetitions 1 \
  --sample_index 0 --seed 0 \
  --use_ddim --timestep_respacing ddim50 --overwrite
```

DDIM 不是阶段 4 必须 gate，但如果实现了 `--use_ddim`，应至少跑一次 smoke。

## 阶段 4 退出条件

必须满足：

```text
sample/sample_label_forecasting_diffusion.py 存在
py_compile 通过
import smoke 通过
p_sample_loop smoke 通过
输出文件检查通过
generated_future40.npy 存在且 shape = [1,4,1,56,6,40]
obs_motion.npy 存在且 shape = [1,56,6,20]
real_future40.npy 存在且 shape = [1,56,6,40]
metadata.json / metrics.json / label_swap_summary.json 可读
generated finite
不同 label 输出不完全相同
metrics.json smoke_only = true
未修改 train/train_mdm.py
未修改 train/train_label_forecasting_diffusion.py 的训练语义
未修改 model/cmdm.py
未修改 diffusion/gaussian_diffusion.py
新增阶段 4 result 文档
AGENTS.md 更新到阶段 4 结果
```

## 失败处理

### 1. checkpoint load 失败

处理：

```text
确认 checkpoint 是阶段 C dict checkpoint，不是旧 MDM raw state_dict。
确认 model_type / train_protocol 字段存在。
strict=True 不通过时先停止，不用 strict=False 绕过。
```

### 2. p_sample_loop 过慢

处理：

```text
先确认 batch_size=4、num_cases=1、num_repetitions=1。
默认 p_sample_loop 仍是必须 gate。
额外跑 DDIM 只能作为补充，不能替代默认 gate，除非后续重新写设计说明。
```

### 3. 输出 NaN/Inf

处理：

```text
检查 initial noise finite。
检查 obs_motion/action/mask finite。
检查模型输出 finite。
检查 checkpoint 是否来自通过阶段 C gate 的模型。
不要通过 clipping 或 smoothing 掩盖 NaN。
```

### 4. 不同 label 输出完全相同

处理：

```text
打印 batch action 序列，确认是 [2,5,8,17]。
确认 guidance_scale=1.0 时没有设置 y["uncond"]。
确认 shared noise 只跨 label 共享，没有把 generated 复制错。
必要时用阶段 B 的 action token smoke 重新验证。
```

### 5. metrics 被误用

处理：

```text
metrics.json 固定写 smoke_only = true。
result 文档只写生成闭环通过，不写模型质量改善。
```

## 阶段 4 结果文档模板

实现和 smoke 后新增：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-4-sampling-label-swap-result.md
```

至少记录：

```text
实现文件
未修改文件
运行环境
py_compile 结果
import smoke 结果
p_sample_loop smoke 命令与结果
输出文件列表
generated / obs / future shape
finite 检查
label_swap_summary
metrics 摘要
是否满足退出条件
下一阶段是否允许进入动作一致性分类器 gate
```

## 下一阶段边界

阶段 4 通过后，下一步才是：

```text
eval/action_consistency_classifier.py
真实 future40 分类器 gate
generated future40 label consistency
per-class / handshaking subset 指标
```

仍不直接写论文结论，也不直接启动正式 5000-step 训练。正式训练需要在生成闭环和评价 gate 都稳定后再进入。
