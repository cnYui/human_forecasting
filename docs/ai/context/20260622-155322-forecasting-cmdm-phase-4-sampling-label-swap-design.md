# ForecastingCMDMDecoder 阶段 4 Sampling / Label Swap 设计

## 参考上下文

本文依据：

```text
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
docs/ai/context/20260622-121949-forecasting-cmdm-phase-b-model-result.md
docs/ai/context/20260622-124111-forecasting-cmdm-phase-c-diffusion-train-result.md
train/train_label_forecasting_diffusion.py
model/forecasting_cmdm.py
data_loaders/forecasting/ntu_label.py
diffusion/gaussian_diffusion.py
```

## 阶段命名

本文中的阶段 4 对应原提交计划的 Commit 4：

```text
采样、label swap 与基础预测评估
```

它是阶段 C diffusion train gate 之后的生成闭环，不是动作一致性分类器，也不是正式 5000-step 训练。

当前已满足：

```text
阶段 A 数据 gate 通过
阶段 B 模型 gate 通过
阶段 C diffusion train gate 通过
save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt 可加载
save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000003.pt 可加载
```

## 阶段 4 目标

新增采样入口：

```text
sample/sample_label_forecasting_diffusion.py
```

本阶段只证明生成闭环可靠：

```text
checkpoint load
ForecastingCMDMDecoder 重建
diffusion 重建
p_sample_loop / DDIM 入口可用
从 test H5 读取 obs20 / real future40 / action
same obs20 + labels [2,5,8,17] label swap
保存 generated_future40.npy
保存 metadata.json
保存 metrics.json
保存 label_swap_summary.json
输出 shape 正确
输出 finite
不同 label 输出不完全相同
```

本阶段通过后，才允许进入：

```text
动作一致性分类器 gate
更完整 per-class / handshaking subset 评估
正式 5000-step 训练
```

## 本阶段不做

不做以下内容：

```text
不训练新模型
不改 train/train_label_forecasting_diffusion.py 的训练协议
不改 train/train_mdm.py
不改 diffusion/gaussian_diffusion.py
不引入 action consistency classifier
不把分类器一致性当作生成质量证据
不默认初始化 SMPL-X xyz / MPJPE
不生成视频
不写论文结论
```

原因：

```text
阶段 4 的必要问题是 checkpoint 是否能生成可保存、可复查、可 label swap 的 future40。
语义一致性和 xyz 指标是更后面的评价 gate。
```

## 代码位置

新增：

```text
sample/sample_label_forecasting_diffusion.py
```

不建议放到：

```text
eval/eval_label_forecasting_diffusion.py
```

原因：

```text
阶段 4 的主产物是生成文件，不是最终评估报告。
基础 metrics 只是采样输出完整性的 gate。
```

本阶段暂不抽公共 util。采样脚本从 checkpoint 中读取 `model_config` 和 `diffusion_config` 作为事实来源，避免依赖训练入口私有函数。如果阶段 4 后 eval 脚本也需要同样逻辑，再抽到：

```text
utils/forecasting_cmdm.py
```

抽取时必须重新跑阶段 C 和阶段 4 smoke。

## 输入协议

必要输入：

```text
checkpoint = save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt
data_path = dataset/ntu120/smplx/conditioned/xsub.test.h5
window_len = 60
obs_len = 20
pred_len = 40
labels = 2 5 8 17
guidance_scale = 1.0
```

标签是 zero-based NTU120 A001-A026 标签：

```text
2  -> A003
5  -> A006
8  -> A009 handshaking
17 -> A018
```

数据读取使用：

```text
NTULabelForecastDataset(split="test")
center crop
```

对单个源样本：

```text
obs_motion = [56,6,20]
real_future = [56,6,40]
source_action = [1]
```

送入模型时扩展为：

```text
obs_motion = [N,56,6,20]
action = [N,1]
mask = [N,1,1,40]
shape = (N,56,6,40)
```

其中：

```text
N = num_cases * num_labels * num_repetitions
```

## Checkpoint Load 设计

Checkpoint 是阶段 C 保存的 dict：

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

加载步骤：

```text
torch.load(checkpoint, map_location=device)
检查 model_type == forecasting_cmdm_decoder
检查 train_protocol.mean_type == START_X
检查 train_protocol.loss_type == MSE
用 model_config 重建 ForecastingCMDMDecoder
load_state_dict(model_state_dict, strict=True)
model.eval()
```

不要使用：

```text
utils.model_util.load_model_wo_clip
```

原因：

```text
阶段 C checkpoint 不是旧 MDM/CMDM raw state_dict，也没有 CLIP 权重兼容问题。
```

## Diffusion 重建设计

使用 checkpoint 中的 `diffusion_config`：

```text
steps = 1000
noise_schedule = cosine
model_mean_type = START_X
loss_type = MSE
rescale_timesteps = False
data_rep = rot6d
num_person = 2
body_model = smplx
```

采样脚本支持 CLI 覆盖：

```text
--timestep_respacing
--use_ddim
```

规则：

```text
默认沿用 checkpoint diffusion_config.timestep_respacing
如果 smoke 需要更快，可显式传 --timestep_respacing ddim50 --use_ddim
无论是否覆盖，metadata.json 必须记录实际 sampling diffusion config
```

`rescale_timesteps` 必须保持：

```text
False
```

原因：

```text
ForecastingCMDMDecoder 明确要求整数 timestep index。
```

## CFG 设计

模型训练时 `cond_mask_prob = 0.1`，采样时默认：

```text
guidance_scale = 1.0
```

当：

```text
guidance_scale != 1.0
```

使用：

```text
ForecastingClassifierFreeSampleModel(model, guidance_scale)
y["scale"] = torch.ones(N) * guidance_scale
```

CFG 语义必须保持：

```text
uncond 只 mask action
obs_motion 始终保留
```

如果 `guidance_scale == 1.0`，不强制包 CFG wrapper。此时模型仍是 action-conditioned forward，且 `model.eval()` 会关闭训练时随机 action mask。

## Label Swap 设计

核心目标：

```text
同一个 obs20
同一组初始 noise
不同 action label
生成多个 future40
```

为减少随机噪声干扰，label swap 使用共享初始噪声：

```text
base_noise = [num_cases, num_repetitions,56,6,40]
对每个 label repeat 同一份 base_noise
flatten 后送入 p_sample_loop / ddim_sample_loop
```

这样 pairwise 差异更接近 label 条件差异，而不是纯随机初始噪声差异。

源样本选择策略：

```text
--sample_index 指定 test dataset index，默认 0
--num_cases 指定连续 test case 数，默认 1
```

后续可以扩展：

```text
--source_label 选择某个真实动作类的第一个样本
```

阶段 4 smoke 不需要该扩展。

## Sampling 流程

伪代码：

```python
fixseed(args.seed)
state = load_checkpoint(args.checkpoint)
model = build_model_from_state(state).to(device).eval()
diffusion = build_diffusion(state["diffusion_config"], args)
dataset = NTULabelForecastDataset(args.data_path, split="test", ...)
cases = [dataset[i] for i in range(sample_index, sample_index + num_cases)]
y, noise, metadata = build_label_swap_batch(cases, labels, num_repetitions)
sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
generated = sample_fn(
    model_or_cfg,
    (N, 56, 6, 40),
    clip_denoised=False,
    model_kwargs={"y": y},
    noise=noise,
    progress=args.progress,
    skip_timesteps=0,
    init_image=None,
    dump_steps=None,
    const_noise=False,
)
reshape generated to [num_cases,num_labels,num_repetitions,56,6,40]
save outputs
```

关键参数：

```text
clip_denoised = False
```

原因：

```text
rot6d/root translation 表示没有被训练为 [-1,1] 图像像素。
```

不使用：

```text
gaussian_filter1d smoothing
```

原因：

```text
本阶段要检查模型原始生成结果，不能在 gate 里后处理掩盖问题。
```

## 输出设计

输出目录：

```text
results/forecasting/ntu120_label/p4_label_swap_smoke
```

文件：

```text
generated_future40.npy
obs_motion.npy
real_future40.npy
metadata.json
metrics.json
label_swap_summary.json
```

### generated_future40.npy

保存 raw float32 array：

```text
shape = [num_cases,num_labels,num_repetitions,56,6,40]
```

### obs_motion.npy

```text
shape = [num_cases,56,6,20]
```

### real_future40.npy

```text
shape = [num_cases,56,6,40]
```

### metadata.json

至少包含：

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
```

### metrics.json

基础 metrics：

```text
finite = true/false
generated_shape
rot_mse_per_label
rot_mse_mean
rot_mse_by_case_label
root_translation_mse_per_label
source_action
```

`rot_mse` 使用阶段 C 同口径：

```text
mean((generated_future40 - real_future40)^2)
```

mask 当前全 1。

阶段 4 不把这些指标解释为模型质量，只用于生成闭环完整性检查。

### label_swap_summary.json

至少包含：

```text
labels
pairwise_mean_abs_diff
pairwise_max_abs_diff
all_labels_identical
pass_non_identical_check
```

通过标准：

```text
pass_non_identical_check = true
```

实现上可使用：

```text
任意 label pair 的 max_abs_diff > 1e-7
```

阈值只用于 smoke，不能作为语义差异证明。

## CLI 设计

第一版参数：

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
--num_workers 0
--use_ddim
--timestep_respacing
--progress
--overwrite
```

`batch_size` 用于把 flattened label-swap batch 分块采样，避免一次性生成过大 batch。

第一版 smoke 可限制：

```text
num_cases = 1
num_repetitions = 1
len(labels) = 4
```

## Smoke 命令

默认 p_sample_loop 命令：

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

更快的 DDIM smoke 可选：

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

阶段 4 退出 gate 至少跑通默认 p_sample_loop；DDIM 是同阶段可选扩展。

## 退出条件

必须同时满足：

```text
sample/sample_label_forecasting_diffusion.py 存在
py_compile 通过
checkpoint load 通过
模型 strict load 通过
p_sample_loop smoke 通过
generated_future40.npy 存在
obs_motion.npy 存在
real_future40.npy 存在
metadata.json 存在且可读
metrics.json 存在且可读
label_swap_summary.json 存在且可读
generated shape = [1,4,1,56,6,40]
输出无 NaN/Inf
不同 label 输出不完全相同
未修改 train/train_mdm.py
未修改 train/train_label_forecasting_diffusion.py 的训练语义
未修改 model/cmdm.py
未修改 diffusion/gaussian_diffusion.py
新增阶段 4 result 文档
```

## 风险与处理

### 风险 1: 1000-step p_sample_loop 慢

处理：

```text
默认 smoke 仍保留 p_sample_loop gate。
如果耗时过长，再额外支持 --use_ddim --timestep_respacing ddim50。
不要因为慢而跳过 p_sample_loop load/shape gate。
```

### 风险 2: label 输出完全相同

处理：

```text
先确认 action tensor 是否随 label 改变。
再确认 model.eval() 没有设置 y["uncond"]。
再确认 shared noise repeat 维度是否正确。
如果仍完全相同，回到阶段 B CFG/action token smoke。
```

### 风险 3: CFG wrapper 使用错误

处理：

```text
guidance_scale == 1.0 时不包 wrapper。
guidance_scale != 1.0 时必须设置 y["scale"]。
wrapper uncond 只 mask action，obs_motion 不清零。
```

### 风险 4: checkpoint config 与 CLI 冲突

处理：

```text
shape 参数以 checkpoint model_config 为准。
CLI window_len/obs_len/pred_len 只用于 dataset gate，并必须与 checkpoint 一致。
不允许 silent override。
```

### 风险 5: 基础 metrics 被误读成质量结论

处理：

```text
metrics.json 明确标记 smoke_only = true。
阶段 4 文档和 result 文档只写“生成闭环通过”，不写模型好坏。
```

## 阶段 4 后续

阶段 4 通过后，下一步才考虑：

```text
eval/action_consistency_classifier.py
真实 future40 分类器 gate
generated future40 label consistency
per-class / handshaking subset 指标
正式 5000-step 训练和 ablation
```

动作一致性分类器必须先证明：

```text
real future40 test accuracy 明显高于 random 1/26
handshaking subset 可被识别
```

否则不能把 generated consistency 作为主结论。
