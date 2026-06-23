# ForecastingCMDMDecoder 阶段 C Diffusion 训练设计

## 参考上下文

本文依据以下文档和当前代码状态：

```text
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
docs/ai/context/20260622-114122-forecasting-cmdm-phase-a-data-gate-result.md
docs/ai/context/20260622-121949-forecasting-cmdm-phase-b-model-result.md
data_loaders/forecasting/ntu_label.py
model/forecasting_cmdm.py
utils/model_util.py
diffusion/gaussian_diffusion.py
diffusion/resample.py
train/train_forecasting.py
train/train_forecasting_xyz.py
```

## 阶段命名

本文中的阶段 C 指阶段 A/B 后的下一步：

```text
阶段 A: NTU120 2P label forecasting 数据 gate
阶段 B: ForecastingCMDMDecoder 模型 gate
阶段 C: diffusion 训练入口、2-step smoke、checkpoint save/resume gate
```

这不是总提交计划里“阶段 C: 标签语义评估”。标签语义评估、label swap 和动作一致性分类器必须等本阶段训练闭环通过后再进入。

## 当前已满足条件

阶段 A 已提供稳定 batch contract：

```text
future      = [B,56,6,40]
obs_motion  = [B,56,6,20]
action      = [B,1]
mask        = [B,1,1,40]
```

数据 gate 结果：

```text
train kept_count = 1956
test kept_count = 1253
train/test 均覆盖 26 类
handshaking label 8 train/test 非零
obs_motion/future 全量 finite
```

阶段 B 已完成：

```text
model/forecasting_cmdm.py
ForecastingCMDMDecoder
ForecastingClassifierFreeSampleModel
随机输入 forward/backward/CFG smoke 通过
dataset batch forward/backward 通过
CFG uncond 只 mask action，不清零 obs_motion
obs PE = 0..19
future PE = 20..59
```

运行环境优先使用：

```text
python_executable = /home/rpartx3080/.local/micromamba/envs/regennet/bin/python
torch_version = 1.7.1
```

## 阶段 C 目标

新增训练入口：

```text
train/train_label_forecasting_diffusion.py
```

本阶段只证明训练闭环可靠：

```text
构建 NTULabelForecastDataset/DataLoader
构建 ForecastingCMDMDecoder
构建 START_X / MSE Gaussian diffusion
从 clean future40 采样 x_t
模型预测 clean future40
计算 rot_mse + 可选 velocity_mse/root_translation_mse/relative_root_mse
2-step smoke loss finite
保存 args.json、train_log.jsonl、model*.pt、opt*.pt
resume_checkpoint 可恢复 model/optimizer/step
```

本阶段通过后，才允许进入：

```text
sample/sample_label_forecasting_diffusion.py
label swap
基础预测评估
动作一致性分类器 gate
正式 5000-step 训练
```

## 本阶段不做

不做以下内容：

```text
不实现采样脚本
不做 label swap
不做动作一致性分类器
不跑正式 5000-step 训练
不改 train/train_mdm.py
不改原始 CMDM.forward 语义
不把 obs_motion 塞进 y["cmotion"]
不把模型回退为 Encoder-only
不在 smoke 阶段引入 SMPL-X xyz / MPJPE 依赖
```

原因：

```text
阶段 C 的必要问题是训练协议是否能闭环。
采样、语义一致性和 xyz 指标是后续问题；提前加入会让失败原因混杂。
```

## 关键设计判断

### 不直接复用 TrainLoop

`train/training_loop.py` 当前接口是：

```text
for motion, cond in data:
    diffusion.training_losses(model, motion, t, model_kwargs=cond, dataset=...)
```

该路径绑定旧 CMDM 协议：

```text
motion 是旧 reactor target
cond["y"]["cmotion"] 是旧 actor condition
评估分支绑定 humanml/humanact/uestc
DDP、logger、rot2xyz 假设都来自旧训练入口
```

阶段 C 需要的协议是：

```text
x_start = batch["future"]
y = {
    "obs_motion": batch["obs_motion"],
    "action": batch["action"],
    "mask": batch["mask"],
}
```

因此第一版使用独立轻量训练循环，参考 `train/train_forecasting.py` / `train/train_forecasting_xyz.py` 的保存、日志、resume 风格。

### 不直接复用 diffusion.training_losses

`diffusion/gaussian_diffusion.py::training_losses` 当前有这些旧耦合：

```text
enc = model.model.module
get_xyz = enc.rot2xyz(...)
model_kwargs["y"]["cmotion"] 用于 relation loss
几何项默认假设旧 CMDM dataset/body_model 语义
```

本阶段只复用 diffusion 对象的：

```text
q_sample
num_timesteps
START_X / MSE 配置
schedule_sampler
```

训练 loss 在新入口内显式计算。这样能保持 diffusion 加噪口径一致，同时避免旧 CMDM 的 condition 语义污染 forecasting 协议。

### 模型输出目标

使用 `ModelMeanType.START_X`：

```text
x_start = future40
noise = torch.randn_like(x_start)
x_t = diffusion.q_sample(x_start, t, noise)
pred_xstart = model(x_t, t, y)
rot_mse = masked_l2(pred_xstart, x_start, mask)
```

不训练 epsilon：

```text
阶段 B 模型设计和历史 v3 架构都以 clean future40 denoising 为目标。
START_X 也和现有 create_gaussian_diffusion 的默认 predict_xstart=True 一致。
```

### mask 语义

数据集返回：

```text
mask = [B,1,1,40]
```

阶段 C 只支持全 1 future mask。仍保留 mask 参与 loss，原因是后续可兼容 variable future 或 padding，但本阶段不扩大数据协议。

## 文件设计

新增：

```text
train/train_label_forecasting_diffusion.py
```

不新增公共 util，除非实现时出现明确重复。第一版训练入口内保留私有函数：

```text
_utc_now
_device
_prepare_save_dir
_write_json
_append_train_log
_build_dataset
_build_loader
_build_model
_build_diffusion
_checkpoint_paths
_save_checkpoint
_load_resume
_masked_l2
_velocity_mse
_root_translation_mse
_relative_root_mse
_diffusion_train_step
run_training
build_arg_parser
main
```

后续若采样脚本也需要 `_build_model` / `_build_diffusion`，再抽到：

```text
utils/forecasting_cmdm.py
```

阶段 C 不提前抽象，避免还没稳定的协议变成公共 API。

## Diffusion 构建

优先复用现有 `utils.model_util.create_gaussian_diffusion(args)`，但要补齐它要求的字段：

```text
noise_schedule = cosine
timestep_respacing = ""
sigma_small = True
lambda_vel = 0.0
lambda_rcxyz = 0.0
lambda_fc = 0.0
lambda_orient = 0.0
lambda_body = 0.0
lambda_transl = 0.0
vel_threshold = 0.01
pose_rep = rot6d
num_person = 2
body_model = smplx
```

如果直接复用 `create_gaussian_diffusion(args)` 因参数集合过宽导致入口脆弱，则在本训练脚本内创建一个等价的 `_build_diffusion(args)`：

```text
gd.get_named_beta_schedule(args.noise_schedule, 1000, 1.0)
SpacedDiffusion(...)
model_mean_type = START_X
model_var_type = FIXED_SMALL if sigma_small else FIXED_LARGE
loss_type = MSE
rescale_timesteps = False
data_rep = rot6d
num_person = 2
body_model = smplx
```

二者必须保持：

```text
rescale_timesteps = False
```

原因：

```text
ForecastingCMDMDecoder 已显式拒绝 float timesteps。
```

## Loss 设计

### rot_mse

主 loss：

```python
loss = ((pred - target) ** 2 * mask.float()).sum(dim=(1, 2, 3))
denom = mask.float().sum(dim=(1, 2, 3)) * pred.shape[1] * pred.shape[2]
rot_mse = loss / denom.clamp_min(1.0)
```

返回 `[B]`，再按 schedule weights 做 batch mean：

```python
loss = (terms["loss"] * weights).mean()
```

### velocity_mse

默认权重为 0，smoke 不启用：

```text
velocity = x[..., 1:] - x[..., :-1]
mask_vel = mask[..., 1:]
```

正式训练前可以开启：

```text
velocity_loss_weight = 0.1
```

但阶段 C smoke 的 gate 不依赖该项。

### root_translation_mse

默认权重为 0，smoke 不启用。若启用，只取最后一个 slot 的前三维：

```text
root_translation = motion[:, -1:, 0:3, :]
```

原因：

```text
当前项目历史约定最后一个 slot 保存 root translation。
```

### relative_root_mse

默认权重为 0，smoke 不启用。若启用：

```text
person1_root = motion[:, 27:28, 0:3, :]
person2_root = motion[:, -1:, 0:3, :]
relative_root = person2_root - person1_root
```

该索引需在实现前通过 NTU120 SMPL-X H5 表示再确认。若不能确认，先不实现该项，只保留 TODO，避免写错关系损失。

### 总 loss

第一版：

```text
loss = rot_mse
```

可选项：

```text
loss = rot_mse
     + velocity_loss_weight * velocity_mse
     + root_translation_loss_weight * root_translation_mse
     + relative_root_loss_weight * relative_root_mse
```

阶段 C smoke 默认只启用 `rot_mse`，因为本阶段目标是闭环而不是调优生成质量。

## 训练循环

核心伪代码：

```python
while step < args.num_steps:
    for batch in train_loader:
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
        terms = compute_losses(pred, future, y["mask"], args)
        loss = (terms["loss"] * weights).mean()
        (loss / grad_accum_steps).backward()
        optimizer.step()
        save/log/resume gate
```

训练时：

```text
model.train()
cond_mask_prob 生效，用于 CFG 训练
```

评估和 smoke 不在本阶段做 full evaluation，但训练日志至少记录：

```text
step
train_loss
rot_mse
lr
effective_batch_size
model_num_params
seed
checkpoint
optimizer
```

## Checkpoint 协议

保存路径沿用项目风格：

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
```

`train_protocol` 固定记录：

```text
dataset = ntu120_2p
window_len = 60
obs_len = 20
pred_len = 40
num_actions = 26
target = future
condition = obs_motion + action
mean_type = START_X
loss_type = MSE
```

optimizer checkpoint 内容：

```text
optimizer_state_dict
step
```

resume 规则：

```text
--resume_checkpoint 指向 model*.pt
检查 model_type 一致
检查 model_config 的核心 shape 参数一致
加载同目录 opt*.pt；不存在时允许只恢复模型并警告
从 checkpoint step 继续到 args.num_steps
```

核心 shape 参数：

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

## CLI 参数

第一版参数：

```text
--dataset ntu120_2p
--data_path dataset/ntu120/smplx/conditioned/xsub.train.h5
--eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5
--save_dir ...
--model_type forecasting_cmdm_decoder
--body_model smplx
--window_len 60
--obs_len 20
--pred_len 40
--batch_size
--eval_batch_size
--num_steps
--save_interval
--eval_interval
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

`eval_data_path` 阶段 C 只保存到 `args.json`，不跑 eval。保留该参数是为了和后续采样/评估入口保持同一实验目录元信息。

## 2-step smoke 命令

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

## Resume smoke 命令

第一次跑到 step 2 后，再跑：

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
step 从 2 继续到 3
model000000003.pt 存在
opt000000003.pt 存在
train_log.jsonl 追加 step=3 记录
loss finite
```

## 阶段 C 退出条件

必须同时满足：

```text
train/train_label_forecasting_diffusion.py 可 import
py_compile 通过
2-step smoke 训练完成
loss / rot_mse finite
model000000002.pt 存在
opt000000002.pt 存在
args.json 存在且记录协议参数
train_log.jsonl 至少含 step=1/2 或最终 step=2
resume 到 step=3 通过
checkpoint model_config 与 ForecastingCMDMDecoder.config() 一致
未修改 train/train_mdm.py
未修改 model/cmdm.py
未引入 Encoder-only 回退路径
```

## 风险与处理

### 风险 1: 旧 diffusion.training_losses 误用

处理：

```text
阶段 C 不调用 diffusion.training_losses。
只调用 q_sample。
```

### 风险 2: save_dir 已存在

处理：

```text
目录存在且非空时，除非 --overwrite 或 --resume_checkpoint，否则报错。
--overwrite 只允许用于 smoke；正式训练建议换新 save_dir。
```

### 风险 3: relative root 索引不确定

处理：

```text
阶段 C smoke 不启用 relative_root_loss。
实现时如不能确认双人 root slot，只写 TODO，不猜索引。
```

### 风险 4: DDP / MPI 复杂度干扰 smoke

处理：

```text
第一版单进程 torch 训练。
后续正式长训如需要多卡，再专门设计 DDP 版本。
```

### 风险 5: CFG 训练被关掉

处理：

```text
cond_mask_prob 默认 0.1。
阶段 C 不在训练入口里把 model.eval() 用于训练 step。
```

## 后续阶段入口

阶段 C 通过后，下一步才是生成闭环：

```text
sample/sample_label_forecasting_diffusion.py
checkpoint load
p_sample_loop / DDIM
same obs20 + labels [2,5,8,17] label swap smoke
generated_future40.npy
metadata.json
metrics.json
label_swap_summary.json
```

动作一致性分类器仍应放在更后面：

```text
eval/action_consistency_classifier.py
```
