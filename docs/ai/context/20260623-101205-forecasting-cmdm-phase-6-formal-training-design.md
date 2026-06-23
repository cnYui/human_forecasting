# ForecastingCMDM 阶段 6 正式训练设计

## 阶段定位

阶段 6 是 ForecastingCMDMDecoder 主模型的第一次正式训练阶段。

它对应原提交计划中的：

```text
Commit 6: 正式训练配置、ablation 开关与报告脚本
```

但当前阶段 6 第一版只做最小正式闭环：

```text
seed0 5000-step 正式训练
正式 checkpoint 采样
正式 generated future40 指标汇总
动作一致性分类器复评
```

它不是 smoke。阶段 A/B/C/4/5 的作用是把数据、模型、训练、采样和动作一致性评估工具都打通；阶段 6 才开始训练要拿来分析的 ForecastingCMDMDecoder 主模型。

## 参考上下文

本设计依据：

```text
docs/ai/context/20260622-105946-forecasting-cmdm-final-target-architecture-v3.md
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
docs/ai/context/20260622-114122-forecasting-cmdm-phase-a-data-gate-result.md
docs/ai/context/20260622-121949-forecasting-cmdm-phase-b-model-result.md
docs/ai/context/20260622-124111-forecasting-cmdm-phase-c-diffusion-train-result.md
docs/ai/context/20260622-222109-forecasting-cmdm-phase-4-sampling-label-swap-result.md
docs/ai/context/20260622-224756-forecasting-cmdm-phase-5-action-classifier-implementation-result.md
```

相关代码入口：

```text
data_loaders/forecasting/ntu_label.py
model/forecasting_cmdm.py
train/train_label_forecasting_diffusion.py
sample/sample_label_forecasting_diffusion.py
eval/action_consistency_classifier.py
```

## 当前前提

已通过的工程 gate：

```text
阶段 A: NTU120 2P length60 数据 gate
阶段 B: ForecastingCMDMDecoder forward/backward gate
阶段 C: diffusion train/resume smoke gate
阶段 4: checkpoint sampling / label swap gate
阶段 5: action consistency classifier 代码与真实 future40 分类 gate
```

数据协议固定：

```text
dataset = ntu120_2p
train_path = dataset/ntu120/smplx/conditioned/xsub.train.h5
test_path = dataset/ntu120/smplx/conditioned/xsub.test.h5
window_len = 60
obs_len = 20
pred_len = 40
input shape = [B,56,6,T]
condition = obs20 + action label
target = future40
num_actions = 26
handshaking = A009 = label 8
```

阶段 A 统计：

```text
train kept_count = 1956
test kept_count = 1253
train/test covered_labels = 26
train min_class_count = 2
test min_class_count = 1
handshaking train = 170
handshaking test = 68
```

## 阶段 6 目标

### 6A: seed0 正式主模型训练

目标：

```text
训练 ForecastingCMDMDecoder 到 5000 steps
保存 model000001000.pt ... model000005000.pt
保存 opt000001000.pt ... opt000005000.pt
保存 args.json
保存 train_log.jsonl
训练 loss finite
checkpoint 协议可被 sample 脚本严格加载
```

通过后，阶段 6 才有真正的主模型 checkpoint。

### 6B: 正式 checkpoint 采样

目标：

```text
加载 model000005000.pt
在 test split 上生成 label swap future40
输出 generated_future40.npy
输出 obs_motion.npy / real_future40.npy
输出 metadata.json / metrics.json / label_swap_summary.json
finite=true
pass_non_identical_check=true
```

采样结果用于后续动作一致性复评和阶段 7 视频。

### 6C: 动作一致性复评

目标：

```text
使用阶段 5 已验证的 action classifier
对阶段 6 generated_future40 做 generated consistency evaluation
输出 generated_consistency.json
valid_for_claim=true 只允许在 classifier_gate_pass=true 时成立
```

如果动作分类器 gate 不通过，阶段 6 仍可报告训练和采样结果，但不能把 generated consistency 当作语义控制结论。

## 本阶段不做

不做：

```text
不生成视频
不做三种子正式训练
不做完整 ablation
不修改 train/train_mdm.py
不修改 train/training_loop.py
不回退到 Encoder-only 主线
不把 5000-step seed0 结果包装成最终论文结论
```

原因：

```text
视频必须依赖正式 checkpoint 和正式 generated 输出，应该进入阶段 7 qualitative visualization。
三种子和 ablation 需要先看 seed0 是否稳定、是否有合理 loss 和采样结果。
旧 train_mdm.py 是 actor->reactor CMDM 协议，不是 obs20+label->future40 协议。
```

## 模型配置

主配置：

```text
model_type = forecasting_cmdm_decoder
latent_dim = 256
decoder_layers = 4
obs_encoder_layers = 2
num_heads = 4
ff_size = 1024
dropout = 0.1
activation = gelu
cond_mask_prob = 0.1
```

理由：

```text
latent_dim=256 是阶段 6 第一版正式容量。
decoder_layers=4 保留足够 target denoising 容量。
obs_encoder_layers=2 用于编码 obs20 memory。
cond_mask_prob=0.1 为后续 CFG 采样保留无条件 action dropout 能力。
```

3080 OOM 降级配置只允许：

```text
latent_dim = 192
decoder_layers = 3
obs_encoder_layers = 1
batch_size = 2
grad_accum_steps = 8
```

不允许降级为：

```text
Encoder-only
删除 action condition
删除 obs memory
缩短 pred_len
```

## Diffusion 配置

训练协议固定：

```text
diffusion_steps = 1000
noise_schedule = cosine
model_mean_type = START_X
model_var_type = FIXED_SMALL
loss_type = MSE
rescale_timesteps = false
target = future40
condition = obs_motion + action
```

训练脚本当前实现：

```text
q_sample(future40, t)
model(x_t, t, y) -> pred_xstart
loss = masked rot_mse(pred_xstart, future40)
```

默认不开启：

```text
velocity_loss_weight = 0.0
root_translation_loss_weight = 0.0
relative_root_loss_weight = 0.0
```

原因：

```text
阶段 6 第一版优先验证主模型 diffusion 训练闭环。
relative_root_loss 需要先确认 NTU120 SMPL-X 双人 root slot 索引，当前训练脚本会禁止启用。
```

## 训练命令

注意：当前 `train/train_label_forecasting_diffusion.py` 会强制要求：

```text
eval_interval = 0
```

因此阶段 6 不设计训练中自动 eval。评估在训练结束后单独用 sampling / classifier 脚本完成。

seed0 正式训练命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m train.train_label_forecasting_diffusion \
  --dataset ntu120_2p \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000 \
  --model_type forecasting_cmdm_decoder \
  --body_model smplx \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 4 --grad_accum_steps 4 --eval_batch_size 4 \
  --num_steps 5000 --save_interval 1000 --eval_interval 0 \
  --latent_dim 256 --decoder_layers 4 --obs_encoder_layers 2 \
  --num_heads 4 --ff_size 1024 --dropout 0.1 --activation gelu \
  --cond_mask_prob 0.1 \
  --lr 1e-4 --weight_decay 1e-4 --clip_grad_norm 1.0 \
  --noise_schedule cosine --schedule_sampler uniform \
  --velocity_loss_weight 0.0 \
  --root_translation_loss_weight 0.0 \
  --relative_root_loss_weight 0.0 \
  --num_workers 0 --seed 0 --overwrite
```

预期输出：

```text
save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/
  args.json
  train_log.jsonl
  model000001000.pt
  model000002000.pt
  model000003000.pt
  model000004000.pt
  model000005000.pt
  opt000001000.pt
  opt000002000.pt
  opt000003000.pt
  opt000004000.pt
  opt000005000.pt
```

## 训练监控

阶段 6 训练期间只看训练日志，不把训练 loss 当作最终效果：

```text
train_loss
rot_mse
lr
effective_batch_size
model_num_params
step
checkpoint
```

最低可接受训练状态：

```text
train_log.jsonl 可逐行 JSON 解析
step 从 1 到 5000 连续记录
loss 全部 finite
model000005000.pt 存在
opt000005000.pt 存在
checkpoint.step = 5000
checkpoint.model_type = forecasting_cmdm_decoder
checkpoint.train_protocol.mean_type = START_X
checkpoint.train_protocol.loss_type = MSE
```

如果中断：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m train.train_label_forecasting_diffusion \
  --dataset ntu120_2p \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000 \
  --model_type forecasting_cmdm_decoder \
  --body_model smplx \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 4 --grad_accum_steps 4 --eval_batch_size 4 \
  --num_steps 5000 --save_interval 1000 --eval_interval 0 \
  --latent_dim 256 --decoder_layers 4 --obs_encoder_layers 2 \
  --num_heads 4 --ff_size 1024 --dropout 0.1 --activation gelu \
  --cond_mask_prob 0.1 \
  --lr 1e-4 --weight_decay 1e-4 --clip_grad_norm 1.0 \
  --noise_schedule cosine --schedule_sampler uniform \
  --velocity_loss_weight 0.0 \
  --root_translation_loss_weight 0.0 \
  --relative_root_loss_weight 0.0 \
  --num_workers 0 --seed 0 \
  --resume_checkpoint save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model00000XXXX.pt
```

其中 `model00000XXXX.pt` 必须替换成最后一个完整 checkpoint。

## 正式采样设计

阶段 6 正式采样使用：

```text
sample/sample_label_forecasting_diffusion.py
```

基础 label set：

```text
labels = [2, 5, 8, 17]
```

其中：

```text
8 = A009 handshaking
```

第一轮正式采样命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m sample.sample_label_forecasting_diffusion \
  --checkpoint save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --labels 2 5 8 17 \
  --guidance_scale 1.0 \
  --batch_size 4 --num_cases 8 --num_repetitions 2 \
  --sample_index 0 --seed 0 \
  --use_ddim --timestep_respacing ddim50 --overwrite
```

选择 DDIM50 作为第一轮正式采样，是为了降低采样时间；若结果异常，再用完整 `p_sample_loop` 对少量 case 做对照。

预期输出：

```text
results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap/
  generated_future40.npy
  obs_motion.npy
  real_future40.npy
  metadata.json
  metrics.json
  label_swap_summary.json
```

输出 shape：

```text
generated_future40.npy = [8,4,2,56,6,40]
obs_motion.npy = [8,56,6,20]
real_future40.npy = [8,56,6,40]
```

## 动作一致性复评设计

阶段 6 复评使用阶段 5 分类器入口：

```text
eval/action_consistency_classifier.py
```

若阶段 5 已有正式 action classifier checkpoint，优先使用对应正式配置。当前脚本会重新训练分类器并评估 generated consistency；因此阶段 6 第一版可直接用正式 5B 配置重新运行一遍，并把 `generated_dir` 指向阶段 6 采样目录。

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m eval.action_consistency_classifier \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --generated_dir results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap \
  --save_dir save/forecasting/ntu120_label/phase6_action_classifier_generated_consistency_h256_b3_s0 \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 16 --eval_batch_size 64 \
  --num_steps 2000 --save_interval 2000 --eval_interval 0 \
  --hidden_dim 256 --num_blocks 3 --dropout 0.2 \
  --seed 0 --overwrite
```

预期输出：

```text
classifier_model.pt
normalizer.pt
real_test_metrics.json
real_test_predictions.jsonl
generated_consistency.json
generated_predictions.jsonl
confusion_matrix.npy
train_log.jsonl
```

可解释条件：

```text
real_test_metrics.classifier_gate_pass = true
generated_consistency.valid_for_claim = true
```

如果 `valid_for_claim=false`：

```text
只记录为 debug 结果
不能写成“生成动作符合标签”
```

## 阶段 6 检查脚本思路

阶段 6 可以新增一个轻量检查脚本，但不是必须：

```text
scripts/check_phase6_formal_outputs.py
```

检查内容：

```text
checkpoint exists
checkpoint.step = 5000
train_log rows = 5000
train loss finite
generated_future40.npy shape 正确
metadata.checkpoint 指向 model000005000.pt
metrics.finite = true
label_swap_summary.pass_non_identical_check = true
generated_consistency.valid_for_claim 字段存在
```

如果不新增脚本，也可以用一次性 Python 命令在 result 文档中记录核对结果。

## 结果文档要求

阶段 6 完成后必须新增 result 文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-6-formal-training-result.md
```

必须记录：

```text
训练命令
训练环境
save_dir
最终 checkpoint
train_log loss 起止值
checkpoint 协议字段
采样命令
generated 输出目录
generated shape
metrics.json 核心指标
label_swap_summary 是否非完全相同
generated_consistency 是否 valid_for_claim
失败或异常处理
```

不要记录：

```text
不把大段 train_log 全贴进文档
不提交 save/ 或 results/ 大文件
不把 seed0 单次结果写成最终论文结论
```

## 退出条件

阶段 6 通过需要同时满足：

```text
model000005000.pt 存在
opt000005000.pt 存在
train_log.jsonl 可解析且 loss finite
checkpoint 可被 sample 脚本 strict load
正式采样输出 generated_future40.npy
generated finite = true
label swap 非完全相同
动作一致性复评输出 generated_consistency.json
新增阶段 6 result 文档
AGENTS.md 更新阶段记忆
```

阶段 6 不要求：

```text
视频输出
三种子结果
完整 ablation 表格
论文最终指标
```

## 下一阶段

阶段 6 通过后，进入阶段 7：

```text
正式 checkpoint qualitative visualization / video
```

阶段 7 才处理：

```text
选择可视化样本
确定 skeleton / xyz 转换路径
渲染 obs20 / gt future40 / generated future40
输出 mp4 或 frame sequence
为 handshaking label swap 生成演示视频
```

阶段 7 的设计必须先确认当前 rot6d + root translation 到可视化 skeleton 的可靠路径，不能直接把 `[56,6,T]` 当作可画 xyz。
