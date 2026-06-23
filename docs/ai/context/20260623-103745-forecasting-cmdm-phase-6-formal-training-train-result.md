# ForecastingCMDM 阶段 6 正式训练结果

## 结论

阶段 6 seed0 5000-step ForecastingCMDMDecoder 主模型训练已完成。

本结果只覆盖训练端：

```text
6A seed0 正式主模型训练
```

尚未覆盖：

```text
6B 正式 checkpoint 采样
6C 动作一致性复评
scripts/check_phase6_formal_outputs.py 完整阶段检查
```

## 训练命令

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

## 运行环境

```text
python = /home/rpartx3080/.local/micromamba/envs/regennet/bin/python
device = cuda
gpu = NVIDIA GeForce RTX 3080
model_num_params = 6258512
effective_batch_size = 16
seed = 0
```

## 输出目录

```text
save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000
```

关键文件：

```text
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

最终 checkpoint：

```text
save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt
```

## 基础核对

训练结束输出：

```text
Training finished. final_checkpoint=save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt
```

核对结果：

```text
train_log rows = 5000
steps_contiguous = true
loss_finite = true
first_train_loss = 0.5163707286119461
last_train_loss = 0.020558608695864677
checkpoint_step = 5000
checkpoint_model_type = forecasting_cmdm_decoder
train_protocol.mean_type = START_X
train_protocol.loss_type = MSE
train_protocol.target = future
train_protocol.condition = obs_motion + action
```

文件大小：

```text
model000005000.pt = 29M
opt000005000.pt = 48M
train_log.jsonl = 1.2M
args.json = 2.4K
```

## 下一步

继续阶段 6B：

```text
用 model000005000.pt 运行正式 DDIM50 label swap 采样。
采样命令必须传 --formal，使 metrics.smoke_only=false。
```

阶段 6B/6C 完成后，再新增完整 `forecasting-cmdm-phase-6-formal-training-result.md` 并运行：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  scripts/check_phase6_formal_outputs.py
```
