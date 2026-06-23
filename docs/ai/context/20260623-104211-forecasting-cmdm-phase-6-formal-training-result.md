# ForecastingCMDM 阶段 6 正式训练完整结果

## 结论

阶段 6 工程闭环已完成：

```text
6A seed0 5000-step 正式训练：完成
6B 正式 checkpoint DDIM50 label swap 采样：完成
6C 动作一致性分类器复评：完成
scripts/check_phase6_formal_outputs.py：通过
```

关键边界：

```text
工程闭环通过，但 generated consistency 结果为 0.0。
这说明分类器 gate 通过后，仍没有任何 generated future40 被 top1 分类成输入条件标签。
因此不能写成“生成动作符合动作标签”或“语义控制成功”。
```

## 6A 训练结果

训练结果已单独记录：

```text
docs/ai/context/20260623-103745-forecasting-cmdm-phase-6-formal-training-train-result.md
```

最终 checkpoint：

```text
save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt
```

训练端核对：

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

## 6B 正式采样

命令：

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
  --use_ddim --timestep_respacing ddim50 \
  --formal --run_name phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap \
  --overwrite
```

输出目录：

```text
results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap
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

采样指标：

```text
generated_shape = [8,4,2,56,6,40]
finite = true
metrics.smoke_only = false
rot_mse_mean = 0.05050671473145485
root_translation_mse_mean = 0.005765178706496954
label_swap_summary.pass_non_identical_check = true
```

## 6C 动作一致性复评

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

输出目录：

```text
save/forecasting/ntu120_label/phase6_action_classifier_generated_consistency_h256_b3_s0
```

真实 future40 分类器 gate：

```text
top1_acc = 0.859537110933759
top5_acc = 0.9696727853152434
balanced_acc = 0.5975166737332928
handshaking_acc = 0.9117647058823529
classifier_gate_pass = true
```

generated consistency：

```text
valid_for_claim = true
consistency_acc = 0.0
num_generated_samples = 64
top1_confidence_mean = 0.1609623150434345
```

Condition label counts：

```text
label 2  = 16
label 5  = 16
label 8  = 16
label 17 = 16
```

Predicted label counts：

```text
label 0  = 16
label 2  = 7
label 3  = 5
label 7  = 33
label 11 = 3
```

解释：

```text
真实 future40 分类器足够强，因此 generated consistency 是可解释指标。
但本次 generated top1 没有命中任何输入条件标签，说明当前 5000-step seed0 模型没有表现出可靠 label semantic control。
label swap 非完全相同只说明标签或采样路径改变了输出，不能说明动作语义正确。
```

## 完整检查脚本

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  scripts/check_phase6_formal_outputs.py
```

结果：

```text
pass = true
errors = []
checkpoint.step = 5000
checkpoint.model_type = forecasting_cmdm_decoder
train_log rows = 5000
generated_shape = [8,4,2,56,6,40]
sampling.finite = true
sampling.smoke_only = false
label_swap_summary.pass_non_identical_check = true
generated_consistency.classifier_gate_pass = true
generated_consistency.valid_for_claim = true
generated_consistency.consistency_acc = 0.0
```

## 后续判断

阶段 6 可以作为正式闭环结果记录，但不是正向语义控制结果。

建议下一步不要直接进入论文正结论，而是先做错误分析：

```text
1. 检查 generated future40 的尺度、normalizer 和 train real future40 是否分布接近。
2. 用少量 p_sample_loop 非 DDIM50 对照，排除 DDIM50 采样退化。
3. 检查 CFG / cond_mask_prob 是否需要 guidance_scale > 1。
4. 检查 action embedding 是否在训练中被有效使用，可做 label shuffle / no-label ablation。
5. 进入阶段 7 可视化前，先挑选 label 8 handshaking 的生成样本做定性审查。
```
