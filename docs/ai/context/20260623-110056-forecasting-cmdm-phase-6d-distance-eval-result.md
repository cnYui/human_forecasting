# ForecastingCMDM Phase 6D 距离评估实现与结果

## 背景

用户指出：

```text
最终目标是生成帧和真实数据集帧更拟合。
不需要用分类器判断 generated 是否被分类成条件标签。
```

因此本阶段补充正式距离评估入口，主指标改为：

```text
generated_future40 vs real_future40
```

## 实现

新增脚本：

```text
eval/eval_label_forecasting_distance.py
```

功能：

```text
1. 加载 ForecastingCMDMDecoder checkpoint。
2. 遍历 NTU120 2P test split。
3. 使用每条样本自己的真实 source action label。
4. 生成 future40。
5. 直接计算 generated / copy-last / zero 与 real_future40 的 MSE、RMSE、MAE。
6. 输出整体指标、per-action 指标和 sample-level jsonl。
```

支持模式：

```text
ddim50
p_sample_loop
one_step_t999
```

输出：

```text
metadata.json
metrics.json
per_action_metrics.json
sample_metrics.jsonl
```

可选保存：

```text
generated_future40.npy
real_future40.npy
obs_motion.npy
```

## 验证

语法检查：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python -m py_compile eval/eval_label_forecasting_distance.py
```

通过。

### one_step_t999 smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m eval.eval_label_forecasting_distance \
  --checkpoint save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/phase6d_distance_smoke_one_step_t999 \
  --mode one_step_t999 \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 4 --sample_batch_size 4 --max_samples 4 \
  --seed 0 --overwrite
```

结果：

```text
samples = 4
generated_mse = 0.039533
copy_last_mse = 0.068108
```

### DDIM50 smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m eval.eval_label_forecasting_distance \
  --checkpoint save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/phase6d_distance_smoke_ddim50 \
  --mode ddim50 \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 2 --sample_batch_size 2 --max_samples 2 \
  --seed 0 --overwrite
```

结果：

```text
samples = 2
generated_mse = 0.071545
copy_last_mse = 0.086034
```

### p_sample_loop smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m eval.eval_label_forecasting_distance \
  --checkpoint save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/phase6d_distance_smoke_p_sample_loop \
  --mode p_sample_loop \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 1 --sample_batch_size 1 --max_samples 1 \
  --seed 0 --overwrite
```

结果：

```text
samples = 1
generated_mse = 0.092917
copy_last_mse = 0.085673
```

该结果只验证接口，不作为正式 p_sample_loop 结论。

## 正式距离结果

### one_step_t999 full test

输出目录：

```text
results/forecasting/ntu120_label/phase6d_distance_one_step_t999_full_test
```

结果：

```text
num_samples = 1253
generated_mse = 0.0273966648
generated_rmse = 0.1655193788
generated_mae = 0.0810767258

copy_last_mse = 0.0358492542
copy_last_rmse = 0.1893389928
copy_last_mae = 0.0792285130

zero_mse = 0.1785055200
zero_rmse = 0.4224991361
zero_mae = 0.2537976741
```

判断：

```text
one_step_t999 在全 test split 上 MSE/RMSE 明显优于 copy-last。
但 MAE 略差于 copy-last。
```

这说明模型确实学到了从 obs/action 到 future 的有用预测能力。

### DDIM50 256 samples

输出目录：

```text
results/forecasting/ntu120_label/phase6d_distance_ddim50_256
```

结果：

```text
num_samples = 256
generated_mse = 0.0447964055
generated_rmse = 0.2116516136
generated_mae = 0.0908588756

copy_last_mse = 0.0453326257
copy_last_rmse = 0.2129145971
copy_last_mae = 0.0782121037

zero_mse = 0.1745593650
zero_rmse = 0.4178030218
zero_mae = 0.2485746191
```

判断：

```text
DDIM50 MSE 只比 copy-last 略好，MAE 明显更差。
这和之前 probe 一致，说明多步 free sampling 没有充分保留 teacher-forced / one-step 学到的预测能力。
```

## 结论

当前问题不是“模型不能根据输入标签和帧生成”。

更准确地说：

```text
模型在训练分布内学到了从 obs/action 预测 future 的能力；
但当前 DDIM50 / p_sample_loop 多步自由采样链路会累积误差；
最终 generated future40 和真实 future40 的距离没有明显优于简单 copy-last baseline。
```

因此后续修复重点应是：

```text
1. high-noise timestep sampling
2. one-step pure-noise prediction fine-tune
3. velocity/root loss
4. K-step final-sample loss
```

其中第 4 项最直接对应用户目标：

```text
loss = MSE(final_generated_future40, real_future40)
```

但它代价最高，需要小 batch 和短 K 先 smoke。
