# ForecastingCMDM high-noise / one-step 微调监控结果

## 目标

继续微调阶段 6 checkpoint，并实时监控：

```text
generated_future40 vs real_future40 的 MSE / MAE 是否下降
```

主指标不使用分类器。

## 微调配置

起点：

```text
save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt
```

主微调目录：

```text
save/forecasting/ntu120_label/phase6d_high_noise_onestep_ft_h256_l4_s0_5100
```

训练配置：

```text
lr = 5e-5
batch_size = 4
grad_accum_steps = 4
effective_batch_size = 16
timestep_sampling = high_noise
high_noise_min_t = 750
one_step_noise_prob = 0.5
one_step_t = 999
save_interval = 100
```

训练段：

```text
5000 -> 5100
5100 -> 5200
5200 -> 5300
5300 -> 5400
5400 -> 5500
5500 -> 5600
```

5600 后发现同参数反弹，因此额外从 5500 出发尝试：

```text
one_step_noise_prob = 0.25
5500 -> 5600
```

输出：

```text
save/forecasting/ntu120_label/phase6d_high_noise_onestep025_ft_h256_l4_s0_5600
```

## 监控结果

### one_step_t999 full test

全 test split，1253 samples。

| checkpoint | MSE | MAE | copy-last MSE | copy-last MAE |
|---|---:|---:|---:|---:|
| 5000 | 0.027396665 | 0.081076726 | 0.035849254 | 0.079228513 |
| 5100 | 0.027100256 | 0.080811010 | 0.035849254 | 0.079228513 |
| 5200 | 0.026947508 | 0.080636046 | 0.035849254 | 0.079228513 |
| 5300 | 0.026962407 | 0.080368759 | 0.035849254 | 0.079228513 |
| 5400 | 0.026898429 | 0.079696646 | 0.035849254 | 0.079228513 |
| 5500 | 0.026942649 | 0.079396160 | 0.035849254 | 0.079228513 |
| 5600 p=0.5 | 0.027513730 | 0.080022805 | 0.035849254 | 0.079228513 |
| 5600 p=0.25 | 0.026528949 | 0.079123937 | 0.035849254 | 0.079228513 |

判断：

```text
one_step_t999 的 MSE/MAE 总体改善。
p=0.25 的 5600 在 one-step 上最佳，且 MAE 首次略优于 copy-last。
```

但 one-step 不是最终视频生成口径，只说明从纯噪声一步预测 future 的能力增强。

### DDIM50 256 samples

用于快速监控最终多步采样趋势。

| checkpoint | MSE | MAE | copy-last MSE | copy-last MAE |
|---|---:|---:|---:|---:|
| 5000 | 0.044796406 | 0.090858876 | 0.045332626 | 0.078212104 |
| 5100 | 0.042983061 | 0.087533941 | 0.045332626 | 0.078212104 |
| 5200 | 0.040557547 | 0.085815316 | 0.045332626 | 0.078212104 |
| 5300 | 0.038834284 | 0.084578923 | 0.045332626 | 0.078212104 |
| 5400 | 0.038598633 | 0.083986718 | 0.045332626 | 0.078212104 |
| 5500 | 0.037719118 | 0.082567059 | 0.045332626 | 0.078212104 |
| 5600 p=0.5 | 0.039540842 | 0.083658710 | 0.045332626 | 0.078212104 |
| 5600 p=0.25 | 0.040283758 | 0.083025351 | 0.045332626 | 0.078212104 |

判断：

```text
DDIM50 256 样本在 5000 -> 5500 持续下降。
继续到 5600 后 MSE 反弹。
p=0.25 能改善 one-step，但不能改善 DDIM50。
```

因此最终多步生成当前最佳 checkpoint 是：

```text
save/forecasting/ntu120_label/phase6d_high_noise_onestep_ft_h256_l4_s0_5100/model000005500.pt
```

## DDIM50 full test 同口径确认

为了避免只看 256 samples，额外跑全 test split DDIM50。

### 5000 原始 checkpoint

输出：

```text
results/forecasting/ntu120_label/phase6d_ft5000_distance_ddim50_full_test
```

结果：

```text
num_samples = 1253
generated MSE = 0.036687691
generated MAE = 0.092765272
copy-last MSE = 0.035849255
copy-last MAE = 0.079228513
```

判断：

```text
原始 5000 的 DDIM50 全量 MSE/MAE 都弱于 copy-last。
```

### 5500 微调 checkpoint

输出：

```text
results/forecasting/ntu120_label/phase6d_ft5500_distance_ddim50_full_test
```

结果：

```text
num_samples = 1253
generated MSE = 0.030736648
generated MAE = 0.083149877
copy-last MSE = 0.035849255
copy-last MAE = 0.079228513
```

判断：

```text
5500 微调后 DDIM50 全量 MSE 明显优于 copy-last。
MAE 相比原始 5000 明显下降，但仍高于 copy-last。
```

## 改善幅度

DDIM50 full test，5000 -> 5500：

```text
MSE: 0.036687691 -> 0.030736648
绝对下降: 0.005951043
相对下降: 约 16.22%

MAE: 0.092765272 -> 0.083149877
绝对下降: 0.009615395
相对下降: 约 10.36%
```

与 copy-last 对比：

```text
5000: generated MSE 比 copy-last 差 0.000838437
5500: generated MSE 比 copy-last 好 0.005112607
```

## 结论

这次微调找到了能提升拟合度的方向：

```text
high-noise timestep sampling + one-step pure-noise branch
```

它确实让最终 DDIM50 生成的 future40 更贴近真实 future40：

```text
全量 DDIM50 MSE 从弱于 copy-last 变成明显优于 copy-last。
```

但还没有完全解决 MAE：

```text
5500 的 DDIM50 MAE 仍高于 copy-last。
```

当前建议：

```text
1. 以 model000005500.pt 作为当前最佳生成 checkpoint。
2. 不继续使用 p=0.5 训到 5600，因为已经反弹。
3. 如果继续优化 MAE，下一步应降低 one_step 比例或加入 velocity/root loss，而不是继续同参数硬训。
```
