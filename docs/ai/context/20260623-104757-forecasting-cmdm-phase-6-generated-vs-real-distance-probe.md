# ForecastingCMDM 阶段 6 Generated vs Real 距离检查

## 目的

用户要求不使用分类器语义命中，而是直接看生成 future40 和真实 future40 差得多不多。

## 已有 6B label swap 输出

已有 6B 输出目录：

```text
results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap
```

该输出是 label swap：

```text
labels = [2,5,8,17]
source_actions = [0,1,2,3,4,5,6,7]
```

只有 case 2 和 case 5 的 source action 在 label swap labels 内，因此直接把全部 generated 和 real future40 比不是完全公平。

整体结果：

```text
overall_mse = 0.050506722182035446
overall_rmse = 0.2247370034456253
overall_mae = 0.09330128878355026
real_abs_mean = 0.24624601006507874
real_std = 0.41235271096229553
copy_last_obs_mse = 0.04473424702882767
copy_last_obs_rmse = 0.211504727602005
copy_last_obs_mae = 0.06692911684513092
```

解释：

```text
在 label swap 输出上，generated 比复制最后一帧 baseline 更差。
但这不是标准预测误差，因为大多数生成标签不是该样本真实动作标签。
```

## Source-label match 公平对照

补跑一个小 probe：

```text
results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_source_label_match_probe
```

采样设置：

```text
labels = [0,1,2,3,4,5,6,7]
num_cases = 8
num_repetitions = 2
sample_index = 0
DDIM50
guidance_scale = 1.0
```

只统计每个 case 的真实 source label 对应生成结果：

```text
source_actions = [0,1,2,3,4,5,6,7]
matched_generated_shape = [8,2,56,6,40]
matched_mse_mean = 0.039382558315992355
matched_rmse = 0.19845038652420044
matched_mae = 0.0848858579993248
matched_best_rep_mse_mean = 0.03616931661963463
matched_best_rep_rmse_mean = 0.1804618090391159
```

同一批真实 future40 的 baseline：

```text
copy_last_obs_mse = 0.04473424702882767
copy_last_obs_rmse = 0.211504727602005
copy_last_obs_mae = 0.06692911684513092
zero_mse = 0.17028947174549103
zero_rmse = 0.41266146302223206
zero_mae = 0.24624601006507874
```

## 判断

直接看距离：

```text
source-label matched generation 的 MSE/RMSE 略优于复制最后一帧 baseline。
但 MAE 明显差于复制最后一帧 baseline。
```

因此不能说“差得很小”。更准确表述：

```text
当前模型生成结果在数值尺度上接近真实数据分布，且比 zero baseline 好很多；
在真实标签匹配的小样本 probe 上，平方误差略优于 copy-last baseline；
但逐元素平均误差仍偏大，且尚未证明比简单 baseline 稳定更好。
```

后续若要做正式数值结论，需要在整个 test split 上按真实 label 生成，再报告 ADE/MSE/MAE 或现有 forecasting metrics，而不是只用 8 个 case 的 probe。
