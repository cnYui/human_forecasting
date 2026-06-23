# ForecastingCMDM high-noise / one-step 微调监控计划

## 目标

用户要求继续微调，并实时监控：

```text
generated_future40 vs real_future40 的 MSE / MAE 是否继续下降
```

本阶段不使用分类器作为主指标。

## 起点

从阶段 6 正式 checkpoint 继续：

```text
save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt
```

已知基线：

```text
one_step_t999 full test:
generated MSE = 0.0273966648
generated MAE = 0.0810767258
copy-last MSE = 0.0358492542
copy-last MAE = 0.0792285130

DDIM50 256 samples:
generated MSE = 0.0447964055
generated MAE = 0.0908588756
copy-last MSE = 0.0453326257
copy-last MAE = 0.0782121037
```

## 微调策略

使用已新增训练开关：

```text
--timestep_sampling high_noise
--high_noise_min_t 750
--one_step_noise_prob 0.5
--one_step_t 999
```

理由：

```text
阶段 6D 显示 one-step high-noise 预测比 DDIM50 free sampling 更贴近真实 future。
微调应优先强化从高噪声/纯噪声还原真实 future40 的能力。
```

## 训练方式

采用分段训练，不一次跑很长：

```text
5000 -> 5100
5100 -> 5200
5200 -> 5300
...
```

每段结束后保存 checkpoint 并评估。

## 监控方式

每段 checkpoint 后运行：

```text
eval/eval_label_forecasting_distance.py
```

第一优先监控：

```text
one_step_t999 full test MSE / MAE
```

第二优先监控：

```text
DDIM50 256 samples MSE / MAE
```

如果 one-step MSE 下降但 DDIM50 不降，说明预测能力继续增强，但多步采样仍是瓶颈。

如果 DDIM50 MSE/MAE 也下降，说明微调开始真正改善最终生成结果。

## 停止/调整条件

继续当前微调：

```text
one_step 或 DDIM50 的 MSE 持续下降，且 MAE 不明显恶化。
```

调整参数：

```text
MSE 不降或 MAE 明显上升。
```

可调整方向：

```text
one_step_noise_prob: 0.5 -> 0.25
high_noise_min_t: 750 -> 500
开启 velocity_loss_weight
开启 root_translation_loss_weight
```

停止当前路线：

```text
连续两个阶段 DDIM50 MSE/MAE 都不改善。
```
