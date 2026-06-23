# ForecastingCMDM 生成拟合问题修复计划

## 用户目标

用户明确要求：

```text
最终目标是通过反向传播学习，让最后生成出来的 future40 和训练/测试数据里的真实 future40 更贴合。
分类器只能作为辅助检查，不能替代 generated vs real 的直接距离判断。
```

因此后续主指标必须回到：

```text
generated_future40 vs real_future40
```

而不是：

```text
generated_future40 -> action classifier -> 是否命中条件标签
```

## 当前问题定位

代码核对后，阶段 6 训练目标确实通过反向传播拟合真实 future40：

```text
x_t = q_sample(real_future40, t)
pred_xstart = model(x_t, t, obs20/action)
loss = MSE(pred_xstart, real_future40)
```

但最终采样使用的是：

```text
x_T = random noise
多步 reverse sampling
generated_future40 = sample_loop(model, x_T, obs20/action)
```

这里没有直接对最终 `generated_future40` 和 `real_future40` 的距离做反向传播。

阶段 6 拟合诊断显示：

```text
teacher-forced denoising 明显优于 copy-last
free DDIM50 / p_sample_loop sampling 只接近或弱于 copy-last
one-step pure-noise t999 pred_xstart 反而比 DDIM50 free sampling 更好
```

所以当前主要问题不是“模型完全没学”，而是：

```text
训练分布内的 denoising 学到了；
自由采样链路存在误差累积；
训练目标和最终使用目标没有完全对齐。
```

## 不能继续做的事

不应继续把以下内容当主结论：

```text
generated consistency_acc
label swap 非完全相同
分类器 top1/top5
```

这些只能说明辅助现象，不能回答“生成帧和真实帧差得多不多”。

## 下一步最小修复

先做 Phase 6D：正式 true-label distance evaluation。

新增入口：

```text
eval/eval_label_forecasting_distance.py
```

功能：

```text
1. 加载阶段 6 checkpoint。
2. 遍历 test split，使用每条样本自己的真实 source action label。
3. 生成 future40。
4. 直接计算 generated_future40 vs real_future40。
5. 同时报告 copy-last / zero baseline。
6. 支持 ddim50、p_sample_loop、one_step_t999 三种模式。
7. 输出整体指标和 per-action 指标。
```

核心输出：

```text
metrics.json
per_action_metrics.json
metadata.json
```

主指标：

```text
MSE
RMSE
MAE
copy-last MSE/RMSE/MAE
zero MSE/RMSE/MAE
generated_vs_copy_last_delta
```

原因：

```text
先把“生成和真实差多少”变成正式、可复现、全 test split 或可控 max_samples 的评估。
否则直接改训练会缺少判断依据。
```

## 训练侧修复路线

如果 Phase 6D 确认 free sampling 明显不贴合，优先按下面顺序改训练。

### 1. high-noise timestep sampling

新增训练配置：

```text
--timestep_sampling uniform|high_noise
--high_noise_min_t 750
```

让训练更常看到接近采样起点的高噪声输入，强化从高噪声还原真实 future40。

### 2. one-step pure-noise 预测分支

新增训练配置：

```text
--one_step_noise_prob p
```

以一定概率直接：

```text
x_t = random noise
t = 999
model(x_t, t, obs/action) -> real_future40
```

这是当前诊断里最贴近“从噪声生成结果”的便宜修复。

### 3. velocity / root loss

打开已有损失：

```text
--velocity_loss_weight
--root_translation_loss_weight
```

原因：

```text
逐元素 rot MSE 不保证视频运动连续，也不保证 root 轨迹更像真实动作。
```

### 4. K-step final-sample loss

更根本但更贵的方案：

```text
unroll K-step DDIM sampling
loss = MSE(final_generated_future40, real_future40)
backprop through sampling steps
```

这才是严格优化“最终生成帧更贴近真实帧”的目标。

第一版 K 可从 5 或 10 开始，3080 上需要小 batch 和 grad accumulation。

## 本次执行范围

本次先实现 Phase 6D 距离评估，不直接启动长训练。

理由：

```text
阶段 6B 是 label swap，不是标准 forecasting distance eval。
已有 probe 样本数有限，不能作为正式结论。
先补评估入口，才能判断后续训练改法是否真的让生成帧更贴近。
```
