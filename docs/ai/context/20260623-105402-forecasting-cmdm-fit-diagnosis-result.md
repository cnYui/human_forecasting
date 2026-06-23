# ForecastingCMDM 生成拟合问题诊断结果

## 结论

当前问题不是“没有通过反向传播学习拟合真实 future40”。

更准确的定位是：

```text
训练目标学到了一定 teacher-forced denoising / xstart prediction 能力；
但最终多步 free sampling 生成误差更高，说明训练目标和最终生成过程存在 mismatch。
```

也就是说：

```text
模型在看到 q_sample(real_future, t) 这种训练分布内的 noised future 时，能较好还原真实 future；
但从纯噪声开始反复 reverse sampling 时，x_t 会进入模型没被显式约束的自由运行分布，误差沿采样链累积。
```

## 代码协议检查

训练入口：

```text
train/train_label_forecasting_diffusion.py
```

当前训练核心：

```text
future, y = batch
t ~ Uniform(0,999)
x_t = q_sample(future, t)
pred = model(x_t, t, y)
loss = MSE(pred, future)
```

这确实是在通过反向传播拟合真实 future40。

采样入口：

```text
sample/sample_label_forecasting_diffusion.py
```

当前采样核心：

```text
x_T ~ N(0,I)
for t in reverse timesteps:
  pred_xstart = model(x_t, t, obs/action)
  x_{t-1} = diffusion posterior / DDIM update
```

这里没有直接对最终 `generated_future40` 和真实 future40 做反向传播。

## Teacher-forced denoising 诊断

使用：

```text
checkpoint = save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt
test split = 1253 samples
```

对真实 future40 加噪：

```text
x_t = q_sample(real_future40, t)
model(x_t, t, obs/action) -> pred_xstart
```

结果：

```text
t=0   MSE=0.012395 RMSE=0.111334 MAE=0.065217
t=10  MSE=0.012337 RMSE=0.111071 MAE=0.065033
t=50  MSE=0.012351 RMSE=0.111135 MAE=0.064975
t=100 MSE=0.012444 RMSE=0.111553 MAE=0.065234
t=250 MSE=0.012861 RMSE=0.113408 MAE=0.066424
t=500 MSE=0.014634 RMSE=0.120971 MAE=0.070464
t=750 MSE=0.018491 RMSE=0.135982 MAE=0.075447
t=999 MSE=0.027295 RMSE=0.165211 MAE=0.081163
random t MSE=0.016396 RMSE=0.128046 MAE=0.071098
```

同一 test split baseline：

```text
copy-last MSE=0.035849 RMSE=0.189339 MAE=0.079229
zero      MSE=0.178506 RMSE=0.422499 MAE=0.253798
```

判断：

```text
teacher-forced denoising 明显优于 zero，也优于 copy-last 的 MSE/RMSE。
但 t 越高误差越大，高噪声条件下拟合仍不足。
```

## Free sampling 诊断

### DDIM50 source-label free sampling

使用前 256 个 test samples，输入真实 source label，DDIM50 从纯噪声自由生成。

结果：

```text
free_ddim50_source_label MSE=0.044811 RMSE=0.211687 MAE=0.091538
copy-last                MSE=0.045333 RMSE=0.212915 MAE=0.078212
```

判断：

```text
DDIM50 free sampling 的 MSE 只比 copy-last 略好，MAE 更差。
```

### 1000-step p_sample_loop source-label free sampling

使用前 32 个 test samples，输入真实 source label，完整 1000-step p_sample_loop。

结果：

```text
free_p_sample_1000_source_label MSE=0.034422 RMSE=0.185532 MAE=0.085683
copy-last                       MSE=0.031725 RMSE=0.178117 MAE=0.063402
```

判断：

```text
完整 1000-step p_sample_loop 没有解决问题，至少在小样本上仍弱于 copy-last。
```

### One-step pure noise xstart prediction

使用前 256 个 test samples，直接：

```text
x = pure Gaussian noise
t = 999
pred = model(x, t, obs/action)
```

结果：

```text
one_step_pure_noise_t999 MSE=0.033510 RMSE=0.183058 MAE=0.081307
copy-last                MSE=0.045333 RMSE=0.212915 MAE=0.078212
```

判断：

```text
单步 high-noise xstart prediction 的 MSE 明显优于 copy-last，也优于 DDIM50 free sampling。
这强烈说明多步 reverse sampling 本身在当前模型上会带来误差累积。
```

## 问题定位

当前主要瓶颈不是分类器，也不是数据完全坏掉，而是以下几点。

### 1. 训练目标和最终使用方式不一致

训练优化的是：

```text
model(q_sample(real_future, t), t, obs/action) -> real_future
```

最终使用的是：

```text
x_T ~ noise
reverse sample 多步迭代 -> generated_future
```

模型从未直接对“多步采样后的最终 generated_future 和 real_future 的距离”做反向传播。

### 2. Free sampling 存在 off-policy / exposure bias

训练时每个 `x_t` 都来自真实 future 的正向扩散分布。

采样时 `x_t` 来自模型上一步输出和扩散 posterior；一旦某一步 pred_xstart 有偏差，后续输入就偏离训练分布，误差会累积。

### 3. 当前 loss 过于单一

阶段 6 只启用了：

```text
rot_mse
velocity_loss_weight = 0
root_translation_loss_weight = 0
relative_root_loss_weight = 0
```

这能降低逐元素 MSE，但不保证轨迹连续性、速度、root 运动和两人关系更贴合。

### 4. 当前阶段没有专门的 true-label forecasting eval

6B 是 label swap，不是标准 forecasting error evaluation。

后面补跑的小 probe 只覆盖 8 或 256 个样本，不能作为正式 test split 结论。

## 改进路线

优先级从高到低：

### A. 先补正式 true-label 距离评估

新增一个评估入口：

```text
eval/eval_label_forecasting_distance.py
```

功能：

```text
对整个 test split 使用真实 source label 生成 future40
输出 generated vs real 的 MSE/RMSE/MAE
输出 copy-last / zero baseline
支持 DDIM50、p_sample_loop、one-step t999 三种模式
```

原因：

```text
先把“生成和真实差多少”变成正式指标，避免继续用 label swap 或分类器间接判断。
```

### B. 加 one-step / high-noise xstart 训练或评估分支

当前 one-step pure-noise t999 的 MSE 比 DDIM50 free sampling 更好。

可以新增训练配置：

```text
--timestep_sampling high_noise
--high_noise_min_t 750
```

或更直接：

```text
x_t = pure noise
t = 999
model(x_t,t,obs/action) -> future
loss = MSE(pred, future)
```

原因：

```text
如果最终希望生成更像真实 future，应该强化模型从高噪声/纯噪声直接预测 future 的能力，而不是让低噪声样本主导训练。
```

### C. 加 final-sample loss / sampling distillation

更严格地对齐最终目标：

```text
unroll K-step DDIM sampling
loss = MSE(final_generated, real_future)
backprop through K steps
```

K 可先从很小开始：

```text
K = 5 或 10
```

原因：

```text
这直接优化最终生成结果和真实 future 的距离，解决 teacher-forced vs free-running mismatch。
```

代价：

```text
显存和时间会明显增加，3080 上需要小 batch/grad accumulation。
```

### D. 加速度和 root loss

下一版训练可以开启：

```text
velocity_loss_weight > 0
root_translation_loss_weight > 0
```

先不要开 `relative_root_loss_weight`，因为 root slot 仍需确认。

原因：

```text
仅逐元素 rot MSE 可能让整体数值接近，但轨迹动态不贴合。
速度/root loss 更接近“视频看起来是否拟合真实动作”。
```

### E. 训练更久不是第一优先级

5000 step 不是很长，但当前诊断显示：

```text
teacher-forced 已明显优于 copy-last
free sampling 反而退化
```

所以单纯加长训练可能有帮助，但不是最根本的修复。应先对齐训练目标和最终生成目标。

## 当前建议的下一步

先做一个小而确定的阶段：

```text
Phase 6D: true-label distance eval + one-step/high-noise generation baseline
```

通过后再决定是否：

```text
1. fine-tune high-noise weighted training
2. 加 velocity/root loss
3. 做 K-step final-sample loss
```
