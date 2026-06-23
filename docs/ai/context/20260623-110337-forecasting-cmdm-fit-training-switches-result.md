# ForecastingCMDM 拟合修复训练开关实现结果

## 目的

阶段 6D 距离评估确认：

```text
one_step_t999 在全 test split 上 MSE 优于 copy-last；
DDIM50 free sampling 在 256 样本上只略优于 copy-last，MAE 更差。
```

因此训练侧需要更直接强化：

```text
从高噪声 / 纯噪声到真实 future40 的预测能力。
```

## 代码改动

修改：

```text
train/train_label_forecasting_diffusion.py
```

新增训练参数：

```text
--timestep_sampling uniform|high_noise
--high_noise_min_t 750
--one_step_noise_prob 0.0
--one_step_t 999
```

默认行为：

```text
timestep_sampling = uniform
one_step_noise_prob = 0.0
```

因此旧阶段 6 命令行为不变。

## 新训练分支

### high_noise timestep sampling

当：

```text
--timestep_sampling high_noise
```

训练使用：

```text
t ~ Uniform(high_noise_min_t, diffusion_steps - 1)
x_t = q_sample(real_future40, t)
loss = MSE(model(x_t,t,obs/action), real_future40)
```

目的：

```text
减少低噪声 timestep 对训练的主导，强化接近采样起点的还原能力。
```

### one_step pure-noise branch

当：

```text
--one_step_noise_prob p
```

每个训练 batch 以概率 p 使用：

```text
x_t = random Gaussian noise
t = one_step_t
loss = MSE(model(x_t,t,obs/action), real_future40)
```

目的：

```text
让反向传播更直接服务于“从噪声生成 future40 更贴近真实 future40”。
```

## 日志新增

`train_log.jsonl` 新增：

```text
t_mean
one_step_noise_active
```

用于确认当前 step 是否进入 one-step 分支，以及 timestep 是否落在预期区间。

## 验证

语法检查：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python -m py_compile \
  train/train_label_forecasting_diffusion.py \
  eval/eval_label_forecasting_distance.py
```

通过。

### mixed high-noise + one-step smoke

输出：

```text
save/forecasting/ntu120_label/phase6d_train_high_noise_one_step_smoke
```

结果：

```text
step 1: train_loss=0.467939, t_mean=999.0, one_step_noise_active=1.0
step 2: train_loss=0.477212, t_mean=827.0, one_step_noise_active=0.0
checkpoint = model000000002.pt
```

说明：

```text
one-step 分支和 high-noise 分支均被覆盖。
```

### one-step only smoke

输出：

```text
save/forecasting/ntu120_label/phase6d_train_one_step_only_smoke
```

结果：

```text
step 1: train_loss=0.467939, t_mean=999.0, one_step_noise_active=1.0
checkpoint = model000000001.pt
```

### high-noise only smoke

输出：

```text
save/forecasting/ntu120_label/phase6d_train_high_noise_only_smoke
```

结果：

```text
step 1: train_loss=0.466146, t_mean=819.0, one_step_noise_active=0.0
checkpoint = model000000001.pt
```

## 后续正式训练建议

下一轮不要只加长原始 uniform 训练。

优先跑一个 fine-tune：

```text
从 model000005000.pt resume
timestep_sampling = high_noise
high_noise_min_t = 750
one_step_noise_prob = 0.25 或 0.5
one_step_t = 999
velocity_loss_weight = 0.0 起步
root_translation_loss_weight = 0.0 起步
```

如果 Phase 6D 距离指标改善，再逐步加入：

```text
velocity_loss_weight
root_translation_loss_weight
K-step final-sample loss
```
