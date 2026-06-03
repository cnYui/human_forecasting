# InterHuman-AS 论文配置小步 smoke 计划

## 目的

在进入长训前，先验证论文模型规模和显式 loss 在单张 RTX 3080 上能否运行。

## 对齐论文的配置

```text
dataset = interhuman
setting = online, unconstrained
num_frames = 150
layers = 8
latent_dim = 512
lambda_orient = 1
lambda_body = 1
lambda_transl = 1
```

## 本次不对齐论文的配置

```text
num_steps = 100
batch_size = 1
grad_accum_steps = 4
```

原因：

- 本次目标是验证显存、显式 loss 和 checkpoint，不是产出论文指标。
- 论文训练为 `batch_size=64, num_steps=500000`，单卡 3080 需要先逐级放大。

## 命令要点

```text
save_dir = save/interhuman/paper_config_l8_d512_loss_smoke
data_path = dataset/interhuman/smpl/conditioned
save_interval = 50
log_interval = 10
```

## 验收

- 训练达到 `num_steps=100`，退出码 0。
- loss 有限，不出现 OOM。
- checkpoint `model000000100.pt` 和 `opt000000100.pt` 可加载。
- 记录显存峰值和耗时。

## 后续门槛

通过后再运行：

```text
layers = 8
latent_dim = 512
batch_size = 1
grad_accum_steps = 64
num_steps = 1000
```

用于验证论文 effective batch size 64 的单卡可行性。
