# InterHuman-AS 论文 batch size 近似 smoke 计划

## 目的

验证单张 RTX 3080 是否能用梯度累积近似论文训练 batch size。

论文训练口径：

```text
batch_size = 64
layers = 8
latent_dim = 512
lambda_inter = 1
num_steps = 500000
```

本次 smoke：

```text
batch_size = 1
grad_accum_steps = 64
effective_batch_size = 64
layers = 8
latent_dim = 512
lambda_orient/body/transl = 1
num_steps = 1000
```

说明：

- `num_steps` 表示 optimizer steps。
- 本次实际需要约 `1000 * 64 = 64000` 个 forward/backward data batch。
- 本次仍不是论文长训，只验证 batch size 近似口径、速度、checkpoint 和自然退出。

## 命令要点

```text
save_dir = save/interhuman/paper_config_l8_d512_accum64_1000_smoke
save_interval = 500
log_interval = 100
```

## 预估

基于 `grad_accum_steps=4, num_steps=100` 的耗时 `40.76s`，本次粗估：

```text
40.76s * (64000 / 400) = 6521.6s = 108.7min
```

实际速度可能因 checkpoint I/O 和调度波动变化。

## 验收

- 训练达到 `num_steps=1000`，退出码 0。
- loss 有限。
- checkpoint `model000001000.pt` 和 `opt000001000.pt` 可加载。
- 无残留训练进程和 GPU compute app。
