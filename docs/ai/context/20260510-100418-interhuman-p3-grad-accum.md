# InterHuman-AS P3 梯度累积实施记录

## 目标

为单张 RTX 3080 训练增加梯度累积能力，使显存 batch size 和 effective batch size 分离。

## 代码改动

新增训练参数：

```text
--grad_accum_steps
```

位置：

```text
utils/parser_util.py
```

训练逻辑：

```text
train/training_loop.py
```

语义：

```text
batch_size: 实际 GPU batch
grad_accum_steps: 每次 optimizer step 前累积的 forward/backward 次数
num_steps: optimizer steps
effective_batch_size = batch_size * world_size * grad_accum_steps
```

实现要点：

- `grad_accum_steps=1` 保持原行为。
- 每次 backward 的 loss 按 `1 / grad_accum_steps` 缩放。
- DDP 下非同步累积 batch 使用 `no_sync()`。
- checkpoint 文件名仍以 optimizer step 为准。
- `args.json` 会保存 `grad_accum_steps`。
- logger 记录 `data_step` 和 `effective_batch_size`。

## 验证 1：兼容原行为

命令要点：

```text
save_dir: save/interhuman/p3_grad_accum_1_smoke
num_steps: 2
grad_accum_steps: 1
max_samples: 2
```

结果：

```text
step[0]: loss[0.67435]
step[1]: loss[0.62534]
```

验证：

```text
args.json grad_accum_steps = 1
model000000002.pt 可加载，50 keys
opt000000002.pt 可加载，2 keys
```

## 验证 2：两步累积

命令要点：

```text
save_dir: save/interhuman/p3_grad_accum_2_smoke
num_steps: 2
grad_accum_steps: 2
max_samples: 2
```

结果：

```text
Starting epoch 0:2
step[0]: loss[0.64726]
Starting epoch 1:2
step[1]: loss[0.62225]
```

说明：

`max_samples=2` 时 dataloader 每个 epoch 只有 2 个 batch。`num_steps=2` 且 `grad_accum_steps=2` 需要 4 个 data batch，因此正确跨 2 个 epoch 完成 2 个 optimizer steps。

验证：

```text
args.json grad_accum_steps = 2
model000000002.pt 可加载，50 keys
opt000000002.pt 可加载，2 keys
```

## 下一步

可进入单卡 baseline：

```text
batch_size: 1
grad_accum_steps: 16-64
layers: 4
latent_dim: 256
num_steps: 50k
```

在长训前建议先用：

```text
num_steps: 1000
grad_accum_steps: 16
```

验证显存、速度和 checkpoint 命名。
