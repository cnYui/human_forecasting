# InterHuman-AS P1/P2 实施记录

## 已完成

P1 离线预处理：

```text
preprocess/interhuman_as.py
```

已生成：

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5
dataset/interhuman/smpl/conditioned/meta.json
```

输出规模：

```text
train h5: 2.0G
val h5:   218M
test h5:  406M
```

样本数：

```text
train: listed 6022, written 6021, skipped 1
val:   listed 580,  written 580,  skipped 0
test:  listed 1177, written 1175, skipped 2
```

跳过原因：

```text
missing_actor_reactor_label
```

P2 H5 loader：

```text
data_loaders/a2m/interhuman.py
```

新增能力：

- `data_path` 指向 `dataset/interhuman/smpl/conditioned` 时读取 H5。
- `data_path` 指向 `interhuman_train.h5` 时从同目录解析 train/val/test H5。
- `data_path` 指向 `dataset/interhuman` 时保留在线 `.pkl` loader。
- H5 文件句柄在 worker 内懒加载，避免跨进程共享句柄。

## 校验

P1 对齐校验：

```text
train/10:    max_abs_diff_vs_online_loader = 0.0
train/3292:  max_abs_diff_vs_online_loader = 0.0
train/4476:  max_abs_diff_vs_online_loader = 0.0
val/1008:    max_abs_diff_vs_online_loader = 0.0
val/1024:    max_abs_diff_vs_online_loader = 0.0
test/1:      max_abs_diff_vs_online_loader = 0.0
test/100:    max_abs_diff_vs_online_loader = 0.0
test/1118:   max_abs_diff_vs_online_loader = 0.0
test/6539:   max_abs_diff_vs_online_loader = 0.0
```

P2 batch 校验：

```text
motion:  [B, 25, 6, 150]
cmotion: [B, 25, 6, 150]
finite:  True
```

`num_workers=8` 可取 batch：

```text
len: 6021
batch: [1, 25, 6, 150]
finite: True
```

最小训练 smoke：

```text
save_dir: save/interhuman/h5_loader_smoke
num_steps: 4
max_samples: 2
layers: 2
latent_dim: 128
batch_size: 1
lambda_orient/body/transl: 0
```

训练进入实际 batch，loss：

```text
step[0]: loss[0.67435]
step[1]: loss[0.62534]
```

## 注意事项

`num_steps=2` 时没有实际训练，因为当前 `TrainLoop` 使用：

```text
num_epochs = num_steps // (len(data) * world_size + 1)
```

当 `num_steps < len(data)+1` 时 `num_epochs=0`。后续 smoke 命令必须保证 `num_steps >= len(data)+1`，或者在训练循环中修正这个边界。

## 下一步

进入 P3 前建议先做一个 1000 step H5 full smoke：

```text
batch_size: 1
layers: 2
latent_dim: 128
num_frames: 150
num_steps: 1000
lambda_orient/body/transl: 0
```

如果 full smoke 稳定，再实现 `--grad_accum_steps`。
