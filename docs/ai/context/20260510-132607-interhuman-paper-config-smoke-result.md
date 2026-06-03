# InterHuman-AS 论文配置小步 smoke 结果

## 目标

验证更接近论文训练设置的模型规模和显式 loss 是否能在单张 RTX 3080 上运行。

## 配置

```text
dataset = interhuman
data_path = dataset/interhuman/smpl/conditioned
setting = cmdm
arch = online
unconstrained = true
body_model = smpl
pose_rep = rot6d
num_frames = 150
layers = 8
latent_dim = 512
lambda_orient = 1
lambda_body = 1
lambda_transl = 1
batch_size = 1
grad_accum_steps = 4
num_steps = 100
```

## 首次运行问题

首次运行输出从 `step[0]` 开始为：

```text
loss[nan]
```

原因：

`diffusion/gaussian_diffusion.py` 中 explicit translation loss 仍硬编码读取：

```text
[:, 55:56, 0:3, :]
```

这适用于旧的 56-slot 表示，但当前 InterHuman H5 是 `[25, 6, T]`，translation 在最后一个 slot，也就是 index `24`。`55:56` 得到空 tensor，`masked_l2()` 中分母为 0，导致 NaN。

修复：

```text
cmotion[:, -1:, 0:3, :]
target[:, -1:, 0:3, :]
model_output[:, -1:, 0:3, :]
```

该写法同时兼容 56-slot 和 25-slot 表示，因为两者都把 root translation 放在最后一个 slot。

## 修复后验证

输出目录：

```text
save/interhuman/paper_config_l8_d512_loss_smoke_fixed
```

结果：

```text
退出码 = 0
耗时 = 0:40.76
最大 CPU RSS = 4216292 KB
训练中观测显存约 2.4GB
退出后显存 = 137 MiB
```

loss 记录：

```text
step[0]:  loss[0.70100]
step[10]: loss[0.20821]
step[20]: loss[0.15965]
step[30]: loss[0.13664]
step[40]: loss[0.12742]
step[50]: loss[0.12056]
step[60]: loss[0.11777]
step[70]: loss[0.11527]
step[80]: loss[0.11294]
step[90]: loss[0.11064]
```

checkpoint：

```text
model000000000.pt: OrderedDict, 158 keys
model000000050.pt: OrderedDict, 158 keys
model000000100.pt: OrderedDict, 158 keys
opt000000100.pt: dict, 2 keys
```

进程状态：

```text
无残留 train_mdm / micromamba run 进程
无残留 GPU compute app
```

## 结论

论文模型规模 `layers=8, latent_dim=512` 和显式 loss 在 RTX 3080 上可运行。当前还不是论文训练，只是小步 smoke。

下一步建议：

```text
layers = 8
latent_dim = 512
batch_size = 1
grad_accum_steps = 64
num_steps = 1000
lambda_orient/body/transl = 1
```

该配置用于验证单卡 effective batch size 64 的论文 batch size 近似口径。
