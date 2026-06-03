# InterHuman-AS P3 grad_accum=16 中等 smoke 记录

## 目的

验证 P3 梯度累积在更接近单卡 baseline 的设置下是否稳定：

```text
num_steps = 1000
grad_accum_steps = 16
batch_size = 1
layers = 2
latent_dim = 128
```

这里的 `num_steps` 仍表示 optimizer steps，因此本次训练实际执行约：

```text
1000 * 16 = 16000
```

个 forward/backward data batch。

## 命令

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m train.train_mdm \
  --setting cmdm \
  --save_dir save/interhuman/p3_grad_accum_16_1000_smoke \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --cond_mask_prob 0 \
  --num_person 2 \
  --layers 2 \
  --latent_dim 128 \
  --num_frames 150 \
  --arch online \
  --overwrite \
  --pose_rep rot6d \
  --body_model smpl \
  --train_platform_type NoPlatform \
  --unconstrained \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --num_steps 1000 \
  --save_interval 500 \
  --log_interval 100 \
  --lambda_orient 0 \
  --lambda_body 0 \
  --lambda_transl 0 \
  --max_samples -1
```

## 结果

训练自然达到 `num_steps=1000` 并退出，退出码为 0。

耗时：

```text
7:05.02
```

资源：

```text
GPU: RTX 3080 10GB
训练中显存: 约 1572 MiB
退出后显存: 137 MiB
Maximum resident set size: 4155348 KB
```

loss 记录：

```text
step[0]:   loss[0.65583]
step[100]: loss[0.14353]
step[200]: loss[0.11022]
step[300]: loss[0.09788]
step[400]: loss[0.09039]
step[500]: loss[0.08481]
step[900]: loss[0.07124]
```

说明：

- 输出日志中 `tqdm` 的百分比是当前 epoch 内 data batch 进度，不是 optimizer step 进度。
- 本次跨 3 个 epoch，符合 `ceil(16000 / 6021) = 3`。
- `save_interval=500` 正常写出 0、500、1000 三组 checkpoint。

## 输出文件

```text
save/interhuman/p3_grad_accum_16_1000_smoke/args.json
save/interhuman/p3_grad_accum_16_1000_smoke/model000000000.pt
save/interhuman/p3_grad_accum_16_1000_smoke/model000000500.pt
save/interhuman/p3_grad_accum_16_1000_smoke/model000001000.pt
save/interhuman/p3_grad_accum_16_1000_smoke/opt000000000.pt
save/interhuman/p3_grad_accum_16_1000_smoke/opt000000500.pt
save/interhuman/p3_grad_accum_16_1000_smoke/opt000001000.pt
```

checkpoint 加载验证：

```text
args.json grad_accum_steps = 16
args.json batch_size = 1
args.json num_steps = 1000
model000000000.pt: OrderedDict, 50 keys
model000000500.pt: OrderedDict, 50 keys
model000001000.pt: OrderedDict, 50 keys
opt000001000.pt: dict, 2 keys
```

进程状态：

```text
无残留 train_mdm / micromamba run 进程
无残留 GPU compute app
```

## 结论

P3 梯度累积路径可用于后续单卡 baseline。建议下一步进入：

```text
layers = 4
latent_dim = 256
batch_size = 1
grad_accum_steps = 16 或 32
num_steps = 50000
```

如显存仍宽松，可以先试 `grad_accum_steps=32`；如果训练时间过长，先用 `grad_accum_steps=16` 产出可对比 baseline。
