# InterHuman-AS 50K baseline 训练计划

## 目标

进入论文配置口径的更长 baseline 训练，验证在单张 RTX 3080 上：

- 论文模型尺寸能否长时间稳定训练。
- 有效 batch size 64 的梯度累积是否能稳定保存和恢复。
- loss 是否在 50K optimizer steps 内保持有限且无明显训练崩溃。

本阶段仍不是论文 1:1 复现，因为论文训练为 500K steps，且 Table 4 还缺 ST-GCN recognition evaluator、InterHuman 类别标签来源、DDIM-5 生成流程和 SMPL-X 口径确认。

## 训练配置

```text
dataset = interhuman
data_path = dataset/interhuman/smpl/conditioned
body_model = smpl
pose_rep = rot6d
num_person = 2
num_frames = 150
layers = 8
latent_dim = 512
batch_size = 1
grad_accum_steps = 64
effective_batch_size = 64
num_steps = 50000
save_interval = 5000
log_interval = 100
lambda_orient = 1
lambda_body = 1
lambda_transl = 1
```

保存目录：

```text
save/interhuman/paper_config_l8_d512_accum64_50000_baseline
```

日志：

```text
save/interhuman/paper_config_l8_d512_accum64_50000_baseline/train.log
```

## 预估

上一阶段 `num_steps=1000, grad_accum_steps=64` 实测耗时：

```text
1:16:09 = 4569s
```

50K baseline 粗估：

```text
4569s * 50 = 228450s = 63.46h = 2.64 days
```

checkpoint 大小约：

```text
model checkpoint ~= 111MB
optimizer checkpoint ~= 203MB
每次保存合计 ~= 314MB
```

`save_interval=5000` 时预计保存 `0, 5000, ..., 50000` 共 11 组，约 `3.5GB`，当前磁盘空间足够。

## 启动方式

由于训练预计超过 2 天，本次用 detached 进程启动，不在交互 session 里等待完整训练结束。

核心命令：

```text
/usr/bin/time -v micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
  python -m train.train_mdm \
  --setting cmdm \
  --save_dir save/interhuman/paper_config_l8_d512_accum64_50000_baseline \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --cond_mask_prob 0 \
  --num_person 2 \
  --layers 8 \
  --latent_dim 512 \
  --num_frames 150 \
  --arch online \
  --overwrite \
  --pose_rep rot6d \
  --body_model smpl \
  --train_platform_type NoPlatform \
  --unconstrained \
  --batch_size 1 \
  --grad_accum_steps 64 \
  --num_steps 50000 \
  --save_interval 5000 \
  --log_interval 100 \
  --lambda_orient 1 \
  --lambda_body 1 \
  --lambda_transl 1 \
  --max_samples -1
```

## 验收

- 后台进程启动并进入训练 loop。
- `args.json`、`model000000000.pt`、`opt000000000.pt` 写出。
- 初始 checkpoint 可加载。
- `step[0]` 或早期 loss 为有限值。
- GPU 显存保持在 10GB 内。
- 中途保存 `model000005000.pt` 后再记录一次状态。

## 风险

- 单卡预计训练 2.6 天，期间 GPU 被长时间占用。
- 若系统重启或进程被杀，需要从最近的 `model*.pt` 和 `opt*.pt` 恢复。
- 当前训练日志使用 tqdm，长日志较大但可接受。
