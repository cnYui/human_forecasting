# JRT XYZ Seed0 正式训练启动记录

## 目标

启动 `jrt_xyz` 在 InterHuman 150/30/120 joint-space 协议下的 seed0 正式训练。

## 启动前状态

GPU：

```text
NVIDIA GeForce RTX 3080
memory.used=137 MiB / 10240 MiB
utilization=0%
```

未发现正在运行的 `train_forecasting_xyz.py` / `jrt_xyz` 训练进程。

## 配置

```text
dataset=interhuman
data_path=dataset/interhuman/smpl/conditioned
model_type=jrt_xyz
save_dir=save/forecasting/interhuman/p9_jrt_xyz_h256_l4_s0_5000
window_len=150
obs_len=30
pred_len=120
hidden_dim=256
num_heads=8
num_layers=4
dropout=0.1
batch_size=8
grad_accum_steps=2
effective_batch_size=16
eval_batch_size=8
num_steps=5000
save_interval=1000
eval_interval=1000
aux_weight=0.2
jrt_relation_weight=0.5
lr=1e-4
weight_decay=1e-4
num_workers=0
seed=0
```

## 取舍

- `batch_size=8, grad_accum_steps=2` 是保守选择；JRT relation stream 有 `[B,48,48,hidden_dim]` 中间张量，显存风险高于 lite SoMoFormer。
- `jrt_relation_weight=0.5` 是 smoke 后的第一版正式值，避免 relation L1 在训练初期压过 xyz pose MSE。
- 本次只启动 seed0；3-seed 需要等 seed0 指标和显存稳定后再决定。

## 输出

```text
save/forecasting/interhuman/p9_jrt_xyz_h256_l4_s0_5000/train.log
save/forecasting/interhuman/p9_jrt_xyz_h256_l4_s0_5000/train.pid
```

## 实际启动状态

第一次 `nohup` wrapper 进程立即退出且 `train.log` 为空；前台 1-step launch check 证明代码和配置正常。随后改用 `setsid bash -c ...` 显式 `cd` 到项目根目录后启动成功。

当前运行信息：

```text
wrapper PID: 712537
micromamba PID: 712539
python PID: 712541
train.pid: save/forecasting/interhuman/p9_jrt_xyz_h256_l4_s0_5000/train.pid
```

早期检查：

```text
GPU memory: 3586 MiB / 10240 MiB
GPU utilization: 88%
latest checked step: 114
latest checked train_loss: 0.1305667869746685
```

注意：`train.log` 因 Python stdout 缓冲暂时为空；训练进度应优先查看：

```text
save/forecasting/interhuman/p9_jrt_xyz_h256_l4_s0_5000/train_log.jsonl
```
