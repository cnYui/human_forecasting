# P9 T2P InterHuman XYZ 启动结果

## 范围

本次按已确认边界启动 P9：

```text
model_type: t2p_interhuman_xyz
dataset: InterHuman
space: xyz [B,T,2,24,3]
protocol: deterministic, num_modes=1
obs/pred: 30/120
```

计划文档：

```text
docs/ai/context/20260615-132220-p9-t2p-interhuman-xyz-plan.md
```

## 已实现

新增：

```text
model/forecasting_t2p.py
```

修改：

```text
model/forecasting_xyz.py
train/train_forecasting_xyz.py
```

实现内容：

- 新增 `T2PInterHumanXYZ`。
- 接入 `XYZ_FORECASTING_MODEL_TYPES`。
- 支持 checkpoint config 恢复。
- 训练脚本新增：
  - `--t2p_root_loss_weight`
  - `--t2p_local_loss_weight`

第一版结构：

```text
obs_xyz
-> root_encoder + local_encoder
-> two-person transformer interaction
-> root trajectory decoder
-> trajectory-conditioned local pose decoder
-> pred_xyz
```

## 已验证

语法检查通过：

```text
python -m py_compile model/forecasting_t2p.py model/forecasting_xyz.py train/train_forecasting_xyz.py eval/eval_forecasting_xyz.py
```

最小模型检查通过：

```text
pred: (2, 120, 2, 24, 3)
config restore: t2p_interhuman_xyz
```

dataset smoke 通过：

```text
save/forecasting/interhuman/p9_t2p_smoke_xyz_dataset/
```

metrics sanity 通过：

```text
save/forecasting/interhuman/p9_t2p_smoke_metrics/
pred == target 时 xyz metrics 全 0
```

2-step train smoke 通过：

```text
save/forecasting/interhuman/p9_t2p_interhuman_xyz_2step_smoke/model000000002.pt
```

2-step test eval 指标 finite。

## 训练启动

第一次尝试：

```text
save/forecasting/interhuman/p9_t2p_interhuman_xyz_h256_l2_s0_5000
batch_size=32
```

失败原因：

```text
CUDA out of memory
```

当时 GPU 上已有 JRT 训练进程占用约 `3444 MiB`，T2P batch32 在 active->xyz 转换处额外显存不足。

正式启动配置改为：

```text
save_dir: save/forecasting/interhuman/p9_t2p_interhuman_xyz_h256_l2_b8a4_s0_5000
model_type: t2p_interhuman_xyz
hidden_dim: 256
num_layers: 2
num_heads: 8
batch_size: 8
grad_accum_steps: 4
effective_batch_size: 32
num_steps: 5000
save_interval: 1000
eval_interval: 1000
seed: 0
lr: 3e-4
weight_decay: 1e-4
num_workers: 0
```

启动命令使用 `setsid` 后台运行，日志和 PID：

```text
save/forecasting/interhuman/p9_t2p_interhuman_xyz_h256_l2_b8a4_s0_5000/train.log
save/forecasting/interhuman/p9_t2p_interhuman_xyz_h256_l2_b8a4_s0_5000/train.pid
```

当前进程：

```text
wrapper PID: 714421
python PID: 714422
```

早期日志：

```text
Training joint-space forecasting model...
model_type=t2p_interhuman_xyz params=5593536 device=cuda effective_batch_size=32
step[1]: train_loss[0.102989]
step[50]: train_loss[0.061938]
```

同时运行的已知 GPU 进程：

```text
JRT python PID: 712541, used_memory≈3444 MiB
T2P python PID: 714422, used_memory≈3326 MiB
```

## 监控命令

```bash
tail -f save/forecasting/interhuman/p9_t2p_interhuman_xyz_h256_l2_b8a4_s0_5000/train.log
```

```bash
ps -p $(cat save/forecasting/interhuman/p9_t2p_interhuman_xyz_h256_l2_b8a4_s0_5000/train.pid) -o pid,stat,etime,args
```

```bash
nvidia-smi
```

## 下一步

等待：

```text
model000001000.pt
opt000001000.pt
metrics_val.json
```

写出后检查 val/test 指标是否 finite。若 seed0 完成并不明显失败，再决定是否启动 seeds `1,2`。
