# InterHuman-AS 50K baseline 已启动

## 状态

已启动 detached 长训进程。

```text
save_dir = save/interhuman/paper_config_l8_d512_accum64_50000_baseline
log = save/interhuman/paper_config_l8_d512_accum64_50000_baseline/train.log
pid_file = save/interhuman/paper_config_l8_d512_accum64_50000_baseline/train.pid
launcher_pid = 150767
start_time = 2026-05-10T19:36:21+09:00
```

启动方式改为 `setsid -f`，原因是本环境会清理普通后台子进程，`nohup ... &` 无法保活训练。

## 启动验收

训练已进入 loop：

```text
Starting epoch 0:532
step[0]: loss[0.67129]
saving model...
```

GPU 状态：

```text
memory.used = 2396 MiB
memory.total = 10240 MiB
utilization.gpu = 43%
power.draw = 103.31 W
```

已写出初始文件：

```text
args.json
model000000000.pt
opt000000000.pt
train.log
train.pid
```

初始 checkpoint 验证：

```text
layers=8
latent_dim=512
batch_size=1
grad_accum_steps=64
num_steps=50000
save_interval=5000
log_interval=100
lambda_orient=1.0
lambda_body=1.0
lambda_transl=1.0
model000000000.pt: OrderedDict len=158
opt000000000.pt: dict len=2
```

## 监控命令

查看进程：

```bash
pid=$(cat save/interhuman/paper_config_l8_d512_accum64_50000_baseline/train.pid)
ps -p "$pid" -o pid,ppid,sid,stat,etime,%cpu,%mem,cmd
ps --ppid "$pid" -o pid,ppid,stat,etime,%cpu,%mem,cmd
```

查看日志：

```bash
tail -n 80 save/interhuman/paper_config_l8_d512_accum64_50000_baseline/train.log
```

查看 checkpoint：

```bash
find save/interhuman/paper_config_l8_d512_accum64_50000_baseline -maxdepth 1 -type f -name 'model*.pt' -o -name 'opt*.pt'
```

查看 GPU：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
```

## 后续

下一次检查重点：

- 是否到达 `step[100]` 并保持有限 loss。
- 是否持续接近 14 micro-batch/s。
- 是否按计划在 `model000005000.pt` / `opt000005000.pt` 写出后继续训练。
