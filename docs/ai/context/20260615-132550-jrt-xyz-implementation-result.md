# JRT XYZ 实现结果

## 结论

已在 ReGenNet 内实现 `jrt_xyz` baseline，并用本地 InterHuman H5 数据完成最小验证。该实现不修改 `/home/rpartx3080/CodeSpace/JRTransformer` 官方仓库，复用 ReGenNet 现有：

```text
InterHumanForecastDataset
active_to_xyz
train/train_forecasting_xyz.py
eval/eval_forecasting_xyz.py
```

## 新增与修改

新增：

```text
model/forecasting_jrt.py
docs/ai/context/20260615-132215-jrt-xyz-implementation-plan.md
docs/ai/context/20260615-132550-jrt-xyz-implementation-result.md
```

修改：

```text
model/forecasting_xyz.py
train/train_forecasting_xyz.py
AGENTS.md
```

## 模型接口

`jrt_xyz` 输入输出：

```text
obs_xyz    [B,30,2,24,3]
target_xyz [B,120,2,24,3]
pred_xyz   [B,120,2,24,3]
```

模型内部 token：

```text
2 persons * 24 SMPL joints = 48 tokens
```

relation tensor：

```text
[B,48,48,obs_len+2]
```

包含：

1. 历史 `exp(-joint_distance)`。
2. SMPL 24-joint skeleton adjacency。
3. same-person connectivity。

训练 loss：

```text
pose_mse + jrt_relation_weight * future_relation_l1 + aux_weight * aux_losses
```

其中 future relation target 使用 target future joints 的 `exp(-distance)`，贴近 JRT 官方实现。

## 验证结果

### compileall

命令：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m compileall model/forecasting_jrt.py model/forecasting_xyz.py train/train_forecasting_xyz.py eval/eval_forecasting_xyz.py
```

结果：通过。

### 前向与 loss 检查

配置：

```text
hidden_dim=64
num_heads=4
num_layers=1
B=2
```

结果：

```text
params=572644
pred=(2,120,2,24,3), finite=True
loss=2.2405364513397217, finite=True
```

### 真实数据 xyz smoke

命令使用：

```text
data_path=dataset/interhuman/smpl/conditioned
max_samples=2
batch_size=1
```

输出：

```text
train/val/test lengths = 2/2/2
obs_active    [1,30,2,147]
target_active [1,120,2,147]
obs_xyz       [1,30,2,24,3]
target_xyz    [1,120,2,24,3]
finite=True
```

结果文件：

```text
save/forecasting/interhuman/p9_jrt_xyz_smoke/xyz_smoke_summary.json
```

### 2-step training

命令核心配置：

```text
model_type=jrt_xyz
hidden_dim=64
num_heads=4
num_layers=1
batch_size=1
max_samples=4
num_steps=2
aux_weight=0.1
jrt_relation_weight=0.5
```

输出：

```text
model_type=jrt_xyz params=572644 device=cuda effective_batch_size=1
step[1]: train_loss[0.160138]
step[2]: train_loss[0.389221]
```

checkpoint：

```text
save/forecasting/interhuman/p9_jrt_xyz_2step_smoke/model000000002.pt
```

自动 test eval：

```text
joint_mse=0.05019832914695144
mpjpe=0.2782419081777334
long_joint_mse=0.09757680032635108
relative_root_distance_error=0.27407389879226685
inter_person_distance_consistency_xyz=0.008990141679532826
```

### checkpoint eval

单独加载 checkpoint 验证通过：

```text
results/forecasting/interhuman/p9_jrt_xyz_2step_smoke_eval/metrics_test.json
```

结果与自动 test eval 一致。

### metrics sanity 回归

`pred == target` 时所有 xyz metrics 仍为 `0.0`。

结果文件：

```text
save/forecasting/interhuman/p9_jrt_xyz_metrics_sanity/metrics_sanity.json
```

## 使用注意

直接执行脚本时需要设置：

```bash
PYTHONPATH=.
```

原因是 `python eval/eval_forecasting_xyz.py` 会把 `eval/` 作为脚本目录加入 import path，未必包含项目根目录。

## 下一步建议

可以进入 P9.2：跑 seed0 正式 5000-step。

建议起步配置：

```text
model_type=jrt_xyz
hidden_dim=256
num_heads=8
num_layers=4
batch_size=8 或 16
eval_batch_size=8 或 16
num_steps=5000
aux_weight=0.2
jrt_relation_weight=0.5 或 1.0
```

显存风险来自 `[B,48,48,hidden_dim]` relation stream，建议正式训练前先用 batch_size=16 观察显存；若接近 10GB 上限，则降到 batch_size=8 并记录 effective batch。
