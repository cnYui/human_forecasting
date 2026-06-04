# Forecasting P5.1 Aggregation + Ablation Infrastructure 结果记录

## 文档定位

本文记录 P5.1 基础设施实现与验收结果，引用以下上游文档：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260603-203757-forecasting-p5-plan.md
docs/ai/context/20260603-202924-forecasting-p4-relation-result.md
```

本文不是 P5 完成记录。P5.2 3-seed 主表、P5.3 消融表和 P5.4 observation-ratio 补充尚未执行。

## 实现文件

修改：

```text
utils/forecasting_motion.py
model/forecasting.py
train/train_forecasting.py
eval/eval_forecasting.py
```

新增 smoke manifest：

```text
results/forecasting/interhuman/p5_aggregate_smoke/manifest.json
```

未修改旧 ReGenNet 主路径：

```text
train/train_mdm.py
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
data_loaders/tensors.py::ccollate
```

## 已实现内容

### Relation Feature Set

`utils.forecasting_motion.extract_relation_features` 新增：

```text
feature_set=all|translation|velocity|orientation
```

维度：

```text
all: 16
translation: 3
velocity: 3
orientation: 9
```

新增 helper：

```text
relation_feature_dim(feature_set)
RELATION_FEATURE_SETS
RELATION_FEATURE_DIMS
```

### Relation Encoder Type

`model.forecasting.RelationAwareForecastingModel` 新增：

```text
relation_feature_set
relation_encoder_type=gru|none
```

定义：

```text
gru: 继续使用 GRU relation_encoder。
none: 对 relation features 做 temporal mean pooling，再用 Linear(feature_dim -> relation_hidden_dim) 投影。
```

checkpoint config 现在记录：

```text
relation_feature_set
relation_encoder_type
relation_feature_dim
```

旧 P4 checkpoint config 兼容性：

```text
旧 config 缺少 relation_feature_set / relation_encoder_type 时，默认恢复为 all + gru。
```

### Training CLI

`train/train_forecasting.py` 新增：

```text
--relation_feature_set all|translation|velocity|orientation
--relation_encoder_type gru|none
```

默认值：

```text
relation_feature_set=all
relation_encoder_type=gru
```

因此 P4 命令不加新参数时语义不变。

### Aggregate Mode

`eval/eval_forecasting.py` 新增：

```text
--mode aggregate
--manifest PATH
```

aggregate 只读取已落盘文件：

```text
metrics_test.json
args.json
checkpoint metadata
```

不会重新计算 metrics。

输出：

```text
summary.json
summary.csv
summary.md
manifest.resolved.json
```

summary 包含：

```text
method / variant / num_runs / seeds
params mean/std
METRIC_KEYS 中所有指标的 mean/std
```

## 验收命令

编译检查：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall utils/forecasting_motion.py model/forecasting.py train/train_forecasting.py eval/eval_forecasting.py
```

前向与 config restore 检查：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python - <<'PY'
import torch
from model.forecasting import create_forecasting_model, create_forecasting_model_from_config
from utils.forecasting_motion import extract_relation_features, relation_feature_dim
obs = torch.randn(2, 30, 2, 147)
for feature_set in ("all", "translation", "velocity", "orientation"):
    assert tuple(extract_relation_features(obs, feature_set=feature_set).shape) == (2, 30, relation_feature_dim(feature_set))
    for encoder_type in ("gru", "none"):
        model = create_forecasting_model("relation", hidden_dim=64, num_layers=1, relation_hidden_dim=32, relation_feature_set=feature_set, relation_encoder_type=encoder_type)
        assert tuple(model(obs).shape) == (2, 120, 2, 147)
        restored = create_forecasting_model_from_config(model.config())
        assert tuple(restored(obs).shape) == (2, 120, 2, 147)
PY
```

aggregate smoke：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode aggregate \
  --manifest results/forecasting/interhuman/p5_aggregate_smoke/manifest.json \
  --save_dir results/forecasting/interhuman/p5_aggregate_smoke
```

ablation knobs train smoke：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p5_ablation_knobs_smoke \
  --model_type relation \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 4 \
  --eval_batch_size 16 \
  --num_steps 2 \
  --hidden_dim 64 \
  --num_layers 1 \
  --relation_hidden_dim 32 \
  --relation_num_layers 1 \
  --relation_feature_set translation \
  --relation_encoder_type none \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --grad_accum_steps 1 \
  --max_samples 16 \
  --num_workers 0 \
  --save_interval 2 \
  --eval_interval 2 \
  --seed 0
```

checkpoint 独立加载评估：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode checkpoint \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --checkpoint save/forecasting/interhuman/p5_ablation_knobs_smoke/model000000002.pt \
  --model_type relation \
  --normalizer save/forecasting/interhuman/p5_ablation_knobs_smoke/normalizer.pt \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 16 \
  --num_workers 0 \
  --max_samples 16 \
  --save_dir save/forecasting/interhuman/p5_ablation_knobs_smoke
```

## 验收结果

编译检查：

```text
通过
```

前向与 config restore：

```text
all/translation/velocity/orientation 均通过。
gru/none 均通过。
旧 P4 relation config 恢复为 all + gru，通过。
```

aggregate smoke 输出：

```text
results/forecasting/interhuman/p5_aggregate_smoke/summary.json
results/forecasting/interhuman/p5_aggregate_smoke/summary.csv
results/forecasting/interhuman/p5_aggregate_smoke/summary.md
results/forecasting/interhuman/p5_aggregate_smoke/manifest.resolved.json
```

aggregate smoke rows：

```text
repeat:      future_mse=0.036892867478446695, long_mse=0.05112874942032371
independent: future_mse=0.02874350040329723,  long_mse=0.03612076791780671
concat:      future_mse=0.031901971752366684, long_mse=0.03789569738167008
relation:    future_mse=0.031443351850382925, long_mse=0.036962207905420166
```

ablation knobs train smoke：

```text
save_dir: save/forecasting/interhuman/p5_ablation_knobs_smoke
model_type: relation
relation_feature_set: translation
relation_encoder_type: none
num_steps: 2
max_samples: 16
num_params: 2,348,688
checkpoint: save/forecasting/interhuman/p5_ablation_knobs_smoke/model000000002.pt
```

checkpoint 独立加载评估：

```text
通过。
test future_mse=0.07065138220787048
test long_mse=0.07377895712852478
```

## 是否允许进入 P5.2

允许进入 P5.2：

```text
relation ablation knobs 可训练、可保存、可独立加载。
aggregate mode 可读取已有 metrics 并生成 json/csv/md。
P4 旧 checkpoint config 兼容。
```

但 P5 尚未完成：

```text
尚未执行 seed 1/2 主表训练。
尚未生成 3-seed main table。
尚未执行 parameter-matched concat。
尚未执行完整 relation feature ablation。
尚未扩展 dynamic pred_len metrics。
```

下一步：

```text
进入 P5.2，先完成 20% 主协议 3-seed 主表。
主表不过 P5 论文主张门槛时，不进入完整 P5.3 / P6 成功展示。
```
