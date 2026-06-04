# Forecasting P4 Relation-Aware Joint Predictor 结果记录

## 文档定位

本文记录 P4 实现与验收结果，引用以下上游文档：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260603-202101-forecasting-p4-relation-plan.md
docs/ai/context/20260603-201148-forecasting-p3-baselines-result.md
```

P4 目标是实现 relation-aware joint predictor，并验证它是否优于 concat no-relation baseline 的长期预测和至少一个 relation metric。

## 实现文件

修改：

```text
utils/forecasting_motion.py
model/forecasting.py
train/train_forecasting.py
eval/eval_forecasting.py
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

### Relation Feature Extractor

新增：

```text
utils.forecasting_motion.extract_relation_features(obs)
```

输入输出：

```text
input:  obs [B,30,2,147]
output: relation_features [B,30,16]
```

特征定义：

```text
relative_root_translation: trans_A - trans_B, dim=3
relative_root_velocity: velocity_A - velocity_B, dim=3
root_distance: ||trans_A - trans_B||, dim=1
relative_root_orientation: flatten(R_A^T R_B), dim=9
```

实现约束：

```text
训练和评估中 relation features 由 normalized obs 直接构造。
没有新增 relation feature normalizer。
extractor 做 shape 和 finite 检查。
```

### Relation-Aware Model

新增：

```text
model.forecasting.RelationAwareForecastingModel
```

模型类型注册：

```text
FORECASTING_MODEL_TYPES = ("independent", "concat", "relation")
```

结构：

```text
shared person GRU(input_dim=147, hidden_dim=256, num_layers=2)
relation GRU(input_dim=16, hidden_dim=128, num_layers=1)
fusion MLP(256*2+128 -> 256)
joint decoder(256 -> 120*294)
```

输入输出：

```text
input:  obs [B,30,2,147]
output: pred [B,120,2,147]
```

参数量：

```text
10,058,704
```

checkpoint `model_config` 已记录：

```text
model_type=relation
obs_len=30
pred_len=120
person_dim=147
hidden_dim=256
num_layers=2
relation_hidden_dim=128
relation_num_layers=1
relation_feature_dim=16
relation_features=[
  relative_root_translation,
  relative_root_velocity,
  root_distance,
  relative_root_orientation
]
```

### Training / Eval 接入

`train/train_forecasting.py` 新增：

```text
--model_type relation
--relation_hidden_dim 128
--relation_num_layers 1
```

`eval/eval_forecasting.py --mode checkpoint` 已支持：

```text
--model_type relation
```

训练 loss 沿用 P3：

```text
normalized active-vector MSE
```

没有加入额外 relation loss。

## 验收命令

编译检查：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall utils/forecasting_motion.py model/forecasting.py train/train_forecasting.py eval/eval_forecasting.py
```

最小前向检查：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python - <<'PY'
import torch
from model.forecasting import create_forecasting_model, create_forecasting_model_from_config
from utils.forecasting_motion import extract_relation_features
obs = torch.randn(2, 30, 2, 147)
assert tuple(extract_relation_features(obs).shape) == (2, 30, 16)
model = create_forecasting_model("relation")
assert tuple(model(obs).shape) == (2, 120, 2, 147)
restored = create_forecasting_model_from_config(model.config())
assert tuple(restored(obs).shape) == (2, 120, 2, 147)
PY
```

relation smoke：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p4_relation_smoke \
  --model_type relation \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 8 \
  --eval_batch_size 64 \
  --num_steps 5 \
  --hidden_dim 256 \
  --num_layers 2 \
  --relation_hidden_dim 128 \
  --relation_num_layers 1 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --grad_accum_steps 1 \
  --max_samples 64 \
  --num_workers 0 \
  --save_interval 5 \
  --eval_interval 5 \
  --seed 0
```

P2 metrics sanity 回归：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode metrics_sanity \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 64 \
  --num_workers 0 \
  --max_samples 64 \
  --save_dir save/forecasting/interhuman/p4_regression_metrics_sanity
```

P4 official run：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000 \
  --model_type relation \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --num_steps 5000 \
  --hidden_dim 256 \
  --num_layers 2 \
  --relation_hidden_dim 128 \
  --relation_num_layers 1 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --grad_accum_steps 1 \
  --max_samples -1 \
  --num_workers 0 \
  --save_interval 1000 \
  --eval_interval 500 \
  --seed 0
```

official checkpoint 独立加载评估已执行：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode checkpoint \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --checkpoint save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt \
  --model_type relation \
  --normalizer save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/normalizer.pt \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 64 \
  --num_workers 0 \
  --save_dir save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000
```

## 验收结果

编译检查：

```text
通过
```

最小前向检查：

```text
relation_features: [2,30,16]
pred: [2,120,2,147]
params: 10,058,704
config restore: 通过
```

relation smoke：

```text
checkpoint: save/forecasting/interhuman/p4_relation_smoke/model000000005.pt
num_params: 10,058,704
test max_samples: 64
future_mse: 0.06092999875545502
long_mse: 0.061928361654281616
checkpoint 独立加载评估: 通过
```

P2 metrics sanity 回归：

```text
future_mse: 0.0
rotation_mse: 0.0
translation_mse: 0.0
short_mse: 0.0
mid_mse: 0.0
long_mse: 0.0
relative_root_distance_error: 0.0
relative_orientation_error: 0.0
inter_person_distance_consistency: 0.0
```

official training：

```text
checkpoint: save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt
optimizer: save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/opt000005000.pt
normalizer: save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/normalizer.pt
num_params: 10,058,704
seed: 0
num_steps: 5000
effective_batch_size: 32
```

official test metrics：

```text
future_mse: 0.031443351850382925
rotation_mse: 0.02872313491589441
translation_mse: 0.1620137987174387
short_mse: 0.025012160438721575
mid_mse: 0.032355690477635916
long_mse: 0.036962207905420166
relative_root_distance_error: 0.40891610895554853
relative_orientation_error: 0.7386043404969643
inter_person_distance_consistency: 0.01641271052163417
```

official checkpoint 独立加载评估：

```text
通过，metrics 与训练末尾 test metrics 一致。
```

## 与基线对比

P2 repeat：

```text
future_mse: 0.036892867478446695
long_mse: 0.05112874942032371
relative_root_distance_error: 0.255221389058068
relative_orientation_error: 0.5552304635836384
inter_person_distance_consistency: 0.006041959892430409
```

P3 concat：

```text
future_mse: 0.031901971752366684
long_mse: 0.03789569738167008
relative_root_distance_error: 0.4946546451313289
relative_orientation_error: 0.7301466854538504
inter_person_distance_consistency: 0.01559202428760491
```

P3 independent：

```text
future_mse: 0.02874350040329723
long_mse: 0.03612076791780671
relative_root_distance_error: 0.40124681779718774
relative_orientation_error: 0.6649324903337974
inter_person_distance_consistency: 0.01343646844276997
```

P4 relation：

```text
future_mse: 0.031443351850382925
long_mse: 0.036962207905420166
relative_root_distance_error: 0.40891610895554853
relative_orientation_error: 0.7386043404969643
inter_person_distance_consistency: 0.01641271052163417
```

P4 最低门槛判断：

```text
future_mse <= concat: 通过，0.031443351850382925 < 0.031901971752366684
long_mse < concat: 通过，0.036962207905420166 < 0.03789569738167008
relative_root_distance_error < concat: 通过，0.40891610895554853 < 0.4946546451313289
```

P4 强门槛判断：

```text
future_mse <= independent: 未通过
long_mse < independent: 未通过
relation metrics 优于 independent: 未通过
```

额外观察：

```text
relation-aware 优于 concat 的核心证据是 future_mse、long_mse 和 relative_root_distance_error。
relation-aware 没有优于 independent。
relation-aware 没有优于 repeat 的 relation metrics。
relative_orientation_error 和 inter_person_distance_consistency 均略差于 concat。
```

## 是否允许进入 P5

允许进入 P5，但论文主张必须收窄：

```text
可以进入 P5 做 multi-seed、relation feature ablation 和容量控制。
当前只能说 seed=0 下 relation-aware 相比 concat 改善 future_mse / long_mse / relative_root_distance_error。
不能声称 relation-aware 全面优于 independent。
不能声称 relation-aware 已解决所有 relation metrics。
```

P5 必须重点检查：

```text
1. relation-aware 是否跨 seed 稳定优于 concat。
2. independent 为什么仍强于 relation-aware。
3. relation features 在 normalized obs 上计算是否影响 orientation 语义。
4. 是否需要容量匹配或 original-scale relation feature。
5. 是否需要把 relation loss 作为消融，而不是默认主模型。
```
