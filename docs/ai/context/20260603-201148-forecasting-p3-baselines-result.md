# Forecasting P3 Independent / Concat Baselines 结果记录

## 文档定位

本文记录 P3 实现与验收结果，引用以下上游文档：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260603-195707-forecasting-p3-baselines-plan.md
docs/ai/context/20260603-194749-forecasting-p2-metrics-repeat-result.md
```

P3 目标是实现 independent predictor 和 concat no-relation predictor，建立可训练 baseline，并复用 P2 original-scale evaluator。

## 实现文件

新增：

```text
model/forecasting.py
train/train_forecasting.py
```

修改：

```text
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

### Independent Predictor

实现：

```text
model.forecasting.IndependentForecastingModel
```

结构：

```text
shared GRU(input_dim=147, hidden_dim=256, num_layers=2)
shared MLP decoder(hidden_dim -> pred_len * 147)
obs reshape: [B,30,2,147] -> [B*2,30,147]
pred reshape: [B*2,120*147] -> [B,120,2,147]
```

约束：

```text
A 分支不读 B_obs。
B 分支不读 A_obs。
不构造 relation features。
```

参数量：

```text
5,305,064
```

### Concat No-Relation Predictor

实现：

```text
model.forecasting.ConcatForecastingModel
```

结构：

```text
GRU(input_dim=294, hidden_dim=256, num_layers=2)
MLP decoder(hidden_dim -> pred_len * 294)
obs reshape: [B,30,2,147] -> [B,30,294]
pred reshape: [B,120*294] -> [B,120,2,147]
```

约束：

```text
读取双人原始历史。
不显式构造 relative translation / velocity / orientation / distance。
不使用 relation_encoder。
不加入 relation loss。
```

参数量：

```text
9,951,440
```

### 训练入口

实现：

```text
train/train_forecasting.py
```

训练 contract：

```text
loss = normalized active-vector MSE
optimizer = AdamW
num_steps = optimizer steps
支持 grad_accum_steps
num_workers 默认 0
train random crop
val/test center crop
```

保存：

```text
args.json
normalizer.pt
normalizer.json
model{step:09}.pt
opt{step:09}.pt
metrics_val.json
metrics_val.yaml
metrics_test.json
metrics_test.yaml
train_log.jsonl
```

与计划相比新增的非核心参数：

```text
eval_batch_size: 评估 batch size，默认 64。
log_interval: 控制训练日志打印。
clip_grad_norm: 默认 1.0，用于防止训练 NaN。
overwrite: 允许显式复跑同一 save_dir。
```

这些参数不改变 P3 数据协议、模型定义、loss 或 metrics 口径。

### Checkpoint Evaluator

扩展：

```text
eval/eval_forecasting.py --mode checkpoint
```

新增参数：

```text
--checkpoint
--model_type independent|concat
--normalizer
```

评估流程：

```text
checkpoint -> model_config -> create_forecasting_model -> load_state_dict
obs original -> normalize -> model -> pred normalized -> denormalize
compute_forecasting_metrics(pred_original, target_original, obs_original)
按 batch_size 加权聚合
保存 metrics_{split}.json/yaml
```

## 验收命令

编译检查：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall model/forecasting.py train/train_forecasting.py eval/eval_forecasting.py
```

concat smoke：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p3_concat_smoke \
  --model_type concat \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 8 \
  --eval_batch_size 64 \
  --num_steps 5 \
  --hidden_dim 256 \
  --num_layers 2 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --grad_accum_steps 1 \
  --max_samples 64 \
  --num_workers 0 \
  --save_interval 5 \
  --eval_interval 5 \
  --seed 0
```

independent smoke 同上，`--model_type independent`，保存目录：

```text
save/forecasting/interhuman/p3_independent_smoke
```

P2 regression：

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
  --save_dir save/forecasting/interhuman/p3_regression_metrics_sanity
```

concat official run：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p3_concat_h256_l2_s0_5000 \
  --model_type concat \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --num_steps 5000 \
  --hidden_dim 256 \
  --num_layers 2 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --grad_accum_steps 1 \
  --max_samples -1 \
  --num_workers 0 \
  --save_interval 1000 \
  --eval_interval 500 \
  --seed 0
```

independent official run 同上，`--model_type independent`，保存目录：

```text
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000
```

最终 checkpoint 独立加载评估：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode checkpoint \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --checkpoint save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt \
  --model_type concat \
  --normalizer save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/normalizer.pt \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 64 \
  --num_workers 0 \
  --save_dir save/forecasting/interhuman/p3_concat_h256_l2_s0_5000
```

independent checkpoint eval 同上，替换 checkpoint、normalizer、model_type 和 save_dir。

## 验收结果

### 编译检查

结果：

```text
通过
```

### Smoke

concat smoke：

```text
通过
checkpoint: save/forecasting/interhuman/p3_concat_smoke/model000000005.pt
num_params: 9,951,440
test max_samples: 64
future_mse: 0.06102471426129341
```

independent smoke：

```text
通过
checkpoint: save/forecasting/interhuman/p3_independent_smoke/model000000005.pt
num_params: 5,305,064
test max_samples: 64
future_mse: 0.06038358062505722
```

两个 smoke checkpoint 均已用 `eval_forecasting --mode checkpoint` 独立加载评估通过。

### P2 Regression

`metrics_sanity` 回归通过：

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

### Official Test Metrics

P2 repeat baseline：

```text
future_mse: 0.036892867478446695
long_mse: 0.05112874942032371
relative_root_distance_error: 0.255221389058068
relative_orientation_error: 0.5552304635836384
inter_person_distance_consistency: 0.006041959892430409
```

P3 concat：

```text
checkpoint: save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt
num_params: 9,951,440
seed: 0
num_steps: 5000
effective_batch_size: 32
future_mse: 0.031901971752366684
rotation_mse: 0.02901937451770925
translation_mse: 0.17026668681403784
short_mse: 0.024624377386424486
mid_mse: 0.03318584122233034
long_mse: 0.03789569738167008
relative_root_distance_error: 0.4946546451313289
relative_orientation_error: 0.7301466854538504
inter_person_distance_consistency: 0.01559202428760491
```

P3 independent：

```text
checkpoint: save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt
num_params: 5,305,064
seed: 0
num_steps: 5000
effective_batch_size: 32
future_mse: 0.02874350040329723
rotation_mse: 0.02674679075345749
translation_mse: 0.12458551774813434
short_mse: 0.01961966261531659
mid_mse: 0.03049006870412451
long_mse: 0.03612076791780671
relative_root_distance_error: 0.40124681779718774
relative_orientation_error: 0.6649324903337974
inter_person_distance_consistency: 0.01343646844276997
```

## 对比结论

P3 两个 trainable baseline 均优于 repeat 的 `future_mse`：

```text
repeat:      0.036892867478446695
concat:      0.031901971752366684
independent: 0.02874350040329723
```

P3 通过。

重要观察：

```text
当前 seed=0 下 independent 强于 concat，且参数量更少。
```

这说明第一阶段数据协议下，简单拼接双人历史并没有自动带来更好的预测。P4/P5 不能只把 concat 当作唯一强基线；后续主表必须同时报告 independent，并解释 relation-aware 相比 independent / concat 的差异。

另外，repeat 在 relation metrics 上仍然很强：

```text
repeat relative_root_distance_error: 0.255221389058068
repeat relative_orientation_error: 0.5552304635836384
repeat inter_person_distance_consistency: 0.006041959892430409
```

P3 模型主要改善 MSE / long_mse，但 relation metrics 尚未优于 repeat。这不阻塞 P4，因为 P3 目标是建立可训练 baseline；但 P4 的 relation-aware 模型必须重点改善 relation metrics，不能只改善 future_mse。

## 输出路径

concat：

```text
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/args.json
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/normalizer.pt
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/normalizer.json
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/opt000005000.pt
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/metrics_val.json
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/metrics_val.yaml
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/metrics_test.json
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/metrics_test.yaml
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/train_log.jsonl
```

independent：

```text
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/args.json
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/normalizer.pt
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/normalizer.json
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/opt000005000.pt
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/metrics_val.json
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/metrics_val.yaml
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/metrics_test.json
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/metrics_test.yaml
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/train_log.jsonl
```

## 是否允许进入 P4

允许进入 P4：

```text
P3 smoke 通过。
P3 official full train/eval 完成。
两个模型 checkpoint 可保存和加载。
两个模型 test metrics key 与 P2 完全一致。
所有指标 finite。
两个模型均优于 repeat future_mse。
参数量、训练预算、seed 已记录。
```

P4 实现时必须保留以下约束：

```text
relation-aware 不得改 P1/P2 数据与指标协议。
relation-aware 必须同时对比 repeat / independent / concat。
如果 relation-aware 只赢 concat 但不赢 independent，论文主张仍然偏弱，必须在 P5 中诚实呈现。
```
