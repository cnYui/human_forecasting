# Forecasting P5 Ablation + Paper Tables 计划文档

## 文档定位

本文使用 `using-superpowers` 工作流生成，是以下最终正式设计的 P5 落地计划：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
```

上游依赖：

```text
docs/ai/context/20260603-191712-forecasting-p1-dataset-result.md
docs/ai/context/20260603-194749-forecasting-p2-metrics-repeat-result.md
docs/ai/context/20260603-201148-forecasting-p3-baselines-result.md
docs/ai/context/20260603-202924-forecasting-p4-relation-result.md
```

本文只规划 P5，不进入实现结果记录。P5 完成后必须新建结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p5-ablation-paper-tables-result.md
```

## P5 目标

P5 目标是生成论文可用的主表和消融表，并验证 P4 的单 seed 观察是否稳定：

```text
relation-aware 是否跨 seed 稳定优于 concat no-relation 的 long_mse 和至少一个 relation metric。
```

P5 不是默认证明 relation-aware 成功。P5 是证伪阶段：

```text
如果 3 seed 后 relation-aware 不稳定优于 concat，
或者消融不能支持 relation module / relation feature 的贡献，
则不能写成功结论，必须回到 P4 修正模型或收窄论文主张。
```

## 当前事实

P4 seed=0 已达到最低门槛：

```text
relation future_mse: 0.031443351850382925
concat future_mse:   0.031901971752366684

relation long_mse:   0.036962207905420166
concat long_mse:     0.03789569738167008

relation relative_root_distance_error: 0.40891610895554853
concat relative_root_distance_error:   0.4946546451313289
```

但 P4 没有达到强门槛：

```text
independent future_mse: 0.02874350040329723
independent long_mse:   0.03612076791780671

relation-aware 没有优于 independent。
relation-aware 没有优于 repeat 的 relation metrics。
relation-aware 的 relative_orientation_error 和 inter_person_distance_consistency 略差于 concat。
```

P5 必须保留这个边界：

```text
不得只用“赢 concat”包装为全面最优。
不得把 single-seed P4 结果直接写成论文主结论。
不得省略 independent。
```

## 必须解决的问题

从第一性原理看，P5 只解决六件事：

```text
1. 为主实验建立可复现的 3-seed manifest。
2. 训练或复用 repeat / independent / concat / relation 的 seed 0/1/2 结果。
3. 增加 aggregate 汇总入口，输出 mean/std 表格，所有指标来自 P2/P3/P4 同一 evaluator。
4. 实现 relation 消融所需的最小模型开关。
5. 完成参数匹配 concat 和 relation feature ablation。
6. 根据稳定性和消融结果决定是否允许进入 P6。
```

## 非目标

P5 不做：

```text
P6 qualitative npy / curves / render
NTU120-AS / Chi3D-AS forecasting
diffusion forecasting
multi-dataset normalizer
best-of-K / diversity / multimodal forecasting
ReGenNet Table 4 evaluator
把 relation loss 默认加入主模型
```

P5 不重新定义 P1/P2/P3/P4 contract：

```text
obs:    [B,30,2,147]
target: [B,120,2,147]
pred:   [B,120,2,147]
训练 loss 默认 normalized active-vector MSE
指标默认 original scale
主表使用 eval/eval_forecasting.py 的 checkpoint/repeat metrics
```

## P5 阶段拆分

P5 分四个阶段推进：

```text
P5.1 Aggregation + Ablation Infrastructure
P5.2 Main Table 3 Seeds
P5.3 Ablation Table 3 Seeds
P5.4 Observation Ratio Supplement
```

门槛：

```text
P5.2 主表不支持核心主张时，不进入 P5.3/P5.4 的论文表格生产。
此时只允许做诊断性消融，并必须写失败/回退记录。
```

## P5.1 Aggregation + Ablation Infrastructure

### Aggregation

扩展：

```text
eval/eval_forecasting.py --mode aggregate
```

新增参数建议：

```text
--manifest results/forecasting/interhuman/p5_main_150_30_120/manifest.json
--save_dir results/forecasting/interhuman/p5_main_150_30_120
```

manifest 格式：

```json
{
  "protocol": {
    "dataset": "interhuman",
    "window_len": 150,
    "obs_len": 30,
    "pred_len": 120,
    "split": "test",
    "seeds": [0, 1, 2]
  },
  "runs": [
    {
      "table": "main",
      "method": "relation",
      "variant": "all_features_gru",
      "seed": 0,
      "run_dir": "save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000",
      "metrics_path": "save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/metrics_test.json",
      "checkpoint": "save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt"
    }
  ]
}
```

aggregate 输出：

```text
summary.json
summary.csv
summary.md
manifest.resolved.json
```

summary 必须包含：

```text
method
variant
num_runs
seeds
params_mean
params_std
future_mse_mean / future_mse_std
rotation_mse_mean / rotation_mse_std
translation_mse_mean / translation_mse_std
long_mse_mean / long_mse_std
relative_root_distance_error_mean / std
relative_orientation_error_mean / std
inter_person_distance_consistency_mean / std
```

实现约束：

```text
aggregate 只能读取已经落盘的 metrics json / args json / checkpoint metadata。
不得重新计算指标。
不得手工复制指标到表格。
缺少 seed、num_params、metrics key 不一致时直接失败。
```

### Ablation Model Knobs

当前 P4 relation 模型只支持 all features + GRU relation encoder。P5 需要新增最小开关：

```text
--relation_feature_set all|translation|velocity|orientation
--relation_encoder_type gru|none
```

建议实现位置：

```text
utils/forecasting_motion.py
  extract_relation_features(obs, feature_set="all")

model/forecasting.py
  RelationAwareForecastingModel 增加 relation_feature_set / relation_encoder_type。

train/train_forecasting.py
  CLI 透传 relation_feature_set / relation_encoder_type。

eval/eval_forecasting.py
  checkpoint 从 model_config 恢复，不新增指标分支。
```

特征维度：

```text
all:         relative translation 3 + relative velocity 3 + root distance 1 + relative orientation 9 = 16
translation: relative translation = 3
velocity:    relative velocity = 3
orientation: relative orientation = 9
```

`relation_encoder_type=none` 定义：

```text
仍使用 relation features。
不使用 GRU relation_encoder。
使用 temporal mean pooling + Linear(feature_dim -> relation_hidden_dim) 作为非时序 relation projection。
```

原因：

```text
这样能隔离“显式关系特征”与“时序 relation encoder”的贡献，
同时保持 joint decoder 和 person encoder 不变。
```

### Parameter-Matched Concat

P4 relation 参数量：

```text
10,058,704
```

concat hidden_dim 参数量核对：

```text
concat hidden_dim=256: 9,951,440
concat hidden_dim=259: 10,075,415
```

P5 parameter-matched concat 使用：

```text
model_type=concat
hidden_dim=259
num_layers=2
```

相对 P4 relation 参数差：

```text
16,711 params, 约 0.17%
```

## P5.2 Main Table 3 Seeds

### Main Methods

主表方法：

```text
Repeat
Independent
Concat no-relation
Relation-aware
```

seeds：

```text
0, 1, 2
```

repeat 是 deterministic baseline，但仍按 seed 0/1/2 跑三次 eval，预期 std 为 0。不得手工复制 repeat 指标。

### 可复用 seed=0 结果

允许复用已完成且配置匹配的 seed=0：

```text
independent:
  save/forecasting/interhuman/p3_independent_h256_l2_s0_5000

concat:
  save/forecasting/interhuman/p3_concat_h256_l2_s0_5000

relation:
  save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000
```

复用要求：

```text
manifest 必须记录真实 run_dir。
不得复制、重命名或覆盖历史 run。
aggregate 必须能读取这些历史 run 的 args / metrics / checkpoint metadata。
```

### 需要新增的 main runs

repeat：

```bash
for seed in 0 1 2; do
  micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
    --mode repeat \
    --dataset interhuman \
    --data_path dataset/interhuman/smpl/conditioned \
    --split test \
    --window_len 150 \
    --obs_len 30 \
    --pred_len 120 \
    --batch_size 64 \
    --num_workers 0 \
    --seed ${seed} \
    --save_dir save/forecasting/interhuman/p5_repeat_s${seed}_150_30_120
done
```

independent seed 1/2：

```bash
for seed in 1 2; do
  micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
    --dataset interhuman \
    --data_path dataset/interhuman/smpl/conditioned \
    --save_dir save/forecasting/interhuman/p5_main_independent_h256_l2_s${seed}_5000 \
    --model_type independent \
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
    --seed ${seed}
done
```

concat seed 1/2：

```bash
for seed in 1 2; do
  micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
    --dataset interhuman \
    --data_path dataset/interhuman/smpl/conditioned \
    --save_dir save/forecasting/interhuman/p5_main_concat_h256_l2_s${seed}_5000 \
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
    --seed ${seed}
done
```

relation seed 1/2：

```bash
for seed in 1 2; do
  micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
    --dataset interhuman \
    --data_path dataset/interhuman/smpl/conditioned \
    --save_dir save/forecasting/interhuman/p5_main_relation_h256_r128_l2_s${seed}_5000 \
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
    --relation_feature_set all \
    --relation_encoder_type gru \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --grad_accum_steps 1 \
    --max_samples -1 \
    --num_workers 0 \
    --save_interval 1000 \
    --eval_interval 500 \
    --seed ${seed}
done
```

### Main Table 验收

工程验收：

```text
每个 trainable method 有 seed 0/1/2 三个 metrics_test.json。
每个 run 有 args.json、normalizer.pt/json、model000005000.pt、opt000005000.pt。
repeat seed 0/1/2 均有 metrics_test.json。
aggregate 生成 summary.json/csv/md。
metrics key 与 P2 完全一致。
所有数值 finite。
```

论文主张门槛：

```text
relation long_mse_mean < concat long_mse_mean。
relation long_mse 在至少 2/3 seeds 优于同 seed concat。
至少一个 relation metric 的 mean 优于 concat。
同一个 relation metric 在至少 2/3 seeds 优于同 seed concat。
```

强门槛：

```text
relation future_mse_mean <= independent future_mse_mean。
relation long_mse_mean <= independent long_mse_mean。
至少一个 relation metric mean 优于 independent。
```

如果只过论文主张门槛但不过强门槛：

```text
允许进入 P5.3。
论文表述只能说 relation-aware 稳定优于 concat no-relation 的目标指标。
不得声称整体优于 independent。
```

如果论文主张门槛不过：

```text
停止主表推进。
新建 P5 失败分析文档。
只允许做诊断性 seed0 ablation，不生成成功论文表。
```

## P5.3 Ablation Table 3 Seeds

只有 P5.2 主表通过论文主张门槛后，才进入完整 P5.3。

### Ablation Rows

必做消融：

```text
concat no-relation
parameter-matched concat
relation without relation encoder
relation with relation encoder
relative translation only
relative velocity only
relative orientation only
all relation features
```

消融统一使用：

```text
seeds: 0,1,2
num_steps: 5000
batch_size: 32
eval_batch_size: 64
lr: 1e-3
weight_decay: 1e-4
num_workers: 0
```

### Parameter-Matched Concat

命令模板：

```bash
for seed in 0 1 2; do
  micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
    --dataset interhuman \
    --data_path dataset/interhuman/smpl/conditioned \
    --save_dir save/forecasting/interhuman/p5_ablation_concat_h259_l2_s${seed}_5000 \
    --model_type concat \
    --window_len 150 \
    --obs_len 30 \
    --pred_len 120 \
    --batch_size 32 \
    --eval_batch_size 64 \
    --num_steps 5000 \
    --hidden_dim 259 \
    --num_layers 2 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --grad_accum_steps 1 \
    --max_samples -1 \
    --num_workers 0 \
    --save_interval 1000 \
    --eval_interval 500 \
    --seed ${seed}
done
```

### Relation Without Encoder

命令模板：

```bash
for seed in 0 1 2; do
  micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
    --dataset interhuman \
    --data_path dataset/interhuman/smpl/conditioned \
    --save_dir save/forecasting/interhuman/p5_ablation_relation_noenc_all_s${seed}_5000 \
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
    --relation_feature_set all \
    --relation_encoder_type none \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --grad_accum_steps 1 \
    --max_samples -1 \
    --num_workers 0 \
    --save_interval 1000 \
    --eval_interval 500 \
    --seed ${seed}
done
```

### Relation Feature Ablations

命令模板：

```bash
for feature_set in translation velocity orientation; do
  for seed in 0 1 2; do
    micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
      --dataset interhuman \
      --data_path dataset/interhuman/smpl/conditioned \
      --save_dir save/forecasting/interhuman/p5_ablation_relation_${feature_set}_gru_s${seed}_5000 \
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
      --relation_feature_set ${feature_set} \
      --relation_encoder_type gru \
      --lr 1e-3 \
      --weight_decay 1e-4 \
      --grad_accum_steps 1 \
      --max_samples -1 \
      --num_workers 0 \
      --save_interval 1000 \
      --eval_interval 500 \
      --seed ${seed}
  done
done
```

`all relation features` 可复用 P5.2 relation 主实验 run，但 manifest 必须记录复用来源。

### Ablation 验收

消融支持 relation module 的条件：

```text
relation with encoder 的 long_mse_mean 优于 relation without encoder。
relation with encoder 的至少一个 relation metric mean 优于 relation without encoder。
```

消融支持 relation features 的条件：

```text
all relation features 的 long_mse_mean 优于 translation/velocity/orientation single-feature variants 中至少两个。
all relation features 的至少一个 relation metric mean 优于 single-feature variants 中至少两个。
```

参数匹配门槛：

```text
relation with encoder 的 long_mse_mean 优于 parameter-matched concat。
至少一个 relation metric mean 优于 parameter-matched concat。
```

如果 ablation 不支持：

```text
不得写 relation module / feature 贡献。
必须在结果文档中说明 P5 不支持该论文主张。
```

## P5.4 Observation Ratio Supplement

正式设计包含：

```text
10%: obs=15, pred=135
20%: obs=30, pred=120
30%: obs=45, pred=105
50%: obs=75, pred=75
```

但当前 P2 metrics 第一版固定：

```text
pred_len=120
short_mse frames 0:40
mid_mse frames 40:80
long_mse frames 80:120
```

因此 P5.4 不能直接开跑。必须先做 metrics contract 扩展：

```text
compute_forecasting_metrics 支持动态 pred_len。
short/mid/long 改为按 pred_len 三等分。
pred_len=120 时必须保持原结果一致。
metrics_sanity 对 pred_len=75/105/120/135 均通过。
repeat baseline 对每个 observation ratio 可评估。
```

P5.4 只作为补充表，不阻塞 20% 主协议：

```text
主表仍以 obs=30/pred=120 为准。
observation ratio 表不用于替代 P5.2/P5.3 的结论。
```

## 结果路径

主表：

```text
results/forecasting/interhuman/p5_main_150_30_120/
  manifest.json
  manifest.resolved.json
  summary.json
  summary.csv
  summary.md
```

消融表：

```text
results/forecasting/interhuman/p5_ablation_150_30_120/
  manifest.json
  manifest.resolved.json
  summary.json
  summary.csv
  summary.md
```

观测比例补充：

```text
results/forecasting/interhuman/p5_observation_ratio/
  manifest.json
  manifest.resolved.json
  summary.json
  summary.csv
  summary.md
```

## P5 最终验收标准

工程验收：

```text
compileall 通过。
main table manifest 完整。
ablation manifest 完整。
aggregate 输出 json/csv/md。
所有表格指标来自 metrics_test.json。
所有 trainable paper rows 至少 3 seeds。
所有 checkpoint、args、metrics 可回溯。
所有 metrics finite。
```

论文验收：

```text
relation-aware 在 mean 和至少 2/3 seeds 上优于 concat 的 long_mse。
relation-aware 在 mean 和至少 2/3 seeds 上优于 concat 的至少一个 relation metric。
parameter-matched concat 不能解释 relation-aware 的全部收益。
relation feature ablation 支持 all features 的收益。
relation encoder ablation 支持 relation encoder 的收益。
```

失败条件：

```text
relation-aware 只赢 repeat。
relation-aware 只赢 future_mse。
relation-aware 只在 seed=0 赢 concat。
relation-aware 不赢 parameter-matched concat。
消融显示 relation encoder 或 relation features 没贡献。
```

失败处理：

```text
新建 P5 failure result 文档。
不得进入 P6 作为成功论文展示。
回到 P4，优先检查 original-scale relation feature、relation loss 消融、fusion 结构和 independent 强基线问题。
```

## P5 完成记录要求

P5 完成后新建结果文档，至少记录：

```text
实现文件变更
所有 experiment manifests
复用的 seed=0 run 路径
新增 seed=1/2 run 路径
所有聚合表路径
每个 method/variant 的 mean/std
主张门槛是否通过
强门槛是否通过
ablation 是否支持 relation module/features
是否允许进入 P6
```
