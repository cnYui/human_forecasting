# Forecasting P3 Independent / Concat Baselines 计划文档

## 文档定位

本文使用 `using-superpowers` 工作流生成，是以下最终正式设计的 P3 落地计划：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
```

上游依赖：

```text
docs/ai/context/20260603-184214-forecasting-p1-p6-complete-design.md
docs/ai/context/20260603-161803-forecasting-p1-p6-roadmap.md
docs/ai/context/20260603-191712-forecasting-p1-dataset-result.md
docs/ai/context/20260603-194249-forecasting-p2-plan.md
docs/ai/context/20260603-194749-forecasting-p2-metrics-repeat-result.md
```

本文只规划 P3，不进入实现结果记录。P3 完成后必须新建结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p3-baselines-result.md
```

## P3 目标

P3 目标是建立两个可训练 baseline：

```text
independent predictor
concat no-relation predictor
```

它们用于回答一个必要问题：

```text
后续 relation-aware model 是否只是赢了 repeat / zero-velocity 这种弱基线。
```

P3 通过后，P4 才允许实现 relation-aware joint predictor。P3 不支撑论文核心贡献，只提供公平可训练对照。

## 必须解决的问题

从第一性原理看，P3 只解决四件事：

```text
1. 建立不读对方观测的 independent predictor。
2. 建立读取双人原始历史但不显式构造关系特征的 concat no-relation predictor。
3. 建立 supervised forecasting 训练循环、checkpoint 和 checkpoint evaluation。
4. 证明两个 trainable baseline 至少在 future_mse 上优于 repeat baseline。
```

P3 的核心验收不是指标达到论文结论，而是：

```text
训练、保存、加载、评估、结果落盘全部闭环。
```

## 非目标

P3 不做：

```text
relation-aware model
relation features
relation_encoder
relative_root_distance_loss
relative_orientation_loss
graph network
cross-attention
diffusion forecasting
multi-seed 主实验
ablation 表
qualitative 可视化
NTU120-AS / Chi3D-AS forecasting 接入
ReGenNet Table 4 evaluator
```

P3 不重新定义 P1/P2 contract：

```text
obs:    [B,30,2,147]
target: [B,120,2,147]
metrics 必须使用 original scale
训练 loss 使用 normalized active-vector MSE
所有模型共用 utils/forecasting_metrics.py
```

P3 不修改旧 ReGenNet 主路径：

```text
train/train_mdm.py
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
data_loaders/tensors.py::ccollate
```

## 当前代码基线

P1/P2 已落地：

```text
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
utils/forecasting_motion.py
utils/forecasting_metrics.py
eval/eval_forecasting.py
```

当前 `eval/eval_forecasting.py` 已支持：

```text
dataset_smoke
metrics_sanity
repeat
checkpoint
```

其中 `checkpoint` 当前是 `NotImplementedError`。P3 必须接入该 mode，不能另写平行 evaluator。

P2 repeat baseline test 指标为：

```text
future_mse: 0.036892867478446695
rotation_mse: 0.03583101653970602
translation_mse: 0.08786168225168244
short_mse: 0.019088565562595063
mid_mse: 0.04046128798774847
long_mse: 0.05112874942032371
relative_root_distance_error: 0.255221389058068
relative_orientation_error: 0.5552304635836384
inter_person_distance_consistency: 0.006041959892430409
```

P3 最低门槛：

```text
independent future_mse < 0.036892867478446695
concat future_mse < 0.036892867478446695
```

## 新增与修改文件

新增：

```text
model/forecasting.py
train/train_forecasting.py
```

修改：

```text
eval/eval_forecasting.py
```

可复用但不应破坏既有行为：

```text
utils/forecasting_motion.py
utils/forecasting_metrics.py
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
```

除非发现 P1/P2 contract 实现错误，否则 P3 不改数据协议和指标定义。

## Model Contract

### 共享输入输出

输入：

```text
obs: [B,30,2,147]
```

输出：

```text
pred: [B,120,2,147]
```

模型内部使用 normalized active vector。评估前必须 denormalize 到 original scale：

```text
pred_original = normalizer.denormalize(pred_normalized)
target_original = target
obs_original = obs
```

### Independent Predictor

定义：

```text
A_obs -> A_future
B_obs -> B_future
concat(A_future, B_future)
```

第一版结构：

```text
obs[:, :, person, :] -> GRU(input_dim=147, hidden_dim=256, num_layers=2)
last_hidden -> MLP(hidden_dim -> pred_len * 147)
reshape -> [B,120,2,147]
```

实现建议：

```text
x = obs.permute or reshape to [B*2,30,147]
shared_encoder(x)
shared_decoder(h)
reshape back to [B,120,2,147]
```

硬约束：

```text
A 分支不能读 B_obs。
B 分支不能读 A_obs。
decoder 不能在 person 维度上做 joint flatten。
不能使用 relation feature helper。
```

共享 person encoder/decoder 是第一版默认方案。原因是它减少参数，且不会引入跨人信息泄漏。

### Concat No-Relation Predictor

定义：

```text
concat(A_obs, B_obs) -> future(A,B)
```

第一版结构：

```text
obs.reshape(B,30,294) -> GRU(input_dim=294, hidden_dim=256, num_layers=2)
last_hidden -> MLP(hidden_dim -> pred_len * 294)
reshape -> [B,120,2,147]
```

约束：

```text
可以看到双人原始历史。
不能显式构造 relative translation / velocity / orientation / distance。
不能使用 relation_encoder。
不能加入 relation loss。
```

论文表述必须保持准确：

```text
concat 不是没有关系信息，而是没有显式关系归纳偏置。
```

## Training Contract

训练入口：

```text
python -m train.train_forecasting
```

必须使用 `regennet` micromamba 环境：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet
```

训练数据：

```text
train split
shuffle=True
num_workers 默认 0
T>=150 样本
train random crop
```

验证/测试：

```text
val/test split
shuffle=False
num_workers 默认 0
center crop
```

归一化：

```text
normalizer 只用 train split、T>=150 序列统计。
obs_normalized = normalizer.normalize(obs)
target_normalized = normalizer.normalize(target)
loss = mean((pred_normalized - target_normalized)^2)
metrics 使用 denormalize 后的 original-scale pred。
```

优化：

```text
optimizer = AdamW
num_steps = optimizer steps
支持 grad_accum_steps
支持 max_samples smoke
支持 seed 固定
```

默认训练配置：

```text
hidden_dim: 256
num_layers: 2
lr: 1e-3
weight_decay: 1e-4
batch_size: 32
grad_accum_steps: 1
num_steps: 5000
eval_interval: 500
save_interval: 1000
seed: 0
num_workers: 0
```

如果 5000 steps 后 train loss 仍明显下降且模型未超过 repeat，可追加到 10000 steps，但必须在 P3 结果文档记录原因。不得直接进入 P4。

## Checkpoint / Save Contract

每个 run 至少保存：

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

`model*.pt` 至少包含：

```text
model_state_dict
model_type
model_config
num_params
step
seed
normalizer_path
```

`opt*.pt` 至少包含：

```text
optimizer_state_dict
step
```

`args.json` 至少记录：

```text
dataset
data_path
save_dir
model_type
window_len
obs_len
pred_len
batch_size
num_steps
hidden_dim
num_layers
lr
weight_decay
grad_accum_steps
effective_batch_size
max_samples
num_workers
seed
num_params
created_at
```

## Evaluation Contract

P3 必须扩展：

```text
eval/eval_forecasting.py --mode checkpoint
```

新增参数：

```text
--checkpoint
--model_type independent|concat
--normalizer
```

checkpoint evaluation 流程：

```text
1. 读取 args / checkpoint / normalizer。
2. 按 checkpoint model_type 构造模型。
3. 加载 model_state_dict。
4. split=val/test，shuffle=False。
5. obs original -> normalize -> model -> pred_normalized。
6. pred_normalized -> denormalize -> pred original。
7. 调用 compute_forecasting_metrics(pred, target, obs)。
8. 按 batch_size 加权聚合。
9. 输出 metrics_{split}.json 和 metrics_{split}.yaml。
```

输出 key 必须与 P2 完全一致：

```text
future_mse
rotation_mse
translation_mse
short_mse
mid_mse
long_mse
relative_root_distance_error
relative_orientation_error
inter_person_distance_consistency
```

不得在 `train/train_forecasting.py` 内实现一套独立指标口径。训练期间 val/test eval 也必须复用 checkpoint evaluation 或同一组 helper。

## CLI 计划

### Concat smoke

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

### Independent smoke

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p3_independent_smoke \
  --model_type independent \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 8 \
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

### Concat official P3 run

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

### Independent official P3 run

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p3_independent_h256_l2_s0_5000 \
  --model_type independent \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 32 \
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

### Checkpoint eval

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

independent checkpoint eval 同上，替换 `save_dir`、`checkpoint` 和 `--model_type independent`。

## 实现顺序

1. 新增 `model/forecasting.py`，实现 model builder、参数量统计、independent 和 concat。
2. 新增 `train/train_forecasting.py` 参数解析、seed、device、dataset/loader。
3. 接入 train-only normalizer，训练保存 `normalizer.pt/json`。
4. 实现 normalized-space MSE、AdamW、grad_accum_steps。
5. 实现 checkpoint / optimizer / args / train_log 保存。
6. 扩展 `eval/eval_forecasting.py --mode checkpoint`。
7. 跑 concat smoke。
8. 跑 independent smoke。
9. 跑 concat official P3 run。
10. 跑 independent official P3 run。
11. 分别跑 val/test checkpoint evaluation。
12. 新建 P3 结果文档，记录配置、参数量、训练预算、seed、metrics 和是否允许进入 P4。

## P3 验收标准

必须全部满足：

```text
independent max_samples smoke 通过。
concat max_samples smoke 通过。
两个模型 checkpoint 可保存。
两个模型 checkpoint 可加载。
两个模型 metrics_val.json/yaml 可落盘。
两个模型 metrics_test.json/yaml 可落盘。
两个模型 metrics key 与 P2 完全一致。
所有指标 finite。
两个模型均记录参数量、训练预算、seed。
independent future_mse 优于 repeat baseline。
concat future_mse 优于 repeat baseline。
P3 结果文档已新建。
AGENTS.md 已记录 P3 完成状态或阻塞原因。
```

P3 不过，不进入 P4。

## 失败回退

```text
loss 不下降：检查 normalizer、target 切片、decoder reshape 和学习率。
checkpoint 不能加载：先修保存/加载 contract，不进入 P4。
metrics key 不稳定：回到 P2 evaluator contract，不进入训练结论。
输出 shape 错误：先固定 [B,120,2,147]，不改 metrics 迁就模型。
出现 NaN：检查 normalized 输入、梯度裁剪需求和 optimizer 状态。
independent 泄漏另一人信息：修输入切分，删除受污染结果，重跑。
concat 使用 relation features：删除该实现，重跑。
模型赢不了 repeat：先排查训练协议、normalizer、seed、steps，不直接加 relation-aware。
concat 明显弱于 independent：检查 concat reshape 是否打乱 person/time 维度。
```

## P3 完成后允许进入 P4 的条件

只有同时满足以下条件，才允许实现 relation-aware model：

```text
P3 两个 smoke 均通过。
independent 和 concat 均完成 full train/eval。
两个模型至少在 test future_mse 上优于 repeat。
checkpoint、args、normalizer、metrics 文件可回溯。
P3 结果文档写明模型参数量、训练预算和 seed。
AGENTS.md 已更新下一步为 P4。
```

如果只完成 smoke，不能进入 P4。

如果 concat 未优于 repeat，不能进入 P4。

如果 independent 未优于 repeat，但 concat 已优于 repeat，也仍不能进入 P4；应先排查 independent 是否存在训练或输入切分问题。
