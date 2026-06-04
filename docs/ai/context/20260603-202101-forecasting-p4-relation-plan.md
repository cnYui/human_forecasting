# Forecasting P4 Relation-Aware Joint Predictor 计划文档

## 文档定位

本文使用 `using-superpowers` 工作流生成，是以下最终正式设计的 P4 落地计划：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
```

上游依赖：

```text
docs/ai/context/20260603-191712-forecasting-p1-dataset-result.md
docs/ai/context/20260603-194749-forecasting-p2-metrics-repeat-result.md
docs/ai/context/20260603-201148-forecasting-p3-baselines-result.md
```

本文只规划 P4，不记录实现结果。P4 完成后必须新建结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p4-relation-result.md
```

## P4 目标

P4 目标是实现论文核心模型：

```text
relation-aware joint predictor
```

它需要回答一个可证伪问题：

```text
显式双人关系归纳偏置是否比 concat no-relation predictor 更有利于长期未来预测和交互关系一致性。
```

P4 只做单 seed 工程闭环和第一版有效性验证。跨 seed 稳定性、消融表和论文主表属于 P5。

## 必须解决的问题

从第一性原理看，P4 只解决五件事：

```text
1. 从 obs active vector 中构造显式关系特征。
2. 新增 relation-aware 模型，并保持输入输出 contract 不变。
3. 复用 P3 supervised training / checkpoint / evaluator，不另写指标。
4. 完成 smoke、checkpoint 独立加载和 full train/eval。
5. 用 P2/P3 固定指标证明 relation-aware 至少优于 concat 的 long_mse 和一个 relation metric。
```

P4 的核心不是堆更大模型，而是验证关系特征是否提供了有效归纳偏置。

## 非目标

P4 不做：

```text
P5 multi-seed 主实验
P5 relation feature ablation 表
graph network
cross-attention
diffusion forecasting
scheduled sampling
autoregressive rollout
NTU120-AS / Chi3D-AS forecasting 接入
ReGenNet Table 4 evaluator
sample/visualize_forecasting.py 定性可视化
relative_root_distance_loss 默认加入
relative_orientation_loss 默认加入
```

P4 不重新定义 P1/P2/P3 contract：

```text
obs:    [B,30,2,147]
target: [B,120,2,147]
pred:   [B,120,2,147]
训练 loss 使用 normalized active-vector MSE
论文指标必须在 original scale 计算
所有 checkpoint metrics 继续使用 eval/eval_forecasting.py
```

P4 不修改旧 ReGenNet 主路径：

```text
train/train_mdm.py
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
data_loaders/tensors.py::ccollate
```

## 当前基线

P2 repeat baseline test：

```text
future_mse: 0.036892867478446695
long_mse: 0.05112874942032371
relative_root_distance_error: 0.255221389058068
relative_orientation_error: 0.5552304635836384
inter_person_distance_consistency: 0.006041959892430409
```

P3 concat baseline test：

```text
checkpoint: save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt
num_params: 9,951,440
seed: 0
num_steps: 5000
effective_batch_size: 32
future_mse: 0.031901971752366684
long_mse: 0.03789569738167008
relative_root_distance_error: 0.4946546451313289
relative_orientation_error: 0.7301466854538504
inter_person_distance_consistency: 0.01559202428760491
```

P3 independent baseline test：

```text
checkpoint: save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt
num_params: 5,305,064
seed: 0
num_steps: 5000
effective_batch_size: 32
future_mse: 0.02874350040329723
long_mse: 0.03612076791780671
relative_root_distance_error: 0.40124681779718774
relative_orientation_error: 0.6649324903337974
inter_person_distance_consistency: 0.01343646844276997
```

P4 最低门槛：

```text
future_mse <= 0.031901971752366684
long_mse < 0.03789569738167008
至少一个 relation metric 优于 concat：
  relative_root_distance_error < 0.4946546451313289
  或 relative_orientation_error < 0.7301466854538504
  或 inter_person_distance_consistency < 0.01559202428760491
```

P4 强门槛：

```text
future_mse <= 0.02874350040329723
long_mse < 0.03612076791780671
至少一个 relation metric 优于 independent。
```

如果只达到最低门槛，P4 可以进入 P5，但论文表述必须谨慎：只能声称显式关系建模优于 concat no-relation，不能声称全面优于 independent。

## 新增与修改文件

修改：

```text
utils/forecasting_motion.py
model/forecasting.py
train/train_forecasting.py
eval/eval_forecasting.py
```

不新增平行 evaluator。若需要新增辅助函数，优先放入已有职责边界：

```text
utils/forecasting_motion.py
  relation feature extract/shape check。

model/forecasting.py
  RelationAwareForecastingModel、model type 注册、config 保存/加载。

train/train_forecasting.py
  让 --model_type relation 复用原训练循环。

eval/eval_forecasting.py
  让 checkpoint mode 接受 relation，不改变 metrics。
```

## Relation Feature Contract

输入：

```text
obs: [B,30,2,147]
```

输出：

```text
relation_features: [B,30,REL_DIM]
```

第一版特征：

```text
relative root translation: trans_A - trans_B, dim=3
relative root velocity: velocity_A - velocity_B, dim=3
root distance: ||trans_A - trans_B||, dim=1
relative root orientation: flatten(R_A^T R_B), dim=9
```

第一版 `REL_DIM`：

```text
16
```

translation 与 velocity 从 active vector 的最后 3 维读取：

```text
trans = obs[..., 144:147]
```

velocity 定义：

```text
velocity[:, 0] = 0
velocity[:, t] = trans[:, t] - trans[:, t - 1]
relative_velocity = velocity_A - velocity_B
```

relative orientation 使用项目已有函数：

```text
utils.rotation_conversions.rotation_6d_to_matrix
root rot6d = obs[..., 0:6]
R_rel = R_A^T R_B
```

实现要求：

```text
特征必须支持 normalized obs，因为模型训练和推理输入均为 normalized active vector。
特征 extractor 必须检查 shape 和 finite。
不在 P4 中新增 relation feature normalizer。
```

风险说明：

```text
在 normalized obs 上计算 relation orientation 会先对 rot6d 做归一化统计变换，再经 Gram-Schmidt 转为矩阵。
这不等价于 original-scale rot6d 的几何关系。
P4 第一版保留该设计以复用 P3 normalized training contract；如果 relation metrics 无改善，P4 结果文档必须记录该风险，并把 original-scale relation feature 或单独 relation normalizer 列为后续修正方案。
```

## Model Contract

新增模型：

```text
model.forecasting.RelationAwareForecastingModel
```

注册：

```text
FORECASTING_MODEL_TYPES = ("independent", "concat", "relation")
```

输入输出：

```text
input:  obs [B,30,2,147]
output: pred [B,120,2,147]
```

第一版结构：

```text
person_encoder(A_obs) -> h_A
person_encoder(B_obs) -> h_B
relation_encoder(relation_features) -> h_rel
fusion = concat(h_A, h_B, h_rel)
fusion_mlp(fusion) -> h_joint
joint_decoder(h_joint) -> future(A,B)
```

推荐参数：

```text
person_encoder: shared GRU(input_dim=147, hidden_dim=256, num_layers=2)
relation_encoder: GRU(input_dim=16, hidden_dim=128, num_layers=1)
fusion_mlp: Linear(256*2+128 -> 256) + ReLU
joint_decoder: _FutureDecoder(256 -> 120*294)
```

配置字段至少保存：

```text
model_type
obs_len
pred_len
person_dim
hidden_dim
num_layers
relation_hidden_dim
relation_num_layers
relation_feature_dim
relation_features
```

参数量预期：

```text
应高于 concat 或接近 concat。
如果显著高于 concat，P4 结果文档必须解释参数量差异，P5 再做容量控制消融。
```

## Training Contract

P4 默认训练预算沿用 P3，便于第一版对比：

```text
hidden_dim=256
num_layers=2
relation_hidden_dim=128
relation_num_layers=1
batch_size=32
eval_batch_size=64
num_steps=5000
lr=1e-3
weight_decay=1e-4
grad_accum_steps=1
seed=0
num_workers=0
save_interval=1000
eval_interval=500
```

训练 loss：

```text
loss = normalized active-vector MSE
```

P4 第一版不默认加入额外 relation loss。原因：

```text
先验证 architecture / feature 本身是否带来收益。
额外 relation loss 会改变优化目标，属于 P5 消融或 P4 失败后的有记录修正。
```

## CLI 变更

`train/train_forecasting.py`：

```text
--model_type relation
--relation_hidden_dim 128
--relation_num_layers 1
```

`eval/eval_forecasting.py`：

```text
--model_type relation
```

必须保持旧命令兼容：

```text
--model_type independent
--model_type concat
```

## 实施步骤

1. 在 `utils/forecasting_motion.py` 中新增 relation feature extractor。
2. 在 `model/forecasting.py` 中新增 `RelationAwareForecastingModel`，并扩展 model factory / config load。
3. 在 `train/train_forecasting.py` 中接入 relation 参数，保持旧模型默认行为不变。
4. 在 `eval/eval_forecasting.py` 中允许 checkpoint evaluator 加载 relation checkpoint。
5. 运行 compileall。
6. 运行 relation smoke。
7. 独立 checkpoint eval 验证 smoke checkpoint 可加载。
8. 回归 P2 metrics sanity，确认 evaluator 没漂移。
9. 运行 P4 official seed=0 full training。
10. 汇总 test metrics，判断是否达到 P4 最低门槛，并新建 P4 result 文档。

## 验收命令

编译检查：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall \
  utils/forecasting_motion.py \
  model/forecasting.py \
  train/train_forecasting.py \
  eval/eval_forecasting.py
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

smoke checkpoint 独立评估：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode checkpoint \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --checkpoint save/forecasting/interhuman/p4_relation_smoke/model000000005.pt \
  --model_type relation \
  --normalizer save/forecasting/interhuman/p4_relation_smoke/normalizer.pt \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 64 \
  --num_workers 0 \
  --max_samples 64 \
  --save_dir save/forecasting/interhuman/p4_relation_smoke
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

official checkpoint 独立评估：

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

## 验收标准

工程验收：

```text
compileall 通过。
relation smoke 通过。
smoke checkpoint 可独立加载评估。
P2 metrics sanity 回归通过。
official run 生成 model000005000.pt / opt000005000.pt。
official test metrics key 与 P2 完全一致。
所有 metrics finite。
args.json 记录 relation 参数。
checkpoint model_config 可恢复 relation 模型。
```

指标验收：

```text
future_mse <= concat future_mse。
long_mse < concat long_mse。
至少一个 relation metric 优于 concat。
记录 relation features、参数量、训练预算、seed。
```

进入 P5 的条件：

```text
达到 P4 指标验收，且 checkpoint evaluation 与训练末尾 test metrics 一致。
```

若 P4 未达标：

```text
不得进入 P5 主表。
必须新建 P4 失败分析文档，优先检查：
  1. normalized obs 上计算 relation orientation 是否破坏几何语义。
  2. relation encoder 容量是否不足或过强。
  3. fusion 是否被 joint decoder 忽略。
  4. 是否需要 original-scale relation feature 或单独 relation normalizer。
  5. 是否需要 relation loss，但不能无记录地直接加入。
```

## 结果记录要求

P4 完成后新建结果文档，至少记录：

```text
实现文件变更
relation feature 维度和定义
模型结构
参数量
训练预算
seed
checkpoint 路径
normalizer 路径
smoke 结果
compileall 结果
P2 regression 结果
official test metrics
与 repeat / independent / concat 的对比
是否允许进入 P5
```
