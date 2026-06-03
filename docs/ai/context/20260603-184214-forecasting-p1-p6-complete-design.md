# Forecasting P1-P6 完整设计文档

## 文档定位

本文件使用 `using-superpowers` 工作流整理，是 P1-P6 的工程设计 contract。它细化以下上游文档，但不覆盖原始路线图：

```text
docs/ai/context/20260603-160713-forecasting-final-goal-design.md
docs/ai/context/20260603-161334-forecasting-p1-p2-design.md
docs/ai/context/20260603-161803-forecasting-p1-p6-roadmap.md
docs/ai/context/20260603-182845-forecasting-roadmap-feasibility-review.md
```

原始路线图仍是第一阶段上位目标。后续每个阶段实现完成时，必须新建阶段结果文档，不得覆盖本文件或路线图。

## 总目标

论文第一阶段目标：

```text
Interaction-aware joint forecasting of two-person human motion from partial observations.
```

中文：

```text
基于部分观测的交互感知双人动作联合未来预测。
```

固定主协议：

```text
dataset: InterHuman
representation: SMPL reproduction active vector
window_len: 150
obs_len: 30
pred_len: 120
input: 前 30 帧双人动作
target: 后 120 帧双人动作
output: 两个人未来动作联合预测
task_type: deterministic forecasting
```

核心可证伪假设：

```text
显式建模两人关系应优于 independent predictor 和 concat no-relation predictor，
尤其应改善 long-horizon error 和 inter-person relation metrics。
```

如果 relation-aware model 不能稳定优于 concat no-relation baseline，则论文主张不成立，不能进入包装结论。

## 本地约束

必须使用项目环境：

```text
micromamba env: /home/rpartx3080/.local/micromamba/envs/regennet
python: 3.7.13
torch: 1.7.1
h5py: 3.7.0
cuda: available
```

裸系统 `python3` 缺少 `h5py`，后续命令必须显式使用：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python ...
```

数据事实：

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5
dataset/interhuman/smpl/conditioned/meta.json

format: [T,25,12]
body_model: smpl
rotation: rot6d
translation_slot: 24
translation_origin: actor frame 0

train: total 6021, T>=150 2910
val:   total 580,  T>=150 226
test:  total 1175, T>=150 508
```

第一阶段不修改旧 ReGenNet 主路径：

```text
train/train_mdm.py
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
```

原因是旧路径绑定 diffusion、actor condition / reactor target 和 ST-GCN evaluator，不符合 prefix-to-future forecasting 语义。

## 全局边界

第一阶段只做：

```text
InterHuman 150-frame deterministic two-person forecasting
active vector 表示
train split normalizer
统一 MSE 类和交互关系指标
repeat / independent / concat / relation-aware 对照
ablation 和 qualitative analysis
```

第一阶段不做：

```text
ReGenNet Table 4 完整复现
自然语言条件预测
动作标签条件预测
SMPL-X 转换
ST-GCN recognition evaluator
diffusion forecasting
best-of-K / diversity / multimodal sampling
NTU120-AS / Chi3D 泛化
```

这些内容可以作为 P7+，不得阻塞 P1-P6。

## 阶段依赖

唯一允许顺序：

```text
P1 Dataset + Normalizer
  -> P2 Metrics + Repeat Baseline
    -> P3 Independent / Concat Baselines
      -> P4 Relation-Aware Predictor
        -> P5 Ablation + Paper Tables
          -> P6 Visualization + Qualitative Analysis
```

禁止跳过：

```text
P1/P2 未完成，不实现 P3。
P3 未完成，不实现 P4。
P4 未完成，不写 P5 主结论。
P5 未完成，不把 P6 可视化作为主证明。
```

## 目录设计

新增 forecasting 专用路径：

```text
data_loaders/forecasting/__init__.py
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
utils/forecasting_motion.py
utils/forecasting_metrics.py
model/forecasting.py
train/train_forecasting.py
eval/eval_forecasting.py
sample/visualize_forecasting.py
```

保存路径：

```text
save/forecasting/interhuman/{run_name}/
results/forecasting/interhuman/{run_name}/
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p{N}-*.md
```

文件职责：

```text
interhuman.py: H5 读取、T>=150 过滤、窗口裁剪、dataset item。
tensors.py: forecasting collate，不做 padding。
forecasting_motion.py: active vector extract/restore、normalizer、relation feature 基础函数。
forecasting_metrics.py: 原始尺度指标。
forecasting.py: independent / concat / relation-aware 模型。
train_forecasting.py: supervised 训练、checkpoint、val/test eval。
eval_forecasting.py: repeat baseline 和 checkpoint 评估入口。
visualize_forecasting.py: npy、曲线、可选 render 输出。
```

## 统一数据表示

H5 输入：

```text
motion: [T, 25, 12]
actor_rot6d:         motion[:, :24, 0:6]
actor_translation:   motion[:, 24, 0:3]
reactor_rot6d:       motion[:, :24, 6:12]
reactor_translation: motion[:, 24, 6:9]
```

active vector：

```text
person_dim = 24 * 6 + 3 = 147
two_person_dim = 2 * 147 = 294
motion_active: [T, 2, 147]
motion_active[:, 0, :144] = actor_rot6d.reshape(T, 144)
motion_active[:, 0, 144:147] = actor_translation
motion_active[:, 1, :144] = reactor_rot6d.reshape(T, 144)
motion_active[:, 1, 144:147] = reactor_translation
```

窗口：

```text
window: [150, 2, 147]
obs:    [30, 2, 147]
target: [120, 2, 147]
```

active vector restore：

```text
motion[:, :24, 0:6] = active[:, 0, :144].reshape(T, 24, 6)
motion[:, 24, 0:3] = active[:, 0, 144:147]
motion[:, :24, 6:12] = active[:, 1, :144].reshape(T, 24, 6)
motion[:, 24, 6:9] = active[:, 1, 144:147]
其他 channel 保持 0
```

第一版保留 frozen H5 的 `actor frame 0` 坐标，不做 window recenter。若后续改为 window recenter，必须作为协议变更记录。

## Normalizer Contract

统计来源：

```text
split: train only
samples: T>=150 的 train sequences
frames: 每条可用序列的全部帧
space: active vector original scale
```

张量形状：

```text
mean: [1, 1, 2, 147]
std:  [1, 1, 2, 147]
eps:  1e-6
std < eps 的维度设为 1.0
```

接口：

```text
normalize(x) = (x - mean) / std
denormalize(x) = x * std + mean
```

保存：

```text
normalizer.pt
normalizer.json
```

`normalizer.json` 至少包含：

```text
dataset
data_path
window_len
obs_len
pred_len
num_train_sequences_used
num_train_frames_used
person_dim
eps
created_at
```

训练 loss 使用 normalized space。论文指标只使用 original scale。

## Metrics Contract

输入：

```text
pred:   Tensor[B, 120, 2, 147]
target: Tensor[B, 120, 2, 147]
obs:    Tensor[B, 30, 2, 147]
```

所有输入必须是 original scale。

输出 key 固定：

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

定义：

```text
future_mse = mean((pred - target)^2)
rotation_mse = mean((pred[..., :144] - target[..., :144])^2)
translation_mse = mean((pred[..., 144:147] - target[..., 144:147])^2)
short_mse = future_mse over pred frames 0:40
mid_mse = future_mse over pred frames 40:80
long_mse = future_mse over pred frames 80:120
```

相对距离：

```text
trans_A = x[:, :, 0, 144:147]
trans_B = x[:, :, 1, 144:147]
dist = norm(trans_A - trans_B, dim=-1)
relative_root_distance_error = mean(abs(pred_dist - target_dist))
```

相对朝向：

```text
root_A = x[:, :, 0, 0:6]
root_B = x[:, :, 1, 0:6]
R_A = rotation_6d_to_matrix(root_A)
R_B = rotation_6d_to_matrix(root_B)
R_rel = R_A.transpose(-1, -2) @ R_B
R_err = R_rel_pred.transpose(-1, -2) @ R_rel_target
angle = acos(clamp((trace(R_err) - 1) / 2, -1 + eps, 1 - eps))
relative_orientation_error = mean(angle)
```

`rotation_6d_to_matrix` 必须用于正交化预测 rot6d，`acos` 前必须 clamp，避免浮点 NaN。

相对距离变化一致性：

```text
dist_full_pred = concat(last_obs_dist, pred_dist)
dist_full_target = concat(last_obs_dist, target_dist)
delta_pred = dist_full_pred[:, 1:] - dist_full_pred[:, :-1]
delta_target = dist_full_target[:, 1:] - dist_full_target[:, :-1]
inter_person_distance_consistency = mean(abs(delta_pred - delta_target))
```

注意：这个指标只能表述为 root distance trend consistency，不能声称覆盖完整 interaction quality。

## P1 设计：Dataset + Normalizer

### 目标

建立唯一 forecasting 数据协议，保证所有 baseline 和模型读取完全一致的 `obs / target`。

### 新增文件

```text
data_loaders/forecasting/__init__.py
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
utils/forecasting_motion.py
```

### 核心类与函数

```text
InterHumanForecastDataset(
    data_path,
    split,
    window_len=150,
    obs_len=30,
    pred_len=120,
    max_samples=-1,
    seed=0,
)

forecasting_collate(batch) -> (obs, target, meta)
extract_active_motion(motion_h5) -> active
restore_active_motion(active) -> motion_h5_like
compute_forecasting_normalizer(data_path, save_path, ...)
load_forecasting_normalizer(path)
```

### Dataset item

```text
{
  "obs": Tensor[30, 2, 147],
  "target": Tensor[120, 2, 147],
  "sample_id": str,
  "start": int,
  "length": int,
}
```

### Batch

```text
obs:    Tensor[B, 30, 2, 147]
target: Tensor[B, 120, 2, 147]
meta:   list[dict]
```

### 采样规则

训练：

```text
保留 T>=150。
每次 __getitem__ 随机选择 start in [0, T - 150]。
同一样本多次读取 start 可以变化。
```

验证/测试：

```text
保留 T>=150。
start = floor((T - 150) / 2)。
同一样本多次读取 start 固定。
```

不做 padding。短序列不进入第一阶段协议。

### P1 验收

必须通过：

```text
train dataset length = 2910
val dataset length = 226
test dataset length = 508
obs shape = [30,2,147]
target shape = [120,2,147]
所有 obs/target 数值有限
train 同一样本多次读取 start 可变化
val/test 同一样本多次读取 start 固定
normalizer 可保存和加载
normalize -> denormalize 最大误差在浮点容忍范围内
active -> h5-like -> active 最大误差在浮点容忍范围内
```

建议验收入口：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode dataset_smoke \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 4 \
  --num_workers 0
```

### P1 失败回退

```text
shape 错：先查 active vector 映射，再查 collate。
样本数错：先查 T>=150 过滤，再查 split 文件。
数值异常：先查 H5 finite，再查 std 防护。
eval start 不固定：修 dataset，不进入 P2。
```

### P1 阶段记录

完成后新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p1-dataset-result.md
```

记录：

```text
引用路线图和本设计文件
实现文件
shape 检查
样本数
normalizer 摘要
smoke 命令和结果
是否允许进入 P2
```

## P2 设计：Metrics + Repeat Baseline

### 目标

在训练模型前完成评估闭环，确保后续所有结果可比较。

### 新增文件

```text
utils/forecasting_metrics.py
eval/eval_forecasting.py
```

### Repeat baseline

定义：

```text
pred[:, t] = obs[:, -1]
```

repeat 不是论文贡献，只是 sanity baseline 和最低对照。

### 评估入口

`eval/eval_forecasting.py` 支持：

```text
--mode dataset_smoke
--mode metrics_sanity
--mode repeat
--mode checkpoint
```

repeat 命令：

```text
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
  --save_dir save/forecasting/interhuman/repeat_150_30_120
```

输出：

```text
save/forecasting/interhuman/repeat_150_30_120/metrics_test.json
save/forecasting/interhuman/repeat_150_30_120/metrics_test.yaml
```

### P2 验收

必须通过：

```text
pred == target 时 MSE 类指标为 0。
pred == target 时 relative_root_distance_error 为 0。
pred == target 时 relative_orientation_error 接近 0。
repeat baseline 可完整评估 test split。
repeat baseline 输出固定 metrics key。
metrics 文件可被 P3/P4/P5 复用。
```

### P2 失败回退

```text
pred == target 非 0：修 metrics，不进入 P3。
relative_orientation_error NaN：检查 rot6d_to_matrix、trace clamp、eps。
repeat 跑不完：修 dataset / collate / dataloader。
metrics key 不稳定：固定 schema，不进入模型训练。
```

### P2 阶段记录

完成后新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p2-metrics-repeat-result.md
```

记录：

```text
sanity check 结果
repeat baseline test 指标
metrics 文件路径
是否允许进入 P3
```

## P3 设计：Independent / Concat Baselines

### 目标

建立两个可训练 baseline，证明 relation-aware 不是只赢 repeat baseline。

### 新增文件

```text
model/forecasting.py
train/train_forecasting.py
```

### 共享训练 contract

输入：

```text
obs:    [B, 30, 2, 147]
target: [B, 120, 2, 147]
```

训练：

```text
loss = MSE(pred_normalized, target_normalized)
optimizer = AdamW
num_steps 表示 optimizer steps
grad_accum_steps 支持梯度累积
num_workers 默认 0
```

保存：

```text
args.json
normalizer.pt
normalizer.json
model{step:09}.pt
opt{step:09}.pt
metrics_val.json / yaml
metrics_test.json / yaml
```

日志至少包含：

```text
step
train_loss
val_future_mse
effective_batch_size
model_num_params
seed
```

### Baseline 1：Independent Predictor

定义：

```text
A_obs -> A_future
B_obs -> B_future
concat(A_future, B_future)
```

硬约束：

```text
A 分支不能读 B_obs。
B 分支不能读 A_obs。
```

建议结构：

```text
person_encoder: GRU(input_dim=147, hidden_dim=256/512, num_layers=2)
decoder: MLP(hidden_dim -> pred_len * 147)
参数可共享，也可 two-tower；第一版优先共享 encoder/decoder，减少参数。
```

输出：

```text
pred: [B, 120, 2, 147]
```

### Baseline 2：Concat No-Relation Predictor

定义：

```text
concat(A_obs, B_obs) -> future(A, B)
```

约束：

```text
可以看到双人原始历史。
不能显式构造 relation features。
不能使用 relation_encoder。
```

建议结构：

```text
encoder: GRU(input_dim=294, hidden_dim=256/512, num_layers=2)
decoder: MLP(hidden_dim -> pred_len * 294)
```

论文表述必须准确：concat 不是“没有关系信息”，而是“没有显式关系归纳偏置”。

### 训练命令草案

concat smoke：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/concat_smoke \
  --model_type concat \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 8 \
  --num_steps 5 \
  --hidden_dim 256 \
  --num_layers 2 \
  --lr 1e-3 \
  --grad_accum_steps 1 \
  --max_samples 64 \
  --num_workers 0 \
  --seed 0
```

independent smoke 同上，`--model_type independent`。

### P3 验收

必须通过：

```text
independent max_samples smoke 通过。
concat max_samples smoke 通过。
两个模型都保存 checkpoint / args / normalizer。
两个模型都能在 val/test 输出 P2 固定 metrics。
两个模型至少优于 repeat baseline 的 future_mse。
记录模型参数量、训练预算、seed。
concat 与 independent 的差异可解释。
```

### P3 失败回退

```text
loss 不下降：检查 normalizer、target 错位、decoder reshape。
checkpoint 不能加载：先修保存/加载，不进入 P4。
模型赢不了 repeat：先检查训练协议，不直接加复杂模型。
independent 泄漏另一人信息：修模型输入切分，重跑结果。
```

### P3 阶段记录

完成后新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p3-baselines-result.md
```

记录：

```text
模型配置
参数量
训练命令
checkpoint
val/test metrics
repeat / independent / concat 对比表
是否允许进入 P4
```

## P4 设计：Relation-Aware Joint Predictor

### 目标

实现论文核心模型：显式关系推理 + 双人联合未来预测。

### 关系特征

从 `obs` 中提取：

```text
trans_A = obs[:, :, 0, 144:147]
trans_B = obs[:, :, 1, 144:147]
relative_root_translation = trans_A - trans_B                # [B,30,3]
velocity_A = trans_A[:, 1:] - trans_A[:, :-1]
velocity_B = trans_B[:, 1:] - trans_B[:, :-1]
relative_root_velocity = velocity_A - velocity_B             # [B,29,3]
root_distance = norm(relative_root_translation, dim=-1)       # [B,30,1]
relative_root_orientation = matrix_to_rotation_6d(R_A^T R_B) # [B,30,6]
```

为了 temporal length 对齐，velocity 第一帧补 0 或复制第一段速度。第一版优先补 0，并记录在 args。

relation feature 最小拼接：

```text
rel_feat: [B, 30, 13]
3 translation + 3 velocity + 1 distance + 6 relative orientation
```

### 模型结构

第一版：

```text
person_encoder(A_obs) -> h_A
person_encoder(B_obs) -> h_B
relation_encoder(rel_feat) -> h_rel
fuse([h_A, h_B, h_rel]) -> joint_decoder -> pred [B,120,2,147]
```

建议模块：

```text
person_encoder: shared GRU(input_dim=147, hidden_dim=256/512, num_layers=2)
relation_encoder: GRU(input_dim=13, hidden_dim=128/256, num_layers=1/2)
fusion: concat + MLP
decoder: MLP(fused_dim -> pred_len * 294)
```

第一版不使用 graph network。只有当 relation-aware 不能赢 concat 且特征/训练已排查后，才考虑 cross-attention 或 graph。

### 参数公平性

必须记录：

```text
independent_num_params
concat_num_params
relation_num_params
training_steps
effective_batch_size
seed
```

P5 中必须加入 parameter-matched 或 no-relation-encoder ablation，避免 relation-aware 只是参数更多。

### Loss

主训练：

```text
loss = normalized active-vector MSE
```

可选 ablation：

```text
loss += lambda_dist * relative_root_distance_loss
loss += lambda_orient * relative_orientation_loss
```

第一版不在主模型加入 relation loss。先看同一 reconstruction loss 下 relation architecture 是否有效。

### P4 验收

必须通过：

```text
relation-aware smoke 通过。
relation-aware checkpoint 可加载。
relation-aware test metrics 完整输出。
relation-aware future_mse 不差于 concat。
relation-aware long_mse 优于 concat。
relation-aware 至少一个 relation metric 优于 concat。
记录 relation features、参数量、训练预算、seed。
```

论文主张最低门槛：

```text
long_mse 和至少一个 relation metric 明显优于 concat no-relation。
```

如果只赢 independent、不赢 concat，不能声称 relation-aware joint modeling 成立。

### P4 失败回退

```text
先检查 relation feature 数值和 shape。
再检查 concat baseline 是否公平且参数量接近。
再加入 relation loss 做 ablation。
再考虑 cross-attention。
最后才考虑 graph network。
```

不得在 P4 失败时直接进入论文包装。

### P4 阶段记录

完成后新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p4-relation-result.md
```

记录：

```text
relation features
模型配置
参数量
训练命令
checkpoint
与 P3 baselines 的完整对比
是否满足论文主张门槛
是否允许进入 P5
```

## P5 设计：Ablation + Paper Tables

### 目标

证明 relation-aware 不是偶然变好，生成论文可用主表和消融表。

### 实验矩阵

主实验：

```text
Repeat
Independent
Concat no-relation
Relation-aware
```

主协议：

```text
window_len=150
obs_len=30
pred_len=120
```

重复实验：

```text
至少 3 seeds: 0, 1, 2
报告 mean/std 或 mean±std
```

结构消融：

```text
concat no-relation
relation-aware without relation encoder
relation-aware with relation encoder
parameter-matched concat
```

关系特征消融：

```text
relative translation only
relative velocity only
relative orientation only
translation + velocity
all relation features
```

观测比例消融：

```text
10%: obs=15, pred=135
20%: obs=30, pred=120
30%: obs=45, pred=105
50%: obs=75, pred=75
```

20% 是主协议，其他比例是补充结果。

可选：

```text
hidden_dim 256 / 512
with / without relation loss
GRU vs small TransformerEncoder
```

### 表格设计

主结果表列：

```text
method
params
future_mse
rotation_mse
translation_mse
long_mse
relative_root_distance_error
relative_orientation_error
inter_person_distance_consistency
```

消融表列：

```text
variant
params
future_mse
long_mse
relative_root_distance_error
relative_orientation_error
inter_person_distance_consistency
```

观测比例表：

```text
obs_ratio
method
future_mse
long_mse
relation_metrics
```

所有表格数据必须来自同一 `eval_forecasting` 协议。

### 结果聚合

建议新增或集成：

```text
eval/eval_forecasting.py --mode aggregate
```

输入：

```text
save/forecasting/interhuman/{run_name}/metrics_test.json
```

输出：

```text
results/forecasting/interhuman/tables/main_results.csv
results/forecasting/interhuman/tables/main_results.md
results/forecasting/interhuman/tables/ablation.csv
results/forecasting/interhuman/tables/obs_ratio.csv
```

### P5 验收

必须通过：

```text
所有主表结果至少 3 seed。
所有实验记录 checkpoint、args、metrics。
所有表格指标来自同一 evaluator。
主表支持论文核心主张。
ablation 支持 relation features 或 relation encoder 的必要性。
失败或不稳定实验被记录。
```

### P5 失败回退

```text
主表不支持主张：回到 P4，不写成功结论。
ablation 不支持主张：收缩贡献表述，或重新设计 relation module。
观测比例不稳定：固定 20% 主协议，其他比例作为补充或限制。
单 seed 波动大：增加 seed，不挑选有利结果。
```

### P5 阶段记录

完成后新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p5-ablation-tables.md
```

记录：

```text
所有实验清单
seed 列表
checkpoint 和 args 路径
表格数据源
是否满足论文主张
失败或不稳定实验
是否允许进入 P6
```

## P6 设计：Visualization + Qualitative Analysis

### 目标

用定性结果解释 relation-aware 是否改善双人关系，而不是用可视化替代指标。

### 新增文件

```text
sample/visualize_forecasting.py
```

也可以先把可视化逻辑放在 `eval/eval_forecasting.py --mode visualize`，但如果逻辑超过一个入口，应拆到 `sample/visualize_forecasting.py`。

### 输入

```text
checkpoint paths
test dataset
normalizer
selected sample ids
```

### 输出目录

```text
results/forecasting/interhuman/{run_name}/qualitative/{sample_id}/
```

每个样本保存：

```text
meta.json
obs.npy
gt.npy
pred_repeat.npy
pred_independent.npy
pred_concat.npy
pred_relation.npy
distance_curve.png
orientation_curve.png
metrics_per_sample.json
```

可选：

```text
obs_h5_like.npy
gt_h5_like.npy
pred_relation_h5_like.npy
rendered frames or videos
```

### 样本选择

至少 8 个 test samples：

```text
2 个 relation-aware 明显优于 concat。
2 个 relation-aware 与 concat 接近。
2 个 relation-aware 失败或更差。
2 个边界样本，覆盖较短 T>=150 和较长序列。
```

不得只保留成功样本。

### 曲线

必须输出：

```text
relative root distance over obs + future
relative orientation error over future
long-horizon segment error curve
```

曲线必须与 metrics 使用同一基础函数，避免图表和指标不一致。

### P6 验收

必须通过：

```text
至少 8 个样本输出完整。
所有 pred 数值有限。
obs / gt / pred 时间轴对齐。
distance / orientation 曲线与 metrics 一致。
失败样本被记录。
render 卡住时仍有 npy + curves 可用于论文分析。
```

### P6 失败回退

```text
render 卡住：先输出 npy + 曲线图，不阻塞论文主指标。
可视化显示动作崩坏但 MSE 低：回到 P4/P5 检查指标遗漏。
样本挑选偏置：固定 selection rule，重新生成。
```

### P6 阶段记录

完成后新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p6-qualitative-result.md
```

记录：

```text
样本列表
输出路径
成功案例
失败案例
图表说明
是否支持 P5 结论
```

## 训练参数建议

初始 smoke：

```text
batch_size=8
num_steps=5
max_samples=64
hidden_dim=256
num_layers=2
lr=1e-3
grad_accum_steps=1
num_workers=0
```

中等训练：

```text
batch_size=32 或 64
num_steps=5000-20000
hidden_dim=256
num_layers=2
lr=1e-3 起，必要时 3e-4
grad_accum_steps 按显存设置
save_interval=1000
eval_interval=1000
```

主实验：

```text
所有方法共享同一 batch_size、optimizer、num_steps、eval split、seed 列表。
relation-aware 额外参数必须在 P5 中通过参数匹配消融解释。
```

## CLI 参数 contract

`train/train_forecasting.py`：

```text
--dataset interhuman
--data_path dataset/interhuman/smpl/conditioned
--save_dir save/forecasting/interhuman/{run_name}
--model_type independent|concat|relation
--window_len 150
--obs_len 30
--pred_len 120
--batch_size
--num_steps
--lr
--weight_decay
--hidden_dim
--num_layers
--relation_hidden_dim
--grad_accum_steps
--max_samples
--seed
--num_workers 0
--save_interval
--eval_interval
--resume_checkpoint
```

`eval/eval_forecasting.py`：

```text
--mode dataset_smoke|metrics_sanity|repeat|checkpoint|aggregate
--dataset interhuman
--data_path dataset/interhuman/smpl/conditioned
--split val|test
--checkpoint
--model_type repeat|independent|concat|relation
--normalizer
--save_dir
--batch_size
--num_workers 0
```

## 论文证据链

论文主张成立需要同时满足：

```text
1. P2 证明 metrics 正确。
2. P3 证明 trainable baselines 优于 repeat。
3. P4 证明 relation-aware 优于 concat 的 long_mse 和至少一个 relation metric。
4. P5 证明结果跨 seed 稳定，且 ablation 支持 relation module / features。
5. P6 展示成功和失败案例，解释指标背后的行为。
```

不能使用的论证：

```text
只赢 repeat baseline。
只赢 independent predictor。
只在 future_mse 上微弱领先。
只展示成功可视化。
单 seed 结果直接写主结论。
```

## 风险与处理

### 风险 1：确定性预测过难

处理：

```text
第一阶段只声称 deterministic forecasting。
多模态 future、best-of-K、diversity 放到 P7+。
```

### 风险 2：concat baseline 太强

处理：

```text
承认 concat 能看到双人历史。
把贡献表述为显式关系归纳偏置。
用 parameter-matched ablation 支撑。
```

### 风险 3：rotation / translation 尺度不一致

处理：

```text
训练用 normalizer。
报告原始尺度指标。
rotation_mse 和 translation_mse 分开报告。
```

### 风险 4：SMPL reproduction 与官方 SMPL-X 口径不同

处理：

```text
论文中明确第一阶段是 SMPL reproduction protocol。
不要声称完全对齐 InterHuman / InterGen 官方 SMPL-X 设置。
```

### 风险 5：可视化和指标冲突

处理：

```text
如果可视化崩坏但 MSE 低，优先修指标和分析。
不使用可视化掩盖主指标失败。
```

## 阶段记录模板

每个阶段结束必须新建文档，包含：

```text
1. 引用的路线图和本设计文件。
2. 实际实现文件。
3. 是否偏离设计。
4. 偏离原因。
5. 验收命令。
6. 验收结果。
7. 输出路径。
8. 是否允许进入下一阶段。
```

如果未达验收标准：

```text
不能写为完成。
不能进入下一阶段做论文结论。
只能写修复记录和继续修复实现。
```

## 最短执行路径

建议后续实现顺序：

```text
1. P1 active vector + dataset + normalizer smoke
2. P2 metrics sanity + repeat baseline
3. P3 concat baseline smoke / train / eval
4. P3 independent baseline smoke / train / eval
5. P4 relation-aware smoke / train / eval
6. P5 3-seed 主表
7. P5 relation feature ablation
8. P6 qualitative npy + curves
9. P6 optional render
```

下一步只允许从 P1 开始。
