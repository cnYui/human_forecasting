# 交互感知双人动作联合预测最终正式设计文档

## 文档定位

本文件是当前项目第一篇论文工程实现的最终正式设计文档。后续实现、验收、阶段记录和论文结果解释均以本文件为准。

本文件整合并收敛以下文档：

```text
docs/ai/context/20260603-184214-forecasting-p1-p6-complete-design.md
docs/ai/context/20260603-185018-three-datasets-usage-review.md
docs/ai/context/20260603-185530-ntu-chi3d-after-interhuman-plan.md
```

本文件不覆盖历史文档。历史文档用于追溯讨论过程；本文件用于执行。

## 最终研究目标

英文题目：

```text
Interaction-aware joint forecasting of two-person human motion from partial observations.
```

中文题目：

```text
基于部分观测的交互感知双人动作联合未来预测。
```

核心问题：

```text
给定一段双人交互动作的前 20% 观测，联合预测两个人后续 80% 的未来动作。
```

第一篇论文只验证一个清晰主张：

```text
双人未来动作预测不是两个单人未来预测的简单拼接。
显式建模两人关系应改善长期预测误差和交互关系一致性。
```

该主张必须可证伪：

```text
如果 relation-aware model 不能稳定优于 concat no-relation baseline，
尤其不能改善 long-horizon error 和至少一个 relation metric，
则不能声称交互感知建模有效。
```

## 数据集最终决策

### 第一阶段主数据集

第一阶段只使用：

```text
InterHuman
```

固定协议：

```text
source: dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5
representation: SMPL reproduction active vector
window_len: 150
obs_len: 30
pred_len: 120
input: 前 30 帧双人动作
target: 后 120 帧双人动作
task: deterministic two-person forecasting
```

### 不合并三套数据集

以下三套数据集不是 train / val / test：

```text
NTU120-AS
Chi3D-AS
InterHuman-AS
```

它们是 ReGenNet 论文中的三个独立 benchmark 数据域。第一阶段不做：

```text
把 NTU120-AS、Chi3D-AS、InterHuman-AS 合并训练。
把 NTU120-AS 当 train、Chi3D-AS 当 val、InterHuman-AS 当 test。
把 NTU/Chi3D 作为 InterHuman P1-P6 的阻塞项。
```

原因：

```text
InterHuman 当前本地表示: [T,25,12], SMPL, 无动作类别主标签。
NTU120-AS 表示: [T,56,6], SMPL-X, 26 类动作。
Chi3D-AS 表示: [T,56,6], SMPL-X, 8 类动作。
三者帧长、动作标签、normalizer、evaluator 和人体模型表示均不一致。
```

直接合并会把当前问题改成 multi-domain forecasting，不是第一篇论文需要解决的问题。

### 后续扩展数据集

InterHuman P1-P6 成立后，后续阶段再考虑：

```text
P7: NTU120-AS action-conditioned forecasting / SMPL-X 大样本对照。
P8: Chi3D-AS 高质量小样本 SMPL-X 泛化验证或 qualitative 补充。
P9: 统一多数据集训练，作为单独研究问题。
```

## 本地数据事实

### InterHuman

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5
dataset/interhuman/smpl/conditioned/meta.json
```

格式：

```text
shape: [T,25,12]
body_model: smpl
rotation: rot6d
translation_slot: 24
translation_origin: actor frame 0
```

样本数：

```text
train: total 6021, T>=150 2910
val:   total 580,  T>=150 226
test:  total 1175, T>=150 508
```

### NTU120-AS

```text
dataset/ntu120/smplx/conditioned/xsub.train.h5
dataset/ntu120/smplx/conditioned/xsub.test.h5

train: 4273
test: 3845
total: 8118
shape: [T,56,6]
recognition checkpoint: recognition_training/ntu_smplx/checkpoint_0100.pth.tar
```

### Chi3D-AS

```text
dataset/chi3d/smplx/conditioned/chi3d_smplx_train.h5
dataset/chi3d/smplx/conditioned/chi3d_smplx_test.h5

train: 293
test: 74
total: 367
shape: [T,56,6]
recognition checkpoint: recognition_training/chi3d_smplx/checkpoint_0060.pth.tar
```

### InterHuman 与 ReGenNet 原论文的差异

当前本地 InterHuman 是 SMPL reproduction H5：

```text
[T,25,12]
```

NTU/Chi3D 是 SMPL-X：

```text
[T,56,6]
```

当前本地没有：

```text
InterHuman text/caption 目录
InterHuman recognition checkpoint
eval_cmdm.py 的 interhuman evaluator 分支
```

因此第一篇 forecasting 论文不依赖 ReGenNet Table 4 的 ST-GCN evaluator，不追 InterHuman-AS Table 4 复现。

## 当前项目旧路径与新路径的关系

### 旧 ReGenNet 路径

旧路径任务：

```text
actor 全段动作 -> generated reactor 全段动作
```

旧路径代码：

```text
train/train_mdm.py
data_loaders/get_data.py
data_loaders/a2m/feeder.py
data_loaders/a2m/interhuman.py
data_loaders/tensors.py::ccollate
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
sample/cgenerate.py
```

关键语义：

```text
ccollate 将双人数据按 channel 拆为 actor 条件 cmotion 和 reactor 目标 motion。
```

### 新 forecasting 路径

新路径任务：

```text
双人前缀观测 -> 双人未来预测
```

新路径不复用 `ccollate`，不改 diffusion 主路径。

第一阶段不得修改：

```text
train/train_mdm.py
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
```

除非后续单独写扩展文档并说明为何需要接入旧 ReGenNet 路径。

## 第一阶段协议

固定窗口：

```text
window_len = 150
obs_len = 30
pred_len = 120
```

长度规则：

```text
T >= 150: 使用
T < 150: 过滤
```

不得 padding：

```text
padding 会伪造未来动作，破坏 forecasting 任务定义。
```

训练采样：

```text
train: random crop 150 frames
val/test: center crop 150 frames
```

归一化：

```text
使用 train split 统计 normalizer。
训练 loss 在 normalized space。
所有论文指标在 original scale。
```

## 数据表示 contract

### H5 原始表示

```text
motion: [T,25,12]

actor_rot6d:         motion[:, :24, 0:6]
actor_translation:   motion[:, 24, 0:3]
reactor_rot6d:       motion[:, :24, 6:12]
reactor_translation: motion[:, 24, 6:9]
```

### Active vector

```text
person_dim = 24 * 6 + 3 = 147
two_person_dim = 2 * 147 = 294
motion_active: [T,2,147]
```

映射：

```text
motion_active[:, 0, :144] = actor_rot6d.reshape(T, 144)
motion_active[:, 0, 144:147] = actor_translation
motion_active[:, 1, :144] = reactor_rot6d.reshape(T, 144)
motion_active[:, 1, 144:147] = reactor_translation
```

窗口：

```text
obs:    [30,2,147]
target: [120,2,147]
```

### Restore

active vector 必须能还原为 H5-like motion：

```text
active [T,2,147] -> motion [T,25,12]
```

用途：

```text
可视化
rot2xyz 输入
后续 qualitative analysis
```

## Normalizer contract

统计来源：

```text
split: train only
samples: T>=150 train sequences
frames: 每条可用序列的全部帧
space: active vector original scale
```

形状：

```text
mean: [1,1,2,147]
std:  [1,1,2,147]
eps: 1e-6
```

防护：

```text
std < eps 的维度设为 1.0
```

保存：

```text
normalizer.pt
normalizer.json
```

`normalizer.json` 至少记录：

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

## Metrics contract

输入：

```text
pred:   Tensor[B,120,2,147]
target: Tensor[B,120,2,147]
obs:    Tensor[B,30,2,147]
```

要求：

```text
pred / target / obs 均为 original scale。
metrics key 固定，所有模型共用同一 evaluator。
```

必须输出：

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
short_mse = future_mse over frames 0:40
mid_mse = future_mse over frames 40:80
long_mse = future_mse over frames 80:120
```

相对 root distance：

```text
dist = norm(trans_A - trans_B)
relative_root_distance_error = mean(abs(pred_dist - target_dist))
```

相对朝向：

```text
root rot6d -> rotation_6d_to_matrix
R_rel = R_A^T R_B
angle = acos(clamp((trace(R_err) - 1) / 2))
relative_orientation_error = mean(angle)
```

必须 clamp：

```text
acos 输入必须 clamp 到 [-1 + eps, 1 - eps]，避免 NaN。
```

inter-person distance consistency：

```text
使用 obs 最后一帧和 future frames 的 root distance delta。
报告 root distance trend consistency，不声称覆盖完整 interaction quality。
```

## 新增文件设计

第一阶段新增：

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

职责：

```text
data_loaders/forecasting/interhuman.py
  InterHuman H5 读取、T>=150 过滤、random/center crop、dataset item。

data_loaders/forecasting/tensors.py
  forecasting_collate，不做 padding。

utils/forecasting_motion.py
  active vector extract/restore、normalizer、relation feature 基础函数。

utils/forecasting_metrics.py
  original scale metrics。

model/forecasting.py
  repeat helper、independent、concat、relation-aware predictor。

train/train_forecasting.py
  supervised training loop、checkpoint、args、normalizer、val/test eval。

eval/eval_forecasting.py
  dataset smoke、metrics sanity、repeat baseline、checkpoint evaluation、aggregate。

sample/visualize_forecasting.py
  npy、曲线图、可选 render。
```

## P1：Dataset + Normalizer

目标：

```text
建立 InterHuman forecasting 数据协议，让后续所有 baseline 和模型读取完全一致的 obs / target。
```

输出 item：

```text
{
  "obs": Tensor[30,2,147],
  "target": Tensor[120,2,147],
  "sample_id": str,
  "start": int,
  "length": int,
}
```

batch：

```text
obs: Tensor[B,30,2,147]
target: Tensor[B,120,2,147]
meta: list[dict]
```

验收：

```text
train dataset length = 2910
val dataset length = 226
test dataset length = 508
obs shape = [30,2,147]
target shape = [120,2,147]
所有 obs/target 有限
train 同一样本多次读取 start 可变化
val/test 同一样本多次读取 start 固定
normalizer 可保存和加载
normalize -> denormalize 误差在浮点容忍范围
active -> h5-like -> active 误差在浮点容忍范围
```

## P2：Metrics + Repeat Baseline

目标：

```text
在训练模型前完成评估闭环。
```

repeat baseline：

```text
pred[:, t] = obs[:, -1]
```

验收：

```text
pred == target 时所有 MSE 类指标为 0。
pred == target 时 relative_root_distance_error 为 0。
pred == target 时 relative_orientation_error 接近 0。
repeat baseline 可完整评估 test split。
metrics 文件保存为 json/yaml。
所有指标 key 固定。
```

P2 不过，不进入 P3。

## P3：Independent / Concat Baselines

目标：

```text
建立两个可训练 baseline，防止 relation-aware 只赢 repeat baseline。
```

### Independent Predictor

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

### Concat No-Relation Predictor

定义：

```text
concat(A_obs, B_obs) -> future(A,B)
```

约束：

```text
可以看到双人原始历史。
不能显式构造 relation features。
不能使用 relation_encoder。
```

论文表述：

```text
concat 不是没有关系信息，而是没有显式关系归纳偏置。
```

训练：

```text
loss = normalized active-vector MSE
optimizer = AdamW
num_steps = optimizer steps
支持 grad_accum_steps
num_workers 默认 0
```

验收：

```text
independent smoke 通过。
concat smoke 通过。
两个模型 checkpoint 可保存和加载。
两个模型都能输出 P2 固定 metrics。
两个模型至少优于 repeat baseline 的 future_mse。
记录参数量、训练预算、seed。
```

P3 不过，不进入 P4。

## P4：Relation-Aware Joint Predictor

目标：

```text
实现论文核心模型：显式关系推理 + 双人联合未来预测。
```

关系特征：

```text
relative root translation: trans_A - trans_B
relative root velocity: velocity_A - velocity_B
root distance: ||trans_A - trans_B||
relative root orientation: R_A^T R_B
```

第一版结构：

```text
person_encoder(A_obs) -> h_A
person_encoder(B_obs) -> h_B
relation_encoder(relation_features) -> h_rel
fuse(h_A, h_B, h_rel) -> joint_decoder -> future(A,B)
```

推荐：

```text
person_encoder: shared GRU
relation_encoder: GRU
fusion: concat + MLP
decoder: MLP direct prediction
```

第一版不使用 graph network。

主 loss：

```text
normalized active-vector MSE
```

可选 ablation：

```text
+ relative_root_distance_loss
+ relative_orientation_loss
```

验收：

```text
relation-aware smoke 通过。
checkpoint 可加载。
test metrics 完整输出。
future_mse 不差于 concat。
long_mse 优于 concat。
至少一个 relation metric 优于 concat。
记录 relation features、参数量、训练预算、seed。
```

论文主张最低门槛：

```text
long_mse 和至少一个 relation metric 明显优于 concat no-relation。
```

如果只赢 independent，不赢 concat，论文核心贡献不成立。

## P5：Ablation + Paper Tables

目标：

```text
证明 relation-aware 不是偶然变好，生成论文可用主表和消融表。
```

主实验：

```text
Repeat
Independent
Concat no-relation
Relation-aware
```

重复实验：

```text
至少 3 seeds: 0, 1, 2
报告 mean/std 或 mean±std
```

必做消融：

```text
concat no-relation
relation-aware without relation encoder
relation-aware with relation encoder
parameter-matched concat
relative translation only
relative velocity only
relative orientation only
all relation features
```

观测比例补充：

```text
10%: obs=15, pred=135
20%: obs=30, pred=120
30%: obs=45, pred=105
50%: obs=75, pred=75
```

主表仍以 20% 为主协议。

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

验收：

```text
所有主表结果至少 3 seed。
所有实验记录 checkpoint、args、metrics。
所有表格指标来自同一 evaluator。
主表支持论文核心主张。
ablation 支持 relation module / relation features。
失败或不稳定实验被记录。
```

P5 不支持主张时，不写成功结论，回到 P4。

## P6：Visualization + Qualitative Analysis

目标：

```text
用定性结果解释模型如何改善或未改善双人关系。
```

输出：

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

样本选择：

```text
至少 8 个 test samples。
包含 relation-aware 明显优于 concat 的样本。
包含 relation-aware 与 concat 接近的样本。
包含 relation-aware 失败或更差的样本。
包含短序列边界和长序列样本。
```

验收：

```text
所有 pred 数值有限。
obs / gt / pred 时间轴对齐。
曲线与 metrics 使用同一基础函数。
失败样本被记录。
render 卡住时仍保留 npy + curves。
```

P6 不能替代 P5 主指标。

## CLI contract

训练入口：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/{run_name} \
  --model_type independent|concat|relation \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size N \
  --num_steps N \
  --hidden_dim N \
  --num_layers N \
  --lr LR \
  --grad_accum_steps N \
  --max_samples -1 \
  --num_workers 0 \
  --seed N
```

评估入口：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode dataset_smoke|metrics_sanity|repeat|checkpoint|aggregate \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split val|test \
  --batch_size N \
  --num_workers 0 \
  --save_dir save/forecasting/interhuman/{run_name}
```

必须使用 `regennet` micromamba 环境。裸 `python3` 不作为验收命令。

## 保存 contract

每个训练 run 保存：

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
```

每个 run 必须记录：

```text
model_type
num_params
seed
effective_batch_size
num_steps
data_path
window_len / obs_len / pred_len
```

## 论文证据链

论文成立需要同时满足：

```text
1. P1 证明数据协议正确。
2. P2 证明 metrics 正确。
3. P3 证明 trainable baselines 优于 repeat。
4. P4 证明 relation-aware 优于 concat 的 long_mse 和至少一个 relation metric。
5. P5 证明结果跨 seed 稳定，且消融支持 relation module / features。
6. P6 展示成功和失败案例，解释指标背后的行为。
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

### 确定性预测过难

处理：

```text
第一篇论文只声称 deterministic forecasting。
best-of-K、diversity、多模态未来放到后续扩展。
```

### Concat baseline 太强

处理：

```text
承认 concat 能看到双人历史。
把贡献定义为显式关系归纳偏置。
做 parameter-matched ablation。
```

### SMPL / SMPL-X 口径不一致

处理：

```text
第一阶段明确为 InterHuman SMPL reproduction protocol。
不声称完全对齐 ReGenNet 官方 SMPL-X 设置。
NTU/Chi3D 的 SMPL-X 对照放到 P7/P8。
```

### 可视化和指标冲突

处理：

```text
如果可视化崩坏但 MSE 低，优先检查 metrics。
不使用可视化掩盖主指标失败。
```

## 阶段记录要求

每个阶段完成后必须新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p{N}-*.md
```

记录：

```text
1. 引用本最终设计文档。
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
只能写修复记录并继续修复。
```

## 最终执行顺序

只允许按以下顺序推进：

```text
1. P1 InterHuman dataset + active vector + normalizer smoke
2. P2 metrics sanity + repeat baseline
3. P3 concat baseline smoke / train / eval
4. P3 independent baseline smoke / train / eval
5. P4 relation-aware smoke / train / eval
6. P5 3-seed main table
7. P5 relation feature ablation
8. P6 qualitative npy + curves
9. P6 optional render
10. P7 NTU120-AS 扩展
11. P8 Chi3D-AS 扩展
```

当前下一步：

```text
开始 P1。
```

不得提前实现：

```text
relation-aware model
NTU forecasting loader
Chi3D forecasting loader
multi-dataset normalizer
SMPL/SMPL-X 统一转换
diffusion forecasting
```
