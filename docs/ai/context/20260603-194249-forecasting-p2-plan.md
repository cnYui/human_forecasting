# Forecasting P2 Metrics + Repeat Baseline 计划文档

## 文档定位

本文使用 `using-superpowers` 工作流生成，是以下正式设计的 P2 落地计划：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
```

上游依赖：

```text
docs/ai/context/20260603-184214-forecasting-p1-p6-complete-design.md
docs/ai/context/20260603-161334-forecasting-p1-p2-design.md
docs/ai/context/20260603-190529-forecasting-p1-plan.md
docs/ai/context/20260603-191712-forecasting-p1-dataset-result.md
```

本文只规划 P2，不进入实现结果记录。P2 完成后必须新建结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p2-metrics-repeat-result.md
```

## P2 目标

在训练任何 forecasting 模型前，先完成评估闭环：

```text
original-scale metrics
metrics_sanity
repeat / zero-velocity baseline
test split 完整评估
固定 metrics schema
json/yaml 结果落盘
```

P2 通过后，P3 的 independent predictor、concat no-relation predictor 和后续 relation-aware predictor 才能使用同一套 evaluator 比较。

## 必须解决的问题

从第一性原理看，P2 只解决三件事：

```text
1. 定义唯一 original-scale 指标 schema，避免后续每个模型各算各的。
2. 用 pred == target 的 sanity check 证明指标本身正确，不把 evaluator bug 带入训练阶段。
3. 用 repeat baseline 跑完整 test split，得到最低对照和评估文件格式样板。
```

P2 的核心验收不是 repeat 指标好不好，而是评估链路可信、可复用、可追溯。

## 非目标

P2 不做：

```text
independent predictor
concat no-relation predictor
relation-aware model
训练循环
checkpoint 训练
可视化
diffusion forecasting
ReGenNet Table 4 evaluator
NTU120-AS / Chi3D-AS 接入
multi-window eval
normalized-space 论文指标
```

P2 不重新定义 P1 数据协议：

```text
InterHumanForecastDataset
forecasting_collate
active vector [T,2,147]
window_len=150
obs_len=30
pred_len=120
train random crop
val/test center crop
T<150 过滤
```

P2 不修改旧 ReGenNet 主路径：

```text
train/train_mdm.py
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
data_loaders/tensors.py::ccollate
```

## 当前代码基线

P1 已落地：

```text
utils/forecasting_motion.py
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
eval/eval_forecasting.py
```

当前 `eval/eval_forecasting.py` 只支持：

```text
--mode dataset_smoke
```

P2 应扩展现有入口，不另起一个平行 evaluator。

## 新增与修改文件

新增：

```text
utils/forecasting_metrics.py
```

修改：

```text
eval/eval_forecasting.py
```

不修改：

```text
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
utils/forecasting_motion.py
```

除非 P2 发现 P1 contract 实现错误。若发现错误，必须先修 P1 contract 并在 P2 结果文档中说明。

## Metrics Contract

输入必须是 original scale：

```text
pred:   Tensor[B,120,2,147]
target: Tensor[B,120,2,147]
obs:    Tensor[B,30,2,147]
```

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

所有输出为 Python float，便于 JSON/YAML 序列化。

### MSE 类指标

定义：

```text
future_mse = mean((pred - target)^2)
rotation_mse = mean((pred[..., :144] - target[..., :144])^2)
translation_mse = mean((pred[..., 144:147] - target[..., 144:147])^2)
short_mse = future_mse over pred frames 0:40
mid_mse = future_mse over pred frames 40:80
long_mse = future_mse over pred frames 80:120
```

实现约束：

```text
pred_len 第一版固定 120。
如果后续支持非 120 pred_len，必须先单独记录协议变更。
```

### relative_root_distance_error

取两人的 root translation：

```text
trans_A = x[:, :, 0, 144:147]
trans_B = x[:, :, 1, 144:147]
dist = norm(trans_A - trans_B, dim=-1)
relative_root_distance_error = mean(abs(pred_dist - target_dist))
```

### relative_orientation_error

取每个人 active vector 的 root rot6d：

```text
root_A = x[:, :, 0, 0:6]
root_B = x[:, :, 1, 0:6]
```

使用项目已有函数：

```text
utils.rotation_conversions.rotation_6d_to_matrix
```

定义：

```text
R_A = rotation_6d_to_matrix(root_A)
R_B = rotation_6d_to_matrix(root_B)
R_rel = R_A.transpose(-1, -2) @ R_B
R_err = R_rel_pred.transpose(-1, -2) @ R_rel_target
angle = acos(clamp((trace(R_err) - 1) / 2, -1 + eps, 1 - eps))
relative_orientation_error = mean(angle)
```

报告单位：

```text
radian
```

防护：

```text
acos 输入必须 clamp，避免 NaN。
所有 metrics 输出必须 finite。
```

### inter_person_distance_consistency

第一版只报告 root distance trend consistency：

```text
last_obs_dist = norm(obs[:, -1, 0, 144:147] - obs[:, -1, 1, 144:147])
last_obs_dist = last_obs_dist.unsqueeze(1)
dist_full_pred = concat(last_obs_dist, pred_dist, dim=1)
dist_full_target = concat(last_obs_dist, target_dist, dim=1)
delta_pred = dist_full_pred[:, 1:] - dist_full_pred[:, :-1]
delta_target = dist_full_target[:, 1:] - dist_full_target[:, :-1]
inter_person_distance_consistency = mean(abs(delta_pred - delta_target))
```

论文表述限制：

```text
该指标只能解释为两人 root distance 变化趋势一致性。
不能声称它覆盖完整 interaction quality。
```

## Aggregation Contract

批级 metrics 不能直接做简单平均，除非所有 batch 大小完全一致。

P2 实现应按样本数加权聚合：

```text
batch_metric_sum += metric_value * batch_size
num_samples += batch_size
final_metric = batch_metric_sum / num_samples
```

MSE 和 relation metrics 当前都是 batch 内均值，因此按 batch size 加权足够。最后一个 batch 小于 `batch_size` 时不会污染结果。

## Repeat Baseline

定义：

```text
pred[:, t] = obs[:, -1]
```

shape：

```text
obs[:, -1]: [B,2,147]
pred:       [B,120,2,147]
```

实现建议：

```text
last = obs[:, -1:].contiguous()
pred = last.expand(-1, pred_len, -1, -1).contiguous()
```

repeat 又称 zero-velocity baseline。它不是论文贡献，只是 sanity baseline 和最低对照。

## eval_forecasting 扩展计划

`eval/eval_forecasting.py` 的 `--mode` 扩展为：

```text
dataset_smoke
metrics_sanity
repeat
checkpoint
```

P2 只实现：

```text
metrics_sanity
repeat
```

`checkpoint` 可先保留为明确的 `NotImplementedError`，给 P3/P4 使用。

### metrics_sanity

构造真实 batch：

```text
dataset = InterHumanForecastDataset(split="test")
obs, target, meta = next(loader)
pred = target.clone()
metrics = compute_forecasting_metrics(pred, target, obs)
```

验收：

```text
future_mse == 0
rotation_mse == 0
translation_mse == 0
short_mse == 0
mid_mse == 0
long_mse == 0
relative_root_distance_error == 0
relative_orientation_error 接近 0
inter_person_distance_consistency == 0
所有 key 完整且顺序稳定
所有值 finite
```

`relative_orientation_error` 允许浮点容忍：

```text
<= 1e-4
```

原因是 rot6d 正交化和 acos clamp 会引入极小数值误差。

输出：

```text
save/forecasting/interhuman/p2_metrics_sanity/metrics_sanity.json
save/forecasting/interhuman/p2_metrics_sanity/metrics_sanity.yaml
```

### repeat

加载 split：

```text
split=test
shuffle=False
num_workers=0
```

逐 batch：

```text
obs, target, meta = batch
pred = repeat_last_observation(obs, pred_len)
metrics = compute_forecasting_metrics(pred, target, obs)
aggregate(metrics, batch_size)
```

输出：

```text
save/forecasting/interhuman/repeat_150_30_120/metrics_test.json
save/forecasting/interhuman/repeat_150_30_120/metrics_test.yaml
```

结果文件至少包含：

```text
mode
dataset
split
data_path
window_len
obs_len
pred_len
batch_size
num_workers
num_samples
metrics
metrics_keys
created_at
```

`metrics_keys` 必须与固定 schema 完全一致。

## YAML 输出策略

JSON 使用标准库 `json` 写出。

YAML 不新增依赖：

```text
如果环境已有 yaml 包，可使用 yaml.safe_dump。
如果没有 yaml 包，使用固定 key/value 的纯文本 YAML 写出。
```

原因是 P2 的重点是评估合同，不应为了结果格式引入额外依赖风险。

## 验收命令

metrics sanity：

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
  --save_dir save/forecasting/interhuman/p2_metrics_sanity
```

repeat baseline：

```bash
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

P2 通过后新建结果记录：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p2-metrics-repeat-result.md
```

## 实现顺序

1. 新增 `utils/forecasting_metrics.py`。
2. 实现 shape/finite 检查、固定 metrics key 和 `compute_forecasting_metrics`。
3. 实现 `repeat_last_observation` 或放在 `eval_forecasting.py` 内部作为局部 helper。
4. 扩展 `eval/eval_forecasting.py --mode metrics_sanity`。
5. 扩展 `eval/eval_forecasting.py --mode repeat`。
6. 写出 JSON/YAML 结果文件。
7. 跑 metrics sanity 命令。
8. 跑 repeat test 命令。
9. 新建 P2 结果文档，记录 sanity、repeat 指标、输出路径和是否允许进入 P3。

## P2 验收标准

必须全部满足：

```text
pred == target 时 MSE 类指标为 0。
pred == target 时 relative_root_distance_error 为 0。
pred == target 时 relative_orientation_error <= 1e-4。
pred == target 时 inter_person_distance_consistency 为 0。
repeat baseline 可完整评估 test split。
repeat 使用 test split 的 508 条 T>=150 样本。
metrics 输出 key 固定。
metrics 文件同时保存 json 和 yaml。
所有指标 finite。
P2 结果文档已新建。
```

P2 不过，不进入 P3。

## 失败回退

```text
pred == target MSE 非 0：先修 slicing / dtype / aggregation。
relative_root_distance_error 非 0：检查 translation slot 是否仍为 active[...,144:147]。
relative_orientation_error NaN：检查 rotation_6d_to_matrix 输入、trace clamp 和 finite 防护。
inter_person_distance_consistency 非 0：检查 last_obs_dist 拼接和 delta 对齐。
repeat 跑不完：先查 DataLoader / collate / max_samples / split。
metrics key 不稳定：固定 schema，不进入模型训练。
repeat 指标异常好：检查 target 泄漏和窗口切分。
```

## P2 完成后允许进入 P3 的条件

只有同时满足以下条件，才允许实现可训练 baseline：

```text
metrics_sanity 通过。
repeat test split 完整结果已保存。
P2 结果文档已写明 metrics 文件路径。
AGENTS.md 已记录 P2 完成状态和下一步 P3。
```

P3 必须复用 P2 evaluator，不允许为 independent / concat baseline 另写一套指标。
