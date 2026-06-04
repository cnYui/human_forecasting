# Forecasting P6 Qualitative Analysis 计划文档

## 文档定位

本文是 P6 qualitative / paper figure 准备阶段的执行计划，依据：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260604-085953-forecasting-p6-qualitative-design.md
docs/ai/context/20260604-085116-forecasting-p5-ablation-result.md
```

本文只规划实现顺序、验收命令和输出路径，不重新定义 P6 设计 contract。

## 当前状态

P1-P5 已完成，P5.3 允许进入 P6：

```text
relation-aware 相对 concat no-relation 和 parameter-matched concat 的 long_mse / relative_root_distance_error 有稳定收益。
relation-aware 不优于 independent，也不优于 repeat 的 relation-style metrics。
```

因此 P6 只做 qualitative / paper figure 准备，不能写成新的主实验证据。

## P6 必须解决的问题

```text
1. 为每个 test 样本生成 repeat / independent / concat / relation 的 original-scale prediction。
2. 计算 sample-level metrics，并保留和 P2 evaluator 一致的口径。
3. 按 success / close / failure / boundary 规则自动选择至少 8 个样本。
4. 为选中样本保存 npy、metrics_per_sample.json 和三类曲线。
5. 生成 summary，说明成功样本和失败样本，不只展示成功案例。
```

## P6 不做

```text
不训练新模型。
不修改 P2 metrics key。
不启动 P5.4 observation-ratio。
不接入 NTU120-AS / Chi3D-AS。
不修改 diffusion / CMDM 主路径。
不把 qualitative 结果当成 P5 主表替代品。
```

## 实现文件计划

新增：

```text
sample/visualize_forecasting.py
```

允许小范围修改：

```text
utils/forecasting_metrics.py
```

修改原则：

```text
sample/visualize_forecasting.py 负责 CLI、checkpoint 加载、全 test 推理、样本选择、落盘和画图。
utils/forecasting_metrics.py 只新增可复用的曲线 / sample-level helper，不改变 compute_forecasting_metrics 的 key 和行为。
eval/eval_forecasting.py 暂不修改，避免 evaluator 职责继续膨胀。
```

## P6.1：样本级指标与曲线 helper

目标：

```text
补足 P6 曲线需要的 per-frame / per-sample 计算，同时保持 P2 aggregate metrics 不漂移。
```

计划新增 helper：

```text
root_distance_sequence(value)
relative_orientation_error_sequence(pred, target)
per_frame_active_mse(pred, target)
compute_forecasting_metrics_for_sample(pred, target, obs)
```

约束：

```text
输入仍使用 original scale。
pred / target / obs 仍是 [B,T,2,147] 或单样本加 batch 维。
compute_forecasting_metrics 的输出 key 和数值不变。
relative orientation 仍复用 rotation_6d_to_matrix 和 acos clamp。
```

验收：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall utils/forecasting_metrics.py
```

额外 sanity：

```text
pred == target 时 sample-level MSE、relative_root_distance_error、relative_orientation_error 为 0 或浮点容忍内接近 0。
```

## P6.2：checkpoint 加载与全 test 推理

目标：

```text
实现 seed0 representative checkpoints 的 deterministic center-crop 推理。
```

checkpoint：

```text
independent:
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt

concat:
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt

relation:
save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt
```

实现要点：

```text
使用 InterHumanForecastDataset(split=test) 保证 center crop。
repeat baseline 直接 repeat obs 最后一帧。
每个 checkpoint 按各自 normalizer normalize obs，再 denormalize prediction。
所有方法对同一个 batch 的同一个 obs / target 推理。
检查 checkpoint config 的 obs_len / pred_len / person_dim / model_type。
```

输出：

```text
metrics_per_sample_all.json
metrics_per_sample_all.csv
run_config.json
```

## P6.3：自动样本选择

目标：

```text
按设计文档固定规则选择至少 8 个样本，避免人工挑成功案例。
```

选择规则：

```text
success: delta_long > 0 且 delta_root_dist > 0，优先 delta_long 最大。
close: abs(delta_long) 最小。
failure: delta_long < 0 或 delta_root_dist < 0，优先 delta_long 最负。
boundary: length 最接近 150 的短序列和 length 最大的长序列。
```

其中：

```text
delta_long = concat.long_mse - relation.long_mse
delta_root_dist = concat.relative_root_distance_error - relation.relative_root_distance_error
```

输出：

```text
selection.json
selection.csv
```

必须记录：

```text
sample_id
length
start
category
selection_reason
delta_long
delta_root_dist
repeat / independent / concat / relation sample-level metrics
```

## P6.4：qualitative 包落盘

目标：

```text
为选中样本生成论文可用的数据包和曲线图。
```

每个样本输出：

```text
qualitative/{sample_id}/meta.json
qualitative/{sample_id}/obs.npy
qualitative/{sample_id}/gt.npy
qualitative/{sample_id}/pred_repeat.npy
qualitative/{sample_id}/pred_independent.npy
qualitative/{sample_id}/pred_concat.npy
qualitative/{sample_id}/pred_relation.npy
qualitative/{sample_id}/metrics_per_sample.json
qualitative/{sample_id}/distance_curve.png
qualitative/{sample_id}/orientation_curve.png
qualitative/{sample_id}/long_mse_curve.png
```

可选：

```text
obs_h5_like.npy
gt_h5_like.npy
pred_relation_h5_like.npy
root_trajectory_xy.png
```

画图要求：

```text
使用 matplotlib Agg 后端，避免无显示环境阻塞。
图中标出 obs_len=30 的分界线。
long_mse_curve 标出 global frame 110..149 的 long horizon 区段。
summary 中说明 sample metrics 不是 P5 全 test aggregate。
```

## P6.5：验收与结果记录

主验收命令：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m sample.visualize_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --independent_checkpoint save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt \
  --concat_checkpoint save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt \
  --relation_checkpoint save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt \
  --save_dir results/forecasting/interhuman/p6_qualitative_150_30_120 \
  --num_samples 8 \
  --batch_size 64 \
  --num_workers 0 \
  --seed 0
```

代码验收：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall \
  utils/forecasting_metrics.py sample/visualize_forecasting.py
```

输出验收：

```text
results/forecasting/interhuman/p6_qualitative_150_30_120/run_config.json 存在。
selection.json / selection.csv 存在。
metrics_per_sample_all.json / metrics_per_sample_all.csv 存在。
至少 8 个 qualitative/{sample_id}/ 目录完整。
success / close / failure / boundary 四类样本都有记录。
所有 pred npy 数值 finite。
三类曲线均可打开。
```

完成后新建结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p6-qualitative-result.md
```

结果文档必须记录：

```text
实现文件
验收命令
输出路径
样本列表
成功样本摘要
失败样本摘要
是否偏离 P6 设计文档
是否支持 P5 结论边界
```

## 风险与处理

### checkpoint normalizer 路径缺失

处理：

```text
先读 checkpoint state["normalizer_path"]。
缺失时回退到 checkpoint 所在目录。
仍缺失则报错，不用全局 normalizer 静默替代。
```

### matplotlib 不可用或后端异常

处理：

```text
使用 Agg 后端。
如果画图失败，先保留 npy、metrics 和 selection，并在结果文档记录阻塞。
```

### 选样类别不足

处理：

```text
记录不足原因。
按未选样本的 delta_long / delta_root_dist 多样性补齐到 num_samples。
不得人工只保留 success 样本。
```

### qualitative 与 aggregate 指标冲突

处理：

```text
优先信任 P5 全 test aggregate。
P6 只解释样本行为，不反转主结论。
如果曲线暴露指标遗漏，再新开设计文档讨论，不在 P6 中临时改 metric。
```

## 下一步

开始实现 `sample/visualize_forecasting.py` 和必要的 `utils/forecasting_metrics.py` helper。实现顺序必须先保证 npy、sample metrics、selection 和 curves，render 后置。
