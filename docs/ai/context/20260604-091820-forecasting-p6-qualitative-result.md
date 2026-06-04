# Forecasting P6 Qualitative Analysis 结果记录

## 文档定位

本文记录 P6 qualitative / paper figure 准备阶段实现和验收结果，依据：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260604-085953-forecasting-p6-qualitative-design.md
docs/ai/context/20260604-090635-forecasting-p6-plan.md
```

## 实现文件

新增：

```text
sample/visualize_forecasting.py
```

修改：

```text
utils/forecasting_metrics.py
```

修改内容：

```text
新增 root_distance_sequence。
新增 relative_orientation_error_sequence。
新增 per_frame_active_mse。
新增 compute_forecasting_metrics_for_sample。
compute_forecasting_metrics 的 metrics key 和 pred_len=120 contract 未改变。
```

## 输出路径

```text
results/forecasting/interhuman/p6_qualitative_150_30_120/
```

根目录输出：

```text
run_config.json
metrics_per_sample_all.json
metrics_per_sample_all.csv
selection.json
selection.csv
summary.md
```

每个选中样本输出：

```text
meta.json
obs.npy
gt.npy
pred_repeat.npy
pred_independent.npy
pred_concat.npy
pred_relation.npy
metrics_per_sample.json
distance_curve.png
orientation_curve.png
long_mse_curve.png
obs_h5_like.npy
gt_h5_like.npy
pred_relation_h5_like.npy
```

## 验收命令

编译检查：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall \
  utils/forecasting_metrics.py sample/visualize_forecasting.py
```

P6 主命令：

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

P2 metrics sanity 回归：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode metrics_sanity \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --batch_size 64 \
  --num_workers 0 \
  --save_dir save/forecasting/interhuman/p6_metrics_sanity_check
```

## 验收结果

已通过：

```text
compileall 通过。
P6 主命令完成。
test split 全量样本数为 508。
选中样本数为 8。
missing_selection_categories 为空。
metrics_per_sample_all.csv 为 509 行，包含 header + 508 samples。
selection.csv 为 9 行，包含 header + 8 samples。
8 个 qualitative sample 目录完整。
所有 obs / gt / pred npy 数值 finite。
三类曲线文件均已生成，文件大小非 0。
P2 metrics_sanity 中 pred == target 的所有固定指标均为 0。
```

## 样本列表

| category | sample_id | length | start | delta_long | delta_root_dist |
| --- | --- | ---: | ---: | ---: | ---: |
| success | 4860 | 150 | 0 | 0.03718680143 | 0.32758769393 |
| success | 4618 | 412 | 131 | 0.02650810033 | 0.16298750043 |
| close | 2508 | 179 | 14 | 0.000005967915 | 0.12066775560 |
| close | 2627 | 242 | 46 | -0.000013232231 | 0.06662647426 |
| failure | 2194 | 440 | 145 | -0.06196782738 | 0.00887624919 |
| failure | 625 | 285 | 67 | -0.02342337742 | -0.02367824316 |
| boundary | 4095 | 150 | 0 | -0.00342843868 | -0.01492154598 |
| boundary | 2613 | 9097 | 4473 | -0.01314016804 | -0.11716908216 |

## 成功样本摘要

```text
sample_id=4860:
delta_long=0.03718680143，delta_root_dist=0.32758769393。
relation 相对 concat 在 long_mse 和 relative root distance 上都明显更好。

sample_id=4618:
delta_long=0.02650810033，delta_root_dist=0.16298750043。
relation 相对 concat 在两个选择指标上都胜出。
```

## 失败样本摘要

```text
sample_id=2194:
delta_long=-0.06196782738。
relation long_mse 明显差于 concat，必须作为 failure case 保留。

sample_id=625:
delta_long=-0.02342337742，delta_root_dist=-0.02367824316。
relation 在 long_mse 和 relative root distance 上都差于 concat。
```

boundary 样本也显示 relation 不是所有样本都优于 concat：

```text
sample_id=4095:
length=150，delta_long=-0.00342843868，delta_root_dist=-0.01492154598。

sample_id=2613:
length=9097，delta_long=-0.01314016804，delta_root_dist=-0.11716908216。
```

## 是否偏离设计

基本未偏离 P6 设计。实际实现多输出了 H5-like npy：

```text
obs_h5_like.npy
gt_h5_like.npy
pred_relation_h5_like.npy
```

这是设计文档允许的可选输出，不影响主验收。

未做：

```text
rendered frames or videos
root_trajectory_xy.png
parameter-matched concat qualitative 输出
```

这些均为可选项，不阻塞 P6。

## 是否支持 P5 结论边界

支持。

P6 qualitative 输出包含 success、close、failure、boundary 四类样本，说明：

```text
relation-aware 可以在部分样本中改善 long-horizon error 和 relative root distance。
relation-aware 不应被描述为全面优于 concat 或 independent。
P6 可视化只能解释 P5 结果，不能替代 P5 全 test aggregate 主表。
```

## 下一步

可以进入论文图表整理：

```text
从 success / failure / boundary 中选择曲线图进入论文 qualitative figure。
写论文时保留 P5/P6 边界，不声称 relation-aware 全面最优。
```

如需动作渲染，应基于本次已保存的 H5-like npy 单独写 render 计划，不要阻塞当前 P6 验收。
