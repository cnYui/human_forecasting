# NTU xyz 最终评估指标与原论文口径对比

## 结论

当前最终 NTU 双人预测评估和原 ReGenNet 论文主评估口径不一样。

当前最终评估是 forecasting / prediction 口径：

```text
前 20 帧双人 xyz + action label -> 后 40 帧双人 xyz
核心看几何预测误差，并且必须超过 copy-last。
```

原 ReGenNet 论文是 action-reaction synthesis 口径：

```text
给 actor motion 和条件，生成 reactor motion / 双人动作；
核心看识别模型特征上的生成质量、动作一致性、多样性和多模态。
```

因此不能把当前 `xyz_mse/mpjpe/copy-last` 结果直接说成已经按原论文完整评估。

## 当前最终 NTU xyz 指标

代码入口：

```text
eval/eval_ntu_label_xyz.py
utils/ntu_smplx_2p_xyz.py
```

当前指标：

```text
xyz_mse
xyz_mae
mpjpe
first_step_error
velocity_error
root_translation_error
relative_root_distance_error
inter_person_distance_consistency
```

并同时报告：

```text
model_metrics
copy_last_metrics
beats_copy_last: xyz_mse / xyz_mae / mpjpe
```

这套指标适合证明：

```text
模型在 xyz 骨架空间预测得比最后一帧延续更接近真实 future。
```

## 原 ReGenNet 论文 / 原仓库评估指标

原仓库入口：

```text
eval/eval_cmdm.py
eval/a2m/stgcn_eval.py
eval/a2m/stgcn/evaluate.py
eval/easy_table.py
```

NTU120-AS / Chi3D-AS action-conditioned 主表指标：

```text
FID
Accuracy
Diversity
Multimodality
```

并按 train/test conditioning 分开输出：

```text
fid_gen_train
accuracy_gen_train
multimodality_gen_train
diversity_gen_train
fid_gen_test
accuracy_gen_test
multimodality_gen_test
diversity_gen_test
```

原评估 full mode 还使用：

```text
num_samples = 1000
num_seeds = 20
mean + interval
```

InterHuman text-conditioned 表还涉及：

```text
R-Precision
FID
MM Dist
Diversity
MModality
```

## 当前相对原论文缺失的指标

如果目标是“和原 ReGenNet 论文口径一致”，当前最终 NTU xyz 评估缺少：

```text
FID
action recognition Accuracy
Diversity
Multimodality
train/test conditioning 分表
20 seeds / confidence interval 统计
```

如果要写 InterHuman text-conditioned 类似表，还缺：

```text
R-Precision
MM Dist
```

如果要复现论文里的速度/采样效率分析，还缺：

```text
Latency
```

## 当前相对早期 forecasting evaluator 的差异

早期 InterHuman forecasting evaluator 使用 active-vector original-scale 指标：

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

当前 NTU xyz evaluator 改为 skeleton xyz 指标后，不再有：

```text
rotation_mse
translation_mse
short_mse / mid_mse / long_mse
relative_orientation_error
```

但新增或强化了：

```text
xyz_mae
mpjpe
first_step_error
velocity_error
root_translation_error
copy-last 对照 gate
```

## 建议

论文若定位为双人 motion forecasting，当前几何误差 + copy-last 是合理主线，但最好补充：

```text
1. horizon breakdown: short / mid / long 或 final-frame error
2. 原论文风格辅助指标: action accuracy / FID / diversity / multimodality
3. 统计稳定性: 多 seed 或 bootstrap confidence interval
```

不能只靠当前 8 个 xyz 指标声称已经完整复现原 ReGenNet 论文评估标准。
