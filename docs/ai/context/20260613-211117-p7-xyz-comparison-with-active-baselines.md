# P7.1 SoMoFormer XYZ 与旧 active baselines 的 xyz 口径对比

## 问题

用户询问 P7.1 SoMoFormer XYZ 的训练结果怎么样，以及是否和之前 P3/P4 的两人预测 baselines，尤其 independent baseline，对比过。

## 对比方式

为了让 P7.1 SoMoFormer XYZ 和旧 active-vector baselines 放到同一 joint-space 指标下，对以下模型做 test split xyz 评估：

- repeat baseline：active repeat 后转 SMPL xyz。
- P3 independent：`save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt`，预测 active 后转 SMPL xyz。
- P3 concat：`save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt`，预测 active 后转 SMPL xyz。
- P4 relation：`save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt`，预测 active 后转 SMPL xyz。
- P7 SoMoFormer XYZ：`save/forecasting/interhuman/p7_somoformer_xyz_h256_l4_dct30_s0_5000/model000005000.pt`，直接预测 xyz。

输出：

```text
results/forecasting/interhuman/p7_xyz_compare_active_baselines/summary.json
results/forecasting/interhuman/p7_xyz_compare_active_baselines/summary.csv
```

## Test split 对比

样本数：

```text
508
```

| run | joint_mse | mpjpe | long_joint_mse | root_translation_error | relative_root_distance_error | consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repeat | 0.1035212231 | 0.3581020008 | 0.1933701291 | 0.3199613036 | 0.2552214026 | 0.0060419603 |
| independent active -> xyz | 0.1394910785 | 0.5289348627 | 0.2022760692 | 0.4967177667 | 0.4012468483 | 0.0134364697 |
| concat active -> xyz | 0.1855908594 | 0.6272615293 | 0.2348735279 | 0.5977439355 | 0.4946546531 | 0.0155920249 |
| relation active -> xyz | 0.1788634248 | 0.6243270955 | 0.2282090737 | 0.5919212694 | 0.4089161324 | 0.0164127117 |
| SoMoFormer XYZ | 0.0596641728 | 0.2897213449 | 0.1165986784 | 0.2483919385 | 0.1982787535 | 0.0055808047 |

## 直接观察

SoMoFormer XYZ 在当前 xyz 口径下是最好的：

- `joint_mse` 明显低于 repeat 和旧 active baselines。
- `mpjpe` 明显低于 repeat 和旧 active baselines。
- `long_joint_mse` 明显低于 repeat 和旧 active baselines。
- `relative_root_distance_error` 也低于 repeat 和旧 active baselines。
- `inter_person_distance_consistency_xyz` 略优于 repeat，也明显优于旧 active baselines。

相对 repeat 的改善：

```text
joint_mse: 0.1035212231 -> 0.0596641728，约 42.37% 降低
mpjpe: 0.3581020008 -> 0.2897213449，约 19.10% 降低
long_joint_mse: 0.1933701291 -> 0.1165986784，约 39.70% 降低
relative_root_distance_error: 0.2552214026 -> 0.1982787535，约 22.31% 降低
```

## 对 independent baseline 的解释

P3 independent 在 P5 active-vector 主表里表现强，但它是按 active-vector MSE 训练的：

```text
pred_active [B,120,2,147]
```

本次对比为了同口径，把它的 active 输出再经 SMPL forward 转成：

```text
pred_xyz [B,120,2,24,3]
```

结果发现 independent active -> xyz 的 joint-space 指标甚至差于 repeat：

```text
repeat joint_mse: 0.1035212231
independent active -> xyz joint_mse: 0.1394910785
```

这说明旧 independent 的 active-vector 优势不等价于 joint-space 优势。可能原因：

- active-vector loss 直接优化 rot6d/translation 数值误差，不直接优化 SMPL joint positions。
- rot6d 小误差经过 SMPL forward 后可能放大为 joint-space 误差。
- 当前 P7.1 SoMoFormer XYZ 直接优化 xyz loss，因此在 xyz 指标上更有优势。

## 公平性边界

这个对比不能被写成“P7 SoMoFormer 全面优于 independent 模型”。

原因：

```text
SoMoFormer XYZ 是用 xyz loss 训练的。
P3 independent / concat / P4 relation 是用 active-vector loss 训练的。
```

所以当前结论只能写成：

```text
在 joint-space 评估口径下，直接训练 xyz 的 SoMoFormer-style baseline 明显优于 repeat，以及旧 active-vector checkpoints 转 xyz 后的结果。
```

如果要严谨比较模型结构，需要补：

```text
xyz-independent baseline
xyz-concat baseline
可选 xyz-relation baseline
```

这些模型也必须用同样的 `pred_xyz` 目标和 `xyz loss` 训练。

## 结论

当前 P7.1 结果是有信号的：SoMoFormer-style joint/person token Transformer 在 InterHuman joint-space forecasting 上明显优于 repeat，也比旧 active checkpoints 转 xyz 的结果更好。

但它还不是 P5 active-vector 主表的替代结果。下一步建议先补 xyz-independent baseline，确认 SoMoFormer XYZ 是否仍然优于“同样用 xyz loss 训练的单人独立预测”。
