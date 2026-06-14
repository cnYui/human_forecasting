# 三篇 Multi-person Forecasting 论文深度调研计划

## 调研对象

```text
docs/download/2021-multi-person-3d-motion-prediction-multi-range-transformers.pdf
docs/download/2023-joint-relation-transformer-multi-person-motion-prediction.pdf
docs/download/2023-stochastic-multi-person-3d-motion-forecasting.pdf
```

## 用户问题

深度调研三篇论文，并用和前面 SoMoFormer / T2P 一样的大白话，说明它们之间方向差异。

## 调研方法

1. 从本地 PDF 抽取摘要、问题定义、方法结构、数据集、指标、baseline、实验表和局限。
2. 联网核对论文元信息和官方页面，避免标题、年份、代码状态记错。
3. 分别回答每篇论文在做什么、不是在做什么、为什么和 SoMoFormer/T2P 不同。
4. 横向比较五篇相关工作：
   - Multi-Range Transformers
   - SoMoFormer
   - Joint-Relation Transformer
   - Stochastic Multi-Person 3D Motion Forecasting
   - Trajectory2Pose / T2P
5. 输出大白话版本，重点解释：
   - 它们具体问题有什么不同
   - 是否同一任务
   - 是否同一输入组织
   - 是否同一 baseline / 指标
   - 对当前 InterHuman SMPL forecasting 的启发

## 约束

- 默认中文。
- 不改训练代码。
- 新增上下文文档，不覆盖历史文档。
- 结论必须保留边界：这些工作都已覆盖 multi-person / interaction-aware forecasting 的强相关方向，当前项目不能声称方向首创。
