# P8 指标表格图片可视化计划

## 目标

根据用户给定的 P8 3-seed 指标，生成一张彩色 PNG 图片，用于直观展示 `independent_pair_xyz`、`somoformer_lite_xyz` 和 `official_somoformer_xyz` 的 test mean/std 对比。

## 设计

- 输出目录：`results/forecasting/interhuman/p8_official_somoformer_xyz_main/`
- 输出图片：`p8_metrics_table_visual.png`
- 上半部分：彩色表格，单元格显示 `mean +- std`。
- 下半部分：按指标分组的柱状图，柱高为 mean，误差线为 std。
- 所有指标均按“越低越好”解释，表格颜色采用 green -> yellow -> red，绿色代表当前指标最优。

## 边界

- 只做结果可视化，不改变模型、训练、评估代码。
- 使用已完成的 3-seed 结果，不重新计算指标。
- 图片用于展示 joint-space P8 结果，不能解释为 P5 active-vector 主表。
