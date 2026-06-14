# Forecasting 终极目标状态审计

## 审计范围

本次只审计 `docs/ai/context/` 中已经落盘的计划、设计、结果记录，以及关键结果目录是否存在。

未重新训练模型，未重新跑完整评估。

## 依据文件

核心设计：

```text
docs/ai/context/20260603-161803-forecasting-p1-p6-roadmap.md
docs/ai/context/20260603-190003-forecasting-final-official-design.md
```

阶段结果：

```text
docs/ai/context/20260603-191712-forecasting-p1-dataset-result.md
docs/ai/context/20260603-194749-forecasting-p2-metrics-repeat-result.md
docs/ai/context/20260603-201148-forecasting-p3-baselines-result.md
docs/ai/context/20260603-202924-forecasting-p4-relation-result.md
docs/ai/context/20260603-232353-forecasting-p5-main-table-result.md
docs/ai/context/20260604-085116-forecasting-p5-ablation-result.md
docs/ai/context/20260604-091820-forecasting-p6-qualitative-result.md
```

已确认存在的结果输出：

```text
results/forecasting/interhuman/p5_main_150_30_120/summary.json
results/forecasting/interhuman/p5_main_150_30_120/summary.md
results/forecasting/interhuman/p5_ablation_150_30_120/summary.json
results/forecasting/interhuman/p5_ablation_150_30_120/summary.md
results/forecasting/interhuman/p6_qualitative_150_30_120/summary.md
results/forecasting/interhuman/p6_qualitative_150_30_120/selection.json
```

## 阶段完成判断

P1 已完成：

```text
dataset / active vector / normalizer smoke 通过。
train/val/test 可用样本为 2910/226/508。
```

P2 已完成：

```text
original-scale metrics、metrics_sanity、repeat baseline 已闭环。
pred == target 时固定指标均为 0。
```

P3 已完成：

```text
independent 和 concat no-relation baseline 已训练、评估、落盘。
两者 future_mse 均优于 repeat。
```

P4 已完成：

```text
relation-aware predictor 已训练、评估、落盘。
达到 P4 最低门槛：future_mse、long_mse、relative_root_distance_error 优于 concat。
未达到强门槛：不优于 independent。
```

P5 已完成当前主协议所需部分：

```text
P5.2 3-seed main table gate 通过。
relation long_mse mean 优于 concat，same-seed 3/3 胜出。
relative_root_distance_error mean 优于 concat，same-seed 3/3 胜出。

P5.3 ablation gate 通过。
relation full model 优于 parameter-matched concat。
relation encoder 对 long_mse 和 relative root distance 有小幅贡献。
all relation features 在 mean 上优于 single-feature variants。
```

P6 已完成：

```text
sample-level metrics、npy、selection、curves 已生成。
success / close / failure / boundary 各 2 个样本。
P6 验收通过，且 P2 metrics_sanity 回归仍为 0。
```

## 终极目标判断

如果“终极目标”指当前正式设计中定义的第一阶段工程与论文证据链：

```text
Interaction-aware joint forecasting of two-person human motion from partial observations
InterHuman 150/30/120 deterministic forecasting
P1-P6 dataset -> metrics -> baselines -> relation-aware -> 3-seed table -> ablation -> qualitative
```

则当前已经达成。

但必须保留论文结论边界：

```text
relation-aware 稳定优于 concat no-relation 和 parameter-matched concat。
relation-aware 不优于 independent 的 future_mse / long_mse。
repeat 在部分 relation-style metrics 上仍最低。
relation encoder 不改善所有 relation metrics。
all-features 不在每个 seed 上都强于 velocity-only。
```

因此论文主张只能写成：

```text
显式关系建模相对 concat no-relation baseline 带来稳定的 long-horizon error 和 relative root distance 改善。
```

不能写成：

```text
relation-aware 全面最优。
relation-aware 全面优于 independent。
显式关系建模解决了所有交互一致性指标。
```

## 尚未完成但不阻塞当前主协议的事项

以下不是 P1-P6 当前主协议的阻塞项：

```text
P5.4 observation-ratio 补充表。
P6 rendered frames or videos。
P6 root_trajectory_xy.png。
parameter-matched concat qualitative 输出。
P7 NTU120-AS 扩展。
P8 Chi3D-AS 扩展。
完整论文正文、投稿格式、参考文献和最终审稿材料。
```

## 下一步建议

如果目标是继续把结果变成论文：

```text
1. 整理 P5 main table 和 ablation table 为论文表格。
2. 从 P6 success / failure / boundary 中选图，形成 qualitative figure。
3. 写 Method / Experiments / Discussion，并明确上述边界。
4. 如需增强论文厚度，再单独设计 P5.4 或 P7/P8，不要改写当前 P1-P6 结论。
```
