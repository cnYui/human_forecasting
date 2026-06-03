# InterHuman 第一阶段之后 NTU120-AS 与 Chi3D-AS 使用计划

## 结论

第一阶段只用 InterHuman。NTU120-AS 和 Chi3D-AS 不在第一阶段合并训练，也不作为 InterHuman 的验证集或测试集。

后续推荐用途：

```text
NTU120-AS：第二阶段 action-conditioned forecasting / SMPL-X 对照 / 更规整大样本验证。
Chi3D-AS：第三阶段小规模高质量 SMPL-X 泛化验证 / qualitative 补充。
```

## 为什么不合并

三套数据的任务和表示不一致：

```text
InterHuman 当前本地表示: [T,25,12], SMPL, 无动作类别主标签。
NTU120-AS 表示: [T,56,6], SMPL-X, 26 类动作。
Chi3D-AS 表示: [T,56,6], SMPL-X, 8 类动作。
```

直接合并会引入额外问题：

```text
1. 需要统一 SMPL / SMPL-X 表示。
2. 需要统一动作标签体系。
3. 需要统一帧长协议，NTU 主口径 60 帧，InterHuman/Chi3D 更适合 150 帧。
4. 需要统一 normalizer，否则 translation / rotation 尺度和采集域会混在一起。
5. 需要重新定义 evaluator，不能再把跨数据集结果解释为单一数据集上的泛化能力。
```

因此合并训练不是当前论文主线的自然下一步，而是单独的 multi-dataset learning 研究问题。

## 推荐阶段安排

### P1-P6：InterHuman 主实验

目标：

```text
证明 relation-aware joint forecasting 在 InterHuman 上优于 repeat / independent / concat。
```

使用：

```text
fixed window_len=150
obs_len=30
pred_len=120
deterministic two-person forecasting
MSE + relation metrics
```

产出：

```text
主结果表
消融表
定性分析
```

只有 InterHuman 主张成立后，才进入其他数据集。

### P7：NTU120-AS 作为动作条件扩展

用途 1：action-conditioned forecasting。

设计：

```text
input: 前若干帧双人动作 + action label
target: 后续双人动作
dataset: NTU120-AS
labels: 26 类双人动作
representation: SMPL-X [T,56,6]
```

目的：

```text
验证 relation-aware predictor 是否能利用动作类别先验。
```

用途 2：规整大样本 SMPL-X 对照。

设计：

```text
window_len=60 或 75
obs_ratio=20%
pred_ratio=80%
```

注意：

```text
NTU 平均帧长约 50-60，不能硬套 InterHuman 150 帧协议。
应单独定义 NTU forecasting protocol。
```

### P8：Chi3D-AS 作为高质量小样本泛化验证

用途：

```text
验证模型在高质量 MoCap / SMPL-X 双人交互上的表现。
```

推荐方式：

```text
1. 不把 Chi3D 当主数据集。
2. 用 InterHuman 学到的设计思想，在 Chi3D 上重新训练小模型。
3. 或把 Chi3D 作为 cross-dataset qualitative / low-data adaptation。
```

注意：

```text
Chi3D 样本只有约 293 train / 74 test，本地 H5 总数 367。
结果波动会大，不适合单独支撑主论文结论。
```

### P9：统一多数据集训练，可选

只有在 P7/P8 都稳定后才考虑。

前置条件：

```text
1. 统一表示到 SMPL-X 或统一 active vector schema。
2. 分别保留 dataset-specific normalizer 或使用 domain embedding。
3. 明确 multi-dataset split，禁止样本泄漏。
4. evaluator 改成 dataset-aware。
5. 论文问题改为 multi-domain two-person forecasting。
```

这不是第一篇论文的必要项。

## 推荐论文写法

第一篇论文：

```text
主数据集：InterHuman
扩展实验：可暂不做 NTU/Chi3D
```

如果时间允许，补充实验可以写：

```text
NTU120-AS：action-conditioned short-window forecasting
Chi3D-AS：small-scale SMPL-X qualitative/generalization
```

但不能把 NTU/Chi3D 的缺失作为阻塞 InterHuman P1-P6 的理由。

## 当前下一步

继续执行：

```text
P1 InterHuman forecasting dataset + normalizer
```

不要提前实现：

```text
NTU forecasting loader
Chi3D forecasting loader
multi-dataset normalizer
unified SMPL/SMPL-X conversion
```
