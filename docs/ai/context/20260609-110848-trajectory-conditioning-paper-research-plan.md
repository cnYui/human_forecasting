# Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning 调研计划

## 目标

深度调研本地 PDF：

`docs/download/2024-multi-agent-long-term-3d-human-pose-forecasting-trajectory-conditioning.pdf`

输出面向当前 ReGenNet / InterHuman forecasting 方向的技术判断：论文解决的问题、核心方法、训练和评估协议、实验可信度、贡献边界、与当前 150/30/120 deterministic two-person forecasting protocol 的关系，以及可以借鉴或不应照搬的部分。

## 方法

1. 先从 PDF 抽取题名、作者、摘要、方法、实验、表格和结论。
2. 联网核对论文元信息、代码或项目页状态，避免只依赖文件名和 PDF 内部信息。
3. 按 paper review 模式分析：问题定义、方法拆解、实验设置、指标、消融、局限。
4. 用 devil's advocate 视角检查：论文是否真正证明了 interaction-aware trajectory conditioning 的收益，是否存在数据协议或评估解释风险。
5. 对照本项目当前 forecasting 结论，给出可迁移设计和不建议迁移的点。

## 预期产物

- 新建一份调研结果文档，包含摘要、方法结构、实验表格解读、局限、对当前项目的启发和下一步建议。
- 最终回复给出压缩版结论，引用结果文档路径。

## 约束

- 默认中文输出。
- 不改动训练代码。
- 不覆盖、重命名或删除既有 `docs/ai/context/` 历史文件。
- 如果需要提出模型改造建议，必须保留当前论文主张边界：当前项目不能声称 multi-person forecasting 或 interaction-aware modeling 首创。
