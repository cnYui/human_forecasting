# T2P 源码与数据复现计划

## 任务

用户要求：

```text
深度调研 docs/download/2024-multi-agent-long-term-3d-human-pose-forecasting-trajectory-conditioning.pdf，
拷贝源码到 /home/rpartx3080/CodeSpace，
并下载数据集复现论文的代码。
```

## 已知上下文

- 论文为 `Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning`，CVPR 2024 Highlight。
- 既有调研文档：`docs/ai/context/20260609-111650-trajectory-conditioning-paper-deep-review.md`。
- 既有结论：T2P 的核心是 `global hip trajectory -> local pose conditioning`，不是 interaction-aware 首创。
- 既有风险：官方仓库可能是 post-paper parser，论文 joint-position 表述和当前 SMPL theta 数据口径可能不完全一致。

## 本次范围

1. 复核论文和既有调研，只补充与复现有关的信息。
2. 克隆官方源码到 `/home/rpartx3080/CodeSpace/T2P`。
3. 审计仓库 README、配置、训练/评估脚本、数据目录约定和 checkpoint 约定。
4. 尝试下载官方公开可获得的数据文件。
5. 如果数据需要注册、人工申请、Google Drive 权限、JRDB 许可或原始数据手工处理，则记录明确阻塞点和复现步骤，不使用未授权镜像。

## 验收

- `/home/rpartx3080/CodeSpace/T2P` 存在且为官方仓库 clone。
- 记录仓库当前 commit、README 数据说明、主要依赖和复现命令。
- 对数据集下载给出实际结果：
  - 成功下载：记录路径、文件列表、大小和基本文件类型检查。
  - 受限失败：记录失败 URL、原因、后续人工动作。
- 新建结果文档到 `docs/ai/context/`，不覆盖历史文档。

## 不做

- 不直接改 T2P 源码以适配 ReGenNet。
- 不启动长训练。
- 不把当前 ReGenNet deterministic InterHuman 协议和 T2P best-of-K / JRDB-GMP 协议混合解释。
- 不绕过数据集授权或下载来源不明的数据。
