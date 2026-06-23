# NTU120 Label-conditioned Forecasting 长度协议更新计划

## 用户确认的新协议

用户决定第一版使用：

```text
dataset = NTU120 2P
window_len = 60
obs_len = 20
pred_len = 40
训练 = 所有 26 个动作类
模型 = ReGenNet / CMDM-derived 条件扩散
条件 = obs_20 + action label
目标 = future_40
```

## 背景

此前设计文档以 `window_len=150, obs_len=30, pred_len=120` 为目标，但 NTU120 当前本地 H5 的长度分布不支持严格 `T>=150`：

```text
train T>=150 = 5
test T>=150 = 5
handshaking T>=150 = 0
```

随后统计显示：

```text
window_len=60
train total=1956, covered_labels=26, handshaking=170
test total=1253, covered_labels=26, handshaking=68
```

因此 `20->40` 是当前不重采样且仍覆盖全部动作类的可执行折中方案。

## 取舍

优点：

- 不需要时间重采样，保留原始连续帧语义。
- train/test 均覆盖 NTU120 2P 的 26 个动作类。
- handshaking 有可用样本。
- 预测长度 40 帧，比 `30->30` 更接近“较长未来”目标。

缺点：

- 覆盖全量样本约 39.53%，不是大多数样本。
- `obs_len=20` 少于用户最初希望的 30。
- 某些类别在 `T>=60` 下样本很少，正式训练需要类别均衡采样或记录类别不平衡风险。

## 文档更新策略

按项目规则不覆写历史 `docs/ai/context/` 文档。本次新增 v2 设计文档，旧文档保留为历史决策记录。

需要新增：

```text
docs/ai/context/*-ntu120-label-conditioned-regennet-smoke-design-v2-length60.md
docs/ai/context/*-ntu120-label-conditioned-regennet-final-design-v2-length60.md
```

同时更新 `AGENTS.md` 的当前入口记忆，明确后续实现默认协议改为：

```text
window_len=60, obs_len=20, pred_len=40
```
