# InterHuman-AS Table 4 复现设计修正版

## 目标校准

复现目标是 `2403.11882v1.pdf` 中 Table 4：

```text
online, unconstrained setting
InterHuman-AS dataset
actor motion -> ReGenNet -> reactor motion
```

本地 PDF 中 Table 4 的指标列是：

```text
FID
Acc.
Div.
Multimod.
```

因此 Table 4 主路径是 actor-conditioned / action-conditioned reaction generation 的 ST-GCN feature 评估链路，不是 text-motion retrieval 链路。

## 非目标

以下指标不属于 Table 4 主路径：

```text
R Precision
MM Dist
text-motion matching evaluator
```

这些属于 text-conditioned motion generation 的评估口径，只能作为后续扩展，不能阻塞 Table 4 复现。

## 当前可行路径

当前仓库已有 InterHuman 在线 loader：

```text
data_loaders/a2m/interhuman.py
```

经 `ccollate()` 后输出：

```text
motion:  [B, 25, 6, 150]
cmotion: [B, 25, 6, 150]
```

这说明训练基础链路的 shape 可用。

## 关键缺口

要复现 Table 4，还缺：

```text
离线 H5 预处理
H5 InterHuman loader
InterHuman 专用生成入口
InterHuman ST-GCN recognition checkpoint
InterHuman evaluator 分支
InterHuman 类别标签来源
SMPL 与 SMPL-X 口径确认
```

如果最终使用 SMPL 而不是论文描述中的 SMPL-X，结果必须标注为：

```text
SMPL reproduction attempt, not exact Table 4 reproduction
```

## 数据格式

H5 entry 使用：

```text
key: sample id
value: [T, 25, 12]
```

其中：

```text
value[:, :24, 0:6]   = actor rot6d
value[:, 24, 0:3]    = actor translation
value[:, :24, 6:12]  = reactor rot6d
value[:, 24, 6:9]    = reactor translation
其他 translation padding 通道为 0
```

translation 规范：

```text
以 actor 第 0 帧 translation 为原点
actor 和 reactor 使用同一个原点
```

该规范与当前 `Dataset._load()` 的双人 translation 处理保持一致。

## 验收标准

最低可接受：

- 完整 train split 可训练 1000 step，不出现 NaN/OOM。
- test split 可 DDIM-5 生成。
- 生成结果和 actor 条件均数值有限。
- 可用文档化 evaluator 输出 `FID / Acc. / Div. / Multimod.`。

Table 4 级别：

- 使用完整 InterHuman-AS train/test split。
- 使用 actor-reactor 标注。
- 使用 150 帧、6D rotation。
- 使用与论文兼容的 ST-GCN feature evaluator。
- 报告 20 次随机种子的均值和 95% 置信区间。
