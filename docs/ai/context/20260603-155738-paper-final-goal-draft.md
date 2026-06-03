# 论文最终目标草案

## 本轮输入

- 用户指定核心文档：`20260603-154759-joint-interaction-forecasting-boundary.md`。
- 导师笔记线索：MSE 指标；前 20% 帧作为观测；预测后 80% 双人动作；关注关节数和维度；动作标签或自然语言可作为条件。
- 当前工程资产：InterHuman / NTU120-AS / Chi3D H5 均已落盘；InterHuman H5 和 ReGenNet 训练链路已有 smoke 和 50K baseline 训练。

## 建议锁定的论文目标

最终目标应定义为：

```text
Interaction-aware joint forecasting of two-person human motion from partial observations.
```

中文表述：

```text
基于部分观测的交互感知双人动作联合未来预测。
```

核心问题：

```text
给定一段双人交互动作的前 20% 观测，联合预测两个人后 80% 的未来动作。
```

关键点是“联合预测”和“交互关系建模”，不是：

```text
person A 独立预测未来
person B 独立预测未来
最后拼接结果
```

## 必须补上的窗口协议

`前 20% / 后 80%` 不应直接作用在 InterHuman 原始全长序列上。当前 InterHuman 长度分布极宽，最长超过 9000 帧，直接预测全序列会让任务定义不可控。

第一阶段建议使用固定窗口：

```text
InterHuman main protocol:
window_len = 150
obs_len = 30
pred_len = 120
input = 前 30 帧双人动作
target = 后 120 帧双人动作
```

采样建议：

```text
训练：从长度 >= 150 的序列中随机裁剪连续 150 帧窗口。
验证/测试：使用确定性中心裁剪或固定多窗口策略。
太短序列：第一阶段过滤，不用 padding 伪造未来。
```

本地长度统计：

```text
InterHuman train: 6021 条，长度 >=150 的有 2910 条。
InterHuman val:   580 条，长度 >=150 的有 226 条。
InterHuman test:  1175 条，长度 >=150 的有 508 条。
```

NTU120-AS 序列短，不适合 150 帧主协议；如果用于辅助实验，应采用：

```text
window_len = 60
obs_len = 12
pred_len = 48
```

## 第一阶段主线

主数据集：

```text
InterHuman
```

主任务：

```text
unconditioned two-person joint forecasting
```

主模型路线：

```text
obs_two_person_motion
  -> person/motion encoder
  -> relation encoder
  -> joint future predictor
  -> future_two_person_motion
```

第一版 relation encoder 不要过大，先建模：

```text
root relative translation
root relative velocity
root orientation difference
inter-person joint distance summary
```

## 必须实验

为了证明论文主张成立，至少需要：

```text
Baseline 0: repeat last observed frame / zero velocity
Baseline 1: independent predictor，A/B 分别预测再拼接
Baseline 2: concat predictor，拼接双人历史但无显式 relation module
Ours: relation-aware joint predictor
```

如果 `Ours` 不能明显优于 `independent` 和 `concat`，尤其是在长期误差和交互关系指标上，论文主张不成立。

## 指标

主指标：

```text
future MSE
rotation MSE
translation MSE
long-horizon MSE
```

交互指标：

```text
relative root distance error
relative orientation error
inter-person distance consistency
```

如果能稳定恢复 joint xyz，再加入：

```text
MPJPE
inter-person joint distance error
```

后续做多模态预测时再加入：

```text
best-of-K error
diversity
coverage
```

## 非目标

第一阶段不要把以下内容作为论文主目标：

```text
ReGenNet Table 4 完整复现
自然语言条件预测
动作标签条件预测
actor-only -> reactor-only 反应生成
SMPL-X 转换
ST-GCN recognition evaluator
```

这些可以作为后续扩展或相关 baseline，但不能继续阻塞 forecasting 主线。

## 工程关系

当前 ReGenNet InterHuman 50K baseline 仍有价值，但定位应调整为：

```text
历史 backbone / 生成式 baseline / 工程资产
```

它不是最终论文任务，因为当前 `cmdm` 训练仍是把双人动作拆成 actor 条件 `cmotion` 与 reactor 目标 `motion`，不等价于：

```text
前 20% 双人动作 -> 后 80% 双人动作
```

因此需要新增 forecasting 数据组织、训练入口和评估脚本。

## 可投稿贡献表述

推荐贡献组织：

```text
1. 定义一个基于部分双人观测的交互动作联合未来预测协议。
2. 提出显式关系推理模块，从可观测运动中学习双人动态关系。
3. 证明关系建模优于独立预测和无关系拼接，尤其改善长期预测与交互一致性。
```

## 外部文献信号

近期相关工作支持本方向：

- ReGenNet 关注 human action-reaction synthesis，但其主任务不是双人联合未来预测：https://arxiv.org/abs/2403.11882
- Joint Relation Transformer 明确指出双人姿态预测需要建模 body-body interaction：https://arxiv.org/abs/2304.05758
- InterGen / multi-human motion generation 系列强调多人体生成中的交互关系和空间约束：https://arxiv.org/abs/2404.05218

## 当前待用户确认

需要用户确认的最终决策：

```text
论文第一阶段是否正式锁定为：
InterHuman 150-frame window, 30-frame observation, 120-frame joint two-person forecasting,
核心创新为 relation-aware joint predictor。
```

确认后再把该目标写入 `AGENTS.md` 的长期记忆，并进入 forecasting loader / baseline / evaluator 的设计与实现。
