# 联合交互预测边界

## 核心判断

双人未来动作预测不应定义为：

```text
person A 单独预测未来
person B 单独预测未来
最后把两个结果拼在一起
```

更合理的定义是：

```text
两个人组成一个相互影响的动态系统，模型需要联合预测两个人的未来。
```

## 研究问题表述

当前主线应表述为：

```text
Interaction-aware joint forecasting of two-person human motion.
```

中文：

```text
交互感知的双人动作联合未来预测。
```

重点不是预测两个独立个体，而是建模：

```text
A 如何影响 B
B 如何影响 A
两个人的相对位置、速度、朝向和身体关系如何共同决定未来动作
```

## 对模型设计的约束

后续模型不能只做两个独立分支：

```text
A_obs -> A_future
B_obs -> B_future
```

至少需要一个跨人交互模块：

```text
A_obs, B_obs -> relation_module -> interaction_tokens
A_obs, B_obs, interaction_tokens -> A_future, B_future
```

可接受的最小结构：

```text
obs_two_person_motion
  -> person encoder
  -> relation encoder / cross-attention / interaction graph
  -> joint future decoder
  -> future_two_person_motion
```

## Baseline 设计

为了证明“联合交互建模”有意义，实验必须包含：

```text
Independent baseline:
分别预测 A 和 B，再拼接。

Concat baseline:
直接拼接 A/B 历史动作，但没有显式 relation module。

Relation-aware model:
加入关系推理模块，联合预测 A/B。
```

如果 relation-aware model 不能明显优于 independent / concat baseline，则论文主张不成立。

## 指标要求

不能只看单人姿态 MSE。还需要能反映交互关系的指标：

```text
future MSE
rotation MSE
translation MSE
relative root distance error
relative orientation error
inter-person distance consistency
long-horizon error
```

后续如果做多模态预测，再加入：

```text
best-of-K error
diversity
coverage
```

## 当前边界

第一阶段实现目标：

```text
InterHuman
输入：前 20% 双人动作
输出：后 80% 双人动作
模型：先做 no-relation baseline，再做 relation-aware joint predictor
```

不做：

```text
把 A/B 分别预测后简单拼接作为最终模型
自然语言条件
ReGenNet Table 4 完整复现
```
