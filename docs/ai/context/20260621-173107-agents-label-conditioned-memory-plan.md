# AGENTS 标签条件预测记忆沉淀计划

## 目标

把以下两份上下文的关键结论沉淀到项目入口 `AGENTS.md`：

```text
docs/ai/context/20260621-172003-label-conditioned-regennet-forecasting-design-note.md
docs/ai/context/20260621-172442-three-datasets-action-label-check.md
```

## 必须保留的信息

- 用户当前新增目标：`前 30 帧双人动作 + 动作标签（如握手） -> 后 120 帧双人动作`。
- 导师要求必须使用 ReGenNet 网络；合理方向是 ReGenNet-style label-conditioned forecasting diffusion，而不是直接跑原始 `train/train_mdm.py`。
- 可复用 ReGenNet 的条件扩散、timestep embedding、action embedding / classifier-free guidance、双人 motion conditioning 和关系约束思想。
- InterHuman 当前本地数据没有“握手”等动作语义标签，只有 actor/reactor 顺序标签；不能把该标签当动作类别。
- NTU120 2P 和 Chi3D 当前本地数据有握手类：NTU120 `8: handshaking`，Chi3D `1: Handshake`。

## 取舍

当前 `AGENTS.md` 已被瘦身，明确详细实验记录应放入 `docs/ai/context/`。因此本次只追加少量长期项目记忆和文档指针，不展开实现方案细节。

## 操作

- 只编辑 `AGENTS.md`。
- 不修改历史上下文文件。
- 不触碰当前工作区中其它未提交改动。
