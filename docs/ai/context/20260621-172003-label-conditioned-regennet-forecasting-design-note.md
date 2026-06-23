# 标签条件 ReGenNet Forecasting 设计判断

## 用户新增目标

希望训练一个条件模型：

```text
condition = 前 30 帧双人动作 + 动作标签，例如“握手”
target = 后 120 帧双人动作
```

并要求使用 ReGenNet 网络。

## 当前数据限制

本地 `dataset/interhuman` 当前可见标注文件只有：

```text
dataset/interhuman/annotations_interhuman/interhuman_label.json
```

该文件不是动作语义类别，而是 actor/reactor 顺序标注：

```text
0: person1 是 actor，person2 是 reactor
1: person2 是 actor，person1 是 reactor
```

当前 `dataset/interhuman/motions/*.pkl` 也只包含：

```text
frames
mocap_framerate
person1
person2
```

没有可直接用于“握手/拥抱/推拉”等动作类别的字段。当前 `data_loaders/a2m/interhuman.py` 在 H5 和 pkl 路径下也都把 InterHuman 的 action 固定为单类：

```text
_action_classes = {0: "interaction"}
num_actions = 1
```

因此现有冻结 H5 不能直接训练“前 30 帧 + 握手 -> 后 120 帧”这种语义标签条件模型。

## ReGenNet 可以学习/复用的内容

如果导师要求必须使用 ReGenNet，合理复用点不是简单拿现有命令跑，而是把 ReGenNet 改成 forecasting 条件扩散模型：

1. 复用 ReGenNet 的条件扩散范式：给未来 120 帧加噪，网络学习在 timestep、观测前缀和标签条件下去噪。
2. 复用 `CMDM` 的 Transformer/MLP denoiser、timestep embedding、action embedding / classifier-free guidance 机制。
3. 复用双人建模经验：actor/reactor 顺序、双人 motion tensor 表示、SMPL rot6d + translation 表示。
4. 复用关系约束思想：在 loss 或辅助目标中加入双人相对距离、相对朝向、root trajectory 等交互一致性项。

不能直接复用的是原始 `train/train_mdm.py` 训练协议，因为它当前是“完整 actor motion 条件 -> 完整 reactor motion”，不是“双人前缀 + label -> 双人未来”。

## 推荐实现路线

最符合“使用 ReGenNet 网络”的路线：

```text
新增 label-conditioned forecasting diffusion
```

数据：

```text
obs:      [B,30,2,147] 或转换回 [B,25,12,30]
label:    [B]
future:   [B,120,2,147] 或 [B,25,12,120]
```

模型：

```text
x_t = noisy future 120 frames
cond = encode(obs_30) + action_embedding(label) + timestep_embedding(t)
output = predicted clean future / noise
```

训练：

```text
MSE diffusion denoising loss
+ xyz reconstruction loss
+ velocity loss
+ interaction relation loss
```

评估：

```text
test split MPJPE / joint_mse / root translation error / interaction consistency
label-conditioned retrieval 或动作分类一致性（需要额外 classifier 或规则）
```

前置条件：

- 需要真正的动作语义标签，不能把 actor/reactor 标签当成“握手”标签。
- 如果只用当前 InterHuman 本地数据，模型最多学到“interaction”单类条件，标签条件没有实际区分度。
