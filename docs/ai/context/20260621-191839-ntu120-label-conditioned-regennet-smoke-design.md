# NTU120 Label-conditioned ReGenNet Forecasting Smoke 设计

## 用户确认的目标边界

目标任务：

```text
dataset = NTU120 2P
condition = 前 30 帧双人动作 + 动作标签
target = 后 120 帧双人动作
window_len = 150
训练 = 从头训练所有动作类
生成比较 = 同一前 30 帧输入，不同标签应生成明显不同的未来 120 帧
表示 = SMPL/SMPL-X rot6d + translation
评估 = 转 xyz 后算 MPJPE / joint MSE，并先用已有动作标签做分类一致性评估
网络 = 必须复用 ReGenNet / CMDM 条件扩散结构
资源 = 单卡 RTX 3080
第一阶段 = smoke 跑通
```

## 必须先指出的数据冲突

用户提出“只保留 `T>=150` 的 NTU120 样本”。当前本地 NTU120 条件 H5 统计如下：

```text
dataset/ntu120/smplx/conditioned/xsub.train.h5
total = 4273
T>=150 = 5
T>=150 labels = {6: 1, 7: 4}
handshaking label 8 = 0

dataset/ntu120/smplx/conditioned/xsub.test.h5
total = 3845
T>=150 = 5
T>=150 labels = {7: 3, 19: 2}
handshaking label 8 = 0
```

因此，严格按 `T>=150` 过滤后：

- 不能训练所有 26 个动作类。
- 不能训练或评估 `handshaking`。
- 训练集只有 5 条，test 只有 5 条，没有统计意义。
- label-conditioned generation 比较不成立。

这不是模型问题，是当前本地 NTU120 H5 的帧长分布与 `30->120` 协议冲突。

## Smoke 的第一道 gate

第一阶段 smoke 不应直接训练，而应先做数据 gate：

```text
检查 xsub.train.h5 / xsub.test.h5
统计每个 label 的 total count
统计每个 label 的 T>=150 count
确认目标标签 handshaking 是否存在
确认每个动作类是否至少有 train/test 样本
```

通过条件：

```text
train 每个动作类 T>=150 count > 0
test 每个动作类 T>=150 count > 0
handshaking train/test 均 > 0
```

当前本地数据不满足。Smoke 应明确报出该失败原因，而不是静默退化为少数类别训练。

## 可执行的 smoke 修正路线

如果用户同意为了保留 NTU120 和全动作标签条件，把严格 `T>=150` 改为“从原序列时间重采样到 150 帧”，则 smoke 可以继续：

```text
原始序列 T 任意
不做 padding 伪造未来
使用线性/均匀索引重采样到 150 帧
前 30 帧作为 obs
后 120 帧作为 future
保留所有 26 类动作标签
```

该路线的取舍：

- 优点：保留 NTU120 全动作类和 handshaking；能做 label-conditioned generation。
- 缺点：预测时间跨度变成 normalized sequence progress，不再是原始帧率下的连续 120 帧物理未来。

如果用户不同意重采样，则可选路线只有：

- 改用 Chi3D 做 150 帧协议，但样本小。
- 改短 NTU120 协议，例如 `obs_len=30,pred_len=30` 或 `obs_len=20,pred_len=40`。
- 寻找/重建更长的 NTU120 motion 源数据。

## Smoke 实现范围

在修正为“重采样到 150”后，第一阶段只做最小闭环：

1. 新增 NTU120 label-conditioned forecasting dataset。
2. 输出：

```text
future: [B,56,6,120]
obs:    [B,56,6,30]
label:  [B,1]
mask:   [B,1,1,120]
```

3. 新增或改造 CMDM forecast wrapper：

```text
x_t = noisy future
cond = obs_30 + action label + timestep
output = denoised future
```

4. 2-step train smoke：

```text
batch_size = 1 或 2
num_steps = 2
hidden/层数使用小配置
保存 checkpoint
```

5. checkpoint eval smoke：

```text
读取 checkpoint
对少量 test 样本采样 future
输出 shape / finite / basic metrics
```

6. label swap generation smoke：

```text
固定同一 obs_30
分别输入 handshaking / hugging / pushing 等标签
生成 120 帧
检查输出不是完全相同
保存对比样本
```

## Smoke 不做的事

- 不做完整 3-seed。
- 不要求生成质量达到可发表。
- 不做完整动作分类器训练。
- 不在第一版改原始 ReGenNet 主训练入口语义，避免破坏历史 baseline。

## 需要新建的模块建议

```text
data_loaders/forecasting/ntu_label.py
model/forecasting_cmdm.py
train/train_label_forecasting_diffusion.py
eval/eval_label_forecasting_diffusion.py
sample/sample_label_forecasting_diffusion.py
```

命名可后续再收敛，但第一版应独立于现有 `train/train_mdm.py`，因为原始入口是 actor->reactor，不是 obs+label->future。

## Smoke 成功标准

最低成功标准：

```text
dataset gate 通过
forward shape 正确
loss finite
2-step training 完成
checkpoint 可加载
sampling 输出 [B,56,6,120]
xyz metrics 可计算
label swap 输出存在可观测差异
```

如果坚持严格 `T>=150`，则当前 smoke 的正确结果是“数据 gate 失败”，不能进入训练。
