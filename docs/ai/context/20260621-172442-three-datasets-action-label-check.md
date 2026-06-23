# 三个数据集动作语义标签检查

## 问题

用户确认：当前三个数据集是否都没有“握手”这样的动作标签。

## 检查结论

不是三个都没有。

## InterHuman

当前本地 InterHuman 冻结数据：

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5
```

H5 key 是纯 sample id，例如：

```text
10
1000
1001
```

可见标注：

```text
dataset/interhuman/annotations_interhuman/interhuman_label.json
```

该标注是 actor/reactor 顺序，不是动作语义类。当前 loader 中 InterHuman 被设为单类：

```text
_action_classes = {0: "interaction"}
num_actions = 1
```

因此当前本地 InterHuman 不能直接提供“握手/拥抱/推拉”这类标签监督。

## NTU120 2P

当前本地 NTU120 条件数据：

```text
dataset/ntu120/smplx/conditioned/xsub.train.h5
dataset/ntu120/smplx/conditioned/xsub.test.h5
```

loader 中 `ntu_action_enumerator` 有 26 个双人动作类，其中：

```text
8: handshaking
```

本地 H5 检查：

```text
xsub.train.h5: handshaking label 8 count = 223
xsub.test.h5:  handshaking label 8 count = 92
```

示例 key：

```text
S001C001P001R001A009
```

注意：预处理后 key 中 `A009` 经 loader `-1` 后对应 label 8，也就是 `handshaking`。

## Chi3D

当前本地 Chi3D 条件数据：

```text
dataset/chi3d/smplx/conditioned/chi3d_smplx_train.h5
dataset/chi3d/smplx/conditioned/chi3d_smplx_test.h5
```

loader 中 `chi3d_action_enumerator` 有 8 类：

```text
0: Grab
1: Handshake
2: Hit
3: HoldingHands
4: Hug
5: Kick
6: Posing
7: Push
```

本地 H5 检查：

```text
train: Handshake label 1 count = 26
test:  Handshake label 1 count = 7
```

示例 key：

```text
s02_Handshake10_1
```

## 后续设计影响

- 如果坚持 InterHuman 150/30/120 主协议，当前没有动作语义标签，需要额外补标签或做伪标签。
- 如果优先做“前 30 帧 + 握手 -> 后 120 帧”的 label-conditioned proof-of-concept，NTU120 2P 和 Chi3D 都能直接提供握手类。
- Chi3D 类别少、语义清楚，但样本量小；NTU120 2P 样本更多，但动作捕捉/SMPL-X 表示和当前 InterHuman SMPL forecasting 主线不同。
