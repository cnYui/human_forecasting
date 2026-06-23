# NTU120 2P 长度分布与 Forecasting 协议建议

## 目标

用户希望查看当前 NTU120 2P 数据集的序列长度分布，以便反向调整 label-conditioned forecasting 的 `obs_len/pred_len/window_len`。

数据：

```text
dataset/ntu120/smplx/conditioned/xsub.train.h5
dataset/ntu120/smplx/conditioned/xsub.test.h5
```

shape：

```text
[T,56,6]
```

标签解析：

```text
key 中 A001-A026 -> label 0-25
handshaking = label 8，对应 key A009
```

## 原始长度分布

### Train

```text
n = 4273
min = 18
max = 203
mean = 59.17
p25 = 44
p50 = 57
p75 = 71
p90 = 87
p95 = 95.4
p99 = 113
```

### Test

```text
n = 3845
min = 17
max = 174
mean = 53.24
p25 = 38
p50 = 49
p75 = 65
p90 = 81
p95 = 92
p99 = 113
```

### Train + Test

```text
n = 8118
min = 17
max = 203
mean = 56.36
p25 = 41
p50 = 54
p75 = 68
p90 = 84
p95 = 94
p99 = 113
```

结论：当前 NTU120 2P SMPL-X 条件 H5 的主长度集中在 50-70 帧，不适合直接使用 `window_len=150`。

## Window 覆盖率

### 全量覆盖

```text
window_len=45:  all 5456 / 8118 = 67.21%
window_len=60:  all 3209 / 8118 = 39.53%
window_len=75:  all 1425 / 8118 = 17.55%
window_len=90:  all 590  / 8118 = 7.27%
window_len=105: all 196  / 8118 = 2.41%
window_len=120: all 46   / 8118 = 0.57%
window_len=150: all 10   / 8118 = 0.12%
```

### Train/Test 均覆盖所有 26 类的最大窗口

```text
window_len=60
```

候选窗口按 split：

```text
window=45
train total=3173, covered_labels=26, min_class_count=14, handshaking=223
test  total=2283, covered_labels=26, min_class_count=21, handshaking=92

window=50
train total=2794, covered_labels=26, min_class_count=8, handshaking=223
test  total=1883, covered_labels=26, min_class_count=10, handshaking=91

window=60
train total=1956, covered_labels=26, min_class_count=2, handshaking=170
test  total=1253, covered_labels=26, min_class_count=1, handshaking=68

window=75
train total=864, covered_labels=25, missing=[17], handshaking=23
test  total=561, covered_labels=25, missing=[14], handshaking=20

window=90
train total=357, covered_labels=18, handshaking=0
test  total=233, covered_labels=15, handshaking=5
```

## Handshaking 长度分布

### Train label 8

```text
n = 223
min = 50
max = 89
mean = 64.92
p25 = 60
p50 = 65
p75 = 69
p90 = 74.8
```

覆盖：

```text
window=45: 223
window=60: 170
window=75: 23
window=90: 0
```

### Test label 8

```text
n = 92
min = 49
max = 103
mean = 67.45
p25 = 59
p50 = 65
p75 = 73.25
p90 = 80
```

覆盖：

```text
window=45: 92
window=60: 68
window=75: 20
window=90: 5
```

结论：如果关心 handshaking，`window_len=60` 是不重采样情况下比较稳的上限；`window_len=75` handshaking 仍有少量样本，但已经不能覆盖全部动作类。

## 推荐协议

### 推荐 A：不重采样，保留所有 26 类

如果坚持使用原始连续帧、不做时间重采样，并且要训练所有动作类：

```text
window_len = 60
obs_len = 30
pred_len = 30
```

优点：

- train/test 均覆盖全部 26 类。
- handshaking train=170，test=68。
- 保留 `obs_len=30`。

缺点：

- 预测未来只有 30 帧，不是用户最初想要的 120 帧。
- 某些类别在 `T>=60` 下样本极少，正式训练可能需要类别均衡采样。

### 推荐 B：不重采样，样本更稳

如果更重视每类样本数，而不是保留 30 帧观测：

```text
window_len = 50
obs_len = 20
pred_len = 30
```

或：

```text
window_len = 45
obs_len = 15
pred_len = 30
```

优点：

- 所有类别覆盖更稳定。
- `window=45/50` 下 handshaking 几乎全量保留。

缺点：

- 观测帧少于 30。

### 推荐 C：保留 30 -> 120 目标

如果必须保留：

```text
obs_len = 30
pred_len = 120
window_len = 150
```

则当前 NTU120 必须使用时间重采样到 150 帧，不能按原始长度 `T>=150` 过滤。

取舍：

- 可以保留所有动作类和 handshaking。
- 未来 120 帧表示 normalized sequence progress，不是原始帧率下的连续 120 帧物理未来。

## 建议决策

当前最合理的第一版 smoke：

```text
NTU120 2P
不重采样
window_len = 60
obs_len = 30
pred_len = 30
训练所有 26 类
重点检查 handshaking label-conditioned generation
```

如果 smoke 跑通，再决定是否做正式版：

1. `30 -> 30` 原始帧协议，强调真实连续帧。
2. `30 -> 120` 重采样协议，强调和最初任务形态对齐。
