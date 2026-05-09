# InterHuman-AS P0 环境与数据冻结

## 输出

已生成 manifest：

```text
dataset/interhuman/reproduction_manifest.json
```

## 目标口径

Table 4 主指标：

```text
FID
Acc.
Div.
Multimod.
```

非主路径指标：

```text
R Precision
MM Dist
```

## 环境

```text
GPU: NVIDIA GeForce RTX 3080, 10240 MiB
Driver: 550.107.02
Python: 3.7.13
Torch: 1.7.1
CUDA: 11.0
NumPy: 1.21.5
h5py: 3.7.0
```

## 数据统计

```text
motions/*.pkl: 7776
actor-reactor labels: 7776
```

split 可用性：

```text
train: listed 6022, valid 6021, skipped 1
val:   listed 580,  valid 580,  skipped 0
test:  listed 1177, valid 1175, skipped 2
```

跳过样本：

```text
train/3945: missing_actor_reactor_label
test/3433:  missing_actor_reactor_label
test/4106:  missing_actor_reactor_label
```

说明：

这些样本即使本地存在 `.pkl`，也不能进入 actor-reactor 训练协议，因为缺少可用的 actor-reactor 标注。

## 缺口

本地未发现 InterHuman text/caption 数据目录。

本地未发现 InterHuman recognition checkpoint：

```text
recognition_training/*interhuman*
```

因此 P5 必须补齐 InterHuman 类别标签与 ST-GCN recognition evaluator，否则无法严格复现 Table 4 的 `Acc.` 和 `Multimod.`。
