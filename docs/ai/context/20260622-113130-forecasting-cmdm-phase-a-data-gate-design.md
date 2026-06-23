# ForecastingCMDMDecoder 阶段 A 数据 Gate 设计

## 参考文档

本设计承接以下文档，不覆盖历史文档：

```text
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
docs/ai/context/20260622-112741-forecasting-cmdm-phase-a-data-gate-plan.md
docs/ai/context/20260622-105946-forecasting-cmdm-final-target-architecture-v3.md
docs/ai/context/20260622-104643-ntu120-label-conditioned-regennet-smoke-design-v2-length60.md
docs/ai/context/20260622-104723-ntu120-label-conditioned-regennet-final-design-v2-length60.md
docs/ai/context/20260622-103022-ntu120-length-distribution-for-forecasting.md
```

`20260622-103022` 中早期建议过 `obs_len=30,pred_len=30`，后续 v2 设计和项目入口已经明确改为：

```text
window_len = 60
obs_len = 20
pred_len = 40
```

阶段 A 以 `20 -> 40` 为唯一实现协议。

## 阶段定位

阶段 A 只建立数据协议和数据 gate。它的输出不是训练结果，而是一个后续阶段可以信任的数据入口：

```text
NTU120 2P conditioned H5
-> 过滤 T >= 60
-> 解析 A001-A026 动作标签
-> 裁剪连续 60 帧
-> 切分 obs20 / future40
-> 输出 label-conditioned forecasting batch
-> 运行可重复的数据检查脚本
```

先做阶段 A 的原因：

- 模型 forward shape 正确不能证明任务协议正确。
- label-conditioned forecasting 的关键风险在数据过滤、标签映射和 obs/future 布局。
- 后续 `ForecastingCMDMDecoder` 依赖显式 `obs_motion`，不能把 obs20 伪装成旧 CMDM 的 `cmotion`。

## 必须满足的协议

```text
dataset = NTU120 2P
train_h5 = dataset/ntu120/smplx/conditioned/xsub.train.h5
test_h5 = dataset/ntu120/smplx/conditioned/xsub.test.h5
H5 item shape = [T,56,6]
window_len = 60
obs_len = 20
pred_len = 40
num_actions = 26
action_code = A001-A026
label = action_number - 1
handshaking = A009 = label 8
```

Dataset 内部读取 H5 后必须转换布局：

```text
H5 window: [60,56,6]
obs_motion: [56,6,20]
future: [56,6,40]
```

这个转换放在 dataset 内完成，训练脚本不再猜测 H5 原始布局。

## 文件边界

阶段 A 新增：

```text
data_loaders/forecasting/ntu_label.py
scripts/check_ntu_label_forecasting_data.py
```

阶段 A 轻量修改：

```text
data_loaders/forecasting/__init__.py
```

阶段 A 不修改：

```text
data_loaders/get_data.py
model/*
train/*
eval/*
sample/*
```

`data_loaders/get_data.py` 暂不接入新 dataset。后续训练入口直接导入：

```python
from data_loaders.forecasting.ntu_label import (
    NTULabelForecastDataset,
    ntu_label_forecasting_collate,
)
```

原因是通用入口当前绑定 MDM/CMDM 旧协议，过早接入会让 `obs_motion/future/action/mask` 的新 batch 语义变得隐式。

## Dataset 设计

新增类：

```text
NTULabelForecastDataset
```

建议初始化参数：

```text
h5_path: str | Path
split: "train" | "test"
window_len: int = 60
obs_len: int = 20
pred_len: int = 40
max_samples: int = -1
seed: int = 0
strict: bool = True
```

基本校验：

- `split` 只接受 `train/test`。
- `obs_len + pred_len == window_len`。
- H5 文件必须存在。
- H5 item 必须是三维，并且后两维是 `[56,6]`。
- 动作标签必须能从 sample key 中解析到 `A001-A026`。

entries 建议结构：

```python
{
    "sample_id": str,
    "length": int,
    "action": int,
    "action_code": str,
}
```

entries 构建规则：

- 使用 `sorted(h5.keys())`，保证统计与 `max_samples` 可复现。
- 解析 label 后再检查长度和 shape。
- 只保留 `length >= window_len`。
- 如果 `strict=True`，发现非法 key、非法 shape 或非法 label 直接报错。
- 如果后续需要兼容脏数据，再显式增加非 strict 模式并在统计中列出跳过数量。

## 标签解析

标签解析必须集中在一个函数中，避免脚本和 dataset 各自写一套规则：

```text
parse_ntu_action_label(sample_id) -> {
  action: int,
  action_code: str,
  action_number: int,
}
```

规则：

```text
从 sample_id 中匹配 A(\d{3})
A001 -> 0
A009 -> 8
A026 -> 25
```

失败策略：

- 没有 `Axxx`：报错。
- `A000`、`A027` 或更大：报错。
- 同一个 key 出现多个 `Axxx`：报错，避免静默取错标签。

阶段 A 不强依赖动作英文名。`meta` 中必须保留 `action_code`，`action_name` 可以先等于 `action_code`。

## H5 Handle 与 Crop

复用现有 `InterHumanForecastDataset` 的 lazy H5 思路：

```text
__getstate__ 时清空 _h5_handle
每个 worker 第一次 __getitem__ 时打开自己的 h5py.File
close / __del__ 负责关闭句柄
```

裁剪规则：

```text
train: random crop 连续 window_len 帧
test: center crop 连续 window_len 帧
```

train random crop：

- 主进程使用 dataset 自己的 `random.Random(seed)`。
- 多 worker 时使用 worker 内已初始化的 Python random 状态。
- 不要求同一 index 每次返回同一 crop，因为训练阶段需要数据增强。

test center crop：

```text
start = (length - window_len) // 2
```

必须 deterministic，供采样和评估复现。

## 单样本返回

`__getitem__` 返回 dict：

```text
obs_motion: torch.float32 [56,6,20]
future:     torch.float32 [56,6,40]
action:     torch.long    [1]
mask:       torch.bool    [1,1,40]
length:     int
start:      int
sample_id:  str
action_code: str
action_name: str
```

`mask` 固定为全 True，因为阶段 A 不 padding、不重采样，只保留完整 future40。

## Collate 设计

新增函数：

```text
ntu_label_forecasting_collate(batch)
```

输出 batch dict：

```text
obs_motion: torch.float32 [B,56,6,20]
future:     torch.float32 [B,56,6,40]
action:     torch.long    [B,1]
mask:       torch.bool    [B,1,1,40]
lengths:    torch.long    [B]
meta:       list[dict]
```

`meta` 至少包含：

```text
sample_id
start
length
action
action_code
action_name
```

不复用 `forecasting_collate` 的原因：

- 旧 collate 返回 `(obs, target, meta)`，缺少 action 和 mask。
- 阶段 B 的 model forward 需要 `y["obs_motion"]`、`y["action"]`、`y["mask"]`。
- dict batch 能让训练入口显式写出 `x_start = batch["future"]`。

## 统计辅助函数

`data_loaders/forecasting/ntu_label.py` 中建议提供轻量统计函数，供检查脚本复用：

```text
summarize_entries(entries, num_actions=26) -> dict
```

输出字段：

```text
raw_count
kept_count
skipped_too_short
label_counts
covered_labels
missing_labels
min_class_count
handshaking_count
length_min
length_max
length_mean
```

统计函数只依赖 entries 或扫描结果，不打开训练逻辑，不调用模型。

## 数据 Gate CLI 设计

新增入口：

```bash
python -m scripts.check_ntu_label_forecasting_data \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --window_len 60 --obs_len 20 --pred_len 40
```

建议参数：

```text
--train_path
--test_path
--window_len
--obs_len
--pred_len
--batch_size default 16
--num_workers default 0
--seed default 0
--max_samples default -1
--scan_all default true
```

`num_workers` 默认 0，降低 gate 自身复杂度；dataset 仍按 lazy H5 支持多 worker。

CLI 必须检查：

- train/test 文件存在。
- `obs_len + pred_len == window_len`。
- train/test 过滤后非空。
- train/test 均覆盖 26 类。
- train/test 的 handshaking label 8 非零。
- action 范围在 `[0,25]`。
- batch shape 完全匹配协议。
- `obs_motion` 和 `future` 全部 finite。
- `mask.shape[-1] == pred_len` 且全 True。

建议 CLI 默认扫描所有 kept samples 的 finite，而不是只检查首个 batch。阶段 A 的目标是 gate，不是快速 smoke。

## 失败策略

以下情况必须非零退出，不能只 warning：

- H5 文件不存在。
- H5 item shape 不是 `[T,56,6]`。
- key 无法解析到唯一 `A001-A026`。
- `T >= 60` 后 train 或 test 为空。
- train/test 任一 split 缺失动作类。
- handshaking 在 train 或 test 为 0。
- batch shape 与协议不一致。
- 任一被扫描样本出现 NaN/Inf。

以下情况可以 warning，但不能阻塞阶段 A：

- 某个类别样本数很少。
- train/test 类别分布不均衡。
- 本地默认 Python 环境缺少 `h5py`，但这属于运行环境问题，真正 gate 需要在项目训练环境执行。

## 预期统计基线

来自长度分布文档，`T >= 60` 后应大致为：

```text
train total = 1956
train covered_labels = 26
train min_class_count = 2
train handshaking = 170

test total = 1253
test covered_labels = 26
test min_class_count = 1
test handshaking = 68
```

如果实现后的 kept count 或 handshaking count 明显偏离，优先检查：

- `Axxx` 标签解析是否偏移一位。
- 是否误用了 `A001-A120` 全 NTU 标签口径。
- H5 key 是否被未排序或 `max_samples` 截断。
- shape 读取是否误把 `[T,56,6]` 当成 `[56,6,T]`。

## 阶段 A 验收

阶段 A 完成时，必须新建结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-a-data-gate-result.md
```

结果文档至少记录：

```text
实际命令
运行环境中的 Python 路径
train/test raw_count 和 kept_count
per-class counts
handshaking counts
首个 batch shape
全量 finite scan 结果
退出码
```

只有检查脚本退出码为 0，才进入阶段 B 模型实现。

## 非目标

阶段 A 不做：

- 不实现 `ForecastingCMDMDecoder`。
- 不实现 CFG wrapper。
- 不接 diffusion 训练循环。
- 不保存 checkpoint。
- 不做 sampling。
- 不训练动作一致性分类器。
- 不做类别均衡采样。
- 不做时间重采样。
- 不做 padding。
- 不修改旧 `train/train_mdm.py`。
- 不修改旧 InterHuman forecasting 协议。

## 后续移交给阶段 B

阶段 A 通过后，阶段 B 可以依赖以下 batch contract：

```python
x_start = batch["future"]
y = {
    "obs_motion": batch["obs_motion"],
    "action": batch["action"],
    "mask": batch["mask"],
}
```

阶段 B 仍需自己做 model shape guard，但不再重复实现 H5 标签解析、crop、obs/future 切分和类别覆盖检查。
