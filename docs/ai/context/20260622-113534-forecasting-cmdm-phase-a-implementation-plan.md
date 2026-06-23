# ForecastingCMDMDecoder 阶段 A 实施 Plan

## 目标

阶段 A 的目标是完成 NTU120 2P label-conditioned forecasting 的数据入口和数据 gate，实现后续阶段 B 可以直接依赖的 batch contract：

```python
x_start = batch["future"]
y = {
    "obs_motion": batch["obs_motion"],
    "action": batch["action"],
    "mask": batch["mask"],
}
```

固定协议：

```text
data = dataset/ntu120/smplx/conditioned/xsub.train.h5
eval_data = dataset/ntu120/smplx/conditioned/xsub.test.h5
window_len = 60
obs_len = 20
pred_len = 40
H5 item = [T,56,6]
batch obs_motion = [B,56,6,20]
batch future = [B,56,6,40]
action = [B,1], label 0-25
handshaking = A009 = label 8
```

本计划基于：

```text
docs/ai/context/20260622-113130-forecasting-cmdm-phase-a-data-gate-design.md
docs/ai/context/20260622-112741-forecasting-cmdm-phase-a-data-gate-plan.md
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
```

## 阶段边界

本阶段只做：

```text
data_loaders/forecasting/ntu_label.py
scripts/check_ntu_label_forecasting_data.py
data_loaders/forecasting/__init__.py
docs/ai/context/*phase-a-data-gate-result.md
```

本阶段不做：

```text
不实现 model/forecasting_cmdm.py
不实现 ForecastingCMDMDecoder
不接 diffusion 训练
不改 train/train_mdm.py
不接 sample / eval
不做类别均衡采样
不做时间重采样
不改 data_loaders/get_data.py
```

`data_loaders/get_data.py` 暂不接入新 dataset，避免旧 MDM/CMDM collate 语义污染新协议。

## 实施 Checklist

### 1. 建立 NTU label parser

文件：

```text
data_loaders/forecasting/ntu_label.py
```

实现：

```text
parse_ntu_action_label(sample_id)
```

验收：

```text
A001 -> label 0
A009 -> label 8
A026 -> label 25
无 Axxx、多个 Axxx、越界 Axxx 均抛 ValueError
```

设计要求：

- 用正则集中解析，脚本和 dataset 不重复写标签逻辑。
- 报错信息包含 `sample_id`，方便定位坏 key。

### 2. 实现 H5 scan 和 entries

实现：

```text
scan_ntu_label_forecasting_entries(h5_path, window_len, strict=True)
```

entries 字段：

```text
sample_id
length
action
action_code
action_name
```

验收：

```text
只保留 T >= 60
H5 item 后两维必须为 [56,6]
使用 sorted(h5.keys()) 保证可复现
raw_count / kept_count / skipped_too_short 可统计
```

失败策略：

- `strict=True` 时，非法 key 或非法 shape 直接失败。
- 不静默跳过无法解析的动作标签。

### 3. 实现统计函数

实现：

```text
summarize_entries(scan_result, num_actions=26)
```

至少输出：

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

验收基线：

```text
train kept_count ~= 1956
train covered_labels = 26
train handshaking_count = 170
test kept_count ~= 1253
test covered_labels = 26
test handshaking_count = 68
```

如果统计明显偏离，先排查标签解析和 H5 shape，不进入后续任务。

### 4. 实现 Dataset

实现：

```text
class NTULabelForecastDataset(Dataset)
```

初始化参数：

```text
h5_path
split
window_len=60
obs_len=20
pred_len=40
max_samples=-1
seed=0
strict=True
```

实现细节：

- lazy H5 handle，参考 `InterHumanForecastDataset`。
- `__getstate__` 清空 `_h5_handle`，兼容 DataLoader 多 worker。
- `train` 使用 random crop。
- `test` 使用 center crop。
- H5 `[60,56,6]` 转为 `[56,6,60]` 后切分。

单样本返回：

```text
obs_motion: float32 [56,6,20]
future: float32 [56,6,40]
action: long [1]
mask: bool [1,1,40]
length: int
start: int
sample_id: str
action_code: str
action_name: str
```

### 5. 实现 Collate

实现：

```text
ntu_label_forecasting_collate(batch)
```

输出：

```text
obs_motion: [B,56,6,20]
future: [B,56,6,40]
action: [B,1]
mask: [B,1,1,40]
lengths: [B]
meta: list[dict]
```

验收：

- 空 batch 抛 `ValueError`。
- dtype 保持稳定：motion 为 `float32`，action/lengths 为 `long`，mask 为 `bool`。
- `meta` 保留 `sample_id/start/length/action/action_code/action_name`。

### 6. 导出 forecasting 入口

文件：

```text
data_loaders/forecasting/__init__.py
```

新增导出：

```text
NTULabelForecastDataset
ntu_label_forecasting_collate
parse_ntu_action_label
summarize_entries
```

不删除旧导出：

```text
InterHumanForecastDataset
forecasting_collate
```

### 7. 新增数据 Gate CLI

文件：

```text
scripts/check_ntu_label_forecasting_data.py
```

命令：

```bash
python -m scripts.check_ntu_label_forecasting_data \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --window_len 60 --obs_len 20 --pred_len 40
```

检查项：

```text
文件存在
obs_len + pred_len == window_len
train/test kept_count > 0
train/test covered_labels = 26
train/test handshaking_count > 0
首个 batch shape 正确
全量 kept samples finite
action 范围为 [0,25]
mask shape 为 [B,1,1,40] 且全 True
```

输出内容：

```text
Python executable
train/test raw_count
train/test kept_count
per-class counts
missing labels
handshaking count
batch shapes
finite scan result
PASS / FAIL
```

失败必须非零退出。

### 8. 运行验证并写结果文档

验证命令：

```bash
python -m scripts.check_ntu_label_forecasting_data \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --window_len 60 --obs_len 20 --pred_len 40
```

结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-a-data-gate-result.md
```

结果文档必须记录：

```text
实际命令
Python executable
退出码
train/test raw_count
train/test kept_count
train/test per-class counts
handshaking counts
batch shape
finite scan 是否通过
是否允许进入阶段 B
```

## 推荐实现顺序

1. 新建 `data_loaders/forecasting/ntu_label.py`，先写 parser、scan、summary。
2. 用小段 Python 调 parser 和 scan，确认统计接近历史基线。
3. 补 Dataset 和 collate。
4. 更新 `data_loaders/forecasting/__init__.py`。
5. 新建 `scripts/check_ntu_label_forecasting_data.py`。
6. 跑数据 gate。
7. 新建 result 文档。

这个顺序先验证最容易出错的标签和长度统计，再进入 batch 形状检查。

## 验收标准

阶段 A 完成必须全部满足：

```text
train/test 均覆盖 26 类
handshaking train/test 非零
obs_motion = [B,56,6,20]
future = [B,56,6,40]
action = [B,1]
mask = [B,1,1,40]
obs_motion/future 全量 finite
检查脚本退出码为 0
结果文档已写入 docs/ai/context/
```

若任一项失败，不进入阶段 B。

## 风险与处理

### h5py 环境缺失

当前普通 `python3` 环境可能缺少 `h5py`。实现和 gate 应在项目训练环境中执行，并把实际 `sys.executable` 写入结果文档。

### 类别样本极少

`window_len=60` 下最小类别数很低。阶段 A 只记录分布，不在 dataset 内做重采样。类别均衡采样属于训练阶段。

### label 口径混淆

本任务只使用本地 conditioned H5 中的 `A001-A026`，映射到 `0-25`。不要使用 NTU120 全 120 类标签编号。

### shape 方向混淆

H5 是 `[T,56,6]`，模型协议是 `[56,6,T]`。转换只放在 dataset 内，后续训练入口不再转置。

## 完成后的下一阶段

阶段 A gate 通过后，进入阶段 B：

```text
新增 model/forecasting_cmdm.py
实现 ForecastingCMDMDecoder
实现 ForecastingClassifierFreeSampleModel
只做 forward / backward / CFG shape smoke
```

阶段 B 不再重新设计 dataset，只依赖阶段 A 的 batch contract。
