# ForecastingCMDMDecoder 阶段 A 数据集与数据 Gate 计划

## 参考上下文

本阶段只参考并承接以下文档的阶段 A / Commit 1 边界：

```text
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-112549-forecasting-cmdm-code-placement-policy.md
docs/ai/context/20260622-105946-forecasting-cmdm-final-target-architecture-v3.md
docs/ai/context/20260622-103022-ntu120-length-distribution-for-forecasting.md
```

当前阶段不是模型实现阶段。`ForecastingCMDMDecoder`、CFG wrapper、diffusion 训练循环、采样和评估全部留到后续阶段。

## 阶段 A 目标

建立 NTU120 2P label-conditioned forecasting 的可靠数据入口和数据检查 gate，保证后续模型阶段拿到的 batch 已满足最终协议：

```text
dataset = NTU120 2P
split = xsub train / test
data = dataset/ntu120/smplx/conditioned/xsub.train.h5
data = dataset/ntu120/smplx/conditioned/xsub.test.h5
window_len = 60
obs_len = 20
pred_len = 40
condition = obs20 + action label
target = future40
representation = [T,56,6] in H5, batch 输出 [B,56,6,T]
num_actions = 26
handshaking = A009 = label 8
```

第一性原则：

- 必须先证明数据协议成立，再实现模型；否则模型 smoke 的 shape 正确没有意义。
- 阶段 A 只解决 H5 读取、动作标签解析、窗口过滤、切分、batch collate 和可重复检查。
- 不用 workaround 把 `obs20` 塞进旧 CMDM 的 `y["cmotion"]`；新协议必须显式输出 `obs_motion` 和 `future`。

## 文件边界

计划新增：

```text
data_loaders/forecasting/ntu_label.py
scripts/check_ntu_label_forecasting_data.py
```

计划轻量修改：

```text
data_loaders/forecasting/__init__.py
```

不修改：

```text
model/cmdm.py
model/forecasting_cmdm.py
train/train_mdm.py
train/train_label_forecasting_diffusion.py
eval/*
sample/*
```

当前仓库没有 `scripts/` 目录。阶段 A 实现时可创建 `scripts/`，该目录只放 CLI 检查入口，不把检查脚本混入 dataset 模块。

## 数据集设计

新增 `NTULabelForecastDataset`，职责只包括：

- 打开 NTU120 2P conditioned H5。
- 遍历 H5 keys，读取每条序列长度。
- 从 key 中解析 `A001` 到 `A026`，映射为 label `0` 到 `25`。
- 过滤 `T >= window_len`，默认 `window_len=60`。
- train split 使用 random crop。
- test split 使用 center crop。
- 返回固定长度窗口，并切分为 `obs20` 和 `future40`。
- 使用 lazy H5 handle，兼容多 worker DataLoader。

每个样本建议返回：

```text
obs_motion: [56,6,20]
future:     [56,6,40]
action:     [1]
mask:       [1,1,40]
length:     int
start:      int
sample_id:  str
```

H5 内部 shape 是 `[T,56,6]`，dataset 读取后需要转为模型协议 `[56,6,T]`。这个转换应放在 dataset 内完成，避免训练脚本再猜数据布局。

## Collate 设计

新增 `ntu_label_forecasting_collate`，输出一个 batch dict：

```text
obs_motion: [B,56,6,20]
future:     [B,56,6,40]
action:     [B,1]
mask:       [B,1,1,40]
lengths:    [B]
meta:       List[dict]
```

`meta` 至少包含：

```text
sample_id
start
length
action
action_name 或 action_code
```

保留 dict batch 的原因：

- 阶段 B 的 model forward 需要 `y["obs_motion"]`、`y["action"]`、`y["mask"]`。
- 不复用现有 `forecasting_collate` 的 `(obs, target, meta)`，避免把 label-conditioned diffusion 协议隐式塞进旧 InterHuman forecasting 协议。
- 后续训练入口可以直接转换为 `x_start = batch["future"]` 和 `y = {...}`。

## 数据检查脚本

新增 CLI：

```bash
python -m scripts.check_ntu_label_forecasting_data \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --window_len 60 --obs_len 20 --pred_len 40
```

脚本必须检查：

- train/test H5 文件存在。
- `obs_len + pred_len == window_len`。
- train/test 过滤后均非空。
- train/test 过滤后均覆盖 26 类。
- handshaking label 8 在 train/test 中均非零。
- batch shape 正确：

```text
obs_motion = [B,56,6,20]
future     = [B,56,6,40]
action     = [B,1]
mask       = [B,1,1,40]
```

- `obs_motion` 和 `future` 无 NaN/Inf。
- `action` 范围在 `[0,25]`。
- `mask` 与 `pred_len=40` 对齐。

脚本建议输出：

```text
raw samples
kept samples after T>=60
label coverage
per-class counts
handshaking counts
one train batch shape
one test batch shape
finite check result
```

任何关键检查失败必须非零退出，不能只打印 warning。

## 通过标准

阶段 A 完成必须满足：

```text
train/test 均覆盖 26 类
handshaking train/test 非零
batch obs_motion = [B,56,6,20]
batch future = [B,56,6,40]
action = [B,1]
mask = [B,1,1,40]
无 NaN/Inf
检查脚本退出码为 0
```

已知长度分布基线：

```text
window_len=60
train total=1956, covered_labels=26, min_class_count=2, handshaking=170
test  total=1253, covered_labels=26, min_class_count=1, handshaking=68
```

如果实现后的统计明显偏离上述基线，优先检查 label 解析、H5 key 遍历和 `T>=60` 过滤逻辑，不进入模型阶段。

## 实现步骤

1. 参考 `data_loaders/forecasting/interhuman.py` 的 lazy H5 handle、split crop 和 worker 兼容写法。
2. 新增 `data_loaders/forecasting/ntu_label.py`，实现 dataset、label parser、collate 和统计辅助函数。
3. 更新 `data_loaders/forecasting/__init__.py`，导出新 dataset/collate。
4. 新建 `scripts/check_ntu_label_forecasting_data.py`，只做数据 gate。
5. 运行阶段 A 检查命令。
6. 将实际检查结果新建到 `docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-a-data-gate-result.md`，不要覆盖本计划文档。

## 非目标

阶段 A 不做：

- 不实现 `ForecastingCMDMDecoder`。
- 不实现 `ForecastingClassifierFreeSampleModel`。
- 不接 diffusion 训练。
- 不做 label swap sampling。
- 不训练动作一致性分类器。
- 不做类别均衡采样。
- 不做时间重采样。
- 不修改原始 `train/train_mdm.py`。
- 不修改旧 InterHuman forecasting dataset 的返回协议。

## 风险与处理

- `window_len=60` 下部分动作类样本很少：阶段 A 只记录统计，不在 dataset 内做类别重采样；正式训练阶段再决定 sampler。
- H5 key 格式如果有异常：dataset 应明确报错或跳过并统计，不能静默映射到错误 label。
- train random crop 可导致同一 index 多次取样不同：这是训练期合理行为；test 必须 center crop，保证评估稳定。
- 多 worker 下随机性容易混乱：优先复用 `get_worker_info()` 分支，阶段 A 不强求逐样本 deterministic，只保证 test deterministic。
