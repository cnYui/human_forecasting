# ForecastingCMDMDecoder 阶段 A 数据 Gate 结果

## 结论

阶段 A 已完成并通过数据 gate，可以进入阶段 B 模型实现。

本阶段实际改动：

```text
新增 data_loaders/forecasting/ntu_label.py
新增 scripts/check_ntu_label_forecasting_data.py
更新 data_loaders/forecasting/__init__.py
```

未改动：

```text
data_loaders/get_data.py
model/*
train/*
eval/*
sample/*
```

## 运行环境

系统默认 `/usr/bin/python3` 缺少项目依赖，因此使用项目 micromamba 环境：

```text
python_executable=/home/rpartx3080/.local/micromamba/envs/regennet/bin/python
python_version=3.7.13
torch_version=1.7.1
h5py_version=3.7.0
numpy_version=1.21.5
```

## 验证命令

### 静态编译

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile \
  data_loaders/forecasting/ntu_label.py \
  scripts/check_ntu_label_forecasting_data.py
```

结果：

```text
exit_code = 0
```

### 标签解析 smoke

验证：

```text
A001 -> label 0
A009 -> label 8
A026 -> label 25
无 Axxx / A000 / A027 / 多个 Axxx 均抛 ValueError
```

结果：

```text
parser ok
exit_code = 0
```

### H5 scan 统计

命令使用 `scan_ntu_label_forecasting_entries` 和 `summarize_entries` 直接扫描：

```text
train 4273 1956 26 2 170
test 3845 1253 26 1 68
```

字段顺序：

```text
split raw_count kept_count covered_labels min_class_count handshaking_count
```

该结果与历史长度分布文档一致。

### 完整数据 gate

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m scripts.check_ntu_label_forecasting_data \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --window_len 60 --obs_len 20 --pred_len 40
```

结果：

```text
exit_code = 0
PASS
```

Train：

```text
raw_count=4273
kept_count=1956
skipped_too_short=2317
covered_labels=26
missing_labels=[]
min_class_count=2
handshaking_count=170
length_min=60
length_max=203
length_mean=76.50
label_counts=0:82 1:76 2:121 3:137 4:38 5:113 6:205 7:214 8:170 9:205 10:183 11:10 12:6 13:5 14:8 15:5 16:3 17:2 18:16 19:100 20:10 21:70 22:9 23:21 24:91 25:56
```

Test：

```text
raw_count=3845
kept_count=1253
skipped_too_short=2592
covered_labels=26
missing_labels=[]
min_class_count=1
handshaking_count=68
length_min=60
length_max=174
length_mean=76.97
label_counts=0:40 1:38 2:59 3:74 4:21 5:57 6:88 7:90 8:68 9:82 10:78 11:8 12:9 13:5 14:5 15:5 16:1 17:6 18:14 19:157 20:15 21:112 22:8 23:32 24:122 25:59
```

Batch shape：

```text
train obs_motion=(16, 56, 6, 20)
train future=(16, 56, 6, 40)
train action=(16, 1)
train mask=(16, 1, 1, 40)

test obs_motion=(16, 56, 6, 20)
test future=(16, 56, 6, 40)
test action=(16, 1)
test mask=(16, 1, 1, 40)
```

Finite scan：

```text
train checked_samples=1956 result=PASS
test checked_samples=1253 result=PASS
```

### 多 worker lazy H5 检查

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m scripts.check_ntu_label_forecasting_data \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 2 --num_workers 2 --no_scan_all
```

结果：

```text
exit_code = 0
PASS
train obs_motion=(2, 56, 6, 20)
train future=(2, 56, 6, 40)
test obs_motion=(2, 56, 6, 20)
test future=(2, 56, 6, 40)
```

## 最终 batch contract

阶段 B 可以直接依赖：

```python
x_start = batch["future"]
y = {
    "obs_motion": batch["obs_motion"],
    "action": batch["action"],
    "mask": batch["mask"],
}
```

对应 shape：

```text
x_start / future = [B,56,6,40]
obs_motion = [B,56,6,20]
action = [B,1]
mask = [B,1,1,40]
```

## 阶段 B 进入条件

以下条件均已满足：

```text
train/test 均覆盖 26 类
handshaking train/test 非零
batch obs_motion = [B,56,6,20]
batch future = [B,56,6,40]
action = [B,1]
mask = [B,1,1,40]
obs_motion/future 全量 finite
检查脚本退出码为 0
```

因此允许进入阶段 B：

```text
新增 model/forecasting_cmdm.py
实现 ForecastingCMDMDecoder
实现 ForecastingClassifierFreeSampleModel
只做 forward / backward / CFG shape smoke
```
