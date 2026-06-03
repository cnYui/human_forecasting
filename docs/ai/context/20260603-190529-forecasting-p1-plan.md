# Forecasting P1 计划文档

## 文档定位

本文使用 `using-superpowers` 工作流生成，是以下正式设计的 P1 落地计划：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
```

补充参考：

```text
docs/ai/context/20260603-184214-forecasting-p1-p6-complete-design.md
docs/ai/context/20260603-161334-forecasting-p1-p2-design.md
```

本文只规划 P1，不进入实现结果记录。P1 完成后必须新建结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p1-dataset-result.md
```

## P1 目标

建立 InterHuman forecasting 的唯一数据协议，让后续 repeat、independent、concat 和 relation-aware 模型读取完全一致的 `obs / target`。

P1 必须交付：

```text
InterHumanForecastDataset
forecasting_collate
active vector extract / restore
train-only normalizer
dataset_smoke 验收入口
```

P1 通过后才允许进入 P2 metrics + repeat baseline。

## 必须解决的问题

从第一性原理看，P1 只解决四件事：

```text
1. 从 H5 原始 [T,25,12] 中抽取真实监督维度，避免 zero channel 污染 loss / metrics。
2. 把固定 150 帧窗口切成 30 帧观测和 120 帧未来目标。
3. 只用 train split 统计 normalizer，训练使用 normalized space，论文指标保留 original scale。
4. 用可重复的 smoke 命令证明 shape、样本数、finite、normalizer 和 roundtrip 都正确。
```

## 非目标

P1 不做：

```text
P2 metrics
repeat baseline
independent / concat / relation-aware model
训练循环
可视化
diffusion forecasting
ReGenNet Table 4 evaluator
NTU120-AS / Chi3D-AS 接入
padding
multi-window eval
window recenter
```

P1 不修改旧 ReGenNet 主路径：

```text
train/train_mdm.py
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
data_loaders/tensors.py::ccollate
```

原因是旧路径语义是 `actor condition -> reactor target`，新任务语义是 `observed prefix -> future suffix`。

## 输入协议

数据源：

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5
dataset/interhuman/smpl/conditioned/meta.json
```

H5 单条样本：

```text
motion: [T,25,12]
actor_rot6d:         motion[:, :24, 0:6]
actor_translation:   motion[:, 24, 0:3]
reactor_rot6d:       motion[:, :24, 6:12]
reactor_translation: motion[:, 24, 6:9]
```

固定 forecasting 协议：

```text
window_len = 150
obs_len = 30
pred_len = 120
T >= 150: 使用
T < 150: 过滤
train: random crop
val/test: center crop
```

预期可用样本数：

```text
train: 2910
val:   226
test:  508
```

## Active Vector 计划

新增文件：

```text
utils/forecasting_motion.py
```

常量：

```text
NUM_PERSONS = 2
NUM_BODY_JOINTS = 24
ROT6D_DIM = 6
ROT_DIM = 144
TRANSL_DIM = 3
PERSON_DIM = 147
```

接口：

```text
extract_active_motion(motion_h5) -> active
restore_active_motion(active) -> motion_h5_like
```

形状：

```text
motion_h5: [T,25,12]
active:    [T,2,147]
```

映射：

```text
active[:, 0, :144] = motion[:, :24, 0:6].reshape(T, 144)
active[:, 0, 144:147] = motion[:, 24, 0:3]
active[:, 1, :144] = motion[:, :24, 6:12].reshape(T, 144)
active[:, 1, 144:147] = motion[:, 24, 6:9]
```

restore 时只还原有效 channel，其他 channel 保持 0。P1 必须验证：

```text
active -> h5-like -> active 最大误差在浮点容忍范围内
```

实现约束：

```text
优先支持 torch.Tensor。
保持 dtype / device。
输入 shape 不合法时直接 ValueError。
不做坐标重中心化。
```

## Dataset 计划

新增文件：

```text
data_loaders/forecasting/__init__.py
data_loaders/forecasting/interhuman.py
```

核心类：

```text
InterHumanForecastDataset(
    data_path,
    split,
    window_len=150,
    obs_len=30,
    pred_len=120,
    max_samples=-1,
    seed=0,
)
```

split 只读取自己的 H5 文件，不复用旧 `InterHuman` loader 的 train/eval 合并索引。

输出 item：

```text
{
  "obs": Tensor[30,2,147],
  "target": Tensor[120,2,147],
  "sample_id": str,
  "start": int,
  "length": int,
}
```

采样规则：

```text
train:
  start 随机取 [0, T - 150]
  同一样本多次读取 start 应可变化

val/test:
  start = floor((T - 150) / 2)
  同一样本多次读取 start 必须固定
```

H5 句柄规则：

```text
惰性打开 h5py.File。
__getstate__ 中清空句柄，避免 DataLoader worker 复用不可 pickle 的 H5 handle。
```

`max_samples` 规则：

```text
先按 T>=150 过滤，再截断 max_samples。
```

## Collate 计划

新增文件：

```text
data_loaders/forecasting/tensors.py
```

接口：

```text
forecasting_collate(batch) -> obs, target, meta
```

输出：

```text
obs:    Tensor[B,30,2,147]
target: Tensor[B,120,2,147]
meta:   list[dict]
```

不做 padding。因为 P1 固定窗口，batch 内 shape 必须一致。

## Normalizer 计划

新增在：

```text
utils/forecasting_motion.py
```

接口：

```text
ForecastingNormalizer(mean, std, eps=1e-6)
compute_forecasting_normalizer(data_path, save_dir, window_len=150, obs_len=30, pred_len=120, eps=1e-6)
load_forecasting_normalizer(path)
```

统计来源：

```text
split: train only
samples: T>=150 train sequences
frames: 每条可用序列的全部帧
space: active vector original scale
```

统计方式：

```text
使用 streaming sum / sum_sq，避免把所有 train frame 堆到内存。
sum 和 sum_sq 使用 float64 CPU tensor。
std < eps 的维度设为 1.0。
```

保存：

```text
normalizer.pt
normalizer.json
```

张量形状：

```text
mean: [1,1,2,147]
std:  [1,1,2,147]
```

`normalizer.json` 至少包含：

```text
dataset
data_path
window_len
obs_len
pred_len
num_train_sequences_used
num_train_frames_used
person_dim
eps
created_at
```

P1 必须验证：

```text
normalizer 可保存和加载
normalize -> denormalize 最大误差在浮点容忍范围内
```

## Smoke 入口计划

正式设计的 P1 验收命令依赖：

```text
python -m eval.eval_forecasting --mode dataset_smoke
```

因此 P1 允许新增：

```text
eval/eval_forecasting.py
```

但 P1 只实现 `dataset_smoke`，不实现 metrics、repeat 或 checkpoint evaluation。P2 再扩展同一入口。

默认输出目录：

```text
save/forecasting/interhuman/p1_dataset_smoke
```

验收命令：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode dataset_smoke \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --batch_size 4 \
  --num_workers 0 \
  --save_dir save/forecasting/interhuman/p1_dataset_smoke
```

`dataset_smoke` 检查项：

```text
train dataset length = 2910
val dataset length = 226
test dataset length = 508
obs shape = [30,2,147]
target shape = [120,2,147]
batch obs shape = [B,30,2,147]
batch target shape = [B,120,2,147]
所有 obs/target finite
train 同一样本多次读取 start 可变化
val/test 同一样本多次读取 start 固定
normalizer.pt 和 normalizer.json 可写出
normalizer 可加载
normalize -> denormalize 误差在容忍范围内
active -> h5-like -> active 误差在容忍范围内
```

容忍范围：

```text
float32 roundtrip max_abs_error <= 1e-5
relative_orientation / metrics 不在 P1 检查
```

## 实现顺序

1. 新建 `data_loaders/forecasting/`，补 `__init__.py`。
2. 实现 `utils/forecasting_motion.py` 的 active vector extract / restore。
3. 实现 `InterHumanForecastDataset` 的 H5 path 解析、长度过滤、crop 和 item 输出。
4. 实现 `forecasting_collate`。
5. 实现 `ForecastingNormalizer`、streaming normalizer 统计、保存和加载。
6. 新增 `eval/eval_forecasting.py --mode dataset_smoke`。
7. 运行 P1 smoke 命令。
8. 新建 P1 result 文档，记录实现文件、命令、输出、normalizer 摘要和是否允许进入 P2。

## 失败处理

```text
样本数不等于 2910/226/508:
  先查 T>=150 过滤和 split 文件，不改协议。

shape 不等于 [30,2,147] / [120,2,147]:
  先查 active vector 映射和 crop 边界。

finite 检查失败:
  先查 H5 原始数值，再查 active vector 和 normalizer std 防护。

train start 不变化:
  先确认样本 T>150，再查 train crop 随机逻辑。

val/test start 不固定:
  修 dataset center crop，不进入 P2。

normalizer roundtrip 失败:
  先查 mean/std shape 和 broadcast 维度。
```

## P1 退出条件

只有同时满足以下条件，才允许进入 P2：

```text
P1 smoke 命令成功退出。
normalizer.pt / normalizer.json 已生成。
P1 result 文档已新建。
AGENTS.md 已记录 P1 完成状态和下一步。
没有修改旧 diffusion / CMDM / ccollate 主路径。
```
