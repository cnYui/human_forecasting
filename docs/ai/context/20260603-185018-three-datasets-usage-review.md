# 三个数据集使用口径梳理

## 问题

用户问：

```text
这三个数据集分别怎么用？
1. 在当前项目里怎么用？
2. 论文 2403.11882v1.pdf 里怎么用？
```

这里的三个数据集指 ReGenNet 主线三套 action-reaction 数据：

```text
NTU RGB+D 120 / NTU120-AS
Chi3D / Chi3D-AS
InterHuman / InterHuman-AS
```

## 当前项目里的用法

### 共同训练语义

当前 ReGenNet 主路径是 action-reaction synthesis，不是 forecasting：

```text
actor motion -> generated reactor motion
```

代码路径：

```text
train/train_mdm.py
  -> data_loaders/get_data.py
  -> data_loaders/a2m/feeder.py 或 data_loaders/a2m/interhuman.py
  -> data_loaders/tensors.py::ccollate
  -> model/cmdm.py
  -> diffusion/gaussian_diffusion.py
```

关键点：

```text
ccollate 把双人输入按 feature channel 拆成：
cmotion = actor 条件
motion  = reactor 训练目标
```

因此旧主路径学习的是“给定 actor 全段动作，生成 reactor 全段动作”。这和当前论文新方向的“前 30 帧双人动作 -> 后 120 帧双人动作”不是同一任务。

### NTU120-AS

本地数据：

```text
dataset/ntu120/smplx/conditioned/xsub.train.h5
dataset/ntu120/smplx/conditioned/xsub.test.h5

train: 4273
test: 3845
total: 8118
shape: [T,56,6]
```

项目用途：

```text
dataset 参数：--dataset ntu
loader：data_loaders/a2m/feeder.py::Feeder
body_model：smplx
num_frames：60
num_actions：26
训练目标：actor-conditioned reactor generation
评估：eval/eval_cmdm.py 支持 ntu 分支
recognition checkpoint：recognition_training/ntu_smplx/checkpoint_0100.pth.tar
```

可直接用来跑原论文式训练、采样、ST-GCN evaluator。也是当前三套数据里最接近原论文主实验主线的数据。

### Chi3D-AS

本地数据：

```text
dataset/chi3d/smplx/conditioned/chi3d_smplx_train.h5
dataset/chi3d/smplx/conditioned/chi3d_smplx_test.h5

train: 293
test: 74
total: 367
shape: [T,56,6]
```

项目用途：

```text
dataset 参数：--dataset chi3d
loader：data_loaders/a2m/feeder.py::Feeder
body_model：smplx
num_frames：150
num_actions：8
训练目标：actor-conditioned reactor generation
评估：eval/eval_cmdm.py 支持 chi3d 分支
recognition checkpoint：recognition_training/chi3d_smplx/checkpoint_0060.pth.tar
```

Chi3D 在项目里可以完整走原论文训练/评估路径，但样本小。论文写 373 条，本地处理后 H5 是 367 条，可能是预处理或过滤差异，需要复现论文时单独核对。

### InterHuman-AS

本地数据：

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5

train: 6021
val: 580
test: 1175
union: 7776
shape: [T,25,12]
```

项目用途：

```text
dataset 参数：--dataset interhuman
loader：data_loaders/a2m/interhuman.py::InterHuman
body_model：当前本地是 smpl
num_frames：模型参数分支按 150
num_actions：当前本地实现设为 1，即 interaction
训练目标：actor-conditioned reactor generation smoke / baseline
评估：eval/eval_cmdm.py 当前不支持 interhuman 分支
recognition checkpoint：本地没有 InterHuman recognition checkpoint
```

InterHuman 是我们后来补的 H5 reproduction 路径，已经能训练 smoke 和 baseline，但还不能严格复现论文 Table 4。当前新的 forecasting 论文主线选择 InterHuman，是因为它更适合“固定 150 帧窗口、前 30 帧观测、预测后 120 帧”的任务。

## 论文里的用法

论文是：

```text
ReGenNet: Towards Human Action-Reaction Synthesis
arXiv:2403.11882v1
```

论文任务不是未来预测，而是 action-reaction synthesis：

```text
given actor motion -> generate plausible reactor motion
```

作者把三套原始交互数据补上 actor-reactor order，形成：

```text
NTU120-AS
Chi3D-AS
InterHuman-AS
```

其中 `AS` 表示 asymmetry。

### 论文中的 NTU120-AS

论文说明：

```text
8118 human interaction sequences
3 cameras
26 action categories
camera 1: main protocol
cross-subject split
camera 2: viewpoint generalization
SMPL-X representation
```

用途：

```text
主实验数据集
online / unconstrained action-reaction synthesis
ablation 主数据集
viewpoint generalization 数据集
offline / constrained 扩展设置数据集
```

论文训练：

```text
batch_size=64
frame_len=60
500K steps
DDIM-5 inference
```

### 论文中的 Chi3D-AS

论文说明：

```text
8 action categories
ground-truth SMPL-X parameters
subtle hand gestures
random train/test split 4:1
frame_len=150
```

用途：

```text
验证 ReGenNet 在高质量 MoCap / SMPL-X 双人交互上的效果
与 NTU120-AS 一起作为主要 comparison 数据集
```

论文训练：

```text
batch_size=16
frame_len=150
500K steps
DDIM-5 inference
```

### 论文中的 InterHuman-AS

论文说明：

```text
InterHuman 原始数据适合 action-reaction 设置
但只包含 human body parameters，没有 dexterous hand movements
Table 1 论文口径写 InterHuman-AS motions = 6022
```

用途：

```text
作为第三个 benchmark 数据集
主要报告 online / unconstrained setting 的 Table 4
评估指标是 FID / Acc. / Div. / Multimod.
```

注意：

```text
论文正文 4.1 写三套数据都用 SMPL-X body models，
但 Table 1 又标 InterHuman modality 为 SMPL。
本地当前 InterHuman H5 是 SMPL reproduction [T,25,12]，不是 NTU/Chi3D 的 SMPL-X [T,56,6]。
```

## 论文评估口径

论文主要指标：

```text
FID
action recognition accuracy
diversity
multi-modality
```

评估方式：

```text
训练 ST-GCN action recognition model。
用 recognition feature 算 FID / diversity / multimodality。
直接算 action recognition accuracy。
root translation 会进入 recognition model，因为双人相对 root translation 对交互重要。
每次生成 1000 samples，20 random seeds，报告 95% confidence interval。
```

## 当前项目与论文的差异

```text
NTU120-AS：当前项目最接近论文口径，数据、checkpoint、eval 分支都在。
Chi3D-AS：当前项目也接近论文口径，但本地 H5 总数 367，不是论文写的 373。
InterHuman-AS：当前项目能训练，但不是完整论文评估口径；缺 InterHuman recognition checkpoint，eval_cmdm 没有 interhuman 分支，本地表示是 SMPL 25-slot。
```

## 对当前 forecasting 主线的含义

如果做 ReGenNet 原论文复现：

```text
优先 NTU120-AS 和 Chi3D-AS。
InterHuman-AS 需要补 evaluator / checkpoint / 口径确认。
```

如果做当前新论文主线：

```text
优先 InterHuman。
原因是样本最多、序列更长、交互语义更适合 150 帧窗口 forecasting。
NTU120-AS 可作为后续 action-conditioned forecasting 扩展。
Chi3D-AS 更适合作小规模泛化或 qualitative 补充。
```
