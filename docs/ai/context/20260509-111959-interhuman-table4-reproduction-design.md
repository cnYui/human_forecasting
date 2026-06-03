# InterHuman-AS Table 4 复现设计

## 目标

在当前 ReGenNet 仓库和本地已有 InterHuman 数据的基础上，尽可能复现 `2403.11882v1.pdf` 中 Table 4 的 InterHuman-AS 实验结果。

目标任务是在线、无意图条件的人体动作-反应合成：

```text
actor motion -> ReGenNet -> reactor motion
```

论文 Table 4 在 InterHuman-AS 上报告的是文本条件风格的指标：

```text
R Precision (Top 3)
FID
MM Dist
Diversity
MModality
```

因此，本复现设计分为两个层级：

1. 使用全部有效的 InterHuman-AS 样本训练一个完整的反应生成模型。
2. 复现 Table 4 的评估协议，包括文本-动作检索相关指标。

第一个层级在当前数据和代码适配基础上可行。第二个层级还需要额外的文本数据，以及兼容 InterHuman 的评估器。

## 当前状态

本地数据：

```text
dataset/interhuman/motions/*.pkl
dataset/interhuman/annotations_interhuman/interhuman_label.json
dataset/interhuman/split/train.txt
dataset/interhuman/split/val.txt
dataset/interhuman/split/test.txt
```

过滤后的数据可用性：

```text
train split: 列出 6022 条，可用 6021 条
val split:   列出 580 条， 可用 580 条
test split:  列出 1177 条，可用 1175 条
```

已知无效或不完整的样本 ID：

```text
3945
3433
4106
```

当前仓库支持情况：

- 原始代码已经支持 `ntu` 和 `chi3d` 数据路径。
- 已通过 `data_loaders/a2m/interhuman.py` 为冒烟测试加入 `interhuman` 支持。
- 小配置下，RTX 3080 可以跑通冒烟训练。
- 冒烟 DDIM 生成能够产生有限数值的样本。

当前冒烟测试使用的动作表示：

```text
body_model: smpl
单人 motion shape: [25, 6, T]
双人样本 shape:    [25, 12, T]
```

这足以验证训练链路，但还不是 Table 4 级别复现。

## 所需数据格式

ReGenNet 的 `cmdm` 路径期望每条样本已经按 actor-reactor 顺序配对：

```text
inp: [J, 2*C, T]
```

对于 rot6d 的 SMPL 身体表示：

```text
J = 25
C = 6
T = 150
inp[:, 0:6, :]  = actor motion
inp[:, 6:12, :] = reactor motion
```

随后 collate 函数会生成：

```text
cond["y"]["cmotion"] = actor motion
motion               = reactor motion
```

每个 InterHuman `.pkl` 文件包含：

```text
person1/trans
person1/root_orient
person1/pose_body
person2/trans
person2/root_orient
person2/pose_body
```

actor-reactor 标注决定两个人的顺序：

```text
0 -> person1 是 actor，person2 是 reactor
1 -> person2 是 actor，person1 是 reactor
```

## 预处理方案

新增一个离线转换脚本：

```text
preprocess/interhuman_as.py
```

输出：

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5
dataset/interhuman/smpl/conditioned/meta.json
```

每个 H5 entry 应存储：

```text
[T, 25, 12]
```

其中：

```text
24 个 SMPL 旋转关节，使用 rot6d 表示
1 个 translation slot
actor/reaction 在 channel 维拼接
```

预处理应执行：

1. 读取 split ID。
2. 丢弃缺失、无标注或零帧样本。
3. 应用 actor-reactor 顺序。
4. 将 axis-angle 转换为 rot6d。
5. 将 root translation 追加为最后一个 slot。
6. 保存固定长度或可变长度序列。
7. 保存 metadata，记录跳过的样本 ID 和原因。

当前在线 loader 可以保留用于开发，但正式训练和评估应使用冻结后的预处理文件，以提升可复现性。

## 模型范围

Table 4 应使用同一类 ReGenNet 架构：

```text
diffusion model
Transformer decoder
online causal mask
actor motion conditioning
unconstrained actor intention
```

论文规模配置：

```text
arch: online
layers: 8
latent_dim: 512
diffusion_steps: 1000
noise_schedule: cosine
num_frames: 150
DDIM evaluation steps: 5
```

论文说明 NTU120-AS 和 InterHuman-AS 使用 batch size 64，并在 4 张 A100 GPU 上训练。这个配置不能在单张 10GB RTX 3080 上原样运行。

## 单张 RTX 3080 训练计划

采用分阶段训练。

阶段 1：完整数据冒烟训练

```text
batch_size: 1
layers: 2
latent_dim: 128
num_frames: 150
num_steps: 1000
lambda_orient/body/transl: 0
```

目的：证明完整 train split 可以被遍历，并且不会出现 NaN 或 OOM。

阶段 2：实用单卡 baseline

```text
batch_size: 1-2
layers: 4
latent_dim: 256
num_frames: 150
num_steps: 50k-100k
DDIM: 5
```

目的：产生可用的定性样本和粗略指标。

阶段 3：更接近论文配置的训练

```text
batch_size: 1-2
gradient_accumulation_steps: 16-64
effective batch size: 32-64
layers: 显存允许则 8，否则 4
latent_dim: 显存允许则 512，否则 256
num_steps: 300k-600k
```

所需代码改动：

- 为 `TrainLoop` 增加真正的梯度累积。
- 将 `microbatch` 和 effective batch size 分离。
- 分开记录 optimizer step 和 data step。

## Loss 设计

当前冒烟测试关闭了：

```text
lambda_orient
lambda_body
lambda_transl
```

原因：原始 interaction loss 假设使用 NTU/Chi3D 的 SMPL-X 风格 layout，其中 translation 索引依赖 joint slot `55` 附近的位置。当前 InterHuman 冒烟数据使用的是 SMPL，只有 `25` 个 slot。

要达到 Table 4 级别复现，需要在以下两条路径中选择一条：

路径 A：SMPL 兼容的 interaction loss

- 重写 interaction loss，让任意 body model 都使用最后一个 slot 作为 translation。
- 从 root joint slot `0` 计算相对朝向。
- 使用 `rot2xyz` 生成的 SMPL joints 计算 body distance。
- 这是最适合本地 InterHuman 数据的务实路径。

路径 B：SMPL-X 转换

- 将 InterHuman 的 SMPL body motion 转成 SMPL-X 兼容表示。
- 保留原始 loss layout。
- 如果作者内部实际使用的是 SMPL-X，这条路径更接近论文，但需要稳定可靠的 SMPL-to-SMPL-X 转换流程。

建议优先实现：路径 A。

## 文本条件需求

Table 4 不只是 action-conditioned generation。它报告了：

```text
R Precision
MM Dist
```

这些指标需要文本描述和文本-动作评估器。

缺失组件：

```text
InterHuman text/caption files
InterHuman text tokenization pipeline
text-conditioned InterHuman dataset loader
text-motion matching evaluator
trained evaluator checkpoint
Table 4 metric runner
```

如果本地只有 `motions/*.pkl` 和 actor-reactor 标注，那么可以训练 actor-conditioned reaction generation，但不能复现 Table 4 指标。

## 评估设计

生成评估应构造两组生成结果：

```text
train-conditioned: actor motions sampled from train split
test-conditioned:  actor motions sampled from test split
```

对于 Table 4：

1. 使用 DDIM-5 生成 reactions。
2. 提取 motion features。
3. 提取 text features。
4. 计算：
   - R Precision Top 3
   - FID
   - MM Dist
   - Diversity
   - MModality
5. 使用多个随机种子重复实验。
6. 报告均值和 95% 置信区间。

需要新增的评估脚本：

```text
eval/eval_interhuman.py
eval/interhuman_metrics.py
```

如果无法获得论文完全一致的 evaluator，结果应标注为 reproduction-attempt metrics，而不是精确的 Table 4 复现。

## 具体实施任务

1. 新增离线 InterHuman-AS 预处理。
2. 用基于 H5 的 InterHuman loader 替代冒烟测试用的在线 loader。
3. 增加 SMPL 兼容的 interaction loss。
4. 为 10GB GPU 训练增加梯度累积。
5. 增加完整数据训练命令模板。
6. 增加 InterHuman 采样生成命令。
7. 获取并验证 InterHuman 文本描述。
8. 实现 InterHuman text-conditioned loader。
9. 实现或移植 text-motion evaluator。
10. 分阶段训练，并与 Table 4 对比。

## 建议命令

完整数据冒烟训练：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m train.train_mdm \
  --setting cmdm \
  --save_dir save/interhuman/smoke_full \
  --dataset interhuman \
  --data_path dataset/interhuman \
  --cond_mask_prob 0 \
  --num_person 2 \
  --layers 2 \
  --latent_dim 128 \
  --num_frames 150 \
  --arch online \
  --overwrite \
  --pose_rep rot6d \
  --body_model smpl \
  --train_platform_type NoPlatform \
  --unconstrained \
  --batch_size 1 \
  --num_steps 1000 \
  --save_interval 500 \
  --log_interval 50 \
  --lambda_orient 0 \
  --lambda_body 0 \
  --lambda_transl 0
```

单卡 baseline 训练：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m train.train_mdm \
  --setting cmdm \
  --save_dir save/interhuman/baseline_3080 \
  --dataset interhuman \
  --data_path dataset/interhuman \
  --cond_mask_prob 0 \
  --num_person 2 \
  --layers 4 \
  --latent_dim 256 \
  --num_frames 150 \
  --arch online \
  --overwrite \
  --pose_rep rot6d \
  --body_model smpl \
  --train_platform_type TensorboardPlatform \
  --unconstrained \
  --batch_size 1 \
  --num_steps 50000 \
  --save_interval 5000 \
  --log_interval 100 \
  --lambda_orient 0 \
  --lambda_body 0 \
  --lambda_transl 0
```

这些命令还不是 Table 4 复现命令。它们是为了在本地硬件上逐步达到稳定 InterHuman-AS 训练的阶段性命令。

## 风险

主要风险：

- 公开仓库没有包含作者的 InterHuman 训练路径。
- 精确的 Table 4 evaluator 可能不在当前仓库中。
- 本地文本数据可能不完整。
- 单张 RTX 3080 的训练速度更慢，batch size 也低于论文规模。
- SMPL 与 SMPL-X 表示差异可能导致指标不可直接比较。
- 如果没有完全一致的 evaluator，复现结果只能作为方向性参考，不能视为数值等价复现。

## 验收标准

最低可接受复现：

- 完整 InterHuman-AS train split 能训练，不出现 NaN/OOM。
- test split 可使用 DDIM-5 生成。
- 生成样本数值有限，并且可以渲染。
- 指标能够用文档化的 evaluator 代码稳定计算。

Table 4 级别复现：

- 使用完整 InterHuman-AS train/test split。
- 使用 actor-reactor 标注。
- 使用与论文一致的帧长和动作表示。
- 使用文本描述。
- 使用与论文兼容的 text-motion evaluator。
- 报告 Table 4 指标及置信区间。
