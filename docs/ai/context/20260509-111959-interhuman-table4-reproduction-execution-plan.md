# InterHuman-AS Table 4 复现执行计划

## 执行计划总览

本计划按“先保证训练链路，再保证数据格式，再追评估协议”的顺序推进。不要直接从 Table 4 指标开始做，因为当前仓库缺 InterHuman 正式路径和 evaluator，直接长训会掩盖数据、loss、评估不一致的问题。

推荐执行顺序：

```text
P0 环境与数据冻结
P1 InterHuman-AS 离线预处理
P2 H5 数据加载与全量 smoke
P3 单卡稳定训练
P4 生成与定性检查
P5 文本数据与 evaluator 补齐
P6 Table 4 指标复现
```

每个阶段都要留下可复跑的命令、日志、配置和输出文件。所有不确定项必须记录在 `meta.json` 或实验 README 中。

## P0：环境与数据冻结

目标：

确认当前机器、依赖、数据和无效样本清单，形成固定输入。后续所有实验以这个状态为基线。

输入：

```text
dataset/interhuman/motions/*.pkl
dataset/interhuman/annotations_interhuman/interhuman_label.json
dataset/interhuman/split/*.txt
/home/rpartx3080/.local/micromamba/envs/regennet
```

输出：

```text
dataset/interhuman/reproduction_manifest.json
docs/ai/interhuman_table4_reproduction_log.md
```

需要记录：

```text
GPU 型号和显存
torch/cuda/numpy 版本
motions 文件数量
actor-reactor 标注数量
train/val/test 可用样本数
过滤掉的样本 ID 和原因
当前 git diff 摘要
```

建议命令：

```bash
nvidia-smi
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python - <<'PY'
import torch, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("numpy", numpy.__version__)
PY
```

验收：

- manifest 文件存在。
- 数据统计和已知无效样本一致。
- 没有训练进程残留。

## P1：InterHuman-AS 离线预处理

目标：

把原始 InterHuman `.pkl` 转换成 ReGenNet 可直接训练的 actor-reactor H5 文件，冻结数据格式。

新增脚本：

```text
preprocess/interhuman_as.py
```

输出目录：

```text
dataset/interhuman/smpl/conditioned/
```

输出文件：

```text
interhuman_train.h5
interhuman_val.h5
interhuman_test.h5
meta.json
```

H5 数据约定：

```text
key: sample id，例如 "3292"
value: [T, 25, 12], float32
```

其中：

```text
value[:, :, 0:6]  = actor rot6d + translation slot
value[:, :, 6:12] = reactor rot6d + translation slot
```

预处理规则：

1. 只处理 split 中存在的 ID。
2. 必须存在对应 `.pkl`。
3. 必须存在 actor-reactor 标注。
4. `frames > 0`。
5. `person1/person2` 的 `trans/root_orient/pose_body` 必须都是有限值。
6. 根据标注把 actor 放前，reactor 放后。
7. `root_orient + pose_body` 组成 22 个 axis-angle joints；补 2 个零关节到 24 个 SMPL rotation joints。
8. axis-angle 转 rot6d。
9. 将 translation 写入第 25 个 slot 的前三维，后三维为 0。

需要确认的技术点：

- 当前 `pose_body` 是 21 个 body joints，即 `[T, 63]`。
- 当前 smoke loader 补 2 个零关节后能通过 SMPL layer。
- 预处理脚本应复用 `utils.rotation_conversions`，避免重复实现旋转转换。

验收：

- 三个 H5 文件均生成。
- `meta.json` 包含输入路径、输出路径、样本数、跳过样本、shape、时间戳。
- 随机抽 10 条 H5 样本，形状均为 `[T,25,12]`，且数值全有限。
- 与在线 loader 对同一 ID 的输出误差为 0 或浮点可接受误差。

## P2：H5 数据加载与全量 Smoke

目标：

将训练路径切到 H5 数据，验证完整 train split 可迭代、可训练、无 NaN/OOM。

代码改动：

```text
data_loaders/a2m/interhuman.py
data_loaders/get_data.py
```

设计：

- `InterHuman` loader 优先读取 `data_path` 指向的 H5。
- 如果 `data_path` 是目录，则回退到当前在线 `.pkl` loader。
- H5 loader 行为与 `Feeder` 保持一致，输出 `inp`。
- `ccollate()` 继续负责拆 actor 和 reactor。

全量 smoke 配置：

```text
batch_size: 1
num_frames: 150
layers: 2
latent_dim: 128
max_samples: -1
num_steps: 1000
lambda_orient/body/transl: 0
```

验收：

- 训练至少 1000 optimizer steps。
- loss 为有限值。
- GPU 显存不超过 10GB。
- 保存 checkpoint。
- 随机取 test split 生成 4 条样本，输出全有限。

继续条件：

如果此阶段出现 NaN，优先排查数据是否存在非有限值；如果出现 OOM，先降低 `num_frames` 做定位，再恢复 150。

## P3：单卡稳定训练

目标：

在 RTX 3080 上训练一个有实际生成质量的 InterHuman-AS baseline。

推荐配置：

```text
batch_size: 1
num_frames: 150
layers: 4
latent_dim: 256
num_steps: 50000
save_interval: 5000
log_interval: 100
DDIM eval steps: 5
```

需要补充的训练能力：

```text
gradient_accumulation_steps
```

建议新增参数：

```text
--grad_accum_steps
```

实现位置：

```text
utils/parser_util.py
train/training_loop.py
```

实现原则：

- `batch_size` 表示实际 GPU batch。
- `grad_accum_steps` 表示多少个 forward/backward 后执行一次 optimizer step。
- 日志中同时记录 `data_step` 和 `optimizer_step`。
- checkpoint 文件名以 optimizer step 为准。

验收：

- `grad_accum_steps=1` 时行为与当前训练一致。
- `grad_accum_steps>1` 时显存不明显增加。
- 能连续训练至少 5000 optimizer steps。
- loss 曲线没有持续 NaN 或爆炸。

## P4：生成与定性检查

目标：

在正式指标前确认生成结果不是数值可行但动作崩坏。

需要新增或适配：

```text
sample/cgenerate_interhuman.py
render/interhuman_render.py
```

最小生成协议：

```text
从 test split 抽 32 条 actor motion
每条生成 1-3 个 reactor motion
DDIM steps = 5
保存 output/cmotion/text/id/length
```

输出：

```text
results/interhuman_baseline_3080/results.npy
results/interhuman_baseline_3080/results.txt
results/interhuman_baseline_3080/rendered/*.mp4
```

验收：

- 所有生成 tensor 数值有限。
- 能转 xyz 或 mesh。
- 随机 10 个视频中，actor 和 reactor 都能正常显示。
- 没有明显全零、爆炸、身体飞散。

## P5：文本数据与 Evaluator

目标：

补齐 Table 4 必须的 text-conditioned 评估链路。

当前缺口：

```text
InterHuman 文本描述文件
InterHuman 文本 token 处理
text-motion matching evaluator
evaluator checkpoint
R Precision / MM Dist 计算脚本
```

需要先确认 InterHuman 原始数据包中是否已下载文本部分。建议检查：

```text
dataset/interhuman/texts
dataset/interhuman/annots
dataset/interhuman/captions
dataset/interhuman/*text*
```

如果本地没有，需要从 InterHuman 官方数据源补下载文本标注。

实现路线：

1. 先找到 InterHuman 官方文本格式。
2. 建立 `sample_id -> captions` 映射。
3. 在 H5 metadata 或单独 JSON 中保存文本。
4. 新增 `InterHumanTextDataset` 或扩展现有 loader。
5. 对接现有 HumanML evaluator 框架，或移植 InterGen/InterHuman 官方 evaluator。

验收：

- 任意 sample ID 能取到 motion 和对应文本。
- batch 中有 `text` 或 token 字段。
- evaluator 能对 real motion 跑出稳定 feature。
- R Precision 和 MM Dist 对真实数据能计算。

重要说明：

没有文本数据和 evaluator 时，不能声称复现 Table 4。最多只能声称完成 InterHuman-AS actor-conditioned reaction generation training。

## P6：Table 4 指标复现

目标：

按论文 Table 4 协议生成和评估，形成可对比表格。

目标表格列：

```text
Method
R Precision (Top 3)
FID
MM Dist
Diversity
MModality
```

复现实验设置：

```text
dataset: InterHuman-AS
setting: online, unconstrained
num_frames: 150
DDIM steps: 5
replication_times: 20 如果时间允许，否则先 3
```

生成规模：

优先级从低到高：

```text
debug: 64 samples
small: 512 samples
paper-attempt: full test split or 1000 samples
```

验收：

- `Real` 指标可计算。
- ReGenNet 指标可计算。
- 每次评估保存原始 JSON/YAML。
- 表格脚本能输出均值和置信区间。

结果标注规则：

- 如果 evaluator 与论文不完全一致，表格标题必须写 `Reproduction Attempt`。
- 如果模型配置低于论文，例如 `layers=4, latent_dim=256`，表格备注必须写明。
- 如果只跑 3 次而不是 20 次，置信区间备注必须写明。

## 文件级任务清单

### 新增文件

```text
preprocess/interhuman_as.py
eval/eval_interhuman.py
eval/interhuman_metrics.py
sample/cgenerate_interhuman.py
docs/ai/interhuman_table4_reproduction_log.md
```

### 修改文件

```text
data_loaders/a2m/interhuman.py
data_loaders/get_data.py
utils/parser_util.py
utils/model_util.py
train/training_loop.py
sample/cgenerate.py 或新增专用脚本
```

### 可选修改

```text
diffusion/gaussian_diffusion.py
```

仅当启用 SMPL-compatible interaction loss 时修改。

## 决策点

### 决策点 1：是否坚持 SMPL-X

当前 InterHuman 原始数据是 SMPL body 参数。若坚持 SMPL-X，需要额外转换流程，风险高。

建议：

```text
先使用 SMPL 完成可复现训练和 evaluator。
再单独评估 SMPL-X 转换必要性。
```

### 决策点 2：是否必须完全复现 Table 4 数值

如果目标是论文级数值对齐，必须获得和作者一致的 evaluator 或非常接近的官方 evaluator。

如果目标是工程复现，则可以使用文档化 evaluator，结果命名为 reproduction attempt。

### 决策点 3：训练预算

RTX 3080 单卡长训成本较高。建议先确认：

```text
是否接受 50k-100k steps baseline
是否需要 300k+ steps 长训
是否能使用远程多卡机器
```

## 预计时间

粗略估计：

```text
P0: 0.5 天
P1: 1 天
P2: 0.5 天
P3: 1-5 天训练时间，取决于 steps
P4: 0.5-1 天
P5: 2-5 天，取决于文本数据和 evaluator 来源
P6: 1-3 天，取决于评估次数
```

如果只做单卡 baseline，不追完整 Table 4：

```text
2-4 天可得到第一个可用模型和可视化结果
```

如果追 Table 4 级别复现：

```text
至少 1-2 周，并且依赖文本数据和 evaluator 是否顺利获得
```

## 下一步建议

立即执行的下一步：

1. 写 `preprocess/interhuman_as.py`，把当前在线转换固化成 H5。
2. 用 H5 loader 跑全量 1000 step smoke。
3. 检查本地是否有 InterHuman 文本标注。
4. 再决定是否开始 50k step 单卡 baseline。

不要立即开始 300k+ steps 长训。当前最大不确定性不是训练能不能跑，而是 Table 4 evaluator 和文本条件链路是否能对齐。
