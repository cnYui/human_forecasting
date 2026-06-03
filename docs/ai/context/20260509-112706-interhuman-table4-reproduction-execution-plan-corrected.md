# InterHuman-AS Table 4 复现执行计划修正版

## 阶段总览

```text
P-1 论文目标校准
P0 环境与数据冻结
P1 InterHuman-AS 离线 H5 预处理
P2 H5 loader 与全量 smoke
P3 单卡稳定训练
P4 InterHuman 专用生成与定性检查
P5 InterHuman ST-GCN evaluator
P6 Table 4 指标复现
```

## P-1：论文目标校准

目标：

纠正 Table 4 指标口径，避免把 text-motion 指标误作为主路径。

验收：

- Table 4 主指标明确为 `FID / Acc. / Div. / Multimod.`。
- text evaluator 从 Table 4 主路径移出。
- 后续计划以 ST-GCN recognition feature evaluator 为主。

## P0：环境与数据冻结

输出：

```text
dataset/interhuman/reproduction_manifest.json
docs/ai/context/YYYYMMDD-HHMMSS-interhuman-p0-manifest.md
```

需要记录：

```text
GPU 型号和显存
torch/cuda/numpy/h5py 版本
motions 文件数量
split 样本数
actor-reactor 标注数量
可用样本数与跳过原因
是否存在 text/caption
是否存在 InterHuman recognition checkpoint
当前 git status 摘要
```

## P1：InterHuman-AS 离线 H5 预处理

新增脚本：

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

H5 entry：

```text
[T, 25, 12] float32
```

规则：

1. 读取 split ID。
2. 过滤缺失 `.pkl`、缺 actor-reactor 标注、零帧、非有限值、shape 异常样本。
3. 按 actor-reactor 标注排序。
4. 将 `root_orient + pose_body` 组成 22 个 axis-angle joints。
5. 补 2 个零关节到 24 个 SMPL rotation joints。
6. axis-angle 转 rot6d。
7. translation 以 actor 第 0 帧为原点归一化。
8. 写入第 25 个 slot。
9. 保存 meta，包括跳过样本和原因。

验收：

- 三个 H5 和 `meta.json` 均生成。
- 随机样本 shape 为 `[T,25,12]` 且数值有限。
- 与当前在线 loader 对同一 ID、同一 frame_ix 输出一致或误差在浮点范围内。

## P2：H5 loader 与全量 smoke

代码改动：

```text
data_loaders/a2m/interhuman.py
data_loaders/get_data.py
```

要求：

- `data_path` 指向 H5 时读取 H5。
- `data_path` 指向目录时保留当前 `.pkl` 在线 loader。
- `ccollate()` 后 actor/reactor 各为 `[25,6,T]`。
- `num_workers=8` 时 H5 不共享失效文件句柄。

验收：

- 1000 step smoke 可完成。
- loss 有限。
- 保存 checkpoint。
- DDIM-5 生成 4 条 test 样本且数值有限。

## P3：单卡稳定训练

新增能力：

```text
--grad_accum_steps
```

要求：

- `batch_size` 是实际 GPU batch。
- `grad_accum_steps` 控制多少个 backward 后 optimizer step。
- `args.json` 保存该参数。
- 日志记录 data step、optimizer step、effective batch size、显存峰值。
- resume 后 step 口径一致。

## P4：InterHuman 专用生成与定性检查

新增脚本：

```text
sample/cgenerate_interhuman.py
```

要求：

- 不依赖 `action_file` 或 `action_name`。
- 只从 train/test loader 取 actor `cmotion`。
- 强制或默认 `num_frames=150`。
- 支持 DDIM-5。
- 保存 `sample_id`、`split`、`length`、`cmotion`、`output`。

## P5：InterHuman ST-GCN evaluator

目标：

补齐 Table 4 的 `FID / Acc. / Div. / Multimod.` 评估链路。

任务：

1. 确认 InterHuman 类别标签来源。
2. 训练或获取 InterHuman-AS recognition checkpoint。
3. 为 `eval/eval_cmdm.py` 增加 `interhuman` 分支。
4. 为 `eval/a2m/stgcn_eval.py` 增加 `interhuman` 参数。
5. 明确 SMPL 或 SMPL-X 评估口径。

门禁：

如果没有可靠类别标签，`Acc.` 和 `Multimod.` 不能声称严格复现 Table 4。

## P6：Table 4 指标复现

目标表格：

```text
Method
FID
Acc.
Div.
Multimod.
```

协议：

```text
dataset: InterHuman-AS
setting: online, unconstrained
num_frames: 150
DDIM steps: 5
num_samples: 1000 或 full test
num_seeds: 20
```

结果标注：

- evaluator 与论文不完全一致时写 `Reproduction Attempt`。
- 使用 SMPL 时写 `SMPL reproduction attempt`。
- 训练配置低于论文时写明 `layers / latent_dim / effective batch size / steps`。
