# InterHuman-AS Table 4 执行计划审查

## 结论

当前执行计划需要补充，而且应先纠正目标指标。

设计文档把 `2403.11882v1.pdf` 的 Table 4 写成了文本条件指标：

```text
R Precision
FID
MM Dist
Diversity
MModality
```

但本地 PDF 中 Table 4 的标题是：

```text
Comparison to state-of-the-arts on the online, unconstrained setting for human action-reaction synthesis on the InterHuman-AS dataset.
```

该表列为：

```text
Method
FID
Acc.
Div.
Multimod.
```

因此，复现 Table 4 的主路径应是 action-conditioned / actor-conditioned reaction generation 的 ST-GCN feature 评估链路，而不是 text-motion evaluator 链路。文本评估属于论文附录中 text-conditioned human reaction generation 的指标说明，不应阻塞 Table 4。

## 当前项目确认

InterHuman 在线 loader 已存在：

```text
data_loaders/a2m/interhuman.py
```

它会读取：

```text
dataset/interhuman/motions/*.pkl
dataset/interhuman/annotations_interhuman/interhuman_label.json
dataset/interhuman/split/*.txt
```

当前输出经 `ccollate()` 后可得到：

```text
motion:  [B, 25, 6, 150]
cmotion: [B, 25, 6, 150]
```

本地 InterHuman 数据没有发现 text/caption 目录，只有 motions、motions_processed、actor-reactor 标注和 split。

当前 `sample/cgenerate.py` 不适合直接作为 InterHuman 生成入口：

- 只对 `chi3d` 设置 `max_frames=150`，`interhuman` 会落到 60。
- `is_using_data=True` 时后续仍使用 `action_text`，存在未定义变量风险。
- 非数据路径依赖 `_get_item_cmotion_index()`，而 `InterHuman` loader 当前没有实现该方法。

当前 `eval/eval_cmdm.py` 不支持 InterHuman：

- 只允许 `ntu` 和 `chi3d`。
- `num_frames` 只为 `ntu/chi3d` 设置。
- 强制 `body_model='smplx'`。

当前 ST-GCN evaluator 也没有 InterHuman 分支：

- `eval/a2m/stgcn_eval.py` 只给 `ntu` 和 `chi3d` 设置 `num_classes`。
- 本地已有 recognition checkpoint 只有 `ntu_smplx` 和 `chi3d_smplx`，没有 InterHuman。

当前 interaction loss 对 SMPL 不完整：

- `lambda_transl` 使用固定索引 `55`。
- InterHuman SMPL 表示只有 `25` 个 slot，translation 在最后一个 slot。

## 计划应补充的内容

### P-1：论文目标校准

在 P0 前新增一个阶段，先确认复现目标：

```text
Table 4 主指标：FID / Acc. / Div. / Multimod.
主 evaluator：ST-GCN action recognition feature evaluator
文本指标：移出 Table 4 主路径，作为可选扩展
```

验收：

- 设计文档中 Table 4 指标已纠正。
- P5/P6 不再把 text-motion evaluator 作为 Table 4 必需项。
- 结果表格列和论文 Table 4 一致。

### P0：补充数据和论文 manifest

P0 除了环境和数据统计，还应记录：

```text
PDF Table 4 指标列
本地是否存在 InterHuman text/caption
本地是否存在 InterHuman recognition checkpoint
当前使用 SMPL 还是 SMPL-X
当前计划是否是 exact reproduction 或 reproduction attempt
```

输出路径建议放在 `docs/ai/context/`，避免使用 `docs/ai/interhuman_table4_reproduction_log.md` 这种不符合项目约定的路径。

### P1：补充 H5 格式对齐检查

P1 需要明确 H5 不只是 `[T,25,12]`，还必须能被现有 `Dataset._load()` 或专用 H5 loader 无歧义读取：

```text
_poses:    [T, 24, 12]
_joints3d: [T, 1, 6]
最终 inp: [25, 12, T]
ccollate 后 actor/reactor 各为 [25, 6, T]
```

还应补充 translation 规范：

```text
以 actor 第 0 帧 translation 为原点
actor 和 reactor 使用同一个原点
```

这是当前 `Dataset._load()` 的实际行为。

### P2：补充 loader 单元测试

P2 应增加独立测试或检查脚本，覆盖：

```text
online pkl loader 与 H5 loader 同 ID、同 frame_ix 输出一致
train/val/test split 不互相污染
max_samples 只用于 smoke，不改变 full manifest
num_workers=8 时 H5 读取不出现句柄共享问题
```

### P3：补充训练稳定性门禁

梯度累积之外，还应补充：

```text
保存 args.json 中的 grad_accum_steps
resume 后 optimizer_step/data_step 一致
有效 batch size 明确写入日志
记录每秒 step、显存峰值、预计完成时间
```

### P4：补充 InterHuman 专用生成入口要求

应明确 `sample/cgenerate_interhuman.py` 的职责：

```text
只从 test/train loader 取 actor cmotion
不依赖 action_file/action_name
强制 num_frames=150
保存 sample_id、split、frame_ix、cmotion、output、length
支持 DDIM-5
```

这比复用当前 `sample/cgenerate.py` 风险低。

### P5：改为 InterHuman ST-GCN recognition evaluator

原 P5 文本链路应改成：

```text
训练或获取 InterHuman-AS recognition checkpoint
支持 SMPL 或确定 SMPL-X 转换
在 eval/eval_cmdm.py 中加入 interhuman 分支
在 eval/a2m/stgcn_eval.py 中加入 interhuman num_classes
确认 InterHuman 类别标签来源
```

关键决策：

```text
如果 InterHuman-AS 没有可靠动作类别标签，则 Acc. 和 Multimod. 的类别分组无法严格复现 Table 4。
```

需要先找官方 InterHuman category/text metadata，而不是直接长训 ReGenNet。

### P6：补充 Table 4 结果口径

P6 的目标表格应改成：

```text
Method
FID
Acc.
Div.
Multimod.
```

复现设置应包括：

```text
train-conditioned
test-conditioned
num_seeds=20
num_samples=1000 或 full test
DDIM steps=5
```

如果使用 SMPL 而不是论文描述中的 SMPL-X，结果必须标注：

```text
SMPL reproduction attempt, not exact Table 4 reproduction
```

## 建议优先级

优先级 1：

```text
修正设计文档和执行计划中的 Table 4 指标定义。
```

优先级 2：

```text
确认 InterHuman 类别标签和 recognition checkpoint 路线。
```

优先级 3：

```text
再实现 H5 预处理、H5 loader 和全量 smoke。
```

如果不先解决 evaluator 和类别标签，后续训练可以产生模型，但无法严肃复现 Table 4。
