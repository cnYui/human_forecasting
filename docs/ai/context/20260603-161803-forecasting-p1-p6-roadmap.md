# Forecasting P1-P6 完整路线图

## 文档定位

这是论文第一阶段实现的原始路线图。后续即使按 P1、P2 逐步实现，也必须能回溯到这里，避免局部实现完成后偏离最终目标。

后续每个阶段完成时，应新建阶段记录文档，并引用本文件：

```text
docs/ai/context/20260603-161803-forecasting-p1-p6-roadmap.md
```

不得用新的局部计划覆盖本路线图。

## 最终目标

论文第一阶段正式目标：

```text
Interaction-aware joint forecasting of two-person human motion from partial observations.
```

中文：

```text
基于部分观测的交互感知双人动作联合未来预测。
```

固定主协议：

```text
dataset: InterHuman
window_len: 150
obs_len: 30
pred_len: 120
input:  前 30 帧双人动作
target: 后 120 帧双人动作
output: 两个人未来动作联合预测
```

核心论文假设：

```text
双人未来动作预测不是两个单人未来预测的简单拼接；
显式建模两人关系应改善长期预测和交互一致性。
```

如果 relation-aware model 不能优于 independent / concat baseline，尤其不能改善 long-horizon 和交互关系指标，则论文主张不成立，不能继续包装成成功结果。

## 全局边界

第一阶段只做：

```text
InterHuman 150-frame deterministic two-person forecasting
active vector 表示
MSE 类和交互关系指标
repeat / independent / concat / relation-aware 四类模型或 baseline
```

第一阶段不做：

```text
ReGenNet Table 4 完整复现
自然语言条件预测
动作标签条件预测
SMPL-X 转换
ST-GCN recognition evaluator
diffusion forecasting
多模态 best-of-K / diversity
```

这些内容可以作为后续扩展，但不能阻塞 P1-P6。

## 阶段依赖图

```text
P1 Dataset + Normalizer
  -> P2 Metrics + Repeat Baseline
    -> P3 Independent / Concat Baselines
      -> P4 Relation-Aware Predictor
        -> P5 Ablation + Paper Tables
          -> P6 Visualization + Qualitative Analysis
```

禁止跳过：

```text
P1/P2 未完成，不实现 P3。
P3 未完成，不实现 P4。
P4 未完成，不写 P5 主结果结论。
P5 未完成，不把 P6 可视化当作主证明。
```

## 统一数据表示

InterHuman H5：

```text
motion: [T, 25, 12]
actor_rot6d:         motion[:, :24, 0:6]
actor_translation:   motion[:, 24, 0:3]
reactor_rot6d:       motion[:, :24, 6:12]
reactor_translation: motion[:, 24, 6:9]
```

Forecasting active vector：

```text
person_dim = 24 * 6 + 3 = 147
two_person_dim = 294
motion_active: [T, 2, 147]
```

窗口：

```text
obs:    [30, 2, 147]
target: [120, 2, 147]
```

长度规则：

```text
T >= 150: 使用
T < 150: 过滤
```

不得 padding，原因是 padding 会伪造未来动作。

## 统一指标

所有模型和 baseline 必须报告同一组指标：

```text
future_mse
rotation_mse
translation_mse
short_mse
mid_mse
long_mse
relative_root_distance_error
relative_orientation_error
inter_person_distance_consistency
```

指标必须在 original scale 上计算。训练可以使用 normalized space，但论文指标不能用 normalized 值。

## P1：Forecasting Dataset + Normalizer

### 目标

建立 InterHuman forecasting 数据协议，让所有后续模型读取完全一致的 `obs / target`。

### 输入

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5
dataset/interhuman/smpl/conditioned/meta.json
```

### 设计任务

新增 forecasting 数据读取逻辑：

```text
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
utils/forecasting_motion.py
```

需要实现：

```text
1. H5 -> active vector [T,2,147]
2. active vector -> H5-like motion，供后续可视化使用
3. T >= 150 过滤
4. train random crop
5. val/test center crop
6. forecasting collate
7. train split normalizer
8. normalizer 保存和加载
```

### 输出

Dataset item：

```text
{
  "obs": Tensor[30, 2, 147],
  "target": Tensor[120, 2, 147],
  "sample_id": str,
  "start": int,
  "length": int,
}
```

Batch：

```text
obs:    Tensor[B, 30, 2, 147]
target: Tensor[B, 120, 2, 147]
meta:   list[dict]
```

Normalizer：

```text
mean: [1,1,2,147]
std:  [1,1,2,147]
```

### 验收标准

P1 通过条件：

```text
train dataset length = 2910
val dataset length = 226
test dataset length = 508
obs shape = [30,2,147]
target shape = [120,2,147]
所有 obs/target 数值有限
train 同一样本多次读取 start 可变化
val/test 同一样本多次读取 start 固定
normalizer 可保存和加载
normalize -> denormalize 最大误差在浮点容忍范围内
```

### 失败回退

如果 shape 或样本数不对：

```text
先检查 T>=150 过滤。
再检查 split 是否读取了正确 H5。
最后检查 active vector 映射。
```

如果 normalized 数值异常：

```text
检查 std 防护。
检查是否把 padding zero channel 纳入统计。
```

### 阶段记录

P1 完成后必须新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p1-dataset-result.md
```

记录：

```text
实现文件
shape 检查
样本数
normalizer 统计摘要
smoke 命令
是否偏离本路线图
```

## P2：Metrics + Repeat Baseline

### 目标

在训练模型前，让评估闭环先成立。

### 设计任务

新增：

```text
utils/forecasting_metrics.py
eval/eval_forecasting.py
```

实现：

```text
1. future_mse
2. rotation_mse
3. translation_mse
4. short/mid/long_mse
5. relative_root_distance_error
6. relative_orientation_error
7. inter_person_distance_consistency
8. repeat baseline
9. metrics yaml/json 保存
```

### Repeat baseline

定义：

```text
pred[:, t] = obs[:, -1]
```

它不是论文贡献，只是 sanity baseline。

### 验收标准

P2 通过条件：

```text
pred == target 时 MSE 类指标为 0
pred == target 时 relative_root_distance_error 为 0
pred == target 时 relative_orientation_error 接近 0
repeat baseline 可完整评估 test split
repeat baseline 输出 metrics 文件
所有指标 key 固定，不随模型变化
```

### 失败回退

如果 `pred == target` 指标非 0：

```text
先修 metrics，不进入 P3。
```

如果 repeat baseline 评估无法跑完：

```text
先修 dataset / collate / dataloader，不进入模型训练。
```

### 阶段记录

P2 完成后必须新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p2-metrics-repeat-result.md
```

记录：

```text
sanity check 结果
repeat baseline test 指标
metrics 文件路径
是否偏离本路线图
```

## P3：Independent / Concat Baselines

### 目标

建立两个可训练 baseline，用于证明 relation-aware model 不是只赢 repeat baseline。

### Baseline 1：Independent Predictor

定义：

```text
A_obs -> A_future
B_obs -> B_future
concat(A_future, B_future)
```

约束：

```text
A 分支不能看到 B_obs。
B 分支不能看到 A_obs。
```

建议结构：

```text
shared GRU encoder 或 two-tower GRU
MLP decoder
输出 [B,120,2,147]
```

### Baseline 2：Concat No-Relation Predictor

定义：

```text
concat(A_obs, B_obs) -> future(A,B)
```

约束：

```text
可以看到双人历史，但不能显式构造 relation features。
```

建议结构：

```text
GRU encoder over [B,30,294]
MLP decoder -> [B,120,2,147]
```

### 设计任务

新增：

```text
model/forecasting.py
train/train_forecasting.py
```

实现：

```text
1. supervised training loop
2. checkpoint 保存
3. args.json 保存
4. normalizer 加载
5. val/test evaluation
6. grad_accum_steps
7. max_samples smoke
```

### 训练 loss

第一版使用：

```text
normalized active-vector MSE
```

不要在 P3 加 relation loss。

### 验收标准

P3 通过条件：

```text
independent max_samples smoke 通过
concat max_samples smoke 通过
两个模型都能保存 checkpoint
两个模型都能在 val/test 输出统一 metrics
两个模型至少优于 repeat baseline 的 future_mse
concat 与 independent 的差异可解释
```

### 失败回退

如果训练 loss 不下降：

```text
检查 normalizer。
检查 target 是否错位。
检查 decoder 输出 shape。
```

如果模型赢不了 repeat：

```text
先降低模型复杂度或学习率排查。
不要直接进入 P4。
```

### 阶段记录

P3 完成后必须新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p3-baselines-result.md
```

记录：

```text
模型配置
训练命令
checkpoint
val/test metrics
repeat / independent / concat 对比表
是否偏离本路线图
```

## P4：Relation-Aware Joint Predictor

### 目标

实现论文核心模型：显式关系推理 + 双人联合未来预测。

### 输入

```text
obs: [B,30,2,147]
```

### 关系特征

第一版关系特征：

```text
relative root translation: trans_A - trans_B
relative root velocity: velocity_A - velocity_B
root distance: ||trans_A - trans_B||
relative root orientation: R_A^T R_B
```

可以派生：

```text
distance velocity
approach / separate trend
```

但第一版不要加入人工阶段标签。

### 模型结构

最小结构：

```text
person_encoder(A_obs) -> h_A
person_encoder(B_obs) -> h_B
relation_encoder(relation_features) -> h_rel
fuse(h_A, h_B, h_rel) -> joint decoder -> future(A,B)
```

允许的 relation encoder：

```text
GRU
MLP over temporal pooled relation features
small TransformerEncoder if PyTorch 1.7.1 支持路径稳定
```

第一版优先：

```text
GRU relation encoder
```

### Loss

第一次训练：

```text
normalized active-vector MSE
```

之后可做 ablation：

```text
+ relative_root_distance_loss
+ relative_orientation_loss
```

### 验收标准

P4 通过条件：

```text
relation-aware smoke 通过
relation-aware test metrics 完整输出
relation-aware future_mse 不差于 concat
relation-aware long_mse 优于 concat
relation-aware 交互指标优于 concat 或 independent
```

最低论文主张门槛：

```text
long_mse 和至少一个 relation metric 明显优于 concat no-relation。
```

如果只在 future_mse 上微弱变化，不能声称关系建模有效。

### 失败回退

如果 relation-aware 不优于 concat：

```text
1. 检查 relation features 是否正确。
2. 检查 concat baseline 是否过强但无 relation 指标提升。
3. 加 relation loss 做 ablation。
4. 增加 cross-attention。
5. 最后才考虑 graph network。
```

不能直接进入论文包装。

### 阶段记录

P4 完成后必须新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p4-relation-result.md
```

记录：

```text
relation features
模型配置
训练命令
checkpoint
与 P3 baselines 的完整对比
是否满足论文主张门槛
是否偏离本路线图
```

## P5：Ablation + Paper Tables

### 目标

证明 relation-aware 不是偶然变好，并整理论文可用表格。

### 必做 ablation

结构消融：

```text
without relation encoder
with relation encoder
```

关系特征消融：

```text
relative translation only
relative orientation only
relative velocity only
all relation features
```

观测比例消融：

```text
10%: obs=15, pred=135
20%: obs=30, pred=120
30%: obs=45, pred=105
50%: obs=75, pred=75
```

主表仍以 20% 为主协议。

### 可选 ablation

```text
hidden_dim 256 / 512
GRU vs small Transformer
with / without relation loss
```

### 论文表格

主结果表：

```text
Repeat
Independent
Concat no-relation
Relation-aware
```

列：

```text
future_mse
rotation_mse
translation_mse
long_mse
relative_root_distance_error
relative_orientation_error
inter_person_distance_consistency
```

Ablation 表：

```text
relation feature set
future_mse
long_mse
relation metrics
```

观测比例表：

```text
obs ratio
independent
concat
ours
```

### 验收标准

P5 通过条件：

```text
所有表格指标来自同一个 eval_forecasting 协议。
所有实验记录 checkpoint 和 args。
主表支持论文核心主张。
ablation 支持 relation features 的必要性。
```

### 失败回退

如果 ablation 不支持主张：

```text
回到 P4 模型设计。
不要只挑选有利指标。
```

如果观测比例不稳定：

```text
先固定 20% 主协议。
其他比例作为补充，不影响第一阶段主线。
```

### 阶段记录

P5 完成后必须新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p5-ablation-tables.md
```

记录：

```text
所有实验清单
表格数据源
是否满足论文主张
失败或不稳定实验
是否偏离本路线图
```

## P6：Visualization + Qualitative Analysis

### 目标

用定性结果解释模型如何改善双人关系，而不是只报告 MSE。

### 任务

实现或复用：

```text
active vector -> H5-like [T,25,12]
H5-like -> render / rot2xyz 输入
obs / gt / pred 对比保存
relative root distance curve
relative orientation curve
```

可视化样本：

```text
至少 8 个 test samples
覆盖短序列边界和长序列
覆盖 relation-aware 明显优于 concat 的样本
覆盖 relation-aware 失败样本
```

### 输出

```text
results/forecasting/interhuman/{run_name}/
```

建议内容：

```text
sample_id
obs.npy
gt.npy
pred_repeat.npy
pred_concat.npy
pred_relation.npy
distance_curve.png
orientation_curve.png
rendered videos or frame sequences
```

### 验收标准

P6 通过条件：

```text
至少 8 个样本可视化输出。
所有 pred 数值有限。
obs / gt / pred 时间轴对齐。
distance / orientation 曲线与 metrics 一致。
失败样本也被记录，而不是只保留成功样本。
```

### 失败回退

如果 render 链路卡住：

```text
先输出 npy + 曲线图。
不要让渲染阻塞论文主指标。
```

如果可视化显示动作崩坏但 MSE 低：

```text
回到 P4/P5，检查指标是否遗漏关键关系质量。
```

### 阶段记录

P6 完成后必须新建：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p6-qualitative-result.md
```

记录：

```text
样本列表
输出路径
成功案例
失败案例
图表说明
是否偏离本路线图
```

## 阶段间全局记录要求

每个阶段实现完成后必须记录：

```text
1. 本阶段引用的路线图文件。
2. 实际实现文件。
3. 是否偏离原计划。
4. 偏离原因。
5. 验收命令。
6. 验收结果。
7. 下一阶段是否允许开始。
```

如果任何阶段未达到验收标准：

```text
不能把状态写为完成。
不能进入下一阶段做论文结论。
可以做修复文档和修复实现。
```

## 最短执行顺序

推荐执行：

```text
1. P1 dataset + normalizer smoke
2. P2 metrics + repeat baseline
3. P3 concat baseline
4. P3 independent baseline
5. P4 relation-aware model
6. P5 main table
7. P5 ablations
8. P6 qualitative analysis
```

原因：

```text
concat baseline 最接近 relation-aware 的公平对照；
如果 concat baseline 都无法赢 repeat，关系模型没有实现意义。
```

## 当前下一步

下一步允许开始 P1，但必须以本文件为原始路线图，并且实现前引用 P1/P2 详细设计：

```text
docs/ai/context/20260603-161334-forecasting-p1-p2-design.md
```

P1 实现期间不得修改：

```text
train/train_mdm.py
model/cmdm.py
diffusion/gaussian_diffusion.py
eval/eval_cmdm.py
```

除非新文档明确说明旧 ReGenNet 路径需要被纳入扩展阶段。
