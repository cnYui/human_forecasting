# Forecasting P6 Qualitative Analysis 设计文档

## 文档定位

本文按用户要求先使用 `using-superpowers` 工作流，再基于最终正式设计文档编写：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
```

本文是 P6 qualitative / paper figure 准备阶段的执行设计，不覆盖历史文档。后续 P6 实现和验收以本文为阶段 contract。

## 上游依据

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260603-184214-forecasting-p1-p6-complete-design.md
docs/ai/context/20260603-232353-forecasting-p5-main-table-result.md
docs/ai/context/20260604-085116-forecasting-p5-ablation-result.md
```

## 当前实验事实

P5.2 主表已经通过 P6 入口门槛：

```text
relation long_mse mean = 0.0373418675
concat long_mse mean   = 0.0380507699
same-seed relation vs concat long_mse: 3/3

relation relative_root_distance_error mean = 0.4220982101
concat relative_root_distance_error mean   = 0.4891153050
same-seed relation vs concat relative_root_distance_error: 3/3
```

P5.3 消融支持容量控制和 relation feature 解释：

```text
relation long_mse mean = 0.0373418675
parameter-matched concat long_mse mean = 0.0378088307

relation relative_root_distance_error mean = 0.4220982101
parameter-matched concat relative_root_distance_error mean = 0.4913293425
```

必须保留的边界：

```text
relation-aware 不优于 independent 的 future_mse / long_mse。
relation-aware 不优于 repeat 的 relation-style metrics。
relation encoder 不能声称改善所有 relation metrics。
all-features 相对 velocity-only 的 long_mse 胜场不足，不能夸大 feature ablation。
```

## P6 目标

P6 的目标是解释 P5 指标背后的行为：

```text
1. 展示 relation-aware 相对 concat no-relation 在部分样本上的长期误差和 root distance 行为改善。
2. 展示 relation-aware 与 concat 接近的样本，说明收益不是所有样本都明显。
3. 展示 relation-aware 失败或更差的样本，保留论文结论边界。
4. 产出可进入论文草稿的曲线图、样本级 metrics 和必要的 numpy 数据。
```

P6 不是新的主实验，也不能替代 P5 主指标。

## P6 不做的事

```text
不训练新模型。
不改 P2 metrics key。
不重新定义 pred_len=120 的主协议。
不启动 P5.4 observation-ratio。
不把成功可视化包装成 relation-aware 全面最优。
不接入 NTU120-AS / Chi3D-AS。
不修改 train/train_mdm.py、model/cmdm.py、diffusion/gaussian_diffusion.py、eval/eval_cmdm.py。
```

## 输入模型与协议

主 qualitative 使用 seed 0 的代表性 checkpoint。3-seed 结论仍只来自 P5 表格。

```text
dataset: InterHuman forecasting
split: test
window_len: 150
obs_len: 30
pred_len: 120
crop: center crop
metrics: P2 original-scale evaluator
normalizer: checkpoint 内记录的 train-only normalizer，加载后校验协议字段
```

输入 checkpoint：

```text
independent:
save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt

concat no-relation:
save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt

relation-aware:
save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt
```

repeat baseline 不需要 checkpoint：

```text
pred_repeat[:, t] = obs[:, -1]
```

parameter-matched concat 可作为补充输出，不作为 P6 必需项：

```text
save/forecasting/interhuman/p5_ablation_concat_h259_l2_s0_5000/model000005000.pt
```

## 新增实现范围

优先新增：

```text
sample/visualize_forecasting.py
```

允许最小改动：

```text
utils/forecasting_metrics.py
  新增 per-frame / per-sample 辅助函数时，必须让 aggregate metrics 仍复用同一基础计算。

eval/eval_forecasting.py
  不新增主逻辑；只在确有必要时复用已有 checkpoint loading helper。
```

不建议把 P6 主逻辑继续塞进 `eval/eval_forecasting.py`，因为 P6 需要样本选择、曲线图、npy 落盘和多模型对比，入口职责已经超过 evaluator。

## CLI contract

P6 主入口：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m sample.visualize_forecasting \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
  --independent_checkpoint save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt \
  --concat_checkpoint save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt \
  --relation_checkpoint save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt \
  --save_dir results/forecasting/interhuman/p6_qualitative_150_30_120 \
  --num_samples 8 \
  --batch_size 64 \
  --num_workers 0 \
  --seed 0
```

可选补充：

```bash
  --param_matched_concat_checkpoint save/forecasting/interhuman/p5_ablation_concat_h259_l2_s0_5000/model000005000.pt
```

实现时如果 checkpoint normalizer 路径缺失，应回退到 checkpoint 所在目录加载 `normalizer.pt`，并记录到输出 `run_config.json`。

## 样本选择 contract

先对 test split 全部 508 个可用样本做 deterministic center-crop 推理，得到每个方法的 sample-level metrics。

每个样本计算：

```text
delta_long = concat.long_mse - relation.long_mse
delta_root_dist = concat.relative_root_distance_error - relation.relative_root_distance_error
```

选择至少 8 个互不重复样本：

```text
2 个 success:
  delta_long > 0 且 delta_root_dist > 0
  按 delta_long 降序，delta_root_dist 降序打破平局

2 个 close:
  abs(delta_long) 最小
  不要求 relation 胜出

2 个 failure:
  delta_long < 0 或 delta_root_dist < 0
  优先选择 delta_long 最负的样本

2 个 length boundary:
  1 个 length 最接近 150 的短序列边界样本
  1 个 test split 中 length 最大的长序列样本
```

如果某一类样本不足：

```text
记录不足原因。
从剩余未选择样本中按 abs(delta_long) 和 abs(delta_root_dist) 多样性补齐。
不得手动只挑成功样本。
```

样本选择必须保存：

```text
selection.json
selection.csv
```

至少记录：

```text
sample_id
length
start
category
delta_long
delta_root_dist
各方法 sample-level metrics
selection_reason
```

## 输出 contract

总输出目录：

```text
results/forecasting/interhuman/p6_qualitative_150_30_120/
```

根目录保存：

```text
run_config.json
selection.json
selection.csv
metrics_per_sample_all.json
metrics_per_sample_all.csv
summary.md
```

每个样本目录：

```text
qualitative/{sample_id}/meta.json
qualitative/{sample_id}/obs.npy
qualitative/{sample_id}/gt.npy
qualitative/{sample_id}/pred_repeat.npy
qualitative/{sample_id}/pred_independent.npy
qualitative/{sample_id}/pred_concat.npy
qualitative/{sample_id}/pred_relation.npy
qualitative/{sample_id}/metrics_per_sample.json
qualitative/{sample_id}/distance_curve.png
qualitative/{sample_id}/orientation_curve.png
qualitative/{sample_id}/long_mse_curve.png
```

可选输出：

```text
qualitative/{sample_id}/obs_h5_like.npy
qualitative/{sample_id}/gt_h5_like.npy
qualitative/{sample_id}/pred_relation_h5_like.npy
qualitative/{sample_id}/pred_concat_h259.npy
qualitative/{sample_id}/root_trajectory_xy.png
rendered frames or videos
```

## 曲线定义

所有曲线使用 original scale 数据。

`distance_curve.png`：

```text
x: frame index 0..149
y: root distance ||trans_A - trans_B||
lines: obs, gt future, repeat, independent, concat, relation
vertical line: obs_len=30
```

`orientation_curve.png`：

```text
x: future frame index 30..149
y: relative orientation error to ground truth
lines: repeat, independent, concat, relation
```

`long_mse_curve.png`：

```text
x: future frame index 30..149
y: per-frame active-vector MSE to ground truth
highlight: long horizon segment frames 110..149 in global frame index
```

曲线计算必须复用或派生自 `utils/forecasting_metrics.py` 的同一基础函数，避免图表口径和 P2 metrics 口径分裂。

## 数值与一致性检查

必须检查：

```text
obs / gt / pred shape 分别为 [30,2,147] / [120,2,147] / [120,2,147]。
所有数组为 finite。
所有 checkpoint 的 window_len / obs_len / pred_len / person_dim 与命令行一致。
所有方法使用同一个 test sample、同一个 center crop start。
sample-level pred == target sanity 在开发阶段可返回 0 指标。
summary 中 aggregate sample metrics 不冒充 P5 全量 test 指标。
```

## 验收标准

P6 完成必须满足：

```text
至少 8 个 test samples 输出完整。
success / close / failure / boundary 四类样本均有记录。
所有 pred 数值有限。
obs / gt / pred 时间轴对齐。
distance / orientation / long_mse 曲线可生成。
metrics_per_sample.json 与曲线基础计算一致。
失败样本被保留并写入 summary。
render 卡住时仍保留 npy + curves，不阻塞 P6。
```

完成后新建阶段结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-p6-qualitative-result.md
```

记录：

```text
实现文件
验收命令
输出路径
样本列表
成功案例
失败案例
是否偏离本文
是否支持 P5 结论
```

## 论文写法边界

可以写：

```text
Qualitative examples show that the relation-aware model can better preserve relative root distance in selected success cases.
Failure cases show that explicit relation modeling is not uniformly better than all baselines.
```

不能写：

```text
relation-aware 全面优于 independent。
relation-aware 在所有 relation metrics 上全面最优。
P6 可视化证明了模型整体最优。
只展示 success cases。
```

## 下一步

按本文实现 `sample/visualize_forecasting.py`，优先完成 npy、per-sample metrics 和三类曲线。render 只作为可选后续，不阻塞 P6 验收。
