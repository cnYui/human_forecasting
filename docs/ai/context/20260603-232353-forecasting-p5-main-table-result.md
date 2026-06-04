# Forecasting P5.2 Main Table 3 Seeds 结果记录

## 文档定位

本文记录 P5.2 主表 3-seed 实验结果，依据正式设计：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260603-203757-forecasting-p5-plan.md
```

P5.2 只判断主表是否支持继续进入 P5.3 消融，不宣称完整 P5 完成。

## 协议

```text
dataset: InterHuman forecasting
window_len: 150
obs_len: 30
pred_len: 120
split: test
seeds: 0, 1, 2
metrics: P2 original-scale evaluator
```

主表方法：

```text
repeat: zero-velocity repeat baseline
independent: per-person independent GRU predictor
concat: concat no-relation GRU predictor
relation: all relation features + GRU relation encoder
```

## 新增完成项

本轮补齐 relation seed2：

```text
save_dir: save/forecasting/interhuman/p5_main_relation_h256_r128_l2_s2_5000
checkpoint: save/forecasting/interhuman/p5_main_relation_h256_r128_l2_s2_5000/model000005000.pt
metrics: save/forecasting/interhuman/p5_main_relation_h256_r128_l2_s2_5000/metrics_test.json
num_params: 10058704
```

test 指标：

```text
future_mse: 0.03208013012359931
long_mse: 0.03760197430145083
relative_root_distance_error: 0.441282676899527
relative_orientation_error: 0.7479855300873284
inter_person_distance_consistency: 0.016602360445448733
```

## 汇总输出

manifest 与 aggregate 输出：

```text
results/forecasting/interhuman/p5_main_150_30_120/manifest.json
results/forecasting/interhuman/p5_main_150_30_120/manifest.resolved.json
results/forecasting/interhuman/p5_main_150_30_120/summary.json
results/forecasting/interhuman/p5_main_150_30_120/summary.csv
results/forecasting/interhuman/p5_main_150_30_120/summary.md
```

aggregate 命令：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode aggregate \
  --manifest results/forecasting/interhuman/p5_main_150_30_120/manifest.json \
  --save_dir results/forecasting/interhuman/p5_main_150_30_120
```

## Main Table Mean / Std

关键指标如下，数值越低越好。

| method | params | future_mse | long_mse | rel_root_dist | rel_orient | ipd_consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repeat | 0 | 0.0368928675 +/- 0.0000000000 | 0.0511287494 +/- 0.0000000000 | 0.2552213891 +/- 0.0000000000 | 0.5552304636 +/- 0.0000000000 | 0.0060419599 +/- 0.0000000000 |
| independent | 5305064 | 0.0287863306 +/- 0.0000504808 | 0.0362148034 +/- 0.0000822461 | 0.4211151787 +/- 0.0191361594 | 0.6603080315 +/- 0.0061870767 | 0.0135530162 +/- 0.0002433916 |
| concat | 9951440 | 0.0320573101 +/- 0.0001377769 | 0.0380507699 +/- 0.0001789489 | 0.4891153050 +/- 0.0072489027 | 0.7315031939 +/- 0.0078048409 | 0.0159595018 +/- 0.0003189634 |
| relation | 10058704 | 0.0317788706 +/- 0.0003197685 | 0.0373418675 +/- 0.0003362215 | 0.4220982101 +/- 0.0169976462 | 0.7339871620 +/- 0.0167900450 | 0.0163852204 +/- 0.0002321092 |

## Same-Seed Relation vs Concat

| metric | seed0 | seed1 | seed2 | wins |
| --- | --- | --- | --- | ---: |
| future_mse | 0.031443351850 < 0.031901971752 | 0.031813129797 < 0.032105230587 | 0.032080130124 < 0.032164727873 | 3/3 |
| long_mse | 0.036962207905 < 0.037895697382 | 0.037461420345 < 0.038246573176 | 0.037601974301 < 0.038010039271 | 3/3 |
| relative_root_distance_error | 0.408916108956 < 0.494654645131 | 0.416095844404 < 0.491780096621 | 0.441282676900 < 0.480911173220 | 3/3 |
| relative_orientation_error | 0.738604340497 > 0.730146685454 | 0.715371615305 < 0.724465525995 | 0.747985530087 > 0.739897370338 | 1/3 |
| inter_person_distance_consistency | 0.016412710522 > 0.015592024288 | 0.016140590312 < 0.016164638190 | 0.016602360445 > 0.016121842818 | 1/3 |

## Gate 判断

P5.2 主表 gate 通过：

```text
relation long_mse mean < concat long_mse mean:
0.0373418675 < 0.0380507699，满足。

relation long_mse same-seed wins >= 2/3:
3/3，满足。

至少一个 relation metric mean 优于 concat:
relative_root_distance_error 0.4220982101 < 0.4891153050，满足。

同一个 relation metric same-seed wins >= 2/3:
relative_root_distance_error 3/3，满足。
```

允许进入 P5.3 消融表。

## 必须保留的边界

P5.2 不能写成 relation-aware 全面最优：

```text
independent future_mse mean: 0.0287863306
relation future_mse mean:    0.0317788706

independent long_mse mean:   0.0362148034
relation long_mse mean:      0.0373418675
```

relation 只稳定优于 concat no-relation，不优于 independent。repeat 在 relation-style metrics 上仍最低，因此这些指标当前更像几何稳定性/交互保持诊断，不等价于 forecasting accuracy 最优性证明。

P5.3 目标应限定为验证：

```text
relation encoder 和 relation feature 是否解释了 relation 相对 concat 的 long_mse / relative_root_distance_error 改善。
```

在 P5.3 完成前，不进入 P6 success showcase，不写最终成功结论。
