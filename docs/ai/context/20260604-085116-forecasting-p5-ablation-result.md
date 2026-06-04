# Forecasting P5.3 Ablation Table 3 Seeds 结果记录

## 文档定位

本文记录 P5.3 消融表 3-seed 实验结果，依据：

```text
docs/ai/context/20260603-190003-forecasting-final-official-design.md
docs/ai/context/20260603-203757-forecasting-p5-plan.md
docs/ai/context/20260603-232353-forecasting-p5-main-table-result.md
```

P5.2 主表 gate 已通过，因此本轮进入 P5.3 完整消融。

## 协议

```text
dataset: InterHuman forecasting
window_len: 150
obs_len: 30
pred_len: 120
split: test
seeds: 0,1,2
num_steps: 5000
batch_size: 32
eval_batch_size: 64
lr: 1e-3
weight_decay: 1e-4
num_workers: 0
metrics: P2 original-scale evaluator
```

## 新增 run

本轮新增 15 个训练 run：

```text
save/forecasting/interhuman/p5_ablation_concat_h259_l2_s0_5000
save/forecasting/interhuman/p5_ablation_concat_h259_l2_s1_5000
save/forecasting/interhuman/p5_ablation_concat_h259_l2_s2_5000

save/forecasting/interhuman/p5_ablation_relation_noenc_all_s0_5000
save/forecasting/interhuman/p5_ablation_relation_noenc_all_s1_5000
save/forecasting/interhuman/p5_ablation_relation_noenc_all_s2_5000

save/forecasting/interhuman/p5_ablation_relation_translation_gru_s0_5000
save/forecasting/interhuman/p5_ablation_relation_translation_gru_s1_5000
save/forecasting/interhuman/p5_ablation_relation_translation_gru_s2_5000

save/forecasting/interhuman/p5_ablation_relation_velocity_gru_s0_5000
save/forecasting/interhuman/p5_ablation_relation_velocity_gru_s1_5000
save/forecasting/interhuman/p5_ablation_relation_velocity_gru_s2_5000

save/forecasting/interhuman/p5_ablation_relation_orientation_gru_s0_5000
save/forecasting/interhuman/p5_ablation_relation_orientation_gru_s1_5000
save/forecasting/interhuman/p5_ablation_relation_orientation_gru_s2_5000
```

复用 P5.2 主表 run：

```text
concat no-relation seed0: save/forecasting/interhuman/p3_concat_h256_l2_s0_5000
concat no-relation seed1: save/forecasting/interhuman/p5_main_concat_h256_l2_s1_5000
concat no-relation seed2: save/forecasting/interhuman/p5_main_concat_h256_l2_s2_5000

all features + GRU seed0: save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000
all features + GRU seed1: save/forecasting/interhuman/p5_main_relation_h256_r128_l2_s1_5000
all features + GRU seed2: save/forecasting/interhuman/p5_main_relation_h256_r128_l2_s2_5000
```

## 汇总输出

```text
results/forecasting/interhuman/p5_ablation_150_30_120/manifest.json
results/forecasting/interhuman/p5_ablation_150_30_120/manifest.resolved.json
results/forecasting/interhuman/p5_ablation_150_30_120/summary.json
results/forecasting/interhuman/p5_ablation_150_30_120/summary.csv
results/forecasting/interhuman/p5_ablation_150_30_120/summary.md
```

注意：

```text
manifest 有 24 个 entries、8 个 aggregate rows。
relation with encoder 和 all relation features 是同一个 full relation run 的两种消融视角，因此在 manifest 中分别引用同一组 P5.2 full relation run。
```

## Ablation Mean / Std

关键指标如下，数值越低越好。

| row | params | future_mse | long_mse | rel_root_dist | rel_orient | ipd_consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| concat h256 no-relation | 9951440 | 0.0320573101 +/- 0.0001377769 | 0.0380507699 +/- 0.0001789489 | 0.4891153050 +/- 0.0072489027 | 0.7315031939 +/- 0.0078048409 | 0.0159595018 +/- 0.0003189634 |
| concat h259 param-matched | 10075415 | 0.0319424076 +/- 0.0000309390 | 0.0378088307 +/- 0.0000678196 | 0.4913293425 +/- 0.0137849415 | 0.7249843689 +/- 0.0060263090 | 0.0158116104 +/- 0.0003099089 |
| relation no encoder all | 10004816 | 0.0320496025 +/- 0.0001077032 | 0.0375836698 +/- 0.0002170996 | 0.4320849805 +/- 0.0026011427 | 0.7278118622 +/- 0.0090623947 | 0.0157649532 +/- 0.0002723236 |
| relation with encoder all | 10058704 | 0.0317788706 +/- 0.0003197685 | 0.0373418675 +/- 0.0003362215 | 0.4220982101 +/- 0.0169976462 | 0.7339871620 +/- 0.0167900450 | 0.0163852204 +/- 0.0002321092 |
| translation only + GRU | 10053712 | 0.0320228866 +/- 0.0002676808 | 0.0376835243 +/- 0.0003360703 | 0.4272217504 +/- 0.0059754293 | 0.7359812581 +/- 0.0157382746 | 0.0164513564 +/- 0.0003783646 |
| velocity only + GRU | 10053712 | 0.0316830241 +/- 0.0002410890 | 0.0373630645 +/- 0.0002973908 | 0.4600573967 +/- 0.0065729065 | 0.7244010452 +/- 0.0134657728 | 0.0172088795 +/- 0.0002645331 |
| orientation only + GRU | 10056016 | 0.0319873445 +/- 0.0001781834 | 0.0374530703 +/- 0.0002845948 | 0.4801889247 +/- 0.0108933745 | 0.7238655465 +/- 0.0094822987 | 0.0170287469 +/- 0.0004352030 |
| all features + GRU | 10058704 | 0.0317788706 +/- 0.0003197685 | 0.0373418675 +/- 0.0003362215 | 0.4220982101 +/- 0.0169976462 | 0.7339871620 +/- 0.0167900450 | 0.0163852204 +/- 0.0002321092 |

## Gate 判断

### Parameter-Matched Concat

通过。

```text
all features + GRU long_mse mean:
0.0373418675 < 0.0378088307
same-seed: 3/3

all features + GRU relative_root_distance_error mean:
0.4220982101 < 0.4913293425
same-seed: 3/3
```

参数量对照没有解释 relation-aware 相对 concat 的全部收益。

### Relation Encoder

通过，但幅度小。

```text
with encoder long_mse mean:
0.0373418675 < 0.0375836698
same-seed: 2/3

with encoder relative_root_distance_error mean:
0.4220982101 < 0.4320849805
same-seed: 2/3
```

不支持的指标：

```text
relative_orientation_error: with encoder 更差，0.7339871620 > 0.7278118622
inter_person_distance_consistency: with encoder 更差，0.0163852204 > 0.0157649532
```

因此只能写 relation encoder 对 long-horizon MSE 和 relative root distance 有帮助，不能写对所有 relation metrics 都有帮助。

### Relation Features

通过。

`all features + GRU` 的 `long_mse` mean 优于三个 single-feature variants：

```text
translation: 0.0373418675 < 0.0376835243
velocity:    0.0373418675 < 0.0373630645
orientation: 0.0373418675 < 0.0374530703
```

same-seed 胜场：

```text
vs translation: 2/3
vs velocity:    1/3
vs orientation: 2/3
```

`all features + GRU` 的 `relative_root_distance_error` mean 优于三个 single-feature variants：

```text
translation: 0.4220982101 < 0.4272217504
velocity:    0.4220982101 < 0.4600573967
orientation: 0.4220982101 < 0.4801889247
```

same-seed 胜场：

```text
vs translation: 2/3
vs velocity:    3/3
vs orientation: 3/3
```

feature contribution 的最强证据来自 `relative_root_distance_error`，不是 `relative_orientation_error`。

## 结论边界

P5.3 支持进入 P6 qualitative / paper figure 准备，但论文主张必须收窄：

```text
可以写：
relation-aware full model 稳定优于 concat no-relation 和 parameter-matched concat；
relation encoder 对 long_mse 和 relative root distance 有小幅贡献；
all relation features 相对 single-feature variants 在 long_mse mean 和 relative root distance 上更稳。

不能写：
relation-aware 全面优于 independent；
relation encoder 改善所有 relation metrics；
all features 在每个 seed 上都优于 velocity-only；
relative_orientation_error 或 inter_person_distance_consistency 是当前 full model 的优势指标。
```

P5.4 observation ratio 仍未启动；若要做，必须先扩展动态 pred_len metrics contract。P5.4 是补充表，不阻塞当前 20% 主协议进入 P6。

## 验证

已通过：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall \
  utils/forecasting_motion.py model/forecasting.py train/train_forecasting.py eval/eval_forecasting.py
```

aggregate 已通过：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting \
  --mode aggregate \
  --manifest results/forecasting/interhuman/p5_ablation_150_30_120/manifest.json \
  --save_dir results/forecasting/interhuman/p5_ablation_150_30_120
```
