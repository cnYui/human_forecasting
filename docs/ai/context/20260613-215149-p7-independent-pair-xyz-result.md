# P7 independent_pair_xyz baseline 实现与结果

## 目标

按用户确认的定义实现并运行 `independent_pair_xyz` baseline：

```text
输入仍是双人样本 obs_xyz [B,30,2,24,3]
模型内部分人独立预测，不允许跨人信息流
输出拼回 pred_xyz [B,120,2,24,3]
```

该 baseline 用于公平对比 P7.1 SoMoFormer XYZ。

## 新增 / 修改

### 新增

```text
model/forecasting_xyz.py
```

新增内容：

- `XYZ_FORECASTING_MODEL_TYPES`
- `IndependentPairXYZModel`
- `create_xyz_forecasting_model`
- `create_xyz_forecasting_model_from_config`

`IndependentPairXYZModel` 结构：

```text
obs_xyz [B,30,2,24,3]
-> reshape [B*2,30,72]
-> shared GRU encoder
-> MLP decoder
-> [B*2,120,72]
-> reshape [B,120,2,24,3]
```

严格限制：

- 每个人只看自己的历史；
- 不使用 relation feature；
- 不融合两个人 hidden；
- 不做 cross-person attention；
- 两人共享预测器权重。

### 修改

```text
train/train_forecasting_xyz.py
eval/eval_forecasting_xyz.py
```

改动：

- 新增 `--model_type independent_pair_xyz|somoformer_xyz`。
- checkpoint 保存真实 `model_type`。
- eval checkpoint 通过 `model/forecasting_xyz.py` factory 加载。
- 保持旧 `somoformer_xyz` checkpoint 兼容。

## 验证

### compile / forward

通过：

```text
compileall model/forecasting_xyz.py model/forecasting_somoformer.py train/train_forecasting_xyz.py eval/eval_forecasting_xyz.py
```

forward smoke：

```text
independent_pair_xyz: [2,30,2,24,3] -> [2,120,2,24,3], finite=true
somoformer_xyz: [2,30,2,24,3] -> [2,120,2,24,3], finite=true
```

### 旧 SoMoFormer XYZ checkpoint 兼容

用 `eval.eval_forecasting_xyz --mode checkpoint` 加载：

```text
save/forecasting/interhuman/p7_somoformer_xyz_h256_l4_dct30_s0_5000/model000005000.pt
```

结果：通过。

### independent_pair 2-step smoke

命令：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting_xyz \
  --model_type independent_pair_xyz \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p7_independent_pair_xyz_2step_smoke \
  --overwrite \
  --batch_size 2 \
  --eval_batch_size 2 \
  --max_samples 2 \
  --num_steps 2 \
  --save_interval 2 \
  --eval_interval 2 \
  --log_interval 1 \
  --hidden_dim 64 \
  --num_layers 1 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --seed 0
```

结果：通过，checkpoint / val eval / test eval 均完成。

### active-vector 主路径回归

命令：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting --mode metrics_sanity \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --save_dir save/forecasting/interhuman/p7_independent_pair_regression_metrics_sanity \
  --batch_size 2 \
  --max_samples 2 \
  --num_workers 0
```

结果：通过，原 active-vector metrics sanity 全 0。

## 正式训练

命令：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting_xyz \
  --model_type independent_pair_xyz \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p7_independent_pair_xyz_h256_l2_s0_5000 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --num_steps 5000 \
  --save_interval 1000 \
  --eval_interval 500 \
  --hidden_dim 256 \
  --num_layers 2 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --seed 0
```

配置：

```text
model_type: independent_pair_xyz
num_params: 2,934,464
batch_size: 32
effective_batch_size: 32
hidden_dim: 256
num_layers: 2
lr: 1e-3
seed: 0
num_steps: 5000
```

最终 checkpoint：

```text
save/forecasting/interhuman/p7_independent_pair_xyz_h256_l2_s0_5000/model000005000.pt
```

## Final test metrics

test samples:

```text
508
```

```text
joint_mse: 0.06980970578165505
mpjpe: 0.3428108572490572
short_joint_mse: 0.019286090273439416
mid_joint_mse: 0.05957724065996531
long_joint_mse: 0.1305657924394908
root_translation_error: 0.3018627288773304
relative_root_distance_error: 0.22909507918076252
inter_person_distance_consistency_xyz: 0.02346360624775173
```

## P7 xyz seed0 主对比

输出：

```text
results/forecasting/interhuman/p7_xyz_main_seed0/summary.json
results/forecasting/interhuman/p7_xyz_main_seed0/summary.csv
results/forecasting/interhuman/p7_xyz_main_seed0/summary.md
```

| run | joint_mse | mpjpe | long_joint_mse | relative_root_distance_error | consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat_xyz | 0.1035212231 | 0.3581020008 | 0.1933701291 | 0.2552214026 | 0.0060419603 |
| independent_pair_xyz | 0.0698097058 | 0.3428108572 | 0.1305657924 | 0.2290950792 | 0.0234636062 |
| somoformer_xyz | 0.0596641728 | 0.2897213449 | 0.1165986784 | 0.1982787535 | 0.0055808047 |

## 判断

SoMoFormer XYZ 优于 independent_pair_xyz：

```text
joint_mse: 0.0698097058 -> 0.0596641728，约 14.53% 降低
mpjpe: 0.3428108572 -> 0.2897213449，约 15.49% 降低
long_joint_mse: 0.1305657924 -> 0.1165986784，约 10.70% 降低
relative_root_distance_error: 0.2290950792 -> 0.1982787535，约 13.45% 降低
```

这说明在当前 joint-space 口径下，允许跨 person/joint token attention 的 SoMoFormer XYZ 比“同样用双人样本、但两人分开独立预测”的 baseline 更好。

## 边界

- 这是 seed0 单次结果，还不是 3-seed 结论。
- 这是 joint-space 口径，不是 active-vector P5 主表。
- `independent_pair_xyz` 的 consistency 指标明显差于 repeat 和 SoMoFormer，说明独立分人预测可能破坏两人相对距离变化的一致性。
- 如果要写成强结论，需要后续补 seeds=`1,2` 或进入 P7 xyz main 3-seed。
