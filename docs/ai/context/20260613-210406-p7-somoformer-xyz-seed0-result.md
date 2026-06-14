# P7.1 SoMoFormer XYZ seed0 正式训练结果

## 目标

运行 P7.1 joint-space SoMoFormer baseline 的 seed0 5000-step 正式训练。

该 run 用 ReGenNet InterHuman active-vector dataset，经 `active -> SMPL xyz` adapter 转为 joint-space，再训练 SoMoFormer-style DCT token Transformer。

## 环境

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python ...
torch: 1.7.1
cuda: true
torch-dct: installed, but current model still uses pure torch DCT matrix
```

## 命令

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting_xyz \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/p7_somoformer_xyz_h256_l4_dct30_s0_5000 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --num_steps 5000 \
  --save_interval 1000 \
  --eval_interval 500 \
  --hidden_dim 256 \
  --num_heads 8 \
  --num_layers 4 \
  --dim_feedforward 1024 \
  --dct_n 30 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --seed 0
```

## 配置

```text
model_type: somoformer_xyz
num_params: 3,182,110
obs_len: 30
pred_len: 120
window_len: 150
batch_size: 16
effective_batch_size: 16
dct_n: 30
hidden_dim: 256
num_heads: 8
num_layers: 4
dim_feedforward: 1024
lr: 3e-4
weight_decay: 1e-4
seed: 0
```

## 输出目录

```text
save/forecasting/interhuman/p7_somoformer_xyz_h256_l4_dct30_s0_5000
```

关键文件：

```text
args.json
train_log.jsonl
metrics_val.json
metrics_test.json
model000005000.pt
opt000005000.pt
```

中间 checkpoint 已按 500 step eval / 1000 step save 间隔写出：

```text
model000000500.pt
model000001000.pt
model000001500.pt
model000002000.pt
model000002500.pt
model000003000.pt
model000003500.pt
model000004000.pt
model000004500.pt
model000005000.pt
```

由于 `eval_interval=500` 且 `save_due or eval_due` 会保存 checkpoint，实际每 500 step 都有 model/optimizer 文件。

## 最终验证集指标

checkpoint:

```text
save/forecasting/interhuman/p7_somoformer_xyz_h256_l4_dct30_s0_5000/model000005000.pt
```

val metrics:

```text
joint_mse: 0.06383317573803717
mpjpe: 0.30179988203850466
short_joint_mse: 0.012063470009747333
mid_joint_mse: 0.054748540299128644
long_joint_mse: 0.12468752014426003
root_translation_error: 0.25941473878590404
relative_root_distance_error: 0.22572190967281308
inter_person_distance_consistency_xyz: 0.006137380772064218
```

## 最终测试集指标

test samples:

```text
508
```

test metrics:

```text
joint_mse: 0.05966417283171744
mpjpe: 0.28972134486896783
short_joint_mse: 0.011117184036008016
mid_joint_mse: 0.051276665180921555
long_joint_mse: 0.11659867844478351
root_translation_error: 0.24839193849113045
relative_root_distance_error: 0.19827875354158597
inter_person_distance_consistency_xyz: 0.00558080468473472
```

## 观察

- 训练完成，最终 checkpoint 可用。
- 验证集 `joint_mse` 从 step500 的 `0.0763873657` 降到 step5000 的 `0.0638331757`。
- 验证集 `long_joint_mse` 从 step500 的 `0.1437492714` 降到 step5000 的 `0.1246875201`。
- 测试集 `mpjpe=0.2897213449`，这是 P7.1 joint-space 口径下的第一条正式 baseline。

## 边界

这是 joint-space baseline：

```text
pred_xyz [B,120,2,24,3]
```

不是 active-vector baseline：

```text
pred_active [B,120,2,147]
```

因此当前指标不能直接和 P5 active-vector 主表的 `future_mse / long_mse / rotation_mse` 数值横比。若要进入论文主表，需要继续 P7.2 `somoformer_active`，输出 `[B,120,2,147]` 并复用 P2/P5 evaluator。

## 下一步建议

1. 增加 xyz repeat baseline，得到 joint-space 下的最小参考线。
2. 可追加 independent/concat 的 xyz-space 对照，判断 SoMoFormer XYZ 是否真的优于简单模型。
3. 若 SoMoFormer XYZ 相对 xyz baselines 有明显优势，再进入 P7.2 `somoformer_active`。
