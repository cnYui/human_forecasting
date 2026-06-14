# P7.1 joint-space SoMoFormer baseline 实现结果

## 环境确认

本项目实际训练环境不是 conda，而是 micromamba：

```text
/home/rpartx3080/.local/micromamba/envs/regennet
```

使用方式：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python ...
```

环境检查结果：

```text
python: /home/rpartx3080/.local/micromamba/envs/regennet/bin/python
torch: 1.7.1
cuda: true
cuda_version: 11.0
smplx: 可 import
h5py: 3.7.0
torch_dct: 未安装
SMPL: 可加载，num_betas=10
```

实现中未引入 `torch_dct` 依赖，改用纯 torch DCT/IDCT 矩阵和 `einsum`。

## 新增实现

### `utils/forecasting_xyz.py`

新增：

- `active_to_smpl_motion`
- `active_to_xyz`
- `batch_active_to_xyz`
- `compute_xyz_metrics`

功能：

```text
active [B,T,2,147]
-> restore_active_motion
-> Rotation2xyz / SMPL
-> xyz [B,T,2,24,3]
```

xyz metrics：

- `joint_mse`
- `mpjpe`
- `short_joint_mse`
- `mid_joint_mse`
- `long_joint_mse`
- `root_translation_error`
- `relative_root_distance_error`
- `inter_person_distance_consistency_xyz`

### `model/forecasting_somoformer.py`

新增 `JointSpaceSoMoFormer`：

```text
input:  obs_xyz [B,30,2,24,3]
output: pred_xyz [B,120,2,24,3]
```

结构：

```text
last-pose padding 到 150 帧
-> DCT over time
-> person / joint / coord embeddings
-> TransformerEncoder
-> DCT coeff decoder
-> residual on DCT coeffs
-> IDCT
-> future xyz
```

### `train/train_forecasting_xyz.py`

新增 joint-space 训练入口：

```text
InterHumanForecastDataset
-> active obs/target
-> active_to_xyz
-> JointSpaceSoMoFormer
-> MSE(pred_xyz, target_xyz)
```

支持 checkpoint 保存、resume、val/test checkpoint eval。

### `eval/eval_forecasting_xyz.py`

新增 joint-space 评估入口：

```text
--mode xyz_smoke
--mode metrics_sanity
--mode checkpoint
```

### `model/rotation2xyz.py`

移除一行旧调试 `print`，避免 P7.1 active->xyz 转换时污染训练日志。

## 验证结果

### compileall

命令：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m compileall utils/forecasting_xyz.py model/forecasting_somoformer.py train/train_forecasting_xyz.py eval/eval_forecasting_xyz.py model/rotation2xyz.py
```

结果：通过。

### xyz smoke

命令：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting_xyz --mode xyz_smoke --data_path dataset/interhuman/smpl/conditioned --save_dir save/forecasting/interhuman/p7_xyz_smoke --batch_size 2 --max_samples 2 --num_workers 0
```

结果：

```text
train/val/test batch:
obs_active: [2,30,2,147]
target_active: [2,120,2,147]
obs_xyz: [2,30,2,24,3]
target_xyz: [2,120,2,24,3]
finite: true
```

### xyz metrics sanity

命令：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting_xyz --mode metrics_sanity --data_path dataset/interhuman/smpl/conditioned --split test --save_dir save/forecasting/interhuman/p7_xyz_metrics_sanity --batch_size 2 --max_samples 2 --num_workers 0
```

结果：`pred == target` 时所有 xyz metrics 均为 `0.0`。

### model forward smoke

结果：

```text
input: [2,30,2,24,3]
output: [2,120,2,24,3]
finite: true
```

### 2-step training smoke

命令：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting_xyz --data_path dataset/interhuman/smpl/conditioned --save_dir save/forecasting/interhuman/p7_xyz_2step_smoke --overwrite --batch_size 2 --eval_batch_size 2 --max_samples 2 --num_steps 2 --save_interval 2 --eval_interval 2 --log_interval 1 --hidden_dim 64 --num_heads 4 --num_layers 1 --dim_feedforward 128 --dct_n 20 --num_workers 0 --seed 0
```

结果：

```text
model_type=somoformer_xyz
params=37972
device=cuda
step[1]: train_loss=0.148617
step[2]: train_loss=0.155437
checkpoint: save/forecasting/interhuman/p7_xyz_2step_smoke/model000000002.pt
```

test metrics：

```text
joint_mse=0.0890086517
mpjpe=0.4276325703
long_joint_mse=0.1577119380
root_translation_error=0.4446732402
relative_root_distance_error=0.2639130652
inter_person_distance_consistency_xyz=0.0405846909
```

### checkpoint eval smoke

命令：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting_xyz --mode checkpoint --data_path dataset/interhuman/smpl/conditioned --split test --checkpoint save/forecasting/interhuman/p7_xyz_2step_smoke/model000000002.pt --save_dir save/forecasting/interhuman/p7_xyz_2step_smoke_eval --batch_size 2 --max_samples 2 --num_workers 0
```

结果：通过，指标与 2-step training smoke 的 test eval 一致。

### active-vector 主路径回归

命令：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting --mode metrics_sanity --data_path dataset/interhuman/smpl/conditioned --split test --save_dir save/forecasting/interhuman/p7_regression_metrics_sanity --batch_size 2 --max_samples 2 --num_workers 0
```

结果：通过，原 active-vector metrics sanity 全为 `0.0`。

## 后续建议

下一步可以启动 P7.1 seed0 正式 baseline：

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

如果训练速度过慢，再进入 P7.1.1：预缓存 `active -> xyz` 到 H5。
