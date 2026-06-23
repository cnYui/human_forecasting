# ReGenNet 150/30/120 双人预测从零训练检查

## 用户目标

- 使用 ReGenNet 项目处理 InterHuman 双人动作预测。
- 前 30 帧作为条件，预测后 120 帧。
- 使用项目内已冻结的 InterHuman train/val/test split 训练和测试。

## 当前项目入口判断

原始 `train/train_mdm.py` / `model/cmdm.py` 不是该任务的直接训练入口。该路径会把一人的完整动作作为 `cmotion` 条件，训练另一人的完整动作生成，不能只靠命令行切到“双人前 30 帧 -> 双人后 120 帧”协议。

当前项目已经为该目标接好独立 forecasting 入口：

```text
train/train_forecasting_xyz.py
eval/eval_forecasting_xyz.py
data_loaders/forecasting/interhuman.py
model/forecasting_xyz.py
```

该入口使用 InterHuman H5，读取 `window_len=150`，切分为 `obs_len=30` 和 `pred_len=120`，并在 SMPL xyz 空间训练/评估：

```text
obs_active:    [B,30,2,147]
target_active: [B,120,2,147]
obs_xyz:       [B,30,2,24,3]
target_xyz:    [B,120,2,24,3]
```

## 数据检查

数据目录存在：

```text
dataset/interhuman/smpl/conditioned
```

文件：

```text
interhuman_train.h5
interhuman_val.h5
interhuman_test.h5
meta.json
```

H5 统计：

```text
train total=6021 usable>=150=2910
val   total=580  usable>=150=226
test  total=1175 usable>=150=508
```

`eval_forecasting_xyz --mode xyz_smoke --max_samples 2` 已通过，形状和 finite 检查正常。

## 推荐从零训练命令

如果目标是当前项目中最贴近“ReGenNet 双人 30->120 forecasting”的主线，优先跑 `somoformer_xyz`：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m train.train_forecasting_xyz \
  --data_path dataset/interhuman/smpl/conditioned \
  --save_dir save/forecasting/interhuman/fresh_somoformer_xyz_h256_l4_dct30_s0_5000 \
  --model_type somoformer_xyz \
  --window_len 150 \
  --obs_len 30 \
  --pred_len 120 \
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

训练结束会自动对 test split 做一次评估，并在 `save_dir` 下写出 checkpoint、`args.json`、`train_log.jsonl` 和 `metrics_test.json/yaml`。

## 评估命令

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m eval.eval_forecasting_xyz \
  --mode checkpoint \
  --data_path dataset/interhuman/smpl/conditioned \
  --split test \
  --checkpoint save/forecasting/interhuman/fresh_somoformer_xyz_h256_l4_dct30_s0_5000/model000005000.pt \
  --save_dir results/forecasting/interhuman/fresh_somoformer_xyz_h256_l4_dct30_s0_5000_test \
  --batch_size 32 \
  --num_workers 0
```

## 注意

- 当前会话中 `nvidia-smi` 无法连接驱动，`torch.cuda.is_available()` 为 `False`；如果在该环境直接训练，会走 CPU，非常慢。
- 在可见 GPU 的终端/容器中运行同样命令时，脚本会自动使用 CUDA。
- 如果显存不足，先改为 `--batch_size 8 --grad_accum_steps 2 --eval_batch_size 8`，保持有效 batch size 约为 16。
