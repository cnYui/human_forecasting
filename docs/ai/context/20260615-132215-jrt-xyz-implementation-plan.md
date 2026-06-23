# JRT XYZ 实现计划

## 目标

按 `docs/ai/context/20260614-185518-jrt-on-regennet-dataset-adaptation-design.md` 实现 ReGenNet 内部的 `jrt_xyz` baseline，用本地 InterHuman SMPL H5 数据跑 JRT-style joint-relation forecasting。

## 范围

本阶段只做：

- 新增 `model/forecasting_jrt.py`。
- 在 `model/forecasting_xyz.py` 接入 `model_type=jrt_xyz`。
- 扩展 `train/train_forecasting_xyz.py` 的 CLI，使 relation loss 权重可配置。
- 复用现有 `InterHumanForecastDataset`、`active_to_xyz`、`eval/eval_forecasting_xyz.py`。
- 跑 `max_samples` smoke、2-step train、checkpoint eval。

本阶段不做：

- 不修改 `/home/rpartx3080/CodeSpace/JRTransformer` 官方仓库。
- 不支持 Chi3D / NTU120。
- 不改 P7/P8 xyz metrics key。
- 不启动 5000-step 正式训练，除非 smoke 全部通过后用户再要求。

## 实现要点

`jrt_xyz` 输入输出：

```text
obs_xyz    [B,30,2,24,3]
target_xyz [B,120,2,24,3]
pred_xyz   [B,120,2,24,3]
```

relation 输入：

```text
relation [B,48,48,obs_len+2]
  - exp(-distance history)
  - SMPL skeleton adjacency
  - same-person connectivity
```

训练损失：

```text
pose_mse + jrt_relation_weight * future_distance_mse
```

为了兼容现有训练器，模型实现：

```python
training_loss(obs_xyz, target_xyz, aux_weight=..., relation_weight=...)
```

## 验收命令

1. compileall：

```bash
python -m compileall model/forecasting_jrt.py model/forecasting_xyz.py train/train_forecasting_xyz.py
```

2. xyz smoke：

```bash
python eval/eval_forecasting_xyz.py --mode xyz_smoke --max_samples 2 ...
```

3. 2-step training：

```bash
python train/train_forecasting_xyz.py --model_type jrt_xyz --num_steps 2 --max_samples 4 ...
```

4. checkpoint eval：

```bash
python eval/eval_forecasting_xyz.py --mode checkpoint --checkpoint ... --max_samples 4 ...
```
