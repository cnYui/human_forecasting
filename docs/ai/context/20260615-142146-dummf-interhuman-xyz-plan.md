# P9.1 DuMMF-style InterHuman XYZ 计划

## 目标

在 ReGenNet 现有 InterHuman forecasting xyz 管线内新增 `dummf_interhuman_xyz`，用于双人未来动作预测：

```text
active-vector batch -> active_to_xyz -> DuMMF-style stochastic predictor -> xyz future
```

第一阶段只打通训练、checkpoint、确定性 eval smoke。随机多未来评估和视频渲染作为后续阶段，不在本次一次性铺开。

## 必须保留的边界

- 不直接修改 `/home/rpartx3080/CodeSpace/DuMMF` 外部仓库。
- 不把 InterHuman 硬转成 DuMMF 的 AMASS-CMU/SMPL-H/DMPL 数据协议。
- 不改 P7/P8 已有 `XYZ_METRIC_KEYS` 和 deterministic checkpoint evaluator 行为。
- `forward()` 继续返回 `[B,120,2,24,3]`，兼容现有 `eval/eval_forecasting_xyz.py`。
- 多未来输出通过 `sample()` / `forward_multi()` 提供 `[B,K,120,2,24,3]`。

## 模型设计

新增 `model/forecasting_dummf.py`：

- 输入 `obs_xyz [B,30,2,24,3]`。
- 每个人 local encoder：共享 GRU，编码单人 root-local 历史。
- 双人 global encoder：GRU 编码两人 root 关系和全身 flatten 历史。
- intent：
  - global intent embedding：每个未来样本共享给两个人，表示互动层面未来。
  - local intent embedding：每个人独立 intent，表示个体姿态变化。
- decoder：
  - global/root decoder 预测两个人未来 root delta。
  - local decoder 预测每个人未来 root-local joints delta。
  - 还原为 `future_root + future_local`。
- 输出：
  - `forward_multi(obs_xyz, num_samples=None)` 返回 K 条未来。
  - `forward(obs_xyz)` 返回第 0 条未来，保证现有 deterministic evaluator 可用。

## 训练损失

`training_loss(obs_xyz, target_xyz, ...)` 实现：

- global Best-of-K loss：K 条完整双人 xyz future 中选每个样本误差最小的一条。
- root Best-of-K loss：K 条双人 root trajectory 中选最小。
- local Best-of-K loss：root-local 姿态误差，按样本选最小。
- velocity loss：对选中的未来样本约束未来速度。
- diversity loss：对 K 条预测的 root trajectory 做 bounded pairwise penalty，鼓励非塌缩多样性。

默认权重保守：

```text
dummf_num_samples=5
dummf_global_loss_weight=1.0
dummf_root_loss_weight=1.0
dummf_local_loss_weight=1.0
dummf_velocity_loss_weight=0.2
dummf_diversity_weight=0.01
```

## 训练脚本改动

扩展 `train/train_forecasting_xyz.py`：

- parser 增加 DuMMF 相关参数。
- `_build_model()` 将参数传给 `create_xyz_forecasting_model()`。
- `_train_step()` 增加 `dummf_interhuman_xyz` 分支，显式传入 DuMMF loss 权重。

## factory 改动

扩展 `model/forecasting_xyz.py`：

- import `DuMMFInterHumanXYZ`。
- `XYZ_FORECASTING_MODEL_TYPES` 增加 `dummf_interhuman_xyz`。
- `create_xyz_forecasting_model()` 和 `create_xyz_forecasting_model_from_config()` 增加对应参数。

## 验收

必须通过：

```bash
python -m compileall model/forecasting_dummf.py model/forecasting_xyz.py train/train_forecasting_xyz.py

micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
  python train/train_forecasting_xyz.py \
  --model_type dummf_interhuman_xyz \
  --save_dir save/forecasting/interhuman/p9_dummf_xyz_smoke \
  --num_steps 2 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --max_samples 8 \
  --save_interval 2 \
  --eval_interval 0 \
  --overwrite

micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
  python eval/eval_forecasting_xyz.py \
  --mode checkpoint \
  --checkpoint save/forecasting/interhuman/p9_dummf_xyz_smoke/model000000002.pt \
  --save_dir save/forecasting/interhuman/p9_dummf_xyz_smoke/eval_test_max8 \
  --batch_size 2 \
  --max_samples 8
```

验收后新增结果文档，记录 checkpoint、参数量、smoke metrics 和边界。
