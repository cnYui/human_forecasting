# P7 XYZ 主对比 3-seed 与显存监控结果

## 背景

用户要求补跑随机种子 1 和 2，并监控显存消耗。当前实验延续 P7 joint-space 口径，不改变 P1-P6 active-vector 主路径。

本次补齐的 run：

- `independent_pair_xyz` seed 1/2
- `somoformer_xyz` seed 1/2

seed 0 复用此前已完成的正式 run。

## 运行环境

- Python 环境：`micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python`
- 数据：`dataset/interhuman/smpl/conditioned`
- 协议：InterHuman fixed window `150`，obs `30`，pred `120`
- 口径：SMPL xyz joint-space `[B,T,2,24,3]`
- 显存监控：手动轮询 `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total`

## 输出位置

- 汇总目录：`results/forecasting/interhuman/p7_xyz_main_3seed/`
- 汇总文件：
  - `summary.json`
  - `summary.csv`
  - `summary.md`
  - `gpu_monitor.csv`
  - `gpu_monitor_summary.json`

## 单 seed checkpoint

`independent_pair_xyz`：

- seed0：`save/forecasting/interhuman/p7_independent_pair_xyz_h256_l2_s0_5000/model000005000.pt`
- seed1：`save/forecasting/interhuman/p7_independent_pair_xyz_h256_l2_s1_5000/model000005000.pt`
- seed2：`save/forecasting/interhuman/p7_independent_pair_xyz_h256_l2_s2_5000/model000005000.pt`

`somoformer_xyz`：

- seed0：`save/forecasting/interhuman/p7_somoformer_xyz_h256_l4_dct30_s0_5000/model000005000.pt`
- seed1：`save/forecasting/interhuman/p7_somoformer_xyz_h256_l4_dct30_s1_5000/model000005000.pt`
- seed2：`save/forecasting/interhuman/p7_somoformer_xyz_h256_l4_dct30_s2_5000/model000005000.pt`

## 3-seed test 结果

| model | joint_mse | mpjpe | long_joint_mse | relative_root_distance_error | inter_person_distance_consistency_xyz |
|---|---:|---:|---:|---:|---:|
| independent_pair_xyz | 0.0682312447 +- 0.0017461769 | 0.3375963565 +- 0.0080529239 | 0.1284009837 +- 0.0019345306 | 0.2280735768 +- 0.0053159672 | 0.0233993507 +- 0.0000829443 |
| somoformer_xyz | 0.0607606915 +- 0.0010012505 | 0.2947844890 +- 0.0055710050 | 0.1186761773 +- 0.0018992134 | 0.2032881472 +- 0.0116076032 | 0.0056123107 +- 0.0001299308 |

SoMoFormer XYZ 相对 `independent_pair_xyz` 的 mean 改善：

- `joint_mse` 降低约 `10.95%`
- `mpjpe` 降低约 `12.68%`
- `long_joint_mse` 降低约 `7.57%`
- `relative_root_distance_error` 降低约 `10.87%`
- `inter_person_distance_consistency_xyz` 降低约 `76.02%`

## 显存监控

- 采样数：`26`
- 采样峰值显存：`9660 MiB / 10240 MiB`
- 平均采样显存：`8957.88 MiB`
- 非空闲采样平均显存：`9310.72 MiB`
- 最高 GPU utilization：`96%`

观察：

- `independent_pair_xyz` 使用 `batch_size=32`，训练期采样基本稳定在 `9660 MiB`，接近 10GB 卡上限，但未 OOM。
- `somoformer_xyz` 使用 `batch_size=16`，初始化训练段约 `5510 MiB`，训练/评估阶段采样峰值约 `9552 MiB`。
- 该监控是手动轮询，不是连续 profiler；因此只能说明采样到的显存峰值，不保证捕捉绝对瞬时峰值。

## 结论边界

本结果只支持 joint-space P7 口径下的结论：SoMoFormer-style joint/person token attention 在 3 个 seed 上稳定优于同样用双人样本训练的 `independent_pair_xyz` baseline。

不能把该结果直接写成优于 P5 active-vector 主表的 independent/relation 模型，因为训练目标、输出空间和 evaluator 口径不同。
