# P8 官方 SoMoFormer XYZ 3-seed baseline 结果

## 背景

用户确认最终目标为完整迁入官方 SoMoFormer 架构，而不是继续使用 `SoMoFormer-lite XYZ`。本阶段在分支 `p8-official-somoformer-xyz` 上完成官方架构第一版接入，并按标准控制变量 baseline 跑完 InterHuman P7/P8 joint-space 3-seed。

## 实现范围

新增官方对齐模型类型：

- `official_somoformer_xyz`

关键设计：

- 保留旧 `somoformer_xyz` 作为 lite baseline，不覆盖旧 checkpoint 语义。
- 官方架构核心组件迁入：
  - `AuxilliaryEncoder`
  - `LearnedDoublePositionalEncoding`
  - 官方 `get_dct_matrix` 逻辑
  - `grid|neck|naive` location method
  - grid embedding
  - residual connection
  - auxiliary outputs / auxiliary loss
  - train-time `metamask` 入口
  - padding mask 入口
- InterHuman 固定两人有效，`padding_mask` 第一版为全 False。
- `tgt_neck` 使用 SMPL root joint 第一帧位置代理。该点是数据适配边界，不是 SoMoF neck joint 的严格复刻。
- DCT/IDCT 矩阵按官方公式生成，并注册为 PyTorch buffer。

## 验证

已通过：

- `compileall model/forecasting_somoformer.py model/forecasting_xyz.py train/train_forecasting_xyz.py eval/eval_forecasting_xyz.py`
- official/lite/independent 三种模型随机 forward shape/finite 检查
- `eval.eval_forecasting_xyz --mode xyz_smoke`
- `eval.eval_forecasting_xyz --mode metrics_sanity`
- `official_somoformer_xyz` 2-step train smoke
- 旧 `somoformer_xyz` seed0 checkpoint eval 回归，指标保持 `joint_mse=0.0596641728`

## 正式训练配置

共同配置：

- dataset：`dataset/interhuman/smpl/conditioned`
- split：train/val/test = `2910/226/508`
- input/output：`obs_len=30,pred_len=120,window_len=150`
- output space：SMPL xyz `[B,T,2,24,3]`
- batch：`16`
- eval batch：`32`
- steps：`5000`
- model：`official_somoformer_xyz`
- hidden：`256`
- layers：`6`
- heads：`8`
- FFN：`1024`
- DCT：`30`
- lr：`1e-4`
- aux weight：`0.2`
- residual：`true`
- location method：`grid`

checkpoint：

- seed0：`save/forecasting/interhuman/p8_official_somoformer_xyz_h256_l6_dct30_s0_5000/model000005000.pt`
- seed1：`save/forecasting/interhuman/p8_official_somoformer_xyz_h256_l6_dct30_s1_5000/model000005000.pt`
- seed2：`save/forecasting/interhuman/p8_official_somoformer_xyz_h256_l6_dct30_s2_5000/model000005000.pt`

## 3-seed 结果

汇总输出：

- `results/forecasting/interhuman/p8_official_somoformer_xyz_main/summary.json`
- `results/forecasting/interhuman/p8_official_somoformer_xyz_main/summary.csv`
- `results/forecasting/interhuman/p8_official_somoformer_xyz_main/summary.md`

| model | joint_mse | mpjpe | long_joint_mse | relative_root_distance_error | inter_person_distance_consistency_xyz |
|---|---:|---:|---:|---:|---:|
| independent_pair_xyz | 0.0682312447 +- 0.0017461769 | 0.3375963565 +- 0.0080529239 | 0.1284009837 +- 0.0019345306 | 0.2280735768 +- 0.0053159672 | 0.0233993507 +- 0.0000829443 |
| somoformer_lite_xyz | 0.0607606915 +- 0.0010012505 | 0.2947844890 +- 0.0055710050 | 0.1186761773 +- 0.0018992134 | 0.2032881472 +- 0.0116076032 | 0.0056123107 +- 0.0001299308 |
| official_somoformer_xyz | 0.0706294909 +- 0.0007989965 | 0.3033010067 +- 0.0015107005 | 0.1376462394 +- 0.0015484277 | 0.1935767838 +- 0.0035398403 | 0.0054475044 +- 0.0000480530 |

official 相对 independent：

- `joint_mse`：`-3.51%`
- `mpjpe`：`+10.16%`
- `long_joint_mse`：`-7.20%`
- `relative_root_distance_error`：`+15.13%`
- `inter_person_distance_consistency_xyz`：`+76.72%`

official 相对 lite：

- `joint_mse`：`-16.24%`
- `mpjpe`：`-2.89%`
- `long_joint_mse`：`-15.98%`
- `relative_root_distance_error`：`+4.78%`
- `inter_person_distance_consistency_xyz`：`+2.94%`

正数表示 official 更好，负数表示 official 更差。

## 显存

显存采样：

- `results/forecasting/interhuman/p8_official_somoformer_xyz_main/gpu_monitor.csv`

采样摘要：

- sample count：`9`
- 采样峰值：`9586 MiB / 10240 MiB`
- 非空闲采样平均：`9008.86 MiB`
- 最高 utilization：`84%`

该监控是手动 `nvidia-smi` 轮询，不是连续 profiler。

## 客观结论

官方完整架构第一版没有全面优于当前 `somoformer_lite_xyz`。

更具体地说：

- official 在 `relative_root_distance_error` 和 `inter_person_distance_consistency_xyz` 上优于 lite。
- official 在 `joint_mse`、`mpjpe`、`long_joint_mse` 上不如 lite。
- official 相对 independent 在 `mpjpe` 和 relation-style metrics 上有收益，但 `joint_mse/long_joint_mse` 更差。

因此当前不能声称“完整官方 SoMoFormer 架构迁入后整体优于 lite baseline”。更准确的表述是：官方结构在相对关系保持上更强，但当前 ReGenNet InterHuman SMPL 24-joint xyz 适配下，整体 joint-space 误差不如 lite。

## 后续风险与下一步

主要风险：

- `tgt_neck` 当前用 SMPL root 代理，不是 SoMoF neck joint。
- 官方 release 的 `normalize_inputs=false`，因此官方 13-joint 统计常量未作用于当前 24-joint SMPL；若后续打开 normalize，需要重新定义 24-joint 统计。
- 官方 grid 逻辑按原代码保留，但它对当前 root 坐标分布可能不理想。
- 官方 `metamask` 代码路径按原公式迁入，但原实现中 mask 分支实际接近 no-op；后续如要修正应作为消融，不能混入主 baseline。

建议下一步：

- 做官方适配消融：`root-as-neck` vs 显式 neck/upper-body joint proxy。
- 做 `location_method=naive/grid/neck` 对比。
- 做 `aux_loss on/off` 和 `metamask fixed/no-op` 对比。
- 若目标是论文主表，必须保留当前失败边界，不能只报告 relation metrics。
