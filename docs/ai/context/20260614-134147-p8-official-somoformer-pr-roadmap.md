# P8 官方 SoMoFormer 架构迁移目标与 PR 路线

## 用户确认的最终目标

最终目标不是继续维护 `SoMoFormer-lite XYZ`，而是把 `/home/rpartx3080/CodeSpace/somoformer` 中官方 SoMoFormer 架构完整迁入 ReGenNet forecasting pipeline，并在 ReGenNet InterHuman two-person forecasting 协议下作为标准控制变量 baseline 训练、评估和对比。

必须避免的偏差：

- 不能把当前简化版 `JointSpaceSoMoFormer` 称为完整 SoMoFormer。
- 不能只做 DCT -> Transformer -> IDCT 的思想复刻后声称官方架构复现。
- 不能把 joint-space P8 结果直接横比 P5 active-vector 主表，除非另做同口径 active-vector adapter。

## 官方 SoMoFormer 架构关键点

官方代码位置：

- `/home/rpartx3080/CodeSpace/somoformer/src/model.py`
- `/home/rpartx3080/CodeSpace/somoformer/src/utils/dct.py`
- `/home/rpartx3080/CodeSpace/somoformer/src/configs/release.yaml`

官方 `SoMoFormer` 关键机制：

- DCT/IDCT 使用 `utils.dct.get_dct_matrix(seq_len)` 得到完整矩阵，再截取前 `dct_n` 个系数。
- 输入先按最后观测帧 padding 到完整 `seq_len`。
- token 是 `N * J * K`，即 person、joint、coordinate 组合。
- `location_method` 支持 `grid|neck|naive`，release 默认 `grid`。
- `grid` 分支使用 neck/root 的 xz 位置离散到 grid embedding，并 concat 到 token feature。
- `LearnedDoublePositionalEncoding` 交错注入 joint-id 和 person-id embedding。
- Transformer 使用 `AuxilliaryEncoder`，返回 final output 和每层 auxiliary output。
- 训练 loss 使用 final prediction + 每层 auxiliary prediction 的加权平均。
- 训练时支持 `metamask`，对 DCT token 做随机 mask。
- 支持 padding mask，对无效 person token 做 attention mask。
- 支持 `normalize_inputs`、`residual_connection`、`output_scale` 等配置。

## 当前 P7 lite 与官方差距

当前 `model/forecasting_somoformer.py` 已有：

- DCT -> linear -> TransformerEncoder -> linear -> IDCT
- person/joint/coord embedding
- residual connection
- two-person SMPL xyz 输入输出

当前缺失或不一致：

- 没有官方 `AuxilliaryEncoder` 和 auxiliary loss。
- 没有官方 `LearnedDoublePositionalEncoding` 的交错 joint/person embedding。
- 没有官方 `grid/neck/naive location_method`，当前是简单 person/joint/coord embedding。
- 没有官方 `padding_mask` 接口。
- 没有官方 `metamask` 训练增强。
- DCT 矩阵构造方式虽数学接近，但不是直接按官方 `get_dct_matrix` 代码路径。
- 当前输出只返回 future，不返回 full sequence 和 aux outputs；官方返回 full sequence 和 aux outputs。

## 小 PR 路线

### PR 1：迁入官方架构模块，不跑完整大实验

目标：

- 新增官方对齐的 ReGenNet adapter，例如 `OfficialSoMoFormerXYZ`。
- 保留当前 `somoformer_xyz` 作为 lite baseline，不覆盖旧 checkpoint 语义。
- 新增 model type，例如 `official_somoformer_xyz`。
- 直接复用官方 `AuxilliaryEncoder`、`LearnedDoublePositionalEncoding`、`get_dct_matrix` 逻辑。
- 在 ReGenNet xyz 数据上适配输入：
  - `obs_xyz [B,30,2,24,3]`
  - reshape 为官方期望的 `tgt [B,30,NJ,K]`
  - `N=2,J=24,K=3`
  - `tgt_neck` 第一版使用 root joint `joint 0` 的第一帧位置作为 grid/neck 位置代理，原因是当前 SMPL 24 joint 中没有显式 SoMoF neck 13-joint 定义。
  - `padding_mask` 第一版全 False，因为 InterHuman 固定两人都有效。
- 训练 loop 支持 official aux loss 和 train-time metamask。

验收：

- `compileall` 通过。
- dataset smoke / xyz metrics sanity 不漂移。
- `official_somoformer_xyz` 2-step train smoke 通过。
- checkpoint eval 可落盘。
- 旧 `somoformer_xyz` checkpoint eval 保持可用。

不做：

- 不跑 5000-step 3-seed。
- 不改 P1-P6 active-vector 主路径。
- 不接 NTU/Chi3D。

### PR 2：官方架构 single-seed 标准 baseline

目标：

- 用 `official_somoformer_xyz` 跑 InterHuman P7 xyz 协议 seed0，预算与当前 P7 对齐。
- 同时复跑或复用同口径控制变量：
  - repeat_xyz
  - independent_pair_xyz seed0
  - somoformer_lite_xyz seed0
  - official_somoformer_xyz seed0

验收：

- 输出 `results/forecasting/interhuman/p8_official_somoformer_xyz_seed0/summary.*`
- 记录显存、参数量、训练命令、checkpoint、metrics。
- 如果官方架构不优于 lite，也必须保留结果，不能调参后只报告成功版本。

### PR 3：官方架构 3-seed 控制变量 baseline

目标：

- 跑 `official_somoformer_xyz` seeds `0,1,2`。
- 汇总与 P7 3-seed 同口径对比。

验收：

- 生成 3-seed mean/std。
- 报告 same-seed official vs lite、official vs independent_pair。
- 记录显存采样。
- 明确结论是否支持“完整官方架构优于 lite / independent_pair”。

### PR 4：可选架构消融

目标：

- 做 official 关键机制消融：
  - grid vs naive/neck
  - aux loss on/off
  - metamask on/off
  - normalize_inputs on/off
  - dct normalization on/off

验收：

- 只在 PR 3 结果有必要解释时启动。
- 不允许先消融再选择性定义主模型。

### PR 5：可选扩展到 active-vector 或其他数据集

目标：

- 若论文需要和 P5 active-vector 主表合流，再设计 `official_somoformer_active`。
- 若需要跨数据集，再单独接 NTU120-AS / Chi3D-AS forecasting loader。

验收：

- 必须单独定义数据协议和 evaluator。
- 不和 InterHuman P8 joint-space 主结果混合写结论。

## 当前分支策略

当前新分支：

- `p8-official-somoformer-xyz`

本分支第一阶段只应完成 PR 1。PR 2/3 是训练实验 PR，不应该和架构迁移 PR 混在一起。

## 当前实现决策

- 官方架构必须作为新 model type 接入，避免覆盖 P7 lite checkpoint 语义。
- 第一版严格以官方代码结构为准，不自行发明替代 embedding 或 attention block。
- DCT 使用官方 `get_dct_matrix` 逻辑，矩阵注册为 buffer 以适配 PyTorch module 迁移；数学路径保持官方。
- InterHuman 固定两人有效，`padding_mask` 默认全 False。
- `tgt_neck` 使用 root joint 第一帧位置代理，后续若要更接近 SoMoF neck，需要单独建立 SMPL 24 joint 到 SoMoF 13 joint/neck 的映射。
