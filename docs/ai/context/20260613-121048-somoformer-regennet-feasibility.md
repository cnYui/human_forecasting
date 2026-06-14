# 用 ReGenNet 数据复现 / 适配 SoMoFormer 的可行性评估

## 问题

用户希望用当前 ReGenNet 的数据集来复现 SoMoFormer，并用 SoMoFormer 的框架实现两个人的未来预测。

本评估只做设计和可行性判断，不修改训练代码。

## 已检查代码

ReGenNet 当前 forecasting 路径：

- `docs/ai/context/20260603-190003-forecasting-final-official-design.md`
- `data_loaders/forecasting/interhuman.py`
- `utils/forecasting_motion.py`
- `utils/forecasting_metrics.py`
- `train/train_forecasting.py`
- `model/forecasting.py`
- `model/rotation2xyz.py`
- `model/smpl.py`

SoMoFormer 当前代码：

- `/home/rpartx3080/CodeSpace/somoformer/src/model.py`
- `/home/rpartx3080/CodeSpace/somoformer/src/dataset.py`
- `/home/rpartx3080/CodeSpace/somoformer/src/train.py`
- `/home/rpartx3080/CodeSpace/somoformer/src/evaluate.py`
- `/home/rpartx3080/CodeSpace/somoformer/src/configs/release.yaml`
- `/home/rpartx3080/CodeSpace/somoformer/requirements.txt`

## 核心结论

可行，但不建议把 SoMoFormer 原仓库直接改成读取 ReGenNet 数据来跑。推荐做法是在 ReGenNet 当前 forecasting pipeline 内新增 `somoformer` model type，复用 ReGenNet 的 dataset / normalizer / evaluator / aggregate 结果闭环，把 SoMoFormer 的 DCT + joint/person token Transformer 结构迁进来。

原因：

- ReGenNet 已有 InterHuman 150/30/120 的固定协议、metrics、repeat / independent / concat / relation 主表和 P5 aggregate 工具。
- SoMoFormer 原仓库绑定 3DPW / SoMoF 数据入口、16/14 帧协议、13 个 3D keypoint、VIM/MPJPE 评估。
- 如果直接在 SoMoFormer 仓库改数据入口，结果无法直接和当前 ReGenNet P5 主表同口径比较。
- 当前论文主张需要继续围绕 ReGenNet 的 deterministic two-person forecasting protocol，而不是改成 SoMoF benchmark 口径。

## 两边协议差异

### ReGenNet 当前主协议

```text
dataset: InterHuman SMPL H5
source: dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5
raw shape: [T,25,12]
active shape: [T,2,147]
obs: [B,30,2,147]
target: [B,120,2,147]
representation: 24 SMPL rot6d + root translation per person
metrics: active-vector original-scale MSE + relation metrics
```

### SoMoFormer 原协议

```text
dataset: 3DPW / SoMoF
model input after batch_process_joints: [B,in_F,N*J,3]
default in_F/out_F: 16/14
default full seq_len: 30
default joints: 13
representation: 3D joint coordinates
token axis: N * J * 3 coordinate tokens
core: DCT over time + learned joint/person embedding + Transformer encoder
loss: masked keypoint MSE on future joints
metrics: VIM / MPJPE
```

## 重要技术判断

### 1. 原版 SoMoFormer 不是 rot6d 模型

SoMoFormer 的建模对象是 joint coordinate trajectory，不是 SMPL rotation 参数。它的 token 是 `N*J*K`，其中 `K=3` 是 xyz 坐标维度。

因此不能简单把 ReGenNet 的 `[B,30,2,147]` reshape 给 SoMoFormer。这样会把 rot6d 维度当成 xyz 坐标 token，破坏模型的结构假设。

### 2. ReGenNet 可以生成 joint-space 输入

当前项目有：

- `model/rotation2xyz.py`
- `model/smpl.py`
- `body_models/smpl`

理论上可以把 active vector 的 `24*6 + trans` 转为 SMPL joints，得到：

```text
[B,T,2,J,3]
```

可选 joint set：

- `smpl`: 24 joints，保留 SMPL 主体关节。
- `a2m`: 18 joints，更接近 action2motion / recognition 旧路径。
- `vibe`: OpenPose 风格 joints，但和 SoMoFormer 13 joints 不完全一致。

第一阶段建议用 `smpl` 24 joints，不强行裁成 SoMoF 13 joints，避免引入不必要的映射误差。

### 3. 直接 joint-space 预测不能替代当前 active-vector 主表

如果训练 SoMoFormer 预测 xyz joints，只能自然评估 MPJPE / joint MSE / root distance 等 joint-space 指标。它不能直接输出 rot6d，所以不能无损接入当前 `future_mse / rotation_mse / translation_mse` 主表。

若要和 P5 的 independent / concat / relation 做完全同口径比较，需要 SoMoFormer 输出 `[B,120,2,147]` active vector，或者设计双头模型：

```text
SoMoFormer-style encoder -> active-vector future decoder
optional auxiliary joint-space loss
```

这是推荐的论文级适配，而不是纯复现。

### 4. 150 帧协议可支持，但要调 DCT

SoMoFormer 用全序列 DCT：先把 30 帧序列扩展为 obs + last-pose padding，再对 full sequence 做 DCT。当前任务 full sequence 是 150 帧，理论可行。

建议配置：

```text
seq_len = 150
input_track_size = 30
output_track_size = 120
dct_n = 30 或 50 起步
num_people = 2
num_joints = 24 if joint-space
```

token 数变化：

```text
SoMoF 原始: 2 * 13 * 3 = 78 tokens
InterHuman SMPL joint-space: 2 * 24 * 3 = 144 tokens
active-vector 维度 token: 2 * 147 = 294 tokens
```

144 tokens 对 Transformer 仍可接受；294 tokens 也可跑，但语义不如 joint-coordinate tokens 清晰。

## 环境现状

默认 shell 的 `python3` 当前缺少：

- `torch`
- `torch_dct`
- `smplx`

SoMoFormer requirements 包含：

```text
torch==1.10.0
torch-dct==0.1.5
pyyaml==6.0
progress==1.6
```

ReGenNet 过去训练显然使用过 PyTorch 环境，但当前 shell 未激活该环境。真正实现前必须先确认训练环境，例如 conda env、CUDA、PyTorch 版本和 `smplx` 是否可安装。

## 推荐路线

### P7.0 环境与协议 smoke

目标：确认能在当前机器把 active vector 转成 SMPL joints，并确定训练环境。

任务：

- 找到 / 激活 ReGenNet 实际训练环境。
- 安装或确认 `smplx`、`torch-dct`。
- 写最小转换 helper：`active [B,T,2,147] -> xyz [B,T,2,24,3]`。
- 做 4 samples finite / shape smoke。

验收：

```text
obs_xyz: [B,30,2,24,3]
target_xyz: [B,120,2,24,3]
finite: true
```

### P7.1 joint-space SoMoFormer baseline

目标：先用 SoMoFormer 原始建模假设验证 InterHuman 数据上的双人 joint prediction。

建议新增：

- `utils/forecasting_xyz.py`
- `model/forecasting_somoformer.py`
- `train/train_forecasting_xyz.py` 或在现有 train 里加独立入口
- `eval/eval_forecasting_xyz.py`

输出：

```text
pred_xyz: [B,120,2,24,3]
```

指标：

- joint_mse
- MPJPE-like
- root_translation_error
- relative_root_distance_error
- long_horizon_joint_mse

价值：

- 最接近 SoMoFormer 原论文框架。
- 可验证 joint/person token Transformer 是否适合 InterHuman。

限制：

- 不能直接和 P5 active-vector 主表比较。
- 不能替代当前论文主结果。

### P7.2 paper-comparable SoMoFormer-active

目标：把 SoMoFormer 的 DCT token Transformer 作为 ReGenNet forecasting 的新 `model_type=somoformer_active`，输出 active vector，复用 P2/P5 指标。

推荐结构：

```text
obs active [B,30,2,147]
-> split rotation/trans tokens
-> optional SMPL joint xyz auxiliary tokens
-> DCT over 150-frame padded sequence
-> joint/person/channel embeddings
-> Transformer encoder
-> active future decoder [B,120,2,147]
```

训练：

```text
main loss: normalized active-vector MSE
optional aux loss: predicted active -> xyz 与 target xyz 的 joint loss
```

价值：

- 可和 repeat / independent / concat / relation 同口径比较。
- 最适合作为当前论文的下一版 strong baseline 或新 backbone。

风险：

- 设计空间更大，需要防止引入过多不可解释变量。
- 如果只用 active-dim token 而不引入 joint-space inductive bias，可能不如 independent。

## 不推荐路线

### 不推荐直接在 SoMoFormer 仓库改数据入口

原因：

- 会绕开 ReGenNet 已有 P1-P6 evaluator 和 aggregate。
- SoMoFormer 原 evaluator 固定 16/14 和 3DPW/SoMoF 假设。
- 结果难以和现有 baselines 公平对比。

### 不推荐把 rot6d 直接当 xyz 坐标喂 SoMoFormer

原因：

- SoMoFormer 的 joint/person/location embedding 建立在 xyz 坐标语义上。
- rot6d 各维度不是空间坐标，DCT token 仍可数学上运行，但“joint-coordinate trajectory token”的归纳偏置失效。

## 粗略工作量

按已有 ReGenNet forecasting 代码基础估计：

- P7.0 环境和 xyz smoke：0.5-1 天。
- P7.1 joint-space SoMoFormer baseline：2-4 天，含 smoke、训练、eval 和结果落盘。
- P7.2 active-vector 同口径 SoMoFormer：4-7 天，含模型设计、训练、P5 aggregate 接入、3-seed 主表。

若训练环境或 `smplx` 依赖冲突，额外增加 0.5-1 天。

## 最终建议

先做 P7.0 + P7.1，不直接冲 P7.2。

原因：

- P7.1 是最接近 SoMoFormer 原框架的最小可验证版本。
- 如果 P7.1 在 joint-space 上都明显弱，说明 SoMoFormer 思路在当前 InterHuman 协议下未必值得继续深挖。
- 如果 P7.1 长期 joint error / relative distance 明显优于 concat，再进入 P7.2，把它改成 active-vector 同口径模型，纳入 P5 风格主表。

论文表述边界：

- 可以写作“引入 SoMoFormer-style joint/person trajectory-token Transformer baseline”。
- 不能写“复现 SoMoFormer 原论文结果”，因为数据集、帧协议、joint set 和指标都不同。
- 不能只和 concat 比。如果 P7.2 成功，仍必须同时报告 independent、concat、relation-aware 和 SoMoFormer-style。
