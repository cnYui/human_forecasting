# 交互感知双人动作联合预测设计文档

## 目标已锁定

论文第一阶段正式锁定为：

```text
Interaction-aware joint forecasting of two-person human motion from partial observations.
```

中文：

```text
基于部分观测的交互感知双人动作联合未来预测。
```

主协议：

```text
dataset: InterHuman
window_len: 150
obs_len: 30
pred_len: 120
input:  前 30 帧双人动作
target: 后 120 帧双人动作
task:   联合预测两个人未来动作
```

## 当前进程状态

已停止当前 ReGenNet 50K baseline 长跑进程。

停止前状态：

```text
save_dir: save/interhuman/paper_config_l8_d512_accum64_50000_baseline
last checkpoint: model000020000.pt / opt000020000.pt
log last status: Interrupted system call
```

当前未发现 ReGenNet 训练进程残留，`nvidia-smi --query-compute-apps` 返回为空。

## 当前工程现状

### 可复用资产

数据：

```text
dataset/interhuman/smpl/conditioned/interhuman_train.h5
dataset/interhuman/smpl/conditioned/interhuman_val.h5
dataset/interhuman/smpl/conditioned/interhuman_test.h5
dataset/interhuman/smpl/conditioned/meta.json
```

InterHuman H5 格式：

```text
shape: [T, 25, 12]
body_model: smpl
rotation: rot6d
translation_slot: 24
actor_rot6d: [:, :24, 0:6]
actor_translation: [:, 24, 0:3]
reactor_rot6d: [:, :24, 6:12]
reactor_translation: [:, 24, 6:9]
translation_origin: actor frame 0
```

本地可用窗口数量：

```text
InterHuman train: 6021 条，长度 >=150 的有 2910 条。
InterHuman val:   580 条，长度 >=150 的有 226 条。
InterHuman test:  1175 条，长度 >=150 的有 508 条。
```

环境：

```text
torch: 1.7.1
GPU: NVIDIA GeForce RTX 3080
CUDA available: true
```

已有代码可复用：

```text
preprocess/interhuman_as.py      # InterHuman H5 生成逻辑
data_loaders/a2m/interhuman.py   # H5 读取经验
utils/rotation_conversions.py    # rot6d / matrix / axis-angle 转换
model/rotation2xyz.py            # 后续 MPJPE / xyz 指标可用
train/training_loop.py           # checkpoint / log / grad accumulation 经验可参考
```

### 不能直接复用的部分

当前 ReGenNet 主训练路径：

```text
train/train_mdm.py
utils/model_util.py
model/cmdm.py
diffusion/gaussian_diffusion.py
data_loaders/tensors.py::ccollate
```

不适合直接作为 forecasting 主线，原因：

```text
ccollate 会把 [T,25,12] 按 channel 拆成 actor 条件 cmotion 和 reactor 目标 motion。
CMDM.forward() 硬依赖 y["cmotion"]。
diffusion training loss 只对 reactor 目标加噪，关系 loss 也是 cmotion 与 reactor 的相对关系。
eval/eval_cmdm.py 只支持 NTU / Chi3D 的 ST-GCN 评估，不支持 InterHuman forecasting MSE 指标。
```

因此不能靠改命令实现：

```text
前 30 帧双人动作 -> 后 120 帧双人动作
```

需要新增 forecasting 专用数据、模型、训练和评估路径。

## 第一阶段设计原则

1. 先做确定性 forecasting，不先做 diffusion。
2. 先证明 relation-aware joint predictor 明显优于 independent / concat。
3. 先用 MSE 类和交互关系指标闭环，不依赖 ST-GCN recognition checkpoint。
4. 保留 ReGenNet 作为历史 backbone / 后续生成式 baseline，不让它阻塞主线。
5. 第一版只用 InterHuman；NTU120-AS 和 Chi3D 后续作为扩展。

## 数据设计

新增数据模块建议：

```text
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
```

输出格式建议从 H5 的 `[T,25,12]` 转成 active vector，避免 zero padding channel 污染 MSE：

```text
person_active_dim = 24 * 6 + 3 = 147
two_person_dim = 294
obs:    [obs_len, 2, 147]
target: [pred_len, 2, 147]
```

active vector 映射：

```text
person A = actor rot6d [:24,0:6] + actor translation [24,0:3]
person B = reactor rot6d [:24,6:12] + reactor translation [24,6:9]
```

训练采样：

```text
只保留 T >= 150 的样本。
每次 __getitem__ 随机裁剪连续 150 帧。
前 30 帧为 obs，后 120 帧为 target。
```

验证/测试采样：

```text
第一版使用 center crop，保证确定性。
后续可扩展为 multi-window evaluation。
```

归一化：

```text
使用 train split 统计 active vector 的 mean/std。
训练 loss 在 normalized space 计算。
评估指标在 original scale 计算。
```

注意：

```text
不要对长度 <150 的序列 padding 后纳入第一阶段主协议。
padding 会伪造未来，破坏 forecasting 任务。
```

## 模型设计

新增模型模块建议：

```text
model/forecasting.py
```

### Baseline 0：Repeat / Zero Velocity

无需训练：

```text
pred[t] = obs[-1]
```

用途：

```text
建立最低可用误差下界。
验证 evaluator 和指标没有实现错误。
```

### Baseline 1：Independent Predictor

定义：

```text
A_obs -> A_future
B_obs -> B_future
concat(A_future, B_future)
```

模型：

```text
共享或独立 GRU/Transformer encoder
每个人只看自己的 obs
decoder 直接输出 pred_len * 147
```

约束：

```text
不能读取另一个人的 obs。
```

### Baseline 2：Concat No-Relation Predictor

定义：

```text
concat(A_obs, B_obs) -> future(A,B)
```

模型：

```text
GRU/Transformer encoder 读取 [obs_len, 294]
MLP decoder 输出 [pred_len, 2, 147]
```

约束：

```text
允许看到两个人历史，但不显式构造 relation features / relation encoder。
```

### Ours：Relation-Aware Joint Predictor

输入：

```text
obs: [B, obs_len, 2, 147]
```

关系特征：

```text
relative root translation: trans_A - trans_B
relative root velocity: velocity_A - velocity_B
root distance: ||trans_A - trans_B||
relative root orientation: R_A^T R_B
```

第一版结构：

```text
person_encoder(A_obs) -> h_A
person_encoder(B_obs) -> h_B
relation_encoder(relation_features) -> h_rel
fuse(h_A, h_B, h_rel) -> decoder -> future(A,B)
```

推荐先用小型 GRU/MLP：

```text
hidden_dim: 256 或 512
num_layers: 2
decoder: MLP direct prediction
```

理由：

```text
PyTorch 版本是 1.7.1，先避免依赖较新的 Transformer API。
任务第一阶段更需要清晰 ablation，而不是复杂生成模型。
```

## Loss 设计

主训练 loss：

```text
normalized active-vector MSE
```

可选关系 loss：

```text
relative root distance MSE
relative orientation MSE
translation MSE
```

第一版建议：

```text
先只训练 reconstruction MSE。
relation loss 作为 ablation 加入。
```

原因：

```text
如果 relation-aware architecture 在相同 reconstruction loss 下已优于 baseline，主张更干净。
否则再判断是否需要 relation loss 强化交互一致性。
```

## 评估设计

新增评估模块建议：

```text
eval/eval_forecasting.py
utils/forecasting_metrics.py
```

必须指标：

```text
future_mse
rotation_mse
translation_mse
short_mse       # pred frames 0-39
mid_mse         # pred frames 40-79
long_mse        # pred frames 80-119
relative_root_distance_error
relative_orientation_error
inter_person_distance_consistency
```

指标计算原则：

```text
rotation_mse 只计算 24 个 rot6d joints。
translation_mse 只计算 active translation，不计算 padding zero channel。
relative_root_distance_error 使用 actor/reactor translation。
relative_orientation_error 使用 root rot6d -> matrix 后计算相对旋转角。
```

后续指标：

```text
MPJPE
inter-person joint distance error
collision / penetration proxy
best-of-K error
diversity
coverage
```

## 训练入口设计

新增入口建议：

```text
train/train_forecasting.py
```

核心参数：

```text
--dataset interhuman
--data_path dataset/interhuman/smpl/conditioned
--save_dir save/forecasting/interhuman/...
--window_len 150
--obs_len 30
--pred_len 120
--model_type repeat|independent|concat|relation
--batch_size
--num_steps
--lr
--hidden_dim
--num_layers
--seed
--max_samples
```

保存内容：

```text
args.json
normalizer.pt 或 normalizer.json
model*.pt
metrics_val.yaml
metrics_test.yaml
```

训练循环不建议直接复用 `TrainLoop`，原因：

```text
现有 TrainLoop 绑定 diffusion.training_losses 和 DDP 逻辑。
forecasting 第一版只需要标准 supervised loss。
```

可以复用经验：

```text
checkpoint 命名
grad_accum_steps
log_interval / save_interval
max_samples smoke
num_workers=0 的 InterHuman 退出策略
```

## 任务分解

### P0：协议冻结与文档

产出：

```text
AGENTS.md 正式记录目标。
docs/ai/context 设计文档。
```

状态：

```text
已完成。
```

### P1：Forecasting Dataset 与 Normalizer

任务：

```text
新增 InterHumanForecastDataset。
实现 active vector extract / restore。
实现 random train crop 和 deterministic eval crop。
实现 train mean/std 统计和缓存。
```

验收：

```text
train/val/test loader 可返回 obs/target。
shape = obs [B,30,2,147], target [B,120,2,147]。
无 NaN / Inf。
长度 <150 样本不会进入第一阶段数据集。
```

### P2：Forecasting Metrics 与 Repeat Baseline

任务：

```text
实现 metrics。
实现 repeat baseline evaluator。
输出 YAML/JSON 指标。
```

验收：

```text
pred == target 时所有 MSE 类指标为 0。
repeat baseline 能在 test split 上完整跑完。
```

### P3：Independent 与 Concat Baseline

任务：

```text
实现 supervised train loop。
实现 independent predictor。
实现 concat no-relation predictor。
训练 smoke 和中等训练。
```

验收：

```text
max_samples smoke 2-5 step 通过。
完整 train 可保存 checkpoint。
val/test 指标可复现。
```

### P4：Relation-Aware Joint Predictor

任务：

```text
实现 relation feature extraction。
实现 relation encoder。
实现 relation-aware predictor。
与 independent / concat 使用相同数据协议和训练设置。
```

验收：

```text
relation-aware 在 long_mse 和交互指标上优于 independent / concat。
如果没有优势，回到模型假设检查，而不是继续堆功能。
```

### P5：Ablation 与论文表格

任务：

```text
去掉 relation encoder。
只用 relative translation。
只用 relative orientation。
不同观测比例：10% / 20% / 30% / 50%。
不同窗口协议可作为补充：60/150。
```

验收：

```text
形成主结果表、ablation 表、长期误差曲线。
```

### P6：可视化与定性分析

任务：

```text
把 active vector 还原为 H5-like motion。
复用 render / rot2xyz 或新增轻量可视化。
保存若干 obs/gt/pred 对比样本。
```

验收：

```text
至少 8 个 test 样本可视化。
动作数值有限，轨迹和相对距离曲线可检查。
```

### P7：扩展路线

不进入第一阶段阻塞项：

```text
NTU120-AS action-conditioned forecasting
Chi3D 小规模泛化验证
InterHuman 文本条件预测
diffusion / ReGenNet forecasting baseline
SMPL-X 统一表示
```

## 最短可执行路径

最短闭环顺序：

```text
1. P1 dataset
2. P2 repeat baseline evaluator
3. P3 concat baseline
4. P3 independent baseline
5. P4 relation-aware model
6. P5 ablation table
```

不要先做：

```text
复杂 graph network
diffusion forecasting
natural language condition
Table 4 evaluator
SMPL-X conversion
```

## 风险与判断标准

### 风险 1：150 帧过滤后训练样本减少

当前 train 仍有 2910 条，第一阶段可接受。

如果训练不稳：

```text
增加每条长序列的随机窗口采样次数。
或补充 120 帧窗口协议作为消融。
```

### 风险 2：translation 和 rotation 尺度不一致

必须使用 normalizer，并在原始尺度报告指标。

### 风险 3：relation-aware 没有明显优势

这不是实现小问题，而是论文主张风险。处理顺序：

```text
先检查 independent / concat 是否公平。
再检查 relation features 是否从 obs 中正确提取。
再增加 relation loss 或 cross-attention。
最后才考虑更复杂图网络。
```

### 风险 4：长期预测 MSE 偏向平均动作

第一阶段先承认 deterministic forecasting 只验证关系建模。
多模态预测、best-of-K 和 diversity 后续再做。

## 下一步立即任务

下一轮实现应从 P1 开始：

```text
新增 data_loaders/forecasting/interhuman.py
新增 data_loaders/forecasting/tensors.py
新增 utils/forecasting_motion.py
新增最小 shape/finite smoke
```

完成 P1/P2 后再进入模型实现。
