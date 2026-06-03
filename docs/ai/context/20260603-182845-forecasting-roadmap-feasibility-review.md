# Forecasting P1-P6 路线图可行性审阅

## 审阅定位

本次按 `$academic-paper` 的 plan / structure / argumentation 视角审阅：

```text
docs/ai/context/20260603-161803-forecasting-p1-p6-roadmap.md
```

审阅目标不是实现 P1-P6，而是判断当前项目条件下该路线图是否能支撑一篇经验型方法论文的第一阶段。

## 结论

总体判断：

```text
工程可行性：中高
论文主张可行性：有条件成立
当前是否允许进入 P1：允许
当前是否允许跳到 P3/P4：不允许
```

路线图的强点是任务边界收得足够窄，先做 InterHuman deterministic forecasting，先闭环 dataset / metrics / repeat baseline，再进入模型和 ablation。这符合经验型方法论文的 IMRaD 结构，也避免继续被 ReGenNet Table 4、SMPL-X、text condition 和 recognition evaluator 阻塞。

核心保留意见是：relation-aware model 是否能形成论文贡献，必须由 P4/P5 的对照实验证明。只要 long_mse 和至少一个 relation metric 不能稳定优于 concat no-relation baseline，就不能把路线图包装成成功论文。

## 本地事实核对

已确认本地可用环境：

```text
micromamba env: /home/rpartx3080/.local/micromamba/envs/regennet
python: 3.7.13
torch: 1.7.1
cuda_available: True
h5py: 3.7.0
```

裸系统 `python3` 缺少 `h5py`，因此后续 smoke / eval 命令应显式使用 regennet 环境。

已确认 H5 数据与路线图一致：

```text
format: [T,25,12]
translation_slot: 24
translation_origin: actor frame 0

train: total 6021, T>=150 2910, bad_shape 0
val:   total 580,  T>=150 226,  bad_shape 0
test:  total 1175, T>=150 508,  bad_shape 0
```

当前 forecasting 专用文件尚不存在：

```text
data_loaders/forecasting/*
utils/forecasting_motion.py
utils/forecasting_metrics.py
eval/eval_forecasting.py
model/forecasting.py
train/train_forecasting.py
```

这与路线图的“下一步从 P1 开始”一致。

## 论文结构可行性

推荐论文结构应是 IMRaD：

```text
Introduction: 双人未来预测中 interaction-aware inductive bias 的必要性。
Method: InterHuman fixed-window forecasting protocol、active vector、relation-aware predictor。
Results: repeat / independent / concat / relation-aware 和 ablation。
Discussion: relation-aware 何时有效、何时失败、deterministic forecasting 的边界。
```

路线图已经具备 Method 和 Results 的基本骨架，但 Literature Review 仍缺位。工程 P1/P2 不需要等待文献综述，但写论文前必须补齐 two-person motion forecasting、human interaction generation / prediction、relation modeling / graph / social forecasting 相关工作矩阵。

## 阶段可行性审阅

### P1 Dataset + Normalizer

可行。H5 shape、样本数、active vector 映射都有本地依据。

需要补强：

```text
1. 验收命令明确使用 regennet micromamba 环境。
2. 明确 normalizer 统计口径：T>=150 全序列所有帧，还是可采样 150-window 分布。
3. 明确是否保留 frozen H5 的 actor-frame-0 坐标；如果后续改成 window recenter，必须作为协议变更记录。
```

建议第一版保留当前 frozen H5 坐标，不在 P1 引入 window recenter，以免扩大实现面。

### P2 Metrics + Repeat Baseline

可行，而且必须优先做。`pred == target` sanity check 是防止后续模型结果无意义的关键。

需要补强：

```text
1. relative_orientation_error 中 acos 前必须 clamp，避免浮点误差产生 NaN。
2. predicted rot6d 需要通过 rotation_6d_to_matrix 正交化后再算相对角度。
3. inter_person_distance_consistency 当前是 root distance delta，适合作为第一版指标，但论文中不能把它说成完整的 interaction quality。
```

### P3 Independent / Concat Baselines

可行。Independent 和 concat 的定义清楚，concat 是 relation-aware 的关键强基线。

需要补强：

```text
1. 报告参数量和训练预算，避免 relation-aware 只是因为更大而赢。
2. Independent 必须严格隔离 A/B observation。
3. Concat 不显式构造 relation features，但它可以从 raw obs 学到关系；论文表述应是“显式关系归纳偏置是否优于原始拼接”，不是“concat 完全没有关系信息”。
```

P3 只要求优于 repeat baseline 合理；如果 concat 都赢不了 repeat，说明数据、normalizer、target 对齐或训练设置有根本问题。

### P4 Relation-Aware Predictor

方向可行，但这是论文成败点。

关系特征选择合理：

```text
relative root translation
relative velocity
root distance
relative root orientation
```

需要补强：

```text
1. 做 parameter-matched ablation，区分 relation feature 贡献和额外参数贡献。
2. 若 relation-aware 只在 future_mse 微弱提升，不能声称 interaction-aware 有效。
3. 若只赢 independent、不赢 concat，论文贡献不成立。
```

最低成立条件应保持路线图原判断：

```text
long_mse 优于 concat
至少一个 relation metric 优于 concat
```

### P5 Ablation + Paper Tables

方向可行，但路线图目前缺少重复实验要求。

必须新增：

```text
主结果至少 3 个 seed，表格报告 mean/std 或 mean±std。
所有模型使用同一 eval_forecasting 协议。
所有 checkpoint、args、metrics 可回溯。
```

如果单 seed 结果被当作主表，论文证据链偏弱。

### P6 Visualization + Qualitative Analysis

可行。P6 应作为解释和失败案例分析，不应替代主指标。

需要补强：

```text
1. 失败样本必须保留。
2. 可视化优先输出 npy 和 relation curves；render 卡住时不能阻塞主结果。
3. 如果可视化显示动作崩坏但 MSE 低，应回头补指标，而不是继续写定性描述。
```

## 主要风险

### 风险 1：确定性预测对多模态未来过强约束

150 帧窗口、30 帧观测、120 帧预测是高难度 deterministic setting。模型可能倾向平均动作，长期预测会自然变差。

处理方式：

```text
第一阶段只声称验证 relation-aware deterministic forecasting。
best-of-K、diversity、多模态生成放到后续扩展。
```

### 风险 2：concat baseline 可能已经足够强

concat 能看到两个人全部历史，因此它不是弱 baseline。relation-aware 的贡献必须是更好的归纳偏置，而不是“首次使用双人信息”。

处理方式：

```text
以 concat 为主对照。
加入参数匹配和 relation feature ablation。
不把赢 independent 当作核心贡献。
```

### 风险 3：SMPL reproduction 口径限制

当前 H5 是 SMPL / rot6d 表示，不是 SMPL-X 官方 InterHuman 口径。

处理方式：

```text
论文中明确这是 SMPL reproduction / first-stage protocol。
不要声称完全复现官方 InterHuman / InterGen 表示。
```

### 风险 4：缺少文献和 novelty 证据链

路线图能支撑实验，但不能单独支撑论文 novelty。

处理方式：

```text
P1/P2 可立即做。
P3/P4 期间并行补 literature matrix。
写论文前必须明确与 two-person forecasting / interaction generation / social prediction 的差异。
```

## 建议的路线图修订

不建议推翻原路线图。建议只追加以下约束：

```text
1. 所有 smoke/eval 命令显式使用 regennet micromamba 环境。
2. P2 relative orientation 指标加入 clamp 和 rot6d 正交化要求。
3. P3/P4 记录参数量、训练预算、seed。
4. P5 主表至少 3 seed，报告 mean/std。
5. P4/P5 明确 concat 是核心对照，relation-aware 必须赢 concat 才能支撑主张。
6. 文献综述并行推进，但不阻塞 P1/P2 工程闭环。
```

## 是否允许进入下一步

允许进入 P1：

```text
data_loaders/forecasting/interhuman.py
data_loaders/forecasting/tensors.py
utils/forecasting_motion.py
```

但 P1/P2 完成前仍不应实现 relation-aware model。当前最正确的下一步仍是：

```text
P1 active vector + dataset + normalizer smoke
P2 metrics + repeat baseline evaluator
```
