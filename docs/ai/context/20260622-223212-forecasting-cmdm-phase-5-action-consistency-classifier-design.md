# ForecastingCMDMDecoder 阶段 5 Action Consistency Classifier 设计

## 参考上下文

本文依据：

```text
docs/ai/context/20260622-111938-forecasting-cmdm-decoder-implementation-commit-test-plan.md
docs/ai/context/20260622-114122-forecasting-cmdm-phase-a-data-gate-result.md
docs/ai/context/20260622-222109-forecasting-cmdm-phase-4-sampling-label-swap-result.md
data_loaders/forecasting/ntu_label.py
sample/sample_label_forecasting_diffusion.py
```

## 阶段命名

本文中的阶段 5 是阶段 4 之后的动作一致性分类器 gate，对应原提交计划的 Commit 5：

```text
动作一致性分类器 gate
```

它不是正式训练，也不是论文结论。它要回答一个前置问题：

```text
真实 NTU120 2P future40 本身是否包含足够动作类别信息？
```

只有该问题过关，才允许用分类器评价 generated future40 是否符合条件 label。

## 当前已满足条件

阶段 A 数据 gate：

```text
train kept_count = 1956
test kept_count = 1253
train/test 均覆盖 26 类
handshaking label 8 train/test 非零
future = [B,56,6,40]
action = [B,1]
```

类别分布边界：

```text
train min_class_count = 2
test min_class_count = 1
handshaking train = 170
handshaking test = 68
```

阶段 4 已生成 label swap 输出：

```text
results/forecasting/ntu120_label/p4_label_swap_smoke/generated_future40.npy
shape = [1,4,1,56,6,40]
labels = [2,5,8,17]
finite = true
```

## 阶段 5 目标

新增：

```text
eval/action_consistency_classifier.py
```

本阶段分两层：

### 5A: 分类器代码 smoke

目标：

```text
真实 future40 DataLoader 可训练
Temporal CNN classifier forward/backward 可跑
50-step smoke loss finite
classifier checkpoint 可保存
real test eval 可输出 metrics
```

5A 只证明代码闭环，不证明动作语义 gate 通过。

### 5B: 真实 future40 分类有效性 gate

目标：

```text
在真实 test future40 上 top1 accuracy 明显高于 random 1/26
handshaking subset 可被识别
per-class 指标可输出
```

只有 5B 通过，才运行或解释 generated consistency：

```text
generated future40 -> classifier -> predicted label 是否等于 conditioning label
```

## 本阶段不做

不做以下内容：

```text
不训练 ForecastingCMDMDecoder
不修改 sample/sample_label_forecasting_diffusion.py
不修改 train/train_label_forecasting_diffusion.py
不把分类器失败解释成生成模型失败
不把 smoke classifier 指标写进论文结论
不把 generated consistency 当作唯一生成质量指标
不做视频或 xyz MPJPE
```

原因：

```text
分类器本身必须先被验证。
如果真实 future40 都不可分类，generated consistency 指标没有可信解释。
```

## 代码位置

新增：

```text
eval/action_consistency_classifier.py
```

理由：

```text
这个脚本训练的是评价 gate，不是 ForecastingCMDM 主模型。
输出是 classifier checkpoint 和 consistency metrics，属于 eval/ 责任边界。
```

不新增：

```text
model/action_classifier.py
```

第一版分类器只服务本 gate，先放在单脚本内。若后续多个 eval 脚本复用，再抽公共模型文件。

## 输入数据协议

训练真实分类器：

```text
train_path = dataset/ntu120/smplx/conditioned/xsub.train.h5
test_path = dataset/ntu120/smplx/conditioned/xsub.test.h5
window_len = 60
obs_len = 20
pred_len = 40
input = future40 only
target = action label 0..25
```

输入张量：

```text
future = [B,56,6,40]
action = [B,1]
```

分类器不看：

```text
obs_motion
sample_id
generated conditioning label 之外的信息
```

原因：

```text
阶段 5 要判断 future40 里是否有动作类别信息。
如果分类器看 obs_motion，会污染“生成 future40 是否符合 label”的评价。
```

## Generated Consistency 输入

可选输入：

```text
generated_dir = results/forecasting/ntu120_label/p4_label_swap_smoke
```

读取：

```text
generated_future40.npy = [C,L,R,56,6,40]
metadata.json
```

metadata 必须包含：

```text
labels = [2,5,8,17]
generated_shape
checkpoint
```

生成一致性计算：

```text
flatten generated -> [C*L*R,56,6,40]
condition_labels -> [C*L*R]
classifier logits -> predicted labels
consistency_acc = mean(pred == condition_label)
per_label_consistency_acc
```

若 5B 不通过：

```text
仍可输出 generated predictions
但必须标记 valid_for_claim = false
不能把 consistency_acc 作为结论
```

## 模型选择

第一版使用小型 Temporal CNN，不使用 Transformer。

理由：

```text
数据量只有 train 1956 / test 1253。
Temporal CNN 更少参数、更快，适合作为 gate。
Transformer 分类器容易在小数据上过拟合，且调参成本更高。
```

输入处理：

```text
future [B,56,6,40]
flatten joints/features -> [B,336,40]
train normalizer 标准化
Temporal CNN
global average pooling over time
Linear -> logits [B,26]
```

推荐结构：

```text
input_proj: Conv1d(336, hidden_dim, kernel_size=1)
residual blocks x num_blocks:
  GroupNorm
  GELU
  Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
  Dropout
  GroupNorm
  GELU
  Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
global_mean_pool over T
Linear(hidden_dim, 26)
```

默认 smoke 配置：

```text
hidden_dim = 128
num_blocks = 2
dropout = 0.1
```

正式 gate 可升为：

```text
hidden_dim = 256
num_blocks = 3
dropout = 0.2
```

## Normalization

分类器训练必须保存 train split normalizer：

```text
mean = train future40 mean over N,T
std = train future40 std over N,T
shape = [1,56,6,1] 或 [1,336,1]
```

保存：

```text
normalizer.pt
```

规则：

```text
训练、真实 test、generated future40 都使用同一 normalizer。
std clamp_min = 1e-6
```

不使用 generated 数据更新 normalizer。

## 类别不均衡处理

NTU120 length60 后类别严重不均衡：

```text
train min_class_count = 2
test min_class_count = 1
```

第一版训练使用：

```text
weighted cross entropy
class_weight = 1 / sqrt(class_count)
class_weight normalize 到 mean=1
```

不建议第一版使用 aggressive oversampling。

原因：

```text
部分类别只有 2 条 train，过采样会极易记忆。
weighted CE 足够作为第一版 gate。
```

评价必须同时输出：

```text
top1_acc micro
top5_acc micro
balanced_acc macro
per_class_acc
per_class_count
handshaking_acc label 8
confusion_matrix
```

宏平均只作参考，因为 test 最小类别只有 1 条，波动很大。

## Gate 阈值

random baseline：

```text
top1_random = 1 / 26 = 0.03846
top5_random = 5 / 26 = 0.19231
```

5A smoke 通过标准：

```text
50-step 训练 loss finite
checkpoint 保存
real test metrics 文件可读
```

5A 不要求 accuracy 达标。

5B 真实分类有效性建议阈值：

```text
top1_acc >= 0.15
top5_acc >= 0.50
handshaking_acc >= 0.20
```

解释：

```text
top1 0.15 约为 random 的 3.9 倍。
top5 0.50 明显高于 random 0.192。
handshaking test 有 68 条，足够作为 label 8 的子集 sanity check。
```

若未达阈值：

```text
标记 classifier_gate_pass = false
不使用 generated consistency 作为主指标
返回检查窗口长度、动作标签、模型容量或是否需要 xyz/骨架特征
```

## 训练流程

伪代码：

```python
fixseed(args.seed)
prepare_save_dir(args)
train_dataset = NTULabelForecastDataset(train_path, split="train")
test_dataset = NTULabelForecastDataset(test_path, split="test")
normalizer = compute_train_future_normalizer(train_dataset)
model = TemporalCNNActionClassifier(...)
optimizer = AdamW(...)
for step in range(num_steps):
    batch = next(train_loader)
    future = normalize(batch["future"])
    label = batch["action"].view(-1)
    logits = model(future)
    loss = weighted_cross_entropy(logits, label)
    backward / optimizer.step
    log train loss / train acc
    optional eval_interval
save classifier checkpoint
evaluate real test
if generated_dir provided:
    evaluate generated consistency
```

训练数据增强：

```text
train dataset 继续使用 random crop
test dataset 使用 center crop
```

理由：

```text
random crop 可作为轻量时间增强。
test center crop 保持可复现。
```

## 输出设计

保存目录：

```text
save/forecasting/ntu120_label/action_classifier_smoke
```

文件：

```text
args.json
train_log.jsonl
classifier_model.pt
normalizer.pt
real_test_metrics.json
real_test_predictions.jsonl
generated_consistency.json    # optional
generated_predictions.jsonl   # optional
confusion_matrix.npy
```

### classifier_model.pt

内容：

```text
model_state_dict
model_config
num_classes = 26
step
seed
train_path
test_path
normalizer_path
real_test_metrics
created_at
```

### real_test_metrics.json

至少包含：

```text
top1_acc
top5_acc
balanced_acc
per_class_acc
per_class_count
handshaking_acc
top1_random
top5_random
classifier_gate_pass
gate_thresholds
loss
num_test_samples
```

### generated_consistency.json

仅在 `--generated_dir` 提供时输出：

```text
generated_dir
classifier_checkpoint
classifier_gate_pass
valid_for_claim
labels
consistency_acc
per_label_consistency_acc
predicted_label_counts
condition_label_counts
num_generated_samples
```

规则：

```text
valid_for_claim = classifier_gate_pass
```

如果 `classifier_gate_pass=false`，该文件只能作为 debug 产物。

## CLI 设计

第一版参数：

```text
--train_path
--test_path
--save_dir
--generated_dir
--window_len 60
--obs_len 20
--pred_len 40
--batch_size 8
--eval_batch_size 32
--num_steps 50
--eval_interval 0
--save_interval 50
--hidden_dim 128
--num_blocks 2
--dropout 0.1
--lr 3e-4
--weight_decay 1e-4
--clip_grad_norm 1.0
--num_workers 0
--seed 0
--overwrite
```

5A smoke 命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m eval.action_consistency_classifier \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/action_classifier_smoke \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 8 --eval_batch_size 32 \
  --num_steps 50 --save_interval 50 --eval_interval 0 \
  --hidden_dim 128 --num_blocks 2 --dropout 0.1 \
  --seed 0 --overwrite
```

5B gate 命令建议：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m eval.action_consistency_classifier \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/action_classifier_gate_h256_b3_s0 \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 16 --eval_batch_size 64 \
  --num_steps 2000 --save_interval 500 --eval_interval 500 \
  --hidden_dim 256 --num_blocks 3 --dropout 0.2 \
  --lr 3e-4 --weight_decay 1e-4 \
  --seed 0 --overwrite
```

generated consistency 命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m eval.action_consistency_classifier \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --generated_dir results/forecasting/ntu120_label/p4_label_swap_smoke \
  --save_dir save/forecasting/ntu120_label/action_classifier_gate_h256_b3_s0 \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 16 --eval_batch_size 64 \
  --num_steps 2000 --save_interval 500 --eval_interval 500 \
  --hidden_dim 256 --num_blocks 3 --dropout 0.2 \
  --lr 3e-4 --weight_decay 1e-4 \
  --seed 0 --overwrite
```

第一版可不支持“只加载已有 classifier checkpoint 评估 generated”。如果实现时很简单，可以加：

```text
--classifier_checkpoint
--eval_only
```

但不要因此拖慢 5A smoke。

## 验证顺序

### 1. 静态编译

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile eval/action_consistency_classifier.py
```

### 2. Import smoke

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
from eval.action_consistency_classifier import build_arg_parser, run_classifier_gate
print("action classifier import ok")
PY
```

### 3. 5A smoke

运行 50-step smoke。

通过标准：

```text
loss finite
classifier_model.pt 存在
normalizer.pt 存在
real_test_metrics.json 存在且可读
confusion_matrix.npy 存在
real_test_predictions.jsonl 可读
```

### 4. 5B gate

运行较长训练，例如 2000-step。

通过标准：

```text
top1_acc >= 0.15
top5_acc >= 0.50
handshaking_acc >= 0.20
classifier_gate_pass = true
```

如果不过：

```text
不进入 generated consistency 主结论
写 result 文档说明 classifier gate blocked
```

### 5. Generated consistency

仅当 5B 通过时检查：

```text
generated_consistency.json 存在
valid_for_claim = true
consistency_acc 可读
per_label_consistency_acc 可读
```

## 风险与处理

### 风险 1: 类别极不均衡

处理：

```text
使用 weighted CE
报告 micro/macro/per-class
不只看 overall top1
handshaking 单独报告
```

### 风险 2: 50-step smoke accuracy 低

处理：

```text
这是预期可能发生的。
5A smoke 只验证代码闭环，不作为 classifier_gate_pass。
```

### 风险 3: 5B 仍低于阈值

处理：

```text
暂停 generated consistency 结论。
检查 future40 是否足够包含动作信息。
尝试更强分类器或特征，但必须新增设计。
```

### 风险 4: 分类器过拟合

处理：

```text
只以 test metrics 作为 gate。
保存 train/test 曲线。
如果 train 高 test 低，不通过 gate。
```

### 风险 5: generated consistency 被误读

处理：

```text
generated_consistency.json 写 valid_for_claim。
result 文档必须明确 consistency 依赖 classifier_gate_pass。
```

## 阶段 5 退出条件

最小退出条件 5A：

```text
eval/action_consistency_classifier.py 存在
py_compile 通过
import smoke 通过
50-step smoke 通过
classifier checkpoint / normalizer / real_test_metrics 输出完整
新增阶段 5 result 文档
```

完整评价 gate 5B：

```text
real future40 classifier_gate_pass = true
top1/top5/handshaking 达阈值
generated consistency 可运行
generated_consistency.json valid_for_claim = true
```

如果只完成 5A：

```text
不能进入论文结果表述
不能把 generated consistency 当主指标
```

## 下一阶段边界

阶段 5B 通过后，才考虑：

```text
seed0 正式 5000-step ForecastingCMDMDecoder 训练
阶段 4 sampling 在正式 checkpoint 上重跑
generated label consistency
per-class / handshaking subset 报告
no_action / no_obs_tokens / no_cfg ablation
```

仍需保持论文表述边界：

```text
不能声称动作语义控制已证明，除非 real classifier gate 和 generated consistency 都过。
不能声称模型优于 baseline，除非正式同口径实验完成。
```
