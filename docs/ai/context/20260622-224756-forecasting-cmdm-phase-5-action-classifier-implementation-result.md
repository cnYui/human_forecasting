# ForecastingCMDM 阶段 5 Action Classifier 实现结果

## 本次改动

新增：

```text
eval/action_consistency_classifier.py
docs/ai/context/20260622-224043-forecasting-cmdm-phase-5-action-classifier-implementation-plan.md
```

更新：

```text
AGENTS.md
```

实现内容：

```text
1. Temporal CNN action classifier。
2. train future40 normalizer 保存到 normalizer.pt。
3. 1/sqrt(class_count) weighted CE。
4. real test top1/top5/balanced/per-class/handshaking/confusion evaluation。
5. checkpoint、args、train_log、predictions、metrics 落盘。
6. 可选 generated_future40 consistency debug evaluation。
```

未修改：

```text
train/train_label_forecasting_diffusion.py
sample/sample_label_forecasting_diffusion.py
model/forecasting_cmdm.py
```

## 验证结果

### 静态检查

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile eval/action_consistency_classifier.py
```

结果：

```text
通过
```

### Import smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
from eval.action_consistency_classifier import build_arg_parser, run_classifier_gate
print("action classifier import ok")
PY
```

结果：

```text
action classifier import ok
```

### 5A 50-step smoke

命令：

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

输出目录：

```text
save/forecasting/ntu120_label/action_classifier_smoke
```

输出文件：

```text
args.json
classifier_model.pt
confusion_matrix.npy
normalizer.pt
real_test_metrics.json
real_test_predictions.jsonl
train_log.jsonl
```

文件核对：

```text
train_log.jsonl = 50 行
real_test_predictions.jsonl = 1253 行
classifier checkpoint step = 50
normalizer mean/std shape = [1,56,6,1]
normalizer count = 78240
confusion_matrix shape = [26,26]
confusion_matrix sum = 1253
```

真实 test 指标：

```text
top1_acc = 0.3343974461292897
top5_acc = 0.6624102154828412
balanced_acc = 0.16697809135415922
handshaking_acc = 0.4264705882352941
classifier_gate_pass = true
```

解释边界：

```text
虽然 50-step smoke 已超过当前 gate 阈值，但它仍是 5A smoke 运行。
正式 5B 仍建议用更长配置复跑，例如 2000-step h256/b3。
per-class acc 仍有多个类别为 0，论文表述不能只看 micro top1/top5。
```

### Generated consistency 代码路径 smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m eval.action_consistency_classifier \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --generated_dir results/forecasting/ntu120_label/p4_label_swap_smoke \
  --save_dir save/forecasting/ntu120_label/action_classifier_generated_path_smoke \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 8 --eval_batch_size 32 \
  --num_steps 1 --save_interval 1 --eval_interval 0 \
  --hidden_dim 128 --num_blocks 2 --dropout 0.1 \
  --seed 0 --overwrite
```

结果：

```text
generated_consistency.json 存在
generated_predictions.jsonl = 4 行
real classifier gate_pass = false
generated valid_for_claim = false
```

解释：

```text
该运行只验证 generated consistency 代码路径能读 metadata 和 generated_future40。
它不是 generated consistency 指标结论。
```

## 后续建议

正式 5B 命令仍使用设计文档中的长训练配置：

```text
hidden_dim=256
num_blocks=3
dropout=0.2
num_steps=2000
batch_size=16
eval_batch_size=64
```

只有正式 5B 的真实 future40 分类 gate 通过后，才使用 generated consistency 作为可解释指标。
