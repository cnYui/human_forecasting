# ForecastingCMDM 阶段 5 Action Classifier 实现计划

## 背景

本次依据 `20260622-223212-forecasting-cmdm-phase-5-action-consistency-classifier-design.md` 开始实现阶段 5。

阶段 5 的真实目标不是证明生成模型好，而是先验证：

```text
真实 NTU120 2P future40 是否包含可由轻量分类器识别的动作类别信息。
```

只有真实 future40 分类 gate 通过后，generated label consistency 才能作为可解释指标。

## 必须实现

新增：

```text
eval/action_consistency_classifier.py
```

脚本职责：

```text
1. 读取 NTULabelForecastDataset train/test。
2. 只使用 future40，不使用 obs_motion。
3. 用 train future40 计算 normalizer 并保存。
4. 训练 Temporal CNN action classifier。
5. 输出 real test top1/top5/balanced/per-class/handshaking/confusion metrics。
6. 可选读取 generated_future40.npy 和 metadata.json，输出 generated consistency debug 指标。
```

## 关键取舍

- 分类器放在 `eval/` 单脚本内，不新增 `model/` 公共模块；原因是当前只服务阶段 5 gate，后续复用再抽象。
- 第一版使用 Temporal CNN，不使用 Transformer；原因是数据量小，CNN 更快且参数更少。
- normalizer 只由 train split 计算，test/generated 不参与统计，避免评价污染。
- 类别不均衡先使用 `1/sqrt(count)` weighted CE，不做 oversampling；原因是部分类别样本极少，过采样更容易记忆。
- smoke 只验证代码闭环，不以 50-step accuracy 判定语义有效。

## 实现步骤

1. 新增脚本参数、保存目录、JSON/JSONL 工具函数。
2. 实现 `TemporalCNNActionClassifier` 和 residual block。
3. 实现 train future normalizer、class weights、训练循环、checkpoint 保存。
4. 实现 real test evaluation 和预测 JSONL。
5. 实现 generated consistency optional evaluation。
6. 更新 `AGENTS.md` 阶段记忆。
7. 运行 `py_compile`、import smoke、50-step smoke。
8. 新增 result 文档记录输出和风险边界。

## 验证命令

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile eval/action_consistency_classifier.py

/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
from eval.action_consistency_classifier import build_arg_parser, run_classifier_gate
print("action classifier import ok")
PY

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

## 不做

```text
不修改 train/train_label_forecasting_diffusion.py
不修改 sample/sample_label_forecasting_diffusion.py
不训练正式 ForecastingCMDMDecoder
不把 smoke classifier 指标写成论文结论
```
