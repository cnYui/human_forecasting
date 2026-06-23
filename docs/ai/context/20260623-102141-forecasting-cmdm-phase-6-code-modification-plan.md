# ForecastingCMDM 阶段 6 代码修改计划

## 目标

参考 `docs/ai/context/20260623-101205-forecasting-cmdm-phase-6-formal-training-design.md`，补齐阶段 6 正式训练闭环的最小代码缺口。

## 必须修改

```text
sample/sample_label_forecasting_diffusion.py
scripts/check_phase6_formal_outputs.py
AGENTS.md
```

## 设计判断

阶段 6 训练入口 `train/train_label_forecasting_diffusion.py` 已满足核心要求：

```text
5000-step num_steps/save_interval 协议由 CLI 控制
checkpoint 保存 step/model_type/model_config/diffusion_config/train_protocol
train_log.jsonl 逐 step 写入
resume 使用 strict config 校验
eval_interval 强制为 0
```

阶段 6 采样入口已有 strict checkpoint load、DDIM、label swap、metadata/metrics/summary 输出，但当前 `metrics.json` 固定写 `smoke_only=true`，与阶段 6 “不是 smoke”冲突。因此新增 CLI 语义开关，默认保持旧 smoke 行为，阶段 6 命令显式传入正式模式。

阶段 6 设计提到检查脚本“可以新增但不是必须”。这里新增轻量脚本，原因是正式训练耗时长，训练后用一个确定性检查脚本比手写一次性 Python 更可复现。

## 修改范围

1. `sample/sample_label_forecasting_diffusion.py`
   - 新增 `--run_name`，写入 `metrics.json` 和 `metadata.json`。
   - 新增 `--formal`，用于让 `metrics.smoke_only=false`。
   - 保持默认 `smoke_only=true`，避免破坏阶段 4 smoke 记录。

2. `scripts/check_phase6_formal_outputs.py`
   - 检查 checkpoint / optimizer 是否存在。
   - 检查 checkpoint step、model_type、train_protocol。
   - 检查 train_log 行数、step 连续、loss finite。
   - 检查 generated 输出 shape、finite、metadata checkpoint、metrics finite。
   - 检查 label swap 非完全相同。
   - 检查 generated_consistency.json 存在并含 `valid_for_claim`。
   - 输出 JSON 摘要，失败时非零退出。

3. `AGENTS.md`
   - 增加阶段 6 代码准备状态，提示正式采样需要 `--formal`。

## 不做

```text
不启动 5000-step 正式训练
不改 train/train_mdm.py
不改 train/training_loop.py
不改 ForecastingCMDMDecoder 架构
不把 seed0 结果写成论文结论
```

## 验证

```text
python -m py_compile sample/sample_label_forecasting_diffusion.py scripts/check_phase6_formal_outputs.py
python -m sample.sample_label_forecasting_diffusion --help
python scripts/check_phase6_formal_outputs.py --help
```
