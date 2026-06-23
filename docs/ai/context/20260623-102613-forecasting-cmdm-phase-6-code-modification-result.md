# ForecastingCMDM 阶段 6 代码修改结果

## 结论

阶段 6 正式训练闭环的代码准备已完成。

本次新增：

```text
scripts/check_phase6_formal_outputs.py
docs/ai/context/20260623-102141-forecasting-cmdm-phase-6-code-modification-plan.md
docs/ai/context/20260623-102613-forecasting-cmdm-phase-6-code-modification-result.md
```

本次更新：

```text
sample/sample_label_forecasting_diffusion.py
AGENTS.md
```

## 实现内容

### 采样输出语义

`sample/sample_label_forecasting_diffusion.py` 新增：

```text
--formal
--run_name
```

默认仍保持早期 smoke 行为：

```text
metrics.smoke_only = true
metadata.formal = false
```

阶段 6 正式采样必须传：

```text
--formal --run_name phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap
```

传入后输出：

```text
metrics.smoke_only = false
metadata.formal = true
```

### 阶段 6 检查脚本

新增 `scripts/check_phase6_formal_outputs.py`，默认参数对齐阶段 6 设计文档。

检查内容：

```text
model000005000.pt / opt000005000.pt 存在
checkpoint.step = 5000
checkpoint.model_type = forecasting_cmdm_decoder
train_protocol.mean_type = START_X
train_protocol.loss_type = MSE
train_log.jsonl 行数为 5000 且 step 连续
train_loss / rot_mse / loss 为 finite
generated_future40.npy shape = [8,4,2,56,6,40]
obs_motion.npy shape = [8,56,6,20]
real_future40.npy shape = [8,56,6,40]
metrics.finite = true
metrics.smoke_only = false
metadata.formal = true
metadata.checkpoint 指向正式 checkpoint
label_swap_summary.pass_non_identical_check = true
generated_consistency.json 存在且含 valid_for_claim
```

脚本不强制 `generated_consistency.valid_for_claim=true`，因为该字段必须由真实 action classifier gate 决定；脚本只核对字段存在并在 JSON 摘要中报告。

## 验证

静态检查：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile sample/sample_label_forecasting_diffusion.py scripts/check_phase6_formal_outputs.py
```

结果：

```text
通过
```

CLI 验证：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m sample.sample_label_forecasting_diffusion --help

/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  scripts/check_phase6_formal_outputs.py --help
```

结果：

```text
均通过，采样入口可见 --formal / --run_name。
```

检查脚本最小通过路径：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  scripts/check_phase6_formal_outputs.py \
  --save_dir /tmp/regennet_phase6_check_smoke/save \
  --generated_dir /tmp/regennet_phase6_check_smoke/results \
  --classifier_dir /tmp/regennet_phase6_check_smoke/classifier \
  --expected_step 3 --num_cases 1 --num_repetitions 1 --labels 2 8
```

结果：

```text
pass = true
errors = []
```

## 正式运行注意

阶段 6 设计文档中的采样命令需要追加：

```text
--formal --run_name phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap
```

正式训练、正式采样和动作一致性复评全部完成后运行：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  scripts/check_phase6_formal_outputs.py
```

通过后再新增阶段 6 formal training result 文档。
