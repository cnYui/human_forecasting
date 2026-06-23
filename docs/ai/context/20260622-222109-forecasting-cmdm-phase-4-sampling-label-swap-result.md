# ForecastingCMDMDecoder 阶段 4 Sampling / Label Swap 结果

## 结论

阶段 4 sampling / label swap gate 已完成并通过。

本阶段新增：

```text
sample/sample_label_forecasting_diffusion.py
docs/ai/context/20260622-222109-forecasting-cmdm-phase-4-sampling-label-swap-result.md
```

本阶段更新：

```text
AGENTS.md
```

本阶段未修改：

```text
train/train_label_forecasting_diffusion.py
train/train_mdm.py
train/training_loop.py
model/forecasting_cmdm.py
model/cmdm.py
data_loaders/forecasting/ntu_label.py
diffusion/gaussian_diffusion.py
eval/*
```

## 实现内容

新增采样入口：

```text
sample/sample_label_forecasting_diffusion.py
```

核心能力：

```text
加载阶段 C dict checkpoint
校验 model_type / train_protocol
用 checkpoint model_config 重建 ForecastingCMDMDecoder
strict=True 加载 model_state_dict
用 checkpoint diffusion_config 重建 SpacedDiffusion
从 NTULabelForecastDataset(test) 读取 obs20 / real future40
same obs20 + labels [2,5,8,17] label swap
同一 case/repetition 下跨 label 共享初始 noise
p_sample_loop 采样
可选 DDIM 采样
保存 generated_future40.npy / obs_motion.npy / real_future40.npy
保存 metadata.json / metrics.json / label_swap_summary.json
```

关键设计保持：

```text
不调用 utils.model_util.load_model_wo_clip
不调用 create_model_and_diffusion
不调用 diffusion.training_losses
不初始化 rot2xyz
不做 gaussian_filter1d smoothing
clip_denoised=False
guidance_scale=1.0 时不包 CFG wrapper
metrics.json 标记 smoke_only=true
```

## 运行环境

```text
python_executable = /home/rpartx3080/.local/micromamba/envs/regennet/bin/python
device = cuda
checkpoint = save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt
checkpoint_step = 2
```

## 验证结果

### 1. 静态编译

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile sample/sample_label_forecasting_diffusion.py
```

结果：

```text
exit_code = 0
```

### 2. Import smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
from sample.sample_label_forecasting_diffusion import build_arg_parser, run_sampling
print('sample entry import ok')
PY
```

结果：

```text
sample entry import ok
exit_code = 0
```

### 3. p_sample_loop smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m sample.sample_label_forecasting_diffusion \
  --checkpoint save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/p4_label_swap_smoke \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --labels 2 5 8 17 \
  --guidance_scale 1.0 \
  --batch_size 4 --num_cases 1 --num_repetitions 1 \
  --sample_index 0 --seed 0 --overwrite
```

结果：

```text
exit_code = 0
generated_shape = [1, 4, 1, 56, 6, 40]
finite = True
pass_non_identical_check = True
rot_mse_mean = 0.419038
max_pair_diff = 0.10049816220998764
```

输出文件：

```text
generated_future40.npy 215168
obs_motion.npy 27008
real_future40.npy 53888
metadata.json 2768
metrics.json 1008
label_swap_summary.json 2147
```

输出目录：

```text
results/forecasting/ntu120_label/p4_label_swap_smoke
```

### 4. 输出文件检查

检查内容：

```text
generated_future40.npy shape = [1,4,1,56,6,40]
obs_motion.npy shape = [1,56,6,20]
real_future40.npy shape = [1,56,6,40]
generated finite = True
metadata.labels = [2,5,8,17]
metrics.finite = True
metrics.smoke_only = True
label_swap_summary.pass_non_identical_check = True
```

结果：

```text
phase4 outputs ok
exit_code = 0
```

### 5. DDIM50 smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m sample.sample_label_forecasting_diffusion \
  --checkpoint save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/p4_label_swap_smoke_ddim50 \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --labels 2 5 8 17 \
  --guidance_scale 1.0 \
  --batch_size 4 --num_cases 1 --num_repetitions 1 \
  --sample_index 0 --seed 0 \
  --use_ddim --timestep_respacing ddim50 --overwrite
```

结果：

```text
exit_code = 0
generated_shape = [1, 4, 1, 56, 6, 40]
finite = True
pass_non_identical_check = True
rot_mse_mean = 0.418891
max_pair_diff = 0.1026286780834198
```

输出目录：

```text
results/forecasting/ntu120_label/p4_label_swap_smoke_ddim50
```

## Label Swap 摘要

p_sample_loop label pair 最大差异：

```text
2 vs 5  max_abs_diff = 0.06662173569202423
2 vs 8  max_abs_diff = 0.10049816220998764
2 vs 17 max_abs_diff = 0.07114972174167633
5 vs 8  max_abs_diff = 0.0892176553606987
5 vs 17 max_abs_diff = 0.07361409068107605
8 vs 17 max_abs_diff = 0.10031171143054962
```

解释边界：

```text
该检查只证明不同 label 路径没有完全塌成同一输出。
不能把该数值解释为动作语义正确或生成质量好。
```

## 退出条件核对

已满足：

```text
sample/sample_label_forecasting_diffusion.py 存在
py_compile 通过
import smoke 通过
p_sample_loop smoke 通过
输出文件检查通过
generated_future40.npy 存在且 shape = [1,4,1,56,6,40]
obs_motion.npy 存在且 shape = [1,56,6,20]
real_future40.npy 存在且 shape = [1,56,6,40]
metadata.json / metrics.json / label_swap_summary.json 可读
generated finite
不同 label 输出不完全相同
metrics.json smoke_only = true
未修改 train/train_mdm.py
未修改 train/train_label_forecasting_diffusion.py 的训练语义
未修改 model/cmdm.py
未修改 diffusion/gaussian_diffusion.py
```

## 下一阶段

允许进入动作一致性分类器 gate：

```text
eval/action_consistency_classifier.py
真实 future40 分类器 gate
generated future40 label consistency
per-class / handshaking subset 指标
```

仍不建议直接进入：

```text
论文结果表述
正式 5000-step 训练
大规模 ablation
```

原因：

```text
当前只证明 checkpoint sampling 和 label swap 文件闭环可靠。
动作语义是否可评估，需要先证明真实 future40 分类器明显高于 random 1/26。
```
