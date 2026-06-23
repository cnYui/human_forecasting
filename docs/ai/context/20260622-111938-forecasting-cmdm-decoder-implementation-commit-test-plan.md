# ForecastingCMDMDecoder 实现提交与测试计划

## 参考上下文

本计划以以下文档为准：

```text
docs/ai/context/20260622-103022-ntu120-length-distribution-for-forecasting.md
docs/ai/context/20260622-104618-ntu120-label-forecasting-length60-update-plan.md
docs/ai/context/20260622-104643-ntu120-label-conditioned-regennet-smoke-design-v2-length60.md
docs/ai/context/20260622-104723-ntu120-label-conditioned-regennet-final-design-v2-length60.md
docs/ai/context/20260622-105124-forecasting-cmdm-architecture-design.md
docs/ai/context/20260622-105749-forecasting-cmdm-final-architecture-plan.md
docs/ai/context/20260622-105946-forecasting-cmdm-final-target-architecture-v3.md
```

当前 `docs/ai/content/` 未发现 2026-06-22 当天文档，只有：

```text
docs/ai/content/20260621-172802-agents-entry-content.md
```

## 最终执行基线

```text
dataset = NTU120 2P
data = dataset/ntu120/smplx/conditioned/xsub.train.h5 / xsub.test.h5
window_len = 60
obs_len = 20
pred_len = 40
condition = obs20 + action label
target = future40
model = ForecastingCMDMDecoder
representation = [B,56,6,T]
num_actions = 26
handshaking = label 8 / A009
```

关键原则：

```text
Smoke 也使用最终 Decoder 架构，只缩小 latent_dim、层数和 batch。
旧 Encoder-only 方案只作为 ablation/debug，不作为主线实现。
CFG uncond 只 mask action，不清零 obs_motion。
obs 使用窗口绝对位置 0..19，future 使用 20..59。
```

## 推荐 Git 提交拆分

### Commit 1: NTU label forecasting dataset 与数据 gate

建议改动：

```text
新增 data_loaders/forecasting/ntu_label.py
扩展 data_loaders/forecasting/__init__.py
新增或复用 collate，输出 obs_motion/future/action/mask/meta
新增轻量数据检查入口，例如 scripts/check_ntu_label_forecasting_data.py
```

提交边界：

```text
只处理 H5 读取、A001-A026 标签解析、T>=60 过滤、train random crop、test center crop、类别统计和 finite 检查。
不放模型代码。
```

必须测试：

```bash
python -m scripts.check_ntu_label_forecasting_data \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --window_len 60 --obs_len 20 --pred_len 40
```

通过标准：

```text
train/test 均覆盖 26 类
handshaking train/test 非零
batch obs_motion = [B,56,6,20]
batch future = [B,56,6,40]
action = [B,1]
mask = [B,1,1,40]
无 NaN/Inf
```

### Commit 2: ForecastingCMDMDecoder 模型与 CFG wrapper

建议改动：

```text
新增 model/forecasting_cmdm.py
实现 ForecastingCMDMDecoder
实现 ForecastingClassifierFreeSampleModel
从 model/cmdm.py 复用 InputProcess / OutputProcess / PositionalEncoding / TimestepEmbedder / EmbedAction
```

提交边界：

```text
只实现模型 forward、shape guard、config/count_parameters。
不接训练循环。
不抽 cmdm_components，除非 import 出现明确副作用或循环依赖。
```

必须测试：

```bash
python - <<'PY'
import torch
from model.forecasting_cmdm import ForecastingCMDMDecoder, ForecastingClassifierFreeSampleModel

model = ForecastingCMDMDecoder(
    njoints=56, nfeats=6, num_actions=26, obs_len=20, pred_len=40,
    latent_dim=128, obs_encoder_layers=1, decoder_layers=2,
    num_heads=4, ff_size=512, dropout=0.1, cond_mask_prob=0.1,
)
x = torch.randn(1, 56, 6, 40)
y = {"obs_motion": torch.randn(1, 56, 6, 20), "action": torch.tensor([[8]])}
t = torch.tensor([10])
out = model(x, t, y)
assert out.shape == x.shape
loss = out.square().mean()
loss.backward()
cfg = ForecastingClassifierFreeSampleModel(model, guidance_scale=2.0)
out_cfg = cfg(x.detach(), t, y)
assert out_cfg.shape == x.shape
assert torch.isfinite(out_cfg).all()
print("ok")
PY
```

通过标准：

```text
forward shape 正确
loss/backward finite
cond/uncond CFG forward finite
action token 单独进入 memory
obs token memory 未被 uncond 清零
```

### Commit 3: Diffusion 训练入口与 2-step smoke

建议改动：

```text
新增 train/train_label_forecasting_diffusion.py
复用 diffusion q_sample / schedule_sampler 思路
实现 START_X clean future 预测目标
实现 rot_mse / velocity_mse / root_translation_mse / relative_root_mse
保存 args.json、train_log.jsonl、model*.pt、opt*.pt
支持 resume_checkpoint、grad_accum_steps、overwrite
```

提交边界：

```text
只跑训练闭环和 checkpoint。
采样、label swap 和正式评估放到后续提交。
```

必须测试：

```bash
python -m train.train_label_forecasting_diffusion \
  --dataset ntu120_2p \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke \
  --model_type forecasting_cmdm_decoder \
  --body_model smplx \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 1 --eval_batch_size 1 \
  --num_steps 2 --save_interval 2 --eval_interval 0 \
  --latent_dim 128 --decoder_layers 2 --obs_encoder_layers 1 \
  --num_heads 4 --ff_size 512 --cond_mask_prob 0.1 \
  --num_workers 0 --seed 0 --overwrite
```

通过标准：

```text
2 step 训练完成
loss finite
checkpoint 和 optimizer state 可保存
resume 能从 step=2 或指定 checkpoint 正确加载
```

### Commit 4: 采样、label swap 与基础预测评估

建议改动：

```text
新增 sample/sample_label_forecasting_diffusion.py 或 eval/eval_label_forecasting_diffusion.py
实现 checkpoint load
实现 p_sample_loop / DDIM 采样入口
实现 same obs20 + labels [2,5,8,17] label swap
保存 generated_future40.npy、metadata.json、metrics.json、label_swap_summary.json
可选复用 SMPL-X xyz 转换计算 MPJPE / joint MSE
```

提交边界：

```text
先保证生成结果 finite 和文件完整。
动作分类一致性不放在本提交。
```

必须测试：

```bash
python -m sample.sample_label_forecasting_diffusion \
  --checkpoint save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt \
  --data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir results/forecasting/ntu120_label/p3_label_swap_smoke \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --labels 2 5 8 17 \
  --guidance_scale 1.0 \
  --batch_size 1 --num_samples 1 --seed 0 --overwrite
```

通过标准：

```text
每个 label 都有输出
输出 shape = [56,6,40] 或 [N,56,6,40]
无 NaN/Inf
不同 label 输出不完全相同
metrics 文件可读
```

### Commit 5: 动作一致性分类器 gate

建议改动：

```text
新增 eval/action_consistency_classifier.py
使用 real future40 训练 temporal transformer / temporal CNN 分类器
保存 classifier checkpoint 与 real test accuracy
把 generated future40 接入 classifier consistency 评估
```

提交边界：

```text
只建立“真实 future40 可分类”的评价 gate。
不把分类器失败的指标用于证明生成质量。
```

必须测试：

```bash
python -m eval.action_consistency_classifier \
  --train_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --test_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/action_classifier_smoke \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 8 --num_steps 50 --seed 0 --overwrite
```

通过标准：

```text
real test accuracy 明显高于 random 1/26
handshaking subset 可被识别
生成结果可输出 classifier_consistency_acc / per_class_consistency_acc
```

### Commit 6: 正式训练配置、ablation 开关与报告脚本

建议改动：

```text
补齐 no_action / no_obs_tokens / encoder_only / no_cfg ablation 参数
整理正式训练命令或 scripts
汇总 metrics_test.json、metrics_per_class.json、label_consistency.json、label_swap_summary.json
```

提交边界：

```text
只提交代码、脚本和小型 JSON 配置。
不提交 save/ 和 results/ 下的大 checkpoint、npy、mp4。
```

正式训练测试：

```bash
python -m train.train_label_forecasting_diffusion \
  --dataset ntu120_2p \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/forecasting_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000 \
  --model_type forecasting_cmdm_decoder \
  --body_model smplx \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 4 --grad_accum_steps 4 --eval_batch_size 4 \
  --num_steps 5000 --save_interval 1000 --eval_interval 1000 \
  --latent_dim 256 --decoder_layers 4 --obs_encoder_layers 2 \
  --num_heads 4 --ff_size 1024 --cond_mask_prob 0.1 \
  --lr 1e-4 --weight_decay 1e-4 \
  --num_workers 0 --seed 0
```

3080 OOM 时只降级为：

```text
latent_dim = 192
decoder_layers = 3
obs_encoder_layers = 1
batch_size = 2
grad_accum_steps = 8
```

不能回退为 Encoder-only。

## 阶段安排

### 阶段 A: 工程 gate

对应 Commit 1-3。

目标：

```text
数据可读
模型可 forward/backward
diffusion 2-step 可训练并保存
```

只有阶段 A 通过后，才进入采样。

### 阶段 B: 生成闭环

对应 Commit 4。

目标：

```text
从 checkpoint 采样 future40
label swap 输出完整
基础 metrics 可计算
```

先不要求语义正确，只要求闭环可靠。

### 阶段 C: 标签语义评估

对应 Commit 5。

目标：

```text
真实 future40 分类器先过 gate
再用分类器评估生成动作标签一致性
```

如果真实数据分类器不过关，暂停使用分类一致性作为主结论。

### 阶段 D: 正式训练与 ablation

对应 Commit 6。

目标：

```text
seed0 5000-step
label swap + handshaking subset
no_action / no_obs_tokens / encoder_only / no_cfg ablation
```

只有 seed0 和 ablation 有合理结果后，再启动 seed=0,1,2 三种子。

## 推荐测试顺序

```text
1. 数据 gate
2. 模型纯随机输入 forward/backward
3. dataset batch -> model forward/backward
4. 2-step train
5. checkpoint resume
6. checkpoint sample
7. label swap smoke
8. real future40 action classifier gate
9. seed0 formal train
10. formal eval + label consistency
11. ablation
12. 3-seed
```

## 不建议做的事

```text
不要继续扩展旧 Encoder-only 主线。
不要把原始 train/train_mdm.py 当作 20->40 forecasting 入口。
不要把 obs20 塞进 y["cmotion"] 复用原始逐帧 add/concat。
不要在 smoke 阶段引入高成本 xyz loss 阻塞调试。
不要只报告 overall metrics，必须至少记录 per-class 和 handshaking subset。
```
