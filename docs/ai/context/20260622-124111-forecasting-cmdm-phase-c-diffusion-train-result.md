# ForecastingCMDMDecoder 阶段 C Diffusion 训练结果

## 结论

阶段 C diffusion train gate 已完成并通过。

本阶段新增：

```text
train/train_label_forecasting_diffusion.py
docs/ai/context/20260622-124111-forecasting-cmdm-phase-c-diffusion-train-result.md
```

本阶段更新：

```text
AGENTS.md
```

本阶段未修改：

```text
train/train_mdm.py
train/training_loop.py
model/cmdm.py
model/forecasting_cmdm.py
data_loaders/forecasting/ntu_label.py
diffusion/gaussian_diffusion.py
eval/*
sample/*
```

## 实现内容

新增训练入口：

```text
train/train_label_forecasting_diffusion.py
```

核心能力：

```text
NTULabelForecastDataset/DataLoader 构建
ForecastingCMDMDecoder 构建
独立 _build_diffusion，使用 START_X / MSE / rescale_timesteps=False
q_sample(future40, t) 加噪
模型预测 clean future40
rot_mse masked loss
可选 velocity/root_translation loss，默认关闭
relative_root_loss 未确认 root slot 前禁止启用
args.json / train_log.jsonl / model*.pt / opt*.pt 保存
resume_checkpoint 恢复 model/optimizer/step
```

关键设计保持：

```text
不调用 diffusion.training_losses
不复用 TrainLoop
不把 obs_motion 塞进 y["cmotion"]
不初始化 rot2xyz
不引入 Encoder-only fallback
```

## 运行环境

```text
python_executable = /home/rpartx3080/.local/micromamba/envs/regennet/bin/python
device = cuda
torch_version = 1.7.1
model_params = 911056
```

## 验证结果

### 1. 静态编译

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile train/train_label_forecasting_diffusion.py
```

结果：

```text
exit_code = 0
```

### 2. Import smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python - <<'PY'
from train.train_label_forecasting_diffusion import build_arg_parser, run_training
print('train entry import ok')
PY
```

结果：

```text
train entry import ok
exit_code = 0
```

### 3. 2-step train smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m train.train_label_forecasting_diffusion \
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
  --lr 1e-4 --weight_decay 1e-4 \
  --num_workers 0 --seed 0 --overwrite
```

结果：

```text
exit_code = 0
resume_step = 0
step[1] train_loss = 0.4831176698207855
step[1] rot_mse = 0.4831176698207855
step[2] train_loss = 0.4620182514190674
step[2] rot_mse = 0.4620182514190674
final_checkpoint = save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt
```

输出文件：

```text
args.json
train_log.jsonl
model000000002.pt
opt000000002.pt
```

文件大小：

```text
args.json 2430
train_log.jsonl 659
model000000002.pt 6233525
opt000000002.pt 7332327
```

### 4. Checkpoint 内容检查

检查内容：

```text
model_type = forecasting_cmdm_decoder
step = 2
model_config.obs_len = 20
model_config.pred_len = 40
model_config.window_len = 60
train_protocol.mean_type = START_X
train_protocol.loss_type = MSE
diffusion_config.rescale_timesteps = False
train_log steps = [1, 2]
```

结果：

```text
checkpoint ok
steps [1, 2]
losses [0.483118, 0.462018]
exit_code = 0
```

### 5. Resume smoke

命令：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m train.train_label_forecasting_diffusion \
  --dataset ntu120_2p \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke \
  --model_type forecasting_cmdm_decoder \
  --body_model smplx \
  --window_len 60 --obs_len 20 --pred_len 40 \
  --batch_size 1 --eval_batch_size 1 \
  --num_steps 3 --save_interval 1 --eval_interval 0 \
  --latent_dim 128 --decoder_layers 2 --obs_encoder_layers 1 \
  --num_heads 4 --ff_size 512 --cond_mask_prob 0.1 \
  --lr 1e-4 --weight_decay 1e-4 \
  --num_workers 0 --seed 0 \
  --resume_checkpoint save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000002.pt
```

结果：

```text
exit_code = 0
resume_step = 2
step[3] train_loss = 0.42363622784614563
step[3] rot_mse = 0.42363622784614563
final_checkpoint = save/forecasting/ntu120_label/p2_cmdm_decoder_len60_o20_p40_smoke/model000000003.pt
```

最终输出文件：

```text
args.json 2512
train_log.jsonl 1091
model000000002.pt 6233525
model000000003.pt 6233525
opt000000002.pt 7332327
opt000000003.pt 7332327
```

最终日志：

```text
steps = [1, 2, 3]
losses = [0.483118, 0.462018, 0.423636]
```

## 退出条件核对

已满足：

```text
train/train_label_forecasting_diffusion.py 存在
py_compile 通过
import smoke 通过
2-step smoke 通过
resume smoke 通过
loss / rot_mse finite
checkpoint 内容检查通过
args.json 记录完整 CLI 和协议
train_log.jsonl 可逐行 JSON 解析
未修改 train/train_mdm.py
未修改 train/training_loop.py
未修改 model/cmdm.py
未修改 diffusion/gaussian_diffusion.py
未引入 Encoder-only 主线
```

## 下一阶段

允许进入生成闭环阶段：

```text
sample/sample_label_forecasting_diffusion.py
checkpoint load
p_sample_loop / DDIM
same obs20 + labels [2,5,8,17] label swap smoke
generated_future40.npy
metadata.json
metrics.json
label_swap_summary.json
```

仍不建议直接进入：

```text
正式 5000-step 训练
动作一致性分类器
论文结果表述
```

原因：

```text
当前只证明 diffusion train/checkpoint gate 可靠；还没有证明 checkpoint sampling 和 label-conditioned output 闭环。
```
