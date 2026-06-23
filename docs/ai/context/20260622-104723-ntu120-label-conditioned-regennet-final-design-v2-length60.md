# NTU120 Label-conditioned ReGenNet Forecasting 最终目标设计 v2

## 最终目标

训练一个 ReGenNet / CMDM-derived 条件扩散模型，实现：

```text
输入：前 20 帧双人动作 + 动作标签
输出：后 40 帧双人动作
数据：NTU120 2P
标签：26 个双人动作类，包含 handshaking
表示：SMPL-X rot6d + root translation
评估：转 xyz 后计算 MPJPE / joint MSE / interaction metrics
生成：同一 obs20 更换标签时，future40 应随标签发生可观察变化
```

本版正式协议：

```text
window_len = 60
obs_len = 20
pred_len = 40
不重采样
不 padding
只保留 T >= 60 的序列
```

## 为什么改为 20 -> 40

此前 `30->120` 目标需要 `window_len=150`，但当前 NTU120 本地 H5 几乎没有这么长的序列：

```text
T>=150: all 10 / 8118 = 0.12%
handshaking T>=150 = 0
```

长度统计显示：

```text
train median = 57
test median = 49
all median = 54
```

在不重采样前提下，`window_len=60` 是 train/test 均覆盖全部 26 个动作类的最大窗口：

```text
train T>=60 = 1956
test T>=60 = 1253
handshaking train = 170
handshaking test = 68
```

因此第一版正式目标改为 `20->40`，优先保证任务真实、标签完整和可训练。

## 数据设计

输入：

```text
dataset/ntu120/smplx/conditioned/xsub.train.h5
dataset/ntu120/smplx/conditioned/xsub.test.h5
```

H5 item：

```text
[T,56,6]
```

标签：

```text
A001-A026 -> label 0-25
handshaking = A009 -> label 8
```

dataset 输出：

```text
obs:        [56,6,20]
future:     [56,6,40]
action:     int
action_text: str
sample_id:  str
```

batch：

```text
obs_motion: [B,56,6,20]
x_start:    [B,56,6,40]
action:     [B,1]
mask:       [B,1,1,40]
```

train crop：

```text
从 T>=60 序列中随机裁剪连续 60 帧
```

test crop：

```text
center crop 连续 60 帧
```

## 模型设计

主模型：

```text
ForecastingCMDM
```

定位：

```text
CMDM/ReGenNet 条件扩散主干的 forecasting 适配版
```

输入输出：

```text
x_t future40 + timestep + obs20 + action label -> denoised future40
```

必须复用的 ReGenNet 组件：

- timestep embedding
- action embedding
- classifier-free guidance 的训练和采样机制
- Transformer/MLP denoiser 主干
- SMPL-X rot6d + translation 表示
- xyz reconstruction / velocity / interaction loss 思路

建议结构：

```text
obs20 -> obs encoder -> obs_cond
action -> EmbedAction -> action_cond
timestep -> TimestepEmbedder -> time_cond
x_t future40 -> InputProcess -> denoiser -> OutputProcess -> pred future40
condition = time_cond + action_cond + obs_cond
```

第一版不要直接复用原始 `CMDM.forward()` 的 `cmotion` 语义。原始 CMDM 是 actor full motion condition -> reactor full motion；本任务是 two-person prefix + label -> two-person future。

## Loss 设计

基础：

```text
rot_mse
velocity_mse
xyz_mse / mpjpe-style reconstruction
root_translation_error
```

交互：

```text
relative_root_distance_error
inter_person_distance_consistency
relative body distance
relative orientation
```

第一版建议从轻量 loss 起步：

```text
rot_mse = 1.0
velocity_mse = 0.5
xyz_mse = 0.5 或 eval-only
relative_root = 0.5
```

原因：

- SMPL-X xyz 转换开销大。
- 先保证 CMDM diffusion 训练稳定，再逐步加重几何 loss。

## 类别不平衡处理

`T>=60` 后虽然覆盖 26 类，但部分类别样本很少：

```text
train min_class_count = 2
test min_class_count = 1
```

正式训练需要至少记录类别分布，并优先加入：

```text
class-balanced sampler
或 per-class repeat sampling
或 class-weighted reporting
```

不能只报告总体指标，否则大类会掩盖小类失败。

## 训练阶段

### P0 数据 gate

必须输出：

```text
T>=60 train/test 总数
每类 train/test 数量
handshaking 数量
obs/future shape
```

### P1 dataset smoke

检查：

```text
[B,56,6,20]
[B,56,6,40]
action [B,1]
action_text
finite
```

### P2 model forward smoke

检查：

```text
x_t [B,56,6,40]
obs_motion [B,56,6,20]
output [B,56,6,40]
loss finite
```

### P3 2-step train smoke

命令形态：

```bash
python -m train.train_label_forecasting_diffusion \
  --dataset ntu \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/p3_cmdm_len60_o20_p40_smoke \
  --model_type forecasting_cmdm \
  --body_model smplx \
  --num_person 2 \
  --window_len 60 \
  --obs_len 20 \
  --pred_len 40 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_steps 2 \
  --save_interval 2 \
  --eval_interval 2 \
  --latent_dim 128 \
  --layers 2 \
  --cond_mask_prob 0.1 \
  --num_workers 0 \
  --seed 0 \
  --overwrite
```

### P4 label swap sampling

固定同一 `obs20`，输入不同标签：

```text
2 pushing other person
5 hugging other person
8 handshaking
17 high-five
```

输出每个 label 的 generated future40，并生成对比可视化。

### P5 正式单 seed 训练

命令形态：

```bash
python -m train.train_label_forecasting_diffusion \
  --dataset ntu \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/forecasting_cmdm_len60_o20_p40_h256_l4_s0_5000 \
  --model_type forecasting_cmdm \
  --body_model smplx \
  --num_person 2 \
  --window_len 60 \
  --obs_len 20 \
  --pred_len 40 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --eval_batch_size 4 \
  --num_steps 5000 \
  --save_interval 1000 \
  --eval_interval 1000 \
  --latent_dim 256 \
  --layers 4 \
  --num_heads 4 \
  --cond_mask_prob 0.1 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --seed 0
```

3080 保守配置：

```text
batch_size = 2
grad_accum_steps = 8
latent_dim = 192
layers = 3
```

### P6 3-seed

仅当 P5 的 loss、metrics 和 label swap 输出合理后启动：

```text
seed = 0,1,2
```

## 评估

预测误差：

```text
future_rot_mse
joint_mse
mpjpe
short/mid/long joint_mse，按 40 帧分段
root_translation_error
relative_root_distance_error
inter_person_distance_consistency
```

动作标签一致性：

```text
generated future -> action classifier -> predicted label
label consistency accuracy
handshaking subset accuracy
label swap distinguishability
```

第一版可以先预留接口，不把动作分类器训练放进 smoke；正式结果必须补齐，否则无法证明“输入 handshaking 后生成的是握手”。

## 可视化输出

至少保存：

```text
GT obs20 + GT future40
GT obs20 + generated future40(label=handshaking)
同一 obs20 + generated future40(label=hugging/pushing/high-five)
```

输出目录建议：

```text
results/forecasting/ntu120_label/forecasting_cmdm_len60_o20_p40_label_swap/
```

## 与旧设计的关系

旧设计：

```text
window_len=150, obs_len=30, pred_len=120
```

保留为长期理想目标或重采样协议目标。

当前执行目标：

```text
window_len=60, obs_len=20, pred_len=40
```

这是当前 NTU120 原始连续帧数据上可训练的第一版任务。

## 当前结论

后续实现应以本 v2 协议为准，除非用户重新确认要使用时间重采样或更换数据集。
