# NTU120 Label-conditioned ReGenNet Forecasting Smoke 设计 v2

## 当前采用协议

本版替代此前 `150/30/120` smoke 目标，当前第一阶段采用：

```text
dataset = NTU120 2P
data_path = dataset/ntu120/smplx/conditioned/xsub.train.h5
test_path = dataset/ntu120/smplx/conditioned/xsub.test.h5
window_len = 60
obs_len = 20
pred_len = 40
num_actions = 26
target_label_example = handshaking, label 8
body_model = smplx
num_person = 2
pose_rep = rot6d + translation
model = ReGenNet / CMDM-derived conditional diffusion
```

## 数据依据

当前本地 NTU120 2P 长度统计：

```text
window=60
train total=1956, covered_labels=26, min_class_count=2, handshaking=170
test total=1253, covered_labels=26, min_class_count=1, handshaking=68
```

因此 `window_len=60` 是不重采样且 train/test 都覆盖全部 26 类的最大可用窗口。

## Smoke 目标

第一阶段目标不是生成高质量视频，而是证明以下闭环可跑通：

```text
NTU120 H5 -> obs20/future40 切分 -> action label condition
-> CMDM-derived denoising model -> 2-step train
-> checkpoint load -> sampling -> basic xyz metrics -> label swap 输出
```

## Dataset 输出

新增 dataset 建议：

```text
data_loaders/forecasting/ntu_label.py
```

单条样本：

```text
obs:        [56,6,20]
future:     [56,6,40]
action:     int, 0-25
action_text: str
sample_id:  str
start:      int
length:     int
```

batch 后：

```text
obs:     [B,56,6,20]
future:  [B,56,6,40]
action:  [B,1]
mask:    [B,1,1,40]
```

采样规则：

- 仅保留 `T >= 60`。
- train split 随机裁剪连续 60 帧。
- test split 使用 center crop，保证评估确定性。
- 不 padding。
- 不时间重采样。

## 模型 smoke

模型建议：

```text
model/forecasting_cmdm.py::ForecastingCMDM
```

输入：

```text
x_t:        [B,56,6,40]
timestep:   [B]
obs_motion: [B,56,6,20]
action:     [B,1]
```

输出：

```text
pred_future: [B,56,6,40]
```

核心复用：

- `InputProcess`
- `OutputProcess`
- `TimestepEmbedder`
- `EmbedAction`
- `PositionalEncoding`
- CMDM 的 Transformer/MLP 主干思想
- classifier-free guidance 所需的 `cond_mask_prob` / `uncond`

## 训练 smoke

命令形态：

```bash
python -m train.train_label_forecasting_diffusion \
  --dataset ntu \
  --data_path dataset/ntu120/smplx/conditioned/xsub.train.h5 \
  --eval_data_path dataset/ntu120/smplx/conditioned/xsub.test.h5 \
  --save_dir save/forecasting/ntu120_label/p0_cmdm_length60_smoke \
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
  --num_heads 4 \
  --cond_mask_prob 0.1 \
  --num_workers 0 \
  --seed 0 \
  --overwrite
```

3080 smoke 预期：

- `batch_size=1` 起步。
- 如果显存正常，再提升到 `batch_size=2`。
- 先关闭高成本 xyz loss 或只对 eval 转 xyz，避免 smoke 被 SMPL-X 转换拖慢。

## 采样 smoke

固定同一个 `obs20`，构造 label swap：

```text
handshaking, label 8
hugging other person, label 5
pushing other person, label 2
```

输出：

```text
results/forecasting/ntu120_label/p0_cmdm_length60_label_swap/
```

保存：

```text
generated_future.npy
run_config.json
metrics.json
label_swap_summary.json
```

最低检查：

- 三个 label 输出 shape 均为 `[56,6,40]`。
- 输出 finite。
- 不同 label 的输出不应完全相同。
- 如果使用 classifier-free guidance，`scale` 生效且不会报错。

## 评估 smoke

基础指标：

```text
future_rot_mse
future_xyz_mse
mpjpe
root_translation_error
relative_root_distance_error
inter_person_distance_consistency
```

标签一致性第一阶段只做接口预留：

```text
action_label = ground truth label
generated_label = classifier/generated retrieval result, later stage
```

Smoke 不强制训练动作分类器。

## 成功标准

必须全部满足：

```text
dataset length gate 通过
26 类 train/test 均有样本
handshaking train/test 均有样本
dataset batch shape 正确
model forward shape 正确
loss finite
2-step training 完成
checkpoint 可加载
sampling 输出 future40
label swap 输出文件存在
```

## 不做的事

- 不做正式 5000-step。
- 不做 3-seed。
- 不承诺 handshaking 视觉质量。
- 不修改旧 `train/train_mdm.py` 语义。
- 不使用 `window_len=150` 或重采样协议。
