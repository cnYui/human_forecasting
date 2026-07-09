# NTU 双人 xyz 长预测误差放大修复计划

## 现象

用户检查 `xyz_transformer_1000_tricolor_2p_videos_zflip` 后发现：

```text
预测越到后面，橙色 generated future 与绿色 real future 的误差越大。
```

## 判断

这不是 Z 轴翻转渲染问题。

当前训练和评估事实：

```text
full test:
model xyz_mse = 0.031281803
copy  xyz_mse = 0.057248837
model xyz_mae = 0.091381698
copy  xyz_mae = 0.120007492
model mpjpe   = 0.189595376
copy  mpjpe   = 0.260575039
first_step_error = 0.0
```

说明模型整体已经超过 copy-last，第一帧也没有跳变；但视频后半段误差放大仍可能存在。

当前代码中的核心限制：

```text
1. training_loss 对 40 帧 future 做均匀 MSE。
2. velocity loss 也是整段平均。
3. eval 只报告整段平均指标，没有 per-frame MSE/MAE/MPJPE。
4. 当前 8 个视频子集上 MAE/MPJPE 没整体超过 copy-last，不应拿它代表 full test。
```

因此后段变差的根因大概率是：

```text
模型优化了整段平均误差，但没有专门约束长 horizon；
decoder 直接输出 40 帧绝对 delta，后半段没有 rollout 纠错信号；
数据类别不均衡，少样本动作的视频更容易在后段回归到平均姿态/平均轨迹。
```

## 必须先补的诊断

新增 per-frame 评估，至少输出：

```text
frame_mse[40]
frame_mae[40]
frame_mpjpe[40]
copy_frame_mse[40]
copy_frame_mae[40]
copy_frame_mpjpe[40]
per_frame_beats_copy_last[40]
```

额外保存：

```text
case_id
action
per_case_mpjpe_mean
per_case_mpjpe_last10
per_case_copy_mpjpe_last10
```

原因：

```text
如果只是最后 10 帧崩，应该调 horizon loss；
如果少数 action 崩，应该做 class-balanced sampling / loss；
如果所有 action 都线性变差，应该加入终点和速度/加速度约束；
如果 copy-last 后段也崩但模型更崩，说明模型学到了错误动态，需要保守残差或 scheduled horizon。
```

## 优先修复方案

### 方案 A：horizon-weighted loss

把 future 后半段权重提高：

```text
weights[t] = linear(1.0, 3.0) 或 quadratic(1.0, 4.0)
loss_pos = mean_t(weights[t] * mse(pred[:,t], target[:,t]))
```

这是第一优先级，因为它直接针对“越到后面误差越大”。

### 方案 B：terminal loss

额外约束最后几帧：

```text
terminal_loss = MSE(pred[:, -5:], target[:, -5:])
loss += terminal_loss_weight * terminal_loss
```

建议从 `terminal_loss_weight=0.5` 起，不要太大，避免牺牲前段连续性。

### 方案 C：velocity + acceleration loss

当前已有 velocity loss，但没有 acceleration：

```text
vel = x[t] - x[t-1]
acc = vel[t] - vel[t-1]
```

建议：

```text
velocity_loss_weight: 0.5
acceleration_loss_weight: 0.1
```

理由：

```text
长 horizon 漂移通常不是单帧位置误差，而是速度趋势错误累积。
acceleration loss 可以抑制后半段轨迹发散和不自然抖动。
```

### 方案 D：root / relative-root 加权

视频里最明显的错通常来自 root translation 和双人相对位置：

```text
root_loss = MSE(pred[:,:,:,0], target[:,:,:,0])
relative_root_loss = MSE(root_dist(pred), root_dist(target))
```

建议先加低权重：

```text
root_loss_weight = 0.5
relative_root_loss_weight = 0.2
```

原因：

```text
只优化全关节平均 MSE 时，root 错误会被 55 个 joint 平均稀释；
但视频观感中 root 漂移最明显。
```

### 方案 E：训练更久，但必须用 per-frame 指标早停

当前正式结果只有 1000 step。可以训练到：

```text
3000 / 5000 / 8000 step
```

但不能只看整段平均 test MSE。保存标准改为：

```text
primary: last10_mpjpe
secondary: full mpjpe, full mae, first_step_error
required: full mpjpe 和 last10_mpjpe 都超过 copy-last
```

## 不建议的方向

暂时不建议回到 diffusion free sampling。

原因：

```text
当前问题是长 horizon 贴合真实 future；
扩散采样会重新引入多步采样误差和随机性；
在 deterministic xyz 模型没把 long-horizon 指标压下来前，扩散只会增加排查复杂度。
```

也不建议继续只改视频渲染：

```text
Z 轴翻转只影响显示，不影响 pred/target 数值误差。
```

## 推荐执行顺序

1. 修改 `eval/eval_ntu_label_xyz.py` 和 `utils/ntu_smplx_2p_xyz.py`，增加 per-frame 指标。
2. 用现有 `model000001000.pt` 重评 full test 和 8 个视频 case，确认后半段是否系统性差于 copy-last。
3. 修改 `NTULabelXYZTransformer.training_loss`，加入 horizon-weighted MSE、terminal loss、acceleration loss、root/relative-root loss，默认权重保持旧行为，命令行显式开启。
4. 从 1000-step checkpoint 继续训练，先跑 3000 step，对比 last10_mpjpe。
5. 只有当 full test 和视频 case 的 last10 指标都超过 copy-last，再重新导出视频。

## 最小实验配置建议

```text
resume_checkpoint = xyz_transformer_len60_o20_p40_h256_l3_s0_1000/model000001000.pt
num_steps = 3000
lr = 1e-4
velocity_loss_weight = 0.5
mae_loss_weight = 0.1
horizon_loss_weight = linear_1_to_3
terminal_loss_weight = 0.5
terminal_frames = 5
acceleration_loss_weight = 0.1
root_loss_weight = 0.5
relative_root_loss_weight = 0.2
```

验收：

```text
full_test_last10_mpjpe < copy_last_last10_mpjpe
full_test_full_mpjpe  < copy_last_full_mpjpe
case8_last10_mpjpe 不应整体差于 copy-last
first_step_error = 0.0
```
