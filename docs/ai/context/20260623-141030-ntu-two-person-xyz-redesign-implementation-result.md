# NTU 双人 xyz 视觉拟合重新设计实现结果

## 已完成改动

新增双人 xyz 转换工具：

```text
utils/ntu_smplx_2p_xyz.py
```

核心口径：

```text
输入 raw NTU motion: [B,56,6,T]
解释: rotvec/translation, 0:3 为 person1, 3:6 为 person2
转换: Rotation2xyz_x(..., pose_rep="rotvec", num_person=2)
输出 xyz: [B,T,2,55,3]
```

新增 direct xyz 预测模型：

```text
model/forecasting_ntu_xyz.py
NTULabelXYZTransformer
```

模型行为：

```text
obs20 双人 xyz + action label -> future40 双人 xyz
输出为 obs 最后一帧 + predicted delta
output_proj 零初始化，未训练时严格等价于 copy-last
delta 使用时间 ramp，第一帧 delta 强制为 0，保证 generated 第一帧等于 obs 最后一帧
```

新增评估入口：

```text
eval/eval_ntu_label_xyz.py
```

每次评估强制同时输出：

```text
model_metrics
copy_last_metrics
beats_copy_last
```

新增训练入口：

```text
train/train_ntu_label_xyz.py
```

新增 xyz cache：

```text
data_loaders/forecasting/ntu_label_xyz_cache.py
scripts/build_ntu_label_xyz_cache.py
```

原因：

```text
当前环境无可用 GPU；现场 SMPL-X 转 xyz 会拖慢正式训练。
cache 后训练直接读取 [N,T,2,55,3]，才能实际监控 MSE/MAE。
```

## 验证结果

编译检查：

```text
py_compile passed
```

双人转换 smoke：

```text
raw_shape = (60,56,6)
xyz_cat_shape = (1,55,6,60)
可整理为 [1,60,2,55,3]
finite = true
```

copy-last 小样本评估：

```text
num_samples = 2
xyz_mse = 0.039860714
xyz_mae = 0.123147614
mpjpe = 0.245317936
first_step_error = 0.0
```

在线转换 1-step 训练 smoke：

```text
checkpoint = save/forecasting/ntu120_label/xyz_redesign_smoke_1step/model000000001.pt
train_loss = 0.044823
```

cache 1-step 训练 smoke：

```text
checkpoint = save/forecasting/ntu120_label/xyz_redesign_smoke_cache_1step/model000000001.pt
train_loss = 0.049483
```

小样本过拟合和连续性检查：

```text
checkpoint = save/forecasting/ntu120_label/xyz_redesign_smoke_cache_overfit10_ramp/model000000010.pt
num_samples = 2
model xyz_mse = 0.015334505
copy-last xyz_mse = 0.049196999
model xyz_mae = 0.061907135
copy-last xyz_mae = 0.102371432
model mpjpe = 0.130926982
copy-last mpjpe = 0.215393111
first_step_error = 0.0
```

结论：

```text
新模型在小样本上确认可以通过反向传播压低 xyz MSE/MAE/MPJPE；
并且结构上保证 generated 第一帧不跳离 obs 最后一帧。
这不是正式泛化结果，只是训练闭环和硬连续性检查。
```

1-step 后未超过 copy-last，这是预期结果：

```text
只训练 1 step，作用是验证反传和评估闭环，不代表正式拟合能力。
```

## 当前环境限制

`nvidia-smi` 失败：

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

因此当前正式路线必须先构建 xyz cache，再训练。

## 下一步

1. 构建完整 train/test xyz cache。
2. 用 cache 启动正式 `NTULabelXYZTransformer` 训练。
3. 每隔固定 step 监控 `xyz_mse / xyz_mae / mpjpe` 是否超过 copy-last。
4. 只有超过 copy-last 后，才生成双人三色 skeleton 视频。
