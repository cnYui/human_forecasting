# NTU 双人 xyz 视觉拟合重新设计计划

## 用户确认边界

用户已确认：

```text
1. 接受直接进入 xyz / skeleton space。
2. 必须做双人互动，不能再做单 skeleton。
3. copy-last 是硬基线；如果生成视频连最后一帧延续都不如，就没有意义。
```

## 当前问题

阶段 6/7 的关键问题不是颜色，也不是视频脚本本身，而是表示口径混乱：

```text
当前 NTU H5 shape: [T,56,6]
阶段 6 训练脚本把 data_rep 写成 rot6d。
阶段 7 可视化用 Rotation2xyz_x(..., pose_rep=rot6d, num_person=1)。
```

但从 `preprocess/actor_reactor.py` 和原始 dataset loader 看，NTU 2P 的 6 维应按双人拆分：

```text
0:3 = person 1 的 axis-angle / translation
3:6 = person 2 的 axis-angle / translation
前 55 个 joint = SMPL-X pose
第 56 个 joint = 双人 root translation
```

因此阶段 7 单 skeleton 视频不能代表双人互动，也不能作为最终视觉拟合结果。

## 数据可行性验证

已用 `xsub.train.h5` 抽样验证：

```text
raw_shape = (60,56,6)
Rotation2xyz_x(..., pose_rep="rotvec", num_person=2)
xyz_cat_shape = (1,55,6,60)
可整理为 [B,T,2,55,3]
finite = true
```

结论：

```text
当前本地 NTU H5 可以恢复双人 skeleton xyz。
不需要放弃 NTU，但必须停止把 [56,6] 当单人 rot6d 使用。
```

## 新训练目标

正式目标改为：

```text
输入: obs20 的双人 skeleton xyz + 真实 source action label
输出: future40 的双人 skeleton xyz
shape: [B,20,2,55,3] -> [B,40,2,55,3]
```

训练损失必须直接作用在 xyz / skeleton space：

```text
主损失: future xyz MSE
监控: MSE, MAE, MPJPE
连续性: 第一帧 future 与 obs 最后一帧的位移误差
动态: velocity error
互动: relative root distance error / inter-person distance consistency
```

## 模型口径

第一版不继续扩散采样，先做 deterministic direct xyz predictor：

```text
obs xyz frames -> Transformer encoder
action embedding -> memory token
future positional query -> Transformer decoder
output delta relative to obs last frame
pred_xyz = last_obs_xyz + delta
```

原因：

```text
1. 当前核心问题是视觉拟合，不是多样性。
2. diffusion 的多步采样误差已经暴露为 free sampling 不贴真实 future。
3. copy-last 是硬基线，所以模型初始输出应等价或接近 copy-last。
4. 输出 delta relative to last frame 可以强制第一帧 future 与 obs 末帧连续。
```

## 硬验收门槛

每次训练必须同时汇报：

```text
generated_xyz_mse
generated_xyz_mae
generated_mpjpe
copy_last_xyz_mse
copy_last_xyz_mae
copy_last_mpjpe
first_step_error
velocity_error
relative_root_distance_error
inter_person_distance_consistency
```

验收规则：

```text
1. 生成结果必须在 MSE 和 MPJPE 上优于 copy-last。
2. MAE 至少不能明显差于 copy-last；若差于 copy-last，需要标记为未通过。
3. 视频必须显示两个人，并且 obs 最后一帧到 generated 第一帧不能明显跳变。
4. 不再用分类器一致性作为主指标。
```

## 实现步骤

1. 新增 NTU 双人 rotvec 到 xyz 的工具函数。
2. 新增 direct xyz 模型，输出 `[B,40,2,55,3]`。
3. 新增 NTU label-conditioned xyz 训练脚本，训练中从原始 H5 转 xyz，但模型和 loss 都在 xyz space。
4. 新增 xyz 评估脚本，固定对比 copy-last。
5. 新增双人三色 skeleton 可视化脚本：蓝色 obs，橙色 generated，绿色 real。

## 当前不做

```text
不继续微调阶段 6 rot6d/CMDM checkpoint。
不再把当前单 skeleton 视频包装成双人互动。
不先追求多样性采样。
不声明动作语义控制成功，除非视觉和距离指标都支持。
```
