# DuMMF 架构用于 InterHuman 双人视频预测的可行性判断

## 用户问题

用户希望使用当前克隆的 DuMMF 项目的模型架构，结合 `/home/rpartx3080/CodeSpace/ReGenNet/dataset/interhuman` 数据集，预测双人的视频。

## 关键判断

可行，但要先明确任务不是“从 RGB 视频直接预测视频像素”，而是：

```text
InterHuman motion / SMPL 参数 -> 预测双人未来 motion -> 渲染成视频
```

如果目标是 RGB 视频到 RGB 视频，DuMMF 和当前 ReGenNet 都不是合适入口；需要额外的视频 pose/SMPL 估计器和视频生成/渲染链路。

## 为什么不能直接搬 DuMMF 训练脚本

DuMMF 官方公开代码当前是 diffusion-based 实现，默认 `data_amass.py` 读取：

```text
./mocap/CMU/
  AMASS-CMU .npz
  keys: poses, trans, betas, gender, mocap_framerate

./mocap/body_models/
  smplh/
  dmpls/
```

而当前 ReGenNet 的 InterHuman 已有：

```text
dataset/interhuman/motions/*.pkl
dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5
```

两者数据协议不同：

- DuMMF：AMASS / SMPL-H pose + trans + betas，默认 3-person，10 observed + 25 future。
- ReGenNet InterHuman：双人 SMPL active-vector / xyz pipeline，当前主协议是 150 帧窗口、30 observed + 120 future。

因此直接运行 DuMMF 原 `train_diffusion.py` 不会读 InterHuman，也不能直接复用 P5/P7 指标。

## 推荐路线

推荐在 ReGenNet 内做 `dummf_interhuman_xyz` 或 `dummf_interhuman_active`，而不是改 DuMMF 外部仓库作为主实现。

优先路线：

1. 以 ReGenNet 现有 `data_loaders/forecasting/interhuman.py` 和 `utils/forecasting_xyz.py` 为数据入口。
2. 先做 xyz 版：输入 `[B, 30, 2, J, 3]`，输出多样本未来 `[B, K, 120, 2, J, 3]`。
3. 迁入 DuMMF 的核心思想，而不是复制原数据 loader：
   - local intent：两个人可用独立 intent code。
   - global/social intent：同一个 sample 内两个人共享 intent code。
   - Best-of-K loss 和 diversity/FPD 风格指标。
4. 渲染只作为输出展示：预测出 motion 后再用现有 SMPL/xyz 可视化链路生成视频。

## 不推荐路线

- 不建议直接把 InterHuman 转成 DuMMF 的 AMASS-CMU 目录结构硬塞给 `data_amass.py`。
- 不建议第一步就做 SMPL-H/DMPL mesh diffusion；当前 InterHuman 本地是 SMPL reproduction 口径，SMPL-H/SMPL-X 口径不完全对齐。
- 不建议把 DuMMF stochastic 多未来结果直接和 P5 deterministic 主表横比；指标口径必须改成 Best-of-K + diversity。

## 结论

如果目标是“用 InterHuman 生成双人未来动作视频”，可以做，而且方向合理。

最小可行实现应是：

```text
InterHuman xyz dataset
  -> DuMMF-style stochastic two-person predictor
  -> Best-of-K / ADE/FDE/FPD / relative metrics
  -> selected samples render video
```

这应作为 ReGenNet 新阶段实现，不应当直接在外部 DuMMF 仓库里完成。
