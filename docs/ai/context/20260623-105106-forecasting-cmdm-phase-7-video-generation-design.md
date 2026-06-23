# ForecastingCMDM 阶段 7 视频生成设计

## 阶段定位

阶段 7 是阶段 6 正式训练后的 qualitative visualization 阶段。

目标不是继续训练模型，而是把阶段 6 的正式 generated future40 转成可检查的视频：

```text
obs20
real future40
generated future40
label swap 输出
```

阶段 7 的视频只能作为定性审查和错误分析材料，不能直接作为“动作语义控制成功”的证据。

## 参考上下文

本设计依据：

```text
docs/ai/context/20260623-101205-forecasting-cmdm-phase-6-formal-training-design.md
docs/ai/context/20260623-102613-forecasting-cmdm-phase-6-code-modification-result.md
docs/ai/context/20260623-103745-forecasting-cmdm-phase-6-formal-training-train-result.md
docs/ai/context/20260623-104211-forecasting-cmdm-phase-6-formal-training-result.md
docs/ai/context/20260623-104757-forecasting-cmdm-phase-6-generated-vs-real-distance-probe.md
docs/ai/context/20260623-105007-forecasting-cmdm-fit-diagnosis-plan.md
docs/ai/context/20260615-132022-p8-official-somoformer-xyz-video-plan.md
```

相关代码入口：

```text
sample/sample_label_forecasting_diffusion.py
sample/visualize_forecasting_xyz.py
model/rotation2xyz.py
model/smpl.py
render/crendermotion.py
render/renderer.py
```

## 阶段 6 当前产物

正式训练 checkpoint：

```text
save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000/model000005000.pt
```

正式采样目录：

```text
results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap
```

采样输出：

```text
generated_future40.npy = [8,4,2,56,6,40]
obs_motion.npy = [8,56,6,20]
real_future40.npy = [8,56,6,40]
labels = [2,5,8,17]
label_action_codes = [A003,A006,A009,A018]
source_action = [0,1,2,3,4,5,6,7]
formal = true
metrics.smoke_only = false
finite = true
label_swap_summary.pass_non_identical_check = true
```

动作一致性结果：

```text
classifier_gate_pass = true
valid_for_claim = true
generated consistency_acc = 0.0
```

解释边界：

```text
真实 future40 分类器可信。
但阶段 6 generated top1 没有命中任何输入 condition label。
因此阶段 7 视频不能写成“动作标签语义控制成功”。
```

## 表示与可视化边界

当前阶段 6 输出形状是：

```text
[56,6,T]
```

本地探针验证：

```text
Rotation2xyz_x(..., jointstype="smplx", num_person=1)
obs+real [1,56,6,60] -> xyz [1,55,3,60], finite=true
generated label8 [1,56,6,40] -> xyz [1,55,3,40], finite=true
```

这说明第一版阶段 7 可以生成：

```text
SMPL-X 55-joint skeleton video
```

但不能默认声称：

```text
双人 skeleton video
two-person interaction video
```

原因：

```text
当前 H5 和阶段 6 generated 文件没有显式 person 维度。
现有 `Rotation2xyz_x` 的 `num_person > 1` 路径更接近处理 feature dim 拼接后的双人表示，例如 [56,12,T]。
而当前输出是 [56,6,T]，直接可转换路径是 num_person=1。
```

阶段 7 第一版表述应为：

```text
可视化当前 ForecastingCMDMDecoder 实际生成的 NTU120 SMPL-X motion。
```

如后续要做真正双人互动视频，必须先确认第二个人在原始数据或预处理产物中的编码路径，不能从 `[56,6,T]` 硬拆。

## 目标

新增视频生成入口：

```text
sample/visualize_label_forecasting_diffusion.py
```

阶段 7 第一版目标：

```text
读取阶段 6 generated_dir
校验 metadata / metrics / generated_consistency
将 obs / real future / generated future 转成 SMPL-X xyz
生成 skeleton mp4
保存 xyz arrays / selection / summary
输出可复现 run_config.json
```

## 本阶段不做

不做：

```text
不重新训练 ForecastingCMDMDecoder
不重新采样 generated_future40
不修改 sample/sample_label_forecasting_diffusion.py
不做 mesh render 第一版
不做双人互动视频宣称
不把视频作为语义控制成功证据
不提交 mp4 到 git
```

原因：

```text
阶段 7 是消费阶段 6 产物的可视化阶段。
mesh render 依赖 vertices、pyrender/EGL 和 SMPL-X faces，失败面比 skeleton 大。
skeleton 视频足够先检查动作是否抖动、冻结、崩坏、是否贴近真实 future。
```

## 输入协议

必需输入：

```text
generated_dir = results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap
classifier_dir = save/forecasting/ntu120_label/phase6_action_classifier_generated_consistency_h256_b3_s0
```

读取文件：

```text
generated_dir/generated_future40.npy
generated_dir/obs_motion.npy
generated_dir/real_future40.npy
generated_dir/metadata.json
generated_dir/metrics.json
generated_dir/label_swap_summary.json
classifier_dir/generated_consistency.json
classifier_dir/generated_predictions.jsonl
```

可选输入：

```text
results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_source_label_match_probe
```

用途：

```text
生成 source-label matched videos，用于更公平地比较 generated 和 real future。
```

## 输出协议

输出目录：

```text
results/forecasting/ntu120_label/phase7_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_videos
```

目录结构：

```text
run_config.json
summary.md
selection.json
selection.csv
conversion_probe.json
arrays/
  caseXXXX_labelA009_rep0.npz
videos/
  caseXXXX_labelA009_rep0.mp4
frames/
  caseXXXX_labelA009_rep0_first.png
```

每个 `.npz` 保存：

```text
obs_xyz          [20,55,3]
real_future_xyz  [40,55,3]
generated_xyz    [40,55,3]
source_meta
condition_label
condition_action_code
classifier_prediction
```

`run_config.json` 必须保存：

```text
generated_dir
classifier_dir
checkpoint
checkpoint_step
formal
labels
source_actions
render_mode
jointstype
pose_rep
translation
glob
num_person_for_conversion
fps
created_at
```

## 转换设计

使用：

```python
from model.rotation2xyz import Rotation2xyz_x
```

参数：

```text
pose_rep = rot6d
translation = true
glob = true
jointstype = smplx
vertstrans = true
num_person = 1
```

输入：

```text
motion = [B,56,6,T]
mask = ones([B,T])
```

输出：

```text
xyz = [B,55,3,T]
```

保存前转置为更方便渲染的：

```text
[T,55,3]
```

转换 gate：

```text
obs_real xyz finite
generated xyz finite
xyz abs mean / std 在合理范围
坐标轴 span > 1e-6
```

如果转换失败：

```text
阶段 7 直接 blocked
写 result 文档说明 rot6d -> xyz 不可用
不要生成伪视频
```

## 骨架连接

第一版使用 SMPL-X kintree：

```text
body_models/smplx/SMPLX_NEUTRAL.npz:kintree_table
```

本地检查：

```text
kintree_table shape = [2,55]
```

处理方式：

```text
child = 1..54
parent = kintree_table[0, child]
edge = (parent, child)
跳过 parent 为 4294967295 的 root
```

理由：

```text
避免手写错误的 SMPL-X 55-joint chain。
如果只想简化画面，可在实现中提供 --body_only，仅绘制 0..21 body joints。
```

## 视频样式

第一版使用 `matplotlib` + `imageio`，沿用 P8 skeleton 视频思路。

颜色：

```text
observed obs20       = blue
real future40        = green
generated future40   = orange
```

时间展示：

```text
frame 0..19: 只画 observed
frame 20..59: 同时画 real future 和 generated future
```

每个视频标题包括：

```text
case index
source action
condition action
classifier predicted action
frame index
```

坐标轴：

```text
每个视频用 obs + real + generated 共同计算固定 axis limits
保持 x/y/z 等比例
隐藏坐标 tick
保留 legend
```

帧率：

```text
fps = 20
```

理由：

```text
window_len=60，20fps 下单个视频约 3 秒，便于快速检查。
```

## 样本选择

阶段 7 不随机挑样本，应覆盖成功和失败风险。

第一版选择：

```text
1. source-label matched:
   case 2, condition label 2/A003
   case 5, condition label 5/A006

2. handshaking condition:
   label 8/A009 的若干 case

3. classifier failure modes:
   generated_predictions 中 predicted_label = 7/A008 或 0/A001 的样本

4. label swap panel:
   同一个 case 下 labels [A003,A006,A009,A018] 的生成对比
```

最小 smoke：

```text
num_videos = 2
```

正式阶段 7：

```text
num_videos = 8
```

注意：

```text
source-label matched 只在当前 label_swap 输出中覆盖 case 2/A003 和 case 5/A006。
如果要更多公平对照，应使用 source_label_match_probe 目录或重新采样真实 source label。
```

## 命令设计

静态检查：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m py_compile sample/visualize_label_forecasting_diffusion.py
```

转换 smoke：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m sample.visualize_label_forecasting_diffusion \
  --generated_dir results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap \
  --classifier_dir save/forecasting/ntu120_label/phase6_action_classifier_generated_consistency_h256_b3_s0 \
  --save_dir results/forecasting/ntu120_label/phase7_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_videos_smoke \
  --render_mode skeleton \
  --jointstype smplx \
  --num_videos 2 \
  --fps 20 \
  --force_cpu \
  --overwrite
```

正式视频：

```bash
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python \
  -m sample.visualize_label_forecasting_diffusion \
  --generated_dir results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap \
  --classifier_dir save/forecasting/ntu120_label/phase6_action_classifier_generated_consistency_h256_b3_s0 \
  --save_dir results/forecasting/ntu120_label/phase7_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_videos \
  --render_mode skeleton \
  --jointstype smplx \
  --num_videos 8 \
  --fps 20 \
  --force_cpu \
  --overwrite
```

## 验收条件

代码 gate：

```text
py_compile 通过
import smoke 通过
```

数据 gate：

```text
generated_dir formal=true
metrics.smoke_only=false
generated_shape=[8,4,2,56,6,40]
generated finite=true
```

转换 gate：

```text
obs_xyz shape = [20,55,3]
real_future_xyz shape = [40,55,3]
generated_xyz shape = [40,55,3]
所有 xyz finite=true
conversion_probe.json 存在
```

视频 gate：

```text
smoke 至少生成 2 个 mp4
正式至少生成 8 个 mp4
每个 mp4 文件大小 > 0
首帧 png 存在且非空
selection.json / selection.csv / summary.md 存在
```

解释 gate：

```text
summary.md 必须写明 generated consistency_acc=0.0
summary.md 必须写明当前视频是 qualitative/debug，不是语义控制成功证据
summary.md 必须写明当前直接转换路径为 num_person=1，不声称双人互动视频
```

## 结果文档

阶段 7 实现完成后新增：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-phase-7-video-generation-result.md
```

必须记录：

```text
实现文件
输入 generated_dir
输入 classifier_dir
转换参数
转换 probe 结果
生成视频数量
输出目录
选样策略
视频文件列表
是否发现冻结/抖动/尺度异常
解释边界
```

不要记录：

```text
不要贴大段 JSON
不要提交 mp4
不要把视频定性观察写成论文结论
```

## 后续方向

阶段 7 通过后，有两条可能路线：

```text
1. 如果视频显示生成质量可接受：
   进入更多样本、更多 label、可能 mesh render。

2. 如果视频显示冻结、抖动或明显偏离真实：
   回到拟合诊断，优先执行 `20260623-105007-forecasting-cmdm-fit-diagnosis-plan.md`。
```

当前阶段 6 已经显示：

```text
generated consistency_acc = 0.0
source-label matched generation 的 MAE 仍差于 copy-last baseline
```

因此阶段 7 更可能服务于错误分析，而不是展示最终效果。
