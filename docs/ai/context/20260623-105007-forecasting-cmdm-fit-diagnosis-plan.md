# ForecastingCMDM 生成拟合问题诊断计划

## 背景

阶段 6 已完成训练、采样和动作一致性复评。用户明确当前核心目标不是用分类器判生成动作类别，而是通过反向传播学习，使生成 future40 更贴近真实训练/测试 future40。

已有直接距离 probe 显示：

```text
source-label matched generation MSE = 0.039382558315992355
copy-last baseline MSE = 0.04473424702882767
source-label matched generation MAE = 0.0848858579993248
copy-last baseline MAE = 0.06692911684513092
```

因此当前生成不是坏掉，但拟合还不够强。

## 本次要回答的问题

```text
1. 当前训练目标是否真的在优化 generated future40 和 real future40 的距离？
2. 模型在 teacher-forced denoising 下是否能把 noised real future40 还原到真实 future40？
3. 如果 teacher-forced loss 低但 free sampling 误差高，问题是否主要在采样反推或条件生成分布？
4. 如果 teacher-forced loss 也高，问题是否主要在模型容量、训练步数、loss 设计或数据尺度？
5. 下一步怎样改，才能让生成帧更贴近真实 future40？
```

## 诊断步骤

1. 检查代码协议：

```text
train/train_label_forecasting_diffusion.py
sample/sample_label_forecasting_diffusion.py
model/forecasting_cmdm.py
diffusion/respace.py
diffusion/gaussian_diffusion.py
```

重点看：

```text
训练目标 = pred_xstart vs future40
采样目标 = 从 noise 反推 future40
mask / shape / action / obs_motion 是否一致
DDIM50 是否可能造成采样误差
```

2. 用 `model000005000.pt` 做 teacher-forced 诊断：

```text
t = 0, 50, 100, 250, 500, 750, 999
x_t = q_sample(real_future40, t)
model(x_t,t,obs/action) -> pred_xstart
计算 pred_xstart vs real_future40 MSE/RMSE/MAE
```

3. 对同一批 case 比较：

```text
copy-last baseline
teacher-forced denoising
free DDIM50 sampling
已有 source-label matched sampling
```

4. 形成判断：

```text
训练拟合问题
采样反推问题
条件弱使用问题
loss/数据尺度问题
```

## 暂不做

```text
不直接改模型
不直接重跑长训练
不把分类器 consistency 当作主要判断标准
不做视频可视化结论
```

## 产出

新增诊断结果文档：

```text
docs/ai/context/YYYYMMDD-HHMMSS-forecasting-cmdm-fit-diagnosis-result.md
```
