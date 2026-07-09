# NTU xyz 引入原 ReGenNet 论文指标可行性判断

## 问题

当前 NTU 双人 xyz forecasting 指标可以说明模型是否在坐标误差上超过 copy-last，但用户观察到动作视觉拟合并不充分，希望参考原 ReGenNet 论文中的 FID 等指标一起评估。

## 结论

可行，但必须分层使用。

当前主任务是：

```text
obs20 双人 xyz + action label -> future40 双人 xyz
```

原 ReGenNet 论文指标主要服务：

```text
action-reaction synthesis / action-conditioned motion generation
```

因此 FID、Accuracy、Diversity、Multimodality 可以作为辅助生成质量和语义指标加入，但不能替代 xyz_mse、mpjpe、copy-last gate。

## 可以优先加入的指标

### 1. Action Recognition Accuracy

目标：

```text
判断预测 future40 是否还能被动作识别模型识别成输入 action label。
```

可行性：

```text
高。项目已有 eval/action_consistency_classifier.py 和原 ReGenNet ST-GCN 评估代码可参考。
```

风险：

```text
高 accuracy 不等于贴近真实 future，只说明动作类别语义可能对。
```

### 2. FID

目标：

```text
比较预测 future40 的识别特征分布与真实 future40 的识别特征分布是否接近。
```

可行性：

```text
中到高。原代码已有 eval/a2m/stgcn/fid.py 和 ST-GCN feature extractor。
```

风险：

```text
如果预测是 deterministic 单输出，FID 可能偏向分布统计，不能说明单样本拟合好。
```

### 3. Diversity

目标：

```text
看预测结果整体是否有变化，不是所有输入都坍缩到相似动作。
```

可行性：

```text
中。可以按特征两两距离算，但 deterministic forecasting 的 diversity 主要来自不同输入样本，不是同一条件多次采样。
```

风险：

```text
不能用来证明同一 obs/action 下有多样性。
```

### 4. Multimodality

目标：

```text
原论文里看同一 action label 下生成结果是否有多种形态。
```

可行性：

```text
低到中。当前模型不是 stochastic sampler，同一个输入不会生成多个未来；只能算同一 action class 内不同样本的特征差异。
```

风险：

```text
如果按原论文名字直接写 multimodality，容易被质疑任务定义不一致。
```

## 建议的指标层级

最终报告应拆成三层：

```text
1. 逐样本预测准确性：xyz_mse, xyz_mae, mpjpe, final_frame_error, short/mid/long error, copy-last 对照。
2. 动作语义和分布质量：action accuracy, FID。
3. 多样性诊断：diversity；multimodality 仅在实现多采样后作为正式指标。
```

## 推荐实现顺序

第一步优先补：

```text
short/mid/long horizon error
final_frame_error
action accuracy
FID
```

第二步再考虑：

```text
diversity
class-wise FID / class-wise accuracy
```

第三步只有当模型支持 stochastic multi-sample prediction 后再补：

```text
multimodality
APD / ADE / FDE / MMADE / MMFDE
```

## 论文表述边界

可以写：

```text
We report both paired forecasting errors and recognition-feature-based generation metrics.
```

不能写：

```text
FID / Accuracy 证明预测轨迹逐帧拟合真实 future。
```

因为 FID 和 Accuracy 是分布与语义指标，不是 paired future accuracy。
