# SoMoFormer 架构分析与当前 Forecasting 模型问题定位

## 分析对象

用户指定论文：

```text
docs/download/2022-somoformer-multi-person-pose-forecasting-transformers.pdf
```

当前项目对照代码：

```text
model/forecasting.py
```

当前现象：

```text
joint / relation-aware predictor 的整体 MSE 不如 independent predictor。
```

## SoMoFormer 的核心架构思想

SoMoFormer 的关键不是简单使用 Transformer，而是重写了 forecasting 的 token 组织方式。

传统做法常把输入看成：

```text
time sequence of poses
[T, people, joints, xyz]
```

SoMoFormer 改成：

```text
joint-sequence series of trajectories
```

即每个 token 不是一个时间步，而是：

```text
某个人某个关节某个坐标维度在整段时间上的轨迹。
```

对于 N 个人、J 个关节、xyz 三个坐标：

```text
token 数 = N * J * 3
每个 token 的内容 = 长度为 obs_len + pred_len 的一维时间轨迹
```

未来部分先用合理初值 padding，通常是延续最后观测姿态；模型做的是 trajectory completion / refinement。

## SoMoFormer 的主要模块

### 1. Future padding + trajectory completion

输入不是只给前 t 帧，而是构造：

```text
[observed trajectory || padded future trajectory]
```

其中 future 初值来自历史轨迹的常数 padding。

好处：

```text
模型不是从一个隐变量直接吐出完整未来，而是在一个合理初值上修正整段未来。
这比从单个 hidden vector 直接生成 120 帧更稳定。
```

### 2. DCT / IDCT 时间频域表示

SoMoFormer 对每个 joint coordinate trajectory 做 DCT：

```text
time trajectory -> DCT coefficients
```

模型在频域里预测整段 completed trajectory，最后用 IDCT 还原。

作用：

```text
利用人体动作连续、平滑、近似周期的特性。
一次性预测全部 future，不用递归，减少误差累积。
```

### 3. Joint-aware token

每个 token 对应具体 joint / coordinate trajectory。

模型不是只看：

```text
person vector = 147 dim
```

而是让不同关节轨迹作为 token 互相 attention。

作用：

```text
保留身体结构粒度。
模型能学某个关节应关注哪些其他关节。
```

### 4. Person identity embedding

每个 token 加 person identity embedding：

```text
这个 joint 属于第几个人。
```

作用：

```text
让模型区分同名关节来自哪个人。
```

### 5. Joint type embedding

每个 token 加 joint type embedding：

```text
left knee / right shoulder / hip / neck 等。
```

作用：

```text
让 Transformer 知道 token 的身体语义。
```

### 6. Global position / grid embedding

SoMoFormer 会移除每个人的 global translation，让局部动作更好学；然后用 learned grid position embedding 告诉模型这个人在场景中的大致位置。

作用：

```text
局部姿态和全局交互分开建模。
既避免 global translation 干扰姿态学习，又保留人与人距离信息。
```

### 7. 数据增强

SoMoFormer 做了：

```text
random first-frame selection
sequence flipping
random scene rotation
random person permutation
```

ablation 中，learned embedding、data augmentation、grid positioning 都明显改善 VIM/MPJPE。

对当前项目的启发：

```text
不要只改模型结构，也要改输入编码和训练增强。
```

### 8. Layer-wise auxiliary loss

SoMoFormer 对每个 Transformer layer 的输出都加辅助 loss。

作用：

```text
鼓励每一层都逐步 refine motion，而不是只靠最后一层学习。
```

## 当前模型为什么 joint predictor 可能不如 independent

当前 `model/forecasting.py` 中：

```text
independent: 每个人 [30,147] -> GRU -> hidden -> 该人 [120,147]
concat:      双人 [30,294] -> GRU -> single hidden -> 双人 [120,294]
relation:    两个单人 GRU hidden + root relation GRU hidden -> single hidden -> 双人 [120,294]
```

### 问题 1：joint 模型把所有未来压进一个 single hidden bottleneck

concat/relation 都是：

```text
obs [B,30,2,147]
-> hidden [B,256]
-> Linear 输出 [B,120*2*147]
```

也就是用一个 256 维向量直接生成 35280 个未来数值。

风险：

```text
个体动作细节被压缩太狠。
两个人的各自运动模式互相干扰。
模型为了学交互，牺牲了单人预测精度。
```

independent 反而更稳，因为它只需要学：

```text
单人 [30,147] -> 单人 [120,147]
```

### 问题 2：concat 没有 joint/person/token 结构

concat 只是把两个人拼成：

```text
[actor 147 | reactor 147] = 294
```

GRU 不知道：

```text
哪些维度是左膝？
哪些维度是 root translation？
哪些维度是 actor？
哪些维度是 reactor？
哪些 joint 之间应该强相关？
```

这些都必须从数据里硬学，样本量又不大，所以容易输给 independent。

### 问题 3：relation 模型只给了 root-level relation，无法解释大部分 rotation MSE

当前 relation features 是：

```text
relative root translation
relative root velocity
root distance
relative root orientation
```

这些主要帮助 root / global interaction。

但 `future_mse` 大量来自：

```text
24 joints * rot6d
```

如果 relation branch 只影响一个全局 hidden，它不一定能改善具体关节旋转，甚至可能干扰单人 motion hidden。

### 问题 4：joint 模型不能自然退化成 independent

当前 relation 模型输出是：

```text
joint_hidden -> full future
```

它不是：

```text
independent_future + interaction_residual
```

所以当交互信息无用或噪声大时，模型不能安全退回 independent predictor。

这解释了为什么：

```text
relation-aware 能赢 concat，但仍输 independent。
```

它改善了无结构 concat 的交互建模，但没有保住 independent 的单人预测能力。

## 对当前项目最有价值的 SoMoFormer 启发

### 启发 1：把 active vector 拆成 token，而不是整人 flatten

当前 active：

```text
[B,T,2,147]
```

建议改成 token 表示：

```text
2 persons * 24 body joints + 2 root tokens = 50 tokens
```

每个 body joint token：

```text
rot6d trajectory: [obs_len + pred_len, 6]
```

每个 root token：

```text
translation trajectory: [obs_len + pred_len, 3]
```

再通过 projection 变成统一 embedding。

### 启发 2：使用 future repeat padding + residual correction

先构造：

```text
active_init [B,150,2,147]
active_init[:, :30] = obs
active_init[:, 30:] = repeat last obs frame
```

模型预测 residual：

```text
pred = active_init_future + residual
```

这样至少有 repeat fallback，比直接从 hidden 生成未来更稳定。

### 启发 3：用 DCT 或 temporal projection 建模整段轨迹

SoMoFormer 的 DCT 可以迁移到 active-vector token：

```text
每个 token 的 150 帧轨迹 -> DCT coefficients
Transformer 预测 corrected coefficients
IDCT -> 150 帧 active
```

第一版也可以先不用 DCT，直接用 temporal MLP / 1D conv，但 DCT 是更贴近论文的设计。

### 启发 4：加入 person / joint / channel type embedding

至少需要：

```text
person embedding: actor / reactor
joint embedding: 0..23 + root
token type embedding: rot6d / translation
```

可选：

```text
left/right side embedding
body part embedding
```

### 启发 5：关系建模应该是 attention 内生的，而不是只拼一个 relation hidden

当前 relation branch 是：

```text
relation_features -> GRU -> hidden -> concat
```

SoMoFormer 风格应让 token 之间直接 attention：

```text
actor left hand token 可以关注 reactor torso/root token
actor root token 可以关注 reactor root token
```

这比一个 root relation hidden 更细。

### 启发 6：用 independent + interaction residual 防止退化

最关键的工程建议：

```text
不要让 joint model 直接替代 independent。
```

建议新模型输出：

```text
base_pred = independent_model(obs)
delta_pred = interaction_model(obs)
pred = base_pred + delta_pred
```

再加一个 residual gate：

```text
pred = base_pred + gate * delta_pred
```

gate 可从 relation features 或 token attention summary 预测。

这样模型至少能学到：

```text
当交互信息没帮助时 gate -> 0，性能接近 independent。
当交互信息有帮助时 gate > 0，修正两人未来。
```

这比当前 relation 模型更符合你现在遇到的问题。

## 推荐下一版模型方向

### P7-A：Interaction Residual Predictor

最小改动，优先级最高。

结构：

```text
shared independent encoder/decoder -> base_pred
relation/token encoder -> delta_pred
pred = base_pred + gate * delta_pred
```

优点：

```text
最可能解决“不如 independent”的问题。
改动小，不必一次实现完整 SoMoFormer。
```

### P7-B：SoMoFormer-lite Active Token Transformer

中等改动。

结构：

```text
obs -> repeat future padding -> active_init [B,150,2,147]
active_init -> 50 tokens
token + person/joint/type embedding
Transformer encoder
predict future residual
```

优点：

```text
真正吸收 SoMoFormer 的核心 token 思想。
比当前 GRU concat 更有结构。
```

### P7-C：DCT Active Token Transformer

较大改动。

结构：

```text
active_init token trajectories -> DCT
Transformer over tokens
predict DCT residual
IDCT -> future active
```

优点：

```text
最接近 SoMoFormer。
能直接建模整段未来轨迹的平滑性。
```

风险：

```text
rot6d 的 DCT residual 是否稳定需要 smoke 验证。
translation 与 rotation 应分开 normalizer 或 token type projection。
```

## 最推荐的下一步

不要马上完整复刻 SoMoFormer。

推荐先做：

```text
P7-A: independent + interaction residual gated model
```

验收标准：

```text
至少不能低于 independent 的 future_mse / long_mse。
如果 gate 学得合理，应接近 independent，并在部分 interaction cases 上改善 root distance。
```

如果 P7-A 可行，再做：

```text
P7-B: SoMoFormer-lite token Transformer
```

最后才考虑：

```text
P7-C: DCT token Transformer
```

## 论文表述建议

如果借鉴 SoMoFormer，不能写成：

```text
我们提出新的 joint trajectory Transformer。
```

更合理写法：

```text
Inspired by joint-trajectory tokenization in SoMoFormer, we adapt the idea to SMPL active-vector forecasting and study whether structured tokenization and interaction residuals can overcome the degradation observed in naive joint predictors.
```

中文：

```text
受 SoMoFormer 的关节轨迹 token 化启发，我们将 InterHuman 的 SMPL active vector 拆成 person/joint/root tokens，并把交互建模设计为 independent predictor 上的 residual correction，避免联合模型破坏单人预测能力。
```
