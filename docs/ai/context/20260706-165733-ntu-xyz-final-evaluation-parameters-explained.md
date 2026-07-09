# NTU xyz 最终评估参数说明

## 结论口径

当前 ReGenNet 项目最后用于判断模型预测准确性的主线，是 NTU120 双人 xyz 预测评估：

```text
eval/eval_ntu_label_xyz.py
```

对应当前完整测试结果：

```text
save/forecasting/ntu120_label/xyz_transformer_len60_o20_p40_h256_l3_s0_1000/metrics_test.json
```

旧的 CMDM diffusion 路线仍有距离评估和动作一致性分类器：

```text
eval/eval_label_forecasting_distance.py
eval/action_consistency_classifier.py
```

但根据现有上下文，动作一致性分类器只能作为辅助说明；最终准确性主结论必须看 xyz 空间误差，并且必须和 copy-last baseline 比。

## 当前最终评估运行参数

| 参数 | 当前值 | 大白话解释 |
|---|---:|---|
| mode | checkpoint | 用训练好的模型 checkpoint 做评估。 |
| dataset | ntu120_2p_smplx | 评估的是 NTU120 双人 SMPL-X 数据。 |
| split | test | 用测试集，不用训练集。 |
| data_path | dataset/ntu120/smplx/conditioned/xsub.test.h5 | 测试集原始 H5 路径。 |
| xyz_cache | results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/test_xyz.pt | 已提前转好的 xyz cache，避免评估时重复跑 SMPL-X 转换。 |
| window_len | 60 | 每条样本总共看 60 帧。 |
| obs_len | 20 | 前 20 帧给模型当已知输入。 |
| pred_len | 40 | 后 40 帧是模型要预测的目标。 |
| num_samples | 1253 | 测试集实际评估了 1253 个窗口样本。 |
| batch_size | 128 | 每批评估 128 条样本；只影响速度和显存，不改变指标含义。 |
| checkpoint | save/forecasting/ntu120_label/xyz_transformer_len60_o20_p40_h256_l3_s0_1000/model000001000.pt | 被评估的模型权重。 |
| checkpoint_step | 1000 | 该模型训练到第 1000 step。 |

## 主评估指标

| 指标 | model 当前值 | copy-last 当前值 | 越小越好 | 大白话解释 |
|---|---:|---:|---|---|
| xyz_mse | 0.031281803 | 0.057248837 | 是 | 所有预测关节坐标的平方误差平均；大错误会被平方放大。 |
| xyz_mae | 0.091381698 | 0.120007492 | 是 | 所有预测关节坐标的绝对误差平均；更像“平均差了多少”。 |
| mpjpe | 0.189595376 | 0.260575039 | 是 | 每个关节的 3D 欧氏距离误差平均；看骨架位置准不准。 |
| first_step_error | 0.0 | 0.0 | 是 | 预测第 1 帧离观测最后 1 帧有多远；当前模型结构保证不跳帧。 |
| velocity_error | 0.031316057 | 0.032093865 | 是 | 预测动作速度变化和真实速度变化的差；看动起来是否接近。 |
| root_translation_error | 0.110962798 | 0.181037420 | 是 | 人体根节点位置误差；看整个人的位置移动准不准。 |
| relative_root_distance_error | 0.105562203 | 0.200716256 | 是 | 两个人根节点之间距离的误差；看两人相对距离准不准。 |
| inter_person_distance_consistency | 0.014223375 | 0.017718861 | 是 | 两人距离变化趋势的误差；看互动距离变化是否更合理。 |

最终 gate：

```text
beats_copy_last.xyz_mse = true
beats_copy_last.xyz_mae = true
beats_copy_last.mpjpe = true
```

也就是当前最终模型在完整 test set 上，三个主指标都超过“直接复制最后一帧”的 baseline。

## 辅助分类器指标

动作一致性分类器用于回答“生成动作看起来是否像条件标签”，不是最终几何准确性主指标。

| 指标 | 当前值 | 大白话解释 |
|---|---:|---|
| top1_acc | 0.859537111 | 真实 future40 输入分类器时，第一预测标签命中真实动作的比例。 |
| top5_acc | 0.969672785 | 真实动作标签出现在分类器前 5 个预测里的比例。 |
| balanced_acc | 0.597516674 | 先算每个动作类准确率再平均，减轻类别不均衡影响。 |
| handshaking_acc | 0.911764706 | handshaking 类的分类准确率。 |
| classifier_gate_pass | true | 分类器自身够用，可以作为辅助检查工具。 |
| consistency_acc | 0.0 | 旧 diffusion 生成结果被分类器判成输入条件标签的比例；该结果不能支持“动作语义控制成功”。 |

## 解释边界

- 最终准确性主结论看 xyz 几何误差，不看分类器一致性。
- copy-last 是硬 baseline：如果模型连“最后一帧不动”都比不过，预测没有说服力。
- 当前 xyz 模型完整测试集上超过 copy-last，但 8 个视频样本只用于人工观察，不能替代 full test 指标。
