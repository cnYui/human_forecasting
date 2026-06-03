# InterHuman-AS 论文 batch size 近似 smoke 结果

## 结论

通过。

本次 run 用单张 RTX 3080 通过梯度累积近似论文 batch size：

```text
layers = 8
latent_dim = 512
batch_size = 1
grad_accum_steps = 64
effective_batch_size = 64
num_steps = 1000
lambda_orient = 1
lambda_body = 1
lambda_transl = 1
```

这仍然只是论文配置口径的 1000 optimizer step smoke，不是论文 500K step 长训，也不是完整 Table 4 评估。

## 训练结果

- save_dir：`save/interhuman/paper_config_l8_d512_accum64_1000_smoke`
- 退出码：0
- wall time：`1:16:09`
- 最大 RSS：`4243284 KB`
- GPU 显存训练中约 `2396 MiB / 10240 MiB`
- 训练后 GPU 显存约 `137 MiB / 10240 MiB`

已写出 checkpoint：

```text
model000000000.pt
opt000000000.pt
model000000500.pt
opt000000500.pt
model000001000.pt
opt000001000.pt
```

已捕获的 loss：

```text
step[0]: loss[0.67129]
step[100]: loss[0.10378]
```

后续 step 日志被 tqdm 进度条密集输出截断，未完整保留；checkpoint 和退出码作为本次 smoke 的主要验收依据。

## 验证

参数核对结果：

```text
layers=8
latent_dim=512
batch_size=1
grad_accum_steps=64
num_steps=1000
lambda_orient=1.0
lambda_body=1.0
lambda_transl=1.0
```

checkpoint 加载结果：

```text
model000000000.pt: OrderedDict len=158
model000000500.pt: OrderedDict len=158
model000001000.pt: OrderedDict len=158
opt000001000.pt: dict len=2
```

残留检查：

- 训练进程已自然退出。
- 后续 `ps` 检查只命中检查命令本身，没有残留训练进程。

## 影响

已确认当前单卡可以跑论文模型尺寸和有效 batch size 64 的 InterHuman H5 训练 smoke。下一步可以进入更长 baseline 训练，但这仍然不能替代以下复现缺口：

- 论文 500K optimizer steps 长训。
- InterHuman 专用生成流程：test-conditioned actor -> generated reactor。
- DDIM-5 生成口径。
- ST-GCN recognition evaluator 和 recognition checkpoint。
- InterHuman 类别标签来源。
- SMPL-X 口径确认；当前仍是 SMPL reproduction attempt。
