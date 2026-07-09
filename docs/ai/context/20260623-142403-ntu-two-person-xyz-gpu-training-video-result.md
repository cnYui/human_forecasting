# NTU 双人 xyz GPU 训练和视频结果

## GPU 问题结论

最初 `nvidia-smi` 失败的原因不是 RTX 3080 或 CUDA 损坏，而是当时工具会话处在受限环境，没有暴露 `/dev/nvidia*` 设备节点。

权限环境切换后复查：

```text
nvidia-smi: NVIDIA GeForce RTX 3080, 10240 MiB
PyTorch: torch 1.7.1, cuda_built 11.0
torch.cuda.is_available() = True
device_count = 1
device name = NVIDIA GeForce RTX 3080
```

## 完整 xyz cache

输出目录：

```text
results/forecasting/ntu120_label/xyz_cache_len60_o20_p40
```

train cache：

```text
path = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/train_xyz.pt
obs = [1956,20,2,55,3]
target = [1956,40,2,55,3]
26 labels covered
min class count = 2
finite = true
```

test cache：

```text
path = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/test_xyz.pt
obs = [1253,20,2,55,3]
target = [1253,40,2,55,3]
26 labels covered
min class count = 1
finite = true
```

## 1000-step GPU 训练

训练命令使用：

```text
train.train_ntu_label_xyz
batch_size = 64
latent_dim = 256
encoder_layers = 3
decoder_layers = 3
device = cuda
```

checkpoint：

```text
save/forecasting/ntu120_label/xyz_transformer_len60_o20_p40_h256_l3_s0_1000/model000001000.pt
```

full test 结果：

```text
num_samples = 1253
model xyz_mse = 0.031281803
copy  xyz_mse = 0.057248837
model xyz_mae = 0.091381698
copy  xyz_mae = 0.120007492
model mpjpe = 0.189595376
copy  mpjpe = 0.260575039
first_step_error = 0.0
```

结论：

```text
完整 test set 上，模型在 MSE、MAE、MPJPE 上都超过 copy-last；
generated 第一帧结构性等于 obs 最后一帧，满足“不跳离最后一帧延续”的硬边界。
```

## 8 个双人三色视频

8 样本导出目录：

```text
results/forecasting/ntu120_label/xyz_transformer_1000_eval8
```

视频输出目录：

```text
results/forecasting/ntu120_label/xyz_transformer_1000_tricolor_2p_videos
```

颜色：

```text
蓝色 = input obs20
橙色 = generated future40
绿色 = real future40
```

边界：

```text
这是直接 xyz skeleton 双人可视化，shape 为 [T,2,55,3]；
不再使用旧的单人 rot6d -> SMPL-X 转换。
```

8 样本注意事项：

```text
8 样本子集上 model MSE 略优于 copy-last；
但 MAE/MPJPE 没有整体优于 copy-last。
该 8 样本只用于人工看视频，不代表 full test 总体结论。
```

视频列表：

```text
case0000_A001.mp4
case0001_A002.mp4
case0002_A003.mp4
case0003_A004.mp4
case0004_A005.mp4
case0005_A006.mp4
case0006_A007.mp4
case0007_A008.mp4
```

完整路径见：

```text
results/forecasting/ntu120_label/xyz_transformer_1000_tricolor_2p_videos/summary.md
```
