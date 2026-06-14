# torch-dct 安装与用途记录

## 用户问题

用户表示可以安装 `torch_dct`，并询问它是用来做什么的。

## 安装环境

实际使用环境：

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python ...
```

环境信息：

```text
Python 3.7.13
pip 22.2.2
torch 1.7.1
CUDA available: true
```

## 安装命令

```text
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python -m pip install torch-dct==0.1.5
```

安装结果：

```text
Successfully installed torch-dct-0.1.5
```

## 验证

验证命令做了 CPU / CUDA 的 DCT -> IDCT roundtrip。

结果：

```text
cpu_roundtrip_max_abs: 7.152557373046875e-07
cuda_roundtrip_max_abs: 9.5367431640625e-07
```

说明 `torch_dct` 在当前环境可 import，且 CPU/CUDA 上 DCT/IDCT 基本可逆。

运行时出现 PyTorch 1.7 的 `torch.rfft` / `torch.irfft` deprecation warning，这是旧 API 警告，不影响当前环境运行。

## torch_dct 的用途

`torch_dct` 用于在 PyTorch tensor 上执行 DCT / IDCT：

```text
DCT:  Discrete Cosine Transform，离散余弦变换
IDCT: Inverse Discrete Cosine Transform，反离散余弦变换
```

在 SoMoFormer 中，它的作用是：

```text
把一段时间轨迹从 frame/time domain 转成 frequency/DCT coefficients。
```

也就是把：

```text
某个人、某个关节、某个坐标在 150 帧里的运动曲线
```

压缩成：

```text
较少的 DCT 系数，例如 dct_n=30
```

模型不直接预测每一帧，而是预测 DCT 系数，再通过 IDCT 还原整段未来轨迹。

这样做的原因：

- 动作轨迹通常在时间上比较平滑，低频 DCT 系数能表达大部分趋势。
- 模型可以一次性预测整段未来，不需要逐帧自回归。
- 对长时间预测更稳定，减少逐帧累积误差。
- 这正是 SoMoFormer 的关键思想之一：用 DCT 表示整段 joint trajectory，再让 Transformer 在 joint/person token 之间建模关系。

## 当前代码状态

P7.1 当前实现 `model/forecasting_somoformer.py` 暂时使用纯 torch DCT/IDCT 矩阵，没有依赖 `torch_dct`。原因是：

- 纯 torch 实现已通过 smoke；
- 避免额外依赖影响可复现性；
- 当前实现和 SoMoFormer 原理一致，只是实现方式不同。

后续如果要进一步贴近 SoMoFormer 原仓库实现，可以把内部 DCT/IDCT 替换为 `torch_dct.dct` / `torch_dct.idct`，但不是当前 P7.1 smoke 的阻塞项。
