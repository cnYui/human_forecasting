# 渲染修复验证记录

## 范围

本次只提交渲染相关修复：

- `render/crendermotion.py`
- `render/renderer.py`

## 改动

- 将 `torch.permute(vidmeshes, (2, 0, 1))` 改为 tensor 方法 `vidmeshes.permute(2, 0, 1)`。
- `PYOPENGL_PLATFORM` 使用 `os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')`，避免强制覆盖调用方环境。
- 渲染输出合成时使用 `rgb[:, :, :3]`，保证与三通道背景图维度一致。

## 验证

语法验证：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m py_compile render/crendermotion.py render/renderer.py
```

实际渲染验证：

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m render.crendermotion \
  --data_path results/chi3d_smoke/results.npy \
  --num_person 2 \
  --setting cmdm \
  --body_model smplx
```

结果：

```text
退出码 0
生成 8 个视频
```

生成文件：

```text
results/chi3d_smoke/rendered/Grab_0.mp4
results/chi3d_smoke/rendered/Handshake_1.mp4
results/chi3d_smoke/rendered/Hit_2.mp4
results/chi3d_smoke/rendered/HoldingHands_3.mp4
results/chi3d_smoke/rendered/Hug_4.mp4
results/chi3d_smoke/rendered/Kick_5.mp4
results/chi3d_smoke/rendered/Posing_6.mp4
results/chi3d_smoke/rendered/Push_7.mp4
```

说明：

- `results/` 已被 `.gitignore` 忽略，渲染产物不进入提交。
- ffmpeg 有 macro block size resize warning，不影响本次渲染成功。
