# NTU 双人 xyz 视频 Z 轴翻转重渲计划

## 问题

用户检查 8 个双人 xyz skeleton 视频后指出：

```text
Z 轴应该反掉了，脚朝上、头朝下。
```

## 判断

当前视频脚本 `sample/visualize_ntu_label_xyz_tricolor.py` 使用：

```text
matplotlib x = xyz[...,0]
matplotlib y = xyz[...,2]
matplotlib z = xyz[...,1]
```

视频里“脚朝上、头朝下”对应渲染竖直轴方向错误。这里不改训练结果、不改 `.pt/.npz` 里的原始 xyz，只在渲染显示坐标上翻转竖直轴：

```text
display_xyz[...,1] = -display_xyz[...,1]
```

## 输出

重新渲染到新目录，避免覆盖上一版：

```text
results/forecasting/ntu120_label/xyz_transformer_1000_tricolor_2p_videos_zflip
```
