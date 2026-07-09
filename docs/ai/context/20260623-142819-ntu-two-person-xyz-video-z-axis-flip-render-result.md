# NTU 双人 xyz 视频 Z 轴翻转重渲结果

## 修改

更新脚本：

```text
sample/visualize_ntu_label_xyz_tricolor.py
```

新增显示坐标变换：

```text
display_xyz[...,1] = -display_xyz[...,1]
```

说明：

```text
只翻转渲染用显示坐标；
不改训练 checkpoint；
不改 eval 导出的原始 xyz 数组；
不改指标。
```

## 新输出目录

```text
results/forecasting/ntu120_label/xyz_transformer_1000_tricolor_2p_videos_zflip
```

配置确认：

```text
flip_z_axis = true
num_videos = 8
```

输出文件：

```text
videos/case0000_A001.mp4
videos/case0001_A002.mp4
videos/case0002_A003.mp4
videos/case0003_A004.mp4
videos/case0004_A005.mp4
videos/case0005_A006.mp4
videos/case0006_A007.mp4
videos/case0007_A008.mp4
```

检查：

```text
8 个 mp4 非空
8 个 first png 非空
run_config.json 记录 flip_z_axis=true
```

完整列表：

```text
results/forecasting/ntu120_label/xyz_transformer_1000_tricolor_2p_videos_zflip/summary.md
```
