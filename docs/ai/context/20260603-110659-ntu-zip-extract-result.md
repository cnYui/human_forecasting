# NTU zip 解压与数据识别结果

## 解压结果

已将：

```text
/home/rpartx3080/CodeSpace/ReGenNet/drive-download-20260603T013120Z-3-001.zip
```

解压到：

```text
dataset/ntu120/smplx/conditioned/
```

生成文件：

- `dataset/ntu120/smplx/conditioned/xsub.train.h5`
- `dataset/ntu120/smplx/conditioned/xsub.test.h5`

## H5 检查

`xsub.train.h5`：

- 样本数：4273
- key 示例：`S001C001P001R001A001`
- 样本 shape 示例：`(74, 56, 6)`
- dtype：`float32`
- 动作编号范围：`A001` 到 `A026`
- 帧数范围：18 到 203

`xsub.test.h5`：

- 样本数：3845
- key 示例：`S001C001P003R001A001`
- 样本 shape 示例：`(83, 56, 6)`
- dtype：`float32`
- 动作编号范围：`A001` 到 `A026`
- 帧数范围：17 到 174

## 判断

这是 NTU RGB+D 120 / NTU120-AS 的双人交互 SMPL-X H5 数据。

依据：

- 文件名 `xsub.train.h5`、`xsub.test.h5` 与 README 中 NTU 训练/采样命令一致。
- 样本 key 使用 NTU 标准命名格式 `SxxxCxxxPxxxRxxxAxxx`。
- 动作编号覆盖 `A001` 到 `A026`，与项目 `data_loaders/a2m/feeder.py` 中 `ntu` 双人动作枚举的 26 类一致。
- 样本 shape 为 `[T, 56, 6]`，符合本项目 NTU/Chi3D SMPL-X feeder 的读取方式。

因此，如果目标是补齐 ReGenNet README 主线三套数据集，这是当前缺失的 NTU120-AS 数据，是需要的数据。

如果目标是补代码兼容入口里的 `HumanAct12Poses` 或 `UESTC`，则这个 zip 不是那两个数据集。
