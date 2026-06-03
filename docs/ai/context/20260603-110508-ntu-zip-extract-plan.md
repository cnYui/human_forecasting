# NTU zip 解压与数据识别计划

## 目标

把 `/home/rpartx3080/CodeSpace/ReGenNet/drive-download-20260603T013120Z-3-001.zip` 解压到当前项目数据目录，并判断其中是否是当前缺失的数据。

## 设计判断

zip 顶层包含：

- `xsub.train.h5`
- `xsub.test.h5`

这两个文件名与 `README.zh-CN.md` 中 NTU RGB+D 120 / NTU120-AS 训练命令的 `xsub.train.h5` 和采样命令的 `xsub.test.h5` 匹配。

为了让路径与 README 和 ST-GCN 识别模型训练命令一致，解压目标采用：

```text
dataset/ntu120/smplx/conditioned/
```

而不是直接放在 `dataset/` 根目录。

## 风险控制

- 使用 `unzip -n`，避免覆盖已有文件。
- 解压后检查 H5 顶层 key、样本 shape、dtype 和样本数量。
- 再与 `data_loaders/a2m/feeder.py`、README 路径约定交叉确认数据类型。
