# 数据集清点记录

## 结论

ReGenNet 主项目 README 的正式动作-反应数据集是三套：

- NTU RGB+D 120 / NTU120-AS
- Chi3D
- InterHuman / InterHuman-AS

本地当前实际落盘的数据集目录只有两套：

- `dataset/chi3d`
- `dataset/interhuman`

因此，如果目标是补齐 README 主线三套数据集，本地还缺：

- `dataset/ntu120` 对应的 NTU RGB+D 120 / NTU120-AS 处理后数据

## 依据

- `README.zh-CN.md` 的“数据准备”只列出 `NTU RGB+D 120`、`Chi3D`、`InterHuman`。
- `dataset/` 当前只有 `chi3d` 和 `interhuman` 两个数据集目录。
- `dataset/chi3d/smplx/conditioned/` 下已有 `chi3d_smplx_train.h5` 和 `chi3d_smplx_test.h5`。
- `dataset/interhuman/smpl/conditioned/` 下已有 `interhuman_train.h5`、`interhuman_val.h5`、`interhuman_test.h5` 和 `meta.json`。
- `data_loaders/get_data.py` 还保留 `humanact12`、`uestc`、`amass` 等上游兼容入口，但这些不是 README 数据准备主线中的三套 ReGenNet 动作-反应数据集。

## 注意

如果问题中的“还需要哪两个”指的是代码兼容入口而不是 README 主线，则额外缺本地数据目录的是：

- `dataset/HumanAct12Poses`
- `dataset/uestc`

但它们更像上游 MDM/ACTOR 单人动作生成兼容数据集，不是 ReGenNet 动作-反应主实验三件套。
