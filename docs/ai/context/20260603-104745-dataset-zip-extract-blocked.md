# 数据集 zip 解压检查记录

## 用户请求

将 Windows 路径中的 zip 解压到当前项目的数据目录，并判断其中数据是否为当前项目需要的数据。

原始路径：

```text
C:\Users\yui\Downloads\drive-download-20260603T013120Z-3-001.zip
```

## 当前检查

当前项目路径：

```text
/home/rpartx3080/CodeSpace/ReGenNet
```

尝试映射到 WSL 常见路径：

```text
/mnt/c/Users/yui/Downloads/drive-download-20260603T013120Z-3-001.zip
```

结果：

- `/mnt` 下没有 `c` 盘挂载。
- `/mnt/c/Users/yui/Downloads/drive-download-20260603T013120Z-3-001.zip` 不存在。
- 在 `/home/rpartx3080`、`/mnt`、`/tmp` 范围内没有找到 `*20260603T013120Z*` 文件。

## 当前数据集状态

`dataset/` 目前可见：

- `dataset/chi3d`
- `dataset/interhuman`

主项目 README 中三套动作-反应数据集为：

- NTU RGB+D 120 / NTU120-AS
- Chi3D
- InterHuman / InterHuman-AS

本地仍缺 `dataset/ntu120`。

## 阻塞原因

目标 zip 文件对当前 Linux 环境不可见，因此不能解压，也不能判断包内数据类型。

## 下一步

需要将 zip 放到当前环境可访问的位置，例如：

```text
/home/rpartx3080/CodeSpace/ReGenNet/drive-download-20260603T013120Z-3-001.zip
```

或提供当前 Linux 环境下可访问的真实路径。拿到文件后先执行 `unzip -l` 查看目录结构，再决定是否解压到 `dataset/`，避免覆盖已有 Chi3D / InterHuman 数据。
