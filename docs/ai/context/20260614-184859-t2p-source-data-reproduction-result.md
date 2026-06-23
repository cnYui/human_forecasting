# T2P 源码与数据复现结果

## 对象

论文：

```text
Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning
Jaewoo Jeong, Daehee Park, Kuk-Jin Yoon
CVPR 2024 Highlight
```

本地 PDF：

```text
docs/download/2024-multi-agent-long-term-3d-human-pose-forecasting-trajectory-conditioning.pdf
```

本次计划：

```text
docs/ai/context/20260614-183846-t2p-source-data-reproduction-plan.md
```

## 源码

已克隆官方仓库到：

```text
/home/rpartx3080/CodeSpace/T2P
```

仓库信息：

```text
origin: https://github.com/Jaewoo97/T2P
HEAD: 25ccf4fbc5a2e4fecdb2aee250201d7d8849ecda
short: 25ccf4f Update t2p_jrdb.yaml
```

## 已下载数据

已从 T2P 官方 GitHub release 下载并解压可公开获取的 release 数据：

```text
/home/rpartx3080/CodeSpace/T2P/downloads/data.zip
/home/rpartx3080/CodeSpace/T2P/downloads/odometry_processed.zip
/home/rpartx3080/CodeSpace/T2P/data/
/home/rpartx3080/CodeSpace/T2P/downloads/odometry_processed/
```

文件大小：

```text
data.zip: 约 1.4G
odometry_processed.zip: 126K
解压后 data/: 约 1.8G
```

`data.zip` 内容：

```text
data/Mocap_UMPM/mix1_6persons.npy
data/Mocap_UMPM/mix2_10persons.npy
data/Mocap_UMPM/mupots_150_2persons.npy
data/Mocap_UMPM/mupots_150_3persons.npy
data/Mocap_UMPM/test_3_75_mocap_umpm.npy
data/Mocap_UMPM/train_3_75_mocap_umpm.npy
data/poseData.pkl
```

基本读取检查通过：

```text
mix1_6persons.npy: (1000, 6, 75, 45), float64
mix2_10persons.npy: (1000, 10, 75, 45), float64
mupots_150_2persons.npy: (176, 2, 150, 45), float64
mupots_150_3persons.npy: (192, 3, 150, 45), float64
test_3_75_mocap_umpm.npy: (3000, 3, 75, 45), float64
train_3_75_mocap_umpm.npy: (13000, 3, 75, 45), float64
poseData.pkl: dict, len=3
```

`odometry_processed.zip` 解压后有 `54` 个 `.npy` 文件，即 27 个 JRDB 场景各自的 `pos/orientation` 文件。样例读取通过：

```text
bytes-cafe-2019-02-07_0_pos.npy: (1726, 3), float32
```

## JRDB-GMP 复现状态

T2P README 明确要求：

1. 从 JRDB 官方网站下载原始 JRDB 数据。
2. 下载 release 中的 preprocessed robot odometry。
3. 运行 `preprocess_1st_jrdb.py` 和 `preprocess_2nd_jrdb.py`。

当前只完成第 2 步。第 1 步未自动完成，原因是 JRDB 官方页面显示下载需要登录或注册账号：

```text
Please note that you are required to log in before downloading JRDB.
Looks like you're not logged in!
```

因此不能在未登录授权状态下下载完整 JRDB 原始数据。需要用户在 JRDB 官方站点注册/登录后获取数据下载权限，再把原始数据放到本机。

官方 JRDB 入口：

```text
https://jrdb.erc.monash.edu/
```

## 复现风险

当前官方仓库不是开箱即跑的完整论文复现包，主要风险：

1. `lightning_train.py` 默认 Hydra config 是 `train_config_jrdb_t2p_v3_2`，但当前仓库 `conf/` 下只有 `train_config_jrdb_t2p.yaml`、`train_config_cmu_t2p.yaml`、`train_config_3dpw_t2p.yaml`。
2. `lightning_eval.py` 默认 config 是 `eval_config_cmu_t2p.yaml`，但当前仓库没有任何 `eval_config*.yaml`。
3. `preprocess_1st_jrdb.py`、`preprocess_2nd_jrdb.py`、`dataset/t2p_dataset.py`、`dataset/3dpw_hivt.py`、`dataset/Mocap_UMPM_hivt.py` 中有大量作者机器绝对路径，如 `/mnt/jaewoo4tb/...`、`/ssd4tb/...`。
4. JRDB preprocessing 依赖 BEV/ROMP、SMPL/SMPL-A 文件、torch-geometric、HiVT `TemporalData` 兼容对象，以及原始 JRDB 的 labels/images 目录。
5. README 写明当前上传的是更新版 JRDB parser：`3D joints => SMPL parameters for pose`，使用 SMPL theta `24x3`。这和 CVPR 论文正文中 joint-position 口径不完全等价，复现实验时必须记录版本差异。

## 可运行检查

已完成：

```text
file data.zip odometry_processed.zip
unzip -l data.zip
unzip -l odometry_processed.zip
release 数据基本读取
py_compile lightning_train.py lightning_eval.py preprocess_1st_jrdb.py preprocess_2nd_jrdb.py dataset/t2p_dataset.py
```

`py_compile` 通过只能说明语法可解析，不能说明依赖、路径和数据已满足。

## 后续复现步骤

1. 用户登录 JRDB 官方站点并下载 T2P README 需要的原始 JRDB train dataset，至少包含：
   - `labels/labels_2d_stitched/`
   - `labels/labels_3d/`
   - `labels/labels_2d_activity_social_stitched/`
   - `images/image_stitched/`
   - `images/image_0/`
2. 把 JRDB 数据放到一个固定本地路径，例如：

```text
/home/rpartx3080/CodeSpace/T2P/jrdb/train_dataset/
```

3. 参数化或修改 T2P 中的硬编码路径：
   - `preprocess_1st_jrdb.py`
   - `preprocess_2nd_jrdb.py`
   - `dataset/t2p_dataset.py`
4. 准备 BEV/ROMP 和 SMPL/SMPL-A 模型文件。
5. 先跑单 scene preprocessing smoke，再跑完整 JRDB-GMP preprocessing。
6. 修复 Hydra config 名称不匹配后再启动训练或评估。

## 对当前 ReGenNet 的结论

T2P 的论文价值仍是 trajectory-first / local-pose-conditioned 结构。它适合作为下一阶段模型设计参考，但不应直接拿官方仓库当前输出和 ReGenNet P5/P7/P8 结果横比：

- T2P 是 JRDB-GMP / in-the-wild pseudo 3D / multi-agent / best-of-K 多模态预测。
- 当前 ReGenNet 主协议是 InterHuman / deterministic two-person / active-vector 或 xyz 同口径预测。
- 两者数据、表示、metric 和评估口径都不同。
