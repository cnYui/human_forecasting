# DuMMF 论文调研与复现资源结果

## 任务结果

- 已克隆官方源码到：`/home/rpartx3080/CodeSpace/DuMMF`。
- 官方仓库：`https://github.com/Sirui-Xu/DuMMF`。
- 当前 commit：`28bf5efc709cdd4dcd5847906920dd8af0b33c48`。
- 已下载并整理可公开直接获取的数据：
  - CMU Mocap ASF/AMC：`/home/rpartx3080/CodeSpace/DuMMF/mocap/allasfamc.zip`。
  - CMU Mocap 解压目录：`/home/rpartx3080/CodeSpace/DuMMF/mocap/all_asfamc/subjects/`。
  - MuPoTS-3D 正确包：`/home/rpartx3080/CodeSpace/DuMMF/mupots3d/MultiPersonTestSet.zip`。
  - MuPoTS-3D 标注路径：`/home/rpartx3080/CodeSpace/DuMMF/mupots3d/data/TS{1..20}/annot.mat`。
  - SoMoF PoseTrack：`/home/rpartx3080/CodeSpace/DuMMF/somof_data.zip` 和 `somof_data_posetrack/`。
  - SoMoF 3DPW processed：`/home/rpartx3080/CodeSpace/DuMMF/somof_data_3dpw.zip` 和 `somof_data_3dpw/`。

## 论文核心

论文：`Stochastic Multi-Person 3D Motion Forecasting`，ICLR 2023。

作者提出 stochastic multi-person 3D motion forecasting：输入多人历史 3D motion，输出多个可能未来。目标不是只做一个 deterministic 最小误差预测，而是同时处理：

- single-person fidelity：每个人动作要真实、连续、身体合理。
- multi-person fidelity：多人互动要合理，避免各自独立生成导致碰撞或不协调。
- overall diversity：多个未来要有多样性。

方法名为 `DuMMF`：Dual-level generative modeling framework for Multi-person Motion Forecasting。

核心机制：

- local level：每个人使用独立 intent code，鼓励个人运动真实性和多样性。
- global level：同一个预测样本内所有人共享 intent code，建模 social intent 和多人互动一致性。
- intent code 包含 learnable discrete intent codes 和 continuous intent codes。
- 框架可接 CGAN / DDPM 和 SC-MPF / XIA / MRT 等多人预测 backbone。

## 数据和指标

论文使用的数据：

- CMU-Mocap：骨架版主实验，构造 2-person / 3-person 预测任务。
- MuPoTS-3D：主要用于 CMU 训练后的泛化评估。
- SoMoF / 3DPW：附录 benchmark 实验。
- AMASS / SMPL-X 或 SMPL-H 表示：论文 Table 1 的 diffusion/mesh 表示实验。

主要指标：

- accuracy：Best-of-N ADE / FDE。
- diversity：FPD。
- 分解指标：rootADE / rootFDE / rootFPD，poseADE / poseFDE / poseFPD。
- 附加指标：lADE / lFDE、foot skating ratio、trajectory collision ratio、average human displacement 等。

## 源码复现状态

公开仓库 README 只说明 `version 1.0` 是 diffusion-based DuMMF，没有完整给出数据下载命令或依赖安装清单。

当前源码入口：

- diffusion 训练：`train_diffusion.py`。
- diffusion 数据：`data_amass.py`，默认读取 `./mocap/CMU/` 下 AMASS CMU `.npz`。
- CMU 骨架预处理：`mocap/preprocess_mocap.py` 和 `mocap/mix_mocap.py`。
- MuPoTS 预处理：`mupots3d/preprocess_mupots.py`。

关键边界：

- `train_diffusion.py` 的默认路径不是 CMU ASF/AMC，而是 AMASS-CMU 的 SMPL-H/DMPL `.npz`：
  - `./mocap/CMU/`
  - `./mocap/body_models/smplh/{neutral,male,female}/model.npz`
  - `./mocap/body_models/dmpls/{neutral,male,female}/model.npz`
- AMASS 和 SMPL/SMPL-H body models 通常需要账号、研究协议和手动下载；本次匿名直连 AMASS 入口两次 60 秒超时，未能合法自动下载。
- 因此，当前已完成的是可公开直接下载的数据落地；diffusion Table 1 级复现仍需要人工准备 AMASS-CMU 和 SMPL-H/DMPL body models。

## 已做验证

- 源码 commit 已确认：`28bf5efc709cdd4dcd5847906920dd8af0b33c48`。
- CMU 解压后存在 `subjects/18/18.asf`、`subjects/18/18_01.amc` 等关键文件；`subjects` 一级目录计数为 `113`。
- MuPoTS 正确包包含 `MultiPersonTestSet/TS1..TS20/annot.mat`。
- 已抽取 20 个 MuPoTS `annot.mat`，并通过软链接对齐为 `mupots3d/data/TS1..TS20/annot.mat`。
- SoMoF 已解压 PoseTrack 和 3DPW 的 JSON 数据，共 `21` 个 JSON 文件；3DPW 图像保留在 zip 中，未展开。

## 复现建议

第一步只做数据预处理 smoke：

```bash
cd /home/rpartx3080/CodeSpace/DuMMF/mocap
python preprocess_mocap.py
python mix_mocap.py
```

MuPoTS 预处理：

```bash
cd /home/rpartx3080/CodeSpace/DuMMF/mupots3d
python preprocess_mupots.py
```

diffusion 训练需要先补：

```text
/home/rpartx3080/CodeSpace/DuMMF/mocap/CMU/
/home/rpartx3080/CodeSpace/DuMMF/mocap/body_models/smplh/
/home/rpartx3080/CodeSpace/DuMMF/mocap/body_models/dmpls/
```

否则 `data_amass.py` 和 `train_diffusion.py` 不能按默认配置运行。

## 与 ReGenNet 的关系

DuMMF 是 stochastic 多未来生成，不应与当前 ReGenNet deterministic InterHuman 150/30/120 或 P7/P8 xyz 结果直接横比。

可借鉴点：

- local/global dual-level intent 设计。
- individual intent 与 social intent 的分离训练。
- Best-of-N accuracy 与 diversity 指标共同报告。

不应直接照搬的点：

- 当前 ReGenNet 主线是 deterministic forecasting；引入 DuMMF 意味着问题定义变成多未来生成。
- DuMMF 的 AMASS/SMPL-H diffusion 路径与 ReGenNet InterHuman active-vector/xyz 路径不是同一数据协议。
