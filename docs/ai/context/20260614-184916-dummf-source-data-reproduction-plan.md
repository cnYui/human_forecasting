# DuMMF 论文调研与复现资源计划

## 任务

- 深度调研论文：`docs/download/2023-stochastic-multi-person-3d-motion-forecasting.pdf`。
- 将源码拷贝到：`/home/rpartx3080/CodeSpace`。
- 下载复现论文所需数据集或确认可下载入口。

## 初步确认

- 论文标题：`Stochastic Multi-Person 3D Motion Forecasting`。
- 会议：ICLR 2023。
- 方法名：`DuMMF`，即 Dual-level generative modeling framework for Multi-person Motion Forecasting。
- 作者项目页：`https://sirui-xu.github.io/DuMMF`。
- 官方代码仓库候选：`https://github.com/Sirui-Xu/DuMMF`。

## 执行计划

1. 读取 PDF 正文与现有项目上下文，提取问题定义、模型结构、数据集、指标和复现关键点。
2. 联网核对项目页、代码仓库、README 和数据集链接，优先使用作者官方来源。
3. 将官方源码 clone 到 `/home/rpartx3080/CodeSpace/DuMMF`；如果目录已存在，先检查远端和状态，不覆盖用户改动。
4. 根据仓库说明下载或定位数据集：
   - 若作者提供可直接下载的预处理数据，下载到仓库 README 指定位置。
   - 若数据集需要申请、登录或许可证确认，只记录入口、路径和阻塞原因，不伪造复现完成状态。
5. 写结果文档，记录源码 commit、数据集状态、复现命令、与当前 ReGenNet/P7/P8 的关系和论文可引用边界。

## 边界

- 不修改 ReGenNet 现有训练代码。
- 不把 DuMMF 指标与当前 deterministic InterHuman active-vector/P7 xyz 结果直接横比。
- 数据下载以可公开访问和许可证允许为前提；需要人工申请的数据只做入口记录。
