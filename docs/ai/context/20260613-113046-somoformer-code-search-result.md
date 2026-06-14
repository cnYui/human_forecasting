# SoMoFormer 开源代码定位记录

## 问题

用户询问本地论文 `docs/download/2022-somoformer-multi-person-pose-forecasting-transformers.pdf` 对应的开源代码位置。

## 检索计划

1. 先从本地 PDF 抽取标题、作者和 arXiv 编号，避免和同名 SoMoFormer 论文混淆。
2. 联网检索论文标题、arXiv 编号和 `SoMoFormer GitHub`。
3. 用 GitHub README / API 和 SoMoF 页面交叉确认仓库是否对应这篇论文。

## 确认信息

- 本地 PDF 标题：`SoMoFormer: Multi-Person Pose Forecasting with Transformers`。
- 本地 PDF 作者：Edward Vendrow, Satyajit Kumar, Ehsan Adeli, Hamid Rezatofighi。
- 本地 PDF arXiv：`2208.14023v1`。
- 论文第一页写明原始代码占位为 `blinded`，这是投稿匿名版本导致的占位。

## 结论

正确仓库是：

- GitHub: https://github.com/evendrow/somoformer
- Clone: `git clone https://github.com/evendrow/somoformer.git`

仓库 README 明确写明这是论文 `SoMoFormer: Multi-Person Pose Forecasting with Transformers` 的代码，并链接到 `https://arxiv.org/abs/2208.14023`。GitHub API 显示仓库描述为 `[Paper] SoMoFormer: Multi-Person Pose Forecasting with Transformers`，默认分支为 `main`，未 archived，最后 push 时间为 `2023-03-01T06:55:55Z`。

## 仓库状态

仓库不是纯占位，当前包含：

- `src/model.py`
- `src/train.py`
- `src/evaluate.py`
- `src/dataset.py`
- `src/configs/release.yaml`
- `requirements.txt`

README 提供了 3DPW / SoMoF 数据目录、训练命令和评估命令。但 README 也说明仓库仍是 work-in-progress，论文中额外使用的 AMASS 训练数据处理脚本仍在计划释放中。因此如果要严格复现论文最优结果，需要注意该仓库可能不包含完整 AMASS preprocessing pipeline。

## 注意

网上还存在另一个容易混淆的同名/近名工作：`SoMoFormer: Social-Aware Motion Transformer for Multi-Person Motion Prediction`。当前本地 PDF 对应的是 arXiv `2208.14023` 的 `Multi-Person Pose Forecasting with Transformers`，正确作者仓库是 `evendrow/somoformer`。

## 来源

- GitHub 仓库：https://github.com/evendrow/somoformer
- GitHub API：https://api.github.com/repos/evendrow/somoformer
- arXiv 页面：https://arxiv.org/abs/2208.14023
- SoMoF 结果页：https://somof.stanford.edu/results/
