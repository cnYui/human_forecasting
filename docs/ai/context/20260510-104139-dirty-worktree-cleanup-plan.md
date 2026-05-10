# 脏工作区整理计划

## 目标

把 P3 之后残留的脏修改按来源拆分，避免后续 InterHuman baseline 训练混入无关文件。

## 处理顺序

1. 保护当前状态：保存当前 dirty diff 摘要，必要时可从 git diff 恢复。
2. 外部资产和实验输出：更新 `.gitignore`，不提交大文件、模型、结果和本地工具。
3. README：不直接覆盖官方 `README.md`，改为新建中文文档 `README.zh-CN.md`，再恢复 `README.md`。
4. 渲染修复：单独验证并提交 `render/*` 的行为修复。
5. 注释类改动：单独验证并提交 `utils/model_util.py`、`utils/parser_util.py` 以及对应 context 文档。

## 不处理范围

- 不删除本地大文件和实验输出，只通过 `.gitignore` 避免误提交。
- 不回滚 P3 已提交内容。
- 不修改 main 分支。
