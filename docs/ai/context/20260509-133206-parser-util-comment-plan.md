# parser_util.py 中文注释补充计划

## 背景

用户希望按 `train/train_mdm.py` 中的编号式中文注释风格，继续解释 `utils/parser_util.py`，当前关注点在 `utils/parser_util.py:255` 附近。

## 必须保持不变

- 不改变任何 argparse 参数名、默认值、choices、required 设置。
- 不改变训练、采样、编辑、评估入口的参数组合顺序。
- 不改变 `parse_and_load_from_model` / `parse_and_load_from_model_wo_data` 的覆盖逻辑。

## 注释策略

- 只补充中文注释，解释参数组和入口函数的用途。
- 使用类似 `train_mdm.py` 的编号式注释，让读者能顺着执行路径理解。
- 重点说明 `rec_model_path` 是评估用动作识别模型，不是 ReGenNet 生成模型 checkpoint。

## 验证

- 修改后运行 `python -m py_compile utils/parser_util.py`，确认注释改动没有破坏语法。
