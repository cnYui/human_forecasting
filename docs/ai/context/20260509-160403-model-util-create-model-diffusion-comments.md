# model_util.py 创建模型与 diffusion 注释计划

## 背景

用户希望给 `utils/model_util.py` 中 `create_model_and_diffusion(args, data)` 每一行加中文注释，解释模型和 diffusion 是如何创建的。

## 范围

- 只修改 `utils/model_util.py` 的 `create_model_and_diffusion()` 函数附近注释。
- 不修改任何训练逻辑、参数、返回值或函数调用顺序。
- 保持当前 `setting == 'cmdm'` 的行为不变。

## 注释重点

- `args.setting` 用来选择训练框架。
- `CMDM(**get_model_args(args, data))` 先根据参数和数据集信息创建网络。
- `args.num_person = 1` 是因为 diffusion 只处理拆分后的 reactor 单人目标动作。
- `create_gaussian_diffusion(args)` 创建扩散训练/采样规则。
- 返回的 `model, diffusion` 会交给训练循环使用。

## 验证

- 修改后运行 `python3 -m py_compile utils/model_util.py`。
