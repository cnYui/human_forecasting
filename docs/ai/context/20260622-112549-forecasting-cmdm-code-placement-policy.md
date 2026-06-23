# ForecastingCMDMDecoder 代码保存位置规范

## 结论

新代码不建议放到单独的隔离源码文件夹，例如：

```text
new_code/
my_code/
experiments/
forecasting_cmdm_project/
```

原因：

```text
这会绕开现有项目模块边界，导致 import、checkpoint、训练入口、评估入口和后续提交都更难维护。
本任务是 ReGenNet 内部新增一个正式 forecasting diffusion baseline，不是外部仓库复现实验。
```

推荐做法是在现有目录结构内按职责放置，同时用文件名明确隔离新任务。

## 推荐位置

### Dataset

```text
data_loaders/forecasting/ntu_label.py
```

用途：

```text
NTU120 2P H5 读取
A001-A026 标签解析
T>=60 过滤
obs20/future40 切分
train random crop / test center crop
forecasting label collate
```

如需导出公共入口，轻量更新：

```text
data_loaders/forecasting/__init__.py
```

### Model

```text
model/forecasting_cmdm.py
```

用途：

```text
ForecastingCMDMDecoder
ForecastingClassifierFreeSampleModel
shape guard
config / count_parameters
```

优先从 `model/cmdm.py` import 可复用组件：

```text
InputProcess
OutputProcess
PositionalEncoding
TimestepEmbedder
EmbedAction
```

只有当 import 明确产生循环依赖或副作用时，才把公共组件无行为改动抽到：

```text
model/cmdm_components.py
```

### Training

```text
train/train_label_forecasting_diffusion.py
```

用途：

```text
NTU label forecasting diffusion 训练入口
2-step smoke
正式 5000-step
checkpoint / resume
train_log.jsonl
```

不建议改造：

```text
train/train_mdm.py
```

原因：

```text
原始 train_mdm.py 对应 actor->reactor CMDM 协议，不是 obs20+label->future40 forecasting 协议。
```

### Evaluation

```text
eval/eval_label_forecasting_diffusion.py
eval/action_consistency_classifier.py
```

用途：

```text
预测误差评估
per-class / handshaking subset 指标
动作一致性分类器 gate
generated future40 classification consistency
```

### Sampling

若仓库已有或后续确定使用 `sample/` 作为采样脚本目录：

```text
sample/sample_label_forecasting_diffusion.py
```

如果不想新增 `sample/` 目录，也可以先放在：

```text
eval/sample_label_forecasting_diffusion.py
```

取舍：

```text
sample/ 更清楚地区分“生成输出”和“评估指标”。
eval/ 更贴合当前仓库已有入口，少建目录。
```

本项目当前已有 `sample/visualize_forecasting_xyz.py` 的未跟踪文件，因此后续如果确认 `sample/` 会成为正式目录，可以把 label swap sampling 放入 `sample/`。

### Small Scripts

小型一次性检查脚本建议放：

```text
scripts/check_ntu_label_forecasting_data.py
```

如果仓库没有正式 `scripts/` 目录，也可以先放：

```text
data_loaders/forecasting/check_ntu_label_data.py
```

但长期更推荐 `scripts/`，避免把 CLI 检查脚本混进 dataset 模块。

### Outputs

训练输出：

```text
save/forecasting/ntu120_label/...
```

采样和评估输出：

```text
results/forecasting/ntu120_label/...
```

不要提交大文件：

```text
model*.pt
opt*.pt
*.npy
*.mp4
大型 metrics dump
```

可以提交小型配置或示例：

```text
configs/forecasting/ntu120_label/*.json
```

前提是后续确实需要固定命令配置。

### AI Context

所有阶段计划、设计取舍、测试结果继续放：

```text
docs/ai/context/YYYYMMDD-HHMMSS-*.md
```

不要覆盖旧文档。

## 是否需要隔离原始代码

需要隔离的是“协议边界”，不是“目录物理隔离”。

正确隔离方式：

```text
新文件名明确使用 forecasting_cmdm / ntu_label / label_forecasting_diffusion
不修改原始 CMDM forward 语义
不复用 train_mdm.py 作为新协议入口
不把 obs20 塞进 y["cmotion"] workaround
旧 Encoder-only 只作为 ablation 明确命名
```

不推荐的隔离方式：

```text
复制一份 model/、train/、data_loaders/ 到新目录
在根目录创建独立实验工程
把正式 baseline 放到临时 notebook 或杂散脚本
```

## 推荐最终文件清单

第一轮实现建议新增：

```text
data_loaders/forecasting/ntu_label.py
model/forecasting_cmdm.py
train/train_label_forecasting_diffusion.py
eval/eval_label_forecasting_diffusion.py
eval/action_consistency_classifier.py
sample/sample_label_forecasting_diffusion.py
scripts/check_ntu_label_forecasting_data.py
```

最少可先实现：

```text
data_loaders/forecasting/ntu_label.py
model/forecasting_cmdm.py
train/train_label_forecasting_diffusion.py
scripts/check_ntu_label_forecasting_data.py
```

后续在 2-step smoke 通过后，再补采样和动作一致性评估。
