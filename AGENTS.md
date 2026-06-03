# ReGenNet AI 上下文记忆

## InterHuman-AS Table 4

- Table 4 主指标是 `FID / Acc. / Div. / Multimod.`，不是 `R Precision / MM Dist`。
- Table 4 主路径应使用 ST-GCN recognition feature evaluator；text-motion evaluator 只作为后续扩展。
- 当前本地 InterHuman 数据没有 text/caption 目录，也没有 InterHuman recognition checkpoint。
- 当前实现先走 SMPL reproduction attempt；若要精确对齐论文的 SMPL-X 口径，需要单独补 SMPL-X 转换或确认官方 evaluator 口径。

## 已完成进度

- 已新增 InterHuman-AS H5 预处理脚本：`preprocess/interhuman_as.py`。
- 已生成冻结 H5：`dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5`。
- 已为 `data_loaders/a2m/interhuman.py` 增加 H5 读取路径，同时保留在线 `.pkl` loader。
- H5 输出与在线 loader 抽样对齐，最大误差为 0。
- H5 loader 最小训练 smoke 已通过。
- H5 full smoke 已通过：`save/interhuman/h5_full_1000_smoke_noworkers/model000001000.pt`。
- InterHuman 训练当前固定 `num_workers=0`，原因是 PyTorch 1.7 多 worker 在提前按 `num_steps` 停止时会阻塞进程退出。
- 已实现 P3 `--grad_accum_steps`；`num_steps` 表示 optimizer steps，`effective_batch_size=batch_size*world_size*grad_accum_steps`。
- `grad_accum_steps=1` 和 `grad_accum_steps=2` 的短 smoke 已通过，checkpoint 可加载。
- `num_steps=1000, grad_accum_steps=16` 中等 smoke 已通过，耗时约 7 分钟，最终 checkpoint：`save/interhuman/p3_grad_accum_16_1000_smoke/model000001000.pt`。
- 论文尺寸模型 smoke 已通过：`layers=8, latent_dim=512, lambda_orient/body/transl=1, grad_accum_steps=4, num_steps=100`。
- 已修复 InterHuman H5 显式 translation loss 的 NaN：translation slot 使用最后一个 joint slot，而不是硬编码 index 55。
- 论文 batch size 近似 smoke 已通过：`layers=8, latent_dim=512, batch_size=1, grad_accum_steps=64, effective_batch_size=64, num_steps=1000`，耗时 `1:16:09`，最终 checkpoint：`save/interhuman/paper_config_l8_d512_accum64_1000_smoke/model000001000.pt`。
- 50K baseline 已启动：`layers=8, latent_dim=512, batch_size=1, grad_accum_steps=64, num_steps=50000, save_interval=5000`，保存目录：`save/interhuman/paper_config_l8_d512_accum64_50000_baseline`，launcher PID 记录在 `train.pid`。
- 50K baseline 已通过早期 `step[100]` 检查：`loss=0.10378`，训练已进入 `epoch 2:532`。

## 下一步

- 监控 50K baseline：等待 `model000005000.pt` 和 `opt000005000.pt` 写出。
- 补 InterHuman 专用生成流程：DDIM-5，test-conditioned actor -> generated reactor。
- P5 前必须确认 InterHuman 类别标签来源和 recognition checkpoint 路线。

## 2026-06-03 论文目标（已正式锁定）

- 当前更合适的论文主线不是继续追 ReGenNet Table 4 完整复现，而是转为 `Interaction-aware joint forecasting of two-person human motion from partial observations`。
- 中文表述：基于部分观测的交互感知双人动作联合未来预测。
- 第一阶段建议锁定 InterHuman：固定 150 帧窗口，前 30 帧作为观测，联合预测后 120 帧双人动作。
- `前 20% / 后 80%` 应作用在固定窗口上，不应直接作用在 InterHuman 原始全长序列上；过短序列第一阶段过滤，不用 padding 伪造未来。
- 核心创新应是 relation-aware joint predictor，从可观测的双人相对位置、速度、朝向和关节距离中学习交互关系。
- 必须比较 repeat/zero-velocity、independent predictor、concat no-relation predictor 和 relation-aware predictor。
- 关键指标除 MSE / rotation MSE / translation MSE 外，还需要 relative root distance error、relative orientation error、inter-person distance consistency 和 long-horizon error。
- ReGenNet InterHuman 50K baseline 定位调整为历史 backbone / 生成式 baseline / 工程资产，不再作为最终论文任务本身。
- 上下文草案见 `docs/ai/context/20260603-155738-paper-final-goal-draft.md`。
- 2026-06-03 已按用户要求停止当前 ReGenNet 50K baseline 长跑进程；最后可用 checkpoint 为 `save/interhuman/paper_config_l8_d512_accum64_50000_baseline/model000020000.pt`。
- 正式实现设计见 `docs/ai/context/20260603-160713-forecasting-final-goal-design.md`。
- P1-P6 完整路线图见 `docs/ai/context/20260603-161803-forecasting-p1-p6-roadmap.md`；后续每个阶段完成时必须新建阶段记录并引用该路线图。
- 已用 `$academic-paper` 审阅 P1-P6 路线图：工程可行性中高，论文主张有条件成立；P5 主表应至少 3 seed 并报告 mean/std，P3/P4 必须记录参数量、训练预算和 seed；审阅记录见 `docs/ai/context/20260603-182845-forecasting-roadmap-feasibility-review.md`。
- 已用 `using-superpowers` 整理 P1-P6 完整工程设计 contract，新增文件 `docs/ai/context/20260603-184214-forecasting-p1-p6-complete-design.md`；后续实现必须从 P1 开始，并按该文档的 CLI、文件职责、metrics、验收和阶段记录要求执行。
- 已梳理 ReGenNet 三套数据集用法：当前项目中 NTU120-AS/Chi3D-AS 可走原论文 `Feeder -> ccollate -> CMDM -> ST-GCN eval` 路径；InterHuman-AS 是后补 SMPL H5 reproduction，可训练但缺论文 Table 4 evaluator/checkpoint；记录见 `docs/ai/context/20260603-185018-three-datasets-usage-review.md`。
- InterHuman 第一阶段完成后，NTU120-AS 建议作为 action-conditioned forecasting / SMPL-X 规整大样本扩展，Chi3D-AS 建议作为高质量小样本 SMPL-X 泛化或 qualitative 补充；不要把三套数据集直接合并作为第一阶段训练集，记录见 `docs/ai/context/20260603-185530-ntu-chi3d-after-interhuman-plan.md`。
- 最终正式设计文档已锁定为 `docs/ai/context/20260603-190003-forecasting-final-official-design.md`；后续实现、验收、阶段记录和论文结果解释以该文档为准。
- 下一阶段从 P1 开始：新增 InterHuman forecasting dataset、active vector extract/restore、150 帧窗口裁剪、normalizer 和 shape/finite smoke。
- P1/P2 完成前不要先做 relation-aware model；必须先让 repeat baseline evaluator 和 metrics 闭环。
- 已新建 P1 计划文档：`docs/ai/context/20260603-190529-forecasting-p1-plan.md`。
- P1 实现允许新增 `eval/eval_forecasting.py --mode dataset_smoke` 作为验收入口，但只做 dataset/normalizer smoke；P2 再扩展 metrics、repeat 和 checkpoint evaluation。
- Forecasting P1 已完成，结果记录见 `docs/ai/context/20260603-191712-forecasting-p1-dataset-result.md`。
- P1 新增实现：`utils/forecasting_motion.py`、`data_loaders/forecasting/interhuman.py`、`data_loaders/forecasting/tensors.py`、`eval/eval_forecasting.py`。
- P1 smoke 已通过：train/val/test 可用样本为 `2910/226/508`，batch shape 为 `obs=[4,30,2,147]`、`target=[4,120,2,147]`，active roundtrip 误差 `0.0`，normalizer roundtrip 误差 `1.1920928955078125e-07`。
- P1 normalizer 已生成：`save/forecasting/interhuman/p1_dataset_smoke/normalizer.pt` 和 `normalizer.json`；统计使用 train-only `T>=150` 序列共 `2910` 条、`1481944` 帧。
- 下一步进入 P2：实现 original-scale metrics 和 repeat baseline evaluator；不得重新定义 P1 数据协议。
- 已新建 P2 计划文档：`docs/ai/context/20260603-194249-forecasting-p2-plan.md`；P2 只实现 original-scale metrics、metrics sanity 和 repeat baseline，不进入 P3 模型训练。
- Forecasting P2 已完成，结果记录见 `docs/ai/context/20260603-194749-forecasting-p2-metrics-repeat-result.md`。
- P2 新增实现：`utils/forecasting_metrics.py`；扩展 `eval/eval_forecasting.py --mode metrics_sanity|repeat|checkpoint`。
- P2 metrics sanity 已通过：test split `508` 条样本，`pred == target` 时所有固定指标均为 `0.0`。
- P2 repeat baseline 已完成 test split 评估，结果保存到 `save/forecasting/interhuman/repeat_150_30_120/metrics_test.json` 和 `metrics_test.yaml`；`future_mse=0.036892867478446695`，`long_mse=0.05112874942032371`，`relative_root_distance_error=0.255221389058068`，`relative_orientation_error=0.5552304635836384`，`inter_person_distance_consistency=0.006041959892430409`。
- 下一步进入 P3：实现 independent predictor 和 concat no-relation predictor；必须复用 P2 evaluator，不得另写指标口径。

## 2026-06-03 Skill 安装

- 已安装 `Imbad0202/academic-research-skills` 仓库中的 4 个本地 Codex skills：`academic-pipeline`、`deep-research`、`academic-paper`、`academic-paper-reviewer`。
- 安装位置为 `/home/rpartx3080/.codex/skills/`；后续会话需要重启 Codex 才能加载。
- 安装记录见 `docs/ai/context/20260603-161246-academic-research-skills-install.md`。
