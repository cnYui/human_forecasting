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
- 已新建 P3 计划文档：`docs/ai/context/20260603-195707-forecasting-p3-baselines-plan.md`；P3 只实现 independent / concat 两个可训练 baseline、supervised training loop、checkpoint evaluation 和结果落盘，不进入 relation-aware model。
- P3 默认训练预算为 `hidden_dim=256, num_layers=2, batch_size=32, num_steps=5000, lr=1e-3, weight_decay=1e-4, seed=0, num_workers=0`；如 5000 steps 未超过 repeat baseline，可追加到 10000 steps，但必须在 P3 结果文档记录原因。
- P3 验收必须记录参数量、训练预算和 seed；independent 与 concat 都至少需要在 test `future_mse` 上优于 repeat baseline `0.036892867478446695`，否则不得进入 P4。
- Forecasting P3 已完成，结果记录见 `docs/ai/context/20260603-201148-forecasting-p3-baselines-result.md`。
- P3 新增实现：`model/forecasting.py`、`train/train_forecasting.py`；扩展 `eval/eval_forecasting.py --mode checkpoint`。
- P3 concat baseline 完整训练已通过：`save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt`，`num_params=9951440`，test `future_mse=0.031901971752366684`，`long_mse=0.03789569738167008`。
- P3 independent baseline 完整训练已通过：`save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt`，`num_params=5305064`，test `future_mse=0.02874350040329723`，`long_mse=0.03612076791780671`。
- P3 重要观察：当前 seed=0 下 independent 强于 concat，且两者 relation metrics 都未优于 repeat；P4/P5 必须同时报告 independent 与 concat，不得只用“赢 concat”包装 relation-aware 结论。
- 下一步进入 P4：实现 relation-aware joint predictor；必须复用 P2 evaluator 和 P3 training/checkpoint contract。
- 已新建 P4 计划文档：`docs/ai/context/20260603-202101-forecasting-p4-relation-plan.md`；P4 只实现 relation-aware joint predictor、relation feature extractor、训练/eval 兼容和单 seed 验收，不进入 P5 multi-seed 或 ablation。
- P4 第一版 relation features 固定为 relative root translation、relative root velocity、root distance、relative root orientation，默认仍使用 normalized active-vector MSE，不默认加入 relation loss。
- P4 最低指标门槛：`future_mse <= concat future_mse`、`long_mse < concat long_mse`，并且至少一个 relation metric 优于 concat；强门槛是同时优于 independent。
- Forecasting P4 已完成，结果记录见 `docs/ai/context/20260603-202924-forecasting-p4-relation-result.md`。
- P4 新增实现：`utils.forecasting_motion.extract_relation_features`、`model.forecasting.RelationAwareForecastingModel`；扩展 `train/train_forecasting.py` 和 `eval/eval_forecasting.py` 支持 `--model_type relation`。
- P4 relation official 训练已通过：`save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt`，`num_params=10058704`，test `future_mse=0.031443351850382925`，`long_mse=0.036962207905420166`，`relative_root_distance_error=0.40891610895554853`。
- P4 达到最低门槛：优于 concat 的 `future_mse`、`long_mse` 和 `relative_root_distance_error`；但未达到强门槛，未优于 independent，也未优于 repeat 的 relation metrics。
- 下一步进入 P5：做 multi-seed、relation feature ablation、容量控制和 possible original-scale relation feature 对照；不得声称 relation-aware 全面优于 independent。
- 已新建 P5 计划文档：`docs/ai/context/20260603-203757-forecasting-p5-plan.md`。
- P5 必须先完成 20% 主协议的 3-seed 主表：Repeat / Independent / Concat no-relation / Relation-aware，seeds=`0,1,2`；可复用 P3/P4 的 seed=0 run，但 manifest 必须记录真实路径。
- P5 论文主张门槛：relation-aware 的 `long_mse` mean 优于 concat，且至少 2/3 seeds 优于同 seed concat；至少一个 relation metric 的 mean 和 2/3 seeds 同时优于 concat。主表不过，不进入完整 ablation 和 P6 成功展示。
- P5 parameter-matched concat 使用 `hidden_dim=259,num_layers=2`，参数量约 `10075415`，与 P4 relation `10058704` 相差约 `0.17%`。
- P5 observation-ratio 补充不能直接开跑；当前 metrics 固定 `pred_len=120`，必须先扩展动态 pred_len 并保证 `pred_len=120` 结果不漂移。
- Forecasting P5.1 基础设施已完成，结果记录见 `docs/ai/context/20260603-205954-forecasting-p5-infrastructure-result.md`。
- P5.1 新增 relation 消融开关：`--relation_feature_set all|translation|velocity|orientation` 和 `--relation_encoder_type gru|none`；旧 P4 checkpoint config 缺少这些字段时默认按 `all+gru` 恢复。
- P5.1 新增 `eval/eval_forecasting.py --mode aggregate --manifest PATH`，可读取已落盘 metrics/args/checkpoint metadata 并输出 `summary.json/csv/md` 和 `manifest.resolved.json`。
- P5.1 smoke 已通过：`results/forecasting/interhuman/p5_aggregate_smoke/summary.json` 已汇总 repeat/independent/concat/relation seed0；`save/forecasting/interhuman/p5_ablation_knobs_smoke/model000000002.pt` 可独立加载评估。
- 下一步进入 P5.2：先跑 20% 主协议 3-seed 主表；不得直接启动完整消融矩阵或 P6 成功展示。
- Forecasting P5.2 主表 3-seed 已完成，结果记录见 `docs/ai/context/20260603-232353-forecasting-p5-main-table-result.md`。
- P5.2 manifest 和汇总输出位于 `results/forecasting/interhuman/p5_main_150_30_120/`：`manifest.json`、`manifest.resolved.json`、`summary.json`、`summary.csv`、`summary.md`。
- P5.2 主表 mean：repeat `future_mse=0.0368928675,long_mse=0.0511287494`；independent `future_mse=0.0287863306,long_mse=0.0362148034`；concat `future_mse=0.0320573101,long_mse=0.0380507699`；relation `future_mse=0.0317788706,long_mse=0.0373418675`。
- P5.2 gate 通过：relation 相对 concat 的 `long_mse` mean 更低且 same-seed 3/3 胜出；`relative_root_distance_error` mean 更低且 same-seed 3/3 胜出。允许进入 P5.3 消融表。
- P5.2 边界必须保留：relation 不优于 independent 的 `future_mse/long_mse`，也不优于 repeat 的 relation-style metrics；论文结论只能写为 relation-aware 稳定优于 concat no-relation，不能写成全面最优。
- 下一步进入 P5.3：训练 relation feature / encoder ablation 和 parameter-matched concat；不得直接进入 P6 success showcase。
- Forecasting P5.3 消融表 3-seed 已完成，结果记录见 `docs/ai/context/20260604-085116-forecasting-p5-ablation-result.md`。
- P5.3 manifest 和汇总输出位于 `results/forecasting/interhuman/p5_ablation_150_30_120/`：`manifest.json`、`manifest.resolved.json`、`summary.json`、`summary.csv`、`summary.md`。
- P5.3 新增 15 个训练 run：parameter-matched concat `h259_l2`、relation no-encoder all-features、translation-only GRU、velocity-only GRU、orientation-only GRU，各 seeds=`0,1,2`；full relation 和 concat no-relation 复用 P5.2/P3/P4 run。
- P5.3 parameter-matched concat gate 通过：all-features relation `long_mse=0.0373418675` 优于 h259 concat `0.0378088307`，`relative_root_distance_error=0.4220982101` 优于 `0.4913293425`，same-seed 均为 3/3。
- P5.3 relation encoder gate 通过但幅度小：with-encoder `long_mse=0.0373418675` 优于 no-encoder `0.0375836698`，same-seed 2/3；`relative_root_distance_error=0.4220982101` 优于 `0.4320849805`，same-seed 2/3。with-encoder 在 `relative_orientation_error` 和 `inter_person_distance_consistency` 上不优于 no-encoder。
- P5.3 relation feature gate 通过：all-features relation 的 `long_mse` mean 优于 translation/velocity/orientation 单特征；`relative_root_distance_error` mean 也优于三个单特征。velocity-only 的 `long_mse=0.0373630645` 与 all-features `0.0373418675` 很接近，且 same-seed long_mse 仅 all-features 1/3 胜出，必须在论文里保留该边界。
- 允许进入 P6 qualitative / paper figure 准备；但论文主张仍只能写 relation-aware 相对 concat no-relation 与 parameter-matched concat 的稳定收益，不能写全面优于 independent 或所有 relation metrics 全面改善。
- P5.4 observation-ratio 仍未启动；若要做，必须先扩展动态 `pred_len` metrics contract，P5.4 不阻塞当前 20% 主协议进入 P6。
- 已按用户要求使用 `using-superpowers` 基于最终正式设计文档新建 P6 设计文档：`docs/ai/context/20260604-085953-forecasting-p6-qualitative-design.md`。
- P6 范围锁定为 qualitative / paper figure 准备：只做 sample-level metrics、npy、distance/orientation/long_mse curves 和可选 render；不训练新模型，不改 P2 metrics key，不启动 P5.4。
- P6 主 qualitative 使用 seed0 representative checkpoints：independent=`save/forecasting/interhuman/p3_independent_h256_l2_s0_5000/model000005000.pt`，concat=`save/forecasting/interhuman/p3_concat_h256_l2_s0_5000/model000005000.pt`，relation=`save/forecasting/interhuman/p4_relation_h256_r128_l2_s0_5000/model000005000.pt`；3-seed 结论仍以 P5 表格为准。
- P6 样本选择必须覆盖 success / close / failure / boundary，保存 `selection.json/csv`；不得只挑成功样本。
- 下一步按 P6 设计实现 `sample/visualize_forecasting.py`，优先完成 npy、per-sample metrics 和曲线输出，render 不阻塞验收。
- 已新建 P6 计划文档：`docs/ai/context/20260604-090635-forecasting-p6-plan.md`。
- P6 实现计划分为：P6.1 sample-level metrics/curves helper，P6.2 seed0 checkpoints 全 test 推理，P6.3 自动样本选择，P6.4 qualitative 数据包和曲线落盘，P6.5 验收与结果记录。
- P6 允许小范围修改 `utils/forecasting_metrics.py` 新增 helper，但不得改变 `compute_forecasting_metrics` 的 key 和行为；`eval/eval_forecasting.py` 暂不扩展，避免 evaluator 职责继续膨胀。
- Forecasting P6 qualitative 已完成，结果记录见 `docs/ai/context/20260604-091820-forecasting-p6-qualitative-result.md`。
- P6 新增实现：`sample/visualize_forecasting.py`；扩展 `utils/forecasting_metrics.py` 增加 sample-level / per-frame helper，未改变 `compute_forecasting_metrics` 的 key 和行为。
- P6 主输出位于 `results/forecasting/interhuman/p6_qualitative_150_30_120/`，包含 `run_config.json`、`metrics_per_sample_all.{json,csv}`、`selection.{json,csv}`、`summary.md` 和 8 个 qualitative sample 目录。
- P6 selected samples 覆盖 success / close / failure / boundary 各 2 个：success=`4860,4618`，close=`2508,2627`，failure=`2194,625`，boundary=`4095,2613`。
- P6 验收已通过：compileall、P6 主命令、selection 类别覆盖、8 个样本文件完整、所有 obs/gt/pred npy finite、曲线图生成、P2 metrics_sanity 回归全 0。
- P6 结论边界：qualitative 支持 relation-aware 在部分样本中改善 long-horizon error 和 relative root distance，但失败/边界样本证明不能声称全面优于 concat 或 independent；P6 只能解释 P5 主表，不能替代 P5 全 test aggregate。
- 2026-06-04 状态审计已完成，记录见 `docs/ai/context/20260604-093950-forecasting-goal-status-audit.md`。结论：当前正式设计定义的 InterHuman 150/30/120 第一阶段工程证据链 P1-P6 已完成，论文主张可以成立但必须收窄为 relation-aware 相对 concat no-relation 和 parameter-matched concat 的稳定收益；完整论文正文、投稿材料、P5.4 observation-ratio、P7/P8 数据集扩展和动作视频渲染尚未完成，且不阻塞当前主协议完成状态。
- 2026-06-08 已做 forecasting 相关工作快速联网检索，记录见 `docs/ai/context/20260608-113335-forecasting-related-work-scan.md`。结论：multi-person / interaction-aware human motion forecasting 已有强相关工作，尤其 Joint-Relation Transformer、Trajectory2Pose、Stochastic Multi-Person 3D Motion Forecasting、SoMoFormer；当前论文不能声称首次做多人预测或首次显式关系建模，只能把贡献收窄为 InterHuman SMPL 150/30/120 deterministic two-person forecasting protocol 下的严谨同口径实证和 relation-aware 相对 concat/parameter-matched concat 的稳定收益。
- 2026-06-08 已按用户要求下载 5 篇强相关 forecasting 论文到 `docs/download/`，并新增下载记录 `docs/ai/context/20260608-113937-related-papers-download-result.md`；PDF 均已用 `file` 检查为有效 PDF。后续写 related work 时应基于这些 PDF 做逐篇摘要和差异表。
- 2026-06-09 已完成 5 篇 forecasting 论文深度调查和当前创新点定位，记录见 `docs/ai/context/20260609-094222-forecasting-five-paper-deep-review-innovation.md`。结论：当前创新点不在 multi-person forecasting、interaction-aware、explicit relation 或 long-horizon forecasting 的方向首创，而在 InterHuman SMPL 150/30/120 deterministic two-person protocol、SMPL active-vector 数据/评估闭环、严格同口径 repeat/independent/concat/relation-aware/parameter-matched concat empirical study、轻量 root-level relation cues 对 concat 的稳定收益，以及诚实报告不优于 independent 和 failure/boundary cases。后续必须补查 Guo et al. 2022 / XIA 类 two-person forecasting 工作，并建议补 joint-space MPJPE-like metric。
- 2026-06-09 已分析 SoMoFormer 架构并对照当前 `model/forecasting.py`，记录见 `docs/ai/context/20260609-095111-somoformer-architecture-analysis-for-regennet.md`。核心结论：当前 concat/relation 模型用单个 hidden bottleneck 直接生成双人全未来，缺少 joint/person token 结构和 independent fallback，容易输给 independent；下一版优先建议做 independent + interaction residual gated model，再做 SoMoFormer-lite active-token Transformer，最后再考虑 DCT token Transformer。
- 2026-06-09 已深度调研 CVPR 2024 Trajectory2Pose / T2P，记录见 `docs/ai/context/20260609-111650-trajectory-conditioning-paper-deep-review.md`。核心结论：T2P 的价值在于先预测多模态 global hip trajectory，再用 trajectory conditioning 生成 local pose；这支持下一版优先做 `independent base + trajectory-conditioned interaction residual + gate`，而不是继续用 single hidden 直接解码双人全未来。
- T2P 不能被用来支持“interaction-aware / multi-person forecasting 首创”主张；它本身证明该方向已有强相关工作。当前项目和 T2P 的关键差异必须保留：T2P 是 multi-agent / in-the-wild pseudo 3D joint / best-of-K 多模态评估；当前主协议是 InterHuman SMPL active-vector / deterministic two-person / same-protocol baselines。
- 2026-06-09 已对比 SoMoFormer 与 T2P 的关节建模关系，记录见 `docs/ai/context/20260609-112253-somoformer-vs-t2p-joint-attention-relation.md`。结论：两者都不是纯时间序列 flatten；SoMoFormer 是 joint-coordinate trajectory token 直接做 joints/persons attention，T2P 是 local body-part/pose embedding + agent-wise trajectory graph + trajectory-conditioned local pose。两者应写成互补的 structured pose forecasting 路线，不要写成直接继承关系。
- 2026-06-09 已澄清 SoMoFormer “不是时序输入”的准确含义，记录见 `docs/ai/context/20260609-112616-somoformer-time-vs-joint-input-clarification.md`。后续应表述为：SoMoFormer 的 token/query 是 joint-coordinate trajectory，不是 timestep；每个 token 内部仍包含 obs+future padding 的时间轨迹。不要把“没有时间信息”和“不是按时间帧做 token”混为一谈。
- 2026-06-09 已深度调研 MRT 2021、JRT 2023 和 DuMMF / Stochastic Multi-Person 3D Motion Forecasting，记录见 `docs/ai/context/20260609-114522-three-forecasting-papers-deep-review-result.md`。核心结论：MRT 是 person/time local-global Transformer，JRT 是显式 joint-to-joint relation + relation supervision，DuMMF 是 stochastic 多未来生成与 dual-level intents；三者和 SoMoFormer/T2P 属于同一大方向下的不同子问题，不能跨 dataset/metric/protocol 直接比分数。
- 后续 related work 推荐大白话定位：MRT = “看自己 + 看别人”；SoMoFormer = “关节轨迹 token 互相看”；JRT = “给关节 attention 加人体关系地图”；DuMMF = “预测多个合理未来”；T2P = “先预测人往哪走，再补身体怎么动”。当前项目只能写 InterHuman SMPL two-person deterministic protocol 下的同口径实证，不能写 multi-person / interaction-aware / explicit relation / stochastic forecasting 首创。

## 2026-06-03 Skill 安装

- 已安装 `Imbad0202/academic-research-skills` 仓库中的 4 个本地 Codex skills：`academic-pipeline`、`deep-research`、`academic-paper`、`academic-paper-reviewer`。
- 安装位置为 `/home/rpartx3080/.codex/skills/`；后续会话需要重启 Codex 才能加载。
- 安装记录见 `docs/ai/context/20260603-161246-academic-research-skills-install.md`。

## 2026-06-08 Codex 升级

- 当前 Codex standalone 已从 `codex-cli 0.134.0` 升级到 `codex-cli 0.137.0`。
- 升级路径使用 `codex update`；因本机 `/usr/bin/awk` 为 `mawk 1.3.4`，installer 的 `{64}` 正则校验解析失败，最终用临时 BusyBox `awk` 包装完成升级，未修改系统 `awk`。
- 验证已通过：`codex --version` 为 `codex-cli 0.137.0`，`codex doctor --summary --ascii --no-color` 为 `17 ok | 1 idle | 0 warn | 0 fail`。
- 记录见 `docs/ai/context/20260608-092121-codex-upgrade-plan.md` 和 `docs/ai/context/20260608-092426-codex-upgrade-result.md`；当前交互进程仍需重启 Codex 才能使用新版本能力。

## 2026-06-13 SoMoFormer 代码定位

- `docs/download/2022-somoformer-multi-person-pose-forecasting-transformers.pdf` 对应 arXiv `2208.14023`，标题为 `SoMoFormer: Multi-Person Pose Forecasting with Transformers`，作者为 Edward Vendrow, Satyajit Kumar, Ehsan Adeli, Hamid Rezatofighi。
- 正确开源仓库为 `https://github.com/evendrow/somoformer`；clone 命令为 `git clone https://github.com/evendrow/somoformer.git`。
- 仓库包含 `src/model.py`、`src/train.py`、`src/evaluate.py`、`src/dataset.py` 和 `src/configs/release.yaml`，但 README 标注仓库仍是 work-in-progress，论文使用的 AMASS 数据处理脚本可能未完整释放。
- 检索记录见 `docs/ai/context/20260613-113046-somoformer-code-search-result.md`。
- 已按用户要求克隆到 `/home/rpartx3080/CodeSpace/somoformer`，当前 HEAD 为 `e9f4fc0 Misc`；结果记录见 `docs/ai/context/20260613-120613-somoformer-clone-result.md`。
- 2026-06-13 已评估用 ReGenNet InterHuman 数据适配 SoMoFormer：可行，但不建议直接改 SoMoFormer 仓库；推荐先做 joint-space SoMoFormer baseline，再做 active-vector 同口径 `somoformer_active`。评估记录见 `docs/ai/context/20260613-121048-somoformer-regennet-feasibility.md`。
- 已为 SoMoFormer 原始组件、当前 ReGenNet independent / concat / relation-aware 双人预测组件，以及 SoMoFormer-style 适配路线绘制 Mermaid 图；记录见 `docs/ai/context/20260613-121955-somoformer-vs-current-forecasting-mermaid.md`。
- SoMoFormer 思想迁入 ReGenNet 的核心设计是不迁移 3DPW/SoMoF dataset，而是在 ReGenNet active-vector pipeline 内加 adapter：P7.1 用 `active -> SMPL xyz` 做 joint-space SoMoFormer baseline，P7.2 做输出 `[B,120,2,147]` 的 `somoformer_active` 以复用 P2/P5 同口径指标；设计说明见 `docs/ai/context/20260613-134606-somoformer-into-regennet-design.md`。
- P7.1 joint-space SoMoFormer baseline 计划已锁定：新增 `utils/forecasting_xyz.py`、`model/forecasting_somoformer.py`、`train/train_forecasting_xyz.py`、`eval/eval_forecasting_xyz.py`；不修改 P1-P6 active-vector 主路径；计划见 `docs/ai/context/20260613-200203-p7-joint-space-somoformer-plan.md`。
- P7.1 joint-space SoMoFormer baseline 已实现并 smoke 通过：实际环境为 `micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet`，新增 `utils/forecasting_xyz.py`、`model/forecasting_somoformer.py`、`train/train_forecasting_xyz.py`、`eval/eval_forecasting_xyz.py`，并移除 `model/rotation2xyz.py` 一行旧调试输出；xyz smoke、xyz metrics sanity、2-step training、checkpoint eval 和 active-vector metrics 回归均通过。结果记录见 `docs/ai/context/20260613-200956-p7-joint-space-somoformer-result.md`。
- 已在 `regennet` micromamba 环境安装 `torch-dct==0.1.5`，CPU/CUDA DCT->IDCT roundtrip 误差约 `1e-6`；用途和安装验证记录见 `docs/ai/context/20260613-201504-torch-dct-install-result.md`。当前 P7.1 实现仍使用纯 torch DCT 矩阵，后续可按需替换为 `torch_dct`。
- P7.1 SoMoFormer XYZ seed0 5000-step 正式训练已完成：`save/forecasting/interhuman/p7_somoformer_xyz_h256_l4_dct30_s0_5000/model000005000.pt`，`num_params=3182110`，test `joint_mse=0.0596641728`、`mpjpe=0.2897213449`、`long_joint_mse=0.1165986784`、`relative_root_distance_error=0.1982787535`；结果记录见 `docs/ai/context/20260613-210406-p7-somoformer-xyz-seed0-result.md`。这是 joint-space 口径，不能直接和 P5 active-vector 主表横比。
- 已将 P7.1 SoMoFormer XYZ 与 repeat 及旧 P3/P4 active checkpoints 转 xyz 后的结果做 test 同口径对比，输出在 `results/forecasting/interhuman/p7_xyz_compare_active_baselines/`；SoMoFormer XYZ 优于 repeat 和旧 active->xyz baselines，但公平性边界是旧 baselines 未用 xyz loss 训练。记录见 `docs/ai/context/20260613-211117-p7-xyz-comparison-with-active-baselines.md`。下一步若要严谨比较结构，需补 xyz-independent baseline。
- 用户修正 P7 xyz independent baseline 定义：应仍使用双人样本，但模型内部把两个人分开独立预测，最后拼回双人输出；正式命名为 `independent_pair_xyz`。设计文档见 `docs/ai/context/20260613-212043-p7-independent-pair-xyz-design.md`，推荐新增 `model/forecasting_xyz.py` 作为 joint-space 模型 factory，并扩展 `train_forecasting_xyz.py --model_type independent_pair_xyz|somoformer_xyz`。
- P7 `independent_pair_xyz` 已实现并完成 seed0 5000-step：`save/forecasting/interhuman/p7_independent_pair_xyz_h256_l2_s0_5000/model000005000.pt`，`num_params=2934464`，test `joint_mse=0.0698097058`、`mpjpe=0.3428108572`、`long_joint_mse=0.1305657924`。P7 xyz seed0 主对比输出在 `results/forecasting/interhuman/p7_xyz_main_seed0/`，SoMoFormer XYZ 相比 independent_pair_xyz 的 test `joint_mse` 约降低 14.53%，`mpjpe` 约降低 15.49%，`long_joint_mse` 约降低 10.70%。结果记录见 `docs/ai/context/20260613-215149-p7-independent-pair-xyz-result.md`。
- P7 xyz 主对比 3-seed 已完成并监控显存，结果记录见 `docs/ai/context/20260614-104849-p7-xyz-main-3seed-gpu-result.md`，汇总输出在 `results/forecasting/interhuman/p7_xyz_main_3seed/`。
- P7 xyz 3-seed mean：`independent_pair_xyz joint_mse=0.0682312447, mpjpe=0.3375963565, long_joint_mse=0.1284009837, relative_root_distance_error=0.2280735768`；`somoformer_xyz joint_mse=0.0607606915, mpjpe=0.2947844890, long_joint_mse=0.1186761773, relative_root_distance_error=0.2032881472`。
- P7 xyz 3-seed 结论：SoMoFormer XYZ 相对同样用双人样本训练的 `independent_pair_xyz`，mean `joint_mse` 降约 `10.95%`，`mpjpe` 降约 `12.68%`，`long_joint_mse` 降约 `7.57%`，`relative_root_distance_error` 降约 `10.87%`；该结论只适用于 joint-space P7 口径，不能直接横比 P5 active-vector 主表。
- P7 显存监控：`gpu_monitor.csv` 手动轮询共 `26` 次，采样峰值 `9660 MiB / 10240 MiB`，非空闲采样平均 `9310.72 MiB`，最高 utilization `96%`。`independent_pair_xyz batch_size=32` 基本稳定在 `9660 MiB`；`somoformer_xyz batch_size=16` 初始化约 `5510 MiB`，训练/评估采样峰值约 `9552 MiB`。
- 2026-06-14 用户确认最终目标是完整迁入官方 SoMoFormer 架构，而不是继续维护 lite 版；PR 路线和设计边界见 `docs/ai/context/20260614-134147-p8-official-somoformer-pr-roadmap.md`。
- P8 已新增 `official_somoformer_xyz`，保留 `somoformer_xyz` 作为 lite baseline；official 迁入 `AuxilliaryEncoder`、`LearnedDoublePositionalEncoding`、官方 DCT matrix 逻辑、`grid|neck|naive` location method、auxiliary loss、padding mask 接口和 train-time `metamask` 入口。InterHuman 第一版适配中 `tgt_neck` 用 SMPL root joint 第一帧代理，固定两人有效所以 padding mask 全 False。
- P8 official SoMoFormer XYZ 3-seed baseline 已跑完，结果记录见 `docs/ai/context/20260614-143029-p8-official-somoformer-xyz-3seed-result.md`，汇总输出在 `results/forecasting/interhuman/p8_official_somoformer_xyz_main/`。
- P8 3-seed mean：`official_somoformer_xyz joint_mse=0.0706294909, mpjpe=0.3033010067, long_joint_mse=0.1376462394, relative_root_distance_error=0.1935767838, inter_person_distance_consistency_xyz=0.0054475044`；相比 `somoformer_lite_xyz`，official 在 relation metrics 上更好，但 `joint_mse/mpjpe/long_joint_mse` 更差，不能声称完整官方架构全面优于 lite baseline。
- P8 official 显存采样峰值为 `9586 MiB / 10240 MiB`，非空闲采样平均 `9008.86 MiB`；配置为 `batch_size=16,num_layers=6,lr=1e-4,aux_weight=0.2,location_method=grid`。
