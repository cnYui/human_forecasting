# AGENTS 入口内容归档

来源：`AGENTS.md`

归档时间：2026-06-21 17:28:02 Asia/Tokyo

## 原文

```markdown
# ReGenNet AI 入口

- 默认使用中文；文档、说明、计划、总结和代码注释都保持中文，除非用户明确要求英文。
- 每次进入实现前先做 design / plan，并把新增上下文、决策、取舍和结果记录到 `docs/ai/context/`；新文件使用 `YYYYMMDD-HHMMSS-文件名.md`，不要覆写历史上下文。
- 长期项目记忆已迁移到 `docs/ai/context/20260614-185753-regennet-key-context-extracted-from-agents.md`；更细的实验和设计以该目录下各阶段时间戳文档为准。
- 当前主线是 InterHuman 150/30/120 two-person motion forecasting，以及 SoMoFormer / T2P / JRT 等结构化预测 baseline 在 ReGenNet 内的同口径迁入。
- 论文结论必须保守：不能声称 multi-person forecasting、interaction-aware 或 explicit relation 首创；relation-aware 只可表述为相对 concat / parameter-matched concat 有稳定收益，不能写成全面优于 independent。
- 近期重点：P8 official SoMoFormer 已完成 3-seed，后续若继续扩展，优先在 ReGenNet 内做 `t2p_interhuman_xyz` 或 `jrt_xyz`，不要直接改外部官方仓库作为主实现。
- 2026-06-15 已实现 `jrt_xyz` baseline：新增 `model/forecasting_jrt.py`，接入 `model/forecasting_xyz.py` 和 `train/train_forecasting_xyz.py`，复用 InterHuman H5 -> active_to_xyz -> xyz evaluator；2-step smoke、checkpoint eval、metrics sanity 均通过，记录见 `docs/ai/context/20260615-132550-jrt-xyz-implementation-result.md`。
- 2026-06-15 已启动 `jrt_xyz` seed0 5000-step 正式训练，目录为 `save/forecasting/interhuman/p9_jrt_xyz_h256_l4_s0_5000`，配置为 `hidden_dim=256,num_layers=4,num_heads=8,batch_size=8,grad_accum_steps=2,effective_batch_size=16,jrt_relation_weight=0.5`；启动记录见 `docs/ai/context/20260615-140005-jrt-xyz-seed0-training-start.md`。训练进度看 `train_log.jsonl`，不要只看因 stdout 缓冲可能为空的 `train.log`。
- 2026-06-15 已实现 `t2p_interhuman_xyz` baseline：新增 `model/forecasting_t2p.py`，接入 `model/forecasting_xyz.py` 和 `train/train_forecasting_xyz.py`，保留 T2P 的 trajectory-first、global-local decoupling 和 trajectory-conditioned local pose decoder；dataset smoke、metrics sanity、2-step train/checkpoint eval 均通过。
- 2026-06-15 已启动 `t2p_interhuman_xyz` seed0 5000-step 正式训练，目录为 `save/forecasting/interhuman/p9_t2p_interhuman_xyz_h256_l2_b8a4_s0_5000`，配置为 `hidden_dim=256,num_layers=2,num_heads=8,batch_size=8,grad_accum_steps=4,effective_batch_size=32,lr=3e-4,seed=0`；启动记录见 `docs/ai/context/20260615-140718-p9-t2p-interhuman-xyz-start-result.md`。第一次 `batch_size=32` 因已有 JRT 训练占用显存而 OOM，已降 batch 并保留 effective batch。
```
