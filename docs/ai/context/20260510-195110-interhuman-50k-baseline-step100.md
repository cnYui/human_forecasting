# InterHuman-AS 50K baseline step[100] 检查

## 结论

50K baseline 已通过早期 `step[100]` 检查。

日志中已出现：

```text
step[0]: loss[0.67129]
step[100]: loss[0.10378]
Starting epoch 2:532
```

说明：

- 训练已跨过 epoch 0 和 epoch 1，进入 epoch 2。
- early loss 为有限值。
- `step[100]` loss 与上一轮 1000-step batch64 smoke 中捕获到的 `0.10378` 一致。
- 后台训练仍应继续运行到 `num_steps=50000`，下一关键 checkpoint 是 `model000005000.pt` / `opt000005000.pt`。

## 后续监控

下一次重点检查：

```bash
find save/interhuman/paper_config_l8_d512_accum64_50000_baseline -maxdepth 1 -type f -name 'model000005000.pt' -o -name 'opt000005000.pt'
tail -n 80 save/interhuman/paper_config_l8_d512_accum64_50000_baseline/train.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
```
