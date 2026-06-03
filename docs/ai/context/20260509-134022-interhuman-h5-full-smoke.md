# InterHuman-AS H5 Full Smoke 记录

## 目的

验证完整冻结 H5 数据路径在 `num_steps=1000` 时确实执行 1000 个训练 step，并能正常保存 checkpoint、自然退出。

## 代码修正

修正 `train/training_loop.py`：

- `num_epochs` 改为按剩余 step 和 dataloader 长度向上取整。
- 每个 batch 前以 `num_steps` 作为硬停止条件。
- `num_steps < len(data)` 时不再出现 `num_epochs=0`。
- 训练提前停止时尝试关闭 dataloader iterator。

修正 `data_loaders/get_data.py`：

- InterHuman 固定 `num_workers=0`。
- 原因是完整 H5 loader 在提前达到 `num_steps` 后，PyTorch 1.7 + 多 worker 会留下 worker 子进程并阻塞进程退出。

修正 `train/train_mdm.py`：

- 修复当前 dirty worktree 中的 `da ta` 拼写错误，恢复为 `data`。

## 最终 smoke 命令

```bash
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet \
python -m train.train_mdm \
  --setting cmdm \
  --save_dir save/interhuman/h5_full_1000_smoke_noworkers \
  --dataset interhuman \
  --data_path dataset/interhuman/smpl/conditioned \
  --cond_mask_prob 0 \
  --num_person 2 \
  --layers 2 \
  --latent_dim 128 \
  --num_frames 150 \
  --arch online \
  --overwrite \
  --pose_rep rot6d \
  --body_model smpl \
  --train_platform_type NoPlatform \
  --unconstrained \
  --batch_size 1 \
  --num_steps 1000 \
  --save_interval 500 \
  --log_interval 100 \
  --lambda_orient 0 \
  --lambda_body 0 \
  --lambda_transl 0 \
  --max_samples -1
```

## 结果

训练正常达到 1000 step 并自然退出。

loss 记录：

```text
step[0]:   loss[0.65867]
step[100]: loss[0.15427]
step[200]: loss[0.11790]
step[300]: loss[0.10618]
step[400]: loss[0.09846]
step[500]: loss[0.09387]
step[600]: loss[0.09114]
step[700]: loss[0.08892]
step[800]: loss[0.08640]
step[900]: loss[0.08422]
```

输出文件：

```text
save/interhuman/h5_full_1000_smoke_noworkers/args.json
save/interhuman/h5_full_1000_smoke_noworkers/model000000000.pt
save/interhuman/h5_full_1000_smoke_noworkers/model000000500.pt
save/interhuman/h5_full_1000_smoke_noworkers/model000001000.pt
save/interhuman/h5_full_1000_smoke_noworkers/opt000000000.pt
save/interhuman/h5_full_1000_smoke_noworkers/opt000000500.pt
save/interhuman/h5_full_1000_smoke_noworkers/opt000001000.pt
```

checkpoint 加载验证：

```text
model000000000.pt: OrderedDict, 50 keys
model000000500.pt: OrderedDict, 50 keys
model000001000.pt: OrderedDict, 50 keys
opt000001000.pt: dict, 2 keys
```

进程状态：

```text
无残留 train_mdm / micromamba run 进程
无残留 GPU compute app
```

## 注意

早先使用 `num_workers=8` 的 1000 step 训练也写出了 `model000001000.pt`，但进程退出时挂住，需要手动终止。最终有效验收以 `h5_full_1000_smoke_noworkers` 为准。

## 下一步

进入 P3：

```text
实现 --grad_accum_steps
```

在实现前保留当前 1000 step smoke checkpoint，作为 H5 数据链路稳定性的基线。
