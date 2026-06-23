# JRT 源码、数据与复现结果

## 结论

已完成用户要求的三件事：

1. 深度调研论文 `Joint-Relation Transformer for Multi-Person Motion Prediction`。
2. 官方源码已克隆到 `/home/rpartx3080/CodeSpace/JRTransformer`。
3. 官方 3DPW 预处理数据与 3DPW checkpoint 已下载，并已跑通官方测试入口，结果与论文 3DPW-SoMoF/RC 主表对齐。

## 来源

- 论文 PDF：`docs/download/2023-joint-relation-transformer-multi-person-motion-prediction.pdf`
- CVF 页面：`https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Joint-Relation_Transformer_for_Multi-Person_Motion_Prediction_ICCV_2023_paper.html`
- 官方源码：`https://github.com/MediaBrain-SJTU/JRTransformer`
- AMASS 官网：`https://amass.is.tue.mpg.de/index.html`

## 源码状态

源码路径：

```text
/home/rpartx3080/CodeSpace/JRTransformer
```

Git 信息：

```text
remote: https://github.com/MediaBrain-SJTU/JRTransformer.git
HEAD: 3765b17cb8b7ba1adfdec42839732fe93a0cebb3
commit message: fix writing mistake
```

主要文件：

```text
README.md
data/smpl_skeleton.npz
data/test_in.json
dataset/dataset_3dpw.py
dataset/dataset_amass.py
model/model.py
pretrain_amass.py
test_3dpw.py
train_3dpw.py
utils/config_3dpw.py
utils/metrics.py
utils/util.py
```

仓库边界：

- README 明确写着 repo 仍在维护中。
- 当前仓库只提供 3DPW-SoMoF/RC 相关训练、测试、AMASS pretrain 代码。
- 论文中的 CMU-Mocap / MuPoTS-3D 实验代码没有在当前仓库中释放。
- README 文件名写 `poseDate.pkl`，但代码实际读取 `data/poseData.pkl`；本地已按代码保存为 `poseData.pkl`。

## 数据与权重

已下载或已有文件：

```text
/home/rpartx3080/CodeSpace/JRTransformer/data/poseData.pkl
/home/rpartx3080/CodeSpace/JRTransformer/data/test_in.json
/home/rpartx3080/CodeSpace/JRTransformer/data/smpl_skeleton.npz
/home/rpartx3080/CodeSpace/JRTransformer/output/best_3dpw.pt
```

文件检查：

```text
poseData.pkl      123811209 bytes  sha256 b6d76d484289a87d1f51785becd09779c9d633a810822a508070b4a17886f6f3
best_3dpw.pt       14416025 bytes  sha256 8ddcfbea64d3aada0e7899ae6eb1822e7e91ed14f82ad9c83ba9d3e12c483049
test_in.json        4131855 bytes  sha256 36162a7d5a2f686c8c6799bf22112cc2d0b197174432c4948946ea4f3495ad89
```

`poseData.pkl` 结构：

```text
keys: ['test', 'train', 'valid']
train: 6479 samples, shape (6479, 2, 30, 39), float64, finite=True
valid:   36 samples, shape (36, 2, 30, 39), float64, finite=True
test:    85 samples, shape (85, 2, 30, 39), float64, finite=True
```

JRT loader smoke：

```text
train_aug len 38874, x_shape (30, 2, 13, 6), para_shape (3,), finite True
valid_pair_aug len 72, x_shape (30, 2, 13, 6), para_shape (3,), finite True
test_json_pair_aug len 170, x_shape (30, 2, 13, 6), para_shape (3,), finite True
```

下载细节：

- `output/best_3dpw.pt` 使用 README 的 Google Drive ID `1W354xCv-q9C2cIADm4Obt8P1RkaUKKmQ` 下载。
- `data/poseData.pkl` 使用 README 的 Google Drive ID `1tatpBjQ1rUyJ6NT5vsjmGOROqa9dw4l8` 下载；`gdown` 没处理大文件确认页，最终用 `curl` 加 `confirm=t` 下载。
- `data/test_in.json` 和 `data/smpl_skeleton.npz` 已包含在官方 Git 仓库中。
- AMASS 需要从官网登录/协议下载，不能绕过；本次没有下载 AMASS 原始数据。

## 论文深度调研

任务定义：

```text
输入：N 个人、J 个关节的历史 3D joint positions
输出：这些人的未来 3D joint positions
```

论文形式化表示：

```text
X_NJ ∈ R^{NJ × (Th × 3)}
Y_NJ ∈ R^{NJ × (Tf × 3)}
```

JRT 的关键点：

1. 每个 token 是“某个人的某个关节”的历史轨迹。
2. joint stream 编码历史关节位置和速度。
3. relation stream 显式编码 joint-to-joint relation。
4. relation-aware attention 把普通 joint attention score 与 relation feature 生成的 relation score 相加。
5. relation stream 还会预测未来 inter-joint distance，用 relation supervision 约束。

显式 relation 输入：

```text
relative distance D_X: 每两个关节之间的历史距离，代码里用 exp(-distance)
adjacent matrix A: 骨骼直接连接关系
connectivity matrix C: 同一人体骨架连通关系
```

训练目标：

```text
joint prediction loss
joint reconstruction loss
relation prediction loss / auxiliary relation loss
deep supervision over intermediate layers
```

主要协议：

```text
3DPW-SoMoF / 3DPW-SoMoF/RC:
  input 16 frames / 1030ms
  predict 14 frames / 900ms
  N=2, J=13
  metric: VIM, MPJPE

CMU-Mocap:
  input 15 frames / 1000ms
  predict 45 frames / 3000ms
  metric: MPJPE

MuPoTS-3D:
  使用与 CMU-Mocap 相同 segment length
  metric: MPJPE
```

实现细节：

```text
depth L = 4
num_heads = 8
feature dimension = 128
3DPW fine-tune lr = 1e-4
AMASS pretrain lr = 1e-3
batch_size = 128
optimizer = AdamW
λJ = λR = 10（论文表述）
```

对当前 ReGenNet 论文方向的影响：

- JRT 已经明确提出 explicit relation-aware multi-person motion prediction。
- 当前项目不能声称“首次做显式关系建模”或“首次把 relation-aware attention 用于多人动作预测”。
- 当前项目可以保留的差异是：InterHuman SMPL active-vector 协议、150/30/120 deterministic two-person forecasting、轻量 root-level relation cues、同口径 repeat/independent/concat/relation/parameter-matched concat 实证。
- 若后续迁入 JRT 思想，推荐作为 `jrt_xyz` 或 `jrt_active` 强 baseline，而不是把当前 root-level relation model 包装成 JRT 等价方法。

## 复现验证

当前使用环境：

```text
micromamba env: /home/rpartx3080/.local/micromamba/envs/regennet
python: 3.7.13
torch: 1.7.1
cuda: True
```

官方 README 推荐环境是 Python 3.10、Torch 1.12.1。当前环境的 PyTorch 1.7.1 缺少 `Tensor.swapaxes`，所以运行测试时只做了运行时兼容补丁：

```python
if not hasattr(torch.Tensor, 'swapaxes'):
    torch.Tensor.swapaxes = torch.Tensor.transpose
```

未修改官方源码。

运行命令：

```bash
cd /home/rpartx3080/CodeSpace/JRTransformer
micromamba run -p /home/rpartx3080/.local/micromamba/envs/regennet python - <<'PY'
import sys, runpy, torch
if not hasattr(torch.Tensor, 'swapaxes'):
    torch.Tensor.swapaxes = torch.Tensor.transpose
sys.argv = ['test_3dpw.py', '--batch_size', '64', '--device', '0']
runpy.run_path('test_3dpw.py', run_name='__main__')
PY
```

输出：

```text
Using device: cuda
Test MPJPE: avg: 9.53 | 100ms: 2.18 | 240ms: 5.05 | 500ms: 10.50 | 640ms: 12.98 | 900ms: 16.93
Test VIM:   avg: 39.51 | 100ms: 9.47 | 240ms: 21.71 | 500ms: 44.09 | 640ms: 53.69 | 900ms: 68.57
```

论文 Table 1 的 3DPW-SoMoF/RC JRT 数字为 `AVG 39.5`、`900ms 68.8`；本地复现 `AVG 39.51`、`900ms 68.57`，视为对齐。

## 未完成和阻塞

1. AMASS pretraining 未复现：AMASS 原始数据需要官网账户和许可下载。
2. CMU-Mocap / MuPoTS-3D 未复现：当前官方 JRT 仓库未释放对应训练/测试代码和预处理数据。
3. 当前只验证了官方 checkpoint 在 3DPW-SoMoF/RC test set 上的 evaluation；完整从 AMASS pretrain 到 3DPW fine-tune 需要 AMASS 数据和更长训练预算。

## 后续建议

1. 如果目标是复现 JRT 论文结果，优先在 Python 3.10 + Torch 1.12.1 新环境中跑 `test_3dpw.py` 和 `train_3dpw.py`，避免 `swapaxes` 兼容补丁。
2. 如果目标是迁入 ReGenNet，先做 joint-space `jrt_xyz` baseline：输入 `[B, obs, 2, J, 3]`，relation tensor 使用 distance + same-person + skeleton adjacency，输出 `[B, pred, 2, J, 3]`，复用 P7/P8 xyz evaluator。
3. 不建议第一步做 active-vector 全 JRT，因为 active-vector 的 rot6d/transl 语义和 JRT 的 joint-to-joint XYZ relation 不一致，容易把论文结构迁坏。
