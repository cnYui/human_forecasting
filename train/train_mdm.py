# 本文件基于 https://github.com/openai/guided-diffusion 改写。
"""
训练扩散模型的入口脚本。

在 ReGenNet 中，这个脚本负责把“参数解析、数据加载、模型构建、扩散过程构建、训练循环”
串起来。真正的网络结构在 model/cmdm.py，真正的训练步在 train/training_loop.py。
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # 评估流程需要这些日志平台类
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from mpi4py import MPI

def main():
    # 1. 解析训练命令行参数，参数定义在 utils/parser_util.py 的 train_args()。
    args = train_args()

    # 2. 固定随机种子，保证实验尽量可复现。
    fixseed(args.seed)

    # 3. 根据参数选择训练日志平台，例如 TensorBoard、ClearML 或不记录。
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    # 4. 检查并创建保存目录，同时把本次训练参数保存为 args.json。
    #    后续采样、评估会从 checkpoint 同目录的 args.json 恢复模型配置。
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # 5. 初始化分布式训练环境。单 GPU 训练时也会走这套初始化逻辑。
    dist_util.setup_dist()

    print("creating data loader...")

    # 6. 根据架构名称判断是离线生成模式还是在线自回归模式。
    #    online/trans_dec 使用 Transformer Decoder；offline/trans_enc 使用 Transformer Encoder。
    if args.arch == 'trans_enc' or args.arch == 'mlp' or args.arch == 'gru' or args.arch == 'offline':
        arch_mode = 'offline'
    elif args.arch == 'trans_dec' or args.arch == 'online':
        arch_mode = 'online'

    # 7. unconstrained=True 表示不使用动作类别或文本作为条件；
    #    对 ReGenNet 的 cmdm 设置来说，仍然会使用另一人的动作 cmotion 作为条件。
    if args.unconstrained:
        action_conditioned = False
    else:
        action_conditioned = True
    print("Setting:", args.setting, "| Dataset:", args.dataset, "| Arch:", arch_mode, "| Action conditioned:", action_conditioned)

    # 8. 创建训练数据加载器。
    #    对 cmdm 双人动作数据，后续 collate 会把 actor 动作拆到 cond['y']['cmotion']，
    #    reactor 动作作为 motion 训练目标。
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, 
                              num_person=args.num_person, data_path = args.data_path, pose_rep = args.pose_rep, body_model=args.body_model, setting=args.setting, ar_shuffle=args.shuffle,
                              shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size(), max_samples=args.max_samples)

    print("creating model and diffusion...")

    # 9. 根据参数和数据集元信息创建 CMDM 模型与 Gaussian Diffusion 训练过程。
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())

    # 10. rot2xyz 内部包含 SMPL/SMPL-X 模型，这里切到 eval，避免训练人体模型参数。
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")

    # 11. 进入训练循环。每一步训练、loss 计算、反向传播、保存 checkpoint 都在 TrainLoop 中完成。
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
