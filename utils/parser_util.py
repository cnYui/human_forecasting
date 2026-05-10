from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # 从已训练模型同目录的 args.json 恢复参数。
    # 不要在命令行手动指定这些参数，因为后面会被 args.json 中的训练配置覆盖。
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # 读取模型 checkpoint 所在目录下保存的训练参数。
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # 向后兼容旧版本 checkpoint
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args

def parse_and_load_from_model_wo_data(parser):
    # 从已训练模型同目录的 args.json 恢复模型和扩散参数，但不覆盖数据集参数。
    # cgenerate 会显式传入数据相关参数，因此这里不加载 data options。
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # 读取模型 checkpoint 所在目录下保存的训练参数。
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # 向后兼容旧版本 checkpoint
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    # 基础参数：训练、采样、评估都会用到。
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--use_ddim", action='store_true',
                       help="Use DDIM to accelerate the inference or not.")
    group.add_argument("--timestep_respacing", default="", type=str, help="ddim timestep respacing.")

def add_diffusion_options(parser):
    # 高斯扩散过程参数：控制噪声日程、扩散步数和方差设置。
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    # 模型参数：控制 MDM/CMDM 框架、Transformer/GRU/MLP 架构和动作专用 loss 权重。
    group = parser.add_argument_group('model')
    group.add_argument("--setting", default='mdm', choices=['mdm', 'cmdm'], type=str,
                       help="Training MDM or CMDM framework")
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru', 'mlp', 'online', 'offline'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--wo_pos_emb", action='store_true',
                       help="Add positional embedding or not.")
    group.add_argument("--cm_mode", default='concat', # 待整理：concat2 当前没有完整实现路径
                       choices=['add', 'concat', 'concat2'], type=str,
                       help="Conditional modeling modes as reported in the paper.")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_orient", default=1.0, type=float, help="Explicit orientation loss.")
    group.add_argument("--lambda_body", default=1.0, type=float, help="Explicit body pose loss.")
    group.add_argument("--lambda_transl", default=1.0, type=float, help="Explicit root translation loss.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")



def add_data_options(parser):
    # 数据参数：决定读取哪个数据集、几个人、动作表示格式和人体模型类型。
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc', 'ntu', 'chi3d', 'interhuman', 'gta', 'sbu'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--num_person", default=1, type=int, help="number of persons")
    group.add_argument("--data_path", default="", type=str, help="Path of the data")
    group.add_argument("--pose_rep", default="rot6d", help="xyz or rotvec etc")
    group.add_argument("--body_model", default='smpl', choices=['smpl', 'smplx'], type=str,
                       help="Use SMPL model or SMPl-X model.")
    group.add_argument("--vel_threshold", default=0.01, type=float, help="Threshold of the velocity.")
    group.add_argument("--shuffle", action='store_true', help="Shuffle the actor-reactor order during training.")
    group.add_argument("--max_samples", default=-1, type=int,
                       help="Limit dataset samples for smoke tests. -1 uses all samples.")

def add_training_options(parser):
    # 训练参数：控制保存目录、优化器、训练步数、日志、checkpoint 和训练中评估。
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=10_000, type=int, # 原始设置是 50_000
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--grad_accum_steps", default=1, type=int,
                       help="Number of forward/backward passes before each optimizer step.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")


def add_sampling_options(parser):
    # 采样参数：控制从哪个 checkpoint 生成、输出目录、样本数量和 CFG guidance 强度。
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def add_generate_options(parser):
    # 生成参数：控制文本、动作类别或动作长度等生成条件。
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=60, type=float,
                       help="The length of the sampled motion [in frames]. ")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")


def add_edit_options(parser):
    # 编辑参数：用于 motion editing，不参与普通训练入口。
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    # 评估参数：控制被评估模型、动作识别模型和评估模式。
    group = parser.add_argument_group('eval')

    # 1. 指定要评估的 ReGenNet/CMDM 生成模型 checkpoint。
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")

    # 2. 指定评估指标使用的动作识别模型 checkpoint；它不是待评估的生成模型。
    group.add_argument("--rec_model_path", required=True, type=str,
                       help="Path to model####.pt of the action recognition model.")

    # 3. 选择评估规模，例如 debug 用于快速检查，full 用于完整指标。
    group.add_argument("--eval_mode", default='debug', type=str, help="Evaluation mode.")

    # 4. 采样阶段的 classifier-free guidance 强度；训练 cond_mask_prob 为 0 时通常会被重置为 1。
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")

    # 5. 控制评估时是否按自回归方式逐段/逐帧生成。
    group.add_argument("--auto_regressive", action='store_true',
                       help="Auto-regressive evaluation or not.")


def train_args():
    # 训练入口 train/train_mdm.py 只会使用下面这五组参数。
    # 参数来源顺序就是：base -> dataset -> model -> diffusion -> training。

    # 1. 创建训练专用 ArgumentParser。
    parser = ArgumentParser()

    # 2. 注册通用参数，例如设备、随机种子、batch size 和 DDIM 设置。
    add_base_options(parser)

    # 3. 注册数据参数，例如 dataset、data_path、num_person、pose_rep 和 body_model。
    add_data_options(parser)

    # 4. 注册模型参数，例如 setting=cmdm、arch=online、cm_mode 和各类 loss 权重。
    add_model_options(parser)

    # 5. 注册扩散过程参数，例如 noise_schedule、diffusion_steps 和 sigma_small。
    add_diffusion_options(parser)

    # 6. 注册训练参数，例如 save_dir、lr、num_steps、日志和 checkpoint 间隔。
    add_training_options(parser)

    # 7. 解析命令行参数并返回给 train/train_mdm.py。
    return parser.parse_args()


def generate_args():
    # 1. 创建普通采样入口的参数解析器，主要服务 sample/generate.py。
    parser = ArgumentParser()

    # 2. 用户只需要指定基础、采样和生成条件；模型/数据/扩散参数从 args.json 恢复。
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)

    # 3. 从 checkpoint 同目录 args.json 加载训练时的数据、模型和扩散配置。
    return parse_and_load_from_model(parser)

def cgenerate_args():
    # 1. 创建条件生成入口的参数解析器，主要服务 sample/cgenerate.py。
    parser = ArgumentParser()

    # 2. cgenerate 需要显式传入数据参数，用来构造 actor motion 条件数据加载器。
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)

    # 3. 只从 args.json 恢复模型和扩散配置，保留命令行传入的数据参数。
    return parse_and_load_from_model_wo_data(parser)


def edit_args():
    # 1. 创建 motion editing 入口的参数解析器。
    parser = ArgumentParser()

    # 2. 编辑任务只需要基础参数、采样 checkpoint 和编辑区域/文本条件。
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)

    # 3. 从 checkpoint 同目录 args.json 恢复训练时的数据、模型和扩散配置。
    return parse_and_load_from_model(parser)


def evaluation_parser():
    # 1. 创建评估入口的参数解析器，主要服务 eval/eval_cmdm.py。
    parser = ArgumentParser()

    # 2. 评估时命令行只指定设备、待评估模型、识别模型和评估模式。
    add_base_options(parser)
    add_evaluation_options(parser)

    # 3. 从待评估模型同目录 args.json 恢复训练配置，保证评估模型结构一致。
    return parse_and_load_from_model(parser)
