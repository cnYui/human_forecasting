from model.cmdm import CMDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    # 1. 读取命令行里的训练框架设置；当前 ReGenNet 主路径使用 setting='cmdm'。
    setting = args.setting

    # 2. 如果是 CMDM 框架，就创建条件动作扩散模型。
    if setting == 'cmdm':
        # 3. get_model_args 会根据命令行参数和数据集信息整理 CMDM 构造参数。
        #    ** 会把返回的 dict 展开成 CMDM(...) 的关键字参数。
        model = CMDM(**get_model_args(args, data))

        # 4. 原始 batch 是双人数据，但 ccollate 已经拆成 actor 条件 cmotion 和 reactor 目标 motion。
        #    diffusion 只负责给 reactor 这个单人目标动作加噪、去噪和算 loss。
        args.num_person = 1

    # 5. 创建高斯扩散过程；它定义训练时怎么加噪、怎么算 loss，以及采样时怎么去噪。
    diffusion = create_gaussian_diffusion(args)

    # 6. 返回神经网络 model 和扩散过程 diffusion，后续会交给 TrainLoop 使用。
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1
    if hasattr(data.dataset, 'num_person'):
        num_person = data.dataset.num_person
    else:
        num_person = 1

    # SMPL defaults
    data_rep = args.pose_rep
    body_model = args.body_model
    if body_model == 'smpl':
        njoints = 25
    elif body_model == 'smplx':
        njoints = 56
    if data_rep == 'rot6d':
        nfeats = 6
    elif data_rep == 'xyz':
        nfeats = 3

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    if args.dataset == 'ntu':
        num_frames = 60
    elif args.dataset in ['chi3d', 'interhuman']:
        num_frames = 150

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions, 'num_person': num_person, 
            'num_frames': num_frames,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch, 'cm_mode': args.cm_mode,
            'body_model': body_model, 'wo_pos_emb': args.wo_pos_emb, 'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = args.timestep_respacing
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_orient=args.lambda_orient,
        lambda_body=args.lambda_body,
        lambda_transl=args.lambda_transl,
        data_rep=args.pose_rep,
        num_person=args.num_person,
        body_model=args.body_model,
        vel_threshold=args.vel_threshold,
    )
