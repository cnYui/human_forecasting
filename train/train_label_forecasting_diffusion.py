import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_label import (
    NTULabelForecastDataset,
    ntu_label_forecasting_collate,
)
from diffusion import gaussian_diffusion as gd
from diffusion.resample import create_named_schedule_sampler
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.forecasting_cmdm import ForecastingCMDMDecoder, count_parameters
from utils.fixseed import fixseed


MODEL_TYPE = "forecasting_cmdm_decoder"
DATASET = "ntu120_2p"
NUM_ACTIONS = 26
NJOINTS = 56
NFEATS = 6
DIFFUSION_STEPS = 1000


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)


def _append_train_log(args, record):
    path = os.path.join(args.save_dir, "train_log.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=False, ensure_ascii=False))
        f.write("\n")


def _is_stage_output(filename):
    if filename in ("args.json", "train_log.jsonl"):
        return True
    return (
        (filename.startswith("model") or filename.startswith("opt"))
        and filename.endswith(".pt")
    )


def _clear_stage_outputs(save_dir):
    for filename in os.listdir(save_dir):
        if _is_stage_output(filename):
            path = os.path.join(save_dir, filename)
            if os.path.isfile(path):
                os.remove(path)


def _prepare_save_dir(args):
    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")

    if os.path.exists(args.save_dir):
        has_files = len(os.listdir(args.save_dir)) > 0
        if has_files and args.resume_checkpoint is None and not args.overwrite:
            raise FileExistsError(
                "save_dir [{}] already exists. 使用 --overwrite 或更换 save_dir。".format(
                    args.save_dir
                )
            )
        if has_files and args.resume_checkpoint is None and args.overwrite:
            _clear_stage_outputs(args.save_dir)
    else:
        os.makedirs(args.save_dir)


def _ensure_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError("{} 存在 NaN 或 Inf".format(name))


def _build_dataset(args, path, split):
    return NTULabelForecastDataset(
        h5_path=path,
        split=split,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
        strict=True,
    )


def _build_loader(args, path, split, shuffle, batch_size):
    dataset = _build_dataset(args, path, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=ntu_label_forecasting_collate,
        drop_last=False,
    )


def _build_model(args):
    if args.model_type != MODEL_TYPE:
        raise ValueError("model_type 必须是 {}".format(MODEL_TYPE))
    return ForecastingCMDMDecoder(
        model_type=args.model_type,
        njoints=NJOINTS,
        nfeats=NFEATS,
        num_actions=NUM_ACTIONS,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        window_len=args.window_len,
        latent_dim=args.latent_dim,
        obs_encoder_layers=args.obs_encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        activation=args.activation,
        cond_mask_prob=args.cond_mask_prob,
        data_rep="rot6d",
        body_model=args.body_model,
        dataset=args.dataset,
        init_rot2xyz=False,
    )


def _timestep_respacing(value):
    if value is None or value == "":
        return [DIFFUSION_STEPS]
    return value


def _diffusion_config(args):
    return {
        "steps": DIFFUSION_STEPS,
        "noise_schedule": args.noise_schedule,
        "timestep_respacing": args.timestep_respacing,
        "sigma_small": bool(args.sigma_small),
        "model_mean_type": "START_X",
        "model_var_type": "FIXED_SMALL" if args.sigma_small else "FIXED_LARGE",
        "loss_type": "MSE",
        "rescale_timesteps": False,
        "data_rep": "rot6d",
        "num_person": 2,
        "body_model": args.body_model,
    }


def _build_diffusion(args):
    betas = gd.get_named_beta_schedule(args.noise_schedule, DIFFUSION_STEPS, 1.0)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(
            DIFFUSION_STEPS,
            _timestep_respacing(args.timestep_respacing),
        ),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=(
            gd.ModelVarType.FIXED_SMALL
            if args.sigma_small
            else gd.ModelVarType.FIXED_LARGE
        ),
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        lambda_rcxyz=0.0,
        lambda_vel=0.0,
        lambda_fc=0.0,
        lambda_orient=0.0,
        lambda_body=0.0,
        lambda_transl=0.0,
        data_rep="rot6d",
        num_person=2,
        body_model=args.body_model,
        vel_threshold=args.vel_threshold,
    )


def _masked_l2(a, b, mask):
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError("masked_l2 输入 shape 不一致: {} vs {}".format(tuple(a.shape), tuple(b.shape)))
    if mask.dim() != 4 or mask.shape[0] != a.shape[0] or mask.shape[-1] != a.shape[-1]:
        raise ValueError("mask 必须是 [B,1,1,T]，当前为 {}".format(tuple(mask.shape)))

    mask = mask.to(device=a.device, dtype=a.dtype)
    loss = ((a - b) ** 2) * mask
    loss = loss.sum(dim=(1, 2, 3))
    denom = mask.sum(dim=(1, 2, 3)) * a.shape[1] * a.shape[2]
    return loss / denom.clamp_min(1.0)


def _velocity_mse(pred, target, mask):
    pred_vel = pred[..., 1:] - pred[..., :-1]
    target_vel = target[..., 1:] - target[..., :-1]
    return _masked_l2(pred_vel, target_vel, mask[..., 1:])


def _root_translation_mse(pred, target, mask):
    pred_root = pred[:, -1:, 0:3, :]
    target_root = target[:, -1:, 0:3, :]
    return _masked_l2(pred_root, target_root, mask)


def _relative_root_mse(pred, target, mask):
    raise NotImplementedError("relative_root_mse 需要先确认 NTU120 SMPL-X 双人 root slot 索引")


def _mean_float(tensor):
    return float(tensor.detach().mean().cpu().item())


def _compute_losses(pred, target, mask, args):
    _ensure_finite("pred", pred)
    _ensure_finite("target", target)
    _ensure_finite("mask", mask.float())

    terms = OrderedDict()
    terms["rot_mse"] = _masked_l2(pred, target, mask)
    loss = terms["rot_mse"]

    if args.velocity_loss_weight > 0.0:
        terms["velocity_mse"] = _velocity_mse(pred, target, mask)
        loss = loss + float(args.velocity_loss_weight) * terms["velocity_mse"]

    if args.root_translation_loss_weight > 0.0:
        terms["root_translation_mse"] = _root_translation_mse(pred, target, mask)
        loss = loss + float(args.root_translation_loss_weight) * terms["root_translation_mse"]

    if args.relative_root_loss_weight > 0.0:
        terms["relative_root_mse"] = _relative_root_mse(pred, target, mask)
        loss = loss + float(args.relative_root_loss_weight) * terms["relative_root_mse"]

    terms["loss"] = loss
    return terms


def _batch_to_model_kwargs(batch, device):
    future = batch["future"].to(device)
    obs_motion = batch["obs_motion"].to(device)
    action = batch["action"].to(device)
    mask = batch["mask"].to(device)

    _ensure_finite("future", future)
    _ensure_finite("obs_motion", obs_motion)
    _ensure_finite("action", action.float())
    _ensure_finite("mask", mask.float())

    return future, {
        "obs_motion": obs_motion,
        "action": action,
        "mask": mask,
    }


def _diffusion_train_step(model, diffusion, schedule_sampler, batch, args, device):
    future, y = _batch_to_model_kwargs(batch, device)
    batch_size = int(future.shape[0])
    use_one_step_noise = (
        float(args.one_step_noise_prob) > 0.0
        and torch.rand((), device=device).item() < float(args.one_step_noise_prob)
    )

    if use_one_step_noise:
        t = torch.full(
            (batch_size,),
            int(args.one_step_t),
            device=device,
            dtype=torch.long,
        )
        weights = torch.ones(batch_size, device=device)
        x_t = torch.randn_like(future)
    elif args.timestep_sampling == "high_noise":
        min_t = int(args.high_noise_min_t)
        t = torch.randint(min_t, int(diffusion.num_timesteps), (batch_size,), device=device)
        weights = torch.ones(batch_size, device=device)
        noise = torch.randn_like(future)
        x_t = diffusion.q_sample(future, t, noise=noise)
    else:
        t, weights = schedule_sampler.sample(batch_size, device)
        noise = torch.randn_like(future)
        x_t = diffusion.q_sample(future, t, noise=noise)
    _ensure_finite("x_t", x_t)

    pred = model(x_t, t, y)
    terms = _compute_losses(pred, future, y["mask"], args)
    loss = (terms["loss"] * weights).mean()
    if not torch.isfinite(loss):
        raise ValueError("训练 loss 为非有限数值: {}".format(float(loss.detach().cpu().item())))

    metrics = OrderedDict()
    metrics["train_loss"] = float(loss.detach().cpu().item())
    for key, value in terms.items():
        metrics[key] = _mean_float(value)
    metrics["t_mean"] = float(t.float().mean().detach().cpu().item())
    metrics["one_step_noise_active"] = 1.0 if use_one_step_noise else 0.0
    return loss, metrics


def _checkpoint_paths(args, step):
    return (
        os.path.join(args.save_dir, "model{:09d}.pt".format(int(step))),
        os.path.join(args.save_dir, "opt{:09d}.pt".format(int(step))),
    )


def _train_protocol(args):
    return {
        "dataset": args.dataset,
        "window_len": int(args.window_len),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "num_actions": NUM_ACTIONS,
        "target": "future",
        "condition": "obs_motion + action",
        "mean_type": "START_X",
        "loss_type": "MSE",
        "timestep_sampling": args.timestep_sampling,
        "high_noise_min_t": int(args.high_noise_min_t),
        "one_step_noise_prob": float(args.one_step_noise_prob),
        "one_step_t": int(args.one_step_t),
    }


def _save_checkpoint(args, model, optimizer, step):
    model_path, opt_path = _checkpoint_paths(args, step)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "model_config": model.config(),
            "num_params": int(args.num_params),
            "step": int(step),
            "seed": int(args.seed),
            "diffusion_config": _diffusion_config(args),
            "train_protocol": _train_protocol(args),
            "created_at": _utc_now(),
        },
        model_path,
    )
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
        },
        opt_path,
    )
    return model_path, opt_path


def _core_config_keys():
    return (
        "njoints",
        "nfeats",
        "num_actions",
        "obs_len",
        "pred_len",
        "window_len",
        "latent_dim",
        "obs_encoder_layers",
        "decoder_layers",
        "num_heads",
        "ff_size",
    )


def _validate_resume_config(args, state, model):
    if state.get("model_type") != args.model_type:
        raise ValueError(
            "resume model_type={} 与当前 model_type={} 不一致".format(
                state.get("model_type"), args.model_type
            )
        )

    checkpoint_config = state.get("model_config", {})
    current_config = model.config()
    for key in _core_config_keys():
        if checkpoint_config.get(key) != current_config.get(key):
            raise ValueError(
                "resume config mismatch: {} checkpoint={} current={}".format(
                    key, checkpoint_config.get(key), current_config.get(key)
                )
            )


def _load_resume(args, model, optimizer, device):
    if args.resume_checkpoint is None:
        return 0
    if not os.path.exists(args.resume_checkpoint):
        raise FileNotFoundError(args.resume_checkpoint)

    state = torch.load(args.resume_checkpoint, map_location=device)
    _validate_resume_config(args, state, model)
    model.load_state_dict(state["model_state_dict"])
    step = int(state.get("step", 0))

    opt_path = os.path.join(os.path.dirname(args.resume_checkpoint), "opt{:09d}.pt".format(step))
    if os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(opt_state["optimizer_state_dict"])
    else:
        print("warning: optimizer checkpoint not found: {}".format(opt_path))
    return step


def _save_args(args):
    serializable = dict(vars(args))
    serializable["created_at"] = _utc_now()
    serializable["train_protocol"] = _train_protocol(args)
    serializable["diffusion_config"] = _diffusion_config(args)
    _write_json(os.path.join(args.save_dir, "args.json"), serializable)


def _validate_args(args):
    if args.dataset != DATASET:
        raise ValueError("dataset 必须是 {}".format(DATASET))
    if args.model_type != MODEL_TYPE:
        raise ValueError("model_type 必须是 {}".format(MODEL_TYPE))
    if args.body_model != "smplx":
        raise ValueError("body_model 必须是 smplx")
    if int(args.obs_len) + int(args.pred_len) != int(args.window_len):
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    if args.eval_interval != 0:
        raise ValueError("阶段 C 不支持 eval_interval > 0")
    if args.relative_root_loss_weight > 0.0:
        raise ValueError("relative_root_loss_weight 需要先确认 NTU120 SMPL-X 双人 root slot 索引")
    if args.timestep_sampling not in ("uniform", "high_noise"):
        raise ValueError("timestep_sampling 必须是 uniform 或 high_noise")
    if int(args.high_noise_min_t) < 0 or int(args.high_noise_min_t) >= DIFFUSION_STEPS:
        raise ValueError("high_noise_min_t 必须在 [0,{}) 内".format(DIFFUSION_STEPS))
    if float(args.one_step_noise_prob) < 0.0 or float(args.one_step_noise_prob) > 1.0:
        raise ValueError("one_step_noise_prob 必须在 [0,1] 内")
    if int(args.one_step_t) < 0 or int(args.one_step_t) >= DIFFUSION_STEPS:
        raise ValueError("one_step_t 必须在 [0,{}) 内".format(DIFFUSION_STEPS))


def run_training(args):
    _validate_args(args)
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    args.effective_batch_size = int(args.batch_size * args.grad_accum_steps)

    fixseed(args.seed)
    _prepare_save_dir(args)
    device = _device()

    train_loader = _build_loader(
        args,
        args.data_path,
        split="train",
        shuffle=True,
        batch_size=args.batch_size,
    )
    model = _build_model(args).to(device)
    diffusion = _build_diffusion(args)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.num_params = count_parameters(model)
    args.device = str(device)
    args.model_config = model.config()
    args.diffusion_num_timesteps = int(diffusion.num_timesteps)
    _save_args(args)

    step = _load_resume(args, model, optimizer, device)
    optimizer.zero_grad()

    print("Training label-conditioned forecasting diffusion...")
    print(
        "model_type={} params={} device={} effective_batch_size={} resume_step={}".format(
            args.model_type,
            args.num_params,
            device,
            args.effective_batch_size,
            step,
        )
    )

    accum_batches = 0
    recent_metrics = []
    latest_checkpoint = None

    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break

            model.train()
            loss, metrics = _diffusion_train_step(
                model,
                diffusion,
                schedule_sampler,
                batch,
                args,
                device,
            )
            (loss / float(args.grad_accum_steps)).backward()
            recent_metrics.append(metrics)
            accum_batches += 1

            if accum_batches % args.grad_accum_steps != 0:
                continue

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            record = _make_log_record(args, recent_metrics, step)
            recent_metrics = []

            if step == 1 or step % args.log_interval == 0:
                print("step[{}]: train_loss[{:.6f}] rot_mse[{:.6f}]".format(
                    step,
                    record["train_loss"],
                    record["rot_mse"],
                ))

            save_due = step % args.save_interval == 0 or step == args.num_steps
            if save_due:
                model_path, opt_path = _save_checkpoint(args, model, optimizer, step)
                latest_checkpoint = model_path
                record["checkpoint"] = model_path
                record["optimizer"] = opt_path

            _append_train_log(args, record)

    if latest_checkpoint is None:
        latest_checkpoint, _ = _save_checkpoint(args, model, optimizer, step)

    print("Training finished. final_checkpoint={}".format(latest_checkpoint))


def _make_log_record(args, metrics_list, step):
    record = OrderedDict()
    record["step"] = int(step)
    for key in metrics_list[0].keys():
        record[key] = sum(item[key] for item in metrics_list) / float(len(metrics_list))
    record["lr"] = float(args.lr)
    record["effective_batch_size"] = int(args.effective_batch_size)
    record["model_num_params"] = int(args.num_params)
    record["seed"] = int(args.seed)
    record["created_at"] = _utc_now()
    return record


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--eval_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_type", default=MODEL_TYPE)
    parser.add_argument("--body_model", default="smplx")
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--obs_encoder_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--cond_mask_prob", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--noise_schedule", default="cosine")
    parser.add_argument("--timestep_respacing", default="")
    parser.add_argument("--sigma_small", dest="sigma_small", action="store_true", default=True)
    parser.add_argument("--no_sigma_small", dest="sigma_small", action="store_false")
    parser.add_argument("--schedule_sampler", default="uniform")
    parser.add_argument("--timestep_sampling", choices=("uniform", "high_noise"), default="uniform")
    parser.add_argument("--high_noise_min_t", type=int, default=750)
    parser.add_argument("--one_step_noise_prob", type=float, default=0.0)
    parser.add_argument("--one_step_t", type=int, default=999)
    parser.add_argument("--velocity_loss_weight", type=float, default=0.0)
    parser.add_argument("--root_translation_loss_weight", type=float, default=0.0)
    parser.add_argument("--relative_root_loss_weight", type=float, default=0.0)
    parser.add_argument("--vel_threshold", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=1)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
