import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loaders.forecasting import InterHumanForecastDataset, forecasting_collate
from eval.eval_forecasting_xyz import evaluate_xyz_model
from model.forecasting_somoformer import ensure_xyz_prediction_shape
from model.forecasting_xyz import (
    XYZ_FORECASTING_MODEL_TYPES,
    count_parameters,
    create_xyz_forecasting_model,
)
from model.rotation2xyz import Rotation2xyz
from utils.fixseed import fixseed
from utils.forecasting_xyz import active_to_xyz


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_save_dir(args):
    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    if os.path.exists(args.save_dir):
        has_files = len(os.listdir(args.save_dir)) > 0
        if has_files and not args.overwrite and args.resume_checkpoint is None:
            raise FileExistsError(
                "save_dir [{}] already exists. 使用 --overwrite 或更换 save_dir。".format(args.save_dir)
            )
    os.makedirs(args.save_dir, exist_ok=True)


def _write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)


def _append_train_log(args, record):
    path = os.path.join(args.save_dir, "train_log.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=False, ensure_ascii=False))
        f.write("\n")


def _build_dataset(args, split):
    return InterHumanForecastDataset(
        data_path=args.data_path,
        split=split,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )


def _build_loader(args, split, shuffle):
    dataset = _build_dataset(args, split)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=forecasting_collate,
    )


def _checkpoint_paths(args, step):
    return (
        os.path.join(args.save_dir, "model{:09d}.pt".format(step)),
        os.path.join(args.save_dir, "opt{:09d}.pt".format(step)),
    )


def _save_checkpoint(args, model, optimizer, step):
    model_path, opt_path = _checkpoint_paths(args, step)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "model_config": model.config(),
            "num_params": args.num_params,
            "step": int(step),
            "seed": int(args.seed),
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


def _load_resume(args, model, optimizer, device):
    if args.resume_checkpoint is None:
        return 0
    state = torch.load(args.resume_checkpoint, map_location=device)
    if state.get("model_type") != args.model_type:
        raise ValueError(
            "resume checkpoint model_type={} 与当前 model_type={} 不一致".format(
                state.get("model_type"), args.model_type
            )
        )
    model.load_state_dict(state["model_state_dict"])
    step = int(state.get("step", 0))
    opt_path = os.path.join(os.path.dirname(args.resume_checkpoint), "opt{:09d}.pt".format(step))
    if os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(opt_state["optimizer_state_dict"])
    return step


def _save_args(args):
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))


def _eval_args(args, split, checkpoint_path):
    return SimpleNamespace(
        dataset=args.dataset,
        data_path=args.data_path,
        split=split,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
        checkpoint=checkpoint_path,
        save_dir=args.save_dir,
    )


def _build_model(args):
    return create_xyz_forecasting_model(
        model_type=args.model_type,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        dct_n=args.dct_n,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        residual_connection=args.residual_connection,
        output_scale=args.output_scale,
        location_method=args.location_method,
        grid_len=args.grid_len,
        grid_emb_size=args.grid_emb_size,
        normalize_inputs=args.normalize_inputs,
        activation=args.activation,
        learned_embedding=args.learned_embedding,
    )


def _train_step(model, converter, obs, target, args, device):
    with torch.no_grad():
        obs_xyz = active_to_xyz(obs, device=device, converter=converter)
        target_xyz = active_to_xyz(target, device=device, converter=converter)
    if hasattr(model, "training_loss"):
        loss = model.training_loss(
            obs_xyz,
            target_xyz,
            aux_weight=args.aux_weight,
            metamask=args.metamask,
        )
        if not torch.isfinite(loss):
            raise ValueError("训练 loss 为非有限数值: {}".format(float(loss.detach().cpu().item())))
        return loss
    pred_xyz = model(obs_xyz)
    ensure_xyz_prediction_shape(pred_xyz, batch_size=int(obs.shape[0]), pred_len=args.pred_len)
    loss = F.mse_loss(pred_xyz, target_xyz)
    if not torch.isfinite(loss):
        raise ValueError("训练 loss 为非有限数值: {}".format(float(loss.detach().cpu().item())))
    return loss


def _evaluate_split(args, model, split, checkpoint_path, step, device):
    eval_args = _eval_args(args, split, checkpoint_path)
    return evaluate_xyz_model(
        args=eval_args,
        model=model,
        checkpoint_state={"step": int(step)},
        device=device,
    )


def run_training(args):
    if args.dataset != "interhuman":
        raise ValueError("P7.1 xyz 训练只支持 interhuman dataset")
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")

    fixseed(args.seed)
    _prepare_save_dir(args)
    device = _device()

    train_loader = _build_loader(args, "train", shuffle=True)
    model = _build_model(args).to(device)
    converter = Rotation2xyz(device=device)
    args.num_params = count_parameters(model)
    args.effective_batch_size = int(args.batch_size * max(1, args.grad_accum_steps))
    args.created_at = _utc_now()
    args.device = str(device)
    _save_args(args)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = _load_resume(args, model, optimizer, device)
    optimizer.zero_grad()

    print("Training joint-space forecasting model...")
    print(
        "model_type={} params={} device={} effective_batch_size={}".format(
            args.model_type, args.num_params, device, args.effective_batch_size
        )
    )

    accum_batches = 0
    recent_losses = []
    latest_checkpoint = None

    while step < args.num_steps:
        for obs, target, meta in train_loader:
            if step >= args.num_steps:
                break

            model.train()
            loss = _train_step(model, converter, obs, target, args, device)
            (loss / float(args.grad_accum_steps)).backward()
            recent_losses.append(float(loss.detach().cpu().item()))
            accum_batches += 1

            if accum_batches % args.grad_accum_steps != 0:
                continue

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            train_loss = sum(recent_losses) / float(len(recent_losses))
            recent_losses = []
            record = OrderedDict()
            record["step"] = int(step)
            record["train_loss"] = train_loss
            record["lr"] = args.lr
            record["effective_batch_size"] = args.effective_batch_size
            record["model_num_params"] = args.num_params
            record["seed"] = args.seed

            if step == 1 or step % args.log_interval == 0:
                print("step[{}]: train_loss[{:.6f}]".format(step, train_loss))

            save_due = step % args.save_interval == 0 or step == args.num_steps
            eval_due = args.eval_interval > 0 and (
                step % args.eval_interval == 0 or step == args.num_steps
            )
            if save_due or eval_due:
                model_path, opt_path = _save_checkpoint(args, model, optimizer, step)
                latest_checkpoint = model_path
                record["checkpoint"] = model_path
                record["optimizer"] = opt_path

            if eval_due:
                val_summary = _evaluate_split(args, model, "val", latest_checkpoint, step, device)
                record["val_joint_mse"] = val_summary["metrics"]["joint_mse"]

            _append_train_log(args, record)

    if latest_checkpoint is None:
        latest_checkpoint, _ = _save_checkpoint(args, model, optimizer, step)

    _evaluate_split(args, model, "test", latest_checkpoint, step, device)
    print("Training finished. final_checkpoint={}".format(latest_checkpoint))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="interhuman")
    parser.add_argument("--data_path", default="dataset/interhuman/smpl/conditioned")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_type", default="somoformer_xyz", choices=XYZ_FORECASTING_MODEL_TYPES)
    parser.add_argument("--window_len", type=int, default=150)
    parser.add_argument("--obs_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--dct_n", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--output_scale", type=float, default=1.0)
    parser.add_argument("--location_method", default="grid", choices=("grid", "neck", "naive"))
    parser.add_argument("--grid_len", type=int, default=3)
    parser.add_argument("--grid_emb_size", type=int, default=8)
    parser.add_argument("--normalize_inputs", action="store_true")
    parser.add_argument("--learned_embedding", action="store_true", default=True)
    parser.add_argument("--aux_weight", type=float, default=0.2)
    parser.add_argument("--metamask", action="store_true", default=True)
    parser.add_argument("--no_metamask", dest="metamask", action="store_false")
    parser.add_argument("--residual_connection", action="store_true", default=True)
    parser.add_argument("--no_residual_connection", dest="residual_connection", action="store_false")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    run_training(args)


if __name__ == "__main__":
    main()
