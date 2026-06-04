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
from eval.eval_forecasting import evaluate_forecasting_model
from model.forecasting import (
    FORECASTING_MODEL_TYPES,
    RELATION_ENCODER_TYPES,
    count_parameters,
    create_forecasting_model,
    ensure_prediction_shape,
)
from utils.fixseed import fixseed
from utils.forecasting_motion import (
    RELATION_FEATURE_SETS,
    compute_forecasting_normalizer,
    load_forecasting_normalizer,
)


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
                "save_dir [{}] already exists. 使用 --overwrite 或更换 save_dir。".format(
                    args.save_dir
                )
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


def _save_checkpoint(args, model, optimizer, step, normalizer_path):
    model_path, opt_path = _checkpoint_paths(args, step)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "model_config": model.config(),
            "num_params": args.num_params,
            "step": int(step),
            "seed": int(args.seed),
            "normalizer_path": normalizer_path,
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
    if "model_state_dict" not in state:
        raise ValueError("resume checkpoint 缺少 model_state_dict")
    if state.get("model_type") != args.model_type:
        raise ValueError(
            "resume model_type={} 与当前 model_type={} 不一致".format(
                state.get("model_type"), args.model_type
            )
        )
    model.load_state_dict(state["model_state_dict"])
    step = int(state.get("step", 0))

    opt_path = os.path.join(
        os.path.dirname(args.resume_checkpoint), "opt{:09d}.pt".format(step)
    )
    if os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(opt_state["optimizer_state_dict"])
    return step


def _save_args(args):
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))


def _eval_args(args, split, checkpoint_path, normalizer_path):
    return SimpleNamespace(
        mode="checkpoint",
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
        model_type=args.model_type,
        normalizer=normalizer_path,
        save_dir=args.save_dir,
    )


def _evaluate_split(args, model, normalizer, split, checkpoint_path, device):
    eval_args = _eval_args(args, split, checkpoint_path, os.path.join(args.save_dir, "normalizer.pt"))
    return evaluate_forecasting_model(
        args=eval_args,
        model=model,
        normalizer=normalizer,
        model_type=args.model_type,
        checkpoint_path=checkpoint_path,
        checkpoint_state={"step": args.current_step},
        device=device,
    )


def _load_or_create_normalizer(args):
    if args.resume_checkpoint is not None:
        normalizer_path = os.path.join(args.save_dir, "normalizer.pt")
        if os.path.exists(normalizer_path):
            return load_forecasting_normalizer(normalizer_path)
    return compute_forecasting_normalizer(
        data_path=args.data_path,
        save_dir=args.save_dir,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
    )


def _train_step(model, normalizer, obs, target, args, device):
    obs = obs.to(device)
    target = target.to(device)
    obs_normalized = normalizer.normalize(obs)
    target_normalized = normalizer.normalize(target)
    pred_normalized = model(obs_normalized)
    ensure_prediction_shape(
        pred_normalized,
        batch_size=int(obs.shape[0]),
        pred_len=args.pred_len,
    )
    loss = F.mse_loss(pred_normalized, target_normalized)
    if not torch.isfinite(loss):
        raise ValueError("训练 loss 为非有限数值: {}".format(float(loss.detach().cpu().item())))
    return loss


def run_training(args):
    if args.dataset != "interhuman":
        raise ValueError("forecasting 训练只支持 interhuman dataset")
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")

    fixseed(args.seed)
    _prepare_save_dir(args)
    device = _device()

    normalizer = _load_or_create_normalizer(args)
    normalizer_path = os.path.join(args.save_dir, "normalizer.pt")

    train_loader = _build_loader(args, "train", shuffle=True)
    model = create_forecasting_model(
        model_type=args.model_type,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        relation_hidden_dim=args.relation_hidden_dim,
        relation_num_layers=args.relation_num_layers,
        relation_feature_set=args.relation_feature_set,
        relation_encoder_type=args.relation_encoder_type,
    )
    model.to(device)
    args.num_params = count_parameters(model)
    args.effective_batch_size = int(args.batch_size * max(1, args.grad_accum_steps))
    args.created_at = _utc_now()
    args.device = str(device)
    _save_args(args)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = _load_resume(args, model, optimizer, device)
    optimizer.zero_grad()

    print("Training forecasting model...")
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
            loss = _train_step(model, normalizer, obs, target, args, device)
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
            args.current_step = step

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
                model_path, opt_path = _save_checkpoint(
                    args, model, optimizer, step, normalizer_path
                )
                latest_checkpoint = model_path
                record["checkpoint"] = model_path
                record["optimizer"] = opt_path

            if eval_due:
                val_summary = _evaluate_split(
                    args, model, normalizer, "val", latest_checkpoint, device
                )
                record["val_future_mse"] = val_summary["metrics"]["future_mse"]

            _append_train_log(args, record)

    if latest_checkpoint is None:
        args.current_step = step
        latest_checkpoint, _ = _save_checkpoint(args, model, optimizer, step, normalizer_path)

    args.current_step = step
    _evaluate_split(args, model, normalizer, "test", latest_checkpoint, device)
    print("Training finished. final_checkpoint={}".format(latest_checkpoint))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="interhuman")
    parser.add_argument("--data_path", default="dataset/interhuman/smpl/conditioned")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_type", required=True, choices=FORECASTING_MODEL_TYPES)
    parser.add_argument("--window_len", type=int, default=150)
    parser.add_argument("--obs_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--relation_hidden_dim", type=int, default=128)
    parser.add_argument("--relation_num_layers", type=int, default=1)
    parser.add_argument("--relation_feature_set", default="all", choices=RELATION_FEATURE_SETS)
    parser.add_argument("--relation_encoder_type", default="gru", choices=RELATION_ENCODER_TYPES)
    parser.add_argument("--lr", type=float, default=1e-3)
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
