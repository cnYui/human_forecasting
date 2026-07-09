import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting import (
    NTULabelForecastDataset,
    NTULabelXYZCacheDataset,
    ntu_label_forecasting_collate,
    ntu_label_xyz_cache_collate,
)
from model.forecasting_ntu_xyz import create_ntu_label_xyz_model_from_config
from model.rotation2xyz import Rotation2xyz_x
from utils.ntu_smplx_2p_xyz import (
    NTU_XYZ_METRIC_KEYS,
    compute_ntu_xyz_metrics,
    copy_last_xyz,
    ntu_rotvec_2p_to_xyz,
)


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=False, ensure_ascii=False)


def _build_dataset(args):
    if args.xyz_cache is not None:
        return NTULabelXYZCacheDataset(args.xyz_cache, max_samples=args.max_samples)
    return NTULabelForecastDataset(
        h5_path=args.data_path,
        split=args.split,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )


def _build_loader(args, dataset):
    collate_fn = ntu_label_xyz_cache_collate if args.xyz_cache is not None else ntu_label_forecasting_collate
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )


def _load_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if "model_state_dict" not in state:
        raise ValueError("checkpoint 缺少 model_state_dict")
    if "model_config" not in state:
        raise ValueError("checkpoint 缺少 model_config")
    model = create_ntu_label_xyz_model_from_config(state["model_config"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


def _empty_sums():
    return OrderedDict((key, 0.0) for key in NTU_XYZ_METRIC_KEYS)


def _add_metrics(sums, metrics, batch_size):
    if tuple(metrics.keys()) != NTU_XYZ_METRIC_KEYS:
        raise AssertionError("metrics key 不稳定")
    for key in NTU_XYZ_METRIC_KEYS:
        sums[key] += float(metrics[key]) * float(batch_size)


def _finalize(sums, num_samples):
    return OrderedDict((key, sums[key] / float(num_samples)) for key in NTU_XYZ_METRIC_KEYS)


def evaluate_ntu_label_xyz(args, model=None, checkpoint_state=None, device=None):
    if device is None:
        device = _device()
    dataset = _build_dataset(args)
    loader = _build_loader(args, dataset)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")

    model_sums = _empty_sums()
    copy_sums = _empty_sums()
    num_samples = 0
    saved = {"obs_xyz": [], "target_xyz": [], "pred_xyz": [], "copy_last_xyz": [], "actions": [], "meta": []}

    if model is not None:
        model.eval()

    with torch.no_grad():
        for batch in loader:
            action = batch["action"].to(device)
            if "obs_xyz" in batch:
                obs_xyz = batch["obs_xyz"].to(device)
                target_xyz = batch["target_xyz"].to(device)
            else:
                obs_xyz = ntu_rotvec_2p_to_xyz(batch["obs_motion"].to(device), device=device, converter=converter)
                target_xyz = ntu_rotvec_2p_to_xyz(batch["future"].to(device), device=device, converter=converter)
            copy_xyz = copy_last_xyz(obs_xyz, args.pred_len)
            if model is None:
                pred_xyz = copy_xyz
            else:
                pred_xyz = model(obs_xyz, action)

            batch_size = int(obs_xyz.shape[0])
            _add_metrics(model_sums, compute_ntu_xyz_metrics(pred_xyz, target_xyz, obs_xyz), batch_size)
            _add_metrics(copy_sums, compute_ntu_xyz_metrics(copy_xyz, target_xyz, obs_xyz), batch_size)
            num_samples += batch_size

            if args.save_arrays and len(saved["meta"]) < int(args.save_array_limit):
                remain = int(args.save_array_limit) - len(saved["meta"])
                take = min(remain, batch_size)
                saved["obs_xyz"].append(obs_xyz[:take].detach().cpu())
                saved["target_xyz"].append(target_xyz[:take].detach().cpu())
                saved["pred_xyz"].append(pred_xyz[:take].detach().cpu())
                saved["copy_last_xyz"].append(copy_xyz[:take].detach().cpu())
                saved["actions"].append(action[:take].detach().cpu())
                saved["meta"].extend(batch["meta"][:take])

    if num_samples != len(dataset):
        raise AssertionError("评估样本数应为 {}，实际为 {}".format(len(dataset), num_samples))

    model_metrics = _finalize(model_sums, num_samples)
    copy_metrics = _finalize(copy_sums, num_samples)
    summary = OrderedDict()
    summary["mode"] = args.mode
    summary["dataset"] = "ntu120_2p_smplx"
    summary["split"] = args.split
    summary["data_path"] = args.data_path
    summary["xyz_cache"] = args.xyz_cache
    summary["window_len"] = args.window_len
    summary["obs_len"] = args.obs_len
    summary["pred_len"] = args.pred_len
    summary["num_samples"] = int(num_samples)
    summary["batch_size"] = args.batch_size
    summary["metrics_keys"] = list(NTU_XYZ_METRIC_KEYS)
    summary["model_metrics"] = model_metrics
    summary["copy_last_metrics"] = copy_metrics
    summary["beats_copy_last"] = {
        "xyz_mse": model_metrics["xyz_mse"] < copy_metrics["xyz_mse"],
        "xyz_mae": model_metrics["xyz_mae"] <= copy_metrics["xyz_mae"],
        "mpjpe": model_metrics["mpjpe"] < copy_metrics["mpjpe"],
    }
    summary["checkpoint"] = args.checkpoint
    summary["checkpoint_step"] = None if checkpoint_state is None else checkpoint_state.get("step")
    summary["created_at"] = _utc_now()

    if args.save_dir is not None:
        _write_json(os.path.join(args.save_dir, "metrics_{}.json".format(args.split)), summary)
        if args.save_arrays and len(saved["meta"]) > 0:
            array_dir = os.path.join(args.save_dir, "arrays")
            os.makedirs(array_dir, exist_ok=True)
            torch.save(
                {
                    "obs_xyz": torch.cat(saved["obs_xyz"], dim=0),
                    "target_xyz": torch.cat(saved["target_xyz"], dim=0),
                    "pred_xyz": torch.cat(saved["pred_xyz"], dim=0),
                    "copy_last_xyz": torch.cat(saved["copy_last_xyz"], dim=0),
                    "actions": torch.cat(saved["actions"], dim=0),
                    "meta": saved["meta"],
                },
                os.path.join(array_dir, "ntu_label_xyz_samples.pt"),
            )
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="checkpoint", choices=("checkpoint", "copy_last"))
    parser.add_argument("--data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--xyz_cache", default=None)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_dir", default="results/forecasting/ntu120_label/xyz_eval")
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--save_array_limit", type=int, default=8)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    device = _device()
    model = None
    state = None
    if args.mode == "checkpoint":
        if args.checkpoint is None:
            raise ValueError("--mode checkpoint 必须提供 --checkpoint")
        model, state = _load_checkpoint(args.checkpoint, device)
    evaluate_ntu_label_xyz(args, model=model, checkpoint_state=state, device=device)


if __name__ == "__main__":
    main()
