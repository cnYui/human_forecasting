import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting import NTULabelForecastDataset, ntu_label_forecasting_collate
from model.rotation2xyz import Rotation2xyz_x
from utils.ntu_smplx_2p_xyz import ntu_rotvec_2p_to_xyz


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_cache(args):
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = _device()
    dataset = NTULabelForecastDataset(
        h5_path=args.data_path,
        split=args.split,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ntu_label_forecasting_collate,
    )
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    obs_items = []
    target_items = []
    action_items = []
    meta = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            obs_xyz = ntu_rotvec_2p_to_xyz(batch["obs_motion"].to(device), device=device, converter=converter)
            target_xyz = ntu_rotvec_2p_to_xyz(batch["future"].to(device), device=device, converter=converter)
            obs_items.append(obs_xyz.cpu())
            target_items.append(target_xyz.cpu())
            action_items.append(batch["action"].cpu())
            meta.extend(batch["meta"])
            if batch_idx == 0 or (batch_idx + 1) % args.log_interval == 0:
                print("converted_batches={} samples={}".format(batch_idx + 1, len(meta)))
    payload = {
        "obs_xyz": torch.cat(obs_items, dim=0),
        "target_xyz": torch.cat(target_items, dim=0),
        "actions": torch.cat(action_items, dim=0),
        "meta": meta,
        "config": {
            "data_path": args.data_path,
            "split": args.split,
            "window_len": args.window_len,
            "obs_len": args.obs_len,
            "pred_len": args.pred_len,
            "num_samples": len(meta),
            "created_at": _utc_now(),
        },
    }
    torch.save(payload, args.save_path)
    summary_path = args.save_path + ".json"
    with open(summary_path, "w") as f:
        json.dump(payload["config"], f, indent=2, sort_keys=False, ensure_ascii=False)
    print(json.dumps(payload["config"], indent=2, sort_keys=False, ensure_ascii=False))
    print("saved={}".format(args.save_path))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    return parser


def main():
    build_cache(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
