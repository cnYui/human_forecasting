import argparse
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_label import (
    HANDSHAKING_LABEL,
    NTU_FEATS,
    NTU_JOINTS,
    NUM_ACTIONS,
    NTULabelForecastDataset,
    ntu_label_forecasting_collate,
    summarize_entries,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="检查 NTU120 2P label-conditioned forecasting 数据协议"
    )
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--no_scan_all",
        action="store_true",
        help="只检查首个 batch，不扫描所有 kept samples 的 finite 状态",
    )
    return parser.parse_args()


def _format_counts(label_counts):
    parts = []
    for idx, count in enumerate(label_counts):
        parts.append("{}:{}".format(idx, count))
    return " ".join(parts)


def _print_summary(name, summary):
    print("[{}] raw_count={}".format(name, summary["raw_count"]))
    print("[{}] kept_count={}".format(name, summary["kept_count"]))
    print("[{}] skipped_too_short={}".format(name, summary["skipped_too_short"]))
    print("[{}] covered_labels={}".format(name, len(summary["covered_labels"])))
    print("[{}] missing_labels={}".format(name, summary["missing_labels"]))
    print("[{}] min_class_count={}".format(name, summary["min_class_count"]))
    print("[{}] handshaking_count={}".format(name, summary["handshaking_count"]))
    print("[{}] length_min={}".format(name, summary["length_min"]))
    print("[{}] length_max={}".format(name, summary["length_max"]))
    print("[{}] length_mean={:.2f}".format(name, summary["length_mean"]))
    print("[{}] label_counts={}".format(name, _format_counts(summary["label_counts"])))


def _check_summary(name, summary):
    if summary["kept_count"] <= 0:
        raise AssertionError("{} 过滤后为空".format(name))
    if len(summary["covered_labels"]) != NUM_ACTIONS:
        raise AssertionError(
            "{} 未覆盖全部 {} 类，missing={}".format(
                name, NUM_ACTIONS, summary["missing_labels"]
            )
        )
    if summary["handshaking_count"] <= 0:
        raise AssertionError("{} handshaking label {} 为 0".format(name, HANDSHAKING_LABEL))


def _make_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=ntu_label_forecasting_collate,
    )


def _check_batch(name, batch, obs_len, pred_len):
    expected_obs = (NTU_JOINTS, NTU_FEATS, obs_len)
    expected_future = (NTU_JOINTS, NTU_FEATS, pred_len)

    print("[{} batch] obs_motion={}".format(name, tuple(batch["obs_motion"].shape)))
    print("[{} batch] future={}".format(name, tuple(batch["future"].shape)))
    print("[{} batch] action={}".format(name, tuple(batch["action"].shape)))
    print("[{} batch] mask={}".format(name, tuple(batch["mask"].shape)))

    if tuple(batch["obs_motion"].shape[1:]) != expected_obs:
        raise AssertionError("{} obs_motion shape 错误".format(name))
    if tuple(batch["future"].shape[1:]) != expected_future:
        raise AssertionError("{} future shape 错误".format(name))
    if tuple(batch["action"].shape[1:]) != (1,):
        raise AssertionError("{} action shape 错误".format(name))
    if tuple(batch["mask"].shape[1:]) != (1, 1, pred_len):
        raise AssertionError("{} mask shape 错误".format(name))
    if batch["obs_motion"].dtype != torch.float32:
        raise AssertionError("{} obs_motion dtype 错误".format(name))
    if batch["future"].dtype != torch.float32:
        raise AssertionError("{} future dtype 错误".format(name))
    if batch["action"].dtype != torch.long:
        raise AssertionError("{} action dtype 错误".format(name))
    if batch["mask"].dtype != torch.bool:
        raise AssertionError("{} mask dtype 错误".format(name))
    if not torch.isfinite(batch["obs_motion"]).all().item():
        raise AssertionError("{} obs_motion 首个 batch 存在 NaN/Inf".format(name))
    if not torch.isfinite(batch["future"]).all().item():
        raise AssertionError("{} future 首个 batch 存在 NaN/Inf".format(name))
    if not bool(batch["mask"].all().item()):
        raise AssertionError("{} mask 不是全 True".format(name))
    if batch["action"].min().item() < 0 or batch["action"].max().item() >= NUM_ACTIONS:
        raise AssertionError("{} action 超出 [0,{}]".format(name, NUM_ACTIONS - 1))


def _scan_entries_finite(name, h5_path, entries):
    bad = []
    with h5py.File(str(h5_path), "r") as h5:
        for entry in entries:
            sample_id = entry["sample_id"]
            data = h5[sample_id][()]
            if not np.isfinite(data).all():
                bad.append(sample_id)
                if len(bad) >= 10:
                    break
    if bad:
        raise AssertionError("{} 存在 NaN/Inf samples: {}".format(name, bad))
    print("[{} finite] checked_samples={} result=PASS".format(name, len(entries)))


def _build_dataset(path, split, args):
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


def main():
    args = _parse_args()
    print("python_executable={}".format(sys.executable))
    print("torch_version={}".format(torch.__version__))
    print("h5py_version={}".format(h5py.__version__))
    print("numpy_version={}".format(np.__version__))

    if args.obs_len + args.pred_len != args.window_len:
        raise AssertionError("obs_len + pred_len 必须等于 window_len")

    train_dataset = _build_dataset(args.train_path, "train", args)
    test_dataset = _build_dataset(args.test_path, "test", args)

    train_summary = summarize_entries(train_dataset.scan_result)
    test_summary = summarize_entries(test_dataset.scan_result)
    _print_summary("train", train_summary)
    _print_summary("test", test_summary)
    _check_summary("train", train_summary)
    _check_summary("test", test_summary)

    train_batch = next(iter(_make_loader(train_dataset, args.batch_size, args.num_workers)))
    test_batch = next(iter(_make_loader(test_dataset, args.batch_size, args.num_workers)))
    _check_batch("train", train_batch, args.obs_len, args.pred_len)
    _check_batch("test", test_batch, args.obs_len, args.pred_len)

    if not args.no_scan_all:
        _scan_entries_finite("train", args.train_path, train_dataset.scan_result["entries"])
        _scan_entries_finite("test", args.test_path, test_dataset.scan_result["entries"])

    print("PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL: {}".format(exc), file=sys.stderr)
        sys.exit(1)
