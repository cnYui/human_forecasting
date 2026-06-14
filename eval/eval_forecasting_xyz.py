import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting import InterHumanForecastDataset, forecasting_collate
from model.forecasting_somoformer import ensure_xyz_prediction_shape
from model.forecasting_xyz import create_xyz_forecasting_model_from_config
from model.rotation2xyz import Rotation2xyz
from utils.forecasting_xyz import XYZ_METRIC_KEYS, active_to_xyz, compute_xyz_metrics


EXPECTED_LENGTHS = {
    "train": 2910,
    "val": 226,
    "test": 508,
}


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _build_loader(args, dataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=forecasting_collate,
    )


def _yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _fallback_yaml_lines(value, indent=0):
    prefix = " " * indent
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append("{}{}:".format(prefix, key))
                lines.extend(_fallback_yaml_lines(item, indent + 2))
            else:
                lines.append("{}{}: {}".format(prefix, key, _yaml_scalar(item)))
        return lines
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append("{}-".format(prefix))
                lines.extend(_fallback_yaml_lines(item, indent + 2))
            else:
                lines.append("{}- {}".format(prefix, _yaml_scalar(item)))
        return lines
    lines.append("{}{}".format(prefix, _yaml_scalar(value)))
    return lines


def _plain_data(value):
    if isinstance(value, dict):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    return value


def _write_yaml(path, summary):
    try:
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(_plain_data(summary), f, sort_keys=False, allow_unicode=True)
        return
    except Exception:
        pass

    with open(path, "w") as f:
        f.write("\n".join(_fallback_yaml_lines(summary)))
        f.write("\n")


def _write_summary(save_dir, json_name, yaml_name, summary):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, json_name)
    yaml_path = os.path.join(save_dir, yaml_name)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False, ensure_ascii=False)
    _write_yaml(yaml_path, summary)
    return json_path, yaml_path


def _assert_xyz_metric_keys(metrics):
    if tuple(metrics.keys()) != XYZ_METRIC_KEYS:
        raise AssertionError(
            "xyz metrics key 应为 {}，实际为 {}".format(XYZ_METRIC_KEYS, tuple(metrics.keys()))
        )


def _assert_metrics_finite(metrics):
    for key, value in metrics.items():
        if not torch.isfinite(torch.tensor(float(value))):
            raise AssertionError("{} 指标为非有限数值: {}".format(key, value))


def _aggregate(metric_sums, metrics, batch_size):
    _assert_xyz_metric_keys(metrics)
    _assert_metrics_finite(metrics)
    for key in XYZ_METRIC_KEYS:
        metric_sums[key] += metrics[key] * float(batch_size)


def _check_dataset_length(dataset, max_samples):
    expected = EXPECTED_LENGTHS[dataset.split]
    if max_samples is not None and max_samples > 0:
        expected = min(expected, max_samples)
    if len(dataset) != expected:
        raise AssertionError("{} dataset length 应为 {}，实际为 {}".format(dataset.split, expected, len(dataset)))


def _load_checkpoint_model(args, device):
    if args.checkpoint is None:
        raise ValueError("--mode checkpoint 必须提供 --checkpoint")
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" not in state:
        raise ValueError("checkpoint 缺少 model_state_dict")
    if "model_config" not in state:
        raise ValueError("checkpoint 缺少 model_config")
    model = create_xyz_forecasting_model_from_config(state["model_config"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


def run_xyz_smoke(args):
    device = _device()
    converter = Rotation2xyz(device=device)
    datasets = {
        "train": _build_dataset(args, "train"),
        "val": _build_dataset(args, "val"),
        "test": _build_dataset(args, "test"),
    }
    summary = OrderedDict()
    summary["mode"] = "xyz_smoke"
    summary["dataset"] = args.dataset
    summary["data_path"] = args.data_path
    summary["device"] = str(device)
    summary["dataset_lengths"] = OrderedDict()
    summary["batch_shapes"] = OrderedDict()
    for split, dataset in datasets.items():
        _check_dataset_length(dataset, args.max_samples)
        summary["dataset_lengths"][split] = len(dataset)
        loader = _build_loader(args, dataset)
        obs, target, meta = next(iter(loader))
        with torch.no_grad():
            obs_xyz = active_to_xyz(obs, device=device, converter=converter)
            target_xyz = active_to_xyz(target, device=device, converter=converter)
        summary["batch_shapes"][split] = {
            "obs_active": list(obs.shape),
            "target_active": list(target.shape),
            "obs_xyz": list(obs_xyz.shape),
            "target_xyz": list(target_xyz.shape),
            "finite": bool(torch.isfinite(obs_xyz).all() and torch.isfinite(target_xyz).all()),
        }
    summary["created_at"] = _utc_now()
    _write_summary(args.save_dir, "xyz_smoke_summary.json", "xyz_smoke_summary.yaml", summary)
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def _check_metrics_sanity(metrics):
    _assert_xyz_metric_keys(metrics)
    _assert_metrics_finite(metrics)
    for key, value in metrics.items():
        if abs(float(value)) > 1e-12:
            raise AssertionError("pred == target 时 {} 应为 0，实际为 {}".format(key, value))


def run_metrics_sanity(args):
    device = _device()
    converter = Rotation2xyz(device=device)
    dataset = _build_dataset(args, args.split)
    _check_dataset_length(dataset, args.max_samples)
    loader = _build_loader(args, dataset)
    obs, target, meta = next(iter(loader))
    with torch.no_grad():
        obs_xyz = active_to_xyz(obs, device=device, converter=converter)
        target_xyz = active_to_xyz(target, device=device, converter=converter)
    metrics = compute_xyz_metrics(target_xyz, target_xyz, obs_xyz)
    _check_metrics_sanity(metrics)

    summary = OrderedDict()
    summary["mode"] = "metrics_sanity"
    summary["dataset"] = args.dataset
    summary["split"] = args.split
    summary["num_samples"] = len(dataset)
    summary["checked_batch_size"] = int(obs.shape[0])
    summary["metrics_keys"] = list(XYZ_METRIC_KEYS)
    summary["metrics"] = metrics
    summary["created_at"] = _utc_now()
    _write_summary(args.save_dir, "metrics_sanity.json", "metrics_sanity.yaml", summary)
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def evaluate_xyz_model(args, model, checkpoint_state=None, device=None):
    if device is None:
        device = _device()
    dataset = _build_dataset(args, args.split)
    _check_dataset_length(dataset, args.max_samples)
    loader = _build_loader(args, dataset)
    converter = Rotation2xyz(device=device)

    metric_sums = OrderedDict((key, 0.0) for key in XYZ_METRIC_KEYS)
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for obs, target, meta in loader:
            obs_xyz = active_to_xyz(obs, device=device, converter=converter)
            target_xyz = active_to_xyz(target, device=device, converter=converter)
            pred_xyz = model(obs_xyz)
            ensure_xyz_prediction_shape(pred_xyz, int(obs.shape[0]), args.pred_len)
            metrics = compute_xyz_metrics(pred_xyz, target_xyz, obs_xyz)
            batch_size = int(obs.shape[0])
            _aggregate(metric_sums, metrics, batch_size)
            num_samples += batch_size

    if num_samples != len(dataset):
        raise AssertionError("xyz 评估样本数应为 {}，实际为 {}".format(len(dataset), num_samples))
    final_metrics = OrderedDict()
    for key in XYZ_METRIC_KEYS:
        final_metrics[key] = metric_sums[key] / float(num_samples)
    _assert_metrics_finite(final_metrics)

    summary = OrderedDict()
    summary["mode"] = "checkpoint"
    summary["dataset"] = args.dataset
    summary["split"] = args.split
    summary["data_path"] = args.data_path
    summary["window_len"] = args.window_len
    summary["obs_len"] = args.obs_len
    summary["pred_len"] = args.pred_len
    summary["batch_size"] = args.batch_size
    summary["num_workers"] = args.num_workers
    summary["num_samples"] = int(num_samples)
    summary["metrics_keys"] = list(XYZ_METRIC_KEYS)
    summary["metrics"] = final_metrics
    summary["checkpoint"] = args.checkpoint
    summary["checkpoint_step"] = None if checkpoint_state is None else checkpoint_state.get("step")
    summary["created_at"] = _utc_now()

    json_name = "metrics_{}.json".format(args.split)
    yaml_name = "metrics_{}.yaml".format(args.split)
    _write_summary(args.save_dir, json_name, yaml_name, summary)
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def run_checkpoint(args):
    device = _device()
    model, state = _load_checkpoint_model(args, device)
    return evaluate_xyz_model(args, model=model, checkpoint_state=state, device=device)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("xyz_smoke", "metrics_sanity", "checkpoint"))
    parser.add_argument("--dataset", default="interhuman")
    parser.add_argument("--data_path", default="dataset/interhuman/smpl/conditioned")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--window_len", type=int, default=150)
    parser.add_argument("--obs_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_dir", default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.dataset != "interhuman":
        raise ValueError("P7.1 xyz eval 只支持 interhuman dataset")
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    if args.save_dir is None:
        args.save_dir = "save/forecasting/interhuman/p7_xyz_{}".format(args.mode)

    if args.mode == "xyz_smoke":
        run_xyz_smoke(args)
    elif args.mode == "metrics_sanity":
        run_metrics_sanity(args)
    elif args.mode == "checkpoint":
        run_checkpoint(args)
    else:
        raise ValueError("unsupported mode: {}".format(args.mode))


if __name__ == "__main__":
    main()
