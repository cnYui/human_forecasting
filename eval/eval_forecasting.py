import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import h5py
import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting import InterHumanForecastDataset, forecasting_collate
from utils.forecasting_metrics import METRIC_KEYS, compute_forecasting_metrics
from utils.forecasting_motion import (
    PERSON_DIM,
    compute_forecasting_normalizer,
    extract_active_motion,
    load_forecasting_normalizer,
    restore_active_motion,
)


EXPECTED_LENGTHS = {
    "train": 2910,
    "val": 226,
    "test": 508,
}


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError("{} 应为 {}，实际为 {}".format(name, expected, actual))


def _assert_shape(name, tensor, expected):
    actual = tuple(tensor.shape)
    if actual != tuple(expected):
        raise AssertionError("{} shape 应为 {}，实际为 {}".format(name, expected, actual))


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


def _check_dataset_lengths(datasets, max_samples):
    summary = {}
    for split, dataset in datasets.items():
        expected = EXPECTED_LENGTHS[split]
        if max_samples is not None and max_samples > 0:
            expected = min(expected, max_samples)
        _assert_equal("{} dataset length".format(split), len(dataset), expected)
        summary[split] = len(dataset)
    return summary


def _assert_metric_keys(metrics):
    if tuple(metrics.keys()) != METRIC_KEYS:
        raise AssertionError("metrics key 应为 {}，实际为 {}".format(METRIC_KEYS, tuple(metrics.keys())))


def _assert_metrics_finite(metrics):
    for key, value in metrics.items():
        if not torch.isfinite(torch.tensor(float(value))):
            raise AssertionError("{} 指标为非有限数值: {}".format(key, value))


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


def _default_save_dir(mode):
    defaults = {
        "dataset_smoke": "save/forecasting/interhuman/p1_dataset_smoke",
        "metrics_sanity": "save/forecasting/interhuman/p2_metrics_sanity",
        "repeat": "save/forecasting/interhuman/repeat_150_30_120",
        "checkpoint": "save/forecasting/interhuman/checkpoint_eval",
    }
    return defaults[mode]


def _repeat_last_observation(obs, pred_len):
    if obs.dim() != 4:
        raise ValueError("obs 必须是 [B,T,2,147]")
    last = obs[:, -1:].contiguous()
    return last.expand(-1, pred_len, -1, -1).contiguous()


def _check_item_shapes_and_finite(datasets, obs_len, pred_len):
    for split, dataset in datasets.items():
        item = dataset[0]
        _assert_shape("{} obs".format(split), item["obs"], (obs_len, 2, PERSON_DIM))
        _assert_shape("{} target".format(split), item["target"], (pred_len, 2, PERSON_DIM))
        if not torch.isfinite(item["obs"]).all():
            raise AssertionError("{} obs 存在非有限数值".format(split))
        if not torch.isfinite(item["target"]).all():
            raise AssertionError("{} target 存在非有限数值".format(split))


def _scan_all_finite(dataset):
    with h5py.File(str(dataset.h5_path), "r") as h5:
        for entry in dataset.entries:
            sample_id = entry["sample_id"]
            motion = torch.from_numpy(h5[sample_id][:]).float()
            active = extract_active_motion(motion)
            if not torch.isfinite(active).all():
                raise AssertionError(
                    "{} split sample_id={} 存在非有限数值".format(dataset.split, sample_id)
                )


def _check_all_finite(datasets):
    for dataset in datasets.values():
        _scan_all_finite(dataset)


def _find_variable_start_index(dataset):
    for index, entry in enumerate(dataset.entries):
        if int(entry["length"]) > dataset.window_len:
            return index
    return None


def _check_start_rules(train_dataset, val_dataset, test_dataset):
    train_index = _find_variable_start_index(train_dataset)
    if train_index is None:
        raise AssertionError("train split 没有 length > window_len 的样本，无法检查随机 start")

    train_starts = [train_dataset[train_index]["start"] for _ in range(64)]
    if len(set(train_starts)) <= 1:
        raise AssertionError("train 同一样本多次读取 start 未变化")

    fixed_summary = {}
    for dataset in (val_dataset, test_dataset):
        item_starts = [dataset[0]["start"] for _ in range(8)]
        if len(set(item_starts)) != 1:
            raise AssertionError("{} 同一样本多次读取 start 不固定".format(dataset.split))
        expected = (int(dataset.entries[0]["length"]) - dataset.window_len) // 2
        _assert_equal("{} center start".format(dataset.split), item_starts[0], expected)
        fixed_summary[dataset.split] = item_starts[0]

    return {
        "train_checked_index": int(train_index),
        "train_unique_starts": sorted([int(value) for value in set(train_starts)]),
        "fixed_eval_starts": fixed_summary,
    }


def _check_batch(dataset, batch_size, num_workers, obs_len, pred_len):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=forecasting_collate,
    )
    obs, target, meta = next(iter(loader))
    _assert_shape("batch obs", obs, (min(batch_size, len(dataset)), obs_len, 2, PERSON_DIM))
    _assert_shape("batch target", target, (min(batch_size, len(dataset)), pred_len, 2, PERSON_DIM))
    if not isinstance(meta, list) or len(meta) != obs.shape[0]:
        raise AssertionError("batch meta 长度不正确")
    if not torch.isfinite(obs).all() or not torch.isfinite(target).all():
        raise AssertionError("batch obs/target 存在非有限数值")
    return {
        "obs_shape": list(obs.shape),
        "target_shape": list(target.shape),
        "meta_size": len(meta),
    }


def _check_active_roundtrip(dataset):
    item = dataset[0]
    active = torch.cat([item["obs"], item["target"]], dim=0)
    restored = restore_active_motion(active)
    recovered = extract_active_motion(restored)
    max_abs_error = float((active - recovered).abs().max().item())
    if max_abs_error > 1e-5:
        raise AssertionError("active roundtrip 误差过大: {}".format(max_abs_error))
    return max_abs_error


def _check_normalizer(args, dataset):
    normalizer = compute_forecasting_normalizer(
        data_path=args.data_path,
        save_dir=args.save_dir,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
    )
    loaded = load_forecasting_normalizer(args.save_dir)

    item = dataset[0]
    active = torch.cat([item["obs"], item["target"]], dim=0)
    restored = loaded.denormalize(loaded.normalize(active))
    max_abs_error = float((active - restored).abs().max().item())
    if max_abs_error > 1e-5:
        raise AssertionError("normalize -> denormalize 误差过大: {}".format(max_abs_error))

    pt_path = os.path.join(args.save_dir, "normalizer.pt")
    json_path = os.path.join(args.save_dir, "normalizer.json")
    if not os.path.exists(pt_path) or not os.path.exists(json_path):
        raise AssertionError("normalizer.pt 或 normalizer.json 未生成")

    return {
        "roundtrip_max_abs_error": max_abs_error,
        "metadata": normalizer.metadata,
    }


def run_dataset_smoke(args):
    if args.dataset != "interhuman":
        raise ValueError("P1 只支持 interhuman dataset")

    os.makedirs(args.save_dir, exist_ok=True)
    datasets = {
        "train": _build_dataset(args, "train"),
        "val": _build_dataset(args, "val"),
        "test": _build_dataset(args, "test"),
    }

    summary = {
        "mode": "dataset_smoke",
        "dataset": args.dataset,
        "data_path": args.data_path,
        "window_len": args.window_len,
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "dataset_lengths": _check_dataset_lengths(datasets, args.max_samples),
    }
    _check_item_shapes_and_finite(datasets, args.obs_len, args.pred_len)
    _check_all_finite(datasets)
    summary["start_rules"] = _check_start_rules(
        datasets["train"], datasets["val"], datasets["test"]
    )
    summary["batch"] = _check_batch(
        datasets["train"], args.batch_size, args.num_workers, args.obs_len, args.pred_len
    )
    summary["active_roundtrip_max_abs_error"] = _check_active_roundtrip(datasets["train"])
    summary["normalizer"] = _check_normalizer(args, datasets["train"])

    summary_path = os.path.join(args.save_dir, "dataset_smoke_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _check_split_dataset_length(dataset, max_samples):
    expected = EXPECTED_LENGTHS[dataset.split]
    if max_samples is not None and max_samples > 0:
        expected = min(expected, max_samples)
    _assert_equal("{} dataset length".format(dataset.split), len(dataset), expected)


def _check_metrics_sanity(metrics):
    _assert_metric_keys(metrics)
    _assert_metrics_finite(metrics)

    zero_keys = (
        "future_mse",
        "rotation_mse",
        "translation_mse",
        "short_mse",
        "mid_mse",
        "long_mse",
        "relative_root_distance_error",
        "inter_person_distance_consistency",
    )
    for key in zero_keys:
        if abs(metrics[key]) > 1e-12:
            raise AssertionError("pred == target 时 {} 应为 0，实际为 {}".format(key, metrics[key]))
    if metrics["relative_orientation_error"] > 1e-4:
        raise AssertionError(
            "pred == target 时 relative_orientation_error 应接近 0，实际为 {}".format(
                metrics["relative_orientation_error"]
            )
        )


def run_metrics_sanity(args):
    if args.dataset != "interhuman":
        raise ValueError("P2 只支持 interhuman dataset")

    dataset = _build_dataset(args, args.split)
    _check_split_dataset_length(dataset, args.max_samples)
    loader = _build_loader(args, dataset)

    obs, target, meta = next(iter(loader))
    pred = target.clone()
    metrics = compute_forecasting_metrics(pred, target, obs)
    _check_metrics_sanity(metrics)

    summary = OrderedDict()
    summary["mode"] = "metrics_sanity"
    summary["dataset"] = args.dataset
    summary["split"] = args.split
    summary["data_path"] = args.data_path
    summary["window_len"] = args.window_len
    summary["obs_len"] = args.obs_len
    summary["pred_len"] = args.pred_len
    summary["batch_size"] = args.batch_size
    summary["num_workers"] = args.num_workers
    summary["num_samples"] = len(dataset)
    summary["checked_batch_size"] = int(obs.shape[0])
    summary["metrics_keys"] = list(METRIC_KEYS)
    summary["metrics"] = metrics
    summary["created_at"] = _utc_now()
    summary["output_files"] = {
        "json": os.path.join(args.save_dir, "metrics_sanity.json"),
        "yaml": os.path.join(args.save_dir, "metrics_sanity.yaml"),
    }

    _write_summary(
        args.save_dir, "metrics_sanity.json", "metrics_sanity.yaml", summary
    )
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def _aggregate_metrics(metric_sums, metrics, batch_size):
    _assert_metric_keys(metrics)
    _assert_metrics_finite(metrics)
    for key in METRIC_KEYS:
        metric_sums[key] += metrics[key] * float(batch_size)


def run_repeat(args):
    if args.dataset != "interhuman":
        raise ValueError("P2 只支持 interhuman dataset")

    dataset = _build_dataset(args, args.split)
    _check_split_dataset_length(dataset, args.max_samples)
    loader = _build_loader(args, dataset)

    metric_sums = OrderedDict((key, 0.0) for key in METRIC_KEYS)
    num_samples = 0

    with torch.no_grad():
        for obs, target, meta in loader:
            pred = _repeat_last_observation(obs, args.pred_len)
            metrics = compute_forecasting_metrics(pred, target, obs)
            batch_size = int(obs.shape[0])
            _aggregate_metrics(metric_sums, metrics, batch_size)
            num_samples += batch_size

    if num_samples != len(dataset):
        raise AssertionError("repeat 评估样本数应为 {}，实际为 {}".format(len(dataset), num_samples))

    final_metrics = OrderedDict()
    for key in METRIC_KEYS:
        final_metrics[key] = metric_sums[key] / float(num_samples)
    _assert_metrics_finite(final_metrics)

    summary = OrderedDict()
    summary["mode"] = "repeat"
    summary["dataset"] = args.dataset
    summary["split"] = args.split
    summary["data_path"] = args.data_path
    summary["window_len"] = args.window_len
    summary["obs_len"] = args.obs_len
    summary["pred_len"] = args.pred_len
    summary["batch_size"] = args.batch_size
    summary["num_workers"] = args.num_workers
    summary["num_samples"] = int(num_samples)
    summary["metrics_keys"] = list(METRIC_KEYS)
    summary["metrics"] = final_metrics
    summary["created_at"] = _utc_now()

    json_name = "metrics_{}.json".format(args.split)
    yaml_name = "metrics_{}.yaml".format(args.split)
    summary["output_files"] = {
        "json": os.path.join(args.save_dir, json_name),
        "yaml": os.path.join(args.save_dir, yaml_name),
    }
    _write_summary(args.save_dir, json_name, yaml_name, summary)
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dataset_smoke", "metrics_sanity", "repeat", "checkpoint"],
    )
    parser.add_argument("--dataset", default="interhuman")
    parser.add_argument("--data_path", default="dataset/interhuman/smpl/conditioned")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--window_len", type=int, default=150)
    parser.add_argument("--obs_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_dir",
        default=None,
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.save_dir is None:
        args.save_dir = _default_save_dir(args.mode)

    if args.mode == "dataset_smoke":
        run_dataset_smoke(args)
        return
    if args.mode == "metrics_sanity":
        run_metrics_sanity(args)
        return
    if args.mode == "repeat":
        run_repeat(args)
        return
    if args.mode == "checkpoint":
        raise NotImplementedError("checkpoint evaluation 将在 P3/P4 接入")
    raise ValueError("unsupported mode: {}".format(args.mode))


if __name__ == "__main__":
    main()
