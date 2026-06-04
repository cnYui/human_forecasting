import argparse
import csv
import json
import os
from collections import OrderedDict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting import InterHumanForecastDataset, forecasting_collate
from model.forecasting import (
    count_parameters,
    create_forecasting_model_from_config,
    ensure_prediction_shape,
)
from utils.fixseed import fixseed
from utils.forecasting_metrics import (
    METRIC_KEYS,
    compute_forecasting_metrics_for_sample,
    per_frame_active_mse,
    relative_orientation_error_sequence,
    root_distance_sequence,
)
from utils.forecasting_motion import PERSON_DIM, load_forecasting_normalizer, restore_active_motion


METHODS = ("repeat", "independent", "concat", "relation")


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path, value):
    with open(path, "w") as f:
        json.dump(_plain_data(value), f, indent=2, sort_keys=False, ensure_ascii=False)


def _plain_data(value):
    if isinstance(value, OrderedDict):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_data(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _build_dataset(args):
    return InterHumanForecastDataset(
        data_path=args.data_path,
        split=args.split,
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


def _repeat_last_observation(obs, pred_len):
    last = obs[:, -1:].contiguous()
    return last.expand(-1, pred_len, -1, -1).contiguous()


def _checkpoint_normalizer_path(checkpoint_path, state):
    path = state.get("normalizer_path")
    if path is not None and os.path.exists(path):
        return path
    fallback = os.path.join(os.path.dirname(checkpoint_path), "normalizer.pt")
    if os.path.exists(fallback):
        return fallback
    if path is not None:
        raise FileNotFoundError(
            "checkpoint normalizer_path 不存在，fallback 也不存在: {} / {}".format(
                path, fallback
            )
        )
    raise FileNotFoundError("checkpoint 缺少 normalizer_path，且 fallback 不存在: {}".format(fallback))


def _validate_model_config(config, expected_type, args, checkpoint_path):
    if config.get("model_type") != expected_type:
        raise ValueError(
            "{} model_type 应为 {}，实际为 {}".format(
                checkpoint_path, expected_type, config.get("model_type")
            )
        )
    expected = {
        "obs_len": args.obs_len,
        "pred_len": args.pred_len,
        "person_dim": PERSON_DIM,
    }
    for key, value in expected.items():
        if int(config.get(key)) != int(value):
            raise ValueError(
                "{} {} 应为 {}，实际为 {}".format(
                    checkpoint_path, key, value, config.get(key)
                )
            )


def _validate_normalizer(normalizer, args, checkpoint_path):
    metadata = normalizer.metadata or {}
    for key, expected in (
        ("window_len", args.window_len),
        ("obs_len", args.obs_len),
        ("pred_len", args.pred_len),
        ("person_dim", PERSON_DIM),
    ):
        if key in metadata and int(metadata[key]) != int(expected):
            raise ValueError(
                "{} normalizer {} 应为 {}，实际为 {}".format(
                    checkpoint_path, key, expected, metadata[key]
                )
            )


def _load_checkpoint_bundle(name, checkpoint_path, expected_type, args, device):
    if checkpoint_path is None:
        return None
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in state or "model_config" not in state:
        raise ValueError("{} 缺少 model_state_dict 或 model_config".format(checkpoint_path))
    config = dict(state["model_config"])
    _validate_model_config(config, expected_type, args, checkpoint_path)

    model = create_forecasting_model_from_config(config)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    normalizer_path = _checkpoint_normalizer_path(checkpoint_path, state)
    normalizer = load_forecasting_normalizer(normalizer_path)
    _validate_normalizer(normalizer, args, checkpoint_path)

    return {
        "name": name,
        "checkpoint": checkpoint_path,
        "checkpoint_step": state.get("step"),
        "config": config,
        "num_params": int(state.get("num_params", count_parameters(model))),
        "normalizer_path": normalizer_path,
        "normalizer": normalizer,
        "model": model,
    }


def _load_model_bundles(args, device):
    bundles = OrderedDict()
    bundles["independent"] = _load_checkpoint_bundle(
        "independent", args.independent_checkpoint, "independent", args, device
    )
    bundles["concat"] = _load_checkpoint_bundle(
        "concat", args.concat_checkpoint, "concat", args, device
    )
    bundles["relation"] = _load_checkpoint_bundle(
        "relation", args.relation_checkpoint, "relation", args, device
    )
    if args.param_matched_concat_checkpoint is not None:
        bundles["concat_h259"] = _load_checkpoint_bundle(
            "concat_h259",
            args.param_matched_concat_checkpoint,
            "concat",
            args,
            device,
        )
    return bundles


def _predict_with_bundle(bundle, obs, args, device):
    obs_device = obs.to(device)
    with torch.no_grad():
        pred_normalized = bundle["model"](bundle["normalizer"].normalize(obs_device))
        ensure_prediction_shape(
            pred_normalized,
            batch_size=int(obs.shape[0]),
            pred_len=args.pred_len,
        )
        pred = bundle["normalizer"].denormalize(pred_normalized).detach().cpu()
    if not torch.isfinite(pred).all():
        raise ValueError("{} prediction 存在非有限数值".format(bundle["name"]))
    return pred


def _sample_metrics(pred, target, obs):
    metrics = compute_forecasting_metrics_for_sample(pred, target, obs)
    return OrderedDict((key, float(metrics[key])) for key in METRIC_KEYS)


def _flatten_metrics(prefix, metrics, row):
    for key in METRIC_KEYS:
        row["{}_{}".format(prefix, key)] = metrics[key]


def _record_to_csv_row(record):
    row = OrderedDict()
    for key in ("sample_id", "length", "start", "delta_long", "delta_root_dist"):
        row[key] = record.get(key)
    for method, metrics in record["metrics"].items():
        _flatten_metrics(method, metrics, row)
    return row


def _selection_to_csv_row(item):
    row = OrderedDict()
    for key in (
        "sample_id",
        "length",
        "start",
        "category",
        "selection_reason",
        "delta_long",
        "delta_root_dist",
    ):
        row[key] = item.get(key)
    for method, metrics in item["metrics"].items():
        _flatten_metrics(method, metrics, row)
    return row


def _all_metrics_fieldnames(records):
    fields = ["sample_id", "length", "start", "delta_long", "delta_root_dist"]
    if not records:
        return fields
    for method in records[0]["metrics"].keys():
        fields.extend(["{}_{}".format(method, key) for key in METRIC_KEYS])
    return fields


def _selection_fieldnames(records):
    fields = [
        "sample_id",
        "length",
        "start",
        "category",
        "selection_reason",
        "delta_long",
        "delta_root_dist",
    ]
    if not records:
        return fields
    for method in records[0]["metrics"].keys():
        fields.extend(["{}_{}".format(method, key) for key in METRIC_KEYS])
    return fields


def _run_inference(args, dataset, loader, bundles, device):
    records = []
    arrays = OrderedDict()

    for obs, target, meta in loader:
        predictions = OrderedDict()
        predictions["repeat"] = _repeat_last_observation(obs, args.pred_len)
        for name, bundle in bundles.items():
            predictions[name] = _predict_with_bundle(bundle, obs, args, device)

        for index, item_meta in enumerate(meta):
            sample_id = item_meta["sample_id"]
            obs_i = obs[index].contiguous()
            target_i = target[index].contiguous()
            sample_predictions = OrderedDict(
                (name, pred[index].contiguous()) for name, pred in predictions.items()
            )

            metrics = OrderedDict()
            for name, pred_i in sample_predictions.items():
                metrics[name] = _sample_metrics(pred_i, target_i, obs_i)
            if "concat" not in metrics or "relation" not in metrics:
                raise ValueError("P6 selection 需要 concat 和 relation metrics")

            delta_long = metrics["concat"]["long_mse"] - metrics["relation"]["long_mse"]
            delta_root_dist = (
                metrics["concat"]["relative_root_distance_error"]
                - metrics["relation"]["relative_root_distance_error"]
            )

            record = OrderedDict()
            record["sample_id"] = sample_id
            record["length"] = int(item_meta["length"])
            record["start"] = int(item_meta["start"])
            record["delta_long"] = float(delta_long)
            record["delta_root_dist"] = float(delta_root_dist)
            record["metrics"] = metrics
            records.append(record)

            arrays[sample_id] = {
                "meta": dict(item_meta),
                "obs": obs_i.numpy(),
                "target": target_i.numpy(),
                "predictions": OrderedDict(
                    (name, pred_i.numpy()) for name, pred_i in sample_predictions.items()
                ),
            }

    if len(records) != len(dataset):
        raise AssertionError("推理样本数应为 {}，实际为 {}".format(len(dataset), len(records)))
    return records, arrays


def _pick_from_candidates(selected, selected_ids, category, candidates, limit, reason):
    for record in candidates:
        if len(selected) >= limit:
            break
        sample_id = record["sample_id"]
        if sample_id in selected_ids:
            continue
        item = OrderedDict(record)
        item["category"] = category
        item["selection_reason"] = reason(record)
        selected.append(item)
        selected_ids.add(sample_id)


def _select_samples(records, num_samples):
    if num_samples < 8:
        raise ValueError("P6 至少需要 8 个 qualitative samples")

    selected = []
    selected_ids = set()

    success = sorted(
        [
            record
            for record in records
            if record["delta_long"] > 0.0 and record["delta_root_dist"] > 0.0
        ],
        key=lambda record: (-record["delta_long"], -record["delta_root_dist"]),
    )
    _pick_from_candidates(
        selected,
        selected_ids,
        "success",
        success,
        2,
        lambda record: "delta_long 和 delta_root_dist 同时为正，且 delta_long 排名靠前",
    )

    close = sorted(records, key=lambda record: abs(record["delta_long"]))
    _pick_from_candidates(
        selected,
        selected_ids,
        "close",
        close,
        4,
        lambda record: "abs(delta_long) 最小的未选样本",
    )

    failure = sorted(
        [
            record
            for record in records
            if record["delta_long"] < 0.0 or record["delta_root_dist"] < 0.0
        ],
        key=lambda record: (record["delta_long"], record["delta_root_dist"]),
    )
    _pick_from_candidates(
        selected,
        selected_ids,
        "failure",
        failure,
        6,
        lambda record: "delta_long < 0 或 delta_root_dist < 0，优先 delta_long 最负",
    )

    short_boundary = sorted(records, key=lambda record: (abs(record["length"] - 150), record["length"]))
    _pick_from_candidates(
        selected,
        selected_ids,
        "boundary",
        short_boundary,
        7,
        lambda record: "length 最接近 150 的短序列边界样本",
    )

    long_boundary = sorted(records, key=lambda record: -record["length"])
    _pick_from_candidates(
        selected,
        selected_ids,
        "boundary",
        long_boundary,
        8,
        lambda record: "test split 中 length 最大的长序列样本",
    )

    if len(selected) < num_samples:
        remaining = sorted(
            [record for record in records if record["sample_id"] not in selected_ids],
            key=lambda record: -(abs(record["delta_long"]) + abs(record["delta_root_dist"])),
        )
        _pick_from_candidates(
            selected,
            selected_ids,
            "fill",
            remaining,
            num_samples,
            lambda record: "类别样本不足或 num_samples 超过 8，按 delta 多样性补齐",
        )

    categories = set(item["category"] for item in selected)
    required = set(["success", "close", "failure", "boundary"])
    missing = sorted(required - categories)
    return selected, missing


def _full_future(obs, future):
    return np.concatenate([obs, future], axis=0)


def _plot_distance_curve(path, obs, target, predictions, args):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(args.window_len)
    gt_full = _full_future(obs, target)
    gt_dist = root_distance_sequence(torch.from_numpy(gt_full).unsqueeze(0))[0].numpy()
    obs_dist = root_distance_sequence(torch.from_numpy(obs).unsqueeze(0))[0].numpy()

    ax.plot(np.arange(args.obs_len), obs_dist, label="obs", linewidth=2.0, color="black")
    ax.plot(x, gt_dist, label="gt", linewidth=2.0)
    for name, pred in predictions.items():
        pred_full = _full_future(obs, pred)
        dist = root_distance_sequence(torch.from_numpy(pred_full).unsqueeze(0))[0].numpy()
        ax.plot(x, dist, label=name, linewidth=1.4)
    ax.axvline(args.obs_len - 0.5, linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel("root distance")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_orientation_curve(path, target, predictions, args):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(args.obs_len, args.window_len)
    target_tensor = torch.from_numpy(target).unsqueeze(0)
    for name, pred in predictions.items():
        pred_tensor = torch.from_numpy(pred).unsqueeze(0)
        error = relative_orientation_error_sequence(pred_tensor, target_tensor)[0].numpy()
        ax.plot(x, error, label=name, linewidth=1.4)
    ax.set_xlabel("frame")
    ax.set_ylabel("relative orientation error")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_long_mse_curve(path, target, predictions, args):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(args.obs_len, args.window_len)
    target_tensor = torch.from_numpy(target).unsqueeze(0)
    for name, pred in predictions.items():
        pred_tensor = torch.from_numpy(pred).unsqueeze(0)
        frame_mse = per_frame_active_mse(pred_tensor, target_tensor)[0].numpy()
        ax.plot(x, frame_mse, label=name, linewidth=1.4)
    ax.axvspan(args.obs_len + 80, args.window_len - 1, color="gray", alpha=0.12)
    ax.set_xlabel("frame")
    ax.set_ylabel("per-frame active MSE")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_h5_like(path, active):
    motion = restore_active_motion(torch.from_numpy(active)).numpy()
    np.save(path, motion)


def _save_selected_samples(args, selected, arrays):
    qualitative_dir = os.path.join(args.save_dir, "qualitative")
    os.makedirs(qualitative_dir, exist_ok=True)

    for item in selected:
        sample_id = item["sample_id"]
        sample = arrays[sample_id]
        sample_dir = os.path.join(qualitative_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        obs = sample["obs"]
        target = sample["target"]
        predictions = sample["predictions"]

        np.save(os.path.join(sample_dir, "obs.npy"), obs)
        np.save(os.path.join(sample_dir, "gt.npy"), target)
        for name, pred in predictions.items():
            file_name = "pred_{}.npy".format(name)
            np.save(os.path.join(sample_dir, file_name), pred)

        _save_h5_like(os.path.join(sample_dir, "obs_h5_like.npy"), obs)
        _save_h5_like(os.path.join(sample_dir, "gt_h5_like.npy"), target)
        if "relation" in predictions:
            _save_h5_like(
                os.path.join(sample_dir, "pred_relation_h5_like.npy"),
                predictions["relation"],
            )

        meta = OrderedDict()
        meta["sample_id"] = sample_id
        meta["length"] = int(item["length"])
        meta["start"] = int(item["start"])
        meta["category"] = item["category"]
        meta["selection_reason"] = item["selection_reason"]
        meta["delta_long"] = float(item["delta_long"])
        meta["delta_root_dist"] = float(item["delta_root_dist"])
        meta["obs_shape"] = list(obs.shape)
        meta["target_shape"] = list(target.shape)
        meta["prediction_shapes"] = {
            name: list(pred.shape) for name, pred in predictions.items()
        }
        _write_json(os.path.join(sample_dir, "meta.json"), meta)
        _write_json(os.path.join(sample_dir, "metrics_per_sample.json"), item["metrics"])

        _plot_distance_curve(
            os.path.join(sample_dir, "distance_curve.png"),
            obs,
            target,
            predictions,
            args,
        )
        _plot_orientation_curve(
            os.path.join(sample_dir, "orientation_curve.png"),
            target,
            predictions,
            args,
        )
        _plot_long_mse_curve(
            os.path.join(sample_dir, "long_mse_curve.png"),
            target,
            predictions,
            args,
        )


def _write_summary_md(path, args, selected, missing_categories):
    with open(path, "w") as f:
        f.write("# Forecasting P6 Qualitative Summary\n\n")
        f.write("本 summary 只解释样本行为，不替代 P5 全 test aggregate 指标。\n\n")
        f.write("## Protocol\n\n")
        f.write("- dataset: {}\n".format(args.dataset))
        f.write("- split: {}\n".format(args.split))
        f.write("- window/obs/pred: {}/{}/{}\n".format(args.window_len, args.obs_len, args.pred_len))
        f.write("- num selected samples: {}\n\n".format(len(selected)))
        if missing_categories:
            f.write("## Missing Categories\n\n")
            f.write("{}\n\n".format(", ".join(missing_categories)))
        f.write("## Selected Samples\n\n")
        f.write("| category | sample_id | length | start | delta_long | delta_root_dist |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for item in selected:
            f.write(
                "| {} | {} | {} | {} | {:.10g} | {:.10g} |\n".format(
                    item["category"],
                    item["sample_id"],
                    item["length"],
                    item["start"],
                    item["delta_long"],
                    item["delta_root_dist"],
                )
            )


def _bundle_summary(bundles):
    summary = OrderedDict()
    for name, bundle in bundles.items():
        if bundle is None:
            continue
        summary[name] = {
            "checkpoint": bundle["checkpoint"],
            "checkpoint_step": bundle["checkpoint_step"],
            "num_params": bundle["num_params"],
            "normalizer_path": bundle["normalizer_path"],
            "config": bundle["config"],
        }
    return summary


def run_visualization(args):
    if args.dataset != "interhuman":
        raise ValueError("P6 只支持 interhuman dataset")
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    if args.num_samples < 8:
        raise ValueError("P6 num_samples 至少为 8")

    fixseed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = _device()

    dataset = _build_dataset(args)
    loader = _build_loader(args, dataset)
    bundles = _load_model_bundles(args, device)
    records, arrays = _run_inference(args, dataset, loader, bundles, device)
    selected, missing_categories = _select_samples(records, args.num_samples)

    run_config = OrderedDict()
    run_config["mode"] = "p6_qualitative"
    run_config["dataset"] = args.dataset
    run_config["data_path"] = args.data_path
    run_config["split"] = args.split
    run_config["window_len"] = args.window_len
    run_config["obs_len"] = args.obs_len
    run_config["pred_len"] = args.pred_len
    run_config["batch_size"] = args.batch_size
    run_config["num_workers"] = args.num_workers
    run_config["max_samples"] = args.max_samples
    run_config["seed"] = args.seed
    run_config["device"] = str(device)
    run_config["num_dataset_samples"] = len(dataset)
    run_config["num_selected_samples"] = len(selected)
    run_config["missing_selection_categories"] = missing_categories
    run_config["checkpoints"] = _bundle_summary(bundles)
    run_config["created_at"] = _utc_now()
    _write_json(os.path.join(args.save_dir, "run_config.json"), run_config)

    _write_json(os.path.join(args.save_dir, "metrics_per_sample_all.json"), records)
    _write_csv(
        os.path.join(args.save_dir, "metrics_per_sample_all.csv"),
        [_record_to_csv_row(record) for record in records],
        _all_metrics_fieldnames(records),
    )

    _write_json(os.path.join(args.save_dir, "selection.json"), selected)
    _write_csv(
        os.path.join(args.save_dir, "selection.csv"),
        [_selection_to_csv_row(item) for item in selected],
        _selection_fieldnames(selected),
    )

    _save_selected_samples(args, selected, arrays)
    _write_summary_md(
        os.path.join(args.save_dir, "summary.md"),
        args,
        selected,
        missing_categories,
    )

    print(json.dumps(_plain_data(run_config), indent=2, sort_keys=False, ensure_ascii=False))
    return {
        "run_config": run_config,
        "records": records,
        "selected": selected,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="interhuman")
    parser.add_argument("--data_path", default="dataset/interhuman/smpl/conditioned")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--window_len", type=int, default=150)
    parser.add_argument("--obs_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=120)
    parser.add_argument("--independent_checkpoint", required=True)
    parser.add_argument("--concat_checkpoint", required=True)
    parser.add_argument("--relation_checkpoint", required=True)
    parser.add_argument("--param_matched_concat_checkpoint", default=None)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
