import argparse
import csv
import json
import os
from collections import OrderedDict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting import InterHumanForecastDataset, forecasting_collate
from model.forecasting_somoformer import ensure_xyz_prediction_shape
from model.forecasting_xyz import create_xyz_forecasting_model_from_config
from model.rotation2xyz import Rotation2xyz
from utils.fixseed import fixseed
from utils.forecasting_xyz import XYZ_METRIC_KEYS, active_to_xyz, compute_xyz_metrics


SMPL24_CHAINS = (
    (0, 1, 4, 7, 10),
    (0, 2, 5, 8, 11),
    (0, 3, 6, 9, 12, 15),
    (12, 13, 16, 18, 20, 22),
    (12, 14, 17, 19, 21, 23),
)

COLORS = OrderedDict(
    (
        ("observed", "#2F6BFF"),
        ("ground_truth", "#2CA02C"),
        ("prediction", "#FF7F0E"),
    )
)


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device(force_cpu):
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _write_json(path, value):
    with open(path, "w") as f:
        json.dump(_plain_data(value), f, indent=2, sort_keys=False, ensure_ascii=False)


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


def _load_checkpoint_model(checkpoint, device):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location=device)
    if "model_state_dict" not in state:
        raise ValueError("checkpoint 缺少 model_state_dict")
    if "model_config" not in state:
        raise ValueError("checkpoint 缺少 model_config")
    config = dict(state["model_config"])
    model = create_xyz_forecasting_model_from_config(config)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


def _repeat_last_obs_xyz(obs_xyz, pred_len):
    return obs_xyz[:, -1:].expand(-1, pred_len, -1, -1, -1).contiguous()


def _metrics_for_sample(pred_xyz, target_xyz, obs_xyz):
    metrics = compute_xyz_metrics(
        pred_xyz.unsqueeze(0),
        target_xyz.unsqueeze(0),
        obs_xyz.unsqueeze(0),
    )
    return OrderedDict((key, float(metrics[key])) for key in XYZ_METRIC_KEYS)


def _run_inference(args, model, device):
    dataset = _build_dataset(args)
    loader = _build_loader(args, dataset)
    converter = Rotation2xyz(device=device)

    records = []
    arrays = OrderedDict()
    model.eval()
    sample_offset = 0
    with torch.no_grad():
        for obs, target, meta in loader:
            obs_xyz = active_to_xyz(obs, device=device, converter=converter)
            target_xyz = active_to_xyz(target, device=device, converter=converter)
            pred_xyz = model(obs_xyz)
            ensure_xyz_prediction_shape(pred_xyz, int(obs.shape[0]), args.pred_len)
            repeat_xyz = _repeat_last_obs_xyz(obs_xyz, args.pred_len)

            obs_cpu = obs_xyz.detach().cpu()
            target_cpu = target_xyz.detach().cpu()
            pred_cpu = pred_xyz.detach().cpu()
            repeat_cpu = repeat_xyz.detach().cpu()

            batch_size = int(obs.shape[0])
            for local_i in range(batch_size):
                sample_id = sample_offset + local_i
                pred_metrics = _metrics_for_sample(
                    pred_cpu[local_i],
                    target_cpu[local_i],
                    obs_cpu[local_i],
                )
                repeat_metrics = _metrics_for_sample(
                    repeat_cpu[local_i],
                    target_cpu[local_i],
                    obs_cpu[local_i],
                )
                delta_long = repeat_metrics["long_joint_mse"] - pred_metrics["long_joint_mse"]
                delta_root = (
                    repeat_metrics["relative_root_distance_error"]
                    - pred_metrics["relative_root_distance_error"]
                )
                record = OrderedDict()
                record["sample_id"] = int(sample_id)
                record["length"] = _safe_meta_value(meta, local_i, "length")
                record["start"] = _safe_meta_value(meta, local_i, "start")
                record["delta_long"] = float(delta_long)
                record["delta_root_dist"] = float(delta_root)
                record["score"] = float(delta_long + delta_root)
                record["metrics"] = OrderedDict(
                    (
                        ("official_somoformer_xyz", pred_metrics),
                        ("repeat_xyz", repeat_metrics),
                    )
                )
                records.append(record)
                arrays[int(sample_id)] = {
                    "obs_xyz": obs_cpu[local_i].numpy(),
                    "gt_xyz": target_cpu[local_i].numpy(),
                    "pred_xyz": pred_cpu[local_i].numpy(),
                    "repeat_xyz": repeat_cpu[local_i].numpy(),
                }
            sample_offset += batch_size
    return records, arrays, len(dataset)


def _safe_meta_value(meta, index, key):
    if meta is None or key not in meta:
        return None
    value = meta[key]
    if torch.is_tensor(value):
        return int(value[index].detach().cpu().item())
    if isinstance(value, np.ndarray):
        return int(value[index].item())
    if isinstance(value, (list, tuple)):
        item = value[index]
        if isinstance(item, np.generic):
            return item.item()
        return item
    return value


def _select_samples(records, num_samples):
    if num_samples < 4:
        raise ValueError("--num_samples 至少为 4")
    per_category = max(1, num_samples // 4)
    selected = []
    used = set()

    def add(category, reason, candidates):
        nonlocal selected
        count = 0
        for record in candidates:
            if record["sample_id"] in used:
                continue
            item = OrderedDict(record)
            item["category"] = category
            item["selection_reason"] = reason
            selected.append(item)
            used.add(record["sample_id"])
            count += 1
            if count >= per_category or len(selected) >= num_samples:
                return

    success = sorted(records, key=lambda item: item["score"], reverse=True)
    failure = sorted(records, key=lambda item: item["score"])
    close = sorted(records, key=lambda item: abs(item["delta_long"]))
    boundary = sorted(
        records,
        key=lambda item: (item["delta_long"] < 0 <= item["delta_root_dist"], abs(item["delta_root_dist"])),
        reverse=True,
    )

    add("success", "largest repeat-vs-official improvement", success)
    add("close", "smallest absolute long-horizon delta", close)
    add("failure", "largest repeat-vs-official degradation", failure)
    add("boundary", "long-horizon and relation deltas disagree", boundary)

    if len(selected) < num_samples:
        add("fallback", "fill remaining slots by sample_id", sorted(records, key=lambda item: item["sample_id"]))
    return selected[:num_samples]


def _metric_row(record):
    row = OrderedDict()
    for key in ("sample_id", "length", "start", "category", "selection_reason", "delta_long", "delta_root_dist", "score"):
        row[key] = record.get(key)
    for method, metrics in record["metrics"].items():
        for metric_key, value in metrics.items():
            row["{}_{}".format(method, metric_key)] = value
    return row


def _all_fieldnames(rows):
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    return keys


def _axis_limits(*motions):
    data = np.concatenate([motion.reshape(-1, 3) for motion in motions], axis=0)
    finite = data[np.isfinite(data).all(axis=1)]
    if finite.size == 0:
        raise ValueError("无法从非有限坐标计算坐标轴范围")
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if span < 1e-6:
        span = 1.0
    radius = span * 0.58
    return center, radius


def _draw_pose(ax, pose, color, linestyle, linewidth):
    for person_idx in range(pose.shape[0]):
        person_pose = pose[person_idx]
        style = linestyle if person_idx == 0 else "--"
        for chain in SMPL24_CHAINS:
            points = person_pose[list(chain)]
            ax.plot(
                points[:, 0],
                points[:, 2],
                points[:, 1],
                color=color,
                linestyle=style,
                linewidth=linewidth,
                alpha=0.95,
            )
        root = person_pose[0]
        ax.scatter(
            [root[0]],
            [root[2]],
            [root[1]],
            color=color,
            s=16,
            alpha=0.95,
        )


def _render_video(path, sample_id, record, obs_xyz, gt_xyz, pred_xyz, fps, dpi):
    center, radius = _axis_limits(obs_xyz, gt_xyz, pred_xyz)
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    legend_items = [
        Line2D([0], [0], color=COLORS["observed"], lw=3, label="Observed"),
        Line2D([0], [0], color=COLORS["ground_truth"], lw=3, label="GT future"),
        Line2D([0], [0], color=COLORS["prediction"], lw=3, label="Pred future"),
    ]

    total_frames = obs_xyz.shape[0] + gt_xyz.shape[0]

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-70)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.legend(handles=legend_items, loc="lower left", frameon=True, framealpha=0.92)
        title = "sample {} | {} | frame {}/{}".format(
            sample_id,
            record.get("category", "selected"),
            frame_idx + 1,
            total_frames,
        )
        ax.set_title(title, fontsize=10)

        if frame_idx < obs_xyz.shape[0]:
            _draw_pose(ax, obs_xyz[frame_idx], COLORS["observed"], "-", 2.4)
        else:
            future_idx = frame_idx - obs_xyz.shape[0]
            _draw_pose(ax, gt_xyz[future_idx], COLORS["ground_truth"], "-", 2.2)
            _draw_pose(ax, pred_xyz[future_idx], COLORS["prediction"], "-", 2.2)

    writer = imageio.get_writer(path, fps=fps)
    try:
        for frame_idx in range(total_frames):
            update(frame_idx)
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(image)
    finally:
        writer.close()
        plt.close(fig)


def _save_sample_outputs(args, selected, arrays):
    videos_dir = os.path.join(args.save_dir, "videos")
    arrays_dir = os.path.join(args.save_dir, "arrays")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)

    for record in selected:
        sample_id = int(record["sample_id"])
        sample_arrays = arrays[sample_id]
        base = "{}_{:04d}".format(record["category"], sample_id)
        array_path = os.path.join(arrays_dir, base + ".npy")
        video_path = os.path.join(videos_dir, base + ".mp4")
        np.save(array_path, sample_arrays)
        _render_video(
            video_path,
            sample_id,
            record,
            sample_arrays["obs_xyz"],
            sample_arrays["gt_xyz"],
            sample_arrays["pred_xyz"],
            fps=args.fps,
            dpi=args.dpi,
        )
        record["array_path"] = array_path
        record["video_path"] = video_path


def _write_summary(args, selected, num_dataset_samples, checkpoint_state, device):
    summary = OrderedDict()
    summary["mode"] = "p8_official_somoformer_xyz_videos"
    summary["dataset"] = args.dataset
    summary["split"] = args.split
    summary["data_path"] = args.data_path
    summary["checkpoint"] = args.checkpoint
    summary["checkpoint_step"] = checkpoint_state.get("step")
    summary["seed"] = args.seed
    summary["device"] = str(device)
    summary["window_len"] = args.window_len
    summary["obs_len"] = args.obs_len
    summary["pred_len"] = args.pred_len
    summary["num_dataset_samples"] = int(num_dataset_samples)
    summary["num_selected_samples"] = int(len(selected))
    summary["fps"] = args.fps
    summary["colors"] = COLORS
    summary["created_at"] = _utc_now()
    _write_json(os.path.join(args.save_dir, "run_config.json"), summary)

    with open(os.path.join(args.save_dir, "summary.md"), "w") as f:
        f.write("# P8 Official SoMoFormer XYZ 骨架视频\n\n")
        f.write("- checkpoint: `{}`\n".format(args.checkpoint))
        f.write("- split: `{}`\n".format(args.split))
        f.write("- seed: `{}`\n".format(args.seed))
        f.write("- selected samples: `{}`\n".format(len(selected)))
        f.write("- colors: observed `{}`, GT future `{}`, pred future `{}`\n\n".format(
            COLORS["observed"],
            COLORS["ground_truth"],
            COLORS["prediction"],
        ))
        f.write("| category | sample_id | delta_long | delta_root_dist | video |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for item in selected:
            f.write(
                "| {} | {} | {:.6f} | {:.6f} | `{}` |\n".format(
                    item["category"],
                    item["sample_id"],
                    item["delta_long"],
                    item["delta_root_dist"],
                    item.get("video_path"),
                )
            )
    return summary


def run_visualization(args):
    if args.dataset != "interhuman":
        raise ValueError("P8 xyz 视频第一版只支持 interhuman dataset")
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")

    fixseed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = _device(args.force_cpu)
    model, checkpoint_state = _load_checkpoint_model(args.checkpoint, device)
    records, arrays, num_dataset_samples = _run_inference(args, model, device)
    selected = _select_samples(records, args.num_samples)

    _write_json(os.path.join(args.save_dir, "metrics_per_sample_all.json"), records)
    metric_rows = [_metric_row(item) for item in records]
    _write_csv(os.path.join(args.save_dir, "metrics_per_sample_all.csv"), metric_rows, _all_fieldnames(metric_rows))

    _save_sample_outputs(args, selected, arrays)
    _write_json(os.path.join(args.save_dir, "selection.json"), selected)
    selection_rows = [_metric_row(item) for item in selected]
    _write_csv(os.path.join(args.save_dir, "selection.csv"), selection_rows, _all_fieldnames(selection_rows))

    summary = _write_summary(args, selected, num_dataset_samples, checkpoint_state, device)
    print(json.dumps(_plain_data(summary), indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="interhuman")
    parser.add_argument("--data_path", default="dataset/interhuman/smpl/conditioned")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--window_len", type=int, default=150)
    parser.add_argument("--obs_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=120)
    parser.add_argument(
        "--checkpoint",
        default="save/forecasting/interhuman/p8_official_somoformer_xyz_h256_l6_dct30_s0_5000/model000005000.pt",
    )
    parser.add_argument(
        "--save_dir",
        default="results/forecasting/interhuman/p8_official_somoformer_xyz_videos",
    )
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--force_cpu", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
