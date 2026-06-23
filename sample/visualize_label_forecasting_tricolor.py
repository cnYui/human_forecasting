import argparse
import csv
import json
import os
from collections import OrderedDict
from datetime import datetime

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch

from model.rotation2xyz import Rotation2xyz_x


COLORS = OrderedDict(
    [
        ("observed", "#2F6BFF"),
        ("generated", "#FF7F0E"),
        ("real", "#2CA02C"),
    ]
)


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device(force_cpu):
    if not bool(force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _json_ready(value):
    if isinstance(value, OrderedDict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_json(path, value):
    with open(path, "w") as f:
        json.dump(_json_ready(value), f, indent=2, sort_keys=False, ensure_ascii=False)


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _append_summary(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prepare_save_dir(args):
    if not args.save_dir:
        raise FileNotFoundError("save_dir was not specified.")
    if os.path.exists(args.save_dir):
        has_files = len(os.listdir(args.save_dir)) > 0
        if has_files and not args.overwrite:
            raise FileExistsError(
                "save_dir [{}] already exists. 使用 --overwrite 或更换 save_dir。".format(
                    args.save_dir
                )
            )
        if has_files and args.overwrite:
            for root, _, files in os.walk(args.save_dir):
                for filename in files:
                    os.remove(os.path.join(root, filename))
    else:
        os.makedirs(args.save_dir)

    for name in ("videos", "frames", "arrays"):
        os.makedirs(os.path.join(args.save_dir, name), exist_ok=True)


def _ensure_finite(name, value):
    if not np.isfinite(np.asarray(value)).all():
        raise ValueError("{} 存在 NaN 或 Inf".format(name))


def _load_source_arrays(source_dir):
    paths = {
        "generated": os.path.join(source_dir, "generated_future40.npy"),
        "real": os.path.join(source_dir, "real_future40.npy"),
        "obs": os.path.join(source_dir, "obs_motion.npy"),
        "metrics": os.path.join(source_dir, "metrics.json"),
        "metadata": os.path.join(source_dir, "metadata.json"),
        "sample_metrics": os.path.join(source_dir, "sample_metrics.jsonl"),
    }
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    generated = np.load(paths["generated"]).astype(np.float32)
    real = np.load(paths["real"]).astype(np.float32)
    obs = np.load(paths["obs"]).astype(np.float32)
    for name, value in (("generated", generated), ("real", real), ("obs", obs)):
        _ensure_finite(name, value)
    if generated.ndim != 4 or real.ndim != 4 or obs.ndim != 4:
        raise ValueError("arrays 必须是 [N,56,6,T]")
    if generated.shape[:3] != (generated.shape[0], 56, 6):
        raise ValueError("generated shape 必须是 [N,56,6,40]，当前 {}".format(generated.shape))
    if real.shape != generated.shape:
        raise ValueError("real shape 必须等于 generated，当前 {} vs {}".format(real.shape, generated.shape))
    if obs.shape[:3] != (generated.shape[0], 56, 6):
        raise ValueError("obs shape 必须是 [N,56,6,20]，当前 {}".format(obs.shape))

    return {
        "generated": generated,
        "real": real,
        "obs": obs,
        "metrics": _load_json(paths["metrics"]),
        "metadata": _load_json(paths["metadata"]),
        "sample_metrics": _load_jsonl(paths["sample_metrics"]),
    }


def _load_smplx_edges(path, body_only):
    data = np.load(path, allow_pickle=True)
    if "kintree_table" not in data.files:
        raise ValueError("{} 缺少 kintree_table".format(path))
    kintree = data["kintree_table"]
    if kintree.shape[0] != 2:
        raise ValueError("kintree_table shape 异常: {}".format(kintree.shape))
    max_joint = 22 if bool(body_only) else 55
    edges = []
    for child in range(1, min(kintree.shape[1], max_joint)):
        parent = int(kintree[0, child])
        if parent < 0 or parent >= max_joint:
            continue
        edges.append((parent, child))
    if len(edges) == 0:
        raise ValueError("没有可绘制的 skeleton edges")
    return edges


def _to_xyz(converter, motion, device):
    tensor = torch.from_numpy(motion).unsqueeze(0).to(device)
    mask = torch.ones((1, tensor.shape[-1]), dtype=torch.bool, device=device)
    with torch.no_grad():
        xyz = converter(
            tensor,
            mask=mask,
            pose_rep="rot6d",
            translation=True,
            glob=True,
            jointstype="smplx",
            vertstrans=True,
            num_person=1,
        )
    xyz_np = xyz.detach().cpu().numpy()[0].transpose(2, 0, 1).astype(np.float32)
    _ensure_finite("xyz", xyz_np)
    if xyz_np.shape[1:] != (55, 3):
        raise ValueError("xyz shape 应为 [T,55,3]，当前 {}".format(xyz_np.shape))
    return xyz_np


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
    return center, span * 0.62


def _draw_pose(ax, pose, edges, color, linewidth, alpha):
    for parent, child in edges:
        points = pose[[parent, child]]
        ax.plot(
            points[:, 0],
            points[:, 2],
            points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    root = pose[0]
    ax.scatter([root[0]], [root[2]], [root[1]], color=color, s=18, alpha=alpha)


def _setup_axis(ax, center, radius, title, legend_items):
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
    ax.set_title(title, fontsize=9)


def _draw_frame(ax, frame_idx, obs_xyz, real_xyz, generated_xyz, edges, center, radius, title_prefix):
    legend_items = [
        Line2D([0], [0], color=COLORS["observed"], lw=3, label="Input obs20"),
        Line2D([0], [0], color=COLORS["generated"], lw=3, label="Generated future40"),
        Line2D([0], [0], color=COLORS["real"], lw=3, label="Real future40"),
    ]
    total_frames = obs_xyz.shape[0] + real_xyz.shape[0]
    title = "{} | frame {}/{}".format(title_prefix, frame_idx + 1, total_frames)
    _setup_axis(ax, center, radius, title, legend_items)
    if frame_idx < obs_xyz.shape[0]:
        _draw_pose(ax, obs_xyz[frame_idx], edges, COLORS["observed"], 2.6, 0.96)
    else:
        future_idx = frame_idx - obs_xyz.shape[0]
        _draw_pose(ax, generated_xyz[future_idx], edges, COLORS["generated"], 2.4, 0.92)
        _draw_pose(ax, real_xyz[future_idx], edges, COLORS["real"], 2.2, 0.92)


def _render_video(video_path, frame_path, obs_xyz, real_xyz, generated_xyz, edges, title_prefix, fps, dpi):
    center, radius = _axis_limits(obs_xyz, real_xyz, generated_xyz)
    total_frames = int(obs_xyz.shape[0] + real_xyz.shape[0])
    fig = plt.figure(figsize=(6.4, 6.4), dpi=int(dpi))
    ax = fig.add_subplot(111, projection="3d")

    writer = imageio.get_writer(video_path, fps=int(fps))
    try:
        for frame_idx in range(total_frames):
            ax.clear()
            _draw_frame(
                ax,
                frame_idx,
                obs_xyz,
                real_xyz,
                generated_xyz,
                edges,
                center,
                radius,
                title_prefix,
            )
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            if frame_idx == 0:
                imageio.imwrite(frame_path, image)
            writer.append_data(image)
    finally:
        writer.close()
        plt.close(fig)


def _base_name(index, record):
    return "case{:04d}_{}".format(int(index), record.get("action_code", "AUNK"))


def _select_records(source, num_videos):
    sample_metrics = source["sample_metrics"]
    count = min(int(num_videos), len(sample_metrics), int(source["generated"].shape[0]))
    if count < 1:
        raise ValueError("没有可视化样本")
    return sample_metrics[:count]


def run_visualization(args):
    _prepare_save_dir(args)
    device = _device(args.force_cpu)
    source = _load_source_arrays(args.source_dir)
    records = _select_records(source, args.num_videos)
    edges = _load_smplx_edges(args.smplx_model, args.body_only)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")

    selection_rows = []
    conversion_probe = []
    for index, record in enumerate(records):
        obs_xyz = _to_xyz(converter, source["obs"][index], device)
        real_xyz = _to_xyz(converter, source["real"][index], device)
        generated_xyz = _to_xyz(converter, source["generated"][index], device)
        base = _base_name(index, record)

        npz_path = os.path.join(args.save_dir, "arrays", base + ".npz")
        video_path = os.path.join(args.save_dir, "videos", base + ".mp4")
        frame_path = os.path.join(args.save_dir, "frames", base + "_first.png")

        np.savez_compressed(
            npz_path,
            obs_xyz=obs_xyz,
            real_future_xyz=real_xyz,
            generated_xyz=generated_xyz,
            record=json.dumps(record, ensure_ascii=False),
        )
        title_prefix = "{} {} mse={:.4f} mae={:.4f}".format(
            base,
            record.get("sample_id", ""),
            float(record.get("generated_mse", 0.0)),
            float(record.get("generated_mae", 0.0)),
        )
        _render_video(
            video_path,
            frame_path,
            obs_xyz,
            real_xyz,
            generated_xyz,
            edges,
            title_prefix,
            args.fps,
            args.dpi,
        )
        video_size = os.path.getsize(video_path)
        frame_size = os.path.getsize(frame_path)
        row = OrderedDict(
            [
                ("index", int(index)),
                ("sample_id", record.get("sample_id")),
                ("action", int(record.get("action", -1))),
                ("action_code", record.get("action_code")),
                ("generated_mse", float(record.get("generated_mse", 0.0))),
                ("generated_mae", float(record.get("generated_mae", 0.0))),
                ("copy_last_mse", float(record.get("copy_last_mse", 0.0))),
                ("copy_last_mae", float(record.get("copy_last_mae", 0.0))),
                ("video_path", video_path),
                ("frame_path", frame_path),
                ("array_path", npz_path),
                ("video_size", int(video_size)),
                ("frame_size", int(frame_size)),
            ]
        )
        selection_rows.append(row)
        conversion_probe.append(
            OrderedDict(
                [
                    ("index", int(index)),
                    ("obs_xyz_shape", list(obs_xyz.shape)),
                    ("real_future_xyz_shape", list(real_xyz.shape)),
                    ("generated_xyz_shape", list(generated_xyz.shape)),
                    ("obs_xyz_finite", bool(np.isfinite(obs_xyz).all())),
                    ("real_future_xyz_finite", bool(np.isfinite(real_xyz).all())),
                    ("generated_xyz_finite", bool(np.isfinite(generated_xyz).all())),
                    ("xyz_abs_mean", float(np.mean(np.abs(np.concatenate([obs_xyz, real_xyz, generated_xyz], axis=0))))),
                ]
            )
        )

    run_config = OrderedDict(
        [
            ("created_at", _utc_now()),
            ("source_dir", args.source_dir),
            ("save_dir", args.save_dir),
            ("device", str(device)),
            ("checkpoint", source["metadata"].get("checkpoint")),
            ("checkpoint_step", source["metadata"].get("checkpoint_step")),
            ("mode", source["metadata"].get("mode")),
            ("num_videos", len(selection_rows)),
            ("fps", int(args.fps)),
            ("dpi", int(args.dpi)),
            ("colors", COLORS),
            ("visualization_boundary", "single SMPL-X 55-joint skeleton from [56,6,T], not a two-person interaction video"),
            ("source_metrics", source["metrics"]),
            ("source_metadata", source["metadata"]),
            ("smplx_model", args.smplx_model),
            ("body_only", bool(args.body_only)),
        ]
    )
    _write_json(os.path.join(args.save_dir, "run_config.json"), run_config)
    _write_json(os.path.join(args.save_dir, "selection.json"), selection_rows)
    _write_json(os.path.join(args.save_dir, "conversion_probe.json"), conversion_probe)
    _write_csv(
        os.path.join(args.save_dir, "selection.csv"),
        selection_rows,
        list(selection_rows[0].keys()),
    )

    summary_lines = [
        "# Phase 7 Tricolor Skeleton Videos",
        "",
        "source_dir: `{}`".format(args.source_dir),
        "checkpoint: `{}`".format(source["metadata"].get("checkpoint")),
        "mode: `{}`".format(source["metadata"].get("mode")),
        "num_videos: `{}`".format(len(selection_rows)),
        "",
        "颜色:",
        "",
        "- 蓝色: input obs20",
        "- 橙色: generated future40",
        "- 绿色: real future40",
        "",
        "边界: 当前视频是 `[56,6,T] -> SMPL-X 55-joint` 的单 skeleton 可视化，不声明双人互动视频。",
        "",
        "| index | action | generated_mse | generated_mae | video |",
        "|---:|---|---:|---:|---|",
    ]
    for row in selection_rows:
        summary_lines.append(
            "| {index} | {action_code} | {generated_mse:.6f} | {generated_mae:.6f} | `{video}` |".format(
                index=row["index"],
                action_code=row["action_code"],
                generated_mse=row["generated_mse"],
                generated_mae=row["generated_mae"],
                video=row["video_path"],
            )
        )
    _append_summary(os.path.join(args.save_dir, "summary.md"), summary_lines)

    print("Tricolor visualization finished. save_dir={}".format(args.save_dir))
    print("videos={}".format(len(selection_rows)))
    return {"run_config": run_config, "selection": selection_rows}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--num_videos", type=int, default=8)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--smplx_model", default="body_models/smplx/SMPLX_NEUTRAL.npz")
    parser.add_argument("--body_only", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
