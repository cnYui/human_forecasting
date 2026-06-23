import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_label import (
    NUM_ACTIONS,
    NTULabelForecastDataset,
    ntu_label_forecasting_collate,
    summarize_entries,
)
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.forecasting_cmdm import ForecastingCMDMDecoder
from utils.fixseed import fixseed


MODEL_TYPE = "forecasting_cmdm_decoder"
FIXED_OUTPUTS = (
    "metadata.json",
    "metrics.json",
    "per_action_metrics.json",
    "sample_metrics.jsonl",
    "generated_future40.npy",
    "real_future40.npy",
    "obs_motion.npy",
)


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
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


def _append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(_json_ready(record), sort_keys=False, ensure_ascii=False))
        f.write("\n")


def _clear_outputs(save_dir):
    for filename in FIXED_OUTPUTS:
        path = os.path.join(save_dir, filename)
        if os.path.isfile(path):
            os.remove(path)


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
            _clear_outputs(args.save_dir)
    else:
        os.makedirs(args.save_dir)


def _ensure_finite(name, value):
    if torch.is_tensor(value):
        ok = torch.isfinite(value).all().item()
    else:
        ok = np.isfinite(np.asarray(value)).all()
    if not ok:
        raise ValueError("{} 存在 NaN 或 Inf".format(name))


def _load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device)
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError(
            "checkpoint model_type={}，预期 {}".format(state.get("model_type"), MODEL_TYPE)
        )
    if "model_state_dict" not in state or "model_config" not in state:
        raise ValueError("checkpoint 缺少 model_state_dict 或 model_config")
    protocol = state.get("train_protocol", {})
    if protocol.get("mean_type") != "START_X":
        raise ValueError("checkpoint train_protocol.mean_type 必须是 START_X")
    if protocol.get("loss_type") != "MSE":
        raise ValueError("checkpoint train_protocol.loss_type 必须是 MSE")
    return state


def _build_model(state, device):
    config = dict(state["model_config"])
    config["init_rot2xyz"] = False
    model = ForecastingCMDMDecoder(**config)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _timestep_respacing(config):
    steps = int(config.get("steps", 1000))
    value = config.get("timestep_respacing", "")
    if value is None or value == "":
        return [steps]
    return value


def _sampling_diffusion_config(state, args):
    config = dict(state.get("diffusion_config", {}))
    if "steps" not in config:
        config["steps"] = 1000
    if "noise_schedule" not in config:
        config["noise_schedule"] = "cosine"
    if "sigma_small" not in config:
        config["sigma_small"] = True

    if args.mode == "ddim50":
        config["timestep_respacing"] = args.timestep_respacing or "ddim50"
    elif args.mode == "p_sample_loop":
        config["timestep_respacing"] = args.timestep_respacing or ""
    else:
        config["timestep_respacing"] = ""

    config["model_mean_type"] = "START_X"
    config["loss_type"] = "MSE"
    config["rescale_timesteps"] = False
    return config


def _build_diffusion(config):
    steps = int(config.get("steps", 1000))
    betas = gd.get_named_beta_schedule(config.get("noise_schedule", "cosine"), steps, 1.0)
    sigma_small = bool(config.get("sigma_small", True))
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, _timestep_respacing(config)),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL if sigma_small else gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        lambda_rcxyz=0.0,
        lambda_vel=0.0,
        lambda_fc=0.0,
        lambda_orient=0.0,
        lambda_body=0.0,
        lambda_transl=0.0,
        data_rep=config.get("data_rep", "rot6d"),
        num_person=int(config.get("num_person", 2)),
        body_model=config.get("body_model", "smplx"),
    )


def _validate_args(args, state):
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(args.data_path)
    if int(args.obs_len) + int(args.pred_len) != int(args.window_len):
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    if int(args.batch_size) < 1:
        raise ValueError("batch_size 必须 >= 1")
    if int(args.max_samples) == 0:
        raise ValueError("max_samples 不能为 0；使用 -1 表示全量")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers 必须 >= 0")
    if int(args.one_step_t) < 0:
        raise ValueError("one_step_t 必须 >= 0")

    model_config = state["model_config"]
    for key in ("window_len", "obs_len", "pred_len"):
        current = int(getattr(args, key))
        expected = int(model_config[key])
        if current != expected:
            raise ValueError("{} CLI={} 与 checkpoint={} 不一致".format(key, current, expected))


def _build_dataset(args):
    return NTULabelForecastDataset(
        h5_path=args.data_path,
        split="test",
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
        strict=True,
    )


def _build_loader(args, dataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ntu_label_forecasting_collate,
        drop_last=False,
    )


def _slice_y(y, start, stop):
    return {key: value[start:stop] for key, value in y.items()}


def _sample_loop(model, diffusion, y, shape, args, device):
    noise = torch.randn(shape, device=device)
    sample_fn = diffusion.ddim_sample_loop if args.mode == "ddim50" else diffusion.p_sample_loop
    outputs = []
    total = int(shape[0])
    for start in range(0, total, int(args.sample_batch_size)):
        stop = min(start + int(args.sample_batch_size), total)
        y_chunk = _slice_y(y, start, stop)
        noise_chunk = noise[start:stop]
        sample = sample_fn(
            model,
            tuple(noise_chunk.shape),
            clip_denoised=False,
            model_kwargs={"y": y_chunk},
            device=device,
            progress=bool(args.progress),
            skip_timesteps=0,
            init_image=None,
            dump_steps=None,
            noise=noise_chunk,
            const_noise=False,
        )
        _ensure_finite("sample", sample)
        outputs.append(sample.detach())
    return torch.cat(outputs, dim=0)


def _one_step_t999(model, y, shape, args, device):
    x_t = torch.randn(shape, device=device)
    t = torch.full((shape[0],), int(args.one_step_t), device=device, dtype=torch.long)
    sample = model(x_t, t, y)
    _ensure_finite("one_step sample", sample)
    return sample.detach()


def _copy_last_baseline(obs_motion, pred_len):
    return obs_motion[..., -1:].repeat(1, 1, 1, int(pred_len))


def _metric_sums(pred, target):
    diff = pred - target
    abs_diff = torch.abs(diff)
    squared = diff * diff
    return {
        "sum_squared": float(squared.sum().detach().cpu().item()),
        "sum_abs": float(abs_diff.sum().detach().cpu().item()),
        "numel": int(diff.numel()),
    }


def _sample_metrics(pred, target):
    diff = pred - target
    mse = (diff * diff).mean(dim=(1, 2, 3))
    mae = torch.abs(diff).mean(dim=(1, 2, 3))
    return mse.detach().cpu(), mae.detach().cpu()


def _empty_sums():
    return {
        "generated": {"sum_squared": 0.0, "sum_abs": 0.0, "numel": 0},
        "copy_last": {"sum_squared": 0.0, "sum_abs": 0.0, "numel": 0},
        "zero": {"sum_squared": 0.0, "sum_abs": 0.0, "numel": 0},
    }


def _add_sums(total, item):
    for key in ("sum_squared", "sum_abs", "numel"):
        total[key] += item[key]


def _finalize_metric_sums(sums):
    if int(sums["numel"]) < 1:
        return {"mse": None, "rmse": None, "mae": None, "numel": 0}
    mse = float(sums["sum_squared"]) / float(sums["numel"])
    mae = float(sums["sum_abs"]) / float(sums["numel"])
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "numel": int(sums["numel"]),
    }


def _action_code(label):
    return "A{:03d}".format(int(label) + 1)


def _finalize_all_metrics(metric_sums, total_samples):
    generated = _finalize_metric_sums(metric_sums["generated"])
    copy_last = _finalize_metric_sums(metric_sums["copy_last"])
    zero = _finalize_metric_sums(metric_sums["zero"])
    return OrderedDict(
        [
            ("num_samples", int(total_samples)),
            ("generated", generated),
            ("copy_last", copy_last),
            ("zero", zero),
            (
                "generated_minus_copy_last_mse",
                None if generated["mse"] is None else generated["mse"] - copy_last["mse"],
            ),
            (
                "generated_minus_copy_last_mae",
                None if generated["mae"] is None else generated["mae"] - copy_last["mae"],
            ),
            (
                "generated_mse_better_than_copy_last",
                None if generated["mse"] is None else generated["mse"] < copy_last["mse"],
            ),
            (
                "generated_mae_better_than_copy_last",
                None if generated["mae"] is None else generated["mae"] < copy_last["mae"],
            ),
        ]
    )


def _append_arrays(store, generated, real_future, obs_motion):
    store["generated"].append(generated.detach().cpu().float())
    store["real_future"].append(real_future.detach().cpu().float())
    store["obs_motion"].append(obs_motion.detach().cpu().float())


def _write_arrays(args, store):
    if not bool(args.save_arrays):
        return {}
    paths = {}
    for key, filename in (
        ("generated", "generated_future40.npy"),
        ("real_future", "real_future40.npy"),
        ("obs_motion", "obs_motion.npy"),
    ):
        value = torch.cat(store[key], dim=0).numpy().astype(np.float32)
        _ensure_finite(key, value)
        path = os.path.join(args.save_dir, filename)
        np.save(path, value)
        paths[key] = path
    return paths


@torch.no_grad()
def run_evaluation(args):
    fixseed(args.seed)
    device = _device()
    state = _load_checkpoint(args.checkpoint, device)
    _validate_args(args, state)
    _prepare_save_dir(args)

    model = _build_model(state, device)
    diffusion_config = _sampling_diffusion_config(state, args)
    diffusion = None if args.mode == "one_step_t999" else _build_diffusion(diffusion_config)
    dataset = _build_dataset(args)
    loader = _build_loader(args, dataset)

    metric_sums = _empty_sums()
    per_action_sums = {idx: _empty_sums() for idx in range(NUM_ACTIONS)}
    per_action_counts = [0 for _ in range(NUM_ACTIONS)]
    array_store = {"generated": [], "real_future": [], "obs_motion": []}
    sample_metrics_path = os.path.join(args.save_dir, "sample_metrics.jsonl")

    total_samples = 0
    for batch_index, batch in enumerate(loader):
        obs_motion = batch["obs_motion"].to(device)
        real_future = batch["future"].to(device)
        action = batch["action"].to(device)
        mask = batch["mask"].to(device)

        y = {"obs_motion": obs_motion, "action": action, "mask": mask}
        shape = tuple(real_future.shape)
        if args.mode == "one_step_t999":
            generated = _one_step_t999(model, y, shape, args, device)
        else:
            generated = _sample_loop(model, diffusion, y, shape, args, device)

        copy_last = _copy_last_baseline(obs_motion, args.pred_len)
        zero = torch.zeros_like(real_future)

        for name, pred in (
            ("generated", generated),
            ("copy_last", copy_last),
            ("zero", zero),
        ):
            _add_sums(metric_sums[name], _metric_sums(pred, real_future))

        generated_mse, generated_mae = _sample_metrics(generated, real_future)
        copy_mse, copy_mae = _sample_metrics(copy_last, real_future)
        zero_mse, zero_mae = _sample_metrics(zero, real_future)

        labels = action.view(-1).detach().cpu().tolist()
        for item_index, label in enumerate(labels):
            label = int(label)
            per_action_counts[label] += 1
            for name, pred in (
                ("generated", generated[item_index : item_index + 1]),
                ("copy_last", copy_last[item_index : item_index + 1]),
                ("zero", zero[item_index : item_index + 1]),
            ):
                _add_sums(
                    per_action_sums[label][name],
                    _metric_sums(pred, real_future[item_index : item_index + 1]),
                )

            meta = batch["meta"][item_index]
            _append_jsonl(
                sample_metrics_path,
                OrderedDict(
                    [
                        ("global_index", int(total_samples + item_index)),
                        ("batch_index", int(batch_index)),
                        ("sample_id", meta["sample_id"]),
                        ("start", int(meta["start"])),
                        ("length", int(meta["length"])),
                        ("action", label),
                        ("action_code", _action_code(label)),
                        ("generated_mse", float(generated_mse[item_index].item())),
                        ("generated_rmse", float(torch.sqrt(generated_mse[item_index]).item())),
                        ("generated_mae", float(generated_mae[item_index].item())),
                        ("copy_last_mse", float(copy_mse[item_index].item())),
                        ("copy_last_rmse", float(torch.sqrt(copy_mse[item_index]).item())),
                        ("copy_last_mae", float(copy_mae[item_index].item())),
                        ("zero_mse", float(zero_mse[item_index].item())),
                        ("zero_rmse", float(torch.sqrt(zero_mse[item_index]).item())),
                        ("zero_mae", float(zero_mae[item_index].item())),
                    ]
                ),
            )

        if bool(args.save_arrays):
            _append_arrays(array_store, generated, real_future, obs_motion)

        total_samples += int(real_future.shape[0])
        if args.log_interval > 0 and total_samples % int(args.log_interval) == 0:
            print("evaluated_samples={}".format(total_samples))

    metrics = _finalize_all_metrics(metric_sums, total_samples)
    per_action_metrics = OrderedDict()
    for label in range(NUM_ACTIONS):
        if per_action_counts[label] == 0:
            continue
        value = _finalize_all_metrics(per_action_sums[label], per_action_counts[label])
        value["action"] = int(label)
        value["action_code"] = _action_code(label)
        per_action_metrics[str(label)] = value

    array_paths = _write_arrays(args, array_store)
    metadata = OrderedDict(
        [
            ("created_at", _utc_now()),
            ("checkpoint", args.checkpoint),
            ("checkpoint_step", int(state.get("step", -1))),
            ("data_path", args.data_path),
            ("save_dir", args.save_dir),
            ("mode", args.mode),
            ("seed", int(args.seed)),
            ("device", str(device)),
            ("window_len", int(args.window_len)),
            ("obs_len", int(args.obs_len)),
            ("pred_len", int(args.pred_len)),
            ("batch_size", int(args.batch_size)),
            ("sample_batch_size", int(args.sample_batch_size)),
            ("max_samples", int(args.max_samples)),
            ("save_arrays", bool(args.save_arrays)),
            ("array_paths", array_paths),
            ("dataset_summary", summarize_entries(dataset.scan_result)),
            ("evaluated_summary", {"num_samples": int(total_samples), "per_action_counts": per_action_counts}),
            ("model_config", state["model_config"]),
            ("checkpoint_diffusion_config", state.get("diffusion_config", {})),
            ("sampling_diffusion_config", diffusion_config),
        ]
    )

    _write_json(os.path.join(args.save_dir, "metadata.json"), metadata)
    _write_json(os.path.join(args.save_dir, "metrics.json"), metrics)
    _write_json(os.path.join(args.save_dir, "per_action_metrics.json"), per_action_metrics)

    print("Distance evaluation finished. save_dir={}".format(args.save_dir))
    print(
        "mode={} samples={} generated_mse={:.6f} copy_last_mse={:.6f}".format(
            args.mode,
            total_samples,
            metrics["generated"]["mse"],
            metrics["copy_last"]["mse"],
        )
    )
    return {"metadata": metadata, "metrics": metrics, "per_action_metrics": per_action_metrics}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--mode", choices=("ddim50", "p_sample_loop", "one_step_t999"), default="ddim50")
    parser.add_argument("--timestep_respacing", default="")
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--one_step_t", type=int, default=999)
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_interval", type=int, default=0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
