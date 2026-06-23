import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch

from data_loaders.forecasting.ntu_label import NTULabelForecastDataset
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.forecasting_cmdm import (
    ForecastingCMDMDecoder,
    ForecastingClassifierFreeSampleModel,
)
from utils.fixseed import fixseed


MODEL_TYPE = "forecasting_cmdm_decoder"
NUM_ACTIONS = 26
FIXED_OUTPUTS = (
    "generated_future40.npy",
    "obs_motion.npy",
    "real_future40.npy",
    "metadata.json",
    "metrics.json",
    "label_swap_summary.json",
)


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path, value):
    with open(path, "w") as f:
        json.dump(_json_ready(value), f, indent=2, sort_keys=True, ensure_ascii=False)


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


def _clear_stage_outputs(save_dir):
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
            _clear_stage_outputs(args.save_dir)
    else:
        os.makedirs(args.save_dir)


def _ensure_finite(name, value):
    if torch.is_tensor(value):
        ok = torch.isfinite(value).all().item()
    else:
        ok = np.isfinite(value).all()
    if not ok:
        raise ValueError("{} 存在 NaN 或 Inf".format(name))


def _to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)


def _validate_checkpoint_state(state):
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError(
            "checkpoint model_type={}，预期 {}".format(state.get("model_type"), MODEL_TYPE)
        )
    protocol = state.get("train_protocol", {})
    if protocol.get("mean_type") != "START_X":
        raise ValueError("checkpoint train_protocol.mean_type 必须是 START_X")
    if protocol.get("loss_type") != "MSE":
        raise ValueError("checkpoint train_protocol.loss_type 必须是 MSE")
    if "model_config" not in state:
        raise ValueError("checkpoint 缺少 model_config")
    if "model_state_dict" not in state:
        raise ValueError("checkpoint 缺少 model_state_dict")


def _build_model_from_checkpoint(state, device):
    model_config = dict(state["model_config"])
    model_config["init_rot2xyz"] = False
    model = ForecastingCMDMDecoder(**model_config)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _timestep_respacing(config):
    steps = int(config["steps"])
    value = config.get("timestep_respacing", "")
    if value is None or value == "":
        return [steps]
    return value


def _sampling_diffusion_config(checkpoint_config, args):
    config = dict(checkpoint_config)
    if args.timestep_respacing:
        config["timestep_respacing"] = args.timestep_respacing
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
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(args.data_path)
    if len(args.labels) == 0:
        raise ValueError("labels 不能为空")
    for label in args.labels:
        if int(label) < 0 or int(label) >= NUM_ACTIONS:
            raise ValueError("label 必须在 [0,{}] 内，当前为 {}".format(NUM_ACTIONS - 1, label))
    if args.num_cases < 1:
        raise ValueError("num_cases 必须 >= 1")
    if args.num_repetitions < 1:
        raise ValueError("num_repetitions 必须 >= 1")
    if args.batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")
    if args.guidance_scale < 0:
        raise ValueError("guidance_scale 必须 >= 0")

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
        max_samples=-1,
        seed=args.seed,
        strict=True,
    )


def _source_meta(item, dataset_index):
    return {
        "dataset_index": int(dataset_index),
        "sample_id": item["sample_id"],
        "start": int(item["start"]),
        "length": int(item["length"]),
        "source_action": int(item["action"].item()),
        "source_action_code": item["action_code"],
        "source_action_name": item["action_name"],
    }


def _select_cases(dataset, args):
    start = int(args.sample_index)
    stop = start + int(args.num_cases)
    if start < 0 or stop > len(dataset):
        raise ValueError(
            "sample_index/num_cases 越界: [{}:{}) dataset_len={}".format(
                start, stop, len(dataset)
            )
        )
    cases = []
    for dataset_index in range(start, stop):
        item = dataset[dataset_index]
        cases.append({"item": item, "meta": _source_meta(item, dataset_index)})
    return cases


def _label_action_codes(labels):
    return ["A{:03d}".format(int(label) + 1) for label in labels]


def _build_label_swap_batch(cases, labels, num_repetitions, device):
    obs_items = []
    action_items = []
    mask_items = []
    noise_items = []
    batch_meta = []
    source_obs = []
    source_future = []

    for case_idx, case in enumerate(cases):
        item = case["item"]
        obs = item["obs_motion"]
        future = item["future"]
        mask = item["mask"].unsqueeze(0)
        source_obs.append(obs)
        source_future.append(future)

        shared_noise = torch.randn(
            int(num_repetitions),
            future.shape[0],
            future.shape[1],
            future.shape[2],
        )
        for label_idx, label in enumerate(labels):
            for rep_idx in range(int(num_repetitions)):
                obs_items.append(obs)
                action_items.append(torch.tensor([int(label)], dtype=torch.long))
                mask_items.append(mask.squeeze(0))
                noise_items.append(shared_noise[rep_idx])
                batch_meta.append(
                    {
                        "case_index": int(case_idx),
                        "label_index": int(label_idx),
                        "label": int(label),
                        "label_action_code": "A{:03d}".format(int(label) + 1),
                        "repetition": int(rep_idx),
                    }
                )

    y = {
        "obs_motion": torch.stack(obs_items, dim=0).float().to(device),
        "action": torch.stack(action_items, dim=0).long().to(device),
        "mask": torch.stack(mask_items, dim=0).bool().to(device),
    }
    noise = torch.stack(noise_items, dim=0).float().to(device)
    source = {
        "obs_motion": torch.stack(source_obs, dim=0).float(),
        "real_future": torch.stack(source_future, dim=0).float(),
    }
    return y, noise, batch_meta, source


def _slice_y(y, start, stop):
    result = {}
    for key, value in y.items():
        result[key] = value[start:stop]
    return result


def _sample_in_batches(model, diffusion, y, noise, args, device):
    if args.guidance_scale == 1.0:
        sample_model = model
    else:
        sample_model = ForecastingClassifierFreeSampleModel(model, args.guidance_scale)
        sample_model.to(device)
        sample_model.eval()

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
    outputs = []
    total = int(noise.shape[0])
    for start in range(0, total, int(args.batch_size)):
        stop = min(start + int(args.batch_size), total)
        y_chunk = _slice_y(y, start, stop)
        if args.guidance_scale != 1.0:
            y_chunk["scale"] = torch.ones(stop - start, device=device) * float(args.guidance_scale)
        noise_chunk = noise[start:stop]
        sample = sample_fn(
            sample_model,
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
        outputs.append(sample.detach().cpu())
    return torch.cat(outputs, dim=0)


def _compute_rot_mse(generated, real_future):
    diff = generated - real_future[:, None, None]
    return (diff ** 2).mean(axis=(3, 4, 5))


def _compute_root_translation_mse(generated, real_future):
    generated_root = generated[:, :, :, -1:, 0:3, :]
    real_root = real_future[:, None, None, -1:, 0:3, :]
    return ((generated_root - real_root) ** 2).mean(axis=(3, 4, 5))


def _compute_label_swap_summary(generated, labels):
    pairwise = []
    pass_non_identical = False
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            diff = np.abs(generated[:, i] - generated[:, j])
            mean_abs = diff.mean(axis=(2, 3, 4))
            max_abs = diff.max(axis=(2, 3, 4))
            max_scalar = float(max_abs.max())
            if max_scalar > 1e-7:
                pass_non_identical = True
            pairwise.append(
                {
                    "label_a": int(labels[i]),
                    "label_b": int(labels[j]),
                    "mean_abs_diff_by_case_rep": mean_abs.tolist(),
                    "max_abs_diff_by_case_rep": max_abs.tolist(),
                    "mean_abs_diff": float(mean_abs.mean()),
                    "max_abs_diff": max_scalar,
                }
            )
    return {
        "labels": [int(label) for label in labels],
        "pairwise": pairwise,
        "all_labels_identical": not pass_non_identical,
        "pass_non_identical_check": bool(pass_non_identical),
        "threshold": 1e-7,
    }


def _compute_metrics(generated, real_future, labels, source_meta, args):
    rot_mse = _compute_rot_mse(generated, real_future)
    root_mse = _compute_root_translation_mse(generated, real_future)
    finite = bool(np.isfinite(generated).all())
    return {
        "run_name": args.run_name,
        "smoke_only": not bool(args.formal),
        "finite": finite,
        "generated_shape": list(generated.shape),
        "labels": [int(label) for label in labels],
        "source_action": [int(item["source_action"]) for item in source_meta],
        "rot_mse_by_case_label_rep": rot_mse.tolist(),
        "rot_mse_by_label": rot_mse.mean(axis=(0, 2)).tolist(),
        "rot_mse_mean": float(rot_mse.mean()),
        "root_translation_mse_by_case_label_rep": root_mse.tolist(),
        "root_translation_mse_by_label": root_mse.mean(axis=(0, 2)).tolist(),
        "root_translation_mse_mean": float(root_mse.mean()),
    }


def _write_outputs(args, state, sampling_config, generated, source, source_meta, batch_meta):
    generated_np = _to_numpy(generated).astype(np.float32)
    generated_np = generated_np.reshape(
        int(args.num_cases),
        len(args.labels),
        int(args.num_repetitions),
        56,
        6,
        int(args.pred_len),
    )
    obs_np = _to_numpy(source["obs_motion"]).astype(np.float32)
    future_np = _to_numpy(source["real_future"]).astype(np.float32)

    _ensure_finite("generated_future40", generated_np)
    _ensure_finite("obs_motion", obs_np)
    _ensure_finite("real_future40", future_np)

    metrics = _compute_metrics(generated_np, future_np, args.labels, source_meta, args)
    summary = _compute_label_swap_summary(generated_np, args.labels)
    metadata = {
        "run_name": args.run_name,
        "formal": bool(args.formal),
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(state.get("step", -1)),
        "data_path": args.data_path,
        "save_dir": args.save_dir,
        "seed": int(args.seed),
        "device": str(_device()),
        "window_len": int(args.window_len),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "labels": [int(label) for label in args.labels],
        "label_action_codes": _label_action_codes(args.labels),
        "guidance_scale": float(args.guidance_scale),
        "use_ddim": bool(args.use_ddim),
        "timestep_respacing": args.timestep_respacing,
        "num_cases": int(args.num_cases),
        "num_repetitions": int(args.num_repetitions),
        "generated_shape": list(generated_np.shape),
        "source_meta": source_meta,
        "batch_meta": batch_meta,
        "model_config": state["model_config"],
        "diffusion_config": state.get("diffusion_config", {}),
        "sampling_diffusion_config": sampling_config,
        "created_at": _utc_now(),
    }

    np.save(os.path.join(args.save_dir, "generated_future40.npy"), generated_np)
    np.save(os.path.join(args.save_dir, "obs_motion.npy"), obs_np)
    np.save(os.path.join(args.save_dir, "real_future40.npy"), future_np)
    _write_json(os.path.join(args.save_dir, "metadata.json"), metadata)
    _write_json(os.path.join(args.save_dir, "metrics.json"), metrics)
    _write_json(os.path.join(args.save_dir, "label_swap_summary.json"), summary)
    return metadata, metrics, summary


def run_sampling(args):
    fixseed(args.seed)
    device = _device()
    state = _load_checkpoint(args.checkpoint, device)
    _validate_checkpoint_state(state)
    _validate_args(args, state)
    _prepare_save_dir(args)

    model = _build_model_from_checkpoint(state, device)
    sampling_config = _sampling_diffusion_config(state.get("diffusion_config", {}), args)
    diffusion = _build_diffusion(sampling_config)
    dataset = _build_dataset(args)
    cases = _select_cases(dataset, args)
    y, noise, batch_meta, source = _build_label_swap_batch(
        cases,
        args.labels,
        args.num_repetitions,
        device,
    )
    generated = _sample_in_batches(model, diffusion, y, noise, args, device)
    source_meta = [case["meta"] for case in cases]
    metadata, metrics, summary = _write_outputs(
        args,
        state,
        sampling_config,
        generated,
        source,
        source_meta,
        batch_meta,
    )
    print("Sampling finished. save_dir={}".format(args.save_dir))
    print("generated_shape={}".format(metadata["generated_shape"]))
    print("finite={}".format(metrics["finite"]))
    print("pass_non_identical_check={}".format(summary["pass_non_identical_check"]))
    return {
        "metadata": metadata,
        "metrics": metrics,
        "label_swap_summary": summary,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--labels", type=int, nargs="+", default=[2, 5, 8, 17])
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_cases", type=int, default=1)
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--timestep_respacing", default="")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run_name", default="label_forecasting_sampling")
    parser.add_argument(
        "--formal",
        action="store_true",
        help="正式评估输出使用 smoke_only=false；默认保持 smoke_only=true 兼容早期 smoke。",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_sampling(args)


if __name__ == "__main__":
    main()
