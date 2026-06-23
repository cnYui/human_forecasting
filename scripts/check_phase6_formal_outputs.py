import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch


MODEL_TYPE = "forecasting_cmdm_decoder"
NUM_JOINTS = 56
NUM_FEATS = 6


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("{} 第 {} 行不是合法 JSON: {}".format(path, line_number, exc))
    return rows


def _load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _require(condition, message, errors):
    if not condition:
        errors.append(message)


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _same_path(a, b):
    if not a or not b:
        return False
    return os.path.abspath(a) == os.path.abspath(b) or os.path.normpath(a) == os.path.normpath(b)


def _check_checkpoint(args, errors):
    model_path = os.path.join(args.save_dir, "model{:09d}.pt".format(args.expected_step))
    opt_path = os.path.join(args.save_dir, "opt{:09d}.pt".format(args.expected_step))
    _require(os.path.exists(model_path), "缺少 checkpoint: {}".format(model_path), errors)
    _require(os.path.exists(opt_path), "缺少 optimizer checkpoint: {}".format(opt_path), errors)
    if not os.path.exists(model_path):
        return OrderedDict([("model_path", model_path), ("exists", False)])

    state = _load_torch(model_path)
    protocol = state.get("train_protocol", {})
    diffusion_config = state.get("diffusion_config", {})
    _require(state.get("step") == args.expected_step, "checkpoint.step 不是 {}".format(args.expected_step), errors)
    _require(state.get("model_type") == MODEL_TYPE, "checkpoint.model_type 不是 {}".format(MODEL_TYPE), errors)
    _require(protocol.get("mean_type") == "START_X", "train_protocol.mean_type 不是 START_X", errors)
    _require(protocol.get("loss_type") == "MSE", "train_protocol.loss_type 不是 MSE", errors)
    _require(protocol.get("target") == "future", "train_protocol.target 不是 future", errors)
    _require(protocol.get("condition") == "obs_motion + action", "train_protocol.condition 不匹配", errors)
    _require(diffusion_config.get("noise_schedule") == args.noise_schedule, "diffusion noise_schedule 不匹配", errors)

    return OrderedDict(
        [
            ("model_path", model_path),
            ("opt_path", opt_path),
            ("exists", True),
            ("step", int(state.get("step", -1))),
            ("model_type", state.get("model_type")),
            ("mean_type", protocol.get("mean_type")),
            ("loss_type", protocol.get("loss_type")),
            ("noise_schedule", diffusion_config.get("noise_schedule")),
        ]
    )


def _check_train_log(args, errors):
    path = os.path.join(args.save_dir, "train_log.jsonl")
    _require(os.path.exists(path), "缺少 train_log.jsonl: {}".format(path), errors)
    if not os.path.exists(path):
        return OrderedDict([("path", path), ("exists", False)])

    rows = _load_jsonl(path)
    steps = [int(row.get("step", -1)) for row in rows]
    expected_steps = list(range(1, int(args.expected_step) + 1))
    _require(len(rows) == args.expected_step, "train_log 行数不是 {}".format(args.expected_step), errors)
    _require(steps == expected_steps, "train_log step 不连续或范围不对", errors)

    for index, row in enumerate(rows):
        for key in ("train_loss", "rot_mse", "loss"):
            if key in row:
                _require(
                    _finite_number(row[key]),
                    "train_log 第 {} 行 {} 非有限".format(index + 1, key),
                    errors,
                )

    return OrderedDict(
        [
            ("path", path),
            ("exists", True),
            ("rows", len(rows)),
            ("first_loss", rows[0].get("train_loss") if rows else None),
            ("last_loss", rows[-1].get("train_loss") if rows else None),
        ]
    )


def _check_sampling(args, errors):
    generated_path = os.path.join(args.generated_dir, "generated_future40.npy")
    obs_path = os.path.join(args.generated_dir, "obs_motion.npy")
    real_path = os.path.join(args.generated_dir, "real_future40.npy")
    metadata_path = os.path.join(args.generated_dir, "metadata.json")
    metrics_path = os.path.join(args.generated_dir, "metrics.json")
    summary_path = os.path.join(args.generated_dir, "label_swap_summary.json")

    for path in (generated_path, obs_path, real_path, metadata_path, metrics_path, summary_path):
        _require(os.path.exists(path), "缺少采样输出: {}".format(path), errors)
    if not all(os.path.exists(path) for path in (generated_path, obs_path, real_path, metadata_path, metrics_path, summary_path)):
        return OrderedDict([("generated_dir", args.generated_dir), ("exists", False)])

    generated = np.load(generated_path)
    obs = np.load(obs_path)
    real = np.load(real_path)
    metadata = _load_json(metadata_path)
    metrics = _load_json(metrics_path)
    summary = _load_json(summary_path)

    expected_shape = [
        int(args.num_cases),
        len(args.labels),
        int(args.num_repetitions),
        NUM_JOINTS,
        NUM_FEATS,
        int(args.pred_len),
    ]
    _require(list(generated.shape) == expected_shape, "generated_future40 shape 不匹配", errors)
    _require(list(obs.shape) == [int(args.num_cases), NUM_JOINTS, NUM_FEATS, int(args.obs_len)], "obs_motion shape 不匹配", errors)
    _require(list(real.shape) == [int(args.num_cases), NUM_JOINTS, NUM_FEATS, int(args.pred_len)], "real_future40 shape 不匹配", errors)
    _require(bool(np.isfinite(generated).all()), "generated_future40 存在 NaN 或 Inf", errors)
    _require(bool(metrics.get("finite")) is True, "metrics.finite 不是 true", errors)
    _require(metrics.get("smoke_only") is False, "正式阶段 metrics.smoke_only 应为 false，请采样时传 --formal", errors)
    _require(metadata.get("formal") is True, "metadata.formal 应为 true，请采样时传 --formal", errors)
    _require(summary.get("pass_non_identical_check") is True, "label swap 非完全相同检查未通过", errors)
    _require([int(item) for item in metadata.get("labels", [])] == [int(item) for item in args.labels], "metadata.labels 不匹配", errors)

    checkpoint = os.path.join(args.save_dir, "model{:09d}.pt".format(args.expected_step))
    _require(_same_path(metadata.get("checkpoint"), checkpoint), "metadata.checkpoint 不指向正式 checkpoint", errors)

    return OrderedDict(
        [
            ("generated_dir", args.generated_dir),
            ("exists", True),
            ("generated_shape", list(generated.shape)),
            ("finite", bool(np.isfinite(generated).all())),
            ("metrics_finite", bool(metrics.get("finite"))),
            ("smoke_only", metrics.get("smoke_only")),
            ("pass_non_identical_check", bool(summary.get("pass_non_identical_check"))),
        ]
    )


def _check_generated_consistency(args, errors):
    path = os.path.join(args.classifier_dir, "generated_consistency.json")
    _require(os.path.exists(path), "缺少 generated_consistency.json: {}".format(path), errors)
    if not os.path.exists(path):
        return OrderedDict([("path", path), ("exists", False)])

    consistency = _load_json(path)
    _require("valid_for_claim" in consistency, "generated_consistency 缺少 valid_for_claim", errors)
    _require("classifier_gate_pass" in consistency, "generated_consistency 缺少 classifier_gate_pass", errors)
    _require(_same_path(consistency.get("generated_dir"), args.generated_dir), "generated_consistency.generated_dir 不匹配", errors)

    return OrderedDict(
        [
            ("path", path),
            ("exists", True),
            ("classifier_gate_pass", consistency.get("classifier_gate_pass")),
            ("valid_for_claim", consistency.get("valid_for_claim")),
            ("consistency_acc", consistency.get("consistency_acc")),
        ]
    )


def run_check(args):
    errors = []
    summary = OrderedDict()
    summary["checkpoint"] = _check_checkpoint(args, errors)
    summary["train_log"] = _check_train_log(args, errors)
    summary["sampling"] = _check_sampling(args, errors)
    summary["generated_consistency"] = _check_generated_consistency(args, errors)
    summary["pass"] = len(errors) == 0
    summary["errors"] = errors
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if len(errors) == 0 else 1


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="save/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000",
    )
    parser.add_argument(
        "--generated_dir",
        default="results/forecasting/ntu120_label/phase6_cmdm_decoder_len60_o20_p40_h256_l4_s0_5000_label_swap",
    )
    parser.add_argument(
        "--classifier_dir",
        default="save/forecasting/ntu120_label/phase6_action_classifier_generated_consistency_h256_b3_s0",
    )
    parser.add_argument("--expected_step", type=int, default=5000)
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--num_cases", type=int, default=8)
    parser.add_argument("--num_repetitions", type=int, default=2)
    parser.add_argument("--labels", type=int, nargs="+", default=[2, 5, 8, 17])
    parser.add_argument("--noise_schedule", default="cosine")
    return parser


def main():
    args = build_arg_parser().parse_args()
    sys.exit(run_check(args))


if __name__ == "__main__":
    main()
