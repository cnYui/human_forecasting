import argparse
import json
import math
import os
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
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
from utils.fixseed import fixseed


MODEL_TYPE = "temporal_cnn_action_classifier"
DATASET = "ntu120_2p"
GATE_THRESHOLDS = OrderedDict(
    [
        ("top1_acc", 0.15),
        ("top5_acc", 0.50),
        ("handshaking_acc", 0.20),
    ]
)
FIXED_OUTPUTS = (
    "args.json",
    "train_log.jsonl",
    "classifier_model.pt",
    "normalizer.pt",
    "real_test_metrics.json",
    "real_test_predictions.jsonl",
    "generated_consistency.json",
    "generated_predictions.jsonl",
    "confusion_matrix.npy",
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
        ok = np.isfinite(np.asarray(value)).all()
    if not ok:
        raise ValueError("{} 存在 NaN 或 Inf".format(name))


def _action_code(label):
    return "A{:03d}".format(int(label) + 1)


def _count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _group_count(channels):
    for group_count in (16, 8, 4, 2):
        if int(channels) % group_count == 0:
            return group_count
    return 1


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(ResidualTemporalBlock, self).__init__()
        groups = _group_count(hidden_dim)
        self.norm1 = nn.GroupNorm(groups, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.gelu(self.norm2(x)))
        x = self.dropout(x)
        return residual + x


class TemporalCNNActionClassifier(nn.Module):
    def __init__(
        self,
        input_channels=NTU_JOINTS * NTU_FEATS,
        hidden_dim=128,
        num_blocks=2,
        dropout=0.1,
        num_classes=NUM_ACTIONS,
        pred_len=40,
    ):
        super(TemporalCNNActionClassifier, self).__init__()
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)
        self.pred_len = int(pred_len)

        self.input_proj = nn.Conv1d(self.input_channels, self.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(self.hidden_dim, self.dropout)
                for _ in range(self.num_blocks)
            ]
        )
        self.out_norm = nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def config(self):
        return {
            "model_type": MODEL_TYPE,
            "input_channels": int(self.input_channels),
            "hidden_dim": int(self.hidden_dim),
            "num_blocks": int(self.num_blocks),
            "dropout": float(self.dropout),
            "num_classes": int(self.num_classes),
            "pred_len": int(self.pred_len),
        }

    def forward(self, future):
        if future.dim() == 4:
            batch_size, num_joints, num_feats, pred_len = future.shape
            expected_channels = int(num_joints) * int(num_feats)
            if expected_channels != self.input_channels:
                raise ValueError(
                    "future channel={}，预期 {}".format(
                        expected_channels, self.input_channels
                    )
                )
            if int(pred_len) != self.pred_len:
                raise ValueError("future pred_len={}，预期 {}".format(pred_len, self.pred_len))
            x = future.reshape(batch_size, self.input_channels, pred_len)
        elif future.dim() == 3:
            if int(future.shape[1]) != self.input_channels:
                raise ValueError(
                    "future channel={}，预期 {}".format(
                        int(future.shape[1]), self.input_channels
                    )
                )
            if int(future.shape[2]) != self.pred_len:
                raise ValueError(
                    "future pred_len={}，预期 {}".format(int(future.shape[2]), self.pred_len)
                )
            x = future
        else:
            raise ValueError("future 必须是 [B,56,6,T] 或 [B,336,T]")

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.out_norm(x))
        x = x.mean(dim=-1)
        return self.head(x)


def _validate_args(args):
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(args.train_path)
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(args.test_path)
    if int(args.obs_len) + int(args.pred_len) != int(args.window_len):
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    if int(args.batch_size) < 1:
        raise ValueError("batch_size 必须 >= 1")
    if int(args.eval_batch_size) < 1:
        raise ValueError("eval_batch_size 必须 >= 1")
    if int(args.num_steps) < 1:
        raise ValueError("num_steps 必须 >= 1")
    if int(args.save_interval) < 1:
        raise ValueError("save_interval 必须 >= 1")
    if int(args.log_interval) < 1:
        raise ValueError("log_interval 必须 >= 1")
    if int(args.hidden_dim) < 1:
        raise ValueError("hidden_dim 必须 >= 1")
    if int(args.num_blocks) < 1:
        raise ValueError("num_blocks 必须 >= 1")
    if float(args.dropout) < 0.0:
        raise ValueError("dropout 必须 >= 0")
    if float(args.lr) <= 0.0:
        raise ValueError("lr 必须 > 0")
    if args.generated_dir and not os.path.isdir(args.generated_dir):
        raise FileNotFoundError(args.generated_dir)


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _build_dataset(args, path, split):
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


def _build_loader(args, dataset, shuffle, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=ntu_label_forecasting_collate,
        drop_last=False,
        worker_init_fn=_seed_worker if int(args.num_workers) > 0 else None,
    )


def _compute_train_future_normalizer(args, dataset):
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ntu_label_forecasting_collate,
        drop_last=False,
    )
    sum_x = torch.zeros(NTU_JOINTS, NTU_FEATS, dtype=torch.float64)
    sum_x2 = torch.zeros(NTU_JOINTS, NTU_FEATS, dtype=torch.float64)
    count = 0

    for batch in loader:
        future = batch["future"].double()
        _ensure_finite("normalizer future", future)
        sum_x = sum_x + future.sum(dim=(0, 3))
        sum_x2 = sum_x2 + (future * future).sum(dim=(0, 3))
        count += int(future.shape[0]) * int(future.shape[-1])

    if count < 1:
        raise ValueError("normalizer count 为空")

    mean = sum_x / float(count)
    var = (sum_x2 / float(count)) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_min(1e-6)

    return {
        "mean": mean.float().view(1, NTU_JOINTS, NTU_FEATS, 1),
        "std": std.float().view(1, NTU_JOINTS, NTU_FEATS, 1),
        "count": int(count),
    }


def _save_normalizer(args, normalizer):
    path = os.path.join(args.save_dir, "normalizer.pt")
    torch.save(
        {
            "mean": normalizer["mean"].cpu(),
            "std": normalizer["std"].cpu(),
            "count": int(normalizer["count"]),
            "shape": list(normalizer["mean"].shape),
            "created_at": _utc_now(),
            "train_path": args.train_path,
        },
        path,
    )
    return path


def _normalizer_to_device(normalizer, device):
    return {
        "mean": normalizer["mean"].to(device),
        "std": normalizer["std"].to(device),
        "count": int(normalizer["count"]),
    }


def _normalize_future(future, normalizer):
    return (future - normalizer["mean"]) / normalizer["std"].clamp_min(1e-6)


def _label_counts_from_dataset(dataset):
    counts = [0 for _ in range(NUM_ACTIONS)]
    for entry in dataset.entries:
        label = int(entry["action"])
        if label < 0 or label >= NUM_ACTIONS:
            raise ValueError("action label 越界: {}".format(label))
        counts[label] += 1
    return counts


def _class_weights_from_counts(counts, device):
    weights = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    for idx, count in enumerate(counts):
        if int(count) > 0:
            weights[idx] = 1.0 / math.sqrt(float(count))
    positive = weights > 0
    if positive.any():
        weights[positive] = weights[positive] / weights[positive].mean().clamp_min(1e-12)
    return weights.to(device)


def _build_model(args):
    return TemporalCNNActionClassifier(
        input_channels=NTU_JOINTS * NTU_FEATS,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        num_classes=NUM_ACTIONS,
        pred_len=args.pred_len,
    )


def _save_args(args, device, model, train_summary, test_summary, normalizer):
    payload = OrderedDict()
    for key, value in sorted(vars(args).items()):
        payload[key] = value
    payload["device"] = str(device)
    payload["num_params"] = int(_count_parameters(model))
    payload["model_config"] = model.config()
    payload["train_summary"] = train_summary
    payload["test_summary"] = test_summary
    payload["normalizer_count"] = int(normalizer["count"])
    payload["created_at"] = _utc_now()
    _write_json(os.path.join(args.save_dir, "args.json"), payload)


def _save_checkpoint(args, model, step, normalizer_path, real_test_metrics=None):
    path = os.path.join(args.save_dir, "classifier_model.pt")
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "num_classes": NUM_ACTIONS,
            "step": int(step),
            "seed": int(args.seed),
            "train_path": args.train_path,
            "test_path": args.test_path,
            "normalizer_path": normalizer_path,
            "real_test_metrics": real_test_metrics,
            "created_at": _utc_now(),
        },
        path,
    )
    return path


def _train_step(model, optimizer, batch, normalizer, class_weights, args, device):
    model.train()
    future = batch["future"].to(device)
    labels = batch["action"].view(-1).to(device)
    _ensure_finite("train future", future)
    _ensure_finite("train action", labels.float())

    future = _normalize_future(future, normalizer)
    logits = model(future)
    loss = F.cross_entropy(logits, labels, weight=class_weights)
    _ensure_finite("train loss", loss)

    optimizer.zero_grad()
    loss.backward()
    if float(args.clip_grad_norm) > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
    optimizer.step()

    preds = logits.argmax(dim=1)
    top1_acc = (preds == labels).float().mean()
    return {
        "train_loss": float(loss.detach().cpu().item()),
        "train_top1_acc": float(top1_acc.detach().cpu().item()),
    }


def _topk_labels_and_probs(logits, k=5):
    k = min(int(k), int(logits.shape[1]))
    probs = torch.softmax(logits, dim=1)
    top_probs, top_labels = torch.topk(probs, k=k, dim=1)
    return top_labels, top_probs


def _make_prediction_record(index, meta, target, pred, top_labels, top_probs):
    return OrderedDict(
        [
            ("index", int(index)),
            ("sample_id", meta.get("sample_id")),
            ("start", int(meta.get("start", -1))),
            ("length", int(meta.get("length", -1))),
            ("target_label", int(target)),
            ("target_action_code", _action_code(target)),
            ("predicted_label", int(pred)),
            ("predicted_action_code", _action_code(pred)),
            ("top5_labels", [int(item) for item in top_labels]),
            ("top5_action_codes", [_action_code(item) for item in top_labels]),
            ("top5_probs", [float(item) for item in top_probs]),
        ]
    )


def _metrics_from_counts(
    loss_sum,
    total,
    top1_correct,
    top5_correct,
    class_correct,
    class_count,
):
    if int(total) < 1:
        raise ValueError("eval total 为空")

    per_class_acc = []
    valid_acc = []
    for idx in range(NUM_ACTIONS):
        count = int(class_count[idx])
        if count > 0:
            acc = float(class_correct[idx]) / float(count)
            valid_acc.append(acc)
            per_class_acc.append(acc)
        else:
            per_class_acc.append(None)

    handshaking_acc = per_class_acc[HANDSHAKING_LABEL]
    top1_acc = float(top1_correct) / float(total)
    top5_acc = float(top5_correct) / float(total)
    balanced_acc = sum(valid_acc) / float(len(valid_acc)) if valid_acc else 0.0
    classifier_gate_pass = (
        top1_acc >= GATE_THRESHOLDS["top1_acc"]
        and top5_acc >= GATE_THRESHOLDS["top5_acc"]
        and handshaking_acc is not None
        and handshaking_acc >= GATE_THRESHOLDS["handshaking_acc"]
    )

    return OrderedDict(
        [
            ("top1_acc", top1_acc),
            ("top5_acc", top5_acc),
            ("balanced_acc", balanced_acc),
            ("per_class_acc", per_class_acc),
            ("per_class_count", [int(item) for item in class_count]),
            ("handshaking_acc", handshaking_acc),
            ("top1_random", 1.0 / float(NUM_ACTIONS)),
            ("top5_random", 5.0 / float(NUM_ACTIONS)),
            ("classifier_gate_pass", bool(classifier_gate_pass)),
            ("gate_thresholds", GATE_THRESHOLDS),
            ("loss", float(loss_sum) / float(total)),
            ("num_test_samples", int(total)),
        ]
    )


@torch.no_grad()
def _evaluate_real_test(
    model,
    loader,
    normalizer,
    device,
    prediction_path=None,
    confusion_path=None,
):
    if prediction_path and os.path.exists(prediction_path):
        os.remove(prediction_path)

    model.eval()
    loss_sum = 0.0
    total = 0
    top1_correct = 0
    top5_correct = 0
    class_correct = np.zeros(NUM_ACTIONS, dtype=np.int64)
    class_count = np.zeros(NUM_ACTIONS, dtype=np.int64)
    confusion = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=np.int64)

    for batch in loader:
        future = batch["future"].to(device)
        labels = batch["action"].view(-1).to(device)
        _ensure_finite("eval future", future)
        future = _normalize_future(future, normalizer)
        logits = model(future)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        _ensure_finite("eval loss", loss)

        top_labels, top_probs = _topk_labels_and_probs(logits, k=5)
        preds = top_labels[:, 0]
        labels_cpu = labels.detach().cpu().numpy()
        preds_cpu = preds.detach().cpu().numpy()
        top_labels_cpu = top_labels.detach().cpu().numpy()
        top_probs_cpu = top_probs.detach().cpu().numpy()

        batch_size = int(labels.shape[0])
        loss_sum += float(loss.detach().cpu().item())
        total += batch_size
        top1_correct += int((preds == labels).sum().detach().cpu().item())
        top5_correct += int((top_labels == labels.unsqueeze(1)).any(dim=1).sum().detach().cpu().item())

        for row_idx in range(batch_size):
            target = int(labels_cpu[row_idx])
            pred = int(preds_cpu[row_idx])
            class_count[target] += 1
            if pred == target:
                class_correct[target] += 1
            confusion[target, pred] += 1
            if prediction_path:
                record = _make_prediction_record(
                    total - batch_size + row_idx,
                    batch["meta"][row_idx],
                    target,
                    pred,
                    top_labels_cpu[row_idx],
                    top_probs_cpu[row_idx],
                )
                _append_jsonl(prediction_path, record)

    metrics = _metrics_from_counts(
        loss_sum,
        total,
        top1_correct,
        top5_correct,
        class_correct,
        class_count,
    )
    if confusion_path:
        np.save(confusion_path, confusion)
    return metrics


def _next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _label_count_list(labels):
    counts = [0 for _ in range(NUM_ACTIONS)]
    for label in labels:
        label = int(label)
        if label < 0 or label >= NUM_ACTIONS:
            raise ValueError("label 越界: {}".format(label))
        counts[label] += 1
    return counts


def _load_generated(generated_dir, pred_len):
    future_path = os.path.join(generated_dir, "generated_future40.npy")
    metadata_path = os.path.join(generated_dir, "metadata.json")
    if not os.path.exists(future_path):
        raise FileNotFoundError(future_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(metadata_path)

    generated = np.load(future_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    missing = [
        key
        for key in ("labels", "generated_shape", "checkpoint")
        if key not in metadata
    ]
    if missing:
        raise ValueError("metadata 缺少字段: {}".format(missing))
    if list(generated.shape) != list(metadata["generated_shape"]):
        raise ValueError(
            "generated shape={} 与 metadata generated_shape={} 不一致".format(
                list(generated.shape), metadata["generated_shape"]
            )
        )
    if generated.ndim != 6:
        raise ValueError("generated_future40 必须是 [case,label,rep,56,6,T]")
    if tuple(generated.shape[-3:]) != (NTU_JOINTS, NTU_FEATS, int(pred_len)):
        raise ValueError(
            "generated future shape 后三维必须是 [{},{},{}]，当前为 {}".format(
                NTU_JOINTS, NTU_FEATS, int(pred_len), tuple(generated.shape[-3:])
            )
        )
    _ensure_finite("generated_future40", generated)
    return generated.astype(np.float32), metadata


def _generated_condition_labels(generated, metadata):
    case_count, label_count, rep_count = generated.shape[:3]
    flat_count = int(case_count) * int(label_count) * int(rep_count)
    labels = [int(item) for item in metadata["labels"]]

    if int(label_count) == len(labels):
        label_grid = np.asarray(labels, dtype=np.int64).reshape(1, label_count, 1)
        return np.broadcast_to(label_grid, (case_count, label_count, rep_count)).reshape(-1)

    batch_meta = metadata.get("batch_meta", [])
    if len(batch_meta) == flat_count:
        return np.asarray([int(item["label"]) for item in batch_meta], dtype=np.int64)

    raise ValueError("无法从 generated shape 和 metadata 推断 condition labels")


def _generated_index_meta(index, generated, metadata):
    case_count, label_count, rep_count = generated.shape[:3]
    case_idx, label_idx, rep_idx = np.unravel_index(index, (case_count, label_count, rep_count))
    record = OrderedDict(
        [
            ("generated_index", int(index)),
            ("case_index", int(case_idx)),
            ("label_index", int(label_idx)),
            ("repetition", int(rep_idx)),
        ]
    )
    batch_meta = metadata.get("batch_meta", [])
    if len(batch_meta) == int(case_count) * int(label_count) * int(rep_count):
        record["batch_meta"] = batch_meta[int(index)]
    return record


@torch.no_grad()
def _evaluate_generated_consistency(
    args,
    model,
    normalizer,
    device,
    real_test_metrics,
):
    generated, metadata = _load_generated(args.generated_dir, args.pred_len)
    condition_labels = _generated_condition_labels(generated, metadata)
    flat_future = generated.reshape(-1, NTU_JOINTS, NTU_FEATS, int(args.pred_len))
    valid_for_claim = bool(real_test_metrics.get("classifier_gate_pass", False))

    prediction_path = os.path.join(args.save_dir, "generated_predictions.jsonl")
    if os.path.exists(prediction_path):
        os.remove(prediction_path)

    model.eval()
    predicted = []
    top1_confidences = []
    for start in range(0, int(flat_future.shape[0]), int(args.eval_batch_size)):
        stop = min(start + int(args.eval_batch_size), int(flat_future.shape[0]))
        future = torch.from_numpy(flat_future[start:stop]).float().to(device)
        future = _normalize_future(future, normalizer)
        logits = model(future)
        top_labels, top_probs = _topk_labels_and_probs(logits, k=5)
        top_labels_cpu = top_labels.detach().cpu().numpy()
        top_probs_cpu = top_probs.detach().cpu().numpy()

        for local_idx in range(stop - start):
            index = start + local_idx
            pred = int(top_labels_cpu[local_idx, 0])
            predicted.append(pred)
            top1_confidences.append(float(top_probs_cpu[local_idx, 0]))
            record = _generated_index_meta(index, generated, metadata)
            condition = int(condition_labels[index])
            record.update(
                OrderedDict(
                    [
                        ("condition_label", condition),
                        ("condition_action_code", _action_code(condition)),
                        ("predicted_label", pred),
                        ("predicted_action_code", _action_code(pred)),
                        ("top5_labels", [int(item) for item in top_labels_cpu[local_idx]]),
                        (
                            "top5_action_codes",
                            [_action_code(item) for item in top_labels_cpu[local_idx]],
                        ),
                        ("top5_probs", [float(item) for item in top_probs_cpu[local_idx]]),
                        ("valid_for_claim", valid_for_claim),
                    ]
                )
            )
            _append_jsonl(prediction_path, record)

    predicted = np.asarray(predicted, dtype=np.int64)
    correct = predicted == condition_labels
    per_label_consistency = []
    for label in range(NUM_ACTIONS):
        mask = condition_labels == label
        if np.any(mask):
            per_label_consistency.append(float(correct[mask].mean()))
        else:
            per_label_consistency.append(None)

    consistency = OrderedDict(
        [
            ("generated_dir", args.generated_dir),
            ("classifier_checkpoint", os.path.join(args.save_dir, "classifier_model.pt")),
            ("classifier_gate_pass", bool(real_test_metrics.get("classifier_gate_pass", False))),
            ("valid_for_claim", valid_for_claim),
            ("labels", [int(item) for item in metadata["labels"]]),
            ("consistency_acc", float(correct.mean()) if len(correct) > 0 else 0.0),
            ("per_label_consistency_acc", per_label_consistency),
            ("predicted_label_counts", _label_count_list(predicted)),
            ("condition_label_counts", _label_count_list(condition_labels)),
            ("num_generated_samples", int(flat_future.shape[0])),
            (
                "top1_confidence_mean",
                float(np.mean(top1_confidences)) if top1_confidences else 0.0,
            ),
            ("metadata_checkpoint", metadata.get("checkpoint")),
        ]
    )
    _write_json(os.path.join(args.save_dir, "generated_consistency.json"), consistency)
    return consistency


def _make_train_log_record(step, train_metrics, eval_metrics, args):
    record = OrderedDict()
    record["step"] = int(step)
    for key, value in train_metrics.items():
        record[key] = value
    if eval_metrics is not None:
        record["eval_top1_acc"] = eval_metrics["top1_acc"]
        record["eval_top5_acc"] = eval_metrics["top5_acc"]
        record["eval_balanced_acc"] = eval_metrics["balanced_acc"]
        record["eval_handshaking_acc"] = eval_metrics["handshaking_acc"]
        record["eval_classifier_gate_pass"] = eval_metrics["classifier_gate_pass"]
    record["lr"] = float(args.lr)
    record["batch_size"] = int(args.batch_size)
    record["seed"] = int(args.seed)
    record["created_at"] = _utc_now()
    return record


def run_classifier_gate(args):
    _validate_args(args)
    fixseed(args.seed)
    _prepare_save_dir(args)
    device = _device()

    stats_dataset = _build_dataset(args, args.train_path, split="train")
    normalizer = _compute_train_future_normalizer(args, stats_dataset)
    normalizer_path = _save_normalizer(args, normalizer)

    train_dataset = _build_dataset(args, args.train_path, split="train")
    test_dataset = _build_dataset(args, args.test_path, split="test")
    train_summary = summarize_entries(train_dataset.entries)
    test_summary = summarize_entries(test_dataset.entries)

    train_loader = _build_loader(args, train_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = _build_loader(args, test_dataset, shuffle=False, batch_size=args.eval_batch_size)

    model = _build_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_counts = _label_counts_from_dataset(train_dataset)
    class_weights = _class_weights_from_counts(train_counts, device)
    normalizer_device = _normalizer_to_device(normalizer, device)

    _save_args(args, device, model, train_summary, test_summary, normalizer)

    print(
        "Training action consistency classifier: params={} device={} train={} test={}".format(
            _count_parameters(model), device, len(train_dataset), len(test_dataset)
        )
    )

    iterator = iter(train_loader)
    latest_checkpoint = None
    for step in range(1, int(args.num_steps) + 1):
        batch, iterator = _next_batch(train_loader, iterator)
        train_metrics = _train_step(
            model,
            optimizer,
            batch,
            normalizer_device,
            class_weights,
            args,
            device,
        )

        eval_metrics = None
        if int(args.eval_interval) > 0 and step % int(args.eval_interval) == 0:
            eval_metrics = _evaluate_real_test(
                model,
                test_loader,
                normalizer_device,
                device,
                prediction_path=None,
                confusion_path=None,
            )

        if step == 1 or step % int(args.log_interval) == 0:
            message = "step[{}]: train_loss[{:.6f}] train_top1[{:.4f}]".format(
                step,
                train_metrics["train_loss"],
                train_metrics["train_top1_acc"],
            )
            if eval_metrics is not None:
                message += " eval_top1[{:.4f}] eval_top5[{:.4f}]".format(
                    eval_metrics["top1_acc"],
                    eval_metrics["top5_acc"],
                )
            print(message)

        if step % int(args.save_interval) == 0 or step == int(args.num_steps):
            latest_checkpoint = _save_checkpoint(
                args,
                model,
                step,
                normalizer_path,
                real_test_metrics=None,
            )

        record = _make_train_log_record(step, train_metrics, eval_metrics, args)
        if latest_checkpoint is not None:
            record["checkpoint"] = latest_checkpoint
        _append_jsonl(os.path.join(args.save_dir, "train_log.jsonl"), record)

    real_metrics = _evaluate_real_test(
        model,
        test_loader,
        normalizer_device,
        device,
        prediction_path=os.path.join(args.save_dir, "real_test_predictions.jsonl"),
        confusion_path=os.path.join(args.save_dir, "confusion_matrix.npy"),
    )
    _write_json(os.path.join(args.save_dir, "real_test_metrics.json"), real_metrics)
    latest_checkpoint = _save_checkpoint(
        args,
        model,
        int(args.num_steps),
        normalizer_path,
        real_test_metrics=real_metrics,
    )

    generated_metrics = None
    if args.generated_dir:
        generated_metrics = _evaluate_generated_consistency(
            args,
            model,
            normalizer_device,
            device,
            real_metrics,
        )

    print(
        "Finished. checkpoint={} top1={:.4f} top5={:.4f} handshaking={} gate_pass={}".format(
            latest_checkpoint,
            real_metrics["top1_acc"],
            real_metrics["top5_acc"],
            real_metrics["handshaking_acc"],
            real_metrics["classifier_gate_pass"],
        )
    )
    return {
        "checkpoint": latest_checkpoint,
        "real_test_metrics": real_metrics,
        "generated_consistency": generated_metrics,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument(
        "--train_path",
        default="dataset/ntu120/smplx/conditioned/xsub.train.h5",
    )
    parser.add_argument(
        "--test_path",
        default="dataset/ntu120/smplx/conditioned/xsub.test.h5",
    )
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--generated_dir", default=None)
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_classifier_gate(args)


if __name__ == "__main__":
    main()
