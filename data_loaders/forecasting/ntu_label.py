import random
import re
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset, get_worker_info


NUM_ACTIONS = 26
HANDSHAKING_LABEL = 8
NTU_JOINTS = 56
NTU_FEATS = 6


def parse_ntu_action_label(sample_id):
    matches = re.findall(r"A(\d{3})", str(sample_id))
    if len(matches) != 1:
        raise ValueError(
            "sample_id 必须包含唯一 Axxx 动作标签，当前为 {}".format(sample_id)
        )

    action_number = int(matches[0])
    if action_number < 1 or action_number > NUM_ACTIONS:
        raise ValueError(
            "sample_id 的动作标签必须在 A001-A026 内，当前为 {}".format(sample_id)
        )

    action_code = "A{:03d}".format(action_number)
    return {
        "action": action_number - 1,
        "action_code": action_code,
        "action_number": action_number,
    }


def _shape_is_valid(shape, num_joints=NTU_JOINTS, num_feats=NTU_FEATS):
    return len(shape) == 3 and int(shape[1]) == num_joints and int(shape[2]) == num_feats


def scan_ntu_label_forecasting_entries(
    h5_path,
    window_len=60,
    num_joints=NTU_JOINTS,
    num_feats=NTU_FEATS,
    strict=True,
):
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(str(h5_path))

    raw_count = 0
    skipped_too_short = 0
    skipped_invalid = []
    entries = []

    with h5py.File(str(h5_path), "r") as h5:
        for key in sorted(h5.keys()):
            raw_count += 1
            sample_id = str(key)
            shape = h5[key].shape

            try:
                label_info = parse_ntu_action_label(sample_id)
                if not _shape_is_valid(shape, num_joints=num_joints, num_feats=num_feats):
                    raise ValueError(
                        "H5 item shape 必须为 [T,{},{}]，{} 当前为 {}".format(
                            num_joints, num_feats, sample_id, tuple(shape)
                        )
                    )
            except ValueError as exc:
                if strict:
                    raise
                skipped_invalid.append({"sample_id": sample_id, "reason": str(exc)})
                continue

            length = int(shape[0])
            if length < int(window_len):
                skipped_too_short += 1
                continue

            entries.append(
                {
                    "sample_id": sample_id,
                    "length": length,
                    "action": int(label_info["action"]),
                    "action_code": label_info["action_code"],
                    "action_name": label_info["action_code"],
                }
            )

    return {
        "h5_path": str(h5_path),
        "raw_count": raw_count,
        "kept_count": len(entries),
        "skipped_too_short": skipped_too_short,
        "skipped_invalid": skipped_invalid,
        "entries": entries,
    }


def summarize_entries(scan_result, num_actions=NUM_ACTIONS):
    if isinstance(scan_result, dict):
        entries = scan_result.get("entries", [])
        raw_count = int(scan_result.get("raw_count", len(entries)))
        skipped_too_short = int(scan_result.get("skipped_too_short", 0))
        skipped_invalid = scan_result.get("skipped_invalid", [])
    else:
        entries = scan_result
        raw_count = len(entries)
        skipped_too_short = 0
        skipped_invalid = []

    label_counts = [0 for _ in range(num_actions)]
    lengths = []
    for entry in entries:
        action = int(entry["action"])
        if action < 0 or action >= num_actions:
            raise ValueError("action 必须在 [0,{}] 内，当前为 {}".format(num_actions - 1, action))
        label_counts[action] += 1
        lengths.append(int(entry["length"]))

    covered_labels = [idx for idx, count in enumerate(label_counts) if count > 0]
    missing_labels = [idx for idx, count in enumerate(label_counts) if count == 0]
    length_mean = sum(lengths) / float(len(lengths)) if lengths else 0.0

    return {
        "raw_count": raw_count,
        "kept_count": len(entries),
        "skipped_too_short": skipped_too_short,
        "skipped_invalid_count": len(skipped_invalid),
        "label_counts": label_counts,
        "covered_labels": covered_labels,
        "missing_labels": missing_labels,
        "min_class_count": min(label_counts) if label_counts else 0,
        "handshaking_count": label_counts[HANDSHAKING_LABEL],
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": length_mean,
    }


class NTULabelForecastDataset(Dataset):
    def __init__(
        self,
        h5_path,
        split,
        window_len=60,
        obs_len=20,
        pred_len=40,
        max_samples=-1,
        seed=0,
        strict=True,
    ):
        if split not in ("train", "test"):
            raise ValueError("split 必须是 train/test，当前为 {}".format(split))
        if int(obs_len) + int(pred_len) != int(window_len):
            raise ValueError("obs_len + pred_len 必须等于 window_len")

        self.h5_path = Path(h5_path)
        self.split = split
        self.window_len = int(window_len)
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.max_samples = int(max_samples) if max_samples is not None else -1
        self.seed = int(seed)
        self.strict = bool(strict)
        self._rng = random.Random(self.seed)
        self._h5_handle = None

        self.scan_result = scan_ntu_label_forecasting_entries(
            self.h5_path,
            window_len=self.window_len,
            strict=self.strict,
        )
        self.entries = list(self.scan_result["entries"])
        if self.max_samples is not None and self.max_samples > 0:
            self.entries = self.entries[: self.max_samples]
        if len(self.entries) == 0:
            raise ValueError("T >= {} 过滤后数据为空: {}".format(self.window_len, self.h5_path))

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handle"] = None
        return state

    def close(self):
        if self._h5_handle is not None:
            self._h5_handle.close()
            self._h5_handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_h5_handle(self):
        if self._h5_handle is None:
            self._h5_handle = h5py.File(str(self.h5_path), "r")
        return self._h5_handle

    def _sample_start(self, length):
        max_start = int(length) - self.window_len
        if max_start < 0:
            raise ValueError("length 小于 window_len")
        if self.split == "train":
            if get_worker_info() is not None:
                return random.randint(0, max_start)
            return self._rng.randint(0, max_start)
        return max_start // 2

    def __getitem__(self, index):
        entry = self.entries[index]
        sample_id = entry["sample_id"]
        length = int(entry["length"])
        start = self._sample_start(length)
        stop = start + self.window_len

        motion = self._get_h5_handle()[sample_id][start:stop]
        motion = torch.from_numpy(motion).float()
        if tuple(motion.shape) != (self.window_len, NTU_JOINTS, NTU_FEATS):
            raise ValueError(
                "窗口 shape 必须为 [{},{},{}]，{} 当前为 {}".format(
                    self.window_len, NTU_JOINTS, NTU_FEATS, sample_id, tuple(motion.shape)
                )
            )

        motion = motion.permute(1, 2, 0).contiguous()
        obs_motion = motion[:, :, : self.obs_len].contiguous()
        future = motion[:, :, self.obs_len :].contiguous()
        mask = torch.ones(1, 1, self.pred_len, dtype=torch.bool)

        return {
            "obs_motion": obs_motion,
            "future": future,
            "action": torch.tensor([int(entry["action"])], dtype=torch.long),
            "mask": mask,
            "length": length,
            "start": int(start),
            "sample_id": sample_id,
            "action_code": entry["action_code"],
            "action_name": entry["action_name"],
        }


def ntu_label_forecasting_collate(batch):
    not_none = [item for item in batch if item is not None]
    if len(not_none) == 0:
        raise ValueError("batch 为空")

    obs_motion = torch.stack([item["obs_motion"] for item in not_none], dim=0).float()
    future = torch.stack([item["future"] for item in not_none], dim=0).float()
    action = torch.stack([item["action"] for item in not_none], dim=0).long()
    mask = torch.stack([item["mask"] for item in not_none], dim=0).bool()
    lengths = torch.as_tensor([int(item["length"]) for item in not_none], dtype=torch.long)

    meta = []
    for item in not_none:
        meta.append(
            {
                "sample_id": item["sample_id"],
                "start": int(item["start"]),
                "length": int(item["length"]),
                "action": int(item["action"].item()),
                "action_code": item["action_code"],
                "action_name": item["action_name"],
            }
        )

    return {
        "obs_motion": obs_motion,
        "future": future,
        "action": action,
        "mask": mask,
        "lengths": lengths,
        "meta": meta,
    }
