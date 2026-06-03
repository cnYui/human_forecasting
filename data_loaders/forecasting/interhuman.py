import random
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset, get_worker_info

from utils.forecasting_motion import extract_active_motion


class InterHumanForecastDataset(Dataset):
    def __init__(
        self,
        data_path,
        split,
        window_len=150,
        obs_len=30,
        pred_len=120,
        max_samples=-1,
        seed=0,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("split 必须是 train/val/test，当前为 {}".format(split))
        if obs_len + pred_len != window_len:
            raise ValueError("obs_len + pred_len 必须等于 window_len")

        self.data_path = data_path
        self.split = split
        self.window_len = int(window_len)
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.max_samples = int(max_samples) if max_samples is not None else -1
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._h5_handle = None
        self.h5_path = self._resolve_h5_path(data_path, split)
        self.entries = self._read_entries()

    def _resolve_h5_path(self, data_path, split):
        path = Path(data_path)
        if path.suffix == ".h5":
            return path
        return path / "interhuman_{}.h5".format(split)

    def _read_entries(self):
        if not self.h5_path.exists():
            raise FileNotFoundError(str(self.h5_path))

        entries = []
        with h5py.File(str(self.h5_path), "r") as h5:
            for key in h5.keys():
                length = int(h5[key].shape[0])
                if length >= self.window_len:
                    entries.append({"sample_id": str(key), "length": length})

        if self.max_samples is not None and self.max_samples > 0:
            entries = entries[: self.max_samples]
        return entries

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
        active = extract_active_motion(motion)
        obs = active[: self.obs_len].contiguous()
        target = active[self.obs_len :].contiguous()

        return {
            "obs": obs,
            "target": target,
            "sample_id": sample_id,
            "start": int(start),
            "length": length,
        }
