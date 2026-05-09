import json
import os
import pickle
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from .dataset import Dataset


class InterHuman(Dataset):
    def __init__(self, datapath, **kwargs):
        self.data_path = datapath or "dataset/interhuman"
        self.max_samples = kwargs.get("max_samples", -1)
        super().__init__(**kwargs)
        self._h5_handles = {}
        self._h5_mode = self._is_h5_path(self.data_path)
        if self._h5_mode:
            self._init_h5()
        else:
            self._init_pkl()

    def _is_h5_path(self, datapath):
        path = Path(datapath)
        return path.suffix == ".h5" or (path / "interhuman_train.h5").exists()

    def _init_common_labels(self):
        self._action_to_label = {0: 0}
        self._label_to_action = {0: 0}
        self._action_classes = {0: "interaction"}
        self.num_actions = 1

    def _init_h5(self):
        self._h5_paths = self._resolve_h5_paths(self.data_path)
        eval_split = "val" if self.split == "val" else "test"
        train_entries = [("train", key) for key in self._read_h5_keys("train")]
        eval_entries = [(eval_split, key) for key in self._read_h5_keys(eval_split)]

        self.keys = [key for _, key in train_entries + eval_entries]
        self._h5_entries = train_entries + eval_entries

        n_train = len(train_entries)
        self._train = np.arange(n_train)
        self._test = np.arange(n_train, len(self._h5_entries))

        if self.max_samples is not None and self.max_samples > 0:
            self._train = self._train[:self.max_samples]
            self._test = self._test[:self.max_samples]

        self._train = self._train[self.shard:][::self.num_shards]
        self._num_frames_in_video = {
            i: self._get_h5_frames(i) for i in range(len(self._h5_entries))
        }
        self._actions = {i: 0 for i in range(len(self._h5_entries))}
        self._init_common_labels()

    def _resolve_h5_paths(self, datapath):
        path = Path(datapath)
        if path.suffix == ".h5":
            directory = path.parent
        else:
            directory = path

        paths = {
            split: directory / f"interhuman_{split}.h5"
            for split in ("train", "val", "test")
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing InterHuman H5 files: " + ", ".join(missing))
        return paths

    def _read_h5_keys(self, split):
        with h5py.File(self._h5_paths[split], "r") as h5:
            return list(h5.keys())

    def _get_h5_frames(self, data_index):
        split, key = self._h5_entries[data_index]
        with h5py.File(self._h5_paths[split], "r") as h5:
            return int(h5[key].shape[0])

    def _get_h5_handle(self, split):
        if split not in self._h5_handles:
            self._h5_handles[split] = h5py.File(self._h5_paths[split], "r")
        return self._h5_handles[split]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        return state

    def _sample_frame_indices(self, nframes):
        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            return np.arange(nframes)

        if self.num_frames == -2:
            if self.min_len <= 0:
                raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
            max_frame = min(nframes, self.max_len) if self.max_len != -1 else nframes
            num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
        else:
            num_frames = self.num_frames if self.num_frames != -1 else self.max_len

        if num_frames > nframes:
            ntoadd = max(0, num_frames - nframes)
            lastframe = nframes - 1
            padding = lastframe * np.ones(ntoadd, dtype=int)
            return np.concatenate((np.arange(0, nframes), padding))

        if self.sampling in ["conseq", "random_conseq"]:
            step_max = (nframes - 1) // (num_frames - 1)
            if self.sampling == "conseq":
                if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                    step = step_max
                else:
                    step = self.sampling_step
            else:
                step = random.randint(1, step_max)

            lastone = step * (num_frames - 1)
            shift_max = nframes - lastone - 1
            shift = random.randint(0, max(0, shift_max - 1))
            return shift + np.arange(0, lastone + 1, step)

        if self.sampling == "random":
            choices = np.random.choice(range(nframes), num_frames, replace=False)
            return sorted(choices)

        raise ValueError("Sampling not recognized.")

    def _load_h5(self, data_index, frame_ix):
        split, key = self._h5_entries[data_index]
        value = self._get_h5_handle(split)[key][:][frame_ix].astype("float32")
        return torch.from_numpy(value).permute(1, 2, 0).contiguous().float()

    def _get_item_data_index(self, data_index):
        if not self._h5_mode:
            return super()._get_item_data_index(data_index)

        nframes = self._num_frames_in_video[data_index]
        frame_ix = self._sample_frame_indices(nframes)
        output = {
            "inp": self._load_h5(data_index, frame_ix),
            "action": self.get_label(data_index),
            "action_text": self.action_to_action_name(self.get_action(data_index)),
            "sample_id": self.keys[data_index],
        }
        return output

    def _init_pkl(self):
        self.motion_dir = os.path.join(self.data_path, "motions")
        self.label_path = os.path.join(
            self.data_path, "annotations_interhuman", "interhuman_label.json"
        )
        split_dir = os.path.join(self.data_path, "split")

        with open(self.label_path, "r") as f:
            self._ar_labels = {str(k): int(v) for k, v in json.load(f).items()}

        with open(os.path.join(split_dir, "train.txt"), "r") as f:
            train_ids = [line.strip() for line in f if line.strip()]
        with open(os.path.join(split_dir, "test.txt"), "r") as f:
            test_ids = [line.strip() for line in f if line.strip()]
        if self.split == "val":
            with open(os.path.join(split_dir, "val.txt"), "r") as f:
                test_ids = [line.strip() for line in f if line.strip()]

        self.keys = self._valid_ids(train_ids) + self._valid_ids(test_ids)
        n_train = len(self._valid_ids(train_ids))
        self._train = np.arange(n_train)
        self._test = np.arange(n_train, len(self.keys))

        if self.max_samples is not None and self.max_samples > 0:
            self._train = self._train[:self.max_samples]
            self._test = self._test[:self.max_samples]

        self._train = self._train[self.shard:][::self.num_shards]
        self._cache = {}
        self._num_frames_in_video = {
            i: self._get_frames(self.keys[i]) for i in range(len(self.keys))
        }
        self._actions = {i: 0 for i in range(len(self.keys))}
        self._init_common_labels()

    def _valid_ids(self, ids):
        valid = []
        for sid in ids:
            path = os.path.join(self.motion_dir, f"{sid}.pkl")
            if sid not in self._ar_labels or not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                item = pickle.load(f)
            if int(item.get("frames", 0)) <= 0:
                continue
            valid.append(sid)
        return valid

    def _load_item(self, sid):
        if sid not in self._cache:
            with open(os.path.join(self.motion_dir, f"{sid}.pkl"), "rb") as f:
                self._cache[sid] = pickle.load(f)
        return self._cache[sid]

    def _get_frames(self, sid):
        return int(self._load_item(sid)["frames"])

    def _person_pose(self, person):
        root = person["root_orient"]
        body = person["pose_body"]
        # InterHuman stores root + 21 SMPL body joints. The local SMPL layer
        # expects 24 rotations including root, so pad the two missing joints.
        pad = np.zeros((root.shape[0], 6), dtype=root.dtype)
        return np.concatenate([root, body, pad], axis=1).reshape(root.shape[0], 24, 3)

    def _ordered_people(self, ind):
        sid = self.keys[ind]
        item = self._load_item(sid)
        p1, p2 = item["person1"], item["person2"]
        if self._ar_labels[sid] == 1:
            p1, p2 = p2, p1
        return p1, p2

    def _load_joints3D(self, ind, frame_ix):
        actor, reactor = self._ordered_people(ind)
        trans = np.concatenate([actor["trans"], reactor["trans"]], axis=1)
        return trans[frame_ix, None, :].astype("float32")

    def _load_rotvec(self, ind, frame_ix):
        actor, reactor = self._ordered_people(ind)
        pose = np.concatenate(
            [self._person_pose(actor), self._person_pose(reactor)], axis=2
        )
        return pose[frame_ix].astype("float32")
