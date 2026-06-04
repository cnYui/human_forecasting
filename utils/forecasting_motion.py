import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import torch

from utils.rotation_conversions import rotation_6d_to_matrix


NUM_PERSONS = 2
NUM_BODY_JOINTS = 24
ROT6D_DIM = 6
ROT_DIM = NUM_BODY_JOINTS * ROT6D_DIM
TRANSL_DIM = 3
PERSON_DIM = ROT_DIM + TRANSL_DIM
H5_JOINTS = 25
H5_CHANNELS = 12
EPS = 1e-6
RELATION_FEATURE_DIM = 16
RELATION_FEATURE_SETS = ("all", "translation", "velocity", "orientation")
RELATION_FEATURE_DIMS = {
    "all": 16,
    "translation": 3,
    "velocity": 3,
    "orientation": 9,
}
RELATION_FEATURE_NAMES = (
    "relative_root_translation",
    "relative_root_velocity",
    "root_distance",
    "relative_root_orientation",
)
RELATION_FEATURE_NAMES_BY_SET = {
    "all": RELATION_FEATURE_NAMES,
    "translation": ("relative_root_translation",),
    "velocity": ("relative_root_velocity",),
    "orientation": ("relative_root_orientation",),
}


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _check_motion_h5_shape(motion):
    if motion.dim() != 3:
        raise ValueError("motion_h5 必须是 [T,25,12]，当前维度数为 {}".format(motion.dim()))
    if motion.shape[1] != H5_JOINTS or motion.shape[2] != H5_CHANNELS:
        raise ValueError(
            "motion_h5 必须是 [T,25,12]，当前 shape 为 {}".format(tuple(motion.shape))
        )


def _check_active_shape(active):
    if active.dim() != 3:
        raise ValueError("active 必须是 [T,2,147]，当前维度数为 {}".format(active.dim()))
    if active.shape[1] != NUM_PERSONS or active.shape[2] != PERSON_DIM:
        raise ValueError(
            "active 必须是 [T,2,147]，当前 shape 为 {}".format(tuple(active.shape))
        )


def _check_forecasting_obs_shape(obs):
    if obs.dim() != 4:
        raise ValueError("obs 必须是 [B,T,2,147]，当前维度数为 {}".format(obs.dim()))
    if obs.shape[2] != NUM_PERSONS or obs.shape[3] != PERSON_DIM:
        raise ValueError("obs 必须是 [B,T,2,147]，当前 shape 为 {}".format(tuple(obs.shape)))
    if not torch.isfinite(obs).all():
        raise ValueError("obs 存在非有限数值")


def relation_feature_dim(feature_set):
    if feature_set not in RELATION_FEATURE_DIMS:
        raise ValueError(
            "relation_feature_set 必须是 {}，当前为 {}".format(
                RELATION_FEATURE_SETS, feature_set
            )
        )
    return int(RELATION_FEATURE_DIMS[feature_set])


def relation_feature_names(feature_set):
    if feature_set not in RELATION_FEATURE_NAMES_BY_SET:
        raise ValueError(
            "relation_feature_set 必须是 {}，当前为 {}".format(
                RELATION_FEATURE_SETS, feature_set
            )
        )
    return tuple(RELATION_FEATURE_NAMES_BY_SET[feature_set])


def extract_active_motion(motion_h5):
    motion = _as_tensor(motion_h5)
    _check_motion_h5_shape(motion)
    num_frames = motion.shape[0]

    actor_rot = motion[:, :NUM_BODY_JOINTS, 0:6].reshape(num_frames, ROT_DIM)
    actor_trans = motion[:, H5_JOINTS - 1, 0:3]
    reactor_rot = motion[:, :NUM_BODY_JOINTS, 6:12].reshape(num_frames, ROT_DIM)
    reactor_trans = motion[:, H5_JOINTS - 1, 6:9]

    actor = torch.cat([actor_rot, actor_trans], dim=-1)
    reactor = torch.cat([reactor_rot, reactor_trans], dim=-1)
    return torch.stack([actor, reactor], dim=1).contiguous()


def restore_active_motion(active):
    active = _as_tensor(active)
    _check_active_shape(active)
    num_frames = active.shape[0]
    motion = active.new_zeros((num_frames, H5_JOINTS, H5_CHANNELS))

    motion[:, :NUM_BODY_JOINTS, 0:6] = active[:, 0, :ROT_DIM].reshape(
        num_frames, NUM_BODY_JOINTS, ROT6D_DIM
    )
    motion[:, H5_JOINTS - 1, 0:3] = active[:, 0, ROT_DIM:PERSON_DIM]
    motion[:, :NUM_BODY_JOINTS, 6:12] = active[:, 1, :ROT_DIM].reshape(
        num_frames, NUM_BODY_JOINTS, ROT6D_DIM
    )
    motion[:, H5_JOINTS - 1, 6:9] = active[:, 1, ROT_DIM:PERSON_DIM]
    return motion


def extract_relation_features(obs, feature_set="all"):
    obs = _as_tensor(obs)
    _check_forecasting_obs_shape(obs)
    if feature_set not in RELATION_FEATURE_SETS:
        raise ValueError(
            "relation_feature_set 必须是 {}，当前为 {}".format(
                RELATION_FEATURE_SETS, feature_set
            )
        )

    batch_size = obs.shape[0]
    num_frames = obs.shape[1]
    translation = obs[..., ROT_DIM:PERSON_DIM]
    rel_translation = translation[:, :, 0] - translation[:, :, 1]

    velocity = translation.new_zeros(translation.shape)
    if num_frames > 1:
        velocity[:, 1:] = translation[:, 1:] - translation[:, :-1]
    rel_velocity = velocity[:, :, 0] - velocity[:, :, 1]

    root_distance = torch.norm(rel_translation, dim=-1, keepdim=True)
    root_rot_a = obs[:, :, 0, :ROT6D_DIM]
    root_rot_b = obs[:, :, 1, :ROT6D_DIM]
    rot_a = rotation_6d_to_matrix(root_rot_a)
    rot_b = rotation_6d_to_matrix(root_rot_b)
    rel_orientation = torch.matmul(rot_a.transpose(-1, -2), rot_b).reshape(
        batch_size, num_frames, 9
    )

    if feature_set == "all":
        features = torch.cat(
            [rel_translation, rel_velocity, root_distance, rel_orientation], dim=-1
        )
    elif feature_set == "translation":
        features = rel_translation
    elif feature_set == "velocity":
        features = rel_velocity
    elif feature_set == "orientation":
        features = rel_orientation
    else:
        raise ValueError("unsupported relation_feature_set: {}".format(feature_set))

    features = features.contiguous()
    expected_dim = relation_feature_dim(feature_set)
    if features.shape[-1] != expected_dim:
        raise ValueError(
            "relation feature dim 应为 {}，实际为 {}".format(
                expected_dim, features.shape[-1]
            )
        )
    if not torch.isfinite(features).all():
        raise ValueError("relation features 存在非有限数值")
    return features


class ForecastingNormalizer(object):
    def __init__(self, mean, std, eps=EPS, metadata=None):
        self.mean = _as_tensor(mean)
        self.std = _as_tensor(std)
        self.eps = float(eps)
        self.metadata = metadata or {}
        self._check_shape()

    def _check_shape(self):
        expected = (1, 1, NUM_PERSONS, PERSON_DIM)
        if tuple(self.mean.shape) != expected:
            raise ValueError("mean shape 必须是 {}，当前为 {}".format(expected, tuple(self.mean.shape)))
        if tuple(self.std.shape) != expected:
            raise ValueError("std shape 必须是 {}，当前为 {}".format(expected, tuple(self.std.shape)))

    def _stats_for(self, value):
        if value.dim() == 3:
            return self.mean[0].to(value.device), self.std[0].to(value.device)
        return self.mean.to(value.device), self.std.to(value.device)

    def normalize(self, value):
        mean, std = self._stats_for(value)
        return (value - mean.to(dtype=value.dtype)) / std.to(dtype=value.dtype)

    def denormalize(self, value):
        mean, std = self._stats_for(value)
        return value * std.to(dtype=value.dtype) + mean.to(dtype=value.dtype)

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
            "metadata": self.metadata,
        }


def _resolve_interhuman_h5(data_path, split):
    path = Path(data_path)
    if path.suffix == ".h5":
        return path
    return path / "interhuman_{}.h5".format(split)


def _normalizer_pt_path(path):
    path = Path(path)
    if path.suffix == ".pt":
        return path
    return path / "normalizer.pt"


def _normalizer_json_path(path):
    path = Path(path)
    if path.suffix == ".json":
        return path
    if path.suffix == ".pt":
        return path.with_suffix(".json")
    return path / "normalizer.json"


def compute_forecasting_normalizer(
    data_path,
    save_dir=None,
    window_len=150,
    obs_len=30,
    pred_len=120,
    eps=EPS,
):
    if obs_len + pred_len != window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")

    h5_path = _resolve_interhuman_h5(data_path, "train")
    if not h5_path.exists():
        raise FileNotFoundError(str(h5_path))

    total_sum = torch.zeros((NUM_PERSONS, PERSON_DIM), dtype=torch.float64)
    total_sq_sum = torch.zeros((NUM_PERSONS, PERSON_DIM), dtype=torch.float64)
    num_sequences = 0
    num_frames = 0

    with h5py.File(str(h5_path), "r") as h5:
        for key in h5.keys():
            shape = h5[key].shape
            length = int(shape[0])
            if length < window_len:
                continue
            motion = torch.from_numpy(h5[key][:]).to(dtype=torch.float64)
            active = extract_active_motion(motion)
            if not torch.isfinite(active).all():
                raise ValueError("normalizer 统计发现非有限数值: sample_id={}".format(key))
            total_sum += active.sum(dim=0)
            total_sq_sum += (active * active).sum(dim=0)
            num_sequences += 1
            num_frames += length

    if num_sequences == 0 or num_frames == 0:
        raise ValueError("没有可用于 normalizer 的 train 序列")

    mean = total_sum / float(num_frames)
    var = total_sq_sum / float(num_frames) - mean * mean
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)
    std = torch.where(std < eps, torch.ones_like(std), std)

    mean = mean.to(dtype=torch.float32).view(1, 1, NUM_PERSONS, PERSON_DIM)
    std = std.to(dtype=torch.float32).view(1, 1, NUM_PERSONS, PERSON_DIM)
    metadata = {
        "dataset": "interhuman",
        "data_path": str(data_path),
        "train_h5_path": str(h5_path),
        "window_len": int(window_len),
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "num_train_sequences_used": int(num_sequences),
        "num_train_frames_used": int(num_frames),
        "person_dim": int(PERSON_DIM),
        "eps": float(eps),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    normalizer = ForecastingNormalizer(mean, std, eps=eps, metadata=metadata)

    if save_dir is not None:
        os.makedirs(str(save_dir), exist_ok=True)
        torch.save(normalizer.state_dict(), str(_normalizer_pt_path(save_dir)))
        with open(str(_normalizer_json_path(save_dir)), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    return normalizer


def load_forecasting_normalizer(path):
    pt_path = _normalizer_pt_path(path)
    state = torch.load(str(pt_path), map_location="cpu")
    return ForecastingNormalizer(
        state["mean"],
        state["std"],
        eps=state.get("eps", EPS),
        metadata=state.get("metadata", {}),
    )
