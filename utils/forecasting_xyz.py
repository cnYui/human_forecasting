from collections import OrderedDict

import torch

from model.rotation2xyz import Rotation2xyz
from utils.forecasting_motion import (
    NUM_BODY_JOINTS,
    NUM_PERSONS,
    PERSON_DIM,
    ROT_DIM,
    TRANSL_DIM,
    restore_active_motion,
)


XYZ_METRIC_KEYS = (
    "joint_mse",
    "mpjpe",
    "short_joint_mse",
    "mid_joint_mse",
    "long_joint_mse",
    "root_translation_error",
    "relative_root_distance_error",
    "inter_person_distance_consistency_xyz",
)

PRED_LEN = 120
SHORT_END = 40
MID_END = 80


def _check_active(value):
    if value.dim() != 4:
        raise ValueError("active 必须是 [B,T,2,147]，当前维度数为 {}".format(value.dim()))
    if value.shape[2] != NUM_PERSONS or value.shape[3] != PERSON_DIM:
        raise ValueError("active 必须是 [B,T,2,147]，当前 shape 为 {}".format(tuple(value.shape)))
    if not torch.isfinite(value).all():
        raise ValueError("active 存在非有限数值")


def _check_xyz(name, value):
    if value.dim() != 5:
        raise ValueError("{} 必须是 [B,T,2,24,3]，当前维度数为 {}".format(name, value.dim()))
    if value.shape[2] != NUM_PERSONS or value.shape[3] != NUM_BODY_JOINTS or value.shape[4] != 3:
        raise ValueError("{} 必须是 [B,T,2,24,3]，当前 shape 为 {}".format(name, tuple(value.shape)))
    if not torch.isfinite(value).all():
        raise ValueError("{} 存在非有限数值".format(name))


def _to_float(value):
    return float(value.detach().cpu().item())


def active_to_smpl_motion(active):
    active = torch.as_tensor(active)
    if active.dim() == 3:
        active = active.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    _check_active(active)

    batch_size = active.shape[0]
    num_frames = active.shape[1]
    flat = active.reshape(batch_size * num_frames, NUM_PERSONS, PERSON_DIM)
    motion = restore_active_motion(flat)
    motion = motion.view(batch_size, num_frames, NUM_BODY_JOINTS + 1, NUM_PERSONS * 6)
    if squeeze:
        return motion.squeeze(0)
    return motion


def active_to_xyz(active, device=None, jointstype="smpl", converter=None):
    active = torch.as_tensor(active)
    squeeze = False
    if active.dim() == 3:
        active = active.unsqueeze(0)
        squeeze = True
    _check_active(active)

    if device is None:
        device = active.device
    device = torch.device(device)
    active = active.to(device)

    motion = active_to_smpl_motion(active).to(device)
    rotations = motion.permute(0, 2, 3, 1).contiguous()
    mask = torch.ones((rotations.shape[0], rotations.shape[-1]), dtype=torch.bool, device=device)
    if converter is None:
        converter = Rotation2xyz(device=device)

    xyz = converter(
        rotations,
        mask=mask,
        pose_rep="rot6d",
        translation=True,
        glob=True,
        jointstype=jointstype,
        vertstrans=True,
        num_person=NUM_PERSONS,
    )
    if jointstype != "smpl":
        raise ValueError("P7.1 第一版只支持 jointstype='smpl'")
    person_xyz = torch.stack(
        [xyz[:, :, 0:3, :], xyz[:, :, 3:6, :]],
        dim=2,
    )
    person_xyz = person_xyz.permute(0, 4, 2, 1, 3).contiguous()
    _check_xyz("xyz", person_xyz)
    if squeeze:
        return person_xyz.squeeze(0)
    return person_xyz


def batch_active_to_xyz(value, device=None, jointstype="smpl", converter=None):
    return active_to_xyz(value, device=device, jointstype=jointstype, converter=converter)


def _root_distance(value):
    return torch.norm(value[:, :, 0, 0] - value[:, :, 1, 0], dim=-1)


def _inter_person_distance_consistency(pred, target, obs):
    pred_dist = _root_distance(pred)
    target_dist = _root_distance(target)
    last_obs_dist = _root_distance(obs[:, -1:]).squeeze(1).unsqueeze(1)
    pred_full = torch.cat([last_obs_dist, pred_dist], dim=1)
    target_full = torch.cat([last_obs_dist, target_dist], dim=1)
    pred_delta = pred_full[:, 1:] - pred_full[:, :-1]
    target_delta = target_full[:, 1:] - target_full[:, :-1]
    return torch.abs(pred_delta - target_delta).mean()


def compute_xyz_metrics(pred_xyz, target_xyz, obs_xyz):
    _check_xyz("pred_xyz", pred_xyz)
    _check_xyz("target_xyz", target_xyz)
    _check_xyz("obs_xyz", obs_xyz)
    if tuple(pred_xyz.shape) != tuple(target_xyz.shape):
        raise ValueError(
            "pred_xyz/target_xyz shape 必须一致，当前为 {} / {}".format(
                tuple(pred_xyz.shape), tuple(target_xyz.shape)
            )
        )
    if pred_xyz.shape[1] != PRED_LEN:
        raise ValueError("P7.1 xyz metrics 第一版要求 pred_len=120，当前为 {}".format(pred_xyz.shape[1]))
    if obs_xyz.shape[0] != pred_xyz.shape[0] or obs_xyz.shape[1] < 1:
        raise ValueError("obs_xyz batch 或时间维不合法，当前 shape 为 {}".format(tuple(obs_xyz.shape)))

    diff = pred_xyz - target_xyz
    per_joint_dist = torch.norm(diff, dim=-1)
    pred_dist = _root_distance(pred_xyz)
    target_dist = _root_distance(target_xyz)

    metrics = OrderedDict()
    metrics["joint_mse"] = _to_float((diff * diff).mean())
    metrics["mpjpe"] = _to_float(per_joint_dist.mean())
    metrics["short_joint_mse"] = _to_float((diff[:, :SHORT_END] * diff[:, :SHORT_END]).mean())
    metrics["mid_joint_mse"] = _to_float((diff[:, SHORT_END:MID_END] * diff[:, SHORT_END:MID_END]).mean())
    metrics["long_joint_mse"] = _to_float((diff[:, MID_END:] * diff[:, MID_END:]).mean())
    metrics["root_translation_error"] = _to_float(per_joint_dist[:, :, :, 0].mean())
    metrics["relative_root_distance_error"] = _to_float(torch.abs(pred_dist - target_dist).mean())
    metrics["inter_person_distance_consistency_xyz"] = _to_float(
        _inter_person_distance_consistency(pred_xyz, target_xyz, obs_xyz)
    )

    if tuple(metrics.keys()) != XYZ_METRIC_KEYS:
        raise AssertionError("xyz metrics key 不稳定")
    for key, value in metrics.items():
        if not torch.isfinite(torch.tensor(float(value))):
            raise ValueError("{} 指标为非有限数值: {}".format(key, value))
    return metrics
