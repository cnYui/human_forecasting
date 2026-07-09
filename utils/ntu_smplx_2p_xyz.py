from collections import OrderedDict

import torch

from model.rotation2xyz import Rotation2xyz_x


NTU_SMPLX_JOINTS_WITH_TRANS = 56
NTU_SMPLX_BODY_JOINTS = 55
NTU_2P_ROTVEC_FEATS = 6
NTU_NUM_PERSONS = 2
XYZ_COORD_DIM = 3

NTU_XYZ_METRIC_KEYS = (
    "xyz_mse",
    "xyz_mae",
    "mpjpe",
    "first_step_error",
    "velocity_error",
    "root_translation_error",
    "relative_root_distance_error",
    "inter_person_distance_consistency",
)


def check_ntu_motion(name, value, seq_len=None):
    if value.dim() != 4:
        raise ValueError("{} 必须是 [B,56,6,T]，当前维度数为 {}".format(name, value.dim()))
    if tuple(value.shape[1:3]) != (NTU_SMPLX_JOINTS_WITH_TRANS, NTU_2P_ROTVEC_FEATS):
        raise ValueError("{} 后三维前两项必须是 [56,6]，当前为 {}".format(name, tuple(value.shape[1:3])))
    if seq_len is not None and int(value.shape[-1]) != int(seq_len):
        raise ValueError("{} 时间长度必须是 {}，当前为 {}".format(name, int(seq_len), int(value.shape[-1])))
    if not torch.isfinite(value).all():
        raise ValueError("{} 存在非有限数值".format(name))


def check_ntu_xyz(name, value, seq_len=None):
    if value.dim() != 5:
        raise ValueError("{} 必须是 [B,T,2,55,3]，当前维度数为 {}".format(name, value.dim()))
    expected_tail = (NTU_NUM_PERSONS, NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM)
    if tuple(value.shape[2:]) != expected_tail:
        raise ValueError("{} 后三维必须是 {}，当前为 {}".format(name, expected_tail, tuple(value.shape[2:])))
    if seq_len is not None and int(value.shape[1]) != int(seq_len):
        raise ValueError("{} 时间长度必须是 {}，当前为 {}".format(name, int(seq_len), int(value.shape[1])))
    if not torch.isfinite(value).all():
        raise ValueError("{} 存在非有限数值".format(name))


def ntu_rotvec_2p_to_xyz(motion, device=None, converter=None):
    motion = torch.as_tensor(motion)
    check_ntu_motion("motion", motion)
    if device is None:
        device = motion.device
    device = torch.device(device)
    motion = motion.to(device=device, dtype=torch.float32)
    if converter is None:
        converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    mask = torch.ones((motion.shape[0], motion.shape[-1]), dtype=torch.bool, device=device)
    xyz_cat = converter(
        motion,
        mask=mask,
        pose_rep="rotvec",
        translation=True,
        glob=True,
        jointstype="smplx",
        vertstrans=True,
        num_person=NTU_NUM_PERSONS,
    )
    if tuple(xyz_cat.shape[1:3]) != (NTU_SMPLX_BODY_JOINTS, NTU_NUM_PERSONS * XYZ_COORD_DIM):
        raise ValueError("xyz_cat 必须是 [B,55,6,T]，当前为 {}".format(tuple(xyz_cat.shape)))
    xyz = torch.stack((xyz_cat[:, :, 0:3], xyz_cat[:, :, 3:6]), dim=2)
    xyz = xyz.permute(0, 4, 2, 1, 3).contiguous()
    check_ntu_xyz("xyz", xyz, seq_len=motion.shape[-1])
    return xyz


def copy_last_xyz(obs_xyz, pred_len):
    check_ntu_xyz("obs_xyz", obs_xyz)
    return obs_xyz[:, -1:].expand(-1, int(pred_len), -1, -1, -1).contiguous()


def _root_distance(value):
    return torch.norm(value[:, :, 0, 0] - value[:, :, 1, 0], dim=-1)


def _prepend_last_obs(pred_xyz, target_xyz, obs_xyz):
    pred_full = torch.cat((obs_xyz[:, -1:], pred_xyz), dim=1)
    target_full = torch.cat((obs_xyz[:, -1:], target_xyz), dim=1)
    return pred_full, target_full


def compute_ntu_xyz_metrics(pred_xyz, target_xyz, obs_xyz):
    check_ntu_xyz("pred_xyz", pred_xyz)
    check_ntu_xyz("target_xyz", target_xyz, seq_len=pred_xyz.shape[1])
    check_ntu_xyz("obs_xyz", obs_xyz)
    if tuple(pred_xyz.shape) != tuple(target_xyz.shape):
        raise ValueError("pred_xyz/target_xyz shape 必须一致")

    diff = pred_xyz - target_xyz
    dist = torch.norm(diff, dim=-1)
    pred_full, target_full = _prepend_last_obs(pred_xyz, target_xyz, obs_xyz)
    pred_vel = pred_full[:, 1:] - pred_full[:, :-1]
    target_vel = target_full[:, 1:] - target_full[:, :-1]
    pred_root_dist = _root_distance(pred_xyz)
    target_root_dist = _root_distance(target_xyz)
    pred_full_root_dist = _root_distance(pred_full)
    target_full_root_dist = _root_distance(target_full)
    inter_delta = torch.abs(
        (pred_full_root_dist[:, 1:] - pred_full_root_dist[:, :-1])
        - (target_full_root_dist[:, 1:] - target_full_root_dist[:, :-1])
    )

    metrics = OrderedDict()
    metrics["xyz_mse"] = float((diff * diff).mean().detach().cpu().item())
    metrics["xyz_mae"] = float(torch.abs(diff).mean().detach().cpu().item())
    metrics["mpjpe"] = float(dist.mean().detach().cpu().item())
    metrics["first_step_error"] = float(
        torch.norm(pred_xyz[:, 0] - obs_xyz[:, -1], dim=-1).mean().detach().cpu().item()
    )
    metrics["velocity_error"] = float(torch.norm(pred_vel - target_vel, dim=-1).mean().detach().cpu().item())
    metrics["root_translation_error"] = float(dist[:, :, :, 0].mean().detach().cpu().item())
    metrics["relative_root_distance_error"] = float(
        torch.abs(pred_root_dist - target_root_dist).mean().detach().cpu().item()
    )
    metrics["inter_person_distance_consistency"] = float(inter_delta.mean().detach().cpu().item())

    if tuple(metrics.keys()) != NTU_XYZ_METRIC_KEYS:
        raise AssertionError("NTU xyz metrics key 不稳定")
    for key, value in metrics.items():
        if not torch.isfinite(torch.tensor(float(value))):
            raise ValueError("{} 指标为非有限数值: {}".format(key, value))
    return metrics
