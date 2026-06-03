from collections import OrderedDict

import torch

from utils.forecasting_motion import NUM_PERSONS, PERSON_DIM, ROT_DIM, TRANSL_DIM
from utils.rotation_conversions import rotation_6d_to_matrix


METRIC_KEYS = (
    "future_mse",
    "rotation_mse",
    "translation_mse",
    "short_mse",
    "mid_mse",
    "long_mse",
    "relative_root_distance_error",
    "relative_orientation_error",
    "inter_person_distance_consistency",
)

PRED_LEN = 120
SHORT_END = 40
MID_END = 80


def _assert_shape(name, tensor, expected):
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError("{} shape 必须是 {}，当前为 {}".format(name, expected, tuple(tensor.shape)))


def _assert_forecasting_inputs(pred, target, obs):
    if pred.dim() != 4:
        raise ValueError("pred 必须是 [B,120,2,147]，当前维度数为 {}".format(pred.dim()))
    if target.dim() != 4:
        raise ValueError("target 必须是 [B,120,2,147]，当前维度数为 {}".format(target.dim()))
    if obs.dim() != 4:
        raise ValueError("obs 必须是 [B,30,2,147]，当前维度数为 {}".format(obs.dim()))

    batch_size = pred.shape[0]
    _assert_shape("target", target, pred.shape)
    if pred.shape[1] != PRED_LEN:
        raise ValueError("P2 指标第一版要求 pred_len=120，当前为 {}".format(pred.shape[1]))
    if pred.shape[2] != NUM_PERSONS or pred.shape[3] != PERSON_DIM:
        raise ValueError("pred 必须是 [B,120,2,147]，当前为 {}".format(tuple(pred.shape)))
    if obs.shape[0] != batch_size or obs.shape[2] != NUM_PERSONS or obs.shape[3] != PERSON_DIM:
        raise ValueError("obs 必须是 [B,30,2,147]，当前为 {}".format(tuple(obs.shape)))
    if obs.shape[1] < 1:
        raise ValueError("obs 至少需要 1 帧")

    for name, tensor in (("pred", pred), ("target", target), ("obs", obs)):
        if not torch.isfinite(tensor).all():
            raise ValueError("{} 存在非有限数值".format(name))


def _to_float(value):
    return float(value.detach().cpu().item())


def _root_distance(value):
    trans_a = value[:, :, 0, ROT_DIM : ROT_DIM + TRANSL_DIM]
    trans_b = value[:, :, 1, ROT_DIM : ROT_DIM + TRANSL_DIM]
    return torch.norm(trans_a - trans_b, dim=-1)


def _relative_rotation(value):
    root_a = value[:, :, 0, 0:6]
    root_b = value[:, :, 1, 0:6]
    rot_a = rotation_6d_to_matrix(root_a)
    rot_b = rotation_6d_to_matrix(root_b)
    return torch.matmul(rot_a.transpose(-1, -2), rot_b)


def _relative_orientation_error(pred, target, eps=1e-6):
    rel_pred = _relative_rotation(pred)
    rel_target = _relative_rotation(target)
    rot_error = torch.matmul(rel_pred.transpose(-1, -2), rel_target)
    trace = rot_error[..., 0, 0] + rot_error[..., 1, 1] + rot_error[..., 2, 2]
    cos_angle = (trace - 1.0) * 0.5
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    same_root = (pred[:, :, :, 0:6] == target[:, :, :, 0:6]).view(
        pred.shape[0], pred.shape[1], -1
    ).all(dim=-1)
    # 完美预测必须返回 0，但不能把真实的微小朝向误差吞掉。
    angle = torch.where(same_root, torch.zeros_like(angle), angle)
    return angle.mean()


def _inter_person_distance_consistency(pred, target, obs):
    pred_dist = _root_distance(pred)
    target_dist = _root_distance(target)
    last_obs_dist = _root_distance(obs[:, -1:]).squeeze(1).unsqueeze(1)

    pred_full = torch.cat([last_obs_dist, pred_dist], dim=1)
    target_full = torch.cat([last_obs_dist, target_dist], dim=1)
    pred_delta = pred_full[:, 1:] - pred_full[:, :-1]
    target_delta = target_full[:, 1:] - target_full[:, :-1]
    return torch.abs(pred_delta - target_delta).mean()


def compute_forecasting_metrics(pred, target, obs):
    _assert_forecasting_inputs(pred, target, obs)

    diff = pred - target
    rot_diff = pred[..., :ROT_DIM] - target[..., :ROT_DIM]
    trans_diff = pred[..., ROT_DIM : ROT_DIM + TRANSL_DIM] - target[
        ..., ROT_DIM : ROT_DIM + TRANSL_DIM
    ]
    pred_dist = _root_distance(pred)
    target_dist = _root_distance(target)

    metrics = OrderedDict()
    metrics["future_mse"] = _to_float((diff * diff).mean())
    metrics["rotation_mse"] = _to_float((rot_diff * rot_diff).mean())
    metrics["translation_mse"] = _to_float((trans_diff * trans_diff).mean())
    metrics["short_mse"] = _to_float((diff[:, :SHORT_END] * diff[:, :SHORT_END]).mean())
    metrics["mid_mse"] = _to_float((diff[:, SHORT_END:MID_END] * diff[:, SHORT_END:MID_END]).mean())
    metrics["long_mse"] = _to_float((diff[:, MID_END:] * diff[:, MID_END:]).mean())
    metrics["relative_root_distance_error"] = _to_float(torch.abs(pred_dist - target_dist).mean())
    metrics["relative_orientation_error"] = _to_float(_relative_orientation_error(pred, target))
    metrics["inter_person_distance_consistency"] = _to_float(
        _inter_person_distance_consistency(pred, target, obs)
    )

    if tuple(metrics.keys()) != METRIC_KEYS:
        raise AssertionError("metrics key 不稳定")
    for key, value in metrics.items():
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("{} 指标为非有限数值: {}".format(key, value))
    return metrics
