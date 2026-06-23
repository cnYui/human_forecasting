import torch
from torch import nn
from torch.nn import functional as F

from model.forecasting_somoformer import ensure_xyz_prediction_shape


def _check_xyz(name, value, seq_len, num_persons, num_joints, coord_dim):
    if value.dim() != 5:
        raise ValueError("{} 必须是 [B,T,2,24,3]，当前维度数为 {}".format(name, value.dim()))
    expected_tail = (int(seq_len), int(num_persons), int(num_joints), int(coord_dim))
    if tuple(value.shape[1:]) != expected_tail:
        raise ValueError("{} 后四维必须是 {}，当前为 {}".format(name, expected_tail, tuple(value.shape[1:])))
    if not torch.isfinite(value).all():
        raise ValueError("{} 存在非有限数值".format(name))


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, value):
        return self.net(value)


def _future_velocity(value, last_obs):
    prev = torch.cat((last_obs.unsqueeze(1), value[:, :-1]), dim=1)
    return value - prev


def _best_of_k_mse(pred, target):
    diff = pred - target.unsqueeze(1)
    per_sample = (diff * diff).mean(dim=tuple(range(2, diff.dim())))
    best_error, best_index = per_sample.min(dim=1)
    return best_error.mean(), best_index


def _gather_k(value, index):
    shape = [value.shape[0], 1] + [1] * (value.dim() - 2)
    gather_index = index.view(*shape).expand(
        value.shape[0],
        1,
        *value.shape[2:],
    )
    return value.gather(1, gather_index).squeeze(1)


def _bounded_diversity_penalty(pred_root):
    if pred_root.shape[1] <= 1:
        return pred_root.new_tensor(0.0)
    batch_size, num_samples = pred_root.shape[:2]
    flat = pred_root.reshape(batch_size, num_samples, -1)
    distance = torch.cdist(flat, flat, p=1)
    mask = torch.triu(
        torch.ones(num_samples, num_samples, device=pred_root.device, dtype=torch.bool),
        diagonal=1,
    )
    if not mask.any():
        return pred_root.new_tensor(0.0)
    # 指数有界项避免直接最大化距离导致训练初期数值不稳。
    return torch.exp(-distance[:, mask] / 100.0).mean()


class DuMMFInterHumanXYZ(nn.Module):
    def __init__(
        self,
        obs_len=30,
        pred_len=120,
        num_persons=2,
        num_joints=24,
        coord_dim=3,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        num_samples=5,
        global_loss_weight=1.0,
        root_loss_weight=1.0,
        local_loss_weight=1.0,
        velocity_loss_weight=0.2,
        diversity_weight=0.01,
    ):
        super(DuMMFInterHumanXYZ, self).__init__()
        self.model_type = "dummf_interhuman_xyz"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.num_persons = int(num_persons)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.num_samples = int(num_samples)
        self.global_loss_weight = float(global_loss_weight)
        self.root_loss_weight = float(root_loss_weight)
        self.local_loss_weight = float(local_loss_weight)
        self.velocity_loss_weight = float(velocity_loss_weight)
        self.diversity_weight = float(diversity_weight)

        if self.num_samples < 1:
            raise ValueError("num_samples 必须 >= 1")

        self.local_dim = (self.num_joints - 1) * self.coord_dim
        self.global_dim = self.num_persons * self.num_joints * self.coord_dim + 7

        self.local_encoder = nn.GRU(
            input_size=self.local_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.global_encoder = nn.GRU(
            input_size=self.global_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        self.global_intent = nn.Parameter(torch.zeros(self.num_samples, self.hidden_dim))
        self.local_intent = nn.Parameter(torch.zeros(self.num_samples, self.hidden_dim))
        nn.init.normal_(self.global_intent, std=0.02)
        nn.init.normal_(self.local_intent, std=0.02)

        self.root_decoder = MLP(
            in_dim=self.hidden_dim * 2,
            out_dim=self.pred_len * self.num_persons * self.coord_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.local_decoder = MLP(
            in_dim=self.hidden_dim * 3,
            out_dim=self.pred_len * self.local_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def _split_root_local(self, xyz):
        root = xyz[:, :, :, 0, :].contiguous()
        local = xyz - root.unsqueeze(3)
        local_no_root = local[:, :, :, 1:, :].contiguous()
        return root, local_no_root

    def _global_input(self, obs_xyz, root_obs):
        batch_size = int(obs_xyz.shape[0])
        centered = obs_xyz - root_obs.mean(dim=2, keepdim=True).unsqueeze(3)
        flat_xyz = centered.reshape(batch_size, self.obs_len, -1)
        rel_root = root_obs[:, :, 0] - root_obs[:, :, 1]
        root_dist = torch.norm(rel_root, dim=-1, keepdim=True)
        root_velocity = root_obs[:, 1:] - root_obs[:, :-1]
        root_velocity = torch.cat((root_velocity[:, :1] * 0.0, root_velocity), dim=1)
        rel_velocity = root_velocity[:, :, 0] - root_velocity[:, :, 1]
        return torch.cat((flat_xyz, rel_root, root_dist, rel_velocity), dim=-1)

    def _encode(self, obs_xyz):
        _check_xyz(
            "obs_xyz",
            obs_xyz,
            self.obs_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )
        batch_size = int(obs_xyz.shape[0])
        root_obs, local_obs = self._split_root_local(obs_xyz)

        local_input = local_obs.permute(0, 2, 1, 3, 4).contiguous().view(
            batch_size * self.num_persons,
            self.obs_len,
            self.local_dim,
        )
        _, local_hidden = self.local_encoder(local_input)
        local_hidden = local_hidden[-1].view(batch_size, self.num_persons, self.hidden_dim)

        global_input = self._global_input(obs_xyz, root_obs)
        _, global_hidden = self.global_encoder(global_input)
        global_hidden = global_hidden[-1]
        return root_obs, local_obs, local_hidden, global_hidden

    def forward_multi(self, obs_xyz, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples
        num_samples = int(num_samples)
        if num_samples < 1 or num_samples > self.num_samples:
            raise ValueError("num_samples 必须在 [1, {}] 内，当前为 {}".format(self.num_samples, num_samples))

        batch_size = int(obs_xyz.shape[0])
        root_obs, local_obs, local_hidden, global_hidden = self._encode(obs_xyz)
        global_intent = self.global_intent[:num_samples]
        local_intent = self.local_intent[:num_samples]

        root_feature = torch.cat(
            (
                global_hidden.unsqueeze(1).expand(batch_size, num_samples, self.hidden_dim),
                global_intent.unsqueeze(0).expand(batch_size, num_samples, self.hidden_dim),
            ),
            dim=-1,
        )
        root_delta = self.root_decoder(root_feature).view(
            batch_size,
            num_samples,
            self.pred_len,
            self.num_persons,
            self.coord_dim,
        )
        future_root = root_obs[:, -1].unsqueeze(1).unsqueeze(2) + root_delta

        sample_global = root_feature[:, :, : self.hidden_dim] + root_feature[:, :, self.hidden_dim :]
        local_feature = torch.cat(
            (
                local_hidden.unsqueeze(1).expand(
                    batch_size,
                    num_samples,
                    self.num_persons,
                    self.hidden_dim,
                ),
                sample_global.unsqueeze(2).expand(
                    batch_size,
                    num_samples,
                    self.num_persons,
                    self.hidden_dim,
                ),
                local_intent.unsqueeze(0).unsqueeze(2).expand(
                    batch_size,
                    num_samples,
                    self.num_persons,
                    self.hidden_dim,
                ),
            ),
            dim=-1,
        )
        local_delta = self.local_decoder(local_feature).view(
            batch_size,
            num_samples,
            self.num_persons,
            self.pred_len,
            self.num_joints - 1,
            self.coord_dim,
        )
        last_local = local_obs[:, -1].unsqueeze(1).unsqueeze(3)
        future_local_no_root = last_local + local_delta
        zero_root = torch.zeros(
            batch_size,
            num_samples,
            self.num_persons,
            self.pred_len,
            1,
            self.coord_dim,
            device=obs_xyz.device,
            dtype=obs_xyz.dtype,
        )
        future_local = torch.cat((zero_root, future_local_no_root), dim=4)
        pred = future_root.permute(0, 1, 3, 2, 4).unsqueeze(4) + future_local
        pred = pred.permute(0, 1, 3, 2, 4, 5).contiguous()

        if not torch.isfinite(pred).all():
            raise ValueError("dummf_interhuman_xyz 预测存在非有限数值")
        return pred

    def sample(self, obs_xyz, num_samples=None):
        return self.forward_multi(obs_xyz, num_samples=num_samples)

    def forward(self, obs_xyz):
        pred = self.forward_multi(obs_xyz, num_samples=1)[:, 0].contiguous()
        ensure_xyz_prediction_shape(
            pred,
            batch_size=int(obs_xyz.shape[0]),
            pred_len=self.pred_len,
            num_persons=self.num_persons,
            num_joints=self.num_joints,
            coord_dim=self.coord_dim,
        )
        return pred

    def training_loss(
        self,
        obs_xyz,
        target_xyz,
        global_loss_weight=None,
        root_loss_weight=None,
        local_loss_weight=None,
        velocity_loss_weight=None,
        diversity_weight=None,
        aux_weight=0.0,
        metamask=False,
    ):
        del aux_weight, metamask
        _check_xyz(
            "target_xyz",
            target_xyz,
            self.pred_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )
        global_loss_weight = self.global_loss_weight if global_loss_weight is None else float(global_loss_weight)
        root_loss_weight = self.root_loss_weight if root_loss_weight is None else float(root_loss_weight)
        local_loss_weight = self.local_loss_weight if local_loss_weight is None else float(local_loss_weight)
        velocity_loss_weight = (
            self.velocity_loss_weight if velocity_loss_weight is None else float(velocity_loss_weight)
        )
        diversity_weight = self.diversity_weight if diversity_weight is None else float(diversity_weight)

        pred = self.forward_multi(obs_xyz)
        pred_root = pred[:, :, :, :, 0, :]
        target_root, target_local = self._split_root_local(target_xyz)
        pred_local = pred - pred_root.unsqueeze(4)
        pred_local_no_root = pred_local[:, :, :, :, 1:, :]

        global_loss, best_index = _best_of_k_mse(pred, target_xyz)
        root_loss, _ = _best_of_k_mse(pred_root, target_root)
        local_loss, _ = _best_of_k_mse(pred_local_no_root, target_local)

        selected = _gather_k(pred, best_index)
        selected_velocity = _future_velocity(selected, obs_xyz[:, -1])
        target_velocity = _future_velocity(target_xyz, obs_xyz[:, -1])
        velocity_loss = F.mse_loss(selected_velocity, target_velocity)
        diversity_loss = _bounded_diversity_penalty(pred_root)

        weighted = pred.new_tensor(0.0)
        denom = 0.0
        if global_loss_weight > 0:
            weighted = weighted + global_loss_weight * global_loss
            denom += global_loss_weight
        if root_loss_weight > 0:
            weighted = weighted + root_loss_weight * root_loss
            denom += root_loss_weight
        if local_loss_weight > 0:
            weighted = weighted + local_loss_weight * local_loss
            denom += local_loss_weight
        if velocity_loss_weight > 0:
            weighted = weighted + velocity_loss_weight * velocity_loss
            denom += velocity_loss_weight
        if diversity_weight > 0:
            weighted = weighted + diversity_weight * diversity_loss
            denom += diversity_weight
        if denom <= 0:
            raise ValueError("DuMMF loss 权重不能全部为 0")
        loss = weighted / denom
        if not torch.isfinite(loss):
            raise ValueError("dummf_interhuman_xyz loss 为非有限数值")
        return loss

    def config(self):
        return {
            "model_type": self.model_type,
            "obs_len": self.obs_len,
            "pred_len": self.pred_len,
            "num_persons": self.num_persons,
            "num_joints": self.num_joints,
            "coord_dim": self.coord_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_samples": self.num_samples,
            "global_loss_weight": self.global_loss_weight,
            "root_loss_weight": self.root_loss_weight,
            "local_loss_weight": self.local_loss_weight,
            "velocity_loss_weight": self.velocity_loss_weight,
            "diversity_weight": self.diversity_weight,
        }
