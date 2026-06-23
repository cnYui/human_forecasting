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


class T2PInterHumanXYZ(nn.Module):
    def __init__(
        self,
        obs_len=30,
        pred_len=120,
        num_persons=2,
        num_joints=24,
        coord_dim=3,
        hidden_dim=256,
        num_heads=8,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        root_loss_weight=1.0,
        local_loss_weight=1.0,
    ):
        super(T2PInterHumanXYZ, self).__init__()
        self.model_type = "t2p_interhuman_xyz"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.num_persons = int(num_persons)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.root_loss_weight = float(root_loss_weight)
        self.local_loss_weight = float(local_loss_weight)
        self.local_dim = (self.num_joints - 1) * self.coord_dim

        self.root_encoder = nn.GRU(
            input_size=self.coord_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.local_encoder = nn.GRU(
            input_size=self.local_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.agent_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
        )
        self.interaction = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.root_decoder = MLP(
            in_dim=self.hidden_dim,
            out_dim=self.pred_len * self.coord_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.traj_condition = nn.Linear(self.pred_len * self.coord_dim, self.hidden_dim)
        self.local_decoder = MLP(
            in_dim=self.hidden_dim * 2,
            out_dim=self.pred_len * self.local_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def _split_root_local(self, xyz):
        root = xyz[:, :, :, 0, :].contiguous()
        local = xyz - root.unsqueeze(3)
        local_no_root = local[:, :, :, 1:, :].contiguous()
        return root, local_no_root

    def forward_full(self, obs_xyz):
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

        root_input = root_obs.permute(0, 2, 1, 3).contiguous().view(
            batch_size * self.num_persons,
            self.obs_len,
            self.coord_dim,
        )
        local_input = local_obs.permute(0, 2, 1, 3, 4).contiguous().view(
            batch_size * self.num_persons,
            self.obs_len,
            self.local_dim,
        )
        _, root_hidden = self.root_encoder(root_input)
        _, local_hidden = self.local_encoder(local_input)
        agent_hidden = torch.cat((root_hidden[-1], local_hidden[-1]), dim=-1)
        agent_hidden = self.agent_fusion(agent_hidden)
        agent_hidden = agent_hidden.view(batch_size, self.num_persons, self.hidden_dim)

        fused = self.interaction(agent_hidden.transpose(0, 1)).transpose(0, 1).contiguous()
        fused_flat = fused.reshape(batch_size * self.num_persons, self.hidden_dim)

        root_delta = self.root_decoder(fused_flat).view(
            batch_size,
            self.num_persons,
            self.pred_len,
            self.coord_dim,
        )
        last_root = root_obs[:, -1].contiguous()
        future_root = last_root.unsqueeze(2) + root_delta

        traj_feature = self.traj_condition(
            future_root.reshape(batch_size * self.num_persons, self.pred_len * self.coord_dim)
        )
        local_condition = torch.cat((fused_flat, traj_feature), dim=-1)
        local_delta = self.local_decoder(local_condition).view(
            batch_size,
            self.num_persons,
            self.pred_len,
            self.num_joints - 1,
            self.coord_dim,
        )
        last_local = local_obs[:, -1].contiguous()
        future_local_no_root = last_local.unsqueeze(2) + local_delta
        zero_root = torch.zeros(
            batch_size,
            self.num_persons,
            self.pred_len,
            1,
            self.coord_dim,
            device=obs_xyz.device,
            dtype=obs_xyz.dtype,
        )
        future_local = torch.cat((zero_root, future_local_no_root), dim=3)
        pred = future_root.unsqueeze(3) + future_local
        pred = pred.permute(0, 2, 1, 3, 4).contiguous()
        ensure_xyz_prediction_shape(
            pred,
            batch_size=batch_size,
            pred_len=self.pred_len,
            num_persons=self.num_persons,
            num_joints=self.num_joints,
            coord_dim=self.coord_dim,
        )
        return pred, future_root.permute(0, 2, 1, 3).contiguous(), future_local.permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, obs_xyz):
        pred, _, _ = self.forward_full(obs_xyz)
        return pred

    def training_loss(self, obs_xyz, target_xyz, aux_weight=0.0, metamask=False):
        del aux_weight, metamask
        _check_xyz(
            "target_xyz",
            target_xyz,
            self.pred_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )
        pred, pred_root, pred_local = self.forward_full(obs_xyz)
        target_root, target_local_no_root = self._split_root_local(target_xyz)
        target_local = torch.cat(
            (
                torch.zeros_like(target_local_no_root[:, :, :, :1, :]),
                target_local_no_root,
            ),
            dim=3,
        )

        xyz_loss = F.mse_loss(pred, target_xyz)
        root_loss = F.mse_loss(pred_root, target_root)
        local_loss = F.mse_loss(pred_local, target_local)
        loss = xyz_loss
        if self.root_loss_weight > 0:
            loss = loss + self.root_loss_weight * root_loss
        if self.local_loss_weight > 0:
            loss = loss + self.local_loss_weight * local_loss
        denom = 1.0 + max(0.0, self.root_loss_weight) + max(0.0, self.local_loss_weight)
        return loss / denom

    def config(self):
        return {
            "model_type": self.model_type,
            "obs_len": self.obs_len,
            "pred_len": self.pred_len,
            "num_persons": self.num_persons,
            "num_joints": self.num_joints,
            "coord_dim": self.coord_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "root_loss_weight": self.root_loss_weight,
            "local_loss_weight": self.local_loss_weight,
        }
