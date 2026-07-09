import torch
from torch import nn
from torch.nn import functional as F

from utils.ntu_smplx_2p_xyz import (
    NTU_NUM_PERSONS,
    NTU_SMPLX_BODY_JOINTS,
    XYZ_COORD_DIM,
    check_ntu_xyz,
)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _normalize_action(action, batch_size, num_actions, device):
    if action is None:
        raise ValueError("action 不能为空")
    if not torch.is_tensor(action):
        action = torch.as_tensor(action)
    if action.dim() == 2 and action.shape[1] == 1:
        action = action[:, 0]
    if action.dim() != 1:
        raise ValueError("action 必须是 [B] 或 [B,1]，当前为 {}".format(tuple(action.shape)))
    if int(action.shape[0]) != int(batch_size):
        raise ValueError("action batch 必须是 {}，当前为 {}".format(int(batch_size), int(action.shape[0])))
    action = action.to(device=device, dtype=torch.long)
    if int(action.min().item()) < 0 or int(action.max().item()) >= int(num_actions):
        raise ValueError("action 必须在 [0,{}] 内".format(int(num_actions) - 1))
    return action


class NTULabelXYZTransformer(nn.Module):
    def __init__(
        self,
        obs_len=20,
        pred_len=40,
        num_actions=26,
        num_persons=NTU_NUM_PERSONS,
        num_joints=NTU_SMPLX_BODY_JOINTS,
        coord_dim=XYZ_COORD_DIM,
        latent_dim=256,
        num_heads=4,
        encoder_layers=3,
        decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        velocity_loss_weight=0.2,
        continuity_loss_weight=1.0,
        first_step_loss_weight=0.1,
        mae_loss_weight=0.0,
    ):
        super(NTULabelXYZTransformer, self).__init__()
        self.model_type = "ntu_label_xyz_transformer"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.num_actions = int(num_actions)
        self.num_persons = int(num_persons)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.velocity_loss_weight = float(velocity_loss_weight)
        self.continuity_loss_weight = float(continuity_loss_weight)
        self.first_step_loss_weight = float(first_step_loss_weight)
        self.mae_loss_weight = float(mae_loss_weight)
        self.frame_dim = self.num_persons * self.num_joints * self.coord_dim

        self.input_proj = nn.Linear(self.frame_dim, self.latent_dim)
        self.output_proj = nn.Linear(self.latent_dim, self.frame_dim)
        self.obs_pos = nn.Parameter(torch.zeros(1, self.obs_len, self.latent_dim))
        self.future_pos = nn.Parameter(torch.zeros(1, self.pred_len, self.latent_dim))
        self.action_embed = nn.Embedding(self.num_actions, self.latent_dim)
        self.action_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_layers)
        self.norm = nn.LayerNorm(self.latent_dim)

        # 初始输出严格等价于 copy-last，后续训练只学习相对最后一帧的位移。
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def config(self):
        return {
            "model_type": self.model_type,
            "obs_len": self.obs_len,
            "pred_len": self.pred_len,
            "num_actions": self.num_actions,
            "num_persons": self.num_persons,
            "num_joints": self.num_joints,
            "coord_dim": self.coord_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "velocity_loss_weight": self.velocity_loss_weight,
            "continuity_loss_weight": self.continuity_loss_weight,
            "first_step_loss_weight": self.first_step_loss_weight,
            "mae_loss_weight": self.mae_loss_weight,
        }

    def forward(self, obs_xyz, action):
        check_ntu_xyz("obs_xyz", obs_xyz, seq_len=self.obs_len)
        batch_size = int(obs_xyz.shape[0])
        action = _normalize_action(action, batch_size, self.num_actions, obs_xyz.device)

        obs_flat = obs_xyz.reshape(batch_size, self.obs_len, self.frame_dim)
        obs_tokens = self.input_proj(obs_flat) + self.obs_pos
        action_token = self.action_embed(action).unsqueeze(1) + self.action_type
        memory = torch.cat((action_token, obs_tokens), dim=1).transpose(0, 1).contiguous()
        memory = self.encoder(memory)

        query = self.future_pos.expand(batch_size, -1, -1)
        query = query + self.action_embed(action).unsqueeze(1)
        query = query.transpose(0, 1).contiguous()
        hidden = self.decoder(query, memory).transpose(0, 1).contiguous()
        delta = self.output_proj(self.norm(hidden))
        delta = delta.view(
            batch_size,
            self.pred_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )
        ramp = torch.linspace(
            0.0,
            1.0,
            self.pred_len,
            device=delta.device,
            dtype=delta.dtype,
        ).view(1, self.pred_len, 1, 1, 1)
        delta = delta * ramp
        pred = obs_xyz[:, -1:].expand(-1, self.pred_len, -1, -1, -1) + delta
        check_ntu_xyz("pred_xyz", pred, seq_len=self.pred_len)
        return pred

    def training_loss(self, obs_xyz, target_xyz, action):
        check_ntu_xyz("target_xyz", target_xyz, seq_len=self.pred_len)
        pred = self.forward(obs_xyz, action)
        loss = F.mse_loss(pred, target_xyz)
        if self.mae_loss_weight > 0:
            loss = loss + self.mae_loss_weight * F.l1_loss(pred, target_xyz)
        if self.velocity_loss_weight > 0:
            pred_full = torch.cat((obs_xyz[:, -1:], pred), dim=1)
            target_full = torch.cat((obs_xyz[:, -1:], target_xyz), dim=1)
            pred_vel = pred_full[:, 1:] - pred_full[:, :-1]
            target_vel = target_full[:, 1:] - target_full[:, :-1]
            loss = loss + self.velocity_loss_weight * F.mse_loss(pred_vel, target_vel)
        if self.continuity_loss_weight > 0:
            loss = loss + self.continuity_loss_weight * F.mse_loss(pred[:, 0], obs_xyz[:, -1])
        if self.first_step_loss_weight > 0:
            loss = loss + self.first_step_loss_weight * F.mse_loss(pred[:, 0], target_xyz[:, 0])
        if not torch.isfinite(loss):
            raise ValueError("ntu_label_xyz_transformer loss 为非有限数值")
        return loss


def create_ntu_label_xyz_model_from_config(config):
    model_type = config.get("model_type")
    if model_type != "ntu_label_xyz_transformer":
        raise ValueError("unsupported NTU xyz model_type: {}".format(model_type))
    return NTULabelXYZTransformer(
        obs_len=config.get("obs_len", 20),
        pred_len=config.get("pred_len", 40),
        num_actions=config.get("num_actions", 26),
        num_persons=config.get("num_persons", NTU_NUM_PERSONS),
        num_joints=config.get("num_joints", NTU_SMPLX_BODY_JOINTS),
        coord_dim=config.get("coord_dim", XYZ_COORD_DIM),
        latent_dim=config.get("latent_dim", 256),
        num_heads=config.get("num_heads", 4),
        encoder_layers=config.get("encoder_layers", 3),
        decoder_layers=config.get("decoder_layers", 3),
        dim_feedforward=config.get("dim_feedforward", 1024),
        dropout=config.get("dropout", 0.1),
        velocity_loss_weight=config.get("velocity_loss_weight", 0.2),
        continuity_loss_weight=config.get("continuity_loss_weight", 1.0),
        first_step_loss_weight=config.get("first_step_loss_weight", 0.1),
        mae_loss_weight=config.get("mae_loss_weight", 0.0),
    )
