import torch
from torch import nn

from model.forecasting_somoformer import (
    JointSpaceSoMoFormer,
    OfficialSoMoFormerXYZ,
    count_parameters,
    ensure_xyz_prediction_shape,
)


XYZ_FORECASTING_MODEL_TYPES = ("somoformer_xyz", "independent_pair_xyz", "official_somoformer_xyz")


def _check_xyz_obs(obs, obs_len, num_persons=2, num_joints=24, coord_dim=3):
    if obs.dim() != 5:
        raise ValueError("obs_xyz 必须是 [B,T,2,24,3]，当前维度数为 {}".format(obs.dim()))
    expected_tail = (int(obs_len), int(num_persons), int(num_joints), int(coord_dim))
    if tuple(obs.shape[1:]) != expected_tail:
        raise ValueError(
            "obs_xyz 后四维必须是 {}，当前为 {}".format(expected_tail, tuple(obs.shape[1:]))
        )
    if not torch.isfinite(obs).all():
        raise ValueError("obs_xyz 存在非有限数值")


class _FutureDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(_FutureDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden):
        return self.net(hidden)


class IndependentPairXYZModel(nn.Module):
    def __init__(
        self,
        obs_len=30,
        pred_len=120,
        num_persons=2,
        num_joints=24,
        coord_dim=3,
        hidden_dim=256,
        num_layers=2,
    ):
        super(IndependentPairXYZModel, self).__init__()
        self.model_type = "independent_pair_xyz"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.num_persons = int(num_persons)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.person_dim = self.num_joints * self.coord_dim

        self.encoder = nn.GRU(
            input_size=self.person_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        # 共享预测器只看单个人的历史，避免任何跨人信息流。
        self.decoder = _FutureDecoder(self.hidden_dim, self.pred_len * self.person_dim)

    def forward(self, obs_xyz):
        _check_xyz_obs(
            obs_xyz,
            self.obs_len,
            num_persons=self.num_persons,
            num_joints=self.num_joints,
            coord_dim=self.coord_dim,
        )
        batch_size = obs_xyz.shape[0]
        x = obs_xyz.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * self.num_persons, self.obs_len, self.person_dim)
        _, hidden = self.encoder(x)
        future = self.decoder(hidden[-1])
        future = future.view(
            batch_size,
            self.num_persons,
            self.pred_len,
            self.num_joints,
            self.coord_dim,
        )
        pred = future.permute(0, 2, 1, 3, 4).contiguous()
        ensure_xyz_prediction_shape(
            pred,
            batch_size=batch_size,
            pred_len=self.pred_len,
            num_persons=self.num_persons,
            num_joints=self.num_joints,
            coord_dim=self.coord_dim,
        )
        return pred

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
        }


def create_xyz_forecasting_model(
    model_type,
    obs_len=30,
    pred_len=120,
    hidden_dim=256,
    num_layers=2,
    num_heads=8,
    dim_feedforward=1024,
    dct_n=30,
    dropout=0.1,
    residual_connection=True,
    output_scale=1.0,
    location_method="grid",
    grid_len=3,
    grid_emb_size=8,
    normalize_inputs=False,
    activation="relu",
    learned_embedding=True,
):
    if model_type == "independent_pair_xyz":
        return IndependentPairXYZModel(
            obs_len=obs_len,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    if model_type == "somoformer_xyz":
        return JointSpaceSoMoFormer(
            obs_len=obs_len,
            pred_len=pred_len,
            dct_n=dct_n,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            residual_connection=residual_connection,
        )
    if model_type == "official_somoformer_xyz":
        return OfficialSoMoFormerXYZ(
            obs_len=obs_len,
            pred_len=pred_len,
            dct_n=dct_n,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            residual_connection=residual_connection,
            output_scale=output_scale,
            location_method=location_method,
            grid_len=grid_len,
            grid_emb_size=grid_emb_size,
            normalize_inputs=normalize_inputs,
            activation=activation,
            learned_embedding=learned_embedding,
        )
    raise ValueError("unsupported xyz model_type: {}".format(model_type))


def create_xyz_forecasting_model_from_config(config):
    model_type = config.get("model_type")
    if model_type not in XYZ_FORECASTING_MODEL_TYPES:
        raise ValueError(
            "model_type 必须是 {}，当前为 {}".format(XYZ_FORECASTING_MODEL_TYPES, model_type)
        )
    return create_xyz_forecasting_model(
        model_type=model_type,
        obs_len=config.get("obs_len", 30),
        pred_len=config.get("pred_len", 120),
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 8),
        dim_feedforward=config.get("dim_feedforward", 1024),
        dct_n=config.get("dct_n", 30),
        dropout=config.get("dropout", 0.1),
        residual_connection=config.get("residual_connection", True),
        output_scale=config.get("output_scale", 1.0),
        location_method=config.get("location_method", "grid"),
        grid_len=config.get("grid_len", 3),
        grid_emb_size=config.get("grid_emb_size", 8),
        normalize_inputs=config.get("normalize_inputs", False),
        activation=config.get("activation", "relu"),
        learned_embedding=config.get("learned_embedding", True),
    )
