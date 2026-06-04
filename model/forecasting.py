import torch
from torch import nn

from utils.forecasting_motion import (
    NUM_PERSONS,
    PERSON_DIM,
    RELATION_FEATURE_SETS,
    extract_relation_features,
    relation_feature_dim as relation_feature_dim_for_set,
    relation_feature_names,
)


FORECASTING_MODEL_TYPES = ("independent", "concat", "relation")
RELATION_ENCODER_TYPES = ("gru", "none")


def _check_obs(obs, obs_len, person_dim):
    if obs.dim() != 4:
        raise ValueError("obs 必须是 [B,T,2,147]，当前维度数为 {}".format(obs.dim()))
    if obs.shape[1] != obs_len:
        raise ValueError("obs_len 应为 {}，实际为 {}".format(obs_len, obs.shape[1]))
    if obs.shape[2] != NUM_PERSONS or obs.shape[3] != person_dim:
        raise ValueError("obs 必须是 [B,T,2,147]，当前 shape 为 {}".format(tuple(obs.shape)))


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


class IndependentForecastingModel(nn.Module):
    def __init__(self, obs_len=30, pred_len=120, person_dim=PERSON_DIM, hidden_dim=256, num_layers=2):
        super(IndependentForecastingModel, self).__init__()
        self.model_type = "independent"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.person_dim = int(person_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.encoder = nn.GRU(
            input_size=self.person_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        # 共享单人分支用于阻断跨人信息泄漏，同时控制参数量。
        self.decoder = _FutureDecoder(self.hidden_dim, self.pred_len * self.person_dim)

    def forward(self, obs):
        _check_obs(obs, self.obs_len, self.person_dim)
        batch_size = obs.shape[0]
        x = obs.permute(0, 2, 1, 3).contiguous().view(
            batch_size * NUM_PERSONS, self.obs_len, self.person_dim
        )
        _, hidden = self.encoder(x)
        future = self.decoder(hidden[-1])
        future = future.view(batch_size, NUM_PERSONS, self.pred_len, self.person_dim)
        return future.permute(0, 2, 1, 3).contiguous()

    def config(self):
        return _model_config(
            self.model_type,
            self.obs_len,
            self.pred_len,
            self.person_dim,
            self.hidden_dim,
            self.num_layers,
        )


class ConcatForecastingModel(nn.Module):
    def __init__(self, obs_len=30, pred_len=120, person_dim=PERSON_DIM, hidden_dim=256, num_layers=2):
        super(ConcatForecastingModel, self).__init__()
        self.model_type = "concat"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.person_dim = int(person_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.input_dim = NUM_PERSONS * self.person_dim

        self.encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.decoder = _FutureDecoder(self.hidden_dim, self.pred_len * self.input_dim)

    def forward(self, obs):
        _check_obs(obs, self.obs_len, self.person_dim)
        batch_size = obs.shape[0]
        x = obs.contiguous().view(batch_size, self.obs_len, self.input_dim)
        _, hidden = self.encoder(x)
        future = self.decoder(hidden[-1])
        return future.view(batch_size, self.pred_len, NUM_PERSONS, self.person_dim).contiguous()

    def config(self):
        return _model_config(
            self.model_type,
            self.obs_len,
            self.pred_len,
            self.person_dim,
            self.hidden_dim,
            self.num_layers,
        )


class RelationAwareForecastingModel(nn.Module):
    def __init__(
        self,
        obs_len=30,
        pred_len=120,
        person_dim=PERSON_DIM,
        hidden_dim=256,
        num_layers=2,
        relation_hidden_dim=128,
        relation_num_layers=1,
        relation_feature_dim=None,
        relation_features=None,
        relation_feature_set="all",
        relation_encoder_type="gru",
    ):
        super(RelationAwareForecastingModel, self).__init__()
        self.model_type = "relation"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.person_dim = int(person_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.relation_hidden_dim = int(relation_hidden_dim)
        self.relation_num_layers = int(relation_num_layers)
        self.relation_feature_set = str(relation_feature_set)
        self.relation_encoder_type = str(relation_encoder_type)
        if self.relation_feature_set not in RELATION_FEATURE_SETS:
            raise ValueError(
                "relation_feature_set 必须是 {}，当前为 {}".format(
                    RELATION_FEATURE_SETS, self.relation_feature_set
                )
            )
        if self.relation_encoder_type not in RELATION_ENCODER_TYPES:
            raise ValueError(
                "relation_encoder_type 必须是 {}，当前为 {}".format(
                    RELATION_ENCODER_TYPES, self.relation_encoder_type
                )
            )
        expected_feature_dim = relation_feature_dim_for_set(self.relation_feature_set)
        if relation_feature_dim is None:
            relation_feature_dim = expected_feature_dim
        if int(relation_feature_dim) != expected_feature_dim:
            raise ValueError(
                "relation_feature_dim={} 与 feature_set={} 的期望维度 {} 不一致".format(
                    relation_feature_dim, self.relation_feature_set, expected_feature_dim
                )
            )
        self.relation_feature_dim = int(relation_feature_dim)
        self.relation_features = list(
            relation_features or relation_feature_names(self.relation_feature_set)
        )
        self.input_dim = NUM_PERSONS * self.person_dim

        self.person_encoder = nn.GRU(
            input_size=self.person_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        if self.relation_encoder_type == "gru":
            self.relation_encoder = nn.GRU(
                input_size=self.relation_feature_dim,
                hidden_size=self.relation_hidden_dim,
                num_layers=self.relation_num_layers,
                batch_first=True,
            )
        else:
            self.relation_encoder = nn.Linear(
                self.relation_feature_dim, self.relation_hidden_dim
            )
        self.fusion = nn.Sequential(
            nn.Linear(NUM_PERSONS * self.hidden_dim + self.relation_hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.decoder = _FutureDecoder(self.hidden_dim, self.pred_len * self.input_dim)

    def forward(self, obs):
        _check_obs(obs, self.obs_len, self.person_dim)
        batch_size = obs.shape[0]

        person_x = obs.permute(0, 2, 1, 3).contiguous().view(
            batch_size * NUM_PERSONS, self.obs_len, self.person_dim
        )
        _, person_hidden = self.person_encoder(person_x)
        person_hidden = person_hidden[-1].view(batch_size, NUM_PERSONS * self.hidden_dim)

        relation_features = extract_relation_features(obs, feature_set=self.relation_feature_set)
        if relation_features.shape[-1] != self.relation_feature_dim:
            raise ValueError(
                "relation feature dim 应为 {}，实际为 {}".format(
                    self.relation_feature_dim, relation_features.shape[-1]
                )
            )
        if self.relation_encoder_type == "gru":
            _, relation_hidden = self.relation_encoder(relation_features)
            relation_hidden = relation_hidden[-1]
        else:
            relation_hidden = self.relation_encoder(relation_features.mean(dim=1))

        joint_hidden = self.fusion(torch.cat([person_hidden, relation_hidden], dim=-1))
        future = self.decoder(joint_hidden)
        return future.view(batch_size, self.pred_len, NUM_PERSONS, self.person_dim).contiguous()

    def config(self):
        return _model_config(
            self.model_type,
            self.obs_len,
            self.pred_len,
            self.person_dim,
            self.hidden_dim,
            self.num_layers,
            relation_hidden_dim=self.relation_hidden_dim,
            relation_num_layers=self.relation_num_layers,
            relation_feature_dim=self.relation_feature_dim,
            relation_features=self.relation_features,
            relation_feature_set=self.relation_feature_set,
            relation_encoder_type=self.relation_encoder_type,
        )


def _model_config(
    model_type,
    obs_len,
    pred_len,
    person_dim,
    hidden_dim,
    num_layers,
    relation_hidden_dim=None,
    relation_num_layers=None,
    relation_feature_dim=None,
    relation_features=None,
    relation_feature_set=None,
    relation_encoder_type=None,
):
    config = {
        "model_type": model_type,
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "person_dim": int(person_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
    }
    if model_type == "relation":
        relation_feature_set = relation_feature_set or "all"
        relation_encoder_type = relation_encoder_type or "gru"
        if relation_feature_dim is None:
            relation_feature_dim = relation_feature_dim_for_set(relation_feature_set)
        config["relation_hidden_dim"] = int(relation_hidden_dim)
        config["relation_num_layers"] = int(relation_num_layers)
        config["relation_feature_dim"] = int(relation_feature_dim)
        config["relation_features"] = list(
            relation_features or relation_feature_names(relation_feature_set)
        )
        config["relation_feature_set"] = str(relation_feature_set)
        config["relation_encoder_type"] = str(relation_encoder_type)
    return config


def create_forecasting_model(
    model_type,
    obs_len=30,
    pred_len=120,
    person_dim=PERSON_DIM,
    hidden_dim=256,
    num_layers=2,
    relation_hidden_dim=128,
    relation_num_layers=1,
    relation_feature_dim=None,
    relation_features=None,
    relation_feature_set="all",
    relation_encoder_type="gru",
):
    if model_type == "independent":
        return IndependentForecastingModel(
            obs_len=obs_len,
            pred_len=pred_len,
            person_dim=person_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    if model_type == "concat":
        return ConcatForecastingModel(
            obs_len=obs_len,
            pred_len=pred_len,
            person_dim=person_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    if model_type == "relation":
        return RelationAwareForecastingModel(
            obs_len=obs_len,
            pred_len=pred_len,
            person_dim=person_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            relation_hidden_dim=relation_hidden_dim,
            relation_num_layers=relation_num_layers,
            relation_feature_dim=relation_feature_dim,
            relation_features=relation_features,
            relation_feature_set=relation_feature_set,
            relation_encoder_type=relation_encoder_type,
        )
    raise ValueError("forecasting model 支持 {}，当前为 {}".format(FORECASTING_MODEL_TYPES, model_type))


def create_forecasting_model_from_config(config):
    config = dict(config)
    model_type = config.pop("model_type")
    return create_forecasting_model(model_type=model_type, **config)


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def ensure_prediction_shape(pred, batch_size, pred_len, person_dim=PERSON_DIM):
    expected = (batch_size, pred_len, NUM_PERSONS, person_dim)
    if tuple(pred.shape) != expected:
        raise ValueError("pred shape 应为 {}，实际为 {}".format(expected, tuple(pred.shape)))
    if not torch.isfinite(pred).all():
        raise ValueError("pred 存在非有限数值")
