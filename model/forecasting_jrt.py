import torch
from torch import nn
from torch.nn import functional as F

from model.forecasting_somoformer import ensure_xyz_prediction_shape


SMPL_24_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
)


def _check_xyz(name, value, seq_len, num_persons, num_joints, coord_dim):
    if value.dim() != 5:
        raise ValueError("{} 必须是 [B,T,2,24,3]，当前维度数为 {}".format(name, value.dim()))
    expected_tail = (int(seq_len), int(num_persons), int(num_joints), int(coord_dim))
    if tuple(value.shape[1:]) != expected_tail:
        raise ValueError("{} 后四维必须是 {}，当前为 {}".format(name, expected_tail, tuple(value.shape[1:])))
    if not torch.isfinite(value).all():
        raise ValueError("{} 存在非有限数值".format(name))


def _tokenize_xyz(value):
    batch_size, seq_len, num_persons, num_joints, coord_dim = value.shape
    return value.permute(0, 2, 3, 1, 4).contiguous().view(
        batch_size,
        num_persons * num_joints,
        seq_len,
        coord_dim,
    )


def _untokenize_xyz(value, batch_size, seq_len, num_persons, num_joints, coord_dim):
    return value.view(batch_size, num_persons, num_joints, seq_len, coord_dim).permute(
        0,
        3,
        1,
        2,
        4,
    ).contiguous()


def _build_adjacency(num_persons, num_joints, edges):
    num_tokens = int(num_persons) * int(num_joints)
    adj = torch.eye(num_tokens, dtype=torch.float32)
    for person_idx in range(int(num_persons)):
        offset = person_idx * int(num_joints)
        for start, end in edges:
            adj[offset + int(start), offset + int(end)] = 1.0
            adj[offset + int(end), offset + int(start)] = 1.0
    return adj


def _build_same_person_connectivity(num_persons, num_joints):
    num_tokens = int(num_persons) * int(num_joints)
    conn = torch.zeros((num_tokens, num_tokens), dtype=torch.float32)
    for person_idx in range(int(num_persons)):
        start = person_idx * int(num_joints)
        stop = start + int(num_joints)
        conn[start:stop, start:stop] = 1.0
    return conn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(256, 256), dropout=0.0):
        super(MLP, self).__init__()
        dims = (int(in_dim),) + tuple(int(dim) for dim in hidden_dims) + (int(out_dim),)
        layers = []
        for idx in range(len(dims) - 1):
            if idx > 0:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, value):
        return self.net(value)


class PositionalEmbedding(nn.Module):
    def __init__(self, num_persons, num_joints, hidden_dim, dropout=0.0):
        super(PositionalEmbedding, self).__init__()
        self.num_persons = int(num_persons)
        self.num_joints = int(num_joints)
        self.person = nn.Parameter(torch.zeros(self.num_persons, int(hidden_dim)))
        self.joint = nn.Parameter(torch.zeros(self.num_joints, int(hidden_dim)))
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.person, std=0.02)
        nn.init.normal_(self.joint, std=0.02)

    def forward_spatial(self):
        person_pe = self.person.repeat_interleave(self.num_joints, dim=0)
        joint_pe = self.joint.repeat(self.num_persons, 1)
        return self.dropout(person_pe + joint_pe)

    def forward_relation(self):
        spatial = self.forward_spatial()
        return self.dropout(spatial.unsqueeze(0) + spatial.unsqueeze(1))


class RelationAwareAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.0):
        super(RelationAwareAttention, self).__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.joint_qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.relation_score = nn.Linear(self.hidden_dim, self.num_heads)
        self.relation_qk = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.attn_drop = nn.Dropout(float(dropout))
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_drop = nn.Dropout(float(dropout))

    def forward(self, joint_feature, relation_feature):
        batch_size, num_tokens, hidden_dim = joint_feature.shape
        qkv = self.joint_qkv(joint_feature)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        joint_q, joint_k, joint_v = qkv[0], qkv[1], qkv[2]

        relation_qk = self.relation_qk(relation_feature)
        relation_qk = relation_qk.view(
            batch_size,
            num_tokens,
            num_tokens,
            2,
            self.num_heads,
            self.head_dim,
        )
        relation_qk = relation_qk.permute(3, 0, 4, 1, 2, 5)
        relation_q, relation_k = relation_qk[0], relation_qk[1]

        joint_score = joint_q @ joint_k.transpose(-2, -1)
        relation_linear_score = self.relation_score(relation_feature)
        relation_linear_score = relation_linear_score.permute(0, 3, 1, 2)
        relation_quadratic_score = (
            relation_q.unsqueeze(-2) @ relation_k.unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)

        attn = (joint_score + relation_linear_score + relation_quadratic_score) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ joint_v).transpose(1, 2).contiguous().view(
            batch_size,
            num_tokens,
            hidden_dim,
        )
        return self.proj_drop(self.proj(out))


class JointRelationBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(JointRelationBlock, self).__init__()
        self.norm_joint_attn = nn.LayerNorm(int(hidden_dim))
        self.norm_relation_attn = nn.LayerNorm(int(hidden_dim))
        self.attn = RelationAwareAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm_joint = nn.LayerNorm(int(hidden_dim))
        self.joint_mlp = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )

        self.norm_relation_input = nn.LayerNorm(int(hidden_dim) * 4)
        self.relation_mlp1 = nn.Sequential(
            nn.Linear(int(hidden_dim) * 4, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )
        self.norm_relation = nn.LayerNorm(int(hidden_dim))
        self.relation_mlp2 = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, joint_feature, relation_feature):
        batch_size, num_tokens, hidden_dim = joint_feature.shape
        joint_feature = joint_feature + self.attn(
            self.norm_joint_attn(joint_feature),
            self.norm_relation_attn(relation_feature),
        )
        joint_feature = joint_feature + self.joint_mlp(self.norm_joint(joint_feature))

        joint_i = joint_feature.unsqueeze(1).expand(batch_size, num_tokens, num_tokens, hidden_dim)
        joint_j = joint_feature.unsqueeze(2).expand(batch_size, num_tokens, num_tokens, hidden_dim)
        relation_rev = relation_feature.transpose(1, 2)
        relation_input = torch.cat((relation_feature, relation_rev, joint_i, joint_j), dim=-1)
        relation_feature = relation_feature + self.relation_mlp1(
            self.norm_relation_input(relation_input)
        )
        relation_feature = relation_feature + self.relation_mlp2(
            self.norm_relation(relation_feature)
        )
        return joint_feature, relation_feature


class JointRelationTransformerXYZ(nn.Module):
    def __init__(
        self,
        obs_len=30,
        pred_len=120,
        num_persons=2,
        num_joints=24,
        coord_dim=3,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        relation_weight=1.0,
    ):
        super(JointRelationTransformerXYZ, self).__init__()
        self.model_type = "jrt_xyz"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.num_persons = int(num_persons)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.relation_weight = float(relation_weight)
        self.num_tokens = self.num_persons * self.num_joints

        self.joint_encoder = MLP(self.obs_len * self.coord_dim * 2, self.hidden_dim)
        self.relation_encoder = MLP(self.obs_len + 2, self.hidden_dim)
        self.pe = PositionalEmbedding(
            self.num_persons,
            self.num_joints,
            self.hidden_dim,
            dropout=self.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                JointRelationBlock(
                    self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.joint_decoder = MLP(self.hidden_dim, self.pred_len * self.coord_dim)
        self.relation_decoder = MLP(self.hidden_dim, self.pred_len)

        self.register_buffer(
            "adjacency",
            _build_adjacency(self.num_persons, self.num_joints, SMPL_24_EDGES),
        )
        self.register_buffer(
            "connectivity",
            _build_same_person_connectivity(self.num_persons, self.num_joints),
        )

    def _joint_input(self, obs_xyz):
        tokens = _tokenize_xyz(obs_xyz)
        velocity = torch.zeros_like(tokens)
        if self.obs_len > 1:
            velocity[:, :, 1:] = tokens[:, :, 1:] - tokens[:, :, :-1]
        return torch.cat((tokens, velocity), dim=-1).view(
            obs_xyz.shape[0],
            self.num_tokens,
            self.obs_len * self.coord_dim * 2,
        )

    def _relation_features_from_tokens(self, tokens):
        pos_i = tokens.unsqueeze(2)
        pos_j = tokens.unsqueeze(1)
        distance = torch.sqrt(((pos_i - pos_j) ** 2).sum(dim=-1).clamp_min(1e-12))
        exp_distance = torch.exp(-distance)
        batch_size = tokens.shape[0]
        adjacency = self.adjacency.to(device=tokens.device, dtype=tokens.dtype)
        connectivity = self.connectivity.to(device=tokens.device, dtype=tokens.dtype)
        relation = torch.cat(
            (
                exp_distance,
                adjacency.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1),
                connectivity.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1),
            ),
            dim=-1,
        )
        return relation

    def _target_relation(self, target_xyz):
        target_tokens = _tokenize_xyz(target_xyz)
        pos_i = target_tokens.unsqueeze(2)
        pos_j = target_tokens.unsqueeze(1)
        distance = torch.sqrt(((pos_i - pos_j) ** 2).sum(dim=-1).clamp_min(1e-12))
        return torch.exp(-distance)

    def _decode_joint(self, feature, last_obs):
        batch_size = feature.shape[0]
        offset = self.joint_decoder(feature).view(
            batch_size,
            self.num_tokens,
            self.pred_len,
            self.coord_dim,
        )
        pred_tokens = last_obs.unsqueeze(2) + offset
        return _untokenize_xyz(
            pred_tokens,
            batch_size,
            self.pred_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )

    def _decode_relation(self, feature):
        return torch.sigmoid(self.relation_decoder(feature))

    def _forward_all(self, obs_xyz):
        _check_xyz(
            "obs_xyz",
            obs_xyz,
            self.obs_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )
        batch_size = obs_xyz.shape[0]
        obs_tokens = _tokenize_xyz(obs_xyz)
        last_obs = obs_tokens[:, :, -1]
        joint_feature = self.joint_encoder(self._joint_input(obs_xyz))
        relation_feature = self.relation_encoder(self._relation_features_from_tokens(obs_tokens))

        pe_joint = self.pe.forward_spatial()
        pe_relation = self.pe.forward_relation()
        aux_pred = [self._decode_joint(joint_feature, last_obs)]
        aux_relation = [self._decode_relation(relation_feature)]

        for block in self.blocks:
            joint_feature = joint_feature + pe_joint
            relation_feature = relation_feature + pe_relation
            joint_feature, relation_feature = block(joint_feature, relation_feature)
            aux_pred.append(self._decode_joint(joint_feature, last_obs))
            aux_relation.append(self._decode_relation(relation_feature))

        pred_xyz = self._decode_joint(joint_feature, last_obs)
        pred_relation = self._decode_relation(relation_feature)
        ensure_xyz_prediction_shape(
            pred_xyz,
            batch_size=batch_size,
            pred_len=self.pred_len,
            num_persons=self.num_persons,
            num_joints=self.num_joints,
            coord_dim=self.coord_dim,
        )
        return pred_xyz, pred_relation, aux_pred, aux_relation

    def forward(self, obs_xyz):
        pred_xyz, _, _, _ = self._forward_all(obs_xyz)
        return pred_xyz

    def training_loss(self, obs_xyz, target_xyz, aux_weight=0.0, relation_weight=None, **kwargs):
        _check_xyz(
            "target_xyz",
            target_xyz,
            self.pred_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )
        pred_xyz, pred_relation, aux_pred, aux_relation = self._forward_all(obs_xyz)
        target_relation = self._target_relation(target_xyz)

        pose_loss = F.mse_loss(pred_xyz, target_xyz)
        relation_loss = F.l1_loss(pred_relation, target_relation)
        weight = self.relation_weight if relation_weight is None else float(relation_weight)
        loss = pose_loss + weight * relation_loss

        if float(aux_weight) > 0 and len(aux_pred) > 0:
            aux_pose = pred_xyz.new_tensor(0.0)
            aux_relation_loss = pred_xyz.new_tensor(0.0)
            for pred_item, relation_item in zip(aux_pred, aux_relation):
                aux_pose = aux_pose + F.mse_loss(pred_item, target_xyz)
                aux_relation_loss = aux_relation_loss + F.l1_loss(relation_item, target_relation)
            aux_pose = aux_pose / float(len(aux_pred))
            aux_relation_loss = aux_relation_loss / float(len(aux_relation))
            loss = loss + float(aux_weight) * (aux_pose + weight * aux_relation_loss)

        if not torch.isfinite(loss):
            raise ValueError("jrt_xyz loss 为非有限数值: {}".format(float(loss.detach().cpu().item())))
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
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "relation_weight": self.relation_weight,
        }
