import math

import numpy as np
import torch
from torch import nn


def _build_dct_matrices(seq_len, dct_n, device, dtype):
    if dct_n > seq_len:
        raise ValueError("dct_n 不能大于 seq_len")
    n = torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len)
    k = torch.arange(seq_len, device=device, dtype=dtype).view(seq_len, 1)
    matrix = torch.cos(math.pi / float(seq_len) * (n + 0.5) * k)
    matrix[0] = matrix[0] * math.sqrt(1.0 / float(seq_len))
    if seq_len > 1:
        matrix[1:] = matrix[1:] * math.sqrt(2.0 / float(seq_len))
    dct = matrix[:dct_n].contiguous()
    idct = dct.transpose(0, 1).contiguous()
    return dct, idct


def _check_xyz_obs(obs, obs_len, num_persons, num_joints, coord_dim):
    if obs.dim() != 5:
        raise ValueError("obs_xyz 必须是 [B,T,2,24,3]，当前维度数为 {}".format(obs.dim()))
    expected_tail = (obs_len, num_persons, num_joints, coord_dim)
    if tuple(obs.shape[1:]) != expected_tail:
        raise ValueError(
            "obs_xyz 后四维必须是 {}，当前为 {}".format(expected_tail, tuple(obs.shape[1:]))
        )
    if not torch.isfinite(obs).all():
        raise ValueError("obs_xyz 存在非有限数值")


def ensure_xyz_prediction_shape(pred, batch_size, pred_len, num_persons=2, num_joints=24, coord_dim=3):
    expected = (int(batch_size), int(pred_len), int(num_persons), int(num_joints), int(coord_dim))
    if tuple(pred.shape) != expected:
        raise ValueError("pred_xyz shape 必须是 {}，当前为 {}".format(expected, tuple(pred.shape)))
    if not torch.isfinite(pred).all():
        raise ValueError("pred_xyz 存在非有限数值")


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _official_get_dct_matrix(seq_len):
    dct_m = np.eye(seq_len)
    for k in np.arange(seq_len):
        for i in np.arange(seq_len):
            weight = np.sqrt(2.0 / seq_len)
            if k == 0:
                weight = np.sqrt(1.0 / seq_len)
            dct_m[k, i] = weight * np.cos(np.pi * (i + 0.5) * k / seq_len)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


class AuxilliaryEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(AuxilliaryEncoder, self).__init__(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=norm,
        )

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        aux_output = []

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            aux_output.append(output)

        if self.norm is not None:
            output = self.norm(output)

        return output, aux_output


class LearnedDoublePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, num_joints=39):
        super(LearnedDoublePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learned_encoding = nn.Embedding(num_joints, d_model // 2, max_norm=True)
        self.person_encoding = nn.Embedding(1000, d_model // 2, max_norm=True)

    def forward(self, x, num_people=1):
        num_joints = x.size(0) // num_people
        half = x.size(2) // 2
        joint_ids = torch.arange(num_joints, device=x.device).repeat(num_people)
        person_ids = torch.arange(num_people, device=x.device).repeat_interleave(num_joints, dim=0)
        x[:, :, 0 : half * 2 : 2] = x[:, :, 0 : half * 2 : 2] + self.learned_encoding(
            joint_ids
        ).unsqueeze(1)
        x[:, :, 1 : half * 2 : 2] = x[:, :, 1 : half * 2 : 2] + self.person_encoding(
            person_ids
        ).unsqueeze(1)
        return self.dropout(x)


class OfficialSoMoFormerXYZ(nn.Module):
    def __init__(
        self,
        obs_len=30,
        pred_len=120,
        num_persons=2,
        num_joints=24,
        coord_dim=3,
        dct_n=30,
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        output_scale=1.0,
        location_method="grid",
        grid_len=3,
        grid_emb_size=8,
        normalize_inputs=False,
        residual_connection=True,
        learned_embedding=True,
    ):
        super(OfficialSoMoFormerXYZ, self).__init__()
        self.model_type = "official_somoformer_xyz"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.seq_len = self.obs_len + self.pred_len
        self.num_persons = int(num_persons)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        self.dct_n = int(dct_n)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.activation = activation
        self.output_scale = float(output_scale)
        self.location_method = location_method
        self.grid_len = int(grid_len)
        self.grid_emb_size = int(grid_emb_size)
        self.normalize_inputs = bool(normalize_inputs)
        self.residual_connection = bool(residual_connection)
        self.learned_embedding = bool(learned_embedding)

        dct_m, idct_m = _official_get_dct_matrix(self.seq_len)
        self.register_buffer("dct", torch.from_numpy(dct_m[: self.dct_n]).float())
        self.register_buffer("idct", torch.from_numpy(idct_m[:, : self.dct_n]).float())

        if self.location_method == "naive":
            self.fc_in = nn.Linear(self.dct_n, self.hidden_dim)
            self.double_id_encoder = LearnedDoublePositionalEncoding(
                self.hidden_dim,
                self.dropout,
                num_joints=self.num_joints * self.coord_dim,
            )
        elif self.location_method == "neck":
            self.fc_in = nn.Linear(self.dct_n, self.hidden_dim - self.coord_dim)
            self.double_id_encoder = LearnedDoublePositionalEncoding(
                self.hidden_dim - self.coord_dim,
                self.dropout,
                num_joints=self.num_joints * self.coord_dim,
            )
        elif self.location_method == "grid":
            self.x_embed = nn.Embedding(self.grid_len, self.grid_emb_size)
            self.y_embed = nn.Embedding(self.grid_len, self.grid_emb_size)
            self.fc_in = nn.Linear(self.dct_n, self.hidden_dim - self.grid_emb_size)
            self.double_id_encoder = LearnedDoublePositionalEncoding(
                self.hidden_dim - self.grid_emb_size,
                self.dropout,
                num_joints=self.num_joints * self.coord_dim,
            )
        else:
            raise ValueError("unsupported location_method: {}".format(self.location_method))

        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([self.hidden_dim])))
        self.fc_out = nn.Linear(self.hidden_dim, self.dct_n)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.transformer = AuxilliaryEncoder(encoder_layer, num_layers=self.num_layers)

        self.register_buffer(
            "m",
            torch.Tensor(
                [
                    0.0000,
                    -0.5622,
                    0.0000,
                    0.0000,
                    -0.5695,
                    0.0000,
                    0.0000,
                    -0.8992,
                    0.0000,
                    0.0000,
                    -0.9009,
                    0.0000,
                    0.0000,
                    -1.3000,
                    0.0000,
                    0.0000,
                    -1.2956,
                    0.0000,
                    0.0000,
                    0.0017,
                    0.0000,
                    0.0000,
                    -0.0681,
                    0.0000,
                    0.0000,
                    -0.0691,
                    0.0000,
                    0.0000,
                    -0.2849,
                    0.0000,
                    0.0000,
                    -0.2850,
                    0.0000,
                    0.0000,
                    -0.4265,
                    0.0000,
                    0.0000,
                    -0.4109,
                    0.0000,
                ]
            ),
        )
        self.register_buffer(
            "s",
            torch.Tensor(
                [
                    0.4289,
                    0.1349,
                    0.4289,
                    0.4286,
                    0.1340,
                    0.4286,
                    0.4425,
                    0.1974,
                    0.4425,
                    0.4409,
                    0.1984,
                    0.4409,
                    0.4587,
                    0.2031,
                    0.4587,
                    0.4537,
                    0.2088,
                    0.4537,
                    0.4200,
                    0.1238,
                    0.4200,
                    0.4410,
                    0.1253,
                    0.4410,
                    0.4381,
                    0.1261,
                    0.4381,
                    0.4640,
                    0.1400,
                    0.4640,
                    0.4635,
                    0.1439,
                    0.4635,
                    0.4836,
                    0.2115,
                    0.4836,
                    0.4835,
                    0.2315,
                    0.4835,
                ]
            )
            * 5.0,
        )
        self.register_buffer(
            "dct_m",
            torch.Tensor(
                [
                    -7.1789e-02,
                    4.1777e-02,
                    2.0464e-02,
                    4.6075e-03,
                    6.3129e-04,
                    2.2184e-03,
                    2.2500e-03,
                    6.2809e-04,
                    1.6267e-04,
                    7.9969e-04,
                    7.8669e-04,
                    1.4957e-04,
                    5.4564e-05,
                    4.1828e-04,
                    3.6941e-04,
                    1.3337e-05,
                    2.7676e-05,
                    2.8088e-04,
                    2.1341e-04,
                    -3.4786e-05,
                    5.6285e-06,
                    1.8702e-04,
                    1.1101e-04,
                    -6.3937e-05,
                    4.1189e-06,
                    1.4035e-04,
                    4.3991e-05,
                    -9.2744e-05,
                    4.7437e-07,
                    1.1380e-04,
                ]
            )[: self.dct_n],
        )
        self.register_buffer(
            "dct_s",
            torch.Tensor(
                [
                    0.5954,
                    0.5319,
                    0.2537,
                    0.0897,
                    0.0567,
                    0.0498,
                    0.0459,
                    0.0383,
                    0.0321,
                    0.0262,
                    0.0189,
                    0.0091,
                    0.0072,
                    0.0085,
                    0.0073,
                    0.0042,
                    0.0036,
                    0.0050,
                    0.0039,
                    0.0027,
                    0.0022,
                    0.0036,
                    0.0024,
                    0.0025,
                    0.0017,
                    0.0029,
                    0.0016,
                    0.0024,
                    0.0015,
                    0.0026,
                ]
            )[: self.dct_n],
        )

    def _active_token_count(self, num_people):
        return int(num_people) * self.num_joints * self.coord_dim

    def dct_forward(self, value):
        tgt_dct = self.dct @ value.reshape(self.dct.shape[1], -1)
        tgt_dct = tgt_dct.reshape(-1, *value.shape[1:]).permute(2, 1, 0)
        tgt_dct = (tgt_dct - self.dct_m[: self.dct_n]) / self.dct_s[: self.dct_n]
        return tgt_dct

    def dct_backward(self, value):
        value = (value * self.dct_s[: self.dct_n]) + self.dct_m[: self.dct_n]
        value = value.transpose(0, 2)
        out_dct = self.idct @ value.flatten(1)
        out_dct = out_dct.reshape(-1, *value.shape[1:])
        return out_dct

    def _make_padding_mask(self, batch_size, device):
        return torch.zeros(batch_size, self.num_persons, dtype=torch.bool, device=device)

    def _make_root_location(self, obs_xyz):
        return obs_xyz[:, 0, :, 0, :].contiguous()

    def forward_full(self, obs_xyz, padding_mask=None, metamask=False):
        _check_xyz_obs(obs_xyz, self.obs_len, self.num_persons, self.num_joints, self.coord_dim)
        batch_size = obs_xyz.shape[0]
        if padding_mask is None:
            padding_mask = self._make_padding_mask(batch_size, obs_xyz.device)
        root_location = self._make_root_location(obs_xyz)
        tgt = obs_xyz.reshape(
            batch_size,
            self.obs_len,
            self.num_persons * self.num_joints,
            self.coord_dim,
        )
        out, aux_out = self._forward_official(tgt, root_location, padding_mask, metamask=metamask)
        out = out.reshape(
            batch_size,
            self.seq_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )
        aux_out = [
            aux.reshape(
                batch_size,
                self.seq_len,
                self.num_persons,
                self.num_joints,
                self.coord_dim,
            )
            for aux in aux_out
        ]
        return out, aux_out

    def _forward_official(self, tgt, tgt_neck, padding_mask, metamask=False):
        batch_size, in_frames, num_person_joints, coord_dim = tgt.shape
        full_frames = self.seq_len
        num_joints = self.num_joints
        out_frames = full_frames - in_frames
        num_people = num_person_joints // num_joints
        num_keys = num_joints * coord_dim

        pad_idx = np.repeat([in_frames - 1], out_frames)
        frame_idx = np.append(np.arange(0, in_frames), pad_idx)
        tgt = tgt[:, frame_idx]
        tgt = tgt.flatten(-2).transpose(0, 1)

        if self.normalize_inputs:
            if self.num_joints * self.coord_dim != int(self.m.numel()):
                raise ValueError("官方 normalize_inputs 统计只适配 13 joints，不适配当前 SMPL 24 joints")
            tgt = (tgt - self.m.repeat(num_people)) / self.s.repeat(num_people)

        if self.location_method == "grid":
            necks_concat = tgt_neck.reshape(batch_size, num_people, coord_dim)[:, :, [0, 2]]
            necks_concat = (torch.clamp(necks_concat, -3, 3) + 3) / 6
            pose_grid = torch.floor((necks_concat * self.grid_len) / self.grid_len).long()
            grid_emb = self.x_embed(pose_grid[:, :, 0].long()) + self.y_embed(
                pose_grid[:, :, 1].long()
            )
            grid_emb = grid_emb.transpose(0, 1).repeat_interleave(num_joints * coord_dim, dim=0)

        tgt_dct = self.dct_forward(tgt)

        if metamask:
            token_count = tgt_dct.shape[0]
            mask_percent = 0.05
            meta_masks = (
                torch.rand((token_count, batch_size, 1), device=tgt_dct.device).float()
                > mask_percent
            ).float()
            constant_tensor = torch.ones_like(tgt_dct).float()
            tgt_dct = tgt_dct * meta_masks + tgt_dct * constant_tensor * (1 - meta_masks)

        encoded = self.fc_in(tgt_dct)
        encoded = self.double_id_encoder(encoded, num_people=num_people)

        if self.location_method == "grid":
            encoded = torch.cat((encoded, grid_emb), dim=2)
        elif self.location_method == "neck":
            necks = tgt_neck.reshape(batch_size, num_people, coord_dim)
            necks = necks.transpose(0, 1).repeat_interleave(num_joints * coord_dim, dim=0)
            encoded = torch.cat((encoded, necks), dim=2)
        elif self.location_method == "naive":
            pass

        tgt_padding_mask = padding_mask.repeat_interleave(num_keys, dim=1)
        out, aux_out = self.transformer(
            encoded,
            mask=None,
            src_key_padding_mask=tgt_padding_mask,
        )
        out = self.fc_out(out)
        aux_out = [self.fc_out(aux) for aux in aux_out]

        if self.residual_connection:
            out = out * self.output_scale + tgt_dct
            aux_out = [aux * self.output_scale + tgt_dct for aux in aux_out]

        out = self.dct_backward(out)
        aux_out = [self.dct_backward(aux) for aux in aux_out]

        if self.normalize_inputs:
            out = (out * self.s.repeat(num_people)) + self.m.repeat(num_people)
            aux_out = [(aux * self.s.repeat(num_people)) + self.m.repeat(num_people) for aux in aux_out]

        out = out.transpose(0, 1).reshape(batch_size, full_frames, num_person_joints, coord_dim)
        aux_out = [
            aux.transpose(0, 1).reshape(batch_size, full_frames, num_person_joints, coord_dim)
            for aux in aux_out
        ]
        return out, aux_out

    def forward(self, obs_xyz):
        full, _ = self.forward_full(obs_xyz, padding_mask=None, metamask=False)
        pred = full[:, self.obs_len :].contiguous()
        ensure_xyz_prediction_shape(
            pred,
            batch_size=obs_xyz.shape[0],
            pred_len=self.pred_len,
            num_persons=self.num_persons,
            num_joints=self.num_joints,
            coord_dim=self.coord_dim,
        )
        return pred

    def training_loss(self, obs_xyz, target_xyz, aux_weight=0.2, metamask=True):
        full, aux_out = self.forward_full(obs_xyz, padding_mask=None, metamask=metamask)
        pred = full[:, self.obs_len :].contiguous()
        ensure_xyz_prediction_shape(
            pred,
            batch_size=obs_xyz.shape[0],
            pred_len=self.pred_len,
            num_persons=self.num_persons,
            num_joints=self.num_joints,
            coord_dim=self.coord_dim,
        )
        loss = torch.nn.functional.mse_loss(pred, target_xyz)
        if len(aux_out) > 0 and aux_weight > 0:
            aux_losses = []
            for aux in aux_out:
                aux_pred = aux[:, self.obs_len :].contiguous()
                aux_losses.append(torch.nn.functional.mse_loss(aux_pred, target_xyz))
            loss = loss + float(aux_weight) * sum(aux_losses)
            loss = loss / (float(aux_weight) * float(len(aux_losses)) + 1.0)
        return loss

    def config(self):
        return {
            "model_type": self.model_type,
            "obs_len": self.obs_len,
            "pred_len": self.pred_len,
            "num_persons": self.num_persons,
            "num_joints": self.num_joints,
            "coord_dim": self.coord_dim,
            "dct_n": self.dct_n,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
            "output_scale": self.output_scale,
            "location_method": self.location_method,
            "grid_len": self.grid_len,
            "grid_emb_size": self.grid_emb_size,
            "normalize_inputs": self.normalize_inputs,
            "residual_connection": self.residual_connection,
            "learned_embedding": self.learned_embedding,
        }


class JointSpaceSoMoFormer(nn.Module):
    def __init__(
        self,
        obs_len=30,
        pred_len=120,
        num_persons=2,
        num_joints=24,
        coord_dim=3,
        dct_n=30,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        residual_connection=True,
    ):
        super(JointSpaceSoMoFormer, self).__init__()
        self.model_type = "somoformer_xyz"
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.seq_len = self.obs_len + self.pred_len
        self.num_persons = int(num_persons)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        self.dct_n = int(dct_n)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.residual_connection = bool(residual_connection)
        self.num_tokens = self.num_persons * self.num_joints * self.coord_dim

        self.fc_in = nn.Linear(self.dct_n, self.hidden_dim)
        self.person_embedding = nn.Embedding(self.num_persons, self.hidden_dim)
        self.joint_embedding = nn.Embedding(self.num_joints, self.hidden_dim)
        self.coord_embedding = nn.Embedding(self.coord_dim, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.fc_out = nn.Linear(self.hidden_dim, self.dct_n)

        person_ids = torch.arange(self.num_persons).view(self.num_persons, 1, 1)
        person_ids = person_ids.expand(self.num_persons, self.num_joints, self.coord_dim)
        joint_ids = torch.arange(self.num_joints).view(1, self.num_joints, 1)
        joint_ids = joint_ids.expand(self.num_persons, self.num_joints, self.coord_dim)
        coord_ids = torch.arange(self.coord_dim).view(1, 1, self.coord_dim)
        coord_ids = coord_ids.expand(self.num_persons, self.num_joints, self.coord_dim)
        self.register_buffer("person_ids", person_ids.reshape(-1).long())
        self.register_buffer("joint_ids", joint_ids.reshape(-1).long())
        self.register_buffer("coord_ids", coord_ids.reshape(-1).long())

    def _dct_forward(self, sequence):
        dct, _ = _build_dct_matrices(
            self.seq_len,
            self.dct_n,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        coeff = torch.einsum("kf,fbt->kbt", dct, sequence)
        return coeff.permute(2, 1, 0).contiguous()

    def _dct_backward(self, coeff):
        _, idct = _build_dct_matrices(
            self.seq_len,
            self.dct_n,
            device=coeff.device,
            dtype=coeff.dtype,
        )
        values = coeff.permute(2, 1, 0).contiguous()
        return torch.einsum("fk,kbt->fbt", idct, values)

    def forward(self, obs_xyz):
        _check_xyz_obs(obs_xyz, self.obs_len, self.num_persons, self.num_joints, self.coord_dim)
        batch_size = obs_xyz.shape[0]
        out_len = self.pred_len
        last = obs_xyz[:, -1:].expand(-1, out_len, -1, -1, -1)
        padded = torch.cat([obs_xyz, last], dim=1)
        flat = padded.view(batch_size, self.seq_len, self.num_tokens).permute(1, 0, 2).contiguous()

        dct_tokens = self._dct_forward(flat)
        hidden = self.fc_in(dct_tokens)
        embedding = (
            self.person_embedding(self.person_ids)
            + self.joint_embedding(self.joint_ids)
            + self.coord_embedding(self.coord_ids)
        ).unsqueeze(1)
        hidden = hidden + embedding
        encoded = self.transformer(hidden)
        pred_coeff = self.fc_out(encoded)
        if self.residual_connection:
            pred_coeff = pred_coeff + dct_tokens

        full = self._dct_backward(pred_coeff)
        full = full.permute(1, 0, 2).contiguous()
        full = full.view(
            batch_size,
            self.seq_len,
            self.num_persons,
            self.num_joints,
            self.coord_dim,
        )
        pred = full[:, self.obs_len :].contiguous()
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
            "dct_n": self.dct_n,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "residual_connection": self.residual_connection,
        }


def create_joint_space_somoformer_from_config(config):
    if config.get("model_type") != "somoformer_xyz":
        raise ValueError("model_type 必须是 somoformer_xyz，当前为 {}".format(config.get("model_type")))
    return JointSpaceSoMoFormer(
        obs_len=config.get("obs_len", 30),
        pred_len=config.get("pred_len", 120),
        num_persons=config.get("num_persons", 2),
        num_joints=config.get("num_joints", 24),
        coord_dim=config.get("coord_dim", 3),
        dct_n=config.get("dct_n", 30),
        hidden_dim=config.get("hidden_dim", 256),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 4),
        dim_feedforward=config.get("dim_feedforward", 1024),
        dropout=config.get("dropout", 0.1),
        residual_connection=config.get("residual_connection", True),
    )
