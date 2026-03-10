"""
sleeve_geomety_model.py
Geometry-aware sleeve model with a ring-structured front-end.

Design goals:
    1) preserve the same temporal backbone as sleeve_model.py
    2) remap flat sleeve channels into a ring/slot representation
    3) apply circular processing only within each ring
    4) avoid extra cylindrical mixing across rings

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_default_ring_map():
    """
    Returns a 26 x 5 map from cylindrical sleeve positions to flat channel indices.

    The mapping follows the inferred physical layout:
      - 10 vertical strips total
      - first 2 rings have 4 electrodes
      - remaining 24 rings have 5 electrodes
      - channels are 1-based in the physical description and converted here to 0-based
    """

    strip_a = list(range(1, 14))
    strip_b = list(range(14, 27))
    strip_c = list(range(27, 40))
    strip_d = list(range(40, 53))
    strip_e = list(range(53, 65))
    strip_f = list(range(65, 77))
    strip_g = list(range(77, 90))
    strip_h = list(range(90, 103))
    strip_i = list(range(103, 116))
    strip_j = list(range(116, 129))

    rings_1_based = [
        [strip_a[0], strip_c[0], strip_g[0], strip_i[0], -1],
        [strip_b[0], strip_d[0], strip_h[0], strip_j[0], -1],
    ]

    for idx in range(1, 13):
        rings_1_based.append(
            [strip_a[idx], strip_c[idx], strip_e[idx - 1], strip_g[idx], strip_i[idx]]
        )
        rings_1_based.append(
            [strip_b[idx], strip_d[idx], strip_f[idx - 1], strip_h[idx], strip_j[idx]]
        )

    ring_map = torch.tensor(rings_1_based, dtype=torch.long)
    valid_mask = ring_map > 0
    ring_map = ring_map - 1
    ring_map[~valid_mask] = -1
    return ring_map, valid_mask


class ChannelToCylinder(nn.Module):
    def __init__(self):
        super().__init__()
        ring_map, valid_mask = build_default_ring_map()
        self.register_buffer("ring_map", ring_map)
        self.register_buffer("valid_mask", valid_mask.to(torch.float32))

        valid_idx = torch.nonzero(valid_mask, as_tuple=False)
        self.register_buffer("valid_rings", valid_idx[:, 0])
        self.register_buffer("valid_slots", valid_idx[:, 1])
        self.register_buffer(
            "valid_channels", ring_map[valid_idx[:, 0], valid_idx[:, 1]]
        )

    @property
    def n_rings(self):
        return int(self.ring_map.shape[0])

    @property
    def n_slots(self):
        return int(self.ring_map.shape[1])

    def forward(self, x):
        if x.ndim != 3:
            raise RuntimeError(f"Expected input shape (B, C, T), got {x.shape}")

        bsz, n_ch, timesteps = x.shape
        if n_ch != 128:
            raise RuntimeError(f"Expected 128 sleeve channels, got {n_ch}")

        grid = x.new_zeros((bsz, self.n_rings, self.n_slots, timesteps))
        grid[:, self.valid_rings, self.valid_slots, :] = x[:, self.valid_channels, :]
        return grid


class RingCircularConv(nn.Module):
    def __init__(self, out_ch=16, slot_k=3, time_k=5):
        super().__init__()
        self.slot_pad = slot_k // 2
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_ch,
            kernel_size=(slot_k, time_k),
            padding=(0, time_k // 2),
            bias=False,
        )

    def forward(self, x):
        if x.ndim != 4:
            raise RuntimeError(
                f"Expected ring input shape (B*R, 1, S, T), got {x.shape}"
            )
        if self.slot_pad > 0:
            x = torch.cat(
                [x[:, :, -self.slot_pad :, :], x, x[:, :, : self.slot_pad, :]], dim=2
            )
        x = self.conv(x)
        return F.relu(x)


class MultiScaleConv1D(nn.Module):
    def __init__(self, in_ch, kernels=(3, 5, 7, 9), branch_ch=48, out_ch=192):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=branch_ch,
                    kernel_size=int(k),
                    padding=int(k) // 2,
                    bias=False,
                )
                for k in kernels
            ]
        )
        merged = branch_ch * len(kernels)
        self.bn = nn.BatchNorm1d(merged)
        self.act = nn.ReLU()
        self.proj = nn.Sequential(
            nn.Conv1d(merged, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.act(self.bn(x))
        return self.proj(x)


class AttentionBlock(nn.Module):
    def __init__(self, d_model=192, n_heads=4, dropout=0.15, ff_mult=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.15, max_len=1200):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class KinematicCouplingHead(nn.Module):
    def __init__(self, in_dim=128, n_joints=14):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_joints)
        self.coupling = nn.Parameter(torch.eye(n_joints) * 0.1)

    def forward(self, x):
        raw = self.fc(x)
        return raw + raw @ self.coupling.T


class SleeveGeometryAttentionModel(nn.Module):
    def __init__(
        self,
        n_ch=128,
        window_size=400,
        n_joints=14,
        hidden=256,
        n_attn=4,
        n_heads=4,
        dropout=0.15,
        geom_ch=16,
    ):
        super().__init__()
        if n_ch != 128:
            raise ValueError(
                "This geometry model currently assumes 128 sleeve channels"
            )
        if hidden % n_heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by n_heads ({n_heads})"
            )

        self.n_ch = n_ch
        self.window_size = window_size
        self.n_joints = n_joints
        self.hidden = hidden
        self.n_attn = n_attn
        self.n_heads = n_heads
        self.geom_ch = geom_ch

        self.mapper = ChannelToCylinder()
        self.register_buffer(
            "ring_mask",
            self.mapper.valid_mask.unsqueeze(0).unsqueeze(1).unsqueeze(-1),
        )

        self.ring_circ = RingCircularConv(
            out_ch=geom_ch,
            slot_k=3,
            time_k=5,
        )

        self.pre = nn.Sequential(
            nn.Conv1d(
                geom_ch * self.mapper.n_rings * self.mapper.n_slots,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        self.ms1 = MultiScaleConv1D(
            in_ch=hidden, kernels=(3, 5, 7, 9), branch_ch=48, out_ch=hidden
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.ms2 = MultiScaleConv1D(
            in_ch=hidden, kernels=(3, 5, 7, 9), branch_ch=48, out_ch=hidden
        )

        self.pos_enc = PositionalEncoding(hidden, dropout=dropout)
        self.attn_layers = nn.ModuleList(
            [
                AttentionBlock(hidden, n_heads=n_heads, dropout=dropout)
                for _ in range(n_attn)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = KinematicCouplingHead(in_dim=128, n_joints=n_joints)

    def _geometry_frontend(self, x):
        x = self.mapper(x)
        bsz, rings, slots, timesteps = x.shape
        x = x * self.ring_mask.squeeze(1)

        x = x.reshape(bsz * rings, 1, slots, timesteps)
        x = self.ring_circ(x)
        x = x.reshape(bsz, rings, self.geom_ch, slots, timesteps)
        x = x.permute(0, 2, 1, 3, 4).reshape(
            bsz, self.geom_ch * rings * slots, timesteps
        )
        return self.pre(x)

    def forward(self, x):
        bsz, ch, timesteps = x.shape
        if ch != self.n_ch:
            raise RuntimeError(f"Expected {self.n_ch} channels, got {ch}")
        if timesteps != self.window_size:
            raise RuntimeError(
                f"Expected window size {self.window_size}, got {timesteps}"
            )

        x = self._geometry_frontend(x)
        x = self.ms1(x)
        x = self.pool(x)
        x = self.ms2(x)

        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        for layer in self.attn_layers:
            x = layer(x)

        x = x.mean(dim=1)
        x = self.mlp(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class KinematicLoss(nn.Module):
    def __init__(self, lambda_smooth=0.0):
        super().__init__()
        self.ls = float(lambda_smooth)
        self.mse = nn.MSELoss()

    def forward(self, pred, target, pred_seq=None):
        loss = self.mse(pred, target)
        if pred_seq is not None and self.ls > 0:
            diff = pred_seq[1:] - pred_seq[:-1]
            loss = loss + self.ls * diff.pow(2).mean()
        return loss


if __name__ == "__main__":
    model = SleeveGeometryAttentionModel(window_size=400, n_ch=128, n_joints=14)
    dummy = torch.randn(8, 128, 400)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {model.count_params():,}")
    print(f"Rings: {model.mapper.n_rings}, Slots: {model.mapper.n_slots}")
