"""
model.py
Student-style MS-Attention baseline ported to PyTorch, with targeted additions:
  1) Circular electrode convolution front-end
  2) Multi-scale kernels (3, 5, 7, 9)
  3) Kinematic coupling output head
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularElectrodeConv(nn.Module):
    def __init__(self, out_ch=16, elec_k=3, time_k=5):
        super().__init__()
        self.elec_pad = elec_k // 2
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_ch,
            kernel_size=(elec_k, time_k),
            padding=(0, time_k // 2),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        if self.elec_pad > 0:
            x = torch.cat(
                [x[:, :, -self.elec_pad :, :], x, x[:, :, : self.elec_pad, :]], dim=2
            )
        x = self.conv(x)
        x = self.bn(x)
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
        x = torch.cat([b(x) for b in self.branches], dim=1)
        x = self.act(self.bn(x))
        return self.proj(x)


class AttentionBlock(nn.Module):
    def __init__(self, d_model=192, n_heads=3, dropout=0.1, ff_mult=4):
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
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1200):
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
    def __init__(self, in_dim=128, n_joints=22):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_joints)
        self.coupling = nn.Parameter(torch.eye(n_joints) * 0.1)

    def forward(self, x):
        raw = self.fc(x)
        return raw + raw @ self.coupling.T


class CNNAttentionImproved(nn.Module):
    def __init__(
        self,
        n_ch=12,
        window_size=400,
        n_joints=22,
        hidden=256,
        n_attn=5,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by n_heads ({n_heads})"
            )

        self.n_ch = n_ch
        self.window_size = window_size
        self.hidden = hidden
        self.n_attn = n_attn
        self.n_heads = n_heads

        self.circ = CircularElectrodeConv(out_ch=16, elec_k=3, time_k=5)
        self.pre = nn.Sequential(
            nn.Conv1d(16 * n_ch, hidden, kernel_size=1, bias=False),
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

    def forward(self, x):
        bsz, ch, timesteps = x.shape
        if ch != self.n_ch:
            raise RuntimeError(f"Expected {self.n_ch} channels, got {ch}")

        x = self.circ(x.unsqueeze(1))
        x = x.reshape(bsz, -1, timesteps)
        x = self.pre(x)

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
            loss = loss + self.ls * (diff.pow(2).mean())
        return loss


if __name__ == "__main__":
    model = CNNAttentionImproved(window_size=400, n_joints=22)
    dummy = torch.randn(8, 12, 400)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {model.count_params():,}")
