"""
sleeve_TCN_model.py
Compact temporal-convolution baseline for sleeve EMG-to-angle regression.

Design goals:
  1) Strong temporal inductive bias via dilated residual TCN blocks
  2) Lower flexibility than the attention-heavy baseline to reduce overfitting
  3) Lightweight temporal attention pooling only at the end
  4) Same I/O convention as the existing sleeve models: (B, 128, T) -> (B, 14)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length padding.")

        pad = dilation * (kernel_size // 2)
        self.dw1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)

        self.dw2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

        self.se = TemporalSE(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x):
        residual = x

        x = self.dw1(x)
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.dw2(x)
        x = self.pw2(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.drop(x)

        return self.act(x + residual)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = max(1, channels // 2)
        self.score = nn.Sequential(
            nn.Conv1d(channels, mid, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(mid, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, C, T)
        logits = self.score(x)  # (B, 1, T)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(x * weights, dim=-1)
        return pooled


class KinematicCouplingHead(nn.Module):
    def __init__(self, in_dim: int = 128, n_joints: int = 14):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_joints)
        self.coupling = nn.Parameter(torch.eye(n_joints) * 0.1)

    def forward(self, x):
        raw = self.fc(x)
        return raw + raw @ self.coupling.T


class SleeveTCNRegressor(nn.Module):
    def __init__(
        self,
        n_ch: int = 128,
        window_size: int = 400,
        n_joints: int = 14,
        hidden: int = 128,
        kernel_size: int = 5,
        dilations=(1, 2, 4, 8),
        dropout: float = 0.15,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.n_ch = n_ch
        self.window_size = window_size
        self.n_joints = n_joints
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.dilations = tuple(int(d) for d in dilations)

        pad = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(
                n_ch,
                n_ch,
                kernel_size=kernel_size,
                padding=pad,
                groups=n_ch,
                bias=False,
            ),
            nn.BatchNorm1d(n_ch),
            nn.ELU(),
            nn.Conv1d(n_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=hidden,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in self.dilations
            ]
        )

        self.pool = TemporalAttentionPooling(hidden)
        self.head_mlp = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ELU(),
            nn.Dropout(min(0.5, dropout * 2.0)),
        )
        self.head = KinematicCouplingHead(in_dim=128, n_joints=n_joints)

    def forward(self, x):
        bsz, ch, timesteps = x.shape
        if ch != self.n_ch:
            raise RuntimeError(f"Expected {self.n_ch} channels, got {ch}")
        if timesteps != self.window_size:
            raise RuntimeError(
                f"Expected window size {self.window_size}, got {timesteps}"
            )

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        x = self.pool(x)
        x = self.head_mlp(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SleeveTCNRegressor(window_size=400, n_ch=128, n_joints=14)
    dummy = torch.randn(8, 128, 400)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {model.count_params():,}")
