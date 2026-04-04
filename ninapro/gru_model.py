"""
gru_model.py
GRU baseline regressor for EMG-to-kinematics prediction on Ninapro windows.
"""

import torch
import torch.nn as nn


class GRUKinematicsRegressor(nn.Module):
    def __init__(
        self,
        n_ch=12,
        n_joints=22,
        hidden=256,
        num_layers=2,
        dropout=0.15,
        bidirectional=False,
    ):
        super().__init__()
        self.n_ch = int(n_ch)
        self.n_joints = int(n_joints)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.dropout = float(dropout)

        gru_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.n_ch,
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=self.bidirectional,
        )

        out_dim = self.hidden * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.n_joints),
        )

    def forward(self, x):
        # Input x: (B, C, T). GRU expects (B, T, C).
        if x.ndim != 3:
            raise RuntimeError(f"Expected 3D input (B,C,T), got {tuple(x.shape)}")
        if x.size(1) != self.n_ch:
            raise RuntimeError(f"Expected {self.n_ch} channels, got {x.size(1)}")

        seq = x.transpose(1, 2)
        out, _ = self.gru(seq)
        features = out[:, -1, :]
        return self.head(features)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GRUKinematicsRegressor(n_ch=12, n_joints=22)
    dummy = torch.randn(8, 12, 400)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {model.count_params():,}")
