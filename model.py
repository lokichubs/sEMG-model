"""
model.py
Defines CNNAttentionImproved — callable nn.Module.
Improvements over Geng et al. 2022:
  1. Circular electrode conv front-end
  2. Multi-scale depthwise separable conv (3,5,7,11)
  3. Deeper attention stack (n=5 vs n=3)
  4. Kinematic coupling output head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CIRCULAR ELECTRODE CONV ──────────────────────────────────
class CircularElectrodeConv(nn.Module):
    """
    2D conv where electrode dim (dim=2) is padded circularly.
    Models the ring geometry of forearm electrode arrays.

    Input:  (B, 1, 12, T)
    Output: (B, out_ch, 12, T)
    """
    def __init__(self, out_ch=32, elec_k=3, time_k=5):
        super().__init__()
        self.ep   = elec_k // 2
        self.conv = nn.Conv2d(1, out_ch, (elec_k, time_k),
                              padding=(0, time_k//2), bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # x: (B, 1, 12, T) — circular pad electrode dim
        x = torch.cat([x[:,:,-self.ep:,:], x, x[:,:,:self.ep,:]], dim=2)
        return F.elu(self.bn(self.conv(x)))


# ── DEPTHWISE SEPARABLE CONV ─────────────────────────────────
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, pad):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, k, padding=pad,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.elu(self.bn(self.pw(self.dw(x))))


# ── MULTI-SCALE CONV BLOCK ───────────────────────────────────
class MultiScaleBlock(nn.Module):
    """4 parallel DSConv paths (k=3,5,7,11) → concat → project."""
    def __init__(self, in_ch, hidden=64):
        super().__init__()
        bch = hidden // 4
        self.branches = nn.ModuleList([
            DSConv(in_ch, bch, k, k//2) for k in [3, 5, 7, 11]
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ELU()
        )

    def forward(self, x):
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))


# ── ATTENTION BLOCK ──────────────────────────────────────────
class AttentionBlock(nn.Module):
    """Single multi-head self-attention + feed-forward layer."""
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + self.drop(a))
        x    = self.norm2(x + self.drop(self.ff(x)))
        return x


# ── POSITIONAL ENCODING ──────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


# ── KINEMATIC COUPLING HEAD ──────────────────────────────────
class KinematicCouplingHead(nn.Module):
    """
    Linear head + learned coupling matrix.
    Encourages anatomically consistent distal/proximal co-variation.
    """
    def __init__(self, d_model=128, n_joints=10):
        super().__init__()
        self.fc       = nn.Linear(d_model, n_joints)
        self.coupling = nn.Parameter(torch.eye(n_joints) * 0.1)

    def forward(self, x):
        raw = self.fc(x)
        return raw + (raw @ self.coupling.T)


# ── FULL MODEL ───────────────────────────────────────────────
class CNNAttentionImproved(nn.Module):
    """
    Improved CNN-Attention for continuous finger kinematics from sEMG.

    Usage:
        model = CNNAttentionImproved()
        pred  = model(emg)   # emg: (B, 12, 50) → pred: (B, 10)

    Args:
        n_ch         : EMG channels (12 for Ninapro DB2)
        window_size  : samples per window (50 = 25ms @ 2kHz)
        n_joints     : joint angles to predict (10)
        hidden       : internal feature dimension
        n_attn       : stacked attention layers (paper=3, we use 5)
        n_heads      : attention heads
        dropout      : dropout rate
    """
    def __init__(self, n_ch=12, window_size=50, n_joints=10,
                 hidden=128, n_attn=5, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_ch        = n_ch
        self.window_size = window_size

        # Front end: circular electrode conv
        self.circ   = CircularElectrodeConv(out_ch=32, elec_k=3, time_k=5)
        self.proj   = nn.Sequential(
            nn.Conv1d(32 * n_ch, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden), nn.ELU()
        )

        # Two multi-scale conv blocks with pooling between them
        self.ms1    = MultiScaleBlock(hidden, hidden)
        self.pool   = nn.AvgPool1d(2, stride=2)
        self.ms2    = MultiScaleBlock(hidden, hidden)

        # Positional encoding + stacked attention
        self.pos_enc = PositionalEncoding(hidden, dropout)
        self.attn_layers = nn.ModuleList([
            AttentionBlock(hidden, n_heads, dropout) for _ in range(n_attn)
        ])

        # Output
        self.head = KinematicCouplingHead(hidden, n_joints)

    def forward(self, x):
        """x: (B, 12, T) → (B, 10)"""
        B, C, T = x.shape

        # Circular electrode conv
        x = self.circ(x.unsqueeze(1))          # (B, 32, 12, T)
        x = x.reshape(B, -1, T)                # (B, 32*12, T)
        x = self.proj(x)                        # (B, hidden, T)

        # Multi-scale temporal conv
        x = self.ms1(x)                         # (B, hidden, T)
        x = self.pool(x)                        # (B, hidden, T/2)
        x = self.ms2(x)                         # (B, hidden, T/2)

        # Attention
        x = self.pos_enc(x.permute(0,2,1))     # (B, T/2, hidden)
        for layer in self.attn_layers:
            x = layer(x)

        x = x.mean(dim=1)                       # (B, hidden) — global avg pool
        return self.head(x)                     # (B, 10)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── LOSS ─────────────────────────────────────────────────────
class KinematicLoss(nn.Module):
    """
    MSE + optional smoothness penalty on consecutive predictions.

    Args:
        lambda_smooth : weight for temporal smoothness term
    """
    def __init__(self, lambda_smooth=0.01):
        super().__init__()
        self.ls  = lambda_smooth
        self.mse = nn.MSELoss()

    def forward(self, pred, target, pred_seq=None):
        """
        pred      : (B, 10)
        target    : (B, 10)
        pred_seq  : (T, B, 10) optional — consecutive preds for smoothness
        """
        loss = self.mse(pred, target)
        if pred_seq is not None and self.ls > 0:
            diff = pred_seq[1:] - pred_seq[:-1]
            loss = loss + self.ls * (diff ** 2).mean()
        return loss


# ── QUICK SANITY CHECK ───────────────────────────────────────
if __name__ == "__main__":
    model = CNNAttentionImproved()
    dummy = torch.randn(8, 12, 50)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {model.count_params():,}")