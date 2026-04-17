"""
sleeve_model.py
Same MS-Attention architecture as the Ninapro model, but with sleeve-oriented
defaults:
  1) 128 EMG input channels
  2) 14 finger-angle outputs
  3) same circular conv + multi-scale conv + attention + coupling head stack
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_activation(name):
    key = str(name).strip().lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    if key in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported activation: {name}")


def normalize_pos_embedding(name):
    key = "sinusoidal" if name is None else str(name).strip().lower()
    aliases = {
        "sin": "sinusoidal",
        "sine": "sinusoidal",
        "sinusoidal": "sinusoidal",
        "rope": "rope",
        "rotary": "rope",
        "rotary_embedding": "rope",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported positional embedding: {name}")
    return aliases[key]


class CircularElectrodeConv(nn.Module):
    def __init__(self, out_ch=16, elec_k=3, time_k=5, activation="elu"):
        super().__init__()
        self.elec_pad = elec_k // 2
        self.act = build_activation(activation)
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
        return self.act(x)


class MultiScaleConv1D(nn.Module):
    def __init__(
        self,
        in_ch,
        kernels=(3, 5, 7, 9),
        branch_ch=48,
        out_ch=192,
        activation="elu",
    ):
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
        self.act = build_activation(activation)
        self.proj = nn.Sequential(
            nn.Conv1d(merged, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            build_activation(activation),
        )

    def forward(self, x):
        x = torch.cat([b(x) for b in self.branches], dim=1)
        x = self.act(self.bn(x))
        return self.proj(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model=192,
        n_heads=4,
        dropout=0.15,
        ff_mult=4,
        ff_activation="elu",
        pos_embedding="sinusoidal",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.pos_embedding = normalize_pos_embedding(pos_embedding)
        self.attn_dropout = float(dropout)
        if self.pos_embedding == "rope" and self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even per-head dimension, got {self.head_dim}"
            )

        if self.pos_embedding == "rope":
            self.attn = None
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.rotary = RotaryPositionalEmbedding(self.head_dim)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None
            self.rotary = None
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            build_activation(ff_activation),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x):
        if self.pos_embedding == "rope":
            bsz, seq_len, _ = x.shape
            q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
            k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
            v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = self.rotary(q, k)
            a = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
            a = a.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
            a = self.out_proj(a)
        else:
            a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / float(head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)

    def _cos_sin(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def forward(self, q, k):
        cos, sin = self._cos_sin(q.size(-2), q.device, q.dtype)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k


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


class SleeveCNNAttentionImproved(nn.Module):
    def __init__(
        self,
        n_ch=128,
        window_size=400,
        n_joints=14,
        hidden=256,
        n_attn=4,
        n_heads=2,
        dropout=0.15,
        use_circ=True,
        cnn_activation="elu",
        attn_ff_activation="elu",
        mlp_activation="elu",
        pos_embedding="sinusoidal",
    ):
        super().__init__()
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
        self.use_circ = bool(use_circ)
        self.cnn_activation = str(cnn_activation)
        self.attn_ff_activation = str(attn_ff_activation)
        self.mlp_activation = str(mlp_activation)
        self.pos_embedding = normalize_pos_embedding(pos_embedding)

        if self.use_circ:
            self.circ = CircularElectrodeConv(
                out_ch=16,
                elec_k=3,
                time_k=5,
                activation=self.cnn_activation,
            )
            pre_in_ch = 16 * n_ch
        else:
            self.circ = None
            pre_in_ch = n_ch
        self.pre = nn.Sequential(
            nn.Conv1d(pre_in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            build_activation(self.cnn_activation),
        )

        self.ms1 = MultiScaleConv1D(
            in_ch=hidden,
            kernels=(3, 5, 7, 9),
            branch_ch=48,
            out_ch=hidden,
            activation=self.cnn_activation,
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.ms2 = MultiScaleConv1D(
            in_ch=hidden,
            kernels=(3, 5, 7, 9),
            branch_ch=48,
            out_ch=hidden,
            activation=self.cnn_activation,
        )

        self.pos_enc = (
            PositionalEncoding(hidden, dropout=dropout)
            if self.pos_embedding == "sinusoidal"
            else None
        )
        self.attn_layers = nn.ModuleList(
            [
                AttentionBlock(
                    hidden,
                    n_heads=n_heads,
                    dropout=dropout,
                    ff_activation=self.attn_ff_activation,
                    pos_embedding=self.pos_embedding,
                )
                for _ in range(n_attn)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden, 128),
            build_activation(self.mlp_activation),
            nn.Dropout(dropout * 2 if dropout * 2 <= 0.5 else 0.5),
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

        if self.use_circ:
            x = self.circ(x.unsqueeze(1))
            x = x.reshape(bsz, -1, timesteps)
        x = self.pre(x)

        x = self.ms1(x)
        x = self.pool(x)
        x = self.ms2(x)

        x = x.permute(0, 2, 1)
        if self.pos_enc is not None:
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


class BaselineKinematicsLoss(nn.Module):
    def __init__(self, n_joints=14, weight=1e-3):
        super().__init__()
        self.n_joints = int(n_joints)
        self.weight = float(weight)
        self.rules = self._build_rules(self.n_joints)

    @staticmethod
    def _build_rules(n_joints):
        if n_joints == 14:
            return [
                (2, 1, 0.90, 1.0),
                (5, 4, 0.90, 1.0),
                (8, 7, 0.90, 1.0),
                (11, 10, 0.90, 1.0),
            ]

        rules = []
        for idx in range(1, n_joints):
            rules.append((idx, idx - 1, 1.0, 0.0))
        return rules

    def forward(self, pred_angles):
        if self.weight <= 0.0 or len(self.rules) == 0:
            return pred_angles.new_tensor(0.0)

        penalties = []
        for distal_idx, proximal_idx, ratio, margin_deg in self.rules:
            proximal = pred_angles[:, proximal_idx]
            distal = pred_angles[:, distal_idx]
            upper_allowed = ratio * proximal + margin_deg
            violation = torch.relu(distal - upper_allowed)
            penalties.append((violation**2).mean())

        return self.weight * torch.stack(penalties).mean()


if __name__ == "__main__":
    model = SleeveCNNAttentionImproved(window_size=200, n_ch=128, n_joints=14)
    dummy = torch.randn(8, 128, 200)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {model.count_params():,}")
