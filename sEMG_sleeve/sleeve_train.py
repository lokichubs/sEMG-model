"""
sleeve_train.py
Training loop for temporal-window SleeveCNNAttentionImproved.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sleeve_model import SleeveCNNAttentionImproved
from sleeve_TCN_model import SleeveTCNRegressor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sleeve_geometry_model import SleeveGeometryAttentionModel
except ImportError:
    SleeveGeometryAttentionModel = None

# ── CONFIG ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = "processed data"
AUGMENTED_DIR = "augmented data"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 64
EPOCHS = 150
LR = 5e-5
ETA_MIN = 3e-5
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 3e-4
N_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PAPER_REPLICATION = False
EMG_TRANSFORM = "none"  # "none" | "log1p"
INPUT_MODE = "raw"  # "raw" | "rms_subframes"
RMS_SUBFRAMES = 100
INPUT_SCALER = "standard"  # "none" | "standard"
TARGET_SCALER = "standard"  # "none" | "standard" | "minmax"
TRAIN_TARGET_LAG = 0
ENABLE_LAG_SWEEP = False
LAG_SWEEP_MAX = 30
CHECKPOINT_SELECTION = "r2"  # "r2" | "lag_r2"
EARLY_STOPPING_PATIENCE = 15
DROPOUT = 0.2
NPZ_LOAD_WORKERS = 8
EXPECTED_CHANNELS = 128


MODEL_REGISTRY = {
    "baseline": {
        "cls": SleeveCNNAttentionImproved,
        "architecture": "SleeveCNNAttentionImproved",
        "default_output_dir": OUTPUT_DIR,
        "default_batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "eta_min": ETA_MIN,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "cnn_activation": "elu",
        "attn_ff_activation": "elu",
        "mlp_activation": "elu",
    },
    "tcn": {
        "cls": SleeveTCNRegressor,
        "architecture": "SleeveTCNRegressor",
        "default_output_dir": f"{OUTPUT_DIR}_tcn",
        "default_batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "eta_min": ETA_MIN,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "dropout": 0.15,
    },
}

if SleeveGeometryAttentionModel is not None:
    MODEL_REGISTRY["geometry"] = {
        "cls": SleeveGeometryAttentionModel,
        "architecture": "SleeveGeometryAttentionModel",
        "default_output_dir": f"{OUTPUT_DIR}_geometry",
        "default_batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "eta_min": ETA_MIN,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a sleeve EMG model on processed NPZ windows."
    )
    activation_choices = ("elu", "relu", "gelu", "silu", "leaky_relu")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="baseline",
        help="Which model architecture to train.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional output directory for checkpoints and logs. "
            "Relative paths are resolved from the sEMG_sleeve folder."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override. If omitted, uses a model-specific default.",
    )
    parser.add_argument(
        "--data-source",
        choices=("processed", "augmented"),
        default="processed",
        help="Choose whether to train from processed data or augmented data.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Optional custom directory containing `*_train.npz` and `*_val.npz` files. "
            "Overrides --data-source if provided."
        ),
    )
    parser.add_argument(
        "--cnn-activation",
        choices=activation_choices,
        default=None,
        help="Baseline only: activation for CNN blocks.",
    )
    parser.add_argument(
        "--attn-ff-activation",
        choices=activation_choices,
        default=None,
        help="Baseline only: activation inside attention feed-forward blocks.",
    )
    parser.add_argument(
        "--mlp-activation",
        choices=activation_choices,
        default=None,
        help="Baseline only: activation for the final MLP head.",
    )
    return parser.parse_args()


def resolve_output_dir(cli_value, model_key):
    if cli_value is None:
        return BASE_DIR / MODEL_REGISTRY[model_key]["default_output_dir"]

    out_dir = Path(cli_value)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    return out_dir


def resolve_data_dir(cli_value, data_source):
    if cli_value is not None:
        data_dir = Path(cli_value)
        if not data_dir.is_absolute():
            data_dir = BASE_DIR / data_dir
        return data_dir

    if data_source == "augmented":
        return BASE_DIR / AUGMENTED_DIR
    return BASE_DIR / PROCESSED_DIR


def resolve_training_hparams(model_key, batch_size_override=None):
    spec = MODEL_REGISTRY[model_key]
    return {
        "batch_size": int(batch_size_override or spec["default_batch_size"]),
        "epochs": int(spec["epochs"]),
        "lr": float(spec["lr"]),
        "eta_min": float(spec["eta_min"]),
        "warmup_epochs": int(spec["warmup_epochs"]),
        "weight_decay": float(spec["weight_decay"]),
        "dropout": float(spec["dropout"]),
    }


# ── DATASET (Multi-File Loader) ──────────────────────────────
class NpzDataset(Dataset):
    def __init__(self, split="train", root=None):
        root = Path(root) if root is not None else BASE_DIR / PROCESSED_DIR
        files = sorted(root.glob(f"*_{split}.npz"))
        assert len(files) > 0, f"No *_{split}.npz files found in {root}"

        def _load_npz_pair(file_path):
            with np.load(file_path, allow_pickle=False) as data:
                emg = np.asarray(data["emg"], dtype=np.float32)
                angles = np.asarray(data["angles"], dtype=np.float32)
            return emg, angles

        workers = max(1, min(int(NPZ_LOAD_WORKERS), len(files)))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                loaded_parts = list(pool.map(_load_npz_pair, files))
        else:
            loaded_parts = [_load_npz_pair(f) for f in files]

        emg_parts = [part[0] for part in loaded_parts]
        angle_parts = [part[1] for part in loaded_parts]

        self.X = np.concatenate(emg_parts, axis=0)
        self.y = np.concatenate(angle_parts, axis=0)
        print(f"[{split}] Loaded {len(files)} files → {len(self.X)} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ── METRICS ──────────────────────────────────────────────────
def compute_metrics(preds, targets):
    cc_list, rmse_list, r2_list = [], [], []
    for j in range(preds.shape[1]):
        p, t = preds[:, j], targets[:, j]
        cc, _ = pearsonr(p, t)
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2) + 1e-8
        r2 = 1 - ss_res / ss_tot
        cc_list.append(cc)
        rmse_list.append(rmse)
        r2_list.append(r2)
    return np.mean(cc_list), np.mean(rmse_list), np.mean(r2_list)


def _apply_emg_transform(emg_3d, mode):
    arr = np.asarray(emg_3d, dtype=np.float32)
    if mode == "none":
        return arr
    if mode == "log1p":
        return np.log1p(np.abs(arr)).astype(np.float32)
    raise ValueError(f"Unknown EMG_TRANSFORM: {mode}")


def _normalize_emg_shape(emg):
    arr = np.asarray(emg, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3:
        s1, s2 = arr.shape[1], arr.shape[2]
        if s1 < s2:
            return arr
        return arr.transpose(0, 2, 1)
    raise ValueError(f"Unexpected EMG shape: {arr.shape}")


def _mask_finite(emg, angles):
    mask = np.isfinite(angles).all(axis=1) & np.isfinite(emg).all(axis=(1, 2))
    return emg[mask], angles[mask], int(mask.size - np.count_nonzero(mask))


def _apply_target_lag(emg, angles, lag):
    lag = int(lag)
    if lag == 0:
        return emg, angles

    n = min(len(emg), len(angles))
    if n <= abs(lag):
        raise ValueError(
            f"TRAIN_TARGET_LAG={lag} is too large for sequence length {n}."
        )

    if lag > 0:
        return emg[:-lag], angles[lag:]
    return emg[-lag:], angles[:lag]


def _rms_subframe_sequence(emg_3d, n_subframes):
    arr = np.asarray(emg_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected EMG shape (N,C,T), got {arr.shape}")

    n, c, t = arr.shape
    n_subframes = int(max(1, n_subframes))
    frame_len = int(max(1, t // n_subframes))
    out = np.zeros((n, c, n_subframes), dtype=np.float32)

    for i in range(n_subframes):
        start = i * frame_len
        end = t if i == (n_subframes - 1) else min(t, start + frame_len)
        chunk = arr[:, :, start:end]
        if chunk.shape[2] == 0:
            chunk = arr[:, :, -1:]
        out[:, :, i] = np.sqrt(np.mean(chunk**2, axis=2) + 1e-8)

    return out


def _fit_input_scaler(train_x, mode):
    params = {"mode": mode}
    if mode == "none":
        return params

    if mode == "standard":
        flat = train_x.reshape(train_x.shape[0], -1)
        mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
        std = flat.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-8] = 1.0
        params["mean"] = mean
        params["std"] = std
        return params

    raise ValueError(f"Unknown INPUT_SCALER: {mode}")


def _apply_input_scaler(x, params):
    mode = params["mode"]
    if mode == "none":
        return x

    flat = x.reshape(x.shape[0], -1)
    flat = (flat - params["mean"]) / params["std"]
    return flat.reshape(x.shape).astype(np.float32)


def _fit_target_scaler(train_y, mode):
    params = {"mode": mode}
    if mode == "none":
        return params

    if mode == "standard":
        mean = train_y.mean(axis=0, keepdims=True).astype(np.float32)
        std = train_y.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-8] = 1.0
        params["mean"] = mean
        params["std"] = std
        return params

    if mode == "minmax":
        minv = train_y.min(axis=0, keepdims=True).astype(np.float32)
        maxv = train_y.max(axis=0, keepdims=True).astype(np.float32)
        rng = (maxv - minv).astype(np.float32)
        rng[rng < 1e-8] = 1.0
        params["min"] = minv
        params["max"] = maxv
        params["range"] = rng
        return params

    raise ValueError(f"Unknown TARGET_SCALER: {mode}")


def _apply_target_scaler(y, params):
    mode = params["mode"]
    if mode == "none":
        return y
    if mode == "standard":
        return ((y - params["mean"]) / params["std"]).astype(np.float32)
    if mode == "minmax":
        return (2.0 * (y - params["min"]) / params["range"] - 1.0).astype(np.float32)
    raise ValueError(f"Unknown target scale mode: {mode}")


def _inverse_target_scaler(y, params):
    mode = params["mode"]
    if mode == "none":
        return y
    if mode == "standard":
        return (y * params["std"] + params["mean"]).astype(np.float32)
    if mode == "minmax":
        return (((y + 1.0) * 0.5) * params["range"] + params["min"]).astype(np.float32)
    raise ValueError(f"Unknown target scale mode: {mode}")


def _align_by_lag(preds, targets, lag):
    lag = int(lag)
    if lag == 0:
        return preds, targets
    if lag > 0:
        if preds.shape[0] <= lag:
            return None, None
        return preds[:-lag], targets[lag:]
    if preds.shape[0] <= -lag:
        return None, None
    return preds[-lag:], targets[:lag]


def sweep_lag_metrics(preds, targets, max_abs_lag):
    max_abs_lag = int(max(0, max_abs_lag))
    records = []

    for lag in range(-max_abs_lag, max_abs_lag + 1):
        aligned_preds, aligned_targets = _align_by_lag(preds, targets, lag)
        if aligned_preds is None or aligned_preds.shape[0] < 4:
            continue
        cc, rmse, r2 = compute_metrics(aligned_preds, aligned_targets)
        records.append(
            {
                "lag": int(lag),
                "n": int(aligned_preds.shape[0]),
                "cc": float(cc),
                "rmse": float(rmse),
                "r2": float(r2),
            }
        )

    if len(records) == 0:
        return None, []

    best = max(records, key=lambda row: (row["r2"], -row["rmse"]))
    return best, records


def _warmup_lr(optimizer, epoch_idx, base_lr, warmup_epochs=5):
    if warmup_epochs <= 0:
        return
    if epoch_idx < warmup_epochs:
        scale = float(epoch_idx + 1) / float(warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = float(base_lr) * scale


# ── PLOTTING ────────────────────────────────────────────────
def plot_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["val_r2"])
    axes[0, 1].set_title("Validation R²")
    axes[0, 1].set_xlabel("Epoch")

    axes[1, 0].plot(epochs, history["val_cc"])
    axes[1, 0].set_title("Validation CC")
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(epochs, history["val_rmse"])
    axes[1, 1].set_title("Validation RMSE")
    axes[1, 1].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# ── TRAINING LOOP ────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, inv_target_fn=None):
    model.train()
    total_loss = 0.0
    n_seen = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc="  Batch", leave=False)
    for emg, label in pbar:
        emg, label = emg.to(device), label.to(device)

        optimizer.zero_grad()
        pred = model(emg)
        loss = criterion(pred, label)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * emg.size(0)
        n_seen += emg.size(0)
        pbar.set_postfix(loss=total_loss / n_seen)

        all_preds.append(pred.detach().cpu().numpy())
        all_targets.append(label.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metric_preds = all_preds
    metric_targets = all_targets
    if inv_target_fn is not None:
        metric_preds = inv_target_fn(metric_preds)
        metric_targets = inv_target_fn(metric_targets)
    _, _, train_r2 = compute_metrics(metric_preds, metric_targets)

    return total_loss / len(loader.dataset), train_r2


@torch.no_grad()
def evaluate(model, loader, criterion, device, inv_target_fn=None, return_arrays=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for emg, label in loader:
        emg, label = emg.to(device), label.to(device)
        pred = model(emg)
        loss = criterion(pred, label)

        total_loss += loss.item() * emg.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metric_preds = all_preds
    metric_targets = all_targets
    if inv_target_fn is not None:
        metric_preds = inv_target_fn(metric_preds)
        metric_targets = inv_target_fn(metric_targets)

    cc, rmse, r2 = compute_metrics(metric_preds, metric_targets)

    if return_arrays:
        return (
            total_loss / len(loader.dataset),
            cc,
            rmse,
            r2,
            metric_preds,
            metric_targets,
        )
    return total_loss / len(loader.dataset), cc, rmse, r2


# ── MAIN ─────────────────────────────────────────────────────
def main():
    startup_t0 = time.perf_counter()
    args = parse_args()
    model_key = args.model
    model_spec = MODEL_REGISTRY[model_key]
    train_cfg = resolve_training_hparams(model_key, args.batch_size)
    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["lr"]
    eta_min = train_cfg["eta_min"]
    warmup_epochs = train_cfg["warmup_epochs"]
    weight_decay = train_cfg["weight_decay"]
    dropout = train_cfg["dropout"]
    baseline_cnn_activation = model_spec.get("cnn_activation", "elu")
    baseline_attn_ff_activation = model_spec.get("attn_ff_activation", "elu")
    baseline_mlp_activation = model_spec.get("mlp_activation", "elu")

    if model_key == "baseline":
        if args.cnn_activation is not None:
            baseline_cnn_activation = args.cnn_activation
        if args.attn_ff_activation is not None:
            baseline_attn_ff_activation = args.attn_ff_activation
        if args.mlp_activation is not None:
            baseline_mlp_activation = args.mlp_activation

    emg_transform = EMG_TRANSFORM
    input_mode = INPUT_MODE
    rms_subframes = RMS_SUBFRAMES
    input_scaler = INPUT_SCALER
    target_scaler = TARGET_SCALER

    if PAPER_REPLICATION:
        emg_transform = "none"
        input_mode = "rms_subframes"
        rms_subframes = 25
        input_scaler = "standard"
        target_scaler = "minmax"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = resolve_output_dir(args.output_dir, model_key)
    data_dir = resolve_data_dir(args.data_dir, args.data_source)
    out_dir.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Model: {model_key} ({model_spec['architecture']})")
    print(f"Outputs: {out_dir}")
    print(f"Data: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"LR: {lr:.2e} | Epochs: {epochs} | Warmup: {warmup_epochs}")
    if model_key == "baseline":
        print(
            "Activations: "
            f"cnn={baseline_cnn_activation}, "
            f"attn_ff={baseline_attn_ff_activation}, "
            f"mlp={baseline_mlp_activation}"
        )
    print("=" * 60)

    train_ds = NpzDataset("train", root=data_dir)
    val_ds = NpzDataset("val", root=data_dir)
    print(f"Startup | dataset load: {time.perf_counter() - startup_t0:.1f}s")

    train_ds.X = _normalize_emg_shape(train_ds.X)
    val_ds.X = _normalize_emg_shape(val_ds.X)

    train_ds.y = np.asarray(train_ds.y, dtype=np.float32)
    val_ds.y = np.asarray(val_ds.y, dtype=np.float32)

    train_ds.X, train_ds.y, dropped_train = _mask_finite(train_ds.X, train_ds.y)
    val_ds.X, val_ds.y, dropped_val = _mask_finite(val_ds.X, val_ds.y)
    if dropped_train > 0 or dropped_val > 0:
        print(f"Dropped non-finite samples | train={dropped_train}, val={dropped_val}")

    if TRAIN_TARGET_LAG != 0:
        train_ds.X, train_ds.y = _apply_target_lag(
            train_ds.X, train_ds.y, TRAIN_TARGET_LAG
        )
        val_ds.X, val_ds.y = _apply_target_lag(val_ds.X, val_ds.y, TRAIN_TARGET_LAG)
        print(f"Applied target lag shift: TRAIN_TARGET_LAG={TRAIN_TARGET_LAG} samples")

    if train_ds.X.ndim != 3 or train_ds.X.shape[1] != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"Expected EMG shape (N, {EXPECTED_CHANNELS}, T), got {train_ds.X.shape}. "
            "Re-run sleeve_preprocessing.py to regenerate temporal windows."
        )

    if train_ds.X.shape[2] < 2:
        raise RuntimeError(
            f"Temporal length T must be >= 2 for this model, got T={train_ds.X.shape[2]}."
        )

    if (
        val_ds.X.ndim != 3
        or val_ds.X.shape[1] != EXPECTED_CHANNELS
        or val_ds.X.shape[2] != train_ds.X.shape[2]
    ):
        raise RuntimeError(
            "Train/val EMG shapes are inconsistent. Re-run sleeve_preprocessing.py to regenerate both splits."
        )

    train_ds.X = _apply_emg_transform(train_ds.X, emg_transform)
    val_ds.X = _apply_emg_transform(val_ds.X, emg_transform)

    if input_mode == "rms_subframes":
        train_ds.X = _rms_subframe_sequence(train_ds.X, rms_subframes)
        val_ds.X = _rms_subframe_sequence(val_ds.X, rms_subframes)
        print(f"Converted EMG to RMS subframes: T={train_ds.X.shape[2]}")
    elif input_mode != "raw":
        raise ValueError(f"Unknown INPUT_MODE: {input_mode}")

    x_scale_params = _fit_input_scaler(train_ds.X, input_scaler)
    train_ds.X = _apply_input_scaler(train_ds.X, x_scale_params)
    val_ds.X = _apply_input_scaler(val_ds.X, x_scale_params)

    y_scale_params = _fit_target_scaler(train_ds.y, target_scaler)
    train_ds.y = _apply_target_scaler(train_ds.y, y_scale_params)
    val_ds.y = _apply_target_scaler(val_ds.y, y_scale_params)
    print(f"Startup | preprocessing: {time.perf_counter() - startup_t0:.1f}s")

    def _inv_target(arr):
        return _inverse_target_scaler(arr, y_scale_params)

    window_size = int(train_ds.X.shape[2])
    n_channels = int(train_ds.X.shape[1])
    n_joints = int(train_ds.y.shape[1])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=True,
    )

    model = model_spec["cls"](
        window_size=window_size,
        n_ch=n_channels,
        n_joints=n_joints,
        dropout=dropout,
        **(
            {
                "cnn_activation": baseline_cnn_activation,
                "attn_ff_activation": baseline_attn_ff_activation,
                "mlp_activation": baseline_mlp_activation,
            }
            if model_key == "baseline"
            else {}
        ),
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=eta_min,
    )
    criterion = nn.SmoothL1Loss(beta=0.5)

    print(f"Parameters: {model.count_params():,}\n")
    print(f"Startup | ready to train: {time.perf_counter() - startup_t0:.1f}s")

    config = {
        "model": {
            "model_key": model_key,
            "architecture": model_spec["architecture"],
            "n_ch": n_channels,
            "window_size": window_size,
            "n_joints": n_joints,
            "hidden": int(getattr(model, "hidden", -1)),
            "n_attn": int(getattr(model, "n_attn", -1)),
            "n_heads": int(getattr(model, "n_heads", -1)),
            "kernel_size": int(getattr(model, "kernel_size", -1)),
            "dilations": list(getattr(model, "dilations", ())),
            "geom_ch": int(getattr(model, "geom_ch", -1)),
            "cnn_activation": getattr(model, "cnn_activation", ""),
            "attn_ff_activation": getattr(model, "attn_ff_activation", ""),
            "mlp_activation": getattr(model, "mlp_activation", ""),
            "parameters": model.count_params(),
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "eta_min": eta_min,
            "warmup_epochs": warmup_epochs,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "SmoothL1Loss(beta=0.5)",
            "grad_clip": 1.0,
            "seed": SEED,
        },
        "data": {
            "data_source": args.data_source,
            "data_dir": str(data_dir),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "emg_shape": list(train_ds.X.shape),
            "paper_replication": PAPER_REPLICATION,
            "emg_transform": emg_transform,
            "input_mode": input_mode,
            "rms_subframes": rms_subframes,
            "input_scaler": input_scaler,
            "target_scaler": target_scaler,
            "train_target_lag": TRAIN_TARGET_LAG,
            "enable_lag_sweep": ENABLE_LAG_SWEEP,
            "lag_sweep_max": LAG_SWEEP_MAX,
            "checkpoint_selection": CHECKPOINT_SELECTION,
        },
        "device": DEVICE,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_r2 = float("-inf")
    history = {
        "train_loss": [],
        "train_r2": [],
        "val_loss": [],
        "val_cc": [],
        "val_rmse": [],
        "val_r2": [],
        "val_lag_best": [],
        "val_lag_cc": [],
        "val_lag_rmse": [],
        "val_lag_r2": [],
        "best_r2": None,
    }

    lag_sweep_last = []
    best_score = float("-inf")
    patience_counter = 0

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training")
    for epoch in epoch_bar:
        _warmup_lr(optimizer, epoch - 1, lr, warmup_epochs)

        train_loss, train_r2 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            inv_target_fn=_inv_target,
        )
        val_loss, cc, rmse, r2, val_preds, val_targets = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            inv_target_fn=_inv_target,
            return_arrays=True,
        )

        lag_best = {"lag": 0, "cc": float(cc), "rmse": float(rmse), "r2": float(r2)}
        if ENABLE_LAG_SWEEP and LAG_SWEEP_MAX > 0:
            lag_best_candidate, lag_sweep_last = sweep_lag_metrics(
                val_preds, val_targets, LAG_SWEEP_MAX
            )
            if lag_best_candidate is not None:
                lag_best = lag_best_candidate

        if epoch > warmup_epochs:
            scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_bar.set_postfix(
            train_r2=f"{train_r2:.4f}", val_r2=f"{r2:.4f}", lr=f"{current_lr:.2e}"
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_loss:.4f} | "
            f"Train R² {train_r2:.4f} | "
            f"Val {val_loss:.4f} | "
            f"CC {cc:.4f} | "
            f"R² {r2:.4f} | "
            f"Lag* {lag_best['lag']:+d} (R² {lag_best['r2']:.4f}, RMSE {lag_best['rmse']:.4f})"
        )

        history["train_loss"].append(float(train_loss))
        history["train_r2"].append(float(train_r2))
        history["val_loss"].append(float(val_loss))
        history["val_cc"].append(float(cc))
        history["val_rmse"].append(float(rmse))
        history["val_r2"].append(float(r2))
        history["val_lag_best"].append(int(lag_best["lag"]))
        history["val_lag_cc"].append(float(lag_best["cc"]))
        history["val_lag_rmse"].append(float(lag_best["rmse"]))
        history["val_lag_r2"].append(float(lag_best["r2"]))

        score = float(r2)
        if CHECKPOINT_SELECTION == "lag_r2":
            score = float(lag_best["r2"])
        if score > best_score:
            best_score = score
            best_r2 = r2
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            scaler_artifacts = {}
            if x_scale_params["mode"] == "standard":
                scaler_artifacts["x_mean"] = x_scale_params["mean"].squeeze()
                scaler_artifacts["x_std"] = x_scale_params["std"].squeeze()
            if y_scale_params["mode"] == "standard":
                scaler_artifacts["y_mean"] = y_scale_params["mean"].squeeze()
                scaler_artifacts["y_std"] = y_scale_params["std"].squeeze()
            if y_scale_params["mode"] == "minmax":
                scaler_artifacts["y_min"] = y_scale_params["min"].squeeze()
                scaler_artifacts["y_max"] = y_scale_params["max"].squeeze()
            if len(scaler_artifacts) > 0:
                np.savez(out_dir / "scaler_params.npz", **scaler_artifacts)

        history["best_r2"] = float(best_r2)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if epoch % 10 == 0 or epoch == epochs:
            plot_curves(history, out_dir)

    if ENABLE_LAG_SWEEP and len(lag_sweep_last) > 0:
        with open(out_dir / "lag_sweep_last_epoch.json", "w") as f:
            json.dump(lag_sweep_last, f, indent=2)

    print("\nBest R²:", best_r2)


if __name__ == "__main__":
    main()
