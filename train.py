"""
train.py
Training loop for temporal-window CNNAttentionImproved.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from model import CNNAttentionImproved
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────
PROCESSED_DIR = "processed data"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 256
EPOCHS = 150
LR = 3e-4
ETA_MIN = 1e-5
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 5e-4
N_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PAPER_REPLICATION = False
EMG_TRANSFORM = "log1p"  # "none" | "log1p"
INPUT_MODE = "raw"  # "raw" | "rms_subframes"
RMS_SUBFRAMES = 25
INPUT_SCALER = "standard"  # "none" | "standard"
TARGET_SCALER = "standard"  # "none" | "standard" | "minmax"


# ── DATASET (Multi-File Loader) ──────────────────────────────
class NpzDataset(Dataset):
    def __init__(self, split="train"):
        root = Path(PROCESSED_DIR)
        files = sorted(root.glob(f"*_{split}.npz"))
        assert len(files) > 0, f"No *_{split}.npz files found in {root}"

        emg_parts, angle_parts = [], []
        for f in files:
            data = np.load(f)
            emg_parts.append(data["emg"].astype(np.float32))
            angle_parts.append(data["angles"].astype(np.float32))

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
        return np.log1p(np.maximum(arr, 0.0)).astype(np.float32)
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
def evaluate(model, loader, criterion, device, inv_target_fn=None):
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

    return total_loss / len(loader.dataset), cc, rmse, r2


# ── MAIN ─────────────────────────────────────────────────────
def main():
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

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print("=" * 60)

    train_ds = NpzDataset("train")
    test_ds = NpzDataset("test")

    train_ds.X = _normalize_emg_shape(train_ds.X)
    test_ds.X = _normalize_emg_shape(test_ds.X)

    train_ds.y = np.asarray(train_ds.y, dtype=np.float32)
    test_ds.y = np.asarray(test_ds.y, dtype=np.float32)

    train_ds.X, train_ds.y, dropped_train = _mask_finite(train_ds.X, train_ds.y)
    test_ds.X, test_ds.y, dropped_test = _mask_finite(test_ds.X, test_ds.y)
    if dropped_train > 0 or dropped_test > 0:
        print(
            f"Dropped non-finite samples | train={dropped_train}, test={dropped_test}"
        )

    if train_ds.X.ndim != 3 or train_ds.X.shape[1] != 12:
        raise RuntimeError(
            f"Expected EMG shape (N, 12, T), got {train_ds.X.shape}. "
            "Re-run data_processing.py to regenerate temporal windows."
        )

    if train_ds.X.shape[2] < 2:
        raise RuntimeError(
            f"Temporal length T must be >= 2 for this model, got T={train_ds.X.shape[2]}. "
            "Re-run data_processing.py with temporal windows (e.g., WINDOW_MS=25)."
        )

    if (
        test_ds.X.ndim != 3
        or test_ds.X.shape[1] != 12
        or test_ds.X.shape[2] != train_ds.X.shape[2]
    ):
        raise RuntimeError(
            "Train/test EMG shapes are inconsistent. Re-run data_processing.py to regenerate both splits."
        )

    train_ds.X = _apply_emg_transform(train_ds.X, emg_transform)
    test_ds.X = _apply_emg_transform(test_ds.X, emg_transform)

    if input_mode == "rms_subframes":
        train_ds.X = _rms_subframe_sequence(train_ds.X, rms_subframes)
        test_ds.X = _rms_subframe_sequence(test_ds.X, rms_subframes)
        print(f"Converted EMG to RMS subframes: T={train_ds.X.shape[2]}")
    elif input_mode != "raw":
        raise ValueError(f"Unknown INPUT_MODE: {input_mode}")

    x_scale_params = _fit_input_scaler(train_ds.X, input_scaler)
    train_ds.X = _apply_input_scaler(train_ds.X, x_scale_params)
    test_ds.X = _apply_input_scaler(test_ds.X, x_scale_params)

    y_scale_params = _fit_target_scaler(train_ds.y, target_scaler)
    train_ds.y = _apply_target_scaler(train_ds.y, y_scale_params)
    test_ds.y = _apply_target_scaler(test_ds.y, y_scale_params)

    def _inv_target(arr):
        return _inverse_target_scaler(arr, y_scale_params)

    window_size = int(train_ds.X.shape[2])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=True,
    )

    n_joints = int(train_ds.y.shape[1])
    model = CNNAttentionImproved(window_size=window_size, n_joints=n_joints).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=2,
        eta_min=ETA_MIN,
    )
    criterion = nn.MSELoss()

    print(f"Parameters: {model.count_params():,}\n")

    # ── Save config ──────────────────────────────────────────
    config = {
        "model": {
            "architecture": "CNNAttentionImproved",
            "n_ch": 12,
            "window_size": window_size,
            "n_joints": n_joints,
            "hidden": int(getattr(model, "hidden", -1)),
            "parameters": model.count_params(),
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "eta_min": ETA_MIN,
            "warmup_epochs": WARMUP_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts",
            "scheduler_t0": 15,
            "scheduler_tmult": 2,
            "loss": "MSELoss()",
            "grad_clip": 1.0,
            "seed": SEED,
        },
        "data": {
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "emg_shape": list(train_ds.X.shape),
            "paper_replication": PAPER_REPLICATION,
            "emg_transform": emg_transform,
            "input_mode": input_mode,
            "rms_subframes": rms_subframes,
            "input_scaler": input_scaler,
            "target_scaler": target_scaler,
        },
        "device": DEVICE,
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
        "best_r2": None,
    }

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training")
    for epoch in epoch_bar:
        _warmup_lr(optimizer, epoch - 1, LR, WARMUP_EPOCHS)

        train_loss, train_r2 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            inv_target_fn=_inv_target,
        )
        val_loss, cc, rmse, r2 = evaluate(
            model, test_loader, criterion, DEVICE, inv_target_fn=_inv_target
        )

        if epoch > WARMUP_EPOCHS:
            scheduler.step(epoch - WARMUP_EPOCHS)

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
            f"R² {r2:.4f}"
        )

        history["train_loss"].append(float(train_loss))
        history["train_r2"].append(float(train_r2))
        history["val_loss"].append(float(val_loss))
        history["val_cc"].append(float(cc))
        history["val_rmse"].append(float(rmse))
        history["val_r2"].append(float(r2))

        if r2 > best_r2:
            best_r2 = r2
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

        if epoch % 10 == 0 or epoch == EPOCHS:
            plot_curves(history, out_dir)

    print("\nBest R²:", best_r2)


if __name__ == "__main__":
    main()
