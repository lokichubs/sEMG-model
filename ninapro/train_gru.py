"""
train_gru.py
Training loop for GRU baseline on Ninapro temporal windows.
Uses the same preprocessing, metrics, and logging protocol as train.py for fair comparison.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import train as shared
from gru_model import GRUKinematicsRegressor

# Keep baseline settings aligned with train.py for fairness.
OUTPUT_DIR = "outputs_gru"
BATCH_SIZE = shared.BATCH_SIZE
EPOCHS = shared.EPOCHS
LR = shared.LR
ETA_MIN = shared.ETA_MIN
WARMUP_EPOCHS = shared.WARMUP_EPOCHS
WEIGHT_DECAY = shared.WEIGHT_DECAY
N_WORKERS = shared.N_WORKERS
DEVICE = shared.DEVICE
SEED = shared.SEED

EMG_TRANSFORM = shared.EMG_TRANSFORM
INPUT_MODE = shared.INPUT_MODE
RMS_SUBFRAMES = shared.RMS_SUBFRAMES
INPUT_SCALER = shared.INPUT_SCALER
TARGET_SCALER = shared.TARGET_SCALER
TRAIN_TARGET_LAG = shared.TRAIN_TARGET_LAG
ENABLE_LAG_SWEEP = shared.ENABLE_LAG_SWEEP
LAG_SWEEP_MAX = shared.LAG_SWEEP_MAX
CHECKPOINT_SELECTION = shared.CHECKPOINT_SELECTION

# GRU-specific hyperparameters.
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.15
GRU_BIDIRECTIONAL = False


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
    _, _, train_r2 = shared.compute_metrics(metric_preds, metric_targets)

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

    cc, rmse, r2 = shared.compute_metrics(metric_preds, metric_targets)

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


def main():
    startup_t0 = time.perf_counter()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print("=" * 60)

    train_ds = shared.NpzDataset("train")
    val_ds = shared.NpzDataset("val")
    print(f"Startup | dataset load: {time.perf_counter() - startup_t0:.1f}s")

    train_ds.X = shared._normalize_emg_shape(train_ds.X)
    val_ds.X = shared._normalize_emg_shape(val_ds.X)

    train_ds.y = np.asarray(train_ds.y, dtype=np.float32)
    val_ds.y = np.asarray(val_ds.y, dtype=np.float32)

    train_ds.X, train_ds.y, dropped_train = shared._mask_finite(train_ds.X, train_ds.y)
    val_ds.X, val_ds.y, dropped_val = shared._mask_finite(val_ds.X, val_ds.y)
    if dropped_train > 0 or dropped_val > 0:
        print(f"Dropped non-finite samples | train={dropped_train}, val={dropped_val}")

    if TRAIN_TARGET_LAG != 0:
        train_ds.X, train_ds.y = shared._apply_target_lag(train_ds.X, train_ds.y, TRAIN_TARGET_LAG)
        val_ds.X, val_ds.y = shared._apply_target_lag(val_ds.X, val_ds.y, TRAIN_TARGET_LAG)
        print(f"Applied target lag shift: TRAIN_TARGET_LAG={TRAIN_TARGET_LAG} samples")

    if train_ds.X.ndim != 3 or train_ds.X.shape[1] != 12:
        raise RuntimeError(
            f"Expected EMG shape (N, 12, T), got {train_ds.X.shape}. "
            "Re-run preprocessing.py to regenerate temporal windows."
        )

    if train_ds.X.shape[2] < 2:
        raise RuntimeError(
            f"Temporal length T must be >= 2 for this model, got T={train_ds.X.shape[2]}."
        )

    if (
        val_ds.X.ndim != 3
        or val_ds.X.shape[1] != 12
        or val_ds.X.shape[2] != train_ds.X.shape[2]
    ):
        raise RuntimeError(
            "Train/val EMG shapes are inconsistent. Re-run preprocessing.py to regenerate both splits."
        )

    train_ds.X = shared._apply_emg_transform(train_ds.X, EMG_TRANSFORM)
    val_ds.X = shared._apply_emg_transform(val_ds.X, EMG_TRANSFORM)

    if INPUT_MODE == "rms_subframes":
        train_ds.X = shared._rms_subframe_sequence(train_ds.X, RMS_SUBFRAMES)
        val_ds.X = shared._rms_subframe_sequence(val_ds.X, RMS_SUBFRAMES)
        print(f"Converted EMG to RMS subframes: T={train_ds.X.shape[2]}")
    elif INPUT_MODE != "raw":
        raise ValueError(f"Unknown INPUT_MODE: {INPUT_MODE}")

    x_scale_params = shared._fit_input_scaler(train_ds.X, INPUT_SCALER)
    train_ds.X = shared._apply_input_scaler(train_ds.X, x_scale_params)
    val_ds.X = shared._apply_input_scaler(val_ds.X, x_scale_params)

    y_scale_params = shared._fit_target_scaler(train_ds.y, TARGET_SCALER)
    train_ds.y = shared._apply_target_scaler(train_ds.y, y_scale_params)
    val_ds.y = shared._apply_target_scaler(val_ds.y, y_scale_params)
    print(f"Startup | preprocessing: {time.perf_counter() - startup_t0:.1f}s")

    def _inv_target(arr):
        return shared._inverse_target_scaler(arr, y_scale_params)

    window_size = int(train_ds.X.shape[2])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=True,
    )

    n_joints = int(train_ds.y.shape[1])
    model = GRUKinematicsRegressor(
        n_ch=12,
        n_joints=n_joints,
        hidden=GRU_HIDDEN,
        num_layers=GRU_LAYERS,
        dropout=GRU_DROPOUT,
        bidirectional=GRU_BIDIRECTIONAL,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,
        eta_min=ETA_MIN,
    )
    criterion = nn.SmoothL1Loss(beta=0.5)

    print(f"Parameters: {model.count_params():,}\n")
    print(f"Startup | ready to train: {time.perf_counter() - startup_t0:.1f}s")

    config = {
        "model": {
            "architecture": "GRUKinematicsRegressor",
            "n_ch": 12,
            "window_size": window_size,
            "n_joints": n_joints,
            "hidden": GRU_HIDDEN,
            "num_layers": GRU_LAYERS,
            "dropout": GRU_DROPOUT,
            "bidirectional": GRU_BIDIRECTIONAL,
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
            "scheduler": "CosineAnnealingLR",
            "loss": "SmoothL1Loss(beta=0.5)",
            "grad_clip": 1.0,
            "seed": SEED,
        },
        "data": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "emg_shape": list(train_ds.X.shape),
            "emg_transform": EMG_TRANSFORM,
            "input_mode": INPUT_MODE,
            "rms_subframes": RMS_SUBFRAMES,
            "input_scaler": INPUT_SCALER,
            "target_scaler": TARGET_SCALER,
            "train_target_lag": TRAIN_TARGET_LAG,
            "enable_lag_sweep": ENABLE_LAG_SWEEP,
            "lag_sweep_max": LAG_SWEEP_MAX,
            "checkpoint_selection": CHECKPOINT_SELECTION,
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
        "val_lag_best": [],
        "val_lag_cc": [],
        "val_lag_rmse": [],
        "val_lag_r2": [],
        "best_r2": None,
    }

    lag_sweep_last = []
    best_score = float("-inf")

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training")
    for epoch in epoch_bar:
        shared._warmup_lr(optimizer, epoch - 1, LR, WARMUP_EPOCHS)

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
            lag_best_candidate, lag_sweep_last = shared.sweep_lag_metrics(
                val_preds, val_targets, LAG_SWEEP_MAX
            )
            if lag_best_candidate is not None:
                lag_best = lag_best_candidate

        if epoch > WARMUP_EPOCHS:
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
            shared.plot_curves(history, out_dir)

    if ENABLE_LAG_SWEEP and len(lag_sweep_last) > 0:
        with open(out_dir / "lag_sweep_last_epoch.json", "w") as f:
            json.dump(lag_sweep_last, f, indent=2)

    print("\nBest R²:", best_r2)


if __name__ == "__main__":
    main()
