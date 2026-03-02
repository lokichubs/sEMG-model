"""
train.py
Training and evaluation loops for CNNAttentionImproved.
Outputs loss curves and per-metric plots to ./outputs/
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import CNNAttentionImproved, KinematicLoss


# ── CONFIG ───────────────────────────────────────────────────
AUGMENTED_DIR = "augmented_data"
OUTPUT_DIR    = "outputs"
BATCH_SIZE    = 256
EPOCHS        = 200
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
LAMBDA_SMOOTH = 0.01
N_WORKERS     = 4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42


# ── DATASET ──────────────────────────────────────────────────
class NpzDataset(Dataset):
    def __init__(self, split="train"):
        self.files = sorted(Path(AUGMENTED_DIR, split).glob("*.npz"))
        assert len(self.files) > 0, f"No .npz files in {AUGMENTED_DIR}/{split}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return torch.from_numpy(d["emg"]), torch.from_numpy(d["label"])


# ── METRICS ──────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    cc_list, rmse_list, r2_list = [], [], []
    for j in range(preds.shape[1]):
        p, t   = preds[:, j], targets[:, j]
        cc, _  = pearsonr(p, t)
        rmse   = np.sqrt(np.mean((p - t) ** 2))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2) + 1e-8
        r2     = 1 - ss_res / ss_tot
        cc_list.append(cc)
        rmse_list.append(rmse)
        r2_list.append(r2)
    return np.mean(cc_list), np.mean(rmse_list), np.mean(r2_list)


# ── PLOTTING ─────────────────────────────────────────────────
def save_plots(history: dict, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CNNAttentionImproved — Training History", fontsize=14)

    axes[0,0].plot(epochs, history["train_loss"], label="Train")
    axes[0,0].plot(epochs, history["val_loss"],   label="Val")
    axes[0,0].set_title("Loss (MSE)"); axes[0,0].set_xlabel("Epoch")
    axes[0,0].legend(); axes[0,0].grid(True)

    axes[0,1].plot(epochs, history["val_cc"])
    axes[0,1].axhline(0.90, color='red', linestyle='--', label='Target (0.90)')
    axes[0,1].set_title("Pearson CC (val)"); axes[0,1].set_xlabel("Epoch")
    axes[0,1].legend(); axes[0,1].grid(True)

    axes[1,0].plot(epochs, history["val_rmse"])
    axes[1,0].set_title("RMSE in degrees (val)"); axes[1,0].set_xlabel("Epoch")
    axes[1,0].grid(True)

    axes[1,1].plot(epochs, history["val_r2"])
    axes[1,1].set_title("R² (val)"); axes[1,1].set_xlabel("Epoch")
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


# ── TRAINING LOOP ────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch_bar):
    model.train()
    total_loss = 0.0

    for emg, label in loader:
        emg, label = emg.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(emg)
        loss = criterion(pred, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * emg.size(0)

        # Update progress bar with current batch loss
        epoch_bar.set_postfix({
            "batch_loss": f"{loss.item():.4f}"
        }, refresh=True)

    return total_loss / len(loader.dataset)


# ── EVAL LOOP ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
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

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    cc, rmse, r2 = compute_metrics(all_preds, all_targets)
    return total_loss / len(loader.dataset), cc, rmse, r2


# ── MAIN ─────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print("="*60)

    # ── Data ──
    train_loader = DataLoader(
        NpzDataset("train"), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=N_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        NpzDataset("test"), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=N_WORKERS, pin_memory=True
    )

    # ── Model ──
    model     = CNNAttentionImproved().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = KinematicLoss(lambda_smooth=LAMBDA_SMOOTH)

    print(f"Parameters: {model.count_params():,}\n")

    history = {k: [] for k in
               ["train_loss", "val_loss", "val_cc", "val_rmse", "val_r2"]}
    best_cc = 0.0

    # ── Outer epoch progress bar ──
    epoch_bar = tqdm(
        range(1, EPOCHS + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
        colour="green"
    )

    for epoch in epoch_bar:

        # Inner bar tracks batches within this epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch_bar
        )
        val_loss, cc, rmse, r2 = evaluate(
            model, test_loader, criterion, DEVICE
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cc"].append(cc)
        history["val_rmse"].append(rmse)
        history["val_r2"].append(r2)

        # ── Update epoch bar with full metrics ──
        epoch_bar.set_description(f"Epoch {epoch:03d}/{EPOCHS}")
        epoch_bar.set_postfix({
            "train_loss" : f"{train_loss:.4f}",
            "val_loss"   : f"{val_loss:.4f}",
            "CC"         : f"{cc:.4f}",
            "RMSE"       : f"{rmse:.4f}",
            "R²"         : f"{r2:.4f}",
            "best_CC"    : f"{best_cc:.4f}"
        }, refresh=True)

        # ── Save best checkpoint ──
        if cc > best_cc:
            best_cc = cc
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cc": cc, "rmse": rmse, "r2": r2
            }, out_dir / "best_model.pt")
            epoch_bar.write(f"  ✓ New best CC={cc:.4f} at epoch {epoch} — checkpoint saved")

        # ── Save plot every 10 epochs ──
        if epoch % 10 == 0 or epoch == 1:
            save_plots(history, out_dir)
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    # ── Final eval on best model ──
    print("\n" + "="*60)
    print("Loading best checkpoint for final evaluation...")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    _, cc, rmse, r2 = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\nFinal Test Results (epoch {ckpt['epoch']}):")
    print(f"  CC   : {cc:.4f}   (target > 0.90)")
    print(f"  RMSE : {rmse:.4f} degrees")
    print(f"  R²   : {r2:.4f}")

    # ── Final save ──
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    save_plots(history, out_dir)
    print("\nAll outputs saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()