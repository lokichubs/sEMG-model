"""
data_processing.py
Loads Ninapro DB2 .mat files, preprocesses,
and saves windows to ./processed_data/train and ./processed_data/test
"""

import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "processed_data"
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
MOVEMENTS = [1, 2, 3, 4, 5, 6]
FS = 2000
WINDOW_SIZE = 50  # 25ms @ 2kHz
STEP = 10  # stride in samples
TRAIN_REPS = [1, 2, 3, 4]
TEST_REPS = [5, 6]
N_AUG_COPIES = 0
# PIP + MCP for each finger from 22-ch CyberGlove
JOINT_INDICES = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]


# ── NORMALIZATION ────────────────────────────────────────────
def norm_emg(emg):
    mean = emg.mean(axis=0, keepdims=True)
    std = emg.std(axis=0, keepdims=True) + 1e-8
    return (emg - mean) / std, mean.squeeze(), std.squeeze()


def norm_angles(angles):
    mn = angles.min(axis=0, keepdims=True)
    mx = angles.max(axis=0, keepdims=True)
    return 2 * (angles - mn) / (mx - mn + 1e-8) - 1, mn.squeeze(), mx.squeeze()


# ── WINDOWING ────────────────────────────────────────────────
def extract_windows(emg, angles):
    """emg: (N,12), angles: (N,10) → list of ((12,W), (10,))"""
    wins = []
    for i in range(0, len(emg) - WINDOW_SIZE + 1, STEP):
        wins.append(
            (
                emg[i : i + WINDOW_SIZE].T.astype(np.float32),
                angles[i + WINDOW_SIZE - 1].astype(np.float32),
            )
        )
    return wins


# ── PER-SUBJECT ──────────────────────────────────────────────
def process_subject(sid, stats):
    path = os.path.join(DATA_DIR, f"DB2_s{sid}", f"DB2_s{sid}", f"S{sid}_E1_A1.mat")
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found")
        return [], []

    mat = sio.loadmat(path)
    emg = mat["emg"].astype(np.float64)
    ang = mat["glove"].astype(np.float64)[:, JOINT_INDICES]
    stim = mat["restimulus"].squeeze().astype(int)
    reps = mat["rerepetition"].squeeze().astype(int)

    train_w, test_w = [], []
    for mov in MOVEMENTS:
        mask = stim == mov
        if not mask.any():
            continue

        em_mov, em_mean, em_std = norm_emg(emg[mask])
        ang_mov, ang_min, ang_max = norm_angles(ang[mask])
        rm = reps[mask]
        stats[f"S{sid}_M{mov}"] = dict(
            emg_mean=em_mean.tolist(),
            emg_std=em_std.tolist(),
            ang_min=ang_min.tolist(),
            ang_max=ang_max.tolist(),
        )

        for r in TRAIN_REPS:
            m = rm == r
            if m.sum() >= WINDOW_SIZE:
                train_w.extend(extract_windows(em_mov[m], ang_mov[m]))

        for r in TEST_REPS:
            m = rm == r
            if m.sum() >= WINDOW_SIZE:
                test_w.extend(extract_windows(em_mov[m], ang_mov[m]))

    return train_w, test_w


# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Ninapro DB2 — Preprocessing (no augmentation)")
    print("=" * 60)

    train_dir = Path(OUTPUT_DIR) / "train"
    test_dir = Path(OUTPUT_DIR) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for old_file in train_dir.glob("*.npz"):
        old_file.unlink()
    for old_file in test_dir.glob("*.npz"):
        old_file.unlink()

    all_train, all_test, stats = [], [], {}

    for sid in SUBJECTS:
        print(f"\nSubject {sid:02d}...")
        tw, vw = process_subject(sid, stats)
        print(f"  train: {len(tw):,}  test: {len(vw):,}")
        all_train.extend(tw)
        all_test.extend(vw)

    np.random.seed(42)
    np.random.shuffle(all_train)

    print("\nSaving train (no augmentation)...")
    c = 0
    for emg, label in tqdm(all_train):
        np.savez_compressed(train_dir / f"{c:08d}.npz", emg=emg, label=label)
        c += 1

    print("Saving test...")
    for i, (emg, label) in enumerate(tqdm(all_test)):
        np.savez_compressed(test_dir / f"{i:08d}.npz", emg=emg, label=label)

    with open(Path(OUTPUT_DIR) / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Train: {c:,} | Test: {len(all_test):,}")
    print(f"Output: {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
