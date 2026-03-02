"""
data_processing.py
Loads Ninapro DB2 .mat files, preprocesses, augments,
and saves all windows to ./augmented_data/train and ./augmented_data/test
"""

import os, json
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch, resample
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────
DATA_DIR      = "data/ninapro_db2"
OUTPUT_DIR    = "augmented_data"
SUBJECTS      = [1, 2, 5, 9, 11, 14, 19, 21, 22, 23, 26, 27, 30, 31, 36]
MOVEMENTS     = [1, 2, 3, 4, 5, 6]
FS            = 2000
WINDOW_SIZE   = 50        # 25ms @ 2kHz
STEP          = 10        # stride in samples
TRAIN_REPS    = [1, 2, 3, 4]
TEST_REPS     = [5, 6]
N_AUG_COPIES  = 3
# PIP + MCP for each finger from 22-ch CyberGlove
JOINT_INDICES = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]

# ── FILTERS ─────────────────────────────────────────────────
def bandpass(emg, low=20, high=500):
    b, a = butter(4, [low/(FS/2), high/(FS/2)], btype='band')
    return filtfilt(b, a, emg, axis=0)

def notch(emg, freq=50, q=30):
    b, a = iirnotch(freq/(FS/2), q)
    return filtfilt(b, a, emg, axis=0)

def lowpass(angles, cutoff=5):
    b, a = butter(4, cutoff/(FS/2), btype='low')
    return filtfilt(b, a, angles, axis=0)

# ── NORMALIZATION ────────────────────────────────────────────
def norm_emg(emg):
    mean = emg.mean(axis=0, keepdims=True)
    std  = emg.std(axis=0, keepdims=True) + 1e-8
    return (emg - mean) / std, mean.squeeze(), std.squeeze()

def norm_angles(angles):
    mn  = angles.min(axis=0, keepdims=True)
    mx  = angles.max(axis=0, keepdims=True)
    return 2*(angles - mn)/(mx - mn + 1e-8) - 1, mn.squeeze(), mx.squeeze()

# ── AUGMENTATION ─────────────────────────────────────────────
def augment(emg):
    """emg: (12, T) — channels first"""
    emg = emg.copy()
    T   = emg.shape[1]

    if np.random.rand() < 0.5:   # magnitude scale
        emg *= np.random.uniform(0.7, 1.3, size=(12, 1))

    if np.random.rand() < 0.5:   # gaussian noise
        emg += np.random.normal(0, 0.02, size=emg.shape)

    if np.random.rand() < 0.3:   # channel dropout (1-2 channels)
        drop = np.random.choice(12, np.random.randint(1,3), replace=False)
        emg[drop, :] = 0.0

    if np.random.rand() < 0.3:   # time warp
        warp = np.random.uniform(0.8, 1.2)
        emg  = resample(resample(emg, max(int(T*warp),2), axis=1), T, axis=1)

    if np.random.rand() < 0.3:   # circular electrode shift
        emg = np.roll(emg, np.random.randint(0, 12), axis=0)

    if np.random.rand() < 0.3:   # DC offset
        emg += np.random.uniform(-0.05, 0.05, size=(12, 1))

    return np.clip(emg, -5.0, 5.0).astype(np.float32)

# ── WINDOWING ────────────────────────────────────────────────
def extract_windows(emg, angles):
    """emg: (N,12), angles: (N,10) → list of ((12,W), (10,))"""
    wins = []
    for i in range(0, len(emg) - WINDOW_SIZE + 1, STEP):
        wins.append((
            emg[i:i+WINDOW_SIZE].T.astype(np.float32),
            angles[i+WINDOW_SIZE-1].astype(np.float32)
        ))
    return wins

# ── PER-SUBJECT ──────────────────────────────────────────────
def process_subject(sid, stats):
    path = os.path.join(DATA_DIR, f"S{sid}_E2_A1.mat")
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found"); return [], []

    mat   = sio.loadmat(path)
    emg   = bandpass(notch(mat['emg'].astype(np.float64)))
    ang   = lowpass(mat['glove'].astype(np.float64)[:, JOINT_INDICES])
    stim  = mat['restimulus'].squeeze().astype(int)
    reps  = mat['rerepetition'].squeeze().astype(int)

    train_w, test_w = [], []
    for mov in MOVEMENTS:
        mask = (stim == mov)
        if not mask.any(): continue

        en, em, es  = norm_emg(emg[mask])
        an, amn, amx = norm_angles(ang[mask])
        rm           = reps[mask]
        stats[f"S{sid}_M{mov}"] = dict(
            emg_mean=em.tolist(), emg_std=es.tolist(),
            ang_min=amn.tolist(), ang_max=amx.tolist()
        )

        for r in TRAIN_REPS:
            m = (rm == r)
            if m.sum() >= WINDOW_SIZE:
                train_w.extend(extract_windows(en[m], an[m]))

        for r in TEST_REPS:
            m = (rm == r)
            if m.sum() >= WINDOW_SIZE:
                test_w.extend(extract_windows(en[m], an[m]))

    return train_w, test_w

# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Ninapro DB2 — Preprocessing & Augmentation")
    print("="*60)

    train_dir = Path(OUTPUT_DIR) / "train"
    test_dir  = Path(OUTPUT_DIR) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    all_train, all_test, stats = [], [], {}

    for sid in SUBJECTS:
        print(f"\nSubject {sid:02d}...")
        tw, vw = process_subject(sid, stats)
        print(f"  train: {len(tw):,}  test: {len(vw):,}")
        all_train.extend(tw); all_test.extend(vw)

    np.random.seed(42)
    np.random.shuffle(all_train)

    print(f"\nSaving train (x{1+N_AUG_COPIES} with aug)...")
    c = 0
    for emg, label in tqdm(all_train):
        np.savez_compressed(train_dir/f"{c:08d}.npz", emg=emg, label=label); c+=1
        for _ in range(N_AUG_COPIES):
            np.savez_compressed(train_dir/f"{c:08d}.npz", emg=augment(emg), label=label); c+=1

    print("Saving test...")
    for i, (emg, label) in enumerate(tqdm(all_test)):
        np.savez_compressed(test_dir/f"{i:08d}.npz", emg=emg, label=label)

    with open(Path(OUTPUT_DIR)/"stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Train: {c:,} | Test: {len(all_test):,}")
    print(f"Output: {Path(OUTPUT_DIR).resolve()}")

if __name__ == "__main__":
    main()