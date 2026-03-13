import argparse
import os
from pathlib import Path

import numpy as np
import pyxdf
from scipy.signal import butter, filtfilt

DEFAULT_EMG_STREAM = "OpenEphys_EMG"
DEFAULT_ANGLE_STREAM = "StereoHandTracker_Angles"
DEFAULT_FS = 2000.0


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def _stream_name(stream):
    info = stream.get("info", {})
    name = info.get("name", [None])
    if isinstance(name, list):
        return name[0]
    return name


def _stream_nominal_srate(stream, default_fs=DEFAULT_FS):
    info = stream.get("info", {})
    nominal_srate = info.get("nominal_srate", [default_fs])
    if isinstance(nominal_srate, list):
        nominal_srate = nominal_srate[0]
    try:
        nominal_srate = float(nominal_srate)
    except (TypeError, ValueError):
        nominal_srate = float(default_fs)
    if nominal_srate <= 0:
        nominal_srate = float(default_fs)
    return nominal_srate


def _get_stream_by_name(streams, stream_name):
    for stream in streams:
        if _stream_name(stream) == stream_name:
            return stream

    available = [_stream_name(stream) for stream in streams]
    raise KeyError(f"Stream '{stream_name}' not found. Available streams: {available}")


def _ensure_2d(array, name):
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D after conversion, got shape {array.shape}")
    return array


def _valid_angle_columns(angle_array):
    return np.where(np.isfinite(angle_array).any(axis=0))[0]


def interpolate_targets_to_emg(target_ts, target_x, emg_ts):
    target_ts = np.asarray(target_ts, dtype=np.float64)
    target_x = np.asarray(target_x, dtype=np.float64)
    emg_ts = np.asarray(emg_ts, dtype=np.float64)

    out = np.full((len(emg_ts), target_x.shape[1]), np.nan, dtype=np.float32)
    for col in range(target_x.shape[1]):
        y = target_x[:, col]
        valid = np.isfinite(target_ts) & np.isfinite(y)
        if np.count_nonzero(valid) < 2:
            continue
        out[:, col] = np.interp(emg_ts, target_ts[valid], y[valid]).astype(np.float32)
    return out


def segment_data(emg, angles, window_len, step_len, label_position="center"):
    n_samples = emg.shape[0]
    n_windows = (n_samples - window_len) // step_len

    if n_windows <= 0:
        return np.array([]), np.array([]), np.array([])

    n_channels = emg.shape[1]
    n_targets = angles.shape[1]

    EMG = np.zeros((n_windows, n_channels, window_len), dtype=np.float32)
    ANG = np.zeros((n_windows, n_targets), dtype=np.float32)
    IDX = np.zeros((n_windows,), dtype=np.int64)

    for i in range(n_windows):
        start = i * step_len
        end = start + window_len
        EMG[i] = emg[start:end, :].T

        if label_position == "last":
            label_idx = end - 1
        else:
            label_idx = start + (window_len // 2)

        ANG[i] = angles[label_idx, :]
        IDX[i] = label_idx

    return EMG, ANG, IDX


def process_sleeve_file(
    xdf_file_path,
    output_dir,
    emg_stream_name=DEFAULT_EMG_STREAM,
    angle_stream_name=DEFAULT_ANGLE_STREAM,
    window_len=400,
    step_len=50,
    emg_lowcut=20.0,
    emg_highcut=500.0,
    angle_lowpass=5.0,
    train_fraction=0.8,
    label_position="last",
    split_mode="blocked",
    val_block_period=5,
):
    print(f"Processing {xdf_file_path}...")

    try:
        streams, _ = pyxdf.load_xdf(str(xdf_file_path))
    except Exception as e:
        print(f"Error loading {xdf_file_path}: {e}")
        return

    try:
        emg_stream = _get_stream_by_name(streams, emg_stream_name)
        angle_stream = _get_stream_by_name(streams, angle_stream_name)
    except KeyError as e:
        print(f"  Skipping {xdf_file_path}: {e}")
        return

    emg_ts = np.asarray(emg_stream.get("time_stamps", []), dtype=np.float64)
    angle_ts = np.asarray(angle_stream.get("time_stamps", []), dtype=np.float64)
    emg_x = _ensure_2d(emg_stream.get("time_series", []), "EMG")
    angle_x = _ensure_2d(angle_stream.get("time_series", []), "Angles")

    if len(emg_ts) == 0 or len(angle_ts) == 0:
        print(f"  Skipping {xdf_file_path}: empty EMG or angle timestamps.")
        return

    shared_start = max(emg_ts[0], angle_ts[0])
    shared_end = min(emg_ts[-1], angle_ts[-1])
    if shared_end <= shared_start:
        print(f"  Skipping {xdf_file_path}: no shared overlap between EMG and angles.")
        return

    emg_mask = (emg_ts >= shared_start) & (emg_ts <= shared_end)
    angle_mask = (angle_ts >= shared_start) & (angle_ts <= shared_end)

    emg_ts = emg_ts[emg_mask]
    emg_x = np.asarray(emg_x[emg_mask], dtype=np.float32)
    angle_ts = angle_ts[angle_mask]
    angle_x = np.asarray(angle_x[angle_mask], dtype=np.float32)

    valid_angle_cols = _valid_angle_columns(angle_x)
    if len(valid_angle_cols) == 0:
        print(f"  Skipping {xdf_file_path}: no usable angle columns found.")
        return

    angle_x = angle_x[:, valid_angle_cols]

    fs_emg = _stream_nominal_srate(emg_stream, default_fs=DEFAULT_FS)

    print(f"  Overlap duration: {shared_end - shared_start:.3f} s")
    print(f"  EMG samples: {len(emg_ts)}")
    print(f"  Angle samples: {len(angle_ts)}")
    print(
        f"  Keeping {len(valid_angle_cols)} angle columns: {valid_angle_cols.tolist()}"
    )
    print(f"  Filtering EMG (Bandpass {emg_lowcut:.1f}-{emg_highcut:.1f}Hz)...")

    b_emg, a_emg = butter_bandpass(emg_lowcut, emg_highcut, fs_emg, order=4)
    filtered_emg = filtfilt(b_emg, a_emg, emg_x, axis=0).astype(np.float32)

    print("  Interpolating angles to EMG timeline...")
    angles_on_emg = interpolate_targets_to_emg(angle_ts, angle_x, emg_ts)

    finite_cols = np.where(np.isfinite(angles_on_emg).all(axis=0))[0]
    if len(finite_cols) == 0:
        print(
            f"  Skipping {xdf_file_path}: no fully finite interpolated angle columns."
        )
        return

    if len(finite_cols) != angles_on_emg.shape[1]:
        kept_source_cols = valid_angle_cols[finite_cols]
        print(
            "  Dropping interpolated columns with remaining NaNs; "
            f"keeping {len(kept_source_cols)} columns."
        )
        angles_on_emg = angles_on_emg[:, finite_cols]
        valid_angle_cols = kept_source_cols

    if angle_lowpass is not None and angle_lowpass > 0:
        print(f"  Filtering kinematics (Lowpass {angle_lowpass:.1f}Hz)...")
        b_ang, a_ang = butter_lowpass(angle_lowpass, fs_emg, order=2)
        filtered_angles = filtfilt(b_ang, a_ang, angles_on_emg, axis=0).astype(
            np.float32
        )
    else:
        filtered_angles = angles_on_emg.astype(np.float32)

    print(f"  Segmenting data (Window {window_len}, Step {step_len})...")
    w_emg, w_ang, w_idx = segment_data(
        filtered_emg,
        filtered_angles,
        window_len,
        step_len,
        label_position=label_position,
    )

    if w_emg.size == 0:
        print(f"  Skipping {xdf_file_path}: not enough samples for one window.")
        return

    window_time = (emg_ts[w_idx] - shared_start).astype(np.float32)

    n_windows = w_emg.shape[0]

    if split_mode == "chronological":
        split_idx = int(np.floor(n_windows * train_fraction))
        split_idx = (
            min(max(split_idx, 1), n_windows - 1) if n_windows > 1 else n_windows
        )
        train_mask = np.zeros(n_windows, dtype=bool)
        train_mask[:split_idx] = True
    elif split_mode == "blocked":
        # Divide session into equal-sized blocks. Every val_block_period-th
        # block goes to validation, the rest to training.  With the default
        # val_block_period=5, ~20% of data is val — matching train_fraction=0.8.
        n_blocks = max(
            val_block_period, int(round(val_block_period / (1.0 - train_fraction)))
        )
        block_size = max(1, n_windows // n_blocks)
        block_ids = np.minimum(np.arange(n_windows) // block_size, n_blocks - 1)
        train_mask = (block_ids % val_block_period) != 0
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    val_mask = ~train_mask
    print(
        f"  Split mode: {split_mode} | train={train_mask.sum()}, val={val_mask.sum()}"
    )

    stem = Path(xdf_file_path).stem
    if stem.endswith("_meg"):
        stem = stem[:-4]

    os.makedirs(output_dir, exist_ok=True)

    common_payload = {
        "fs": np.float32(fs_emg),
        "window_len": np.int32(window_len),
        "step_len": np.int32(step_len),
        "valid_angle_cols": valid_angle_cols.astype(np.int32),
        "label_position": np.array(label_position),
        "source_file": np.array(str(xdf_file_path)),
        "emg_stream": np.array(emg_stream_name),
        "angle_stream": np.array(angle_stream_name),
    }

    out_train = Path(output_dir) / f"{stem}_train.npz"
    np.savez_compressed(
        out_train,
        emg=w_emg[train_mask],
        angles=w_ang[train_mask],
        window_time_s=window_time[train_mask],
        split=np.array("train"),
        **common_payload,
    )
    print(f"  Saved Train to {out_train}: {w_emg[train_mask].shape}")

    if np.any(val_mask):
        out_val = Path(output_dir) / f"{stem}_val.npz"
        np.savez_compressed(
            out_val,
            emg=w_emg[val_mask],
            angles=w_ang[val_mask],
            window_time_s=window_time[val_mask],
            split=np.array("val"),
            **common_payload,
        )
        print(f"  Saved Val   to {out_val}: {w_emg[val_mask].shape}")
    else:
        print("  Validation split skipped because only one window was available.")


def main():
    parser = argparse.ArgumentParser(
        description="Import sleeve XDF files to windowed train/val .npz files"
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default="data",
        help="Directory containing .xdf files (default: data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed data",
        help="Output directory for .npz files (default: processed data)",
    )
    parser.add_argument(
        "--emg_stream",
        type=str,
        default=DEFAULT_EMG_STREAM,
        help=f"EMG stream name (default: {DEFAULT_EMG_STREAM})",
    )
    parser.add_argument(
        "--angle_stream",
        type=str,
        default=DEFAULT_ANGLE_STREAM,
        help=f"Angle stream name (default: {DEFAULT_ANGLE_STREAM})",
    )
    parser.add_argument(
        "--window_len",
        type=int,
        default=400,
        help="Window length in EMG samples (default: 200 = 100ms at 2kHz)",
    )
    parser.add_argument(
        "--step_len",
        type=int,
        default=50,
        help="Step length in EMG samples (default: 50 = 25ms at 2kHz)",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of windows used for train split before the final contiguous val split (default: 0.8)",
    )
    parser.add_argument(
        "--emg_lowcut",
        type=float,
        default=20.0,
        help="EMG bandpass low cutoff in Hz (default: 20)",
    )
    parser.add_argument(
        "--emg_highcut",
        type=float,
        default=500.0,
        help="EMG bandpass high cutoff in Hz (default: 500)",
    )
    parser.add_argument(
        "--angle_lowpass",
        type=float,
        default=5.0,
        help="Angle lowpass cutoff in Hz; set <=0 to disable (default: 5.0)",
    )
    parser.add_argument(
        "--split_mode",
        choices=["chronological", "blocked"],
        default="blocked",
        help=(
            "How to split windows into train/val. "
            "'chronological' puts the first train_fraction in train and the rest in val. "
            "'blocked' interleaves train/val blocks throughout the session to reduce "
            "temporal distribution shift (default: blocked)."
        ),
    )
    parser.add_argument(
        "--val_block_period",
        type=int,
        default=5,
        help=(
            "For blocked split: every Nth block is validation (default: 5, i.e. ~20%% val). "
            "Ignored when split_mode=chronological."
        ),
    )
    parser.add_argument(
        "--label_position",
        choices=["center", "last"],
        default="last",
        help="Which sample inside each window supplies the target label (default: last)",
    )

    args = parser.parse_args()

    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("--train_fraction must be in the interval (0, 1].")

    db_dir = Path(args.db_dir)
    xdf_files = sorted(db_dir.rglob("*.xdf"))

    if not xdf_files:
        raise FileNotFoundError(f"No .xdf files found in {db_dir.resolve()}")

    print(f"Found {len(xdf_files)} XDF file(s) in {db_dir.resolve()}")
    for path in xdf_files:
        process_sleeve_file(
            path,
            args.output_dir,
            emg_stream_name=args.emg_stream,
            angle_stream_name=args.angle_stream,
            window_len=args.window_len,
            step_len=args.step_len,
            emg_lowcut=args.emg_lowcut,
            emg_highcut=args.emg_highcut,
            angle_lowpass=args.angle_lowpass,
            train_fraction=args.train_fraction,
            label_position=args.label_position,
            split_mode=args.split_mode,
            val_block_period=args.val_block_period,
        )


if __name__ == "__main__":
    main()
