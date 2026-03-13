"""
augment_processed_data.py
Create ring-aware rotation-augmented training data from processed sleeve `.npz` files.

Default behavior:
- reads from `processed data/`
- writes to `augmented data/`
- augments only `*_train.npz`
- copies `*_val.npz` unchanged
- keeps data in the flat `(N, 128, T)` format so it can be used with `sleeve_model.py`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "processed data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "augmented data"
DEFAULT_SEED = 42


def build_default_ring_map() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a 26 x 5 map from ring/slot positions to flat channel indices.

    The mapping follows the inferred physical sleeve layout:
      - 10 vertical strips total
      - first 2 rings have 4 electrodes
      - remaining 24 rings have 5 electrodes
      - channels are described 1-based and converted here to 0-based indices
    """

    strip_a = list(range(1, 14))
    strip_b = list(range(14, 27))
    strip_c = list(range(27, 40))
    strip_d = list(range(40, 53))
    strip_e = list(range(53, 65))
    strip_f = list(range(65, 77))
    strip_g = list(range(77, 90))
    strip_h = list(range(90, 103))
    strip_i = list(range(103, 116))
    strip_j = list(range(116, 129))

    rings_1_based = [
        [strip_a[0], strip_c[0], strip_g[0], strip_i[0], -1],
        [strip_b[0], strip_d[0], strip_h[0], strip_j[0], -1],
    ]

    for idx in range(1, 13):
        rings_1_based.append(
            [strip_a[idx], strip_c[idx], strip_e[idx - 1], strip_g[idx], strip_i[idx]]
        )
        rings_1_based.append(
            [strip_b[idx], strip_d[idx], strip_f[idx - 1], strip_h[idx], strip_j[idx]]
        )

    ring_map = np.asarray(rings_1_based, dtype=np.int64)
    valid_mask = ring_map > 0
    ring_map = ring_map - 1
    ring_map[~valid_mask] = -1
    return ring_map, valid_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create ring-aware rotation-augmented sleeve training data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing processed `*_train.npz` and `*_val.npz` files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where augmented `.npz` files will be written.",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Number of augmented copies to create for each training sample.",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=2,
        help="Maximum absolute circular slot shift for each augmented copy.",
    )
    parser.add_argument(
        "--exclude-original",
        action="store_true",
        help="If set, save only augmented copies and omit the original training samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible augmentation.",
    )
    return parser.parse_args()


def resolve_dir(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def flat_to_rings(emg_flat: np.ndarray, ring_map: np.ndarray) -> np.ndarray:
    n_samples, _, n_time = emg_flat.shape
    n_rings, n_slots = ring_map.shape
    out = np.zeros((n_samples, n_rings, n_slots, n_time), dtype=emg_flat.dtype)

    valid_mask = ring_map >= 0
    valid_r, valid_s = np.nonzero(valid_mask)
    valid_c = ring_map[valid_r, valid_s]
    out[:, valid_r, valid_s, :] = emg_flat[:, valid_c, :]
    return out


def rings_to_flat(
    emg_rings: np.ndarray, ring_map: np.ndarray, n_channels: int
) -> np.ndarray:
    n_samples, _, _, n_time = emg_rings.shape
    out = np.zeros((n_samples, n_channels, n_time), dtype=emg_rings.dtype)

    valid_mask = ring_map >= 0
    valid_r, valid_s = np.nonzero(valid_mask)
    valid_c = ring_map[valid_r, valid_s]
    out[:, valid_c, :] = emg_rings[:, valid_r, valid_s, :]
    return out


def rotate_valid_slots_inplace(
    ring_tensor: np.ndarray,
    ring_map: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    """
    Apply one global sample-wise shift across all rings, rotating only valid slots.

    Parameters
    ----------
    ring_tensor : np.ndarray
        Shape `(N, R, S, T)`.
    ring_map : np.ndarray
        Shape `(R, S)`, entries are flat channel indices or -1 for invalid slots.
    shifts : np.ndarray
        Shape `(N,)`, integer shift per sample.
    """
    out = ring_tensor.copy()
    n_samples, n_rings, _, _ = out.shape

    for ring_idx in range(n_rings):
        valid_slots = np.flatnonzero(ring_map[ring_idx] >= 0)
        n_valid = int(valid_slots.size)
        if n_valid <= 1:
            continue

        ring_vals = out[:, ring_idx, valid_slots, :].copy()
        for sample_idx in range(n_samples):
            shift = int(shifts[sample_idx]) % n_valid
            if shift != 0:
                ring_vals[sample_idx] = np.roll(
                    ring_vals[sample_idx], shift=shift, axis=0
                )
        out[:, ring_idx, valid_slots, :] = ring_vals

    return out


def augment_train_split(
    emg: np.ndarray,
    angles: np.ndarray,
    ring_map: np.ndarray,
    copies: int,
    max_shift: int,
    include_original: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if emg.ndim != 3:
        raise ValueError(f"Expected EMG shape (N, C, T), got {emg.shape}")
    if copies < 0:
        raise ValueError("copies must be >= 0")
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0")

    emg = np.asarray(emg, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)

    n_samples, n_channels, _ = emg.shape
    ring_tensor = flat_to_rings(emg, ring_map)

    emg_parts = []
    angle_parts = []
    if include_original:
        emg_parts.append(emg)
        angle_parts.append(angles)

    if copies == 0 or max_shift == 0:
        if not emg_parts:
            return emg.copy(), angles.copy()
        return np.concatenate(emg_parts, axis=0), np.concatenate(angle_parts, axis=0)

    shift_choices = np.arange(-max_shift, max_shift + 1, dtype=np.int64)
    for _ in range(copies):
        sample_shifts = rng.choice(shift_choices, size=n_samples, replace=True)
        aug_rings = rotate_valid_slots_inplace(ring_tensor, ring_map, sample_shifts)
        aug_flat = rings_to_flat(aug_rings, ring_map, n_channels)
        emg_parts.append(aug_flat.astype(np.float32, copy=False))
        angle_parts.append(angles)

    return np.concatenate(emg_parts, axis=0), np.concatenate(angle_parts, axis=0)


def process_file(
    src_path: Path,
    dst_path: Path,
    ring_map: np.ndarray,
    copies: int,
    max_shift: int,
    include_original: bool,
    rng: np.random.Generator,
):
    with np.load(src_path, allow_pickle=False) as data:
        emg = np.asarray(data["emg"], dtype=np.float32)
        angles = np.asarray(data["angles"], dtype=np.float32)

    if src_path.name.endswith("_train.npz"):
        out_emg, out_angles = augment_train_split(
            emg=emg,
            angles=angles,
            ring_map=ring_map,
            copies=copies,
            max_shift=max_shift,
            include_original=include_original,
            rng=rng,
        )
        print(
            f"[train] {src_path.name}: {len(emg)} -> {len(out_emg)} windows "
            f"(copies={copies}, max_shift={max_shift}, include_original={include_original})"
        )
    else:
        out_emg = emg
        out_angles = angles
        print(f"[val]   {src_path.name}: copied unchanged ({len(emg)} windows)")

    np.savez_compressed(dst_path, emg=out_emg, angles=out_angles)


def main():
    args = parse_args()
    input_dir = resolve_dir(args.input_dir)
    output_dir = resolve_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    ring_map, _ = build_default_ring_map()
    rng = np.random.default_rng(args.seed)
    include_original = not args.exclude_original

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files : {len(files)}")

    for src_path in files:
        dst_path = output_dir / src_path.name
        process_file(
            src_path=src_path,
            dst_path=dst_path,
            ring_map=ring_map,
            copies=args.copies,
            max_shift=args.max_shift,
            include_original=include_original,
            rng=rng,
        )

    print("Done.")


if __name__ == "__main__":
    main()
