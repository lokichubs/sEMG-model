"""
sEMG_inference.py

Replay a raw sleeve XDF file as if it were a live EMG stream, run the trained model
window-by-window, and emit predicted joint angles over UDP.

Default behavior is set up for the current baseline model in `outputs_best_v8/`.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyxdf
import torch
from scipy.signal import filtfilt

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sleeve_model import SleeveCNNAttentionImproved
from sleeve_preprocessing import (
    DEFAULT_ANGLE_STREAM,
    DEFAULT_EMG_STREAM,
    _ensure_2d,
    _get_stream_by_name,
    _stream_nominal_srate,
    _valid_angle_columns,
    butter_bandpass,
    interpolate_targets_to_emg,
)
from sleeve_TCN_model import SleeveTCNRegressor

DEFAULT_MODEL_DIR = PROJECT_DIR / "outputs_v16_best"
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_PRED_UDP_PORT = 5020
DEFAULT_GT_UDP_PORT = 5025
DEFAULT_STEP_LEN = 50
DEFAULT_LABEL_POSITION = "last"
DEFAULT_PACKET_FORMAT = "json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay raw sleeve EMG from an XDF file, predict angles with a trained "
            "model, and stream predictions over UDP."
        )
    )
    parser.add_argument(
        "--xdf-file",
        type=str,
        default=None,
        help=(
            "Path to a raw .xdf file. If omitted, the first .xdf file in data/ is used."
        ),
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help=(
            "Optional subject selector, e.g. 1, 001, or sub-001. "
            "Used to locate a matching XDF file when --xdf-file is not provided."
        ),
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help=(
            "Optional session selector, e.g. 2, 002, ses-002, or sess-002. "
            "Used to locate a matching XDF file when --xdf-file is not provided."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing best_model.pt, config.json, and scaler_params.npz.",
    )
    parser.add_argument(
        "--emg-stream",
        type=str,
        default=DEFAULT_EMG_STREAM,
        help=f"EMG stream name inside the XDF file (default: {DEFAULT_EMG_STREAM}).",
    )
    parser.add_argument(
        "--angle-stream",
        type=str,
        default=DEFAULT_ANGLE_STREAM,
        help=f"Angle stream name inside the XDF file (default: {DEFAULT_ANGLE_STREAM}).",
    )
    parser.add_argument(
        "--udp-host",
        type=str,
        default=DEFAULT_UDP_HOST,
        help=f"UDP host to send predictions to (default: {DEFAULT_UDP_HOST}).",
    )
    parser.add_argument(
        "--pred-udp-port",
        type=int,
        default=DEFAULT_PRED_UDP_PORT,
        help=f"UDP port to send predicted angles to (default: {DEFAULT_PRED_UDP_PORT}).",
    )
    parser.add_argument(
        "--gt-udp-port",
        type=int,
        default=DEFAULT_GT_UDP_PORT,
        help=f"UDP port to send ground-truth angles to (default: {DEFAULT_GT_UDP_PORT}).",
    )
    parser.add_argument(
        "--step-len",
        type=int,
        default=DEFAULT_STEP_LEN,
        help="Stride between inference windows in EMG samples (default: 50).",
    )
    parser.add_argument(
        "--label-position",
        choices=["center", "last"],
        default=DEFAULT_LABEL_POSITION,
        help="Which sample inside each window gets the output timestamp (default: last).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between windows to replay the XDF file in wall-clock time.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed factor when --realtime is enabled (default: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference but do not send UDP packets.",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Optional limit on the number of UDP packets/predictions to emit.",
    )
    parser.add_argument(
        "--packet-format",
        choices=["json", "csv"],
        default=DEFAULT_PACKET_FORMAT,
        help="Serialization format for UDP packets (default: json).",
    )
    return parser.parse_args()


def _normalize_subject_token(subject: str | None) -> str | None:
    if subject is None:
        return None
    s = str(subject).strip().lower()
    if s.startswith("sub-"):
        s = s[4:]
    elif s.startswith("sub"):
        s = s[3:]
    if s == "":
        return None
    try:
        return f"sub-{int(s):03d}"
    except ValueError:
        return f"sub-{s}"


def _normalize_session_token(session: str | None) -> str | None:
    if session is None:
        return None
    s = str(session).strip().lower()
    if s.startswith("sess-"):
        s = s[5:]
    elif s.startswith("sess"):
        s = s[4:]
    elif s.startswith("ses-"):
        s = s[4:]
    elif s.startswith("ses"):
        s = s[3:]
    if s == "":
        return None
    try:
        return f"ses-{int(s):03d}"
    except ValueError:
        return f"ses-{s}"


def resolve_xdf_file(
    xdf_file: str | None,
    subject: str | None = None,
    session: str | None = None,
) -> Path:
    if xdf_file is not None:
        path = Path(xdf_file)
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"XDF file not found: {path}")
        return path

    candidates = sorted(DEFAULT_DATA_DIR.glob("*.xdf"))
    if not candidates:
        raise FileNotFoundError(f"No .xdf files found in {DEFAULT_DATA_DIR}")

    subject_token = _normalize_subject_token(subject)
    session_token = _normalize_session_token(session)
    if subject_token is not None or session_token is not None:
        filtered = []
        for candidate in candidates:
            stem = candidate.stem.lower()
            if subject_token is not None and subject_token not in stem:
                continue
            if session_token is not None and session_token not in stem:
                continue
            filtered.append(candidate)

        if len(filtered) == 0:
            raise FileNotFoundError(
                "No .xdf file matched the requested subject/session: "
                f"subject={subject!r}, session={session!r}"
            )
        if len(filtered) > 1:
            if session_token is None and subject_token is not None:
                chosen = random.choice(filtered)
                print(
                    "Multiple sessions matched the requested subject; "
                    f"randomly selected: {chosen.name}"
                )
                return chosen

            options = ", ".join(path.name for path in filtered)
            raise RuntimeError(
                "Multiple .xdf files matched the requested subject/session. "
                f"Please be more specific or pass --xdf-file directly. Matches: {options}"
            )
        return filtered[0]

    return candidates[0]


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scalers(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    train_cfg = config["training"]
    model_key = model_cfg.get("model_key", "baseline")

    if (
        model_key == "baseline"
        or model_cfg.get("architecture") == "SleeveCNNAttentionImproved"
    ):
        model = SleeveCNNAttentionImproved(
            n_ch=int(model_cfg["n_ch"]),
            window_size=int(model_cfg["window_size"]),
            n_joints=int(model_cfg["n_joints"]),
            hidden=int(model_cfg["hidden"]),
            n_attn=int(model_cfg["n_attn"]),
            n_heads=int(model_cfg["n_heads"]),
            dropout=float(train_cfg["dropout"]),
            cnn_activation=str(model_cfg.get("cnn_activation", "elu")),
            attn_ff_activation=str(model_cfg.get("attn_ff_activation", "elu")),
            mlp_activation=str(model_cfg.get("mlp_activation", "elu")),
        )
        return model

    if model_key == "tcn" or model_cfg.get("architecture") == "SleeveTCNRegressor":
        model = SleeveTCNRegressor(
            n_ch=int(model_cfg["n_ch"]),
            window_size=int(model_cfg["window_size"]),
            n_joints=int(model_cfg["n_joints"]),
            hidden=int(model_cfg["hidden"]),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            dilations=tuple(model_cfg.get("dilations", [1, 2, 4, 8])),
            dropout=float(train_cfg["dropout"]),
        )
        return model

    raise ValueError(
        f"Unsupported model configuration: model_key={model_key}, architecture={model_cfg.get('architecture')}"
    )


def apply_emg_transform(window: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return window.astype(np.float32)
    if mode == "log1p":
        return np.log1p(np.abs(window)).astype(np.float32)
    raise ValueError(f"Unsupported EMG transform: {mode}")


def apply_input_scaler(
    window: np.ndarray, config: dict[str, Any], scalers: dict[str, np.ndarray]
) -> np.ndarray:
    mode = config["data"].get("input_scaler", "none")
    if mode == "none":
        return window.astype(np.float32)
    if mode != "standard":
        raise ValueError(f"Unsupported input scaler mode for inference: {mode}")

    flat = window.reshape(1, -1)
    mean = scalers["x_mean"].reshape(1, -1)
    std = scalers["x_std"].reshape(1, -1)
    return ((flat - mean) / std).reshape(window.shape).astype(np.float32)


def inverse_target_scaler(
    pred: np.ndarray, config: dict[str, Any], scalers: dict[str, np.ndarray]
) -> np.ndarray:
    mode = config["data"].get("target_scaler", "none")
    if mode == "none":
        return pred.astype(np.float32)
    if mode == "standard":
        mean = scalers["y_mean"].reshape(1, -1)
        std = scalers["y_std"].reshape(1, -1)
        return (pred * std + mean).astype(np.float32)
    raise ValueError(f"Unsupported target scaler mode for inference: {mode}")


def load_stream_data(
    xdf_path: Path,
    emg_stream_name: str,
    angle_stream_name: str,
    lowcut: float = 20.0,
    highcut: float = 500.0,
):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    emg_stream = _get_stream_by_name(streams, emg_stream_name)
    angle_stream = _get_stream_by_name(streams, angle_stream_name)

    emg_ts = np.asarray(emg_stream.get("time_stamps", []), dtype=np.float64)
    angle_ts = np.asarray(angle_stream.get("time_stamps", []), dtype=np.float64)
    emg_x = _ensure_2d(emg_stream.get("time_series", []), "EMG")
    angle_x = _ensure_2d(angle_stream.get("time_series", []), "Angles")
    emg_x = np.asarray(emg_x, dtype=np.float32)
    angle_x = np.asarray(angle_x, dtype=np.float32)

    if emg_ts.size == 0 or emg_x.shape[0] == 0:
        raise RuntimeError(f"EMG stream '{emg_stream_name}' in {xdf_path} is empty.")
    if angle_ts.size == 0 or angle_x.shape[0] == 0:
        raise RuntimeError(
            f"Angle stream '{angle_stream_name}' in {xdf_path} is empty."
        )

    shared_start = max(emg_ts[0], angle_ts[0])
    shared_end = min(emg_ts[-1], angle_ts[-1])
    if shared_end <= shared_start:
        raise RuntimeError("EMG and angle streams have no overlapping timestamps.")

    emg_mask = (emg_ts >= shared_start) & (emg_ts <= shared_end)
    angle_mask = (angle_ts >= shared_start) & (angle_ts <= shared_end)
    emg_ts = emg_ts[emg_mask]
    emg_x = emg_x[emg_mask]
    angle_ts = angle_ts[angle_mask]
    angle_x = angle_x[angle_mask]

    valid_angle_cols = _valid_angle_columns(angle_x)
    if len(valid_angle_cols) == 0:
        raise RuntimeError("No usable angle columns found in the angle stream.")
    angle_x = angle_x[:, valid_angle_cols]

    fs_emg = _stream_nominal_srate(emg_stream)
    b_emg, a_emg = butter_bandpass(lowcut, highcut, fs_emg, order=4)
    filtered_emg = filtfilt(b_emg, a_emg, emg_x, axis=0).astype(np.float32)

    gt_angles = interpolate_targets_to_emg(angle_ts, angle_x, emg_ts)
    finite_cols = np.where(np.isfinite(gt_angles).all(axis=0))[0]
    if len(finite_cols) == 0:
        raise RuntimeError("No fully finite interpolated angle columns remained.")
    gt_angles = gt_angles[:, finite_cols].astype(np.float32)

    valid_rows = (~np.isnan(filtered_emg).all(axis=1)) & (
        ~np.isnan(gt_angles).all(axis=1)
    )
    dropped_rows = int(valid_rows.size - np.count_nonzero(valid_rows))
    if dropped_rows > 0:
        filtered_emg = filtered_emg[valid_rows]
        gt_angles = gt_angles[valid_rows]
        emg_ts = emg_ts[valid_rows]
        print(f"Dropped {dropped_rows} all-NaN aligned row(s) before visualization.")

    if filtered_emg.shape[0] == 0 or gt_angles.shape[0] == 0:
        raise RuntimeError(
            "No valid aligned samples remained after dropping all-NaN rows."
        )

    return filtered_emg, gt_angles, emg_ts, float(fs_emg)


def iter_windows(
    emg: np.ndarray,
    timestamps: np.ndarray,
    window_len: int,
    step_len: int,
    label_position: str,
):
    n_samples = emg.shape[0]
    n_windows = ((n_samples - window_len) // step_len) + 1
    if n_windows <= 0:
        return

    for i in range(n_windows):
        start = i * step_len
        end = start + window_len
        window = emg[start:end, :].T.astype(np.float32)
        if label_position == "last":
            label_idx = end - 1
        else:
            label_idx = start + (window_len // 2)
        yield i, start, end, timestamps[label_idx], window


def serialize_packet(packet: dict[str, Any], packet_format: str) -> bytes:
    if packet_format == "json":
        return json.dumps(packet).encode("utf-8")
    if packet_format == "csv":
        values = [f"{packet['timestamp_s']:.6f}"] + [
            f"{v:.6f}" for v in packet["angles_deg"]
        ]
        return ",".join(values).encode("utf-8")
    raise ValueError(f"Unsupported packet format: {packet_format}")


def main() -> None:
    args = parse_args()

    xdf_path = resolve_xdf_file(args.xdf_file, args.subject, args.session)
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = BASE_DIR / model_dir

    config = load_json(model_dir / "config.json")
    scalers = load_scalers(model_dir / "scaler_params.npz")

    model = build_model(config)
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    filtered_emg, gt_angles, emg_ts, fs_emg = load_stream_data(
        xdf_path,
        args.emg_stream,
        args.angle_stream,
    )

    window_len = int(config["model"]["window_size"])
    step_len = int(args.step_len)
    emg_transform = config["data"].get("emg_transform", "none")

    print(f"Model dir: {model_dir}")
    print(f"XDF file : {xdf_path}")
    print(f"Device   : {device}")
    print(f"EMG fs   : {fs_emg:.2f} Hz")
    print(f"Window   : {window_len} samples")
    print(f"Step     : {step_len} samples")
    print(
        f"UDP      : pred={args.udp_host}:{args.pred_udp_port}, "
        f"gt={args.udp_host}:{args.gt_udp_port} | dry_run={args.dry_run}"
    )
    print("Starting playback inference...")

    pred_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gt_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_emit_ts = None
    n_sent = 0

    with torch.no_grad():
        for i, start, end, packet_ts, window in iter_windows(
            filtered_emg,
            emg_ts,
            window_len,
            step_len,
            args.label_position,
        ):
            window = apply_emg_transform(window, emg_transform)
            window = apply_input_scaler(window, config, scalers)
            x = torch.from_numpy(window[None, ...]).to(device)

            pred_scaled = model(x).detach().cpu().numpy().astype(np.float32)
            pred = inverse_target_scaler(pred_scaled, config, scalers)[0]
            gt = (
                gt_angles[end - 1]
                if args.label_position == "last"
                else gt_angles[start + (window_len // 2)]
            )

            if args.realtime and last_emit_ts is not None:
                dt = max(0.0, float(packet_ts - last_emit_ts))
                speed = max(1e-6, float(args.speed))
                time.sleep(dt / speed)
            last_emit_ts = float(packet_ts)

            pred_packet = {
                "source_file": xdf_path.name,
                "packet_index": int(i),
                "timestamp_s": float(packet_ts),
                "window_start": int(start),
                "window_end": int(end),
                "angles_deg": [float(v) for v in pred.tolist()],
                "kind": "predicted",
            }
            gt_packet = {
                "source_file": xdf_path.name,
                "packet_index": int(i),
                "timestamp_s": float(packet_ts),
                "window_start": int(start),
                "window_end": int(end),
                "angles_deg": [float(v) for v in gt.tolist()],
                "kind": "ground_truth",
            }
            pred_payload = serialize_packet(pred_packet, args.packet_format)
            gt_payload = serialize_packet(gt_packet, args.packet_format)

            if not args.dry_run:
                pred_sock.sendto(pred_payload, (args.udp_host, args.pred_udp_port))
                gt_sock.sendto(gt_payload, (args.udp_host, args.gt_udp_port))

            n_sent += 1
            if n_sent <= 3 or n_sent % 50 == 0:
                print(
                    f"[{n_sent:05d}] t={packet_ts:.3f}s "
                    f"pred[0:3]={np.array2string(pred[:3], precision=2, suppress_small=True)} "
                    f"gt[0:3]={np.array2string(gt[:3], precision=2, suppress_small=True)}"
                )

            if args.max_packets is not None and n_sent >= int(args.max_packets):
                break

    pred_sock.close()
    gt_sock.close()
    print(
        f"Done. Emitted {n_sent} synchronized packet pair(s) "
        f"to ports {args.pred_udp_port}/{args.gt_udp_port}."
    )


if __name__ == "__main__":
    main()
