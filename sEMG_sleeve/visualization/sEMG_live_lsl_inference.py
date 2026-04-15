from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import filtfilt, resample

try:
    from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
except Exception as exc:
    raise ImportError(
        "pylsl is required for live LSL inference. Install with: pip install pylsl"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sEMG_inference import (
    DEFAULT_PACKET_FORMAT,
    DEFAULT_PRED_UDP_PORT,
    DEFAULT_STEP_LEN,
    DEFAULT_UDP_HOST,
    apply_emg_transform,
    apply_input_scaler,
    build_model,
    inverse_target_scaler,
    load_json,
    load_scalers,
    serialize_packet,
)
from sleeve_preprocessing import DEFAULT_EMG_STREAM, butter_bandpass

DEFAULT_MODEL_DIR = PROJECT_DIR / "outputs_v16_best"
DEFAULT_MODEL_FS = 2000.0
DEFAULT_PRED_LSL_STREAM = "PredictedJointAngles"
DEFAULT_MAX_CHUNK = 512
DEFAULT_BUFFER_SECONDS = 5.0
DEFAULT_LOWCUT = 20.0
DEFAULT_HIGHCUT = 500.0
ANGLE_NAMES_14 = [
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "thumb_cmc_mcp",
    "thumb_ip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe to a live Open Ephys EMG LSL stream, run the trained sleeve "
            "regressor in realtime, and publish predicted joint angles over UDP "
            "and optionally LSL."
        )
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
        help=f"LSL EMG stream name (default: {DEFAULT_EMG_STREAM}).",
    )
    parser.add_argument(
        "--model-fs",
        type=float,
        default=DEFAULT_MODEL_FS,
        help=(
            "Sampling rate used during model training. Used to map live LSL windows "
            "onto the saved model window size."
        ),
    )
    parser.add_argument(
        "--input-fs",
        type=float,
        default=0.0,
        help=(
            "Override the input LSL sample rate. Leave at 0 to use the stream's "
            "nominal rate."
        ),
    )
    parser.add_argument(
        "--step-len",
        type=int,
        default=DEFAULT_STEP_LEN,
        help="Stride between successive predictions in model-domain samples.",
    )
    parser.add_argument(
        "--label-position",
        choices=["center", "last"],
        default="last",
        help="Which sample inside each processed window gets the packet timestamp.",
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default=DEFAULT_LOWCUT,
        help="Bandpass low cutoff in Hz.",
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=DEFAULT_HIGHCUT,
        help="Bandpass high cutoff in Hz.",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=DEFAULT_BUFFER_SECONDS,
        help="Maximum rolling live buffer length in seconds.",
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=DEFAULT_MAX_CHUNK,
        help="Maximum number of samples to pull from the LSL inlet per poll.",
    )
    parser.add_argument(
        "--udp-host",
        type=str,
        default=DEFAULT_UDP_HOST,
        help=f"UDP host for predicted angle packets (default: {DEFAULT_UDP_HOST}).",
    )
    parser.add_argument(
        "--pred-udp-port",
        type=int,
        default=DEFAULT_PRED_UDP_PORT,
        help=f"UDP port for predicted angle packets (default: {DEFAULT_PRED_UDP_PORT}).",
    )
    parser.add_argument(
        "--packet-format",
        choices=["json", "csv"],
        default=DEFAULT_PACKET_FORMAT,
        help="Serialization format for predicted UDP packets.",
    )
    parser.add_argument(
        "--pred-lsl-stream",
        type=str,
        default=DEFAULT_PRED_LSL_STREAM,
        help=(
            "Optional predicted-angle LSL stream name. Pass an empty string to disable "
            "predicted-angle LSL output."
        ),
    )
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the input LSL stream before failing.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="Print one status line every N predictions.",
    )
    return parser.parse_args()


def resolve_inlet(stream_name: str, timeout_s: float) -> StreamInlet:
    deadline = time.time() + max(0.0, float(timeout_s))
    while True:
        streams = resolve_byprop("name", stream_name, timeout=0.25)
        if streams:
            return StreamInlet(streams[0], max_buflen=60)
        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for LSL stream '{stream_name}'")


def get_angle_names(n_joints: int) -> list[str]:
    if int(n_joints) == len(ANGLE_NAMES_14):
        return list(ANGLE_NAMES_14)
    return [f"angle_{idx:02d}" for idx in range(int(n_joints))]


def create_prediction_outlet(
    stream_name: str, angle_names: list[str]
) -> StreamOutlet | None:
    if not stream_name:
        return None

    info = StreamInfo(
        name=stream_name,
        type="JointAngles",
        channel_count=len(angle_names),
        nominal_srate=0.0,
        channel_format="float32",
        source_id=f"semg-live-{int(time.time())}",
    )
    try:
        channels = info.desc().append_child("channels")
        for angle_name in angle_names:
            channel = channels.append_child("channel")
            channel.append_child_value("label", angle_name)
            channel.append_child_value("unit", "deg")
    except Exception:
        pass
    return StreamOutlet(info)


def infer_input_fs(inlet: StreamInlet, override_fs: float) -> float:
    if override_fs > 0:
        return float(override_fs)

    try:
        nominal = float(inlet.info().nominal_srate())
    except Exception:
        nominal = 0.0

    if nominal <= 0:
        raise RuntimeError(
            "Input LSL stream did not report a valid nominal sample rate. "
            "Re-run with --input-fs <Hz>."
        )
    return nominal


def select_packet_timestamp(
    timestamps: np.ndarray, label_position: str, input_window_samples: int
) -> float:
    if label_position == "center":
        return float(timestamps[input_window_samples // 2])
    return float(timestamps[input_window_samples - 1])


def format_named_angles(angle_names: list[str], angle_values: np.ndarray) -> str:
    return ", ".join(
        f"{name}={float(value):6.2f}"
        for name, value in zip(angle_names, np.asarray(angle_values).reshape(-1))
    )


def main() -> None:
    args = parse_args()

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

    model_window_size = int(config["model"]["window_size"])
    expected_channels = int(config["model"]["n_ch"])
    n_joints = int(config["model"]["n_joints"])
    angle_names = get_angle_names(n_joints)
    emg_transform = config["data"].get("emg_transform", "none")
    model_fs = float(args.model_fs)

    inlet = resolve_inlet(args.emg_stream, args.stream_timeout)
    input_fs = infer_input_fs(inlet, args.input_fs)

    info = inlet.info()
    try:
        inlet_channels = int(info.channel_count())
    except Exception:
        inlet_channels = expected_channels

    if inlet_channels != expected_channels:
        raise RuntimeError(
            f"Channel count mismatch: stream has {inlet_channels} channels, "
            f"model expects {expected_channels}."
        )

    input_window_samples = max(1, int(round(model_window_size * input_fs / model_fs)))
    input_step_samples = max(1, int(round(int(args.step_len) * input_fs / model_fs)))
    max_buffer_samples = max(
        input_window_samples * 4, int(round(float(args.buffer_seconds) * input_fs))
    )

    b_emg, a_emg = butter_bandpass(args.lowcut, args.highcut, input_fs, order=4)

    pred_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pred_outlet = create_prediction_outlet(args.pred_lsl_stream.strip(), angle_names)

    sample_buffer = np.zeros((0, expected_channels), dtype=np.float32)
    time_buffer = np.zeros((0,), dtype=np.float64)
    prediction_count = 0
    start_time = time.time()

    print(f"Model dir          : {model_dir}")
    print(f"Input stream       : {args.emg_stream}")
    print(f"Device             : {device}")
    print(f"Input fs           : {input_fs:.2f} Hz")
    print(f"Model fs           : {model_fs:.2f} Hz")
    print(f"Channels           : {expected_channels}")
    print(f"Output joints      : {', '.join(angle_names)}")
    print(f"Model window       : {model_window_size} samples")
    print(f"Input window       : {input_window_samples} samples")
    print(f"Model step         : {int(args.step_len)} samples")
    print(f"Input step         : {input_step_samples} samples")
    print(f"UDP output         : {args.udp_host}:{args.pred_udp_port}")
    print(
        f"Predicted LSL      : {args.pred_lsl_stream if pred_outlet is not None else 'disabled'}"
    )
    print("Live prediction started. Press Ctrl+C to stop.")

    try:
        while True:
            samples, timestamps = inlet.pull_chunk(
                timeout=0.1, max_samples=args.max_chunk
            )
            if not samples:
                continue

            chunk = np.asarray(samples, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            chunk_ts = np.asarray(timestamps, dtype=np.float64)
            if chunk.shape[1] != expected_channels:
                raise RuntimeError(
                    f"Live chunk shape mismatch: got {chunk.shape[1]} channels, "
                    f"expected {expected_channels}."
                )

            sample_buffer = np.vstack([sample_buffer, chunk])
            time_buffer = np.concatenate([time_buffer, chunk_ts])

            if sample_buffer.shape[0] > max_buffer_samples:
                keep = sample_buffer.shape[0] - max_buffer_samples
                sample_buffer = sample_buffer[keep:]
                time_buffer = time_buffer[keep:]

            while sample_buffer.shape[0] >= input_window_samples:
                live_window = sample_buffer[:input_window_samples].T
                window_ts = time_buffer[:input_window_samples]

                filtered = filtfilt(b_emg, a_emg, live_window, axis=1).astype(
                    np.float32
                )
                if input_window_samples != model_window_size:
                    filtered = resample(filtered, model_window_size, axis=1).astype(
                        np.float32
                    )

                filtered = apply_emg_transform(filtered, emg_transform)
                filtered = apply_input_scaler(filtered, config, scalers)
                x = torch.from_numpy(filtered[None, ...]).to(device)

                with torch.no_grad():
                    pred_scaled = model(x).detach().cpu().numpy().astype(np.float32)
                pred = inverse_target_scaler(pred_scaled, config, scalers)[0]

                packet_ts = select_packet_timestamp(
                    window_ts, args.label_position, input_window_samples
                )
                packet = {
                    "source_stream": args.emg_stream,
                    "packet_index": int(prediction_count),
                    "timestamp_s": float(packet_ts),
                    "window_start": int(
                        0
                        if time_buffer.size == 0
                        else prediction_count * input_step_samples
                    ),
                    "window_end": int(input_window_samples),
                    "angle_names": angle_names,
                    "angles_deg": [float(v) for v in pred.tolist()],
                    "kind": "predicted",
                }
                payload = serialize_packet(packet, args.packet_format)
                pred_sock.sendto(payload, (args.udp_host, args.pred_udp_port))

                if pred_outlet is not None:
                    pred_outlet.push_sample(np.asarray(pred, dtype=np.float32).tolist())

                prediction_count += 1
                if prediction_count <= 3 or (
                    args.print_every > 0
                    and prediction_count % int(args.print_every) == 0
                ):
                    elapsed = max(1e-6, time.time() - start_time)
                    rate_hz = prediction_count / elapsed
                    named_angles = format_named_angles(angle_names, pred)
                    print(
                        f"[{prediction_count:05d}] t={packet_ts:.3f}s rate={rate_hz:6.2f} Hz "
                        f"{named_angles}"
                    )

                sample_buffer = sample_buffer[input_step_samples:]
                time_buffer = time_buffer[input_step_samples:]

    except KeyboardInterrupt:
        print("\nStopping live prediction...")
    finally:
        pred_sock.close()


if __name__ == "__main__":
    main()
