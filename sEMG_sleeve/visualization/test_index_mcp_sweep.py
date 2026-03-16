"""Synthetic UDP publisher for testing the index MCP mapping in Unity.

This sends paired predicted and ground-truth packets to the same ports used by
`handtrack_data_handler.py`:

- predicted -> 5020
- ground truth -> 5025

By default, it sweeps the first model angle (`index_mcp`) from 0 -> 90 -> 0
while keeping the other 13 angles at 0. This lets you verify whether the
handler maps the index MCP to the intended Unity joint or if it is shifted by
one bone.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Iterable, List

PRED_HOST = "127.0.0.1"
PRED_PORT = 5020
GT_HOST = "127.0.0.1"
GT_PORT = 5025

ANGLE_NAMES = [
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


def build_sweep(max_angle: float, step_deg: float) -> List[float]:
    if step_deg <= 0:
        raise ValueError("step_deg must be > 0")

    up_count = max(1, int(round(max_angle / step_deg)))
    upward = [i * max_angle / up_count for i in range(up_count + 1)]
    downward = upward[-2::-1]
    return upward + downward


def send_packet(
    sock: socket.socket,
    host: str,
    port: int,
    kind: str,
    packet_index: int,
    timestamp_s: float,
    angles_deg: Iterable[float],
) -> None:
    packet = {
        "kind": kind,
        "packet_index": packet_index,
        "timestamp_s": timestamp_s,
        "angles_deg": [float(v) for v in angles_deg],
    }
    sock.sendto(json.dumps(packet).encode("utf-8"), (host, port))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep synthetic index_mcp packets through handtrack_data_handler.py"
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=90.0,
        help="Maximum sweep angle in degrees.",
    )
    parser.add_argument(
        "--step-deg",
        type=float,
        default=5.0,
        help="Angular step per packet in degrees.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Packet rate in Hz.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of 0->max->0 cycles. Use 0 to loop forever.",
    )
    parser.add_argument(
        "--joint-index",
        type=int,
        default=0,
        help="Angle index to sweep. Default 0 = index_mcp.",
    )
    parser.add_argument(
        "--pred-scale",
        type=float,
        default=1.0,
        help="Scale applied to the predicted sweep angle.",
    )
    parser.add_argument(
        "--gt-scale",
        type=float,
        default=1.0,
        help="Scale applied to the ground-truth sweep angle.",
    )
    args = parser.parse_args()

    if not 0 <= args.joint_index < len(ANGLE_NAMES):
        raise ValueError(
            f"joint-index must be between 0 and {len(ANGLE_NAMES) - 1}, got {args.joint_index}"
        )
    if args.fps <= 0:
        raise ValueError("fps must be > 0")

    sweep_values = build_sweep(args.max_angle, args.step_deg)
    frame_interval = 1.0 / args.fps
    joint_name = ANGLE_NAMES[args.joint_index]

    print("Synthetic sEMG sweep test")
    print(f"  Swept angle: {joint_name} (index {args.joint_index})")
    print(f"  Sweep: 0 -> {args.max_angle:.1f} -> 0 deg")
    print(f"  Step: {args.step_deg:.1f} deg | Rate: {args.fps:.1f} Hz")
    print(f"  Pred port: {PRED_PORT} | GT port: {GT_PORT}")
    print("  Start `handtrack_data_handler.py` and both Unity listeners first.")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet_index = 0
    cycles_completed = 0

    try:
        while True:
            cycle_start = time.perf_counter()
            for angle in sweep_values:
                loop_start = time.perf_counter()
                timestamp_s = time.time()

                pred_angles = [0.0] * len(ANGLE_NAMES)
                gt_angles = [0.0] * len(ANGLE_NAMES)
                pred_angles[args.joint_index] = angle * args.pred_scale
                gt_angles[args.joint_index] = angle * args.gt_scale

                send_packet(
                    sock,
                    PRED_HOST,
                    PRED_PORT,
                    "predicted",
                    packet_index,
                    timestamp_s,
                    pred_angles,
                )
                send_packet(
                    sock,
                    GT_HOST,
                    GT_PORT,
                    "ground_truth",
                    packet_index,
                    timestamp_s,
                    gt_angles,
                )

                if packet_index % 10 == 0:
                    print(
                        f"packet={packet_index:05d} | {joint_name} pred={pred_angles[args.joint_index]:6.2f} | "
                        f"gt={gt_angles[args.joint_index]:6.2f}"
                    )

                packet_index += 1
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            cycles_completed += 1
            cycle_elapsed = time.perf_counter() - cycle_start
            print(
                f"Completed cycle {cycles_completed} in {cycle_elapsed:.2f}s"
            )

            if args.cycles > 0 and cycles_completed >= args.cycles:
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
