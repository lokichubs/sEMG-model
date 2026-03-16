"""
Angle Forwarder: Receives predicted and ground-truth angle packets from
`sEMG_inference.py` and forwards each stream to Unity in the existing 78-element
comma-separated format.
"""

import json
import select
import socket

# --- Config ---
UDP_IP = "127.0.0.1"
PRED_UDP_PORT = 5020
GT_UDP_PORT = 5025
UNITY_IP = "127.0.0.1"
UNITY_PRED_PORT = 5016
UNITY_GT_PORT = 5017

pred_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pred_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
pred_sock.bind((UDP_IP, PRED_UDP_PORT))
pred_sock.setblocking(False)

gt_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
gt_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
gt_sock.bind((UDP_IP, GT_UDP_PORT))
gt_sock.setblocking(False)

unity_pred_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
unity_gt_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Unity 26-joint hierarchy mapping
# 0: Wrist, 1: Palm (skipped)
# Fingers start at index 2
finger_starts = {"index": 2, "middle": 7, "ring": 12, "little": 17, "thumb": 22}

# Tracker angle names per finger
# These must match what finger_bend_angles() in the GUI actually produces:
#   index_mcp, index_pip, index_dip
#   middle_mcp, middle_pip, middle_dip
#   ring_mcp, ring_pip, ring_dip
#   pinky_mcp, pinky_pip, pinky_dip
#   thumb_cmc_mcp, thumb_ip
finger_angle_names = {
    "index": ["index_mcp", "index_pip", "index_dip"],
    "middle": ["middle_mcp", "middle_pip", "middle_dip"],
    "ring": ["ring_mcp", "ring_pip", "ring_dip"],
    "little": ["pinky_mcp", "pinky_pip", "pinky_dip"],
    "thumb": ["thumb_cmc_mcp", "thumb_ip"],
}

# # (comment out this entire block to disable splay forwarding)
# # --- Splay angles ---
# finger_splay_names = {
#     "index": "index_splay",
#     "middle": "middle_splay",
#     "ring": "ring_splay",
#     "little": "pinky_splay",
# }
# # --- End splay ---


print(f"FORWARDER: Listening for predicted angles on {UDP_IP}:{PRED_UDP_PORT}")
print(f"FORWARDER: Listening for ground-truth angles on {UDP_IP}:{GT_UDP_PORT}")
print(
    f"FORWARDER: Sending predicted hand to Unity {UNITY_IP}:{UNITY_PRED_PORT} "
    f"and ground-truth hand to Unity {UNITY_IP}:{UNITY_GT_PORT}"
)
print("Waiting for sEMG inference data...\n")


def map_tracker_to_unity(tracker_angles):
    """Map tracker angle dict to Unity's 78-element (26 joints × 3 axes) string."""
    data_list = ["nan"] * 78

    for finger, start_idx in finger_starts.items():
        angles_for_finger = finger_angle_names[finger]
        # All fingers: skip bone 0 (metacarpal), angles map to bones 1, 2, 3
        # Thumb: 22+1=23 (Proximal), 22+2=24 (Distal)
        # Index: 2+1=3 (Proximal), 2+2=4 (Intermediate), 2+3=5 (Distal)
        for i, angle_name in enumerate(angles_for_finger):
            if angle_name in tracker_angles:
                joint_idx = (start_idx + i + 1) * 3
                val = tracker_angles[angle_name]
                data_list[joint_idx] = str(round(val, 4))

        # # --- Splay: write to Y slot of metacarpal (comment out to disable) ---
        # if finger in finger_splay_names:
        #     splay_name = finger_splay_names[finger]
        #     if splay_name in tracker_angles:
        #         metacarpal_y_slot = start_idx * 3 + 1  # Y slot of bone 0
        #         data_list[metacarpal_y_slot] = str(round(tracker_angles[splay_name], 4))
        # # --- End splay ---

    return ",".join(data_list)


def map_angle_vector_to_unity(angle_values):
    """Map the 14-angle sEMG model output to Unity's 78-element CSV format."""
    data_list = ["nan"] * 78
    angle_values = list(angle_values)

    ordered_fingers = ["index", "middle", "ring", "little", "thumb"]
    cursor = 0
    for finger in ordered_fingers:
        start_idx = finger_starts[finger]
        n_angles = len(finger_angle_names[finger])
        finger_values = angle_values[cursor : cursor + n_angles]
        for i, val in enumerate(finger_values):
            joint_idx = (start_idx + i + 1) * 3
            data_list[joint_idx] = str(round(float(val), 4))
        cursor += n_angles

    return ",".join(data_list)


def forward_packet(raw_data, unity_socket, unity_port, label):
    packet = json.loads(raw_data.decode("utf-8"))

    if "angles_deg" in packet:
        unity_message = map_angle_vector_to_unity(packet.get("angles_deg", []))
    else:
        hand_list = packet.get("angles")
        if hand_list is None:
            hand_list = packet.get("hands", [])
        if not hand_list:
            return None
        first_hand = hand_list[0]
        unity_message = map_tracker_to_unity(first_hand.get("angles", {}))

    unity_socket.sendto(unity_message.encode("utf-8"), (UNITY_IP, unity_port))
    return packet


# --- Main Loop ---
frame_count = 0
first_pred_packet = True
first_gt_packet = True

try:
    while True:
        try:
            readable, _, _ = select.select([pred_sock, gt_sock], [], [], 1.0)
            if not readable:
                continue

            for ready_sock in readable:
                data, _ = ready_sock.recvfrom(65536)
                if not data:
                    continue

                if ready_sock is pred_sock:
                    packet = forward_packet(
                        data, unity_pred_sock, UNITY_PRED_PORT, "predicted"
                    )
                    if packet is not None and first_pred_packet:
                        print("First predicted packet received:")
                        print(f"  Packet index: {packet.get('packet_index')}")
                        print(f"  Timestamp: {packet.get('timestamp_s')}")
                        print(f"  Kind: {packet.get('kind')}")
                        print(f"  First 5 angles: {packet.get('angles_deg', [])[:5]}")
                        print()
                        first_pred_packet = False
                else:
                    packet = forward_packet(
                        data, unity_gt_sock, UNITY_GT_PORT, "ground_truth"
                    )
                    if packet is not None and first_gt_packet:
                        print("First ground-truth packet received:")
                        print(f"  Packet index: {packet.get('packet_index')}")
                        print(f"  Timestamp: {packet.get('timestamp_s')}")
                        print(f"  Kind: {packet.get('kind')}")
                        print(f"  First 5 angles: {packet.get('angles_deg', [])[:5]}")
                        print()
                        first_gt_packet = False

                if packet is not None:
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(
                            f"\rForwarding packets | Count: {frame_count}",
                            end="",
                            flush=True,
                        )

        except json.JSONDecodeError as e:
            print(f"\nJSON decode error: {e}")
            continue
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            continue

except KeyboardInterrupt:
    print("\n\nStopping forwarder...")
    pred_sock.close()
    gt_sock.close()
    unity_pred_sock.close()
    unity_gt_sock.close()
    print("Forwarder stopped cleanly")
