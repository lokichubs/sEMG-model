# Visualization Run Guide

This folder contains the playback-to-Unity pipeline for sleeve EMG angle prediction.

## What this pipeline does
There are two main Python scripts involved:

1. `sEMG_inference.py`
   - loads a trained model from `outputs_best_v8/`
   - loads a raw `.xdf` recording from `../data/`
   - runs the model window-by-window on EMG
   - sends:
     - **predicted angles** over UDP on port `5020`
     - **ground-truth angles** over UDP on port `5025`

2. `handtrack_data_handler.py`
   - listens to those two UDP streams
   - converts each 14-angle vector into the existing Unity 78-value hand format
   - forwards:
     - **predicted hand** to Unity on port `5016`
     - **ground-truth hand** to Unity on port `5017`

Unity then listens on those two ports using the two hand listener scripts.

---

## Unity port mapping
### Python internal ports
- Predicted angles from inference: `5020`
- Ground-truth angles from inference: `5025`

### Unity listener ports
- Predicted hand in Unity: `5016`
- Ground-truth hand in Unity: `5017`

---

## Unity scripts
Use these listener scripts in Unity:

- `Right_Hand_Listener_sEMG.cs`
  - listens on `5016`
  - drives the predicted hand

- `Right_Hand_Listener_GT.cs`
  - listens on `5017`
  - drives the ground-truth hand

Each should be attached to the corresponding hand rig.

---

## Typical run order
### 1. Start Unity
Open the Unity scene containing:
- predicted hand object
- ground-truth hand object
- both UDP listener components attached

Color convention in Unity:
- **green hand** = predicted hand
- **red hand** = ground-truth hand

### 2. Start the data handler
From this `visualization/` folder, run:

`python handtrack_data_handler.py`

This must be running before or alongside inference so it can receive the packets.

### 3. Start inference playback
Also from this folder, run one of the following.

#### Specific subject and session
`python sEMG_inference.py --subject 3 --session 2`

#### Subject only
`python sEMG_inference.py --subject 4`

If only `--subject` is provided and multiple sessions exist, the script will randomly choose one matching session and print which file was selected.

#### Direct file path
`python sEMG_inference.py --xdf-file ../data/sub-004_ses-003_task-emg2angles_run-001_meg.xdf`

---

## Real-time vs fast offline playback
### Fast offline playback (default)
If you do **not** pass `--realtime`, playback runs as fast as the machine can process the data.

Example:
`python sEMG_inference.py --subject 3 --session 2`

### Real-time playback
If you want it to replay at the original recording speed, use:

`python sEMG_inference.py --subject 3 --session 2 --realtime`

This should make a ~10 minute recording take ~10 minutes to replay.

### Faster/slower real-time playback
You can also scale playback speed:

- `--speed 2.0` → 2x realtime
- `--speed 0.5` → half realtime

Example:
`python sEMG_inference.py --subject 3 --session 2 --realtime --speed 2.0`

---

## Useful options
### Dry run without sending UDP
`python sEMG_inference.py --subject 4 --session 3 --dry-run`

### Limit number of packets
`python sEMG_inference.py --subject 4 --session 3 --max-packets 100`

### Change playback stride
`python sEMG_inference.py --subject 4 --session 3 --step-len 50`

### Change packet format
Supported formats:
- `json` (default)
- `csv`

Example:
`python sEMG_inference.py --subject 4 --session 3 --packet-format csv`

---

## Example recommended workflow
### Fast check
1. `python handtrack_data_handler.py`
2. `python sEMG_inference.py --subject 3 --session 2`

### Real-time demo
1. `python handtrack_data_handler.py`
2. `python sEMG_inference.py --subject 3 --session 2 --realtime`

---

## Notes
- `sEMG_inference.py` loads the model from `../outputs_best_v8/` by default.
- It loads raw `.xdf` files from `../data/` by default.
- The script applies the same EMG bandpass used in preprocessing.
- Ground-truth angles are interpolated to the EMG timeline so predicted and ground-truth packets share the same timestamps.

---

## Troubleshooting
### Nothing moves in Unity
Check that:
- Unity is running
- the correct listener scripts are attached
- the listener ports are set correctly:
  - predicted = `5016`
  - ground truth = `5017`
- `handtrack_data_handler.py` is running
- `sEMG_inference.py` is running

### Subject-only command errors or picks a session you did not expect
- Use `--session` to force a specific recording
- otherwise subject-only mode may randomly select one session for that subject

### Playback is too fast
Use:
`--realtime`

### Playback is still too slow or too fast in demo mode
Use:
`--realtime --speed X`

---

## Current pipeline summary
`raw XDF -> sEMG_inference.py -> UDP 5020/5025 -> handtrack_data_handler.py -> Unity 5016/5017`
