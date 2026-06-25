# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

`pinky_perception` benchmarks **on-device (edge) object detection** against a
**local AI server** for the Pinky robot. The same YOLO model is used on both
paths; we measure and compare latency, FPS, resource usage, and accuracy.

- **Edge path**: Pinky (Raspberry Pi, aarch64) runs a lightweight YOLO via **NCNN** on CPU.
- **Server path**: a local PC (GPU) serves YOLO (PyTorch) over **FastAPI REST**; the robot
  POSTs camera frames and receives detection JSON.
- **Input**: USB/CSI camera, real-time frames.

## Environment

- ROS 2 **Jazzy**, Python **3.12**, **aarch64** (Raspberry Pi).
- Edge runtime: **NCNN** (Ultralytics export → `.param`/`.bin`).
- Server runtime: PyTorch + Ultralytics, served with FastAPI/uvicorn.
- Model: Ultralytics YOLO (default `YOLO11n`). Keep the **same base weights** on
  both paths so the comparison is fair; the edge side additionally absorbs the
  NCNN conversion/quantization effect, which is itself part of what we measure.

## Repository layout

```
pinky_pro/                 # ROS 2 workspace (robot base)
  src/
    pinky_pro/             # forked upstream robot packages — GIT-IGNORED
    sllidar_ros2/          # RPLIDAR C1 driver, pinky-specific edits (tracked)
perception/                # ⭐ this project's work (scaffold stage)
  edge/                    # on-device NCNN inference + metrics (runs on the Pi)
  server/                  # FastAPI detection server (runs on local PC/GPU)
  client/                  # robot-side client: capture → POST → metrics
  common/                  # camera, metrics logging, visualization
  benchmark/               # benchmark scripts + results (CSV/plots)
  scripts/                 # utilities, e.g. export_ncnn.py
  models/                  # weights (.pt / .param / .bin) — GIT-IGNORED
```

## Runtime setup (as deployed on this Pi)

- **venv**: `perception/.venv` (created with `--system-site-packages` to reuse the
  system OpenCV/numpy that ROS uses). Run edge/preview scripts with
  `perception/.venv/bin/python`. System numpy is 1.26.4 — do NOT let an install
  upgrade it (would break ROS cv_bridge).
- **Edge inference = raw NCNN, NOT Ultralytics** (`edge/detector_ncnn_raw.py`).
  Only `ncnn` + numpy + opencv installed — **no torch / no ultralytics** (torch is
  426MB and unused for NCNN compute; we decode YOLO output + NMS by hand). The
  Ultralytics-based `detector_ncnn.py` is kept for reference but needs torch.
- **Camera is CSI → must use picamera2/libcamera** (`--source csi`). OpenCV V4L2
  (`/dev/video0`) returns all-black frames for it. `common/camera.py` has both
  backends; picamera2 "RGB888" arrays are already BGR.
- **Edge model**: `models/pinky_pro_and_person_ncnn_model/` from HF
  `ASD0821/pinky_pro_and_person-ncnn`. Classes `0=person, 1=mobile_robot`,
  **imgsz 320**. Measured ~3 FPS / ~313 ms on this Pi.
- **Server model is different (full PyTorch)** — intentional asymmetry: edge-light
  vs server-full. Interpret mAP with that in mind.
- **UDP protocol requirement (mandatory)**: `common/protocol.py` does per-datagram
  CRC32 checksum + drops stale/older incomplete frames (always reassembles the
  newest frame). Keep these if editing the protocol.
- **Headless verify**: `detect_edge.py --save N` writes annotated frames to
  `benchmark/results/frames/`; `edge/preview_server.py` streams live MJPEG to a
  browser at `http://<pi-ip>:8080/` (stdlib http.server, no Flask).

## Important conventions & gotchas

- **`pinky_pro/src/pinky_pro/` is an upstream fork and is git-ignored.** Do not
  commit it or assume edits there are tracked. It builds the robot base (bringup,
  description, navigation, led, lamp_control, etc.).
- **`sllidar_ros2` is third-party but tracked**, because it carries pinky-specific
  edits in `launch/sllidar_c1_launch.py`: `serial_port=/dev/ttyAMA0`,
  `frame_id=rplidar_link`. Preserve these if updating the driver.
- **Model weights and benchmark results are git-ignored** (large/regenerable).
- Keep the `perception/` tree the source of new work; the ROS workspace is the
  robot base, not where detection-benchmark code should live.

## Common commands

ROS 2 workspace build (on the Pi):
```bash
cd pinky_pro
colcon build --symlink-install
source install/setup.bash
```

Perception (planned entry points):
```bash
# export YOLO weights to NCNN
cd perception/scripts && python3 export_ncnn.py

# AI server (local PC/GPU)
cd perception/server && uvicorn app:app --host 0.0.0.0 --port 8000

# edge NCNN benchmark (Pi)
cd perception/edge && python3 detect_edge.py

# server-path benchmark (Pi)
cd perception/client && python3 detect_client.py --server http://<server-ip>:8000
```

## Benchmark methodology

Compare the two paths on identical input. Always log per-frame, then aggregate.

- **Latency**
  - Edge: preprocess + NCNN inference + postprocess.
  - Server: capture + JPEG encode + network up + server inference + network down + decode.
- **Throughput**: sustained FPS (not a single warm frame).
- **Resources (Pi)**: CPU%, RAM, and **temperature / throttling**
  (`vcgencmd measure_temp`, `vcgencmd get_throttled`). The Pi throttles under
  sustained load — record it or FPS numbers are misleading.
- **NCNN knobs**: `num_threads` (and CPU power state) materially change edge FPS;
  fix and report them per run.
- **Accuracy**: detection parity between edge (light) and server (full) models;
  mAP delta vs. latency cost trade-off.

## Working norms

- Don't commit or push unless the user asks. The user manages this repo (`leekt`).
- When adding perception code, update `README.md` (structure + quick start) so the
  scaffold notes stay accurate.
