# perception — edge vs. server object detection benchmark

On-device (NCNN on the Pi) vs. local AI server (YOLO/PyTorch over FastAPI+UDP),
compared on the same input. See the repo root `README.md` for the overall plan.

## Layout

```
common/    camera, UDP framing (protocol.py), metrics, viz, yolo_utils
server/    detector (PyTorch), FastAPI app, UDP server
edge/      NCNN detector + on-device benchmark loop
client/    robot-side client: capture -> send (UDP/HTTP) -> metrics
scripts/   export_ncnn.py, run_server.py (FastAPI + UDP, two ports)
benchmark/ run_edge.sh, run_server_path.sh, compare.py, results/
eval/      eval_map.py (offline mAP), compare_map.py, datasets/
tests/     test_smoke.py (no deps), test_loopback.py (dummy detector)
models/    YOLO weights / NCNN model dir (git-ignored)
```

> **어떤 지표를 평가할지**는 [`METRICS.md`](METRICS.md) 참고 (기술조사 + 측정 위치 정리).
> **실제 테스트 절차**(엣지/서버 측정 + 박스 시각화)는 [`WORKFLOW.md`](WORKFLOW.md) 런북 참고.

## Install

```bash
pip install -r server/requirements.txt    # on the server PC (GPU)
pip install -r edge/requirements.txt      # on the Pi (edge path)
pip install -r client/requirements.txt    # on the Pi (server path)
```

## 1. Prepare the model (locally)

```bash
# server keeps the .pt; edge uses an NCNN export
python3 scripts/export_ncnn.py --weights yolo11n.pt --imgsz 640
mv yolo11n_ncnn_model models/
```

## 2. Run the AI server (PC, one process / two ports)

```bash
python3 scripts/run_server.py --weights yolo11n.pt --device cuda
# HTTP :8000  (POST /detect)   |   UDP :9000  (chunked JPEG)
# add --preview to serve a live annotated MJPEG at http://<server-ip>:8000/
#   (quality check only — drawing+encoding adds load; leave OFF while benchmarking)
```

## 3. Benchmark

```bash
# edge (on the Pi, no network)
./benchmark/run_edge.sh --model models/yolo11n_ncnn_model --frames 300

# server path (on the Pi)
./benchmark/run_server_path.sh --transport udp  --host <server-ip> --frames 300
./benchmark/run_server_path.sh --transport http --host <server-ip> --frames 300

# compare
python3 benchmark/compare.py benchmark/results/*.csv
```

### Record an annotated video of the server path

`detect_client.py --record` draws the server's detection boxes (+ live FPS
overlay) onto each frame and saves an `.mp4`. Playback FPS is auto-estimated
from the run (`--record-fps N` to force it). CSI cam on this Pi needs `--rotate 180`.

```bash
python3 client/detect_client.py --transport udp --host <server-ip> \
    --source csi --rotate 180 --frames 300 --record benchmark/results/server_udp.mp4
```

## Metrics

| now (no labels needed) | later (needs labeled test set) |
|------------------------|--------------------------------|
| FPS (sustained, wall-clock) | mAP@0.5, mAP@0.5:0.95 |
| latency mean / p50 / p95 / p99 | precision / recall / F1 |
| 인식률 proxy: avg detections, avg confidence | edge(NCNN) vs server(full) accuracy delta |
| Pi CPU / RAM / temp / throttle | |
| UDP lost-frame count | |

### Accuracy (mAP) — when a labeled test set is ready

mAP is a property of the model, not the transport, so evaluate each model
offline (no server needed). See `eval/datasets/README.md` for the dataset layout.

```bash
python3 eval/eval_map.py --model models/yolo11n_ncnn_model --data eval/datasets/test/data.yaml --label edge
python3 eval/eval_map.py --model yolo11n.pt                 --data eval/datasets/test/data.yaml --label server
python3 eval/compare_map.py benchmark/results/map_edge.json benchmark/results/map_server.json
```

## Verify (no model/camera needed)

```bash
python3 tests/test_smoke.py       # protocol + metrics math
python3 tests/test_loopback.py    # full UDP/HTTP path with a dummy detector
```
