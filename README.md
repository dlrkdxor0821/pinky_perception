# pinky_perception

Pinky 로봇에서 **온디바이스(엣지) 객체탐지**와 **로컬 AI 서버 객체탐지**의 성능을
같은 조건에서 비교하기 위한 레포.

동일한 YOLO 모델을 기반으로, "라즈베리파이 위에서 직접 경량 추론(NCNN)" vs
"프레임을 로컬 AI 서버로 보내 추론" 두 방식의 **지연시간 / FPS / 자원사용량 / 정확도**를
정량 비교하는 것이 목표다.

## 목표

- 🤖 **엣지(온디바이스)**: Pinky(라즈베리파이, aarch64)에서 **NCNN 경량화 모델**로 직접 YOLO 추론
- 🖥️ **AI 서버**: 로컬 PC(GPU)에서 FastAPI로 추론 서비스, 로봇이 프레임을 보내고 결과를 받음
- 📊 **비교**: 같은 입력으로 두 경로의 성능을 측정·기록·비교

## 시스템 구성

```
              ┌──────────────────────────── Pinky (Raspberry Pi, aarch64) ────────────────────────────┐
  USB/CSI     │                                                                                        │
  카메라  ───▶│  camera capture ──┬─▶ [엣지 경로]  NCNN 경량 YOLO 추론 ──▶ 박스/메트릭                │
              │                   │                                                                    │
              │                   └─▶ [서버 경로]  프레임 JPEG 인코딩 ──HTTP POST──┐                  │
              └───────────────────────────────────────────────────────────────────┼──────────────────┘
                                                                                    │  (로컬 네트워크)
                                                              ┌─────────────────────▼─────────────────┐
                                                              │   AI 서버 (로컬 PC / GPU)              │
                                                              │   FastAPI  POST /detect                │
                                                              │   YOLO(PyTorch) 추론 ──▶ detection JSON│
                                                              └────────────────────────────────────────┘
```

- **모델 (비대칭 — 의도된 비교)**: 클래스 `0=person`, `1=mobile_robot`
  - 엣지: **NCNN 경량 모델** (`pinky_pro_and_person`, imgsz **320**) — HuggingFace
    [`ASD0821/pinky_pro_and_person-ncnn`](https://huggingface.co/ASD0821/pinky_pro_and_person-ncnn).
    Pi CPU(aarch64)에서 NCNN 런타임으로 추론.
  - 서버: **다른(full) 모델** — 로컬 AI 서버(GPU)에서 PyTorch로 추론. 엣지처럼
    경량화하지 않은 더 무거운/정확한 모델을 사용.
  - ⚠️ 두 경로의 모델이 다르므로, 이건 "같은 모델의 런타임 차이"가 아니라
    **"엣지에 맞춘 경량모델 vs 서버의 full 모델"이라는 실제 배포 시나리오 비교**다.
    mAP를 볼 땐 이 비대칭을 전제로 해석한다(같은 라벨 테스트셋 기준).
- **통신**: 한 서버 프로세스가 두 포트 — **UDP**(이미지 스트리밍, 저지연) + **HTTP/FastAPI**
  (`POST /detect`). 둘 다 측정해 전송 방식 차이도 비교.
- **입력**: USB/CSI 카메라 실시간 프레임

## 디렉토리 구조

```
pinky_perception/
├── pinky_pro/                 # ROS 2 (Jazzy) 워크스페이스 — 로봇 베이스
│   └── src/
│       ├── pinky_pro/         # 포크된 로봇 패키지 (git 미추적, 업스트림 fork)
│       └── sllidar_ros2/      # RPLIDAR C1 드라이버 (pinky 전용 수정 포함)
├── perception/                # ⭐ 이 프로젝트 작업 영역
│   ├── edge/                  # 온디바이스 NCNN 추론 (라즈베리파이에서 실행)
│   ├── server/                # AI 서버 (FastAPI, 로컬 PC/GPU 에서 실행)
│   ├── client/                # 로봇측 클라이언트 (프레임 캡처→서버 전송→메트릭)
│   ├── common/                # 공용: 카메라, 메트릭 로깅, 시각화
│   ├── benchmark/             # 벤치마크 스크립트 + 결과(CSV/그래프)
│   ├── scripts/               # 모델 export 등 유틸 (export_ncnn.py)
│   └── models/                # 모델 가중치 (.pt / .param / .bin) — git 미추적
├── README.md
└── CLAUDE.md                  # Claude Code 작업 가이드
```

> 위 `perception/` 하위는 아직 스캐폴드 단계다. 구현이 추가되면 이 README를 갱신한다.

## 비교 지표 (벤치마크)

| 구분 | 지표 |
|------|------|
| 지연시간 | 엣지: 전처리+NCNN추론+후처리 / 서버: 캡처+인코딩+네트워크 왕복+서버추론+디코딩 |
| 처리량 | 지속 FPS |
| 자원 | 라즈베리파이 CPU%, RAM, **온도/스로틀링**(`vcgencmd measure_temp`) |
| 정확도 | 경량(NCNN) vs 서버(full) detection 일치도 / mAP 차이 vs 지연 비용 트레이드오프 |

> ⚠️ 라즈베리파이는 부하 지속 시 **열 스로틀링**으로 FPS가 떨어질 수 있어, 엣지 벤치마크는
> 온도/throttle 상태를 함께 기록하는 게 중요하다. NCNN 스레드 수(`num_threads`)도 결과에 영향을 준다.

## ROS 2 워크스페이스 (로봇 베이스)

- 환경: **ROS 2 Jazzy / Python 3.12 / aarch64**
- 빌드:
  ```bash
  cd pinky_pro
  colcon build --symlink-install
  source install/setup.bash
  ```
- 라이다 드라이버(`sllidar_ros2`)는 C1 모델 / `/dev/ttyAMA0` / `frame_id=rplidar_link` 로 수정돼 있음.

## 진행 상황 (Status)

- [x] 레포 구조 / `perception/` 스캐폴드 + 동작 코드
- [x] 검증: 전체 컴파일, 스모크 테스트, **UDP end-to-end 루프백**
- [x] 평가 지표 조사 정리 ([`perception/METRICS.md`](perception/METRICS.md))
- [x] 엣지 NCNN 모델 다운로드 (`perception/models/pinky_pro_and_person_ncnn_model/`, person/mobile_robot, imgsz 320)
- [x] 카메라: CSI라 **picamera2(libcamera)** 백엔드 필요 (OpenCV V4L2는 까만 화면). `--source csi`
- [x] venv(`perception/.venv`)에 **ncnn 설치** (torch 불필요 — raw ncnn 경로)
- [x] **엣지 추론 = raw ncnn** (torch/ultralytics 없이 `detector_ncnn_raw.py`, 디코딩+NMS 직접)
- [x] 엣지 스모크런 동작 확인 (CSI, 320, ~3 FPS / 지연 ~313ms / throttle 0x0)
- [x] UDP 프로토콜 **체크섬(CRC32) + 이전 프레임 폐기** 구현·검증
- [x] **라이브 프리뷰** (`edge/preview_server.py`, 브라우저 MJPEG, 박스+FPS 오버레이)
- [ ] 카메라를 person/mobile_robot으로 향해 **탐지 정확도 눈으로 확인** ← **현재 여기**
- [ ] 서버 PC: full 모델로 `run_server.py` 기동
- [ ] 서버 경로 벤치마크 (UDP/HTTP)
- [ ] `compare.py`로 엣지 vs 서버 비교표
- [ ] (라벨셋 준비되면) mAP 측정 + `eval/compare_map.py`

> 진행하면서 이 체크리스트를 갱신하고 git으로 커밋해 진행상황을 공유한다.

## 워크플로우 (단계별)

### 0. 사전 (Pi, 1회) — 이미 완료됨
```bash
sudo apt install -y python3.12-venv          # venv용 (sudo 필요)
python3 -m venv --system-site-packages perception/.venv
perception/.venv/bin/pip install ncnn        # raw ncnn 경로 (torch 불필요)
# 클라이언트(서버경로)도 쓸 거면: perception/.venv/bin/pip install requests
```

### 1. 온디바이스(엣지) perception — Pi 단독, 네트워크 X
> CSI 카메라라 **`--source csi`** (picamera2). USB캠이면 `--source 0`.
```bash
perception/.venv/bin/python perception/edge/detect_edge.py \
  --model perception/models/pinky_pro_and_person_ncnn_model \
  --imgsz 320 --source csi --frames 300
# 결과: benchmark/results/edge.csv  (FPS / 지연 / CPU / 온도)

# 박스 라이브로 보기 (브라우저): 아래 띄우고 PC에서 http://<Pi-IP>:8080/ 접속
perception/.venv/bin/python perception/edge/preview_server.py --source csi --imgsz 320 --port 8080

# 헤드리스 검증: 박스 그린 프레임을 파일로 저장 (--save N: N프레임마다)
perception/.venv/bin/python perception/edge/detect_edge.py --source csi --imgsz 320 \
  --frames 30 --save 5   # -> benchmark/results/frames/*.jpg
```

### 2. AI 서버 경로 — Pi가 카메라 프레임을 로컬 서버로 전송
```bash
# (2-1) 서버 PC(GPU)에서 full 모델로 서버 기동 — UDP + HTTP 동시
python3 perception/scripts/run_server.py --weights <서버용_full_model>.pt --device cuda

# (2-2) Pi에서 프레임 전송 + 측정 (서버 IP 지정)
python3 perception/client/detect_client.py --transport udp  --host <서버IP> --frames 300
python3 perception/client/detect_client.py --transport http --host <서버IP> --frames 300
# 결과: benchmark/results/server_udp.csv, server_http.csv
```

### 3. 비교 — 뭐가 더 좋은지
```bash
python3 perception/benchmark/compare.py perception/benchmark/results/*.csv
```
→ 엣지(빠르지만 경량/온도 제약) vs 서버(정확하지만 네트워크 지연)의 트레이드오프 확인.

### 4. (나중) 정확도(mAP) — 라벨 테스트셋 생기면
```bash
python3 perception/eval/eval_map.py --model perception/models/pinky_pro_and_person_ncnn_model \
  --data perception/eval/datasets/test/data.yaml --imgsz 320 --label edge
python3 perception/eval/eval_map.py --model <서버용_full_model>.pt \
  --data perception/eval/datasets/test/data.yaml --label server
python3 perception/eval/compare_map.py perception/benchmark/results/map_*.json
```
