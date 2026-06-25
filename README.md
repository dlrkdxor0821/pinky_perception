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

- **모델**: Ultralytics YOLO (예: `YOLO11n`).
  - 엣지: Ultralytics export 로 **NCNN** 포맷(`.param`/`.bin`)으로 변환 후 NCNN 런타임(CPU, aarch64)에서 추론. ARM CPU에 최적화된 경량 추론.
  - 서버: `.pt` 그대로 PyTorch(GPU) 추론
  - 공정 비교를 위해 **동일 가중치**에서 출발(엣지는 경량화/양자화 영향까지 함께 측정).
- **통신**: REST (FastAPI). 로봇이 이미지를 `POST /detect` → 서버가 detection JSON 반환
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

## 빠른 시작 (예정)

```bash
# 0) (어디서든) YOLO 가중치를 NCNN 으로 export
cd perception/scripts && python3 export_ncnn.py    # YOLO11n -> .param/.bin

# 1) (서버 PC) AI 서버 실행
cd perception/server && uvicorn app:app --host 0.0.0.0 --port 8000

# 2) (라즈베리파이) 엣지 NCNN 추론 벤치마크
cd perception/edge && python3 detect_edge.py

# 3) (라즈베리파이) 서버 경로 벤치마크
cd perception/client && python3 detect_client.py --server http://<서버IP>:8000
```
