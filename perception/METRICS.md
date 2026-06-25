# 평가 지표 (기술조사 정리)

온디바이스(엣지, NCNN) vs 로컬 AI 서버(YOLO/PyTorch) 객체탐지 비교를 위한
평가 지표 정리. 엣지 AI / YOLO 벤치마크 문헌에서 공통으로 쓰는 지표를 모았고,
각 지표가 이 레포 어디서 측정되는지도 함께 표기한다.

> **핵심 원칙**: 문헌상 *엣지 벤치마크는 단일 추론 지연·전력효율·메모리 제약*을,
> *서버/클라우드는 처리량·확장성*을 강조한다. 우리는 둘을 같은 입력으로 비교하므로
> **(1) 정확도, (2) 속도, (3) 자원·에너지, (4) 엣지↔서버 통신** 네 축을 모두 본다.

---

## 1. 정확도 (Accuracy) — 모델의 성질, 라벨 데이터셋 필요

전송 방식과 무관. `eval/eval_map.py`로 **오프라인** 측정 (서버 안 띄워도 됨).

| 지표 | 정의 | 비고 |
|------|------|------|
| **mAP@0.5** | IoU 0.5 기준 전 클래스 평균 AP | 가장 흔한 1차 지표 |
| **mAP@0.5:0.95** | IoU 0.5~0.95(0.05 간격) 평균 | COCO 표준, 더 엄격 |
| **Precision** | 검출한 것 중 맞은 비율(오탐 적을수록↑) | conf threshold에 민감 |
| **Recall** | 실제 객체 중 찾아낸 비율(미탐 적을수록↑) | conf threshold에 민감 |
| **F1** | Precision·Recall 조화평균 | 단일 운영점 비교용 |
| (옵션) 클래스별 AP | 클래스마다 AP | 약한 클래스 파악 |

→ **우리 비교의 핵심**: 엣지(NCNN 경량/양자화) vs 서버(full) 의 **mAP 손실폭**.
경량화로 속도를 얼마 얻는 대신 정확도를 얼마 잃는가 (정확도-지연 트레이드오프).
측정: `eval/eval_map.py` → `eval/compare_map.py`.

> ⚠️ mAP는 **정답 라벨(GT)** 이 있어야 계산됨. 실시간 카메라만으로는 불가
> → 라벨 테스트셋 준비(`eval/datasets/README.md`). 공개 sanity check는 COCO val2017.

---

## 2. 속도 (Speed / Latency) — 실제 배포대로 측정

| 지표 | 정의 | 어디서 |
|------|------|--------|
| **Inference latency** | 모델 추론 1프레임 시간(ms) | edge/server 공통 |
| **End-to-end latency** | 캡처→(인코딩→네트워크→)추론→결과 전체 | client(서버경로) |
| **FPS (throughput)** | 초당 처리 프레임 (지속, wall-clock) | `metrics.py` |
| **지연 분해** | preprocess / inference / postprocess (+네트워크) | 아래 참고 |
| **Tail latency p95/p99** | 꼬리 지연 — 실시간성엔 평균보다 중요 | `metrics.py` |

- **FPS는 end-to-end 파이프라인**(decode+preprocess+inference+postprocess)으로 재는 게
  표준. 순수 추론시간만 보면 실제 체감보다 과대평가됨.
- Ultralytics 결과의 `result.speed` dict가 preprocess/inference/postprocess(ms)를 분해
  제공 → 지연 분해가 필요하면 여기서 뽑으면 됨.
- 우리 `metrics.py`는 frame별 latency를 기록하고 mean/p50/**p95/p99**/FPS(wall-clock)를 요약.

> 문헌 예: 자율주행 연구에서 cloud 240~310ms vs edge 45~55ms(~5x). 단, 결과는
> 네트워크 대역폭·모델 크기에 크게 좌우됨 → 우리 환경(로컬 GPU 서버 vs Pi)에서 직접 재야 함.

---

## 3. 자원 · 에너지 (Resource / Energy) — 엣지 특화

라즈베리파이 같은 제약 환경에서 가장 중요한 축. `detect_edge.py`가 frame별로 기록.

| 지표 | 정의 | 측정 |
|------|------|------|
| **CPU 사용률** | 추론 중 CPU% | psutil (`metrics.py`) |
| **RAM 사용량** | 메모리 점유(MB) | psutil |
| **SoC 온도** | 칩 온도(℃) | `vcgencmd measure_temp` |
| **Throttling 플래그** | 과열/저전압 스로틀 발생 여부 | `vcgencmd get_throttled` |
| **모델 크기** | 디스크 용량(MB) / params / FLOPs | export 산출물 |
| **전력(W)** | 추론 중 소비전력 | 외부 USB 전력계 필요(옵션) |
| **에너지 효율 (FPS/W)** | 와트당 처리량 | 전력계 있을 때 |

- **온도/스로틀링이 핵심 함정**: Pi는 지속 부하 시 스로틀로 FPS가 떨어짐 → 온도와
  throttle을 같이 기록하지 않으면 FPS 숫자가 거짓말이 됨. (이미 코드에 반영)
- **FPS/W (에너지 효율)** 은 엣지 문헌의 대표 지표지만 Pi는 온보드 전력측정이 없어
  **외부 USB 전력계**가 있어야 정확. 없으면 이 지표는 생략하거나 정성적으로만.
- NCNN num_threads(스레드 수)도 엣지 FPS에 큰 영향 → 고정하고 리포트.

> 문헌: Pi5에서 큰 모델 CPU-only는 초당 프레임이 안 나와 비현실적 → **경량 모델(우리의
> YOLO11n) + NCNN** 조합이 Pi 엣지에선 사실상 정석. (우리 설계가 맞는 방향)

---

## 4. 엣지 ↔ 서버 통신 (Communication) — 서버 경로 전용

서버 방식의 "진짜 비용". `detect_client.py`가 기록.

| 지표 | 정의 |
|------|------|
| **네트워크 왕복 지연** | end-to-end − server_infer_ms (인코딩+전송+왕복) |
| **전송 페이로드 크기** | JPEG 바이트수(`jpeg_bytes`) — 품질/해상도 영향 |
| **프레임 유실률 (UDP)** | 타임아웃/손실 프레임 비율(`lost`) |
| **통신-연산 트레이드오프** | 전송시간 vs 서버추론 절감의 균형 |

- **통신-연산 트레이드오프**가 엣지vs서버 결정의 핵심: 서버가 빨리 추론해도 네트워크
  왕복이 크면 엣지가 이김. JPEG 품질↓ → 전송↓ 지만 정확도↓.
- UDP는 저지연이지만 **유실 가능** → `lost` 카운트가 정확도/신뢰성에 영향. HTTP(TCP)는
  유실 없지만 오버헤드. 두 전송을 같이 재서 비교(`--transport udp|http`).

---

## 권장 측정 세트

**Must (지금 바로 가능, 라벨 불필요)**
- 속도: inference latency, end-to-end latency, FPS, p95/p99
- 자원: CPU%, RAM, 온도, throttling
- 통신: 네트워크 왕복지연, payload 크기, UDP 유실률

**라벨 데이터셋 생기면**
- 정확도: mAP@0.5, mAP@0.5:0.95, precision, recall, F1
- 엣지 vs 서버 mAP 델타 (정확도-지연 트레이드오프 완성)

**장비 있으면 (옵션)**
- 전력(W), 에너지 효율 FPS/W (외부 USB 전력계 필요)

## 공정 비교 원칙
1. **같은 모델·같은 클래스셋**에서 출발 (엣지는 NCNN 변환/양자화 영향까지 포함해 측정).
2. **같은 입력 해상도·imgsz·conf threshold** 사용.
3. NCNN **num_threads**, 서버 **device(GPU/CPU)** 등 런타임 설정을 **고정·기록**.
4. 충분한 프레임 수(예: 300+)로 **지속 성능** 측정 (워밍업 1프레임은 제외 고려).
5. 엣지 측정 시 **온도/스로틀 동시 기록** (안 그러면 FPS가 왜곡됨).

---

## 참고문헌
- [Benchmark Analysis of YOLO Performance on Edge Intelligence Devices (MDPI)](https://www.mdpi.com/2410-387X/6/2/16)
- [Bridging AI and edge computing: comprehensive benchmark of YOLO models (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2542660526000569)
- [Review of YOLOv8/RT-DETR energy efficiency on edge devices (Nature Sci. Reports)](https://www.nature.com/articles/s41598-026-46453-6)
- [Edge AI Chip Benchmark Metrics That Matter (Troy Lendman)](https://troylendman.com/edge-ai-chip-benchmark-metrics-that-matter/)
- [Cloud vs Edge AI Inference: Hybrid Decision Guide (Spheron)](https://www.spheron.network/blog/hybrid-cloud-edge-ai-inference-guide/)
- [Communication-Computation Trade-Off in Resource-Constrained Edge Inference (arXiv 2006.02166)](https://arxiv.org/pdf/2006.02166)
- [Evaluation of Object Detection Models — FLOPs, FPS, Latency, mAP… (Nikita Malviya, Medium)](https://medium.com/@nikitamalviya/evaluation-of-object-detection-models-flops-fps-latency-params-size-memory-storage-map-8dc9c7763cfe)
- [The Complete Guide to Object Detection Evaluation Metrics (Medium)](https://medium.com/@prathameshamrutkar3/the-complete-guide-to-object-detection-evaluation-metrics-from-iou-to-map-and-more-1a23c0ea3c9d)
