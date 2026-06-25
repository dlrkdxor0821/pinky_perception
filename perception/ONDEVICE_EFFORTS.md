# 온디바이스(엣지) 구동 노력 정리

라즈베리파이 **Pi 4 (Cortex-A72, 4코어, 64-bit)** 에서 YOLO를 직접 돌리기 위한 작업 기록.
대상: `pinky_pro_and_person` (NCNN, imgsz 320, 클래스 `person`/`mobile_robot`).

---

## ✅ 최종 적용된 설정 (현재 동작 상태)

| 항목 | 적용값 | 비고 |
|------|--------|------|
| 모델 | NCNN `pinky_pro_and_person` (imgsz **320**) | HF `ASD0821/pinky_pro_and_person-ncnn` |
| 추론 런타임 | **raw ncnn** (torch/ultralytics 없음) | 디코딩+NMS 직접 구현 |
| 파이썬 환경 | `perception/.venv` (`--system-site-packages`) + `ncnn`만 | ROS(numpy 1.26.4) 격리 |
| 카메라 | **picamera2(libcamera)** 백엔드, `--source csi` | OpenCV V4L2는 까만 화면이라 |
| 화면 회전 | **`rotate=180`** | 카메라 마운트 보정 (기본값) |
| 임계값 | **conf 0.6** | 과한 오탐 제거 |
| conv 가속 | winograd / sgemm **ON** | ncnn 기본, ARM에 유효 |
| 스레드 | **threads=4** | A72 4코어 최대 |
| 검증 수단 | `--save`(프레임 저장) + `preview_server`(브라우저 MJPEG) | 헤드리스용 |
| **성능 (실측 300프레임)** | **FPS 4.3 / 지연 평균 217ms / p50·p95·p99 = 201·319·387ms** | `benchmark/results/edge.csv` |
| 성능 (참고) | 부하 없을 때 추론 ~180ms (~5.5 FPS) | 순수 온디바이스, 네트워크 X |

**실행 (현재 최종 설정 그대로):**
```bash
# 벤치마크
perception/.venv/bin/python perception/edge/detect_edge.py --source csi --frames 300
# 라이브 프리뷰 (브라우저 http://<Pi-IP>:8080/)
perception/.venv/bin/python perception/edge/preview_server.py --source csi
```

---

## 거쳐온 노력 (한 것들)

- **경량 모델 확보** — HF에서 NCNN 모델 다운로드하였음 (이미 NCNN 포맷, export 불필요).
- **추론을 raw ncnn으로 구현** — torch/ultralytics 없이 `ncnn`+numpy+opencv만으로, YOLO 디코딩·NMS 직접 작성하였음. (torch 426MB 회피, 경량·빠른 시작)
- **의존성 격리** — venv를 `--system-site-packages`로 만들어 시스템 cv2/numpy 재사용, `ncnn`만 추가하였음. (ROS 안 건드림)
- **카메라 문제 해결** — CSI라 OpenCV V4L2는 까만 화면 → picamera2 백엔드 추가하였음.
- **화면 회전 보정** — 90° 틀어진 마운트 → `rotate` 옵션 추가하고 180으로 맞췄음.
- **임계값 조정** — conf 0.6으로 설정하였음.
- **헤드리스 검증 수단** — 박스 프레임 디스크 저장 + 브라우저 MJPEG 라이브 프리뷰(무설치 stdlib) 만들었음.
- **성능 측정** — 320 기준 ~180ms / ~5.5 FPS 실측하였음 (네트워크 X).

---

## 성능 최적화: 적용 / 보류 / 폐기

**✅ 적용 (효과 확인)**
- winograd·sgemm conv (ARM conv 가속)
- threads=4 (코어 최대 활용)
- 회전 보정 (정상 방향 → person 검출 가능)

**⏳ 보류 (후보, 정확도/장비 trade)**
- **imgsz 축소 재export** (320→256/192) — 남은 것 중 효과 가장 큼, 정확도 약간 ↓
- **오버클럭 + 쿨링** (1.8→2.0GHz) — 방열 필요
- **캡처/추론 병렬화** — 체감 FPS↑ (단일 추론 지연은 그대로)

**❌ 폐기 (A72에선 효과 없음)**
- **fp16** — HW(`asimdhp`) 없음 → 실측 **-9% 느림** → 롤백
- **int8 양자화** — dotprod(`asimddp`) 없음 → fp32 대비 이득 없음(↓ 가능) + 변환도구 직접 빌드 부담 → 안 함
- **bf16 / Vulkan GPU / threads>4** — 미지원이거나 오히려 손해

---

## 교훈 (실측)
- int8 도구를 Pi에서 빌드하던 중 프리뷰가 **1 FPS로 급락** → int8 탓이 아니라 **빌드(4코어 풀로드)의 CPU 경쟁** 때문. 빌드 중단 즉시 **4 FPS 복귀**.
  → **무거운 작업과 추론을 동시에 돌리면 FPS 측정이 왜곡됨. 벤치마크는 다른 부하 없이 측정할 것.**
- 양자화 빌드 디렉토리/도구는 정리함(미사용).

## 핵심 결론
지금 ~5.5 FPS는 **A72 CPU 추론 한계**이지 네트워크/UDP 때문이 아님. 더 올리려면 모델 레벨(**imgsz↓**)과 하드웨어(**오버클럭+쿨링**)가 실질 레버. fp16/int8은 이 CPU에선 무효.
