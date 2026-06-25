# AI 서버 셋업 (서버 PC, GPU)

Pi가 카메라 프레임을 보내면 서버가 추론해 결과를 돌려준다. 한 프로세스가
**HTTP(:8000) + UDP(:9000)** 두 포트를 동시에 연다.

## 1. 코드 가져오기
이 레포(또는 최소한 `perception/common/`, `perception/server/`,
`perception/scripts/run_server.py`)를 서버 PC로 복사/clone.

## 2. 의존성 설치
```bash
cd perception
python3 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt   # ultralytics, torch, fastapi, uvicorn, opencv, numpy
```
> GPU면 CUDA용 torch 설치 권장 (https://pytorch.org 의 환경별 명령). CPU만 있어도 동작은 함.

## 3. 모델(.pt) 준비
서버는 **full PyTorch 모델**을 쓴다. 같은 클래스(`person`,`mobile_robot`) 비교가 목적이면
**엣지 NCNN을 만든 그 학습 결과 `best.pt`** 를 쓰는 게 정확하다.
(임시 확인용으론 `yolo11n.pt`도 가능하나 클래스가 COCO라 mobile_robot은 안 잡힘.)

## 4. 서버 실행
```bash
python3 scripts/run_server.py --weights <your_full_model>.pt --device cuda
# HTTP :8000  (POST /detect)   |   UDP :9000  (chunked JPEG)
```
- 방화벽에서 **8000(TCP), 9000(UDP)** 열기.
- 서버 IP 확인: `hostname -I` (예: 192.168.0.x). Pi와 같은 LAN이어야 함.

## 5. (Pi에서) 측정
```bash
perception/.venv/bin/python perception/client/detect_client.py --transport udp  --host <서버IP> --frames 300
perception/.venv/bin/python perception/client/detect_client.py --transport http --host <서버IP> --frames 300
# 결과: benchmark/results/server_udp.csv, server_http.csv
```

## 6. 동작 확인 (선택)
```bash
curl http://<서버IP>:8000/health        # {"status":"ok"}
```
