# 테스트 워크플로우 (런북)

온디바이스(엣지) vs AI 서버, **두 경로를 같은 카메라 입력으로** 테스트한다.
각 경로마다 할 수 있는 게 두 가지다:

1. **성능 수치 측정** — FPS, 지연(latency), Pi 부하 등을 CSV로 기록
2. **박스 눈으로 확인 (시각화)** — 브라우저로 실시간 영상 + 바운딩박스

> ⚠️ **수치 측정과 시각화는 따로 돌린다.** 시각화(스트리밍/녹화)는 그리기·인코딩
> 부하 때문에 FPS를 떨어뜨려서, 켜놓고 잰 수치는 실제보다 낮게 나온다.

| | 추론·박스 그리기 | 보는 주소 | 영상 출처 |
|---|---|---|---|
| **엣지** | 라즈베리파이 | `http://<Pi-IP>:8080/` | Pi 카메라 |
| **서버** | 서버 PC | `http://<서버-IP>:8000/` | Pi 카메라(전송됨) |

→ 서버 경로도 **카메라는 Pi에 있다.** Pi가 프레임을 서버로 쏘고, 서버가 박스를 쳐서
서버 주소로 보여줄 뿐이다.

---

## 0. 환경 (머신 2대, 같은 LAN)

| 머신 | 역할 | 파이썬 |
|---|---|---|
| 라즈베리파이 | 카메라 + 엣지 추론 + 서버경로 클라이언트 | `perception/.venv/bin/python` |
| 서버 PC (GPU) | AI 서버(추론) | 서버쪽 venv — `server/SETUP.md` 참고 |

- **Pi에서는 반드시 venv 파이썬을 쓴다.** `ncnn`/`opencv`가 `perception/.venv`에만
  있어서, 시스템 `python3`로 돌리면 `ModuleNotFoundError`가 난다.
  - 활성화하면 명령이 짧아짐: `cd perception && source .venv/bin/activate`
  - 활성화 안 하면 풀패스로: `perception/.venv/bin/python ...`
  - `./benchmark/run_*.sh` 래퍼는 `.venv`가 있으면 자동으로 그걸 쓴다.
- 서버 IP 확인: 서버 PC에서 `hostname -I`.
- **로봇(Pi) IP를 모르거나 ssh가 안 되면** → 아래 [부록: 로봇 IP 찾기](#부록-로봇-ip-가-안-잡힐-때).

아래 예시는 모두 `cd perception` 한 상태 + venv 활성화 기준이다.

---

## A. 온디바이스 (엣지) — Pi에서

### A-1. 성능 측정 (수치)
```bash
./benchmark/run_edge.sh --source csi --imgsz 320 --frames 300
# 동일: .venv/bin/python edge/detect_edge.py --source csi --imgsz 320 --frames 300
```
- 출력: 요약(표준출력) + `benchmark/results/edge.csv` + 스로틀 플래그.
- 기록 항목: FPS, 지연 mean/p50/p95/p99, 검출 수/신뢰도, **Pi CPU/RAM/온도**.

### A-2. 박스 눈으로 보기 (로컬 MJPEG 스트리밍)
```bash
.venv/bin/python edge/preview_server.py --source csi --imgsz 320 --port 8080
```
PC 브라우저에서 **`http://<Pi-IP>:8080/`** → 카메라 + 박스 + FPS 실시간.
(Pi가 캡처·추론·박스·스트리밍을 전부 한다. 종료는 `Ctrl-C`.)

---

## B. AI 서버 — 서버 PC + Pi

### B-1. 서버 띄우기 (서버 PC)
```bash
cd perception && source .venv/bin/activate
python3 scripts/run_server.py --weights <full_model>.pt --device cuda            # 측정만
python3 scripts/run_server.py --weights <full_model>.pt --device cuda --preview  # 시각화도 켤 때
```
- 방화벽에서 **8000(TCP) / 9000(UDP)** 열기.
- 셋업 전체는 `server/SETUP.md`. 동작 확인: `curl http://<서버IP>:8000/health`.

### B-2. 성능 측정 (Pi에서)
```bash
./benchmark/run_server_path.sh --transport udp  --host <서버IP> --source csi --rotate 180 --frames 300
./benchmark/run_server_path.sh --transport http --host <서버IP> --source csi --rotate 180 --frames 300
```
- 출력: `benchmark/results/server_udp.csv`, `server_http.csv`.
- 기록 항목: FPS, **end-to-end 지연**, `server_infer_ms`(서버 순수 추론),
  프레임 유실률(UDP), **Pi CPU/RAM/온도**.
  - 네트워크 왕복 지연 = `e2e − server_infer_ms`.

### B-3. 박스 눈으로 보기 (둘 중 택1)
- **방법 1 — 서버에서 라이브** (B-1을 `--preview`로 띄웠을 때):
  브라우저 **`http://<서버IP>:8000/`** → Pi가 보낸 영상 + 서버가 친 박스.
- **방법 2 — Pi에서 녹화(mp4)**:
  ```bash
  .venv/bin/python client/detect_client.py --transport udp --host <서버IP> \
      --source csi --rotate 180 --frames 300 --record benchmark/results/server_udp.mp4
  ```

---

## C. 비교
```bash
python3 benchmark/compare.py benchmark/results/edge.csv \
                             benchmark/results/server_udp.csv \
                             benchmark/results/server_http.csv
```
나온 수치를 `성능비교.md` 표에 옮긴다.

---

## 공정한 비교를 위한 규칙

- **한 세션에서 엣지 → 서버(UDP) → 서버(HTTP)를 연달아** 측정한다(같은 장면·전원·발열).
  안 그러면 특히 **Pi 부하/스로틀** 비교가 사과-오렌지가 된다.
- **측정 중엔 시각화(프리뷰/녹화)를 끈다.** 눈으로 확인할 때만 켠다.
- `imgsz`, `conf`, `threads`(NCNN)를 **고정하고 결과에 같이 적는다.** FPS를 크게 바꾼다.
- 엣지 모델은 `imgsz 320` 기준이다(`--imgsz 320` 빠뜨리지 말 것).
- CSI 카메라는 `--source csi`, 이 Pi에서는 `--rotate 180`.

---

## 부록: 로봇 IP 가 안 잡힐 때

`ssh pinky@<IP>`가 안 되면 대부분 **DHCP로 IP가 바뀐 것**이다(전압·전원과 무관).

```bash
ping -c2 <IP>                       # 응답은 오는데
nc -zv <IP> 22                      # 22번이 "Connection refused" 면 → 그 IP는 다른 기기
```

- 같은 망에서 SSH(22) 열린 호스트를 훑어 로봇을 찾는다(로봇은 Raspberry Pi —
  MAC OUI가 `e4:5f:01`/`2c:cf:67`/`b8:27:eb`/`dc:a6:32` 계열).
- 가장 확실: 로봇 콘솔/화면에서 직접 `hostname -I`.
- 옛 호스트키 충돌(`REMOTE HOST IDENTIFICATION HAS CHANGED`)나면:
  `ssh-keygen -R <IP>` 후 다시 접속.
- IP가 자주 바뀌면 `~/.ssh/config`에 별칭을 만들어 두면 편하다.
