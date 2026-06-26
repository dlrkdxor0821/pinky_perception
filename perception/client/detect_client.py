#!/usr/bin/env python3
"""Server-path benchmark (runs on the Pi).

Capture a frame, send it to the AI server, get detections back, and measure
end-to-end latency (encode + network round-trip + server inference). Supports
two transports so they can be compared:
  --transport udp   raw UDP datagrams (low overhead, lossy)   [default]
  --transport http  HTTP POST to FastAPI (reliable, TCP)
"""
import argparse
import json
import pathlib
import socket
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def send_udp(sock, server_addr, frame_id, jpeg, timeout=2.0):
    from common.protocol import encode_frame, decode_result
    for dg in encode_frame(frame_id, jpeg):
        sock.sendto(dg, server_addr)
    sock.settimeout(timeout)
    try:
        while True:
            data, _ = sock.recvfrom(65535)
            fid, payload = decode_result(data)
            if fid == (frame_id & 0xFFFFFFFF):
                return json.loads(payload.decode())
    except socket.timeout:
        return None


def _open_writer(path, frame, fps):
    """Lazily create an mp4 VideoWriter sized to the first annotated frame."""
    import os
    import cv2
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(path, fourcc, max(1.0, float(fps)), (w, h))
    if not wr.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {path!r}")
    return wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["udp", "http"], default="udp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=9000)
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=None)
    ap.add_argument("--record", default=None,
                    help="save annotated detection video to this .mp4 path")
    ap.add_argument("--record-fps", type=float, default=0.0, dest="record_fps",
                    help="playback FPS for --record (0 = auto-estimate from the run)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate frames upright (CSI cam on this Pi needs 180)")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--vflip", action="store_true")
    args = ap.parse_args()

    import cv2
    from common.camera import Camera
    from common.metrics import (Metrics, cpu_percent, pi_temp_c, pi_throttled,
                                ram_used_mb)

    src = int(args.source) if args.source.isdigit() else args.source
    metrics = Metrics(label=f"server-{args.transport}")
    out = args.out or f"benchmark/results/server_{args.transport}.csv"

    sock = None
    url = None
    requests = None
    if args.transport == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_addr = (args.host, args.udp_port)
    else:
        import requests as _requests
        requests = _requests
        url = f"http://{args.host}:{args.http_port}/detect"

    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    print(f"[client] transport={args.transport} -> {args.host} frames={args.frames}")

    record = args.record
    writer = None
    buffered = []        # holds annotated frames until fps is known (auto mode)
    rec_t0 = None
    t_prev = None
    WARMUP = 20          # frames used to estimate fps when --record-fps not given
    if record:
        from common.viz import draw_detections

    with Camera(src, width=args.width, height=args.height,
                rotate=args.rotate, hflip=args.hflip, vflip=args.vflip) as cam:
        cpu_percent()  # prime psutil's delta counter
        for i in range(args.frames):
            frame = cam.read()
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if not ok:
                continue
            jpeg = buf.tobytes()
            t0 = time.perf_counter()
            if args.transport == "udp":
                resp = send_udp(sock, server_addr, i, jpeg)
            else:
                r = requests.post(
                    url, data=jpeg,
                    headers={"Content-Type": "application/octet-stream"},
                )
                resp = r.json() if r.ok else None
            latency_ms = (time.perf_counter() - t0) * 1000  # end-to-end incl. net
            dets = (resp or {}).get("detections", [])
            confs = [d["conf"] for d in dets]
            metrics.add(
                frame=i, t_wall=time.perf_counter(),
                latency_ms=latency_ms if resp is not None else None,
                lost=(resp is None),
                server_infer_ms=(resp or {}).get("server_infer_ms"),
                n_det=len(dets),
                avg_conf=(sum(confs) / len(confs) if confs else None),
                jpeg_bytes=len(jpeg),
                cpu=cpu_percent(), ram_mb=ram_used_mb(), temp_c=pi_temp_c(),
            )

            if record:
                draw_detections(frame, dets)
                now = time.perf_counter()
                inst_fps = 1.0 / (now - t_prev) if t_prev and now > t_prev else 0.0
                t_prev = now
                cv2.putText(
                    frame, f"{inst_fps:4.1f} FPS  {len(dets)} det  {args.transport}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                if writer is not None:
                    writer.write(frame)
                elif args.record_fps > 0:
                    writer = _open_writer(record, frame, args.record_fps)
                    writer.write(frame)
                else:  # auto: buffer a few frames, estimate fps, then flush
                    if rec_t0 is None:
                        rec_t0 = now
                    buffered.append(frame.copy())
                    if len(buffered) >= WARMUP:
                        est = (len(buffered) - 1) / max(1e-6, now - rec_t0)
                        writer = _open_writer(record, buffered[0], est)
                        for f in buffered:
                            writer.write(f)
                        buffered = []

    if record and writer is None and buffered:  # ran fewer frames than WARMUP
        est = ((len(buffered) - 1) / max(1e-6, time.perf_counter() - rec_t0)
               if len(buffered) > 1 else 20.0)
        writer = _open_writer(record, buffered[0], est)
        for f in buffered:
            writer.write(f)
    if writer is not None:
        writer.release()
        print(f"[client] recorded annotated video -> {record}")

    metrics.print_summary()
    print("[client] throttled flags:", pi_throttled())
    lost = sum(1 for r in metrics.rows if r.get("lost"))
    print(f"[client] lost/timeout frames: {lost}/{len(metrics.rows)}")
    metrics.save_csv(out)


if __name__ == "__main__":
    main()
