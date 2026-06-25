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
    args = ap.parse_args()

    import cv2
    from common.camera import Camera
    from common.metrics import Metrics

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

    with Camera(src, width=args.width, height=args.height) as cam:
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
            )

    metrics.print_summary()
    lost = sum(1 for r in metrics.rows if r.get("lost"))
    print(f"[client] lost/timeout frames: {lost}/{len(metrics.rows)}")
    metrics.save_csv(out)


if __name__ == "__main__":
    main()
