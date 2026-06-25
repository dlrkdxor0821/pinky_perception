#!/usr/bin/env python3
"""Live browser preview of the SERVER-path detections (GPU), viewed via MJPEG.

The Pi captures camera frames, sends them to the AI server (UDP), gets the
detections back, draws the boxes, and streams the annotated frames to a browser.
So the *detection* is done by the server GPU (~27 FPS), but the viewer runs here
on the Pi (because the camera is here).

    python3 perception/client/preview_client.py --host <server-ip> --source csi

Then open  http://<pi-ip>:8090/  in a browser. Records like preview_server.py.
"""
import argparse
import json
import pathlib
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

_latest = {"jpeg": None}
_lock = threading.Lock()
_stop = threading.Event()


def capture_loop(args):
    import cv2
    from common.camera import Camera
    from common.viz import draw_detections
    from common.protocol import encode_frame, decode_result

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_addr = (args.host, args.udp_port)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    src = int(args.source) if args.source.isdigit() else args.source
    print(f"[preview-client] -> server {args.host}:{args.udp_port}")

    with Camera(src, width=args.width, height=args.height, rotate=args.rotate) as cam:
        t_prev = time.perf_counter()
        i = 0
        while not _stop.is_set():
            frame = cam.read()
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if not ok:
                continue
            # send to server, wait for detections
            for dg in encode_frame(i, buf.tobytes()):
                sock.sendto(dg, server_addr)
            sock.settimeout(1.0)
            dets = []
            try:
                while True:
                    data, _ = sock.recvfrom(65535)
                    fid, payload = decode_result(data)
                    if fid == (i & 0xFFFFFFFF):
                        dets = json.loads(payload.decode()).get("detections", [])
                        break
            except socket.timeout:
                pass
            i += 1

            draw_detections(frame, dets)
            now = time.perf_counter()
            fps = 1.0 / (now - t_prev) if now > t_prev else 0.0
            t_prev = now
            cv2.putText(frame, f"{fps:4.1f} FPS  {len(dets)} det  [server]", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            ok, out = cv2.imencode(".jpg", frame, enc)
            if ok:
                with _lock:
                    _latest["jpeg"] = out.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path not in ("/", "/stream"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while not _stop.is_set():
                with _lock:
                    jpg = _latest["jpeg"]
                if jpg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1.0 / self.server.max_fps)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="AI server IP")
    ap.add_argument("--udp-port", type=int, default=9000)
    ap.add_argument("--source", default="csi")
    ap.add_argument("--rotate", type=int, default=180, choices=[0, 90, 180, 270])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--port", type=int, default=8090, help="local browser port")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--max-fps", type=float, default=30.0, dest="max_fps")
    args = ap.parse_args()

    worker = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    worker.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.max_fps = args.max_fps
    print(f"[preview-client] open http://<this-pi-ip>:{args.port}/  in a browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()
