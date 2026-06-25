#!/usr/bin/env python3
"""Live MJPEG preview of edge NCNN detections — view in a browser, no client app.

    python3 perception/edge/preview_server.py \
        --model perception/models/pinky_pro_and_person_ncnn_model \
        --source csi --imgsz 320 --port 8080

Then open  http://<라즈베리파이-IP>:8080/  in a browser on your PC. You'll see the
live camera with bounding boxes + FPS overlay.

Uses only the Python stdlib http.server (no Flask/FastAPI install) to stay light
on the Pi. A background thread captures + runs inference; the HTTP handler streams
the latest annotated JPEG as multipart/x-mixed-replace (MJPEG).
"""
import argparse
import pathlib
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
    from edge.detector_ncnn_raw import NCNNRawDetector
    from common.camera import Camera
    from common.viz import draw_detections

    det = NCNNRawDetector(model_dir=args.model, conf=args.conf,
                          imgsz=args.imgsz, num_threads=args.threads)
    src = int(args.source) if args.source.isdigit() else args.source
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    if args.save:
        import os
        os.makedirs(args.save_dir, exist_ok=True)
    print(f"[preview] capture loop started (source={src}, rotate={args.rotate})")
    i = 0
    with Camera(src, width=args.width, height=args.height,
                rotate=args.rotate, hflip=args.hflip, vflip=args.vflip) as cam:
        t_prev = time.perf_counter()
        while not _stop.is_set():
            frame = cam.read()
            dets = det.detect(frame)
            draw_detections(frame, dets)
            now = time.perf_counter()
            fps = 1.0 / (now - t_prev) if now > t_prev else 0.0
            t_prev = now
            cv2.putText(frame, f"{fps:4.1f} FPS  {len(dets)} det", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if args.save and i % args.save == 0:
                cv2.imwrite(f"{args.save_dir}/preview_{i:05d}.jpg", frame)
            i += 1
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if ok:
                with _lock:
                    _latest["jpeg"] = buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

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
            pass  # browser closed the tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(PKG_ROOT / "models/pinky_pro_and_person_ncnn_model"))
    ap.add_argument("--source", default="csi")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--rotate", type=int, default=180, choices=[0, 90, 180, 270],
                    help="rotate frames to make the scene upright (this cam needs 180)")
    ap.add_argument("--save", type=int, default=0,
                    help="also save every Nth annotated frame to disk (0=off)")
    ap.add_argument("--save-dir", default="benchmark/results/frames")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--vflip", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--max-fps", type=float, default=15.0, dest="max_fps")
    args = ap.parse_args()

    worker = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    worker.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.max_fps = args.max_fps
    print(f"[preview] open http://<this-pi-ip>:{args.port}/  in a browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()
