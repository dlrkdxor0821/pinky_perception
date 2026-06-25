#!/usr/bin/env python3
"""On-device (edge) benchmark: capture -> NCNN YOLO -> metrics. Runs on the Pi.

Measures pure on-device latency (no network). Logs FPS, latency percentiles,
detection count / confidence ("인식률" proxy until labeled mAP), and Pi
CPU / RAM / temperature so thermal throttling is visible in the results.
"""
import argparse
import pathlib
import sys
import time

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(PKG_ROOT / "models/pinky_pro_and_person_ncnn_model"))
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--rotate", type=int, default=180, choices=[0, 90, 180, 270],
                    help="rotate frames upright (this cam needs 180)")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--vflip", action="store_true")
    ap.add_argument("--threads", type=int, default=4, help="NCNN num_threads (Pi has 4 cores)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", type=int, default=0,
                    help="save every Nth annotated frame to disk (0=off) — for headless verify")
    ap.add_argument("--save-dir", default="benchmark/results/frames")
    ap.add_argument("--out", default="benchmark/results/edge.csv")
    args = ap.parse_args()

    from edge.detector_ncnn_raw import NCNNRawDetector
    from common.camera import Camera
    from common.metrics import (Metrics, cpu_percent, pi_temp_c, pi_throttled,
                                ram_used_mb)

    src = int(args.source) if args.source.isdigit() else args.source
    detector = NCNNRawDetector(model_dir=args.model, conf=args.conf,
                               imgsz=args.imgsz, num_threads=args.threads)
    metrics = Metrics(label="edge-ncnn")
    print(f"[edge] model={args.model} source={src} frames={args.frames}")

    if args.save:
        import os
        os.makedirs(args.save_dir, exist_ok=True)

    with Camera(src, width=args.width, height=args.height,
                rotate=args.rotate, hflip=args.hflip, vflip=args.vflip) as cam:
        cpu_percent()  # prime psutil's delta counter
        for i in range(args.frames):
            frame = cam.read()
            t0 = time.perf_counter()
            dets = detector.detect(frame)
            latency_ms = (time.perf_counter() - t0) * 1000
            confs = [d["conf"] for d in dets]
            metrics.add(
                frame=i, t_wall=time.perf_counter(), latency_ms=latency_ms,
                n_det=len(dets),
                avg_conf=(sum(confs) / len(confs) if confs else None),
                cpu=cpu_percent(), ram_mb=ram_used_mb(), temp_c=pi_temp_c(),
            )
            if args.save and i % args.save == 0:
                import cv2
                from common.viz import draw_detections
                annotated = draw_detections(frame.copy(), dets)
                cv2.imwrite(f"{args.save_dir}/edge_{i:04d}.jpg", annotated)
            if args.show:
                import cv2
                from common.viz import draw_detections
                cv2.imshow("edge", draw_detections(frame, dets))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    metrics.print_summary()
    print("[edge] throttled flags:", pi_throttled())
    metrics.save_csv(args.out)


if __name__ == "__main__":
    main()
