#!/usr/bin/env python3
"""Export a YOLO .pt model to NCNN for edge inference.

    python3 scripts/export_ncnn.py --weights yolo11n.pt --imgsz 640

Produces e.g. `yolo11n_ncnn_model/` next to the weights. Move it under
`perception/models/` and point detect_edge.py --model at it.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="fp16 export")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    path = model.export(format="ncnn", imgsz=args.imgsz, half=args.half)
    print(f"[export] NCNN model -> {path}")


if __name__ == "__main__":
    main()
