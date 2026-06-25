#!/usr/bin/env python3
"""Capture calibration images for ncnn int8 quantization.

int8 quantization needs a representative set of images (the scenes the model
will actually see) to calibrate per-layer scales. This grabs N frames from the
camera and writes an imagelist.txt that ncnn2table consumes.

    python3 perception/scripts/capture_calib.py --source csi --n 200
"""
import argparse
import pathlib
import sys
import time

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="csi")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rotate", type=int, default=180, choices=[0, 90, 180, 270])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--interval", type=float, default=0.1, help="seconds between frames")
    ap.add_argument("--out", default=str(PKG_ROOT / "eval/calib"))
    args = ap.parse_args()

    import cv2
    from common.camera import Camera

    out = pathlib.Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    src = int(args.source) if args.source.isdigit() else args.source
    listing = []
    print(f"[calib] capturing {args.n} frames from {src} ... (move objects/people around)")
    with Camera(src, width=args.width, height=args.height, rotate=args.rotate) as cam:
        for i in range(args.n):
            frame = cam.read()
            p = out / "images" / f"calib_{i:04d}.jpg"
            cv2.imwrite(str(p), frame)
            listing.append(str(p.resolve()))
            time.sleep(args.interval)
    (out / "imagelist.txt").write_text("\n".join(listing) + "\n")
    print(f"[calib] {len(listing)} images -> {out}/images")
    print(f"[calib] list -> {out}/imagelist.txt  (feed this to quantize_int8.sh)")


if __name__ == "__main__":
    main()
