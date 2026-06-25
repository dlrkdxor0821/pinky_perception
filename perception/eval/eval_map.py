#!/usr/bin/env python3
"""Compute detection accuracy (mAP) for a YOLO model on a labeled test set.

Accuracy is a property of the MODEL, not the transport, so each model is
evaluated offline against a labeled dataset — no server or network needed.
Run it once per model, then compare with compare_map.py:

    # edge (light, NCNN) model
    python3 eval/eval_map.py --model models/yolo11n_ncnn_model \
        --data eval/datasets/test/data.yaml --label edge

    # server (full, PyTorch) model
    python3 eval/eval_map.py --model yolo11n.pt \
        --data eval/datasets/test/data.yaml --label server

Output: mAP@0.5, mAP@0.5:0.95, precision, recall -> JSON in benchmark/results/.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt weights or NCNN export dir")
    ap.add_argument("--data", required=True, help="YOLO dataset yaml (with a val/test split)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="standard low conf for mAP")
    ap.add_argument("--split", default="val", help="dataset split key in the yaml")
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    m = model.val(data=args.data, imgsz=args.imgsz, conf=args.conf,
                  split=args.split, verbose=False)

    res = {
        "label": args.label,
        "model": args.model,
        "data": args.data,
        "map50": round(float(m.box.map50), 4),       # mAP@0.5
        "map50_95": round(float(m.box.map), 4),      # mAP@0.5:0.95
        "precision": round(float(m.box.mp), 4),      # mean precision
        "recall": round(float(m.box.mr), 4),         # mean recall
    }
    print(json.dumps(res, indent=2))

    out = args.out or f"benchmark/results/map_{args.label}.json"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[eval] saved -> {out}")


if __name__ == "__main__":
    main()
