"""On-device YOLO detector using the raw NCNN runtime (no torch / no ultralytics).

Loads an Ultralytics-exported NCNN model (model.ncnn.param / model.ncnn.bin +
metadata.yaml) and does YOLOv8/v11 letterbox preprocessing, NCNN inference, and
box decode + NMS by hand. This keeps the edge path truly lightweight on the Pi
(only ncnn + numpy + opencv), which is the whole point of on-device inference.

Same .detect(image_bgr) -> [{cls,name,conf,xyxy}] interface as the other
detectors, so detect_edge.py uses it unchanged.
"""
import os

import numpy as np
import cv2
import ncnn


def _load_names(metadata_path):
    try:
        import yaml
        with open(metadata_path) as f:
            meta = yaml.safe_load(f)
        return meta.get("names", {}), meta.get("imgsz")
    except Exception:
        return {}, None


def letterbox(img, new_size, color=114):
    """Resize keeping aspect ratio, pad to square (YOLO-style)."""
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    dh, dw = (new_size - nh) // 2, (new_size - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, scale, dw, dh


class NCNNRawDetector:
    def __init__(self, model_dir="models/yolo11n_ncnn_model", conf=0.25, imgsz=320,
                 nms=0.45, num_threads=4):
        self.conf = conf
        self.imgsz = imgsz
        self.nms = nms
        param = f"{model_dir}/model.ncnn.param"
        bin_ = f"{model_dir}/model.ncnn.bin"
        # ncnn doesn't raise on a missing file — it silently loads an empty net,
        # which later fails with "find_blob_index_by_name in0 failed". Check here.
        for p in (param, bin_):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"NCNN model file not found: {p}\n"
                    f"  (cwd={os.getcwd()}) — pass --model with a correct/absolute path.")
        self.net = ncnn.Net()
        self.net.opt.num_threads = num_threads
        self.net.opt.use_vulkan_compute = False
        # NOTE: Cortex-A72 (Pi 4) has NO hardware fp16 arithmetic (no asimdhp),
        # so use_fp16_* is ~9% SLOWER here — leave it off. winograd/sgemm conv
        # are on by default in ncnn and help on ARM.
        self.net.opt.use_winograd_convolution = True
        self.net.opt.use_sgemm_convolution = True
        self.net.load_param(param)
        self.net.load_model(bin_)
        names, meta_imgsz = _load_names(f"{model_dir}/metadata.yaml")
        self.names = names
        # honor the exported input size if metadata provides it
        if isinstance(meta_imgsz, (list, tuple)) and meta_imgsz:
            self.imgsz = int(meta_imgsz[0])
        elif isinstance(meta_imgsz, int):
            self.imgsz = meta_imgsz
        self.nc = len(names) if names else None

    def _name(self, cls):
        if isinstance(self.names, dict):
            return self.names.get(cls, str(cls))
        return str(cls)

    def detect(self, image_bgr):
        h0, w0 = image_bgr.shape[:2]
        canvas, scale, dw, dh = letterbox(image_bgr, self.imgsz)

        mat_in = ncnn.Mat.from_pixels(canvas, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                                      self.imgsz, self.imgsz)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0],
                                        [1 / 255.0, 1 / 255.0, 1 / 255.0])

        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        _, mat_out = ex.extract("out0")

        out = np.squeeze(np.array(mat_out))            # (4+nc, N) or (N, 4+nc)
        if out.ndim != 2:
            return []
        nc = self.nc if self.nc else (min(out.shape) - 4)
        if out.shape[0] == 4 + nc:                     # (4+nc, N) -> (N, 4+nc)
            out = out.T

        boxes_xywh = out[:, :4]
        scores = out[:, 4:4 + nc]
        class_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)

        keep = confs >= self.conf
        if not np.any(keep):
            return []
        boxes_xywh, confs, class_ids = boxes_xywh[keep], confs[keep], class_ids[keep]

        # center xywh (in letterboxed imgsz space) -> xyxy in original frame
        cx, cy, bw, bh = boxes_xywh.T
        x1 = (cx - bw / 2 - dw) / scale
        y1 = (cy - bh / 2 - dh) / scale
        x2 = (cx + bw / 2 - dw) / scale
        y2 = (cy + bh / 2 - dh) / scale
        x1 = np.clip(x1, 0, w0); y1 = np.clip(y1, 0, h0)
        x2 = np.clip(x2, 0, w0); y2 = np.clip(y2, 0, h0)

        # NMS (OpenCV expects [x, y, w, h])
        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), self.conf, self.nms)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        dets = []
        for i in idxs:
            cls = int(class_ids[i])
            dets.append({
                "cls": cls,
                "name": self._name(cls),
                "conf": float(confs[i]),
                "xyxy": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            })
        return dets
