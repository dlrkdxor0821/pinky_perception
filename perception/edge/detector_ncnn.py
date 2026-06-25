"""YOLO detector for on-device edge inference (NCNN backend via Ultralytics).

Load an Ultralytics-exported NCNN model directory, produced by:
    yolo export model=yolo11n.pt format=ncnn        # or scripts/export_ncnn.py
which creates `yolo11n_ncnn_model/` (model.ncnn.param / model.ncnn.bin).
Ultralytics runs it through the NCNN runtime, optimized for ARM CPUs.
"""
from common.yolo_utils import ultralytics_to_dets


class NCNNDetector:
    def __init__(self, model_dir="models/yolo11n_ncnn_model", conf=0.25, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(model_dir, task="detect")
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names

    def detect(self, image_bgr):
        result = self.model.predict(
            image_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False,
        )[0]
        return ultralytics_to_dets(result, self.names)
