"""YOLO detector for the AI server (PyTorch backend, GPU if available).

The same YOLO weights are used on the edge side (exported to NCNN), so the
two paths are comparable. `ultralytics` is imported lazily so this module can
be imported on machines without torch installed.
"""
from common.yolo_utils import ultralytics_to_dets


class Detector:
    def __init__(self, weights="yolo11n.pt", device=None, conf=0.25, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names

    def detect(self, image_bgr):
        result = self.model.predict(
            image_bgr, conf=self.conf, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )[0]
        return ultralytics_to_dets(result, self.names)
