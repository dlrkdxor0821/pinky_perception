"""USB / CSI camera capture via OpenCV VideoCapture.

`source` can be an integer camera index (USB webcam, CSI via libcamera/V4L2)
or a path to a video file for reproducible benchmarking.
"""
import cv2


class Camera:
    def __init__(self, source=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(source)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source!r}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
