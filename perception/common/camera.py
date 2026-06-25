"""Camera capture with two backends.

- V4L2 (OpenCV VideoCapture): USB webcams, video files. `source` = int index or path.
- picamera2 (libcamera): Raspberry Pi CSI cameras. `source` = "csi" (or "picamera2").

On the Pi, CSI cameras must go through libcamera/picamera2 — OpenCV's V4L2 path
returns all-black frames for them. Returned frames are BGR (OpenCV convention).
"""
import time

import cv2


_ROTATE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class Camera:
    def __init__(self, source=0, width=640, height=480, fps=30,
                 rotate=0, hflip=False, vflip=False):
        self.rotate = int(rotate) % 360
        self.hflip = hflip
        self.vflip = vflip
        if isinstance(source, str) and source.lower() in ("csi", "picamera2", "libcamera"):
            self.backend = "picamera2"
            self._init_picamera2(width, height)
        else:
            self.backend = "v4l2"
            self._init_v4l2(source, width, height, fps)

    def _orient(self, frame):
        if self.rotate in _ROTATE:
            frame = cv2.rotate(frame, _ROTATE[self.rotate])
        if self.hflip:
            frame = cv2.flip(frame, 1)
        if self.vflip:
            frame = cv2.flip(frame, 0)
        return frame

    def _init_picamera2(self, width, height):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"})
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.5)  # sensor warmup / auto-exposure settle

    def _init_v4l2(self, source, width, height, fps):
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
        if self.backend == "picamera2":
            # picamera2 "RGB888" returns a BGR-ordered array (OpenCV-compatible)
            frame = self.picam2.capture_array()
            if frame is None:
                raise RuntimeError("Failed to capture frame from picamera2")
            return self._orient(frame)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera")
        return self._orient(frame)

    def release(self):
        if self.backend == "picamera2":
            try:
                self.picam2.stop()
            except Exception:
                pass
        elif getattr(self, "cap", None) is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
