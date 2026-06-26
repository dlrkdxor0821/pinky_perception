"""Thread-safe holder for the latest annotated JPEG.

Shared by the HTTP and UDP detection paths so the FastAPI /stream endpoint can
show live boxes on whatever frame the robot last sent. Visualization ONLY —
drawing + JPEG re-encode adds load, so enable it (run_server.py --preview) for
quality checks, never during a benchmark run.
"""
import threading


class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None

    def update(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes

    def get(self):
        with self._lock:
            return self._jpeg
