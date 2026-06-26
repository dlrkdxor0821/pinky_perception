"""Thread-safe holder for the latest annotated JPEG.

Shared by the HTTP and UDP detection paths so the FastAPI /stream endpoint can
show live boxes on whatever frame the robot last sent. Visualization ONLY —
drawing + JPEG re-encode adds load, so enable it (run_server.py --preview) for
quality checks, never during a benchmark run.

It also tracks the rate at which frames arrive (each update() == one processed
frame) so the viewer can show live FPS and server inference latency.
"""
import collections
import threading
import time


class LatestFrame:
    # rolling window (seconds) over which FPS is averaged; long enough to be
    # stable at the edge path's ~3 FPS, short enough to react to changes.
    _WINDOW_S = 2.0

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._stamps = collections.deque()  # perf_counter() of recent updates
        self._infer_ms = None               # latest server inference time

    def update(self, jpeg_bytes, infer_ms=None):
        now = time.perf_counter()
        with self._lock:
            self._jpeg = jpeg_bytes
            self._infer_ms = infer_ms
            self._stamps.append(now)
            while self._stamps and now - self._stamps[0] > self._WINDOW_S:
                self._stamps.popleft()

    def get(self):
        with self._lock:
            return self._jpeg

    def stats(self):
        """Live throughput/latency for the viewer overlay."""
        with self._lock:
            stamps = self._stamps
            infer_ms = self._infer_ms
            # FPS = frames per second averaged over the window. Needs >= 2
            # samples and the most recent one to be fresh (else the sender
            # stopped and FPS has decayed to 0).
            now = time.perf_counter()
            fresh = stamps and now - stamps[-1] <= self._WINDOW_S
            if fresh and len(stamps) >= 2 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            else:
                fps = 0.0
        return {
            "fps": round(fps, 1),
            "server_infer_ms": round(infer_ms, 1) if infer_ms is not None else None,
        }
