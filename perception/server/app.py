"""FastAPI app for the REST detection path.

POST raw JPEG bytes to /detect; get detections + server inference time back.
This is the HTTP/TCP transport; the UDP transport lives in udp_server.py and
runs in the same process on a different port (see scripts/run_server.py).
"""
import time


def create_app(detector):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import numpy as np
    import cv2

    app = FastAPI(title="pinky_perception detection server")
    state = {"requests": 0}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/stats")
    def stats():
        return state

    @app.post("/detect")
    async def detect(request: Request):
        raw = await request.body()
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "could not decode image"}, status_code=400)
        t0 = time.perf_counter()
        dets = detector.detect(img)
        infer_ms = (time.perf_counter() - t0) * 1000
        state["requests"] += 1
        return {"detections": dets, "server_infer_ms": round(infer_ms, 2)}

    return app
