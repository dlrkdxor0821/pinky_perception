"""FastAPI app for the REST detection path.

POST raw JPEG bytes to /detect; get detections + server inference time back.
This is the HTTP/TCP transport; the UDP transport lives in udp_server.py and
runs in the same process on a different port (see scripts/run_server.py).
"""
import time


def create_app(detector, preview=None):
    """`preview` is an optional common.preview.LatestFrame; when given, each
    request stores an annotated JPEG so /stream can show live boxes."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
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
        if preview is not None:
            from common.viz import draw_detections
            ok, buf = cv2.imencode(".jpg", draw_detections(img, dets))
            if ok:
                preview.update(buf.tobytes())
        return {"detections": dets, "server_infer_ms": round(infer_ms, 2)}

    @app.get("/")
    def viewer():
        if preview is None:
            return HTMLResponse("<p>preview disabled — start the server with --preview</p>")
        return HTMLResponse(
            "<html><body style='margin:0;background:#111'>"
            "<img src='/stream' style='width:100vw;height:100vh;object-fit:contain'>"
            "</body></html>")

    @app.get("/stream")
    def stream():
        if preview is None:
            return JSONResponse(
                {"error": "preview disabled; start server with --preview"},
                status_code=404)

        def gen():
            while True:
                jpeg = preview.get()
                if jpeg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpeg + b"\r\n")
                time.sleep(0.03)  # ~30 Hz poll; actual rate = inbound frame rate

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
