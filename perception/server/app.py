"""FastAPI app for the REST detection path.

POST raw JPEG bytes to /detect; get detections + server inference time back.
This is the HTTP/TCP transport; the UDP transport lives in udp_server.py and
runs in the same process on a different port (see scripts/run_server.py).
"""
import time


# Live viewer: fullscreen MJPEG stream with an FPS / inference-latency overlay
# that polls /fps a few times a second (kept off the image so it adds no
# per-frame CPU on the detection path).
_VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>pinky_perception live</title>
<style>
  html,body{margin:0;height:100%;background:#111;overflow:hidden}
  #v{width:100vw;height:100vh;object-fit:contain;display:block}
  #hud{position:fixed;top:12px;left:12px;padding:8px 12px;border-radius:8px;
       background:rgba(0,0,0,.55);color:#0f0;font:600 22px/1.25 monospace;
       white-space:pre}
</style></head>
<body>
  <img id="v" src="/stream" alt="live">
  <div id="hud">FPS --</div>
  <script>
    const hud = document.getElementById('hud');
    async function tick(){
      try{
        const s = await (await fetch('/fps', {cache:'no-store'})).json();
        const fps = (s.fps ?? 0).toFixed(1);
        const ms  = s.server_infer_ms == null ? '--' : s.server_infer_ms.toFixed(1);
        hud.textContent = `FPS ${fps}\\ninfer ${ms} ms`;
      }catch(e){ hud.textContent = 'FPS --'; }
    }
    tick(); setInterval(tick, 400);
  </script>
</body></html>"""


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
                preview.update(buf.tobytes(), infer_ms=infer_ms)
        return {"detections": dets, "server_infer_ms": round(infer_ms, 2)}

    @app.get("/fps")
    def fps():
        if preview is None:
            return JSONResponse({"fps": 0.0, "server_infer_ms": None})
        return preview.stats()

    @app.get("/")
    def viewer():
        if preview is None:
            return HTMLResponse("<p>preview disabled — start the server with --preview</p>")
        return HTMLResponse(_VIEWER_HTML)

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
