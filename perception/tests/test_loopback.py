#!/usr/bin/env python3
"""End-to-end loopback test with a dummy detector (no YOLO weights needed).

Spins up the real UDP server and (if FastAPI is installed) the HTTP app, sends a
synthetic JPEG through the same client code path, and checks the detections come
back. Verifies the transport wiring independently of the model.

    python3 tests/test_loopback.py

Skips a transport whose optional deps (fastapi/uvicorn) are missing.
"""
import pathlib
import socket
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


class DummyDetector:
    """Returns a fixed detection regardless of the image."""
    names = {0: "dummy"}

    def detect(self, image_bgr):
        h, w = image_bgr.shape[:2]
        return [{"cls": 0, "name": "dummy", "conf": 0.9,
                 "xyxy": [0.0, 0.0, float(w), float(h)]}]


def _make_jpeg():
    import numpy as np
    import cv2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_udp_loopback():
    from server.udp_server import run_udp_server
    from client.detect_client import send_udp

    port = 9555
    stop = threading.Event()
    t = threading.Thread(
        target=run_udp_server,
        args=(DummyDetector(), "127.0.0.1", port, stop),
        daemon=True,
    )
    t.start()
    time.sleep(0.5)  # let the socket bind
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        resp = send_udp(sock, ("127.0.0.1", port), 123, _make_jpeg(), timeout=3.0)
        assert resp is not None, "no UDP reply"
        assert resp["detections"][0]["name"] == "dummy"
        assert "server_infer_ms" in resp
    finally:
        stop.set()
        t.join(timeout=2.0)


def test_http_loopback():
    try:
        import uvicorn  # noqa: F401
        from fastapi.testclient import TestClient
    except Exception:
        print("SKIP test_http_loopback (fastapi/uvicorn not installed)")
        return
    from server.app import create_app

    client = TestClient(create_app(DummyDetector()))
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/detect", content=_make_jpeg(),
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200
    assert r.json()["detections"][0]["name"] == "dummy"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
