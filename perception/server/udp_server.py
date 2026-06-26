"""UDP image-ingest server: receive chunked JPEG, run detection, reply JSON.

Runs alongside the FastAPI app (different port) so one server process exposes
both transports. The robot's frames arrive as UDP datagrams (see
common/protocol.py); the detection result is sent back to the sender address.

Note: the detector instance is shared with the FastAPI app. Run one transport
at a time when benchmarking (Ultralytics predict is not meant for concurrent
calls on the same model).
"""
import json
import socket
import time


def run_udp_server(detector, host="0.0.0.0", port=9000, stop=None, preview=None):
    import numpy as np
    import cv2
    from common.protocol import Reassembler, encode_result

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((host, port))
    sock.settimeout(1.0)
    reasm = Reassembler()
    print(f"[udp] listening on {host}:{port}")

    while stop is None or not stop.is_set():
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        res = reasm.push(data)
        if res is None:
            continue
        frame_id, jpeg = res
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        t0 = time.perf_counter()
        dets = detector.detect(img)
        infer_ms = (time.perf_counter() - t0) * 1000
        if preview is not None:
            from common.viz import draw_detections
            ok, buf = cv2.imencode(".jpg", draw_detections(img, dets))
            if ok:
                preview.update(buf.tobytes())
        payload = json.dumps(
            {"detections": dets, "server_infer_ms": round(infer_ms, 2)}
        ).encode()
        sock.sendto(encode_result(frame_id, payload), addr)

    sock.close()
