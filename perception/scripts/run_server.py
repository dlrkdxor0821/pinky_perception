#!/usr/bin/env python3
"""Launch the detection server: FastAPI (HTTP) + UDP socket — two ports, one
process, one shared model.

    python3 scripts/run_server.py --weights yolo11n.pt --device cuda

HTTP path: POST JPEG to http://<host>:<http-port>/detect
UDP path:  send chunked JPEG to <host>:<udp-port> (see common/protocol.py)
"""
import argparse
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--device", default=None, help="cuda / cpu / 0 ...")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--udp-port", type=int, default=9000)
    args = ap.parse_args()

    from server.detector import Detector
    from server.app import create_app
    from server.udp_server import run_udp_server
    import uvicorn

    detector = Detector(weights=args.weights, device=args.device,
                        conf=args.conf, imgsz=args.imgsz)
    print(f"[server] model loaded: {args.weights}")

    # UDP server in a background thread (shares the single detector instance).
    t = threading.Thread(
        target=run_udp_server,
        args=(detector, args.host, args.udp_port),
        daemon=True,
    )
    t.start()

    # FastAPI / HTTP in the main thread.
    app = create_app(detector)
    print(f"[server] HTTP {args.host}:{args.http_port} | UDP {args.host}:{args.udp_port}")
    uvicorn.run(app, host=args.host, port=args.http_port, log_level="warning")


if __name__ == "__main__":
    main()
