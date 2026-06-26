#!/usr/bin/env bash
# On-device (Pi) edge NCNN benchmark. Extra args are passed through, e.g.:
#   ./benchmark/run_edge.sh --source csi --imgsz 320 --frames 300
set -e
cd "$(dirname "$0")/.."
# Prefer the project venv (ncnn lives there); fall back to system python3.
PY="${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"
exec "$PY" edge/detect_edge.py "$@"
