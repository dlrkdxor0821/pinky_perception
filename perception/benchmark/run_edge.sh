#!/usr/bin/env bash
# On-device (Pi) edge NCNN benchmark. Extra args are passed through, e.g.:
#   ./benchmark/run_edge.sh --frames 500 --imgsz 320
set -e
cd "$(dirname "$0")/.."
python3 edge/detect_edge.py "$@"
