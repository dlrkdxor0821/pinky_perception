#!/usr/bin/env bash
# Server-path benchmark from the Pi. Extra args are passed through, e.g.:
#   ./benchmark/run_server_path.sh --transport udp  --host 192.168.0.10 --source csi --rotate 180
#   ./benchmark/run_server_path.sh --transport http --host 192.168.0.10 --source csi --rotate 180
set -e
cd "$(dirname "$0")/.."
# Prefer the project venv (opencv/requests live there); fall back to system python3.
PY="${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"
exec "$PY" client/detect_client.py "$@"
