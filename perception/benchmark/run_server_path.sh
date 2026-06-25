#!/usr/bin/env bash
# Server-path benchmark from the Pi. Extra args are passed through, e.g.:
#   ./benchmark/run_server_path.sh --transport udp --host 192.168.0.10
#   ./benchmark/run_server_path.sh --transport http --host 192.168.0.10
set -e
cd "$(dirname "$0")/.."
python3 client/detect_client.py "$@"
