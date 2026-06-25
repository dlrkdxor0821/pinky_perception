#!/usr/bin/env bash
# NCNN int8 quantization for the edge model.
#
# Requires the ncnn CLI tools `ncnn2table` and `ncnnoptimize`, which are NOT in
# the pip `ncnn` package. Get them one of two ways:
#   (A) build on the Pi:   ncnn with -DNCNN_BUILD_TOOLS=ON  (see README)
#   (B) on an x86 dev box: download a prebuilt ncnn release (tools included),
#       run this there, then copy the *_int8_ncnn_model/ folder back to the Pi.
#
# Preprocessing here MUST match inference (detector_ncnn_raw.py):
#   BGR->RGB, normalize 1/255 (=0.00392157), input 320x320.
#
# Usage:
#   ./perception/scripts/quantize_int8.sh \
#       perception/models/pinky_pro_and_person_ncnn_model \
#       perception/eval/calib/imagelist.txt
set -e

MODEL_DIR="${1:-perception/models/pinky_pro_and_person_ncnn_model}"
CALIB_LIST="${2:-perception/eval/calib/imagelist.txt}"
OUT_DIR="${MODEL_DIR%_ncnn_model}_int8_ncnn_model"

command -v ncnn2table  >/dev/null || { echo "ERROR: ncnn2table not found (see header for how to get ncnn tools)"; exit 1; }
command -v ncnnoptimize >/dev/null || { echo "ERROR: ncnnoptimize not found"; exit 1; }
[ -f "$CALIB_LIST" ] || { echo "ERROR: calib list missing: $CALIB_LIST (run scripts/capture_calib.py first)"; exit 1; }

mkdir -p "$OUT_DIR"

echo "[1/2] ncnn2table -> calibration table"
ncnn2table "$MODEL_DIR/model.ncnn.param" "$MODEL_DIR/model.ncnn.bin" \
  "$CALIB_LIST" "$OUT_DIR/model.table" \
  mean=[0,0,0] norm=[0.00392157,0.00392157,0.00392157] \
  shape=[320,320,3] pixel=BGR2RGB thread=4 method=kl

echo "[2/2] ncnnoptimize -> int8 model"
ncnnoptimize "$MODEL_DIR/model.ncnn.param" "$MODEL_DIR/model.ncnn.bin" \
  "$OUT_DIR/model.ncnn.param" "$OUT_DIR/model.ncnn.bin" 0 "$OUT_DIR/model.table"

cp "$MODEL_DIR/metadata.yaml" "$OUT_DIR/" 2>/dev/null || true

echo "done -> $OUT_DIR"
echo "compare speed:"
echo "  fp32: detect_edge.py --model $MODEL_DIR --imgsz 320"
echo "  int8: detect_edge.py --model $OUT_DIR  --imgsz 320"
