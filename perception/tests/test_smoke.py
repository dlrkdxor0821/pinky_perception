#!/usr/bin/env python3
"""Dependency-free smoke tests: no model, camera, or network required.

Verifies the UDP framing round-trips (including out-of-order delivery) and the
metrics math. Run before deploying to catch protocol/metric regressions:

    python3 tests/test_smoke.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.protocol import (Reassembler, decode_result, encode_frame,  # noqa: E402
                             encode_result)
from common.metrics import Metrics, percentile  # noqa: E402


def test_protocol_roundtrip():
    payload = bytes(range(256)) * 1000  # ~256 KB -> several chunks
    dgs = encode_frame(42, payload, chunk_size=60000)
    assert len(dgs) >= 4
    reasm = Reassembler()
    out = None
    for dg in dgs:
        out = reasm.push(dg) or out
    assert out is not None, "frame never reassembled"
    fid, data = out
    assert fid == 42
    assert data == payload, "payload mismatch after reassembly"


def test_protocol_out_of_order():
    payload = b"x" * 200000
    dgs = encode_frame(7, payload, chunk_size=60000)
    reasm = Reassembler()
    out = None
    for dg in reversed(dgs):  # deliver chunks in reverse order
        out = reasm.push(dg) or out
    assert out and out[1] == payload


def test_protocol_single_chunk():
    dgs = encode_frame(1, b"hello")
    assert len(dgs) == 1
    fid, data = Reassembler().push(dgs[0])
    assert fid == 1 and data == b"hello"


def test_result_codec():
    fid, data = decode_result(encode_result(99, b'{"ok":1}'))
    assert fid == 99 and data == b'{"ok":1}'


def test_percentile():
    vals = list(range(1, 101))
    assert percentile(vals, 50) == 50.5
    assert percentile(vals, 95) == 95.05
    assert percentile([], 95) is None


def test_metrics_summary():
    m = Metrics("t")
    for i in range(10):
        m.add(frame=i, t_wall=i * 0.1, latency_ms=10 + i, n_det=2, avg_conf=0.5)
    s = m.summary()
    assert s["frames"] == 10
    assert s["latency_ms_mean"] is not None
    assert s["fps"] is not None


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
