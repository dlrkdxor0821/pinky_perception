#!/usr/bin/env python3
"""Aggregate result CSVs into a side-by-side edge-vs-server comparison table.

    python3 benchmark/compare.py benchmark/results/edge.csv \
                                 benchmark/results/server_udp.csv \
                                 benchmark/results/server_http.csv
"""
import argparse
import csv
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.metrics import percentile  # noqa: E402


def summarize(path):
    rows = list(csv.DictReader(open(path)))

    def col(key, cast=float):
        out = []
        for r in rows:
            v = r.get(key, "")
            if v not in ("", "None", None):
                try:
                    out.append(cast(v))
                except ValueError:
                    pass
        return out

    lat = col("latency_ms")
    tw = col("t_wall")
    ndet = col("n_det")
    conf = col("avg_conf")
    fps = (len(tw) - 1) / (tw[-1] - tw[0]) if len(tw) >= 2 and tw[-1] > tw[0] else None
    lost = sum(1 for r in rows if str(r.get("lost", "")).lower() == "true")
    return {
        "frames": len(rows),
        "fps": round(fps, 2) if fps else None,
        "lat_mean": round(statistics.mean(lat), 2) if lat else None,
        "lat_p95": round(percentile(lat, 95), 2) if lat else None,
        "avg_det": round(statistics.mean(ndet), 2) if ndet else None,
        "avg_conf": round(statistics.mean(conf), 3) if conf else None,
        "lost": lost,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="result CSV files to compare")
    args = ap.parse_args()

    cols = ["frames", "fps", "lat_mean", "lat_p95", "avg_det", "avg_conf", "lost"]
    print(f'{"file":26s}' + "".join(f"{c:>10s}" for c in cols))
    print("-" * (26 + 10 * len(cols)))
    for path in args.csvs:
        s = summarize(path)
        name = pathlib.Path(path).stem
        print(f"{name:26s}" + "".join(f"{str(s[c]):>10s}" for c in cols))


if __name__ == "__main__":
    main()
