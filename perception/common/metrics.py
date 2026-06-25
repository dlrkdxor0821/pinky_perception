"""Per-frame benchmark metrics: latency, FPS, detection stats, Pi resources.

Records one row per frame, then summarizes. FPS is wall-clock (from frame
timestamps), not 1000/latency, so it reflects sustained throughput.
"""
import csv
import os
import shutil
import statistics
import subprocess

try:
    import psutil
except ImportError:
    psutil = None

_HAS_VCGENCMD = shutil.which("vcgencmd") is not None


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def pi_temp_c():
    """Raspberry Pi SoC temperature in Celsius (None if not on a Pi)."""
    if not _HAS_VCGENCMD:
        return None
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        return float(out.strip().split("=")[1].split("'")[0])
    except Exception:
        return None


def pi_throttled():
    """Raspberry Pi throttle/undervoltage flags (hex string), None if N/A."""
    if not _HAS_VCGENCMD:
        return None
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True)
        return out.strip().split("=")[1]
    except Exception:
        return None


def cpu_percent():
    return psutil.cpu_percent(interval=None) if psutil else None


def ram_used_mb():
    return round(psutil.virtual_memory().used / 1e6, 1) if psutil else None


class Metrics:
    def __init__(self, label=""):
        self.label = label
        self.rows = []

    def add(self, **row):
        self.rows.append(row)

    def _col(self, key):
        return [r[key] for r in self.rows if r.get(key) is not None]

    def summary(self):
        lat = self._col("latency_ms")
        tw = self._col("t_wall")
        ndet = self._col("n_det")
        conf = self._col("avg_conf")
        fps = None
        if len(tw) >= 2 and tw[-1] > tw[0]:
            fps = (len(tw) - 1) / (tw[-1] - tw[0])
        return {
            "label": self.label,
            "frames": len(self.rows),
            "fps": round(fps, 2) if fps else None,
            "latency_ms_mean": round(statistics.mean(lat), 2) if lat else None,
            "latency_ms_p50": round(percentile(lat, 50), 2) if lat else None,
            "latency_ms_p95": round(percentile(lat, 95), 2) if lat else None,
            "latency_ms_p99": round(percentile(lat, 99), 2) if lat else None,
            "avg_detections": round(statistics.mean(ndet), 2) if ndet else None,
            "avg_confidence": round(statistics.mean(conf), 3) if conf else None,
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n=== Benchmark summary: {s['label']} ===")
        for k, v in s.items():
            if k == "label":
                continue
            print(f"  {k:20s}: {v}")

    def save_csv(self, path):
        if not self.rows:
            print("[metrics] no rows to save")
            return
        keys = sorted({k for r in self.rows for k in r})
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        print(f"[metrics] saved {len(self.rows)} rows -> {path}")
